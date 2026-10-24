import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rerank
import rerank2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_SCHEDULE = "export_arh_11-20-final.csv"
EVAL_SCHEDULE = "export_arh_21-30-final.csv"
FINAL_SCHEDULE = "export_arh_31-42-final.csv"
REFERENCE = Path("submissions/mf_blend.csv")
SUBMISSION = Path("submissions/rerank5.csv")

BASE_SPECS = [
    ("train1", TRAIN_SCHEDULE, (16, 20), (11, 15)),
    ("train2", TRAIN_SCHEDULE, (18, 20), (11, 17)),
]
EXTRA_SPECS = [
    ("f3", TRAIN_SCHEDULE, (14, 16), (11, 13)),
    ("f4", TRAIN_SCHEDULE, (17, 20), (11, 16)),
    ("f5", TRAIN_SCHEDULE, (15, 17), (11, 14)),
]
EVAL_SPEC = ("eval", EVAL_SCHEDULE, (21, 30), (11, 20))
FINAL_SPEC = ("final", FINAL_SCHEDULE, (31, 42), (11, 30))
LATE_SPECS = [
    ("fold4", EVAL_SCHEDULE, (26, 30), (11, 25)),
    ("late1", EVAL_SCHEDULE, (24, 26), (11, 23)),
    ("late2", EVAL_SCHEDULE, (27, 30), (11, 26)),
    ("late3", EVAL_SCHEDULE, (25, 27), (11, 24)),
]


def write_submission(recommendations, users, reference, pool, fill):
    rows = []
    for user in users:
        recommendation = rerank.fill_with_popularity(
            recommendations.get(user, []), fill
        )
        rows.append(" ".join(map(str, recommendation)))
    output = pd.DataFrame(
        {"user_id": reference["user_id"].values, "tv_show_id": rows}
    )
    SUBMISSION.parent.mkdir(exist_ok=True)
    csv_text = output.to_csv(index=False, lineterminator="\n")
    SUBMISSION.write_bytes(csv_text.rstrip("\n").encode("utf-8"))

    reference_bytes = REFERENCE.read_bytes()
    produced_bytes = SUBMISSION.read_bytes()
    produced = pd.read_csv(
        SUBMISSION,
        dtype={"user_id": "uint64", "tv_show_id": str},
    )
    reference_frame = pd.read_csv(
        REFERENCE, dtype={"user_id": "uint64", "tv_show_id": str}
    )
    produced_rows = produced["tv_show_id"].str.split()
    produced_ids = [
        [int(item) for item in row] for row in produced_rows
    ]
    checks = {
        "header": produced_bytes.split(b"\n")[0]
        == reference_bytes.split(b"\n")[0]
        == b"user_id,tv_show_id",
        "row_count": len(produced) == len(reference_frame) == 1260,
        "user_order": np.array_equal(
            produced["user_id"].values,
            reference_frame["user_id"].values,
        ),
        "five_ids": bool(
            produced["tv_show_id"].str.fullmatch(
                r"\d+( \d+){4}"
            ).all()
        ),
        "five_unique_ids": all(
            len(set(row)) == rerank.TOP_K for row in produced_rows
        ),
        "positive_ids": all(
            all(item > 0 for item in row) for row in produced_ids
        ),
        "candidates_only": set(
            item for row in produced_ids for item in row
        ).issubset(set(pool)),
        "line_endings": b"\r\n" not in produced_bytes
        and produced_bytes.count(b"\n")
        == reference_bytes.count(b"\n"),
        "no_trailing_newline": not produced_bytes.endswith(b"\n")
        and not reference_bytes.endswith(b"\n"),
    }
    return checks


def main():
    total_started = time.perf_counter()
    interactions = rerank.load_interactions()
    occurrences = pd.read_parquet(
        rerank.DATA / "occurrences.parquet"
    )
    reference = pd.read_csv(
        REFERENCE, dtype={"user_id": "uint64"}
    )
    users = [int(user) for user in reference["user_id"]]
    raw_views = rerank2.load_raw_views()

    aggregates = {}
    slot_data = {}
    target_slots = {}

    def get_aggregate(weeks):
        key = tuple(weeks)
        if key not in aggregates:
            aggregates[key] = rerank2.build_view_aggregate(
                raw_views, key
            )
        return aggregates[key]

    def get_slot_data(weeks):
        key = tuple(weeks)
        if key not in slot_data:
            slot_data[key] = rerank2.build_slot_data(raw_views, key)
        return slot_data[key]

    def get_target_slots(schedule, target):
        key = (schedule, tuple(target))
        if key not in target_slots:
            target_slots[key] = rerank2.load_item_slots(
                schedule, target
            )
        return target_slots[key]

    def build_fold(spec):
        name, schedule, target, history = spec
        base = rerank.build_fold(
            interactions,
            occurrences,
            {tuple(history): get_aggregate(history)},
            schedule,
            target,
            history,
            users,
            tuple(history),
        )
        return rerank2.add_slots_to_fold(
            base,
            get_slot_data(history),
            get_target_slots(schedule, target),
        )

    fold_started = time.perf_counter()
    base_folds = {
        name: build_fold(spec) for name, spec in (
            (spec[0], spec) for spec in BASE_SPECS
        )
    }
    extra_folds = {
        spec[0]: build_fold(spec) for spec in EXTRA_SPECS
    }
    eval_fold = build_fold(EVAL_SPEC)
    fold_elapsed = time.perf_counter() - fold_started

    eval_rows = eval_fold["features"]
    eval_truth = eval_fold["truth"]
    eval_fill = eval_fold["fill"]
    train1_rows = rerank2.training_frame(base_folds["train1"])
    train2_rows = rerank2.training_frame(base_folds["train2"])
    f3_rows = rerank2.training_frame(extra_folds["f3"])
    f4_rows = rerank2.training_frame(extra_folds["f4"])
    f5_rows = rerank2.training_frame(extra_folds["f5"])

    base_rows = pd.concat(
        [train1_rows, train2_rows], ignore_index=True
    )
    f34_rows = pd.concat(
        [base_rows, f3_rows, f4_rows], ignore_index=True
    )
    f345_rows = pd.concat(
        [f34_rows, f5_rows], ignore_index=True
    )

    model_started = time.perf_counter()

    def evaluate_average(name, train_rows):
        classifier_scores, classifier, classifier_seconds = (
            rerank2.fit_classifier(
                train_rows, eval_rows, rerank2.ALL_FEATURES
            )
        )
        ranker_scores, ranker, ranker_seconds = (
            rerank2.fit_ranker(
                train_rows, eval_rows, rerank2.ALL_FEATURES
            )
        )
        scores = 0.5 * (
            rerank2.minmax_by_user(eval_rows, classifier_scores)
            + rerank2.minmax_by_user(eval_rows, ranker_scores)
        )
        result = rerank2.evaluate_scores(
            name,
            eval_rows,
            scores,
            users,
            eval_truth,
            eval_fill,
        )
        result["classifier"] = classifier
        result["ranker"] = ranker
        result["fit_seconds"] = (
            classifier_seconds + ranker_seconds
        )
        return result

    base_result = evaluate_average("base C", base_rows)
    f34_result = evaluate_average("base + F3+F4", f34_rows)
    f345_result = evaluate_average("base + F3+F4+F5", f345_rows)
    model_elapsed = time.perf_counter() - model_started

    results = [base_result, f34_result, f345_result]
    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs base"
    )
    for result in results:
        print(
            f"{result['name']} | {result['map5']:.5f} | "
            f"{result['map5'] - base_result['map5']:+.5f}"
        )
    print(
        "training rows: base "
        f"{len(base_rows):,}; +F3+F4 {len(f34_rows):,}; "
        f"+F3+F4+F5 {len(f345_rows):,}"
    )
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )

    better = max(
        [f34_result, f345_result], key=lambda result: result["map5"]
    )
    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        better["aps"] - base_result["aps"],
        n_resamples=2000,
        seed=0,
    )
    print(
        f"better addition {better['name']} vs base: "
        f"mean diff {mean_diff:+.5f}, 95% CI "
        f"[{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "better addition beats base with CI above zero; "
            "building final submission"
        )
        final_started = time.perf_counter()
        late_folds = {
            spec[0]: build_fold(spec) for spec in LATE_SPECS
        }
        final_fold = build_fold(FINAL_SPEC)
        final_train = pd.concat(
            [
                base_rows,
                rerank2.training_frame(eval_fold),
                rerank2.training_frame(late_folds["fold4"]),
                rerank2.training_frame(late_folds["late1"]),
                rerank2.training_frame(late_folds["late2"]),
                rerank2.training_frame(late_folds["late3"]),
            ],
            ignore_index=True,
        )
        final_rows = final_fold["features"]
        final_classifier_scores, _, classifier_seconds = (
            rerank2.fit_classifier(
                final_train, final_rows, rerank2.ALL_FEATURES
            )
        )
        final_ranker_scores, _, ranker_seconds = (
            rerank2.fit_ranker(
                final_train, final_rows, rerank2.ALL_FEATURES
            )
        )
        final_scores = 0.5 * (
            rerank2.minmax_by_user(
                final_rows, final_classifier_scores
            )
            + rerank2.minmax_by_user(
                final_rows, final_ranker_scores
            )
        )
        recommendations = (
            rerank.recommendations_from_probabilities(
                final_rows, final_scores
            )
        )
        checks = write_submission(
            recommendations,
            users,
            reference,
            final_fold["pool"],
            final_fold["fill"],
        )
        final_elapsed = time.perf_counter() - final_started
        print(
            "final training folds: train1, train2, eval, fold4, "
            "late1, late2, late3"
        )
        print(
            f"final training rows: {len(final_train):,}; "
            f"classifier fit {classifier_seconds:.2f}s; "
            f"ranker fit {ranker_seconds:.2f}s"
        )
        print(
            "final checks: "
            + ", ".join(
                f"{name}={'OK' if passed else 'FAIL'}"
                for name, passed in checks.items()
            )
        )
        print(
            f"wrote {SUBMISSION} ({len(reference):,} rows); "
            f"final time {final_elapsed:.2f}s"
        )
        if not all(checks.values()):
            raise SystemExit(1)
    else:
        print("no rerank5 submission: CI condition not met")
    print(
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
