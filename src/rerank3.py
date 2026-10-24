import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rerank
import rerank2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REFERENCE = rerank2.REFERENCE
SUBMISSION = Path("submissions/rerank3.csv")


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
    started = time.perf_counter()
    interactions = rerank.load_interactions()
    occurrences = pd.read_parquet(
        rerank.DATA / "occurrences.parquet"
    )
    reference = pd.read_csv(
        REFERENCE, dtype={"user_id": "uint64"}
    )
    users = [int(user) for user in reference["user_id"]]
    raw_views = rerank2.load_raw_views()
    view_aggregates = rerank.load_view_aggregates()
    view_aggregates["train2"] = (
        rerank2.build_view_aggregate(
            raw_views, rerank2.TRAIN2_HISTORY
        )
    )
    view_aggregates["fold4"] = (
        rerank2.build_view_aggregate(
            raw_views, rerank2.FOLD4_HISTORY
        )
    )

    fold_started = time.perf_counter()
    train1_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        rerank2.TRAIN_SCHEDULE,
        rerank.TRAIN_TARGET,
        rerank.TRAIN_HISTORY,
        users,
        "train",
    )
    eval_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        rerank2.EVAL_SCHEDULE,
        rerank.EVAL_TARGET,
        rerank.EVAL_HISTORY,
        users,
        "eval",
    )
    train2_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        rerank2.TRAIN_SCHEDULE,
        rerank2.TRAIN2_TARGET,
        rerank2.TRAIN2_HISTORY,
        users,
        "train2",
    )
    fold4_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        rerank2.EVAL_SCHEDULE,
        rerank2.FOLD4_TARGET,
        rerank2.FOLD4_HISTORY,
        users,
        "fold4",
    )
    fold_elapsed = time.perf_counter() - fold_started

    slot_data = {
        "train": rerank2.build_slot_data(
            raw_views, rerank.TRAIN_HISTORY
        ),
        "train2": rerank2.build_slot_data(
            raw_views, rerank2.TRAIN2_HISTORY
        ),
        "eval": rerank2.build_slot_data(
            raw_views, rerank.EVAL_HISTORY
        ),
    }
    target_slots = {
        "train": rerank2.load_item_slots(
            rerank2.TRAIN_SCHEDULE, rerank.TRAIN_TARGET
        ),
        "train2": rerank2.load_item_slots(
            rerank2.TRAIN_SCHEDULE, rerank2.TRAIN2_TARGET
        ),
        "eval": rerank2.load_item_slots(
            rerank2.EVAL_SCHEDULE, rerank.EVAL_TARGET
        ),
    }
    train1_slot = rerank2.add_slots_to_fold(
        train1_base, slot_data["train"], target_slots["train"]
    )
    train2_slot = rerank2.add_slots_to_fold(
        train2_base, slot_data["train2"], target_slots["train2"]
    )
    eval_slot = rerank2.add_slots_to_fold(
        eval_base, slot_data["eval"], target_slots["eval"]
    )

    model_started = time.perf_counter()
    train1_slot_rows = rerank2.training_frame(train1_slot)
    train2_slot_rows = rerank2.training_frame(train2_slot)
    eval_slot_rows = eval_slot["features"]
    slot_train_rows = pd.concat(
        [train1_slot_rows, train2_slot_rows], ignore_index=True
    )
    slot_clf_scores, _, slot_clf_seconds = (
        rerank2.fit_classifier(
            slot_train_rows,
            eval_slot_rows,
            rerank2.ALL_FEATURES,
        )
    )
    slot_rank_scores, _, slot_rank_seconds = (
        rerank2.fit_ranker(
            slot_train_rows,
            eval_slot_rows,
            rerank2.ALL_FEATURES,
        )
    )
    slot_average_scores = 0.5 * (
        rerank2.minmax_by_user(
            eval_slot_rows, slot_clf_scores
        )
        + rerank2.minmax_by_user(
            eval_slot_rows, slot_rank_scores
        )
    )
    slot_result = rerank2.evaluate_scores(
        "C with slot features",
        eval_slot_rows,
        slot_average_scores,
        users,
        eval_base["truth"],
        eval_base["fill"],
    )

    train1_base_rows = rerank2.training_frame(train1_base)
    train2_base_rows = rerank2.training_frame(train2_base)
    eval_base_rows = eval_base["features"]
    base_train_rows = pd.concat(
        [train1_base_rows, train2_base_rows],
        ignore_index=True,
    )
    base_clf_scores, _, base_clf_seconds = (
        rerank2.fit_classifier(
            base_train_rows,
            eval_base_rows,
            rerank2.BASE_FEATURES,
        )
    )
    base_rank_scores, _, base_rank_seconds = (
        rerank2.fit_ranker(
            base_train_rows,
            eval_base_rows,
            rerank2.BASE_FEATURES,
        )
    )
    base_average_scores = 0.5 * (
        rerank2.minmax_by_user(
            eval_base_rows, base_clf_scores
        )
        + rerank2.minmax_by_user(
            eval_base_rows, base_rank_scores
        )
    )
    base_result = rerank2.evaluate_scores(
        "C without slot features",
        eval_base_rows,
        base_average_scores,
        users,
        eval_base["truth"],
        eval_base["fill"],
    )
    model_elapsed = time.perf_counter() - model_started

    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        base_result["aps"] - slot_result["aps"]
    )
    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs C"
    )
    print(
        f"C with slot features | {slot_result['map5']:.5f} | "
        f"{slot_result['map5'] - slot_result['map5']:+.5f}"
    )
    print(
        f"C without slot features | {base_result['map5']:.5f} | "
        f"{base_result['map5'] - slot_result['map5']:+.5f}"
    )
    print(
        f"without-slot C vs C: mean diff {mean_diff:+.5f}, "
        f"95% CI [{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )
    print(
        f"slot model fit times: classifier {slot_clf_seconds:.2f}s, "
        f"ranker {slot_rank_seconds:.2f}s"
    )
    print(
        f"base model fit times: classifier {base_clf_seconds:.2f}s, "
        f"ranker {base_rank_seconds:.2f}s"
    )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "without-slot C beats C with CI above zero; "
            "building final submission"
        )
        final_started = time.perf_counter()
        final_base = rerank.build_fold(
            interactions,
            occurrences,
            view_aggregates,
            rerank2.FINAL_SCHEDULE,
            rerank2.FINAL_TARGET,
            rerank2.FINAL_HISTORY,
            users,
            "final",
        )
        final_rows = final_base["features"]
        train3_rows = rerank2.training_frame(eval_base)
        train4_rows = rerank2.training_frame(fold4_base)
        all_train = pd.concat(
            [
                train1_base_rows,
                train2_base_rows,
                train3_rows,
                train4_rows,
            ],
            ignore_index=True,
        )
        final_clf_scores, _, final_clf_seconds = (
            rerank2.fit_classifier(
                all_train,
                final_rows,
                rerank2.BASE_FEATURES,
            )
        )
        final_rank_scores, _, final_rank_seconds = (
            rerank2.fit_ranker(
                all_train,
                final_rows,
                rerank2.BASE_FEATURES,
            )
        )
        final_scores = 0.5 * (
            rerank2.minmax_by_user(
                final_rows, final_clf_scores
            )
            + rerank2.minmax_by_user(
                final_rows, final_rank_scores
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
            final_base["pool"],
            final_base["fill"],
        )
        final_elapsed = time.perf_counter() - final_started
        print(
            "final checks: "
            + ", ".join(
                f"{name}={'OK' if passed else 'FAIL'}"
                for name, passed in checks.items()
            )
        )
        print(
            f"wrote {SUBMISSION} ({len(reference):,} rows); "
            f"final time {final_elapsed:.2f}s; "
            f"classifier {final_clf_seconds:.2f}s; "
            f"ranker {final_rank_seconds:.2f}s"
        )
        if not all(checks.values()):
            raise SystemExit(1)
    else:
        print(
            "no rerank3 submission: CI condition not met"
        )
    print(
        f"total runtime: {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
