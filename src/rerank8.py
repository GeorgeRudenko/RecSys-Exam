import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rerank
import rerank2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_SCHEDULE = "export_arh_11-20-final.csv"
EVAL_SCHEDULE = "export_arh_21-30-final.csv"
FINAL_SCHEDULE = "export_arh_31-42-final.csv"
REFERENCE = Path("submissions/mf_blend.csv")
SUBMISSION = Path("submissions/rerank8.csv")

BASE_SPECS = [
    ("train1", TRAIN_SCHEDULE, (16, 20), (11, 15)),
    ("train2", TRAIN_SCHEDULE, (18, 20), (11, 17)),
]
EVAL_SPEC = ("eval", EVAL_SCHEDULE, (21, 30), (11, 20))
FOLD4_SPEC = ("fold4", EVAL_SCHEDULE, (26, 30), (11, 25))
FINAL_SPEC = ("final", FINAL_SCHEDULE, (31, 42), (11, 30))
SEED = 0


def build_target_fractions(occurrences, target_weeks, users, pool):
    subset = occurrences[
        occurrences["iso_week"].between(*target_weeks)
        & occurrences["tv_show_id"].isin(pool)
        & occurrences["user_id"].isin(users)
    ]
    grouped = (
        subset.groupby(["user_id", "tv_show_id"], sort=False)[
            "fraction"
        ]
        .max()
        .reset_index()
    )
    return {
        (int(user), int(item)): float(fraction)
        for user, item, fraction in zip(
            grouped["user_id"],
            grouped["tv_show_id"],
            grouped["fraction"],
        )
    }


def make_labels(
    frame,
    fractions,
    scheme,
):
    max_values = np.fromiter(
        (
            fractions.get((int(user), int(item)), 0.0)
            for user, item in zip(
                frame["user_id"], frame["tv_show_id"]
            )
        ),
        dtype=np.float64,
        count=len(frame),
    )
    if scheme == "L1":
        labels = (max_values > 0.5).astype(np.int8)
        weights = np.ones(len(frame), dtype=np.float64)
    else:
        hard = frame["label"].to_numpy(dtype=np.int8).astype(bool)
        soft = (max_values > 0.5) & (max_values <= 0.8)
        labels = (hard | soft).astype(np.int8)
        weights = np.where(soft, 0.5, 1.0).astype(np.float64)
    return labels, weights


def fit_classifier(
    train_frame,
    eval_frame,
    features,
    labels,
    weights,
):
    started = time.perf_counter()
    x_train = train_frame[features].to_numpy(dtype=np.float64)
    x_eval = eval_frame[features].to_numpy(dtype=np.float64)
    model = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        labels,
        sample_weight=weights,
    )
    scores = model.predict_proba(x_eval)[:, 1]
    return scores, model, time.perf_counter() - started


def fit_ranker(
    train_frame,
    eval_frame,
    features,
    labels,
    weights,
):
    started = time.perf_counter()
    frame = train_frame[
        ["user_id", "tv_show_id"]
    ].copy()
    frame["_label"] = labels
    frame["_weight"] = weights
    order = np.lexsort(
        (
            frame["tv_show_id"].to_numpy(),
            frame["user_id"].to_numpy(),
        )
    )
    frame = frame.iloc[order].reset_index(drop=True)
    groups = frame.groupby("user_id", sort=False).size().to_numpy()
    x_train = train_frame[features].to_numpy(dtype=np.float64)[order]
    x_eval = eval_frame[features].to_numpy(dtype=np.float64)
    model = LGBMRanker(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        min_data_in_leaf=50,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        frame["_label"].to_numpy(dtype=np.int8),
        group=groups,
        sample_weight=frame["_weight"].to_numpy(
            dtype=np.float64
        ),
    )
    scores = model.predict(x_eval)
    return scores, model, time.perf_counter() - started


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
    train1 = build_fold(BASE_SPECS[0])
    train2 = build_fold(BASE_SPECS[1])
    eval_fold = build_fold(EVAL_SPEC)
    fold_elapsed = time.perf_counter() - fold_started

    train1_rows = rerank2.training_frame(train1)
    train2_rows = rerank2.training_frame(train2)
    base_rows = pd.concat(
        [train1_rows, train2_rows], ignore_index=True
    )
    eval_rows = eval_fold["features"]
    eval_truth = eval_fold["truth"]
    eval_fill = eval_fold["fill"]
    max_fractions = {
        "train1": build_target_fractions(
            occurrences,
            BASE_SPECS[0][2],
            users,
            train1["pool"],
        ),
        "train2": build_target_fractions(
            occurrences,
            BASE_SPECS[1][2],
            users,
            train2["pool"],
        ),
    }
    label_data = {}
    for name, rows in (
        ("train1", train1_rows),
        ("train2", train2_rows),
    ):
        label_data[name] = {
            "L1": make_labels(
                rows, max_fractions[name], "L1"
            ),
            "L2": make_labels(
                rows, max_fractions[name], "L2"
            ),
        }

    model_started = time.perf_counter()

    def evaluate(
        name,
        scheme=None,
    ):
        if scheme is None:
            labels = base_rows["label"].to_numpy(dtype=np.int8)
            weights = np.ones(len(base_rows), dtype=np.float64)
        else:
            labels = np.concatenate(
                [
                    label_data["train1"][scheme][0],
                    label_data["train2"][scheme][0],
                ]
            )
            weights = np.concatenate(
                [
                    label_data["train1"][scheme][1],
                    label_data["train2"][scheme][1],
                ]
            )
        classifier_scores, classifier, classifier_seconds = (
            fit_classifier(
                base_rows,
                eval_rows,
                rerank2.ALL_FEATURES,
                labels,
                weights,
            )
        )
        ranker_scores, ranker, ranker_seconds = fit_ranker(
            base_rows,
            eval_rows,
            rerank2.ALL_FEATURES,
            labels,
            weights,
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
        result["scores"] = scores
        result["classifier"] = classifier
        result["ranker"] = ranker
        result["fit_seconds"] = (
            classifier_seconds + ranker_seconds
        )
        result["labels"] = labels
        result["weights"] = weights
        return result

    base_result = evaluate("base C")
    l1_result = evaluate("L1 > 0.5", scheme="L1")
    l2_result = evaluate("L2 soft 0.5-0.8", scheme="L2")
    model_elapsed = time.perf_counter() - model_started

    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs base"
    )
    results = [base_result, l1_result, l2_result]
    for result in results:
        print(
            f"{result['name']} | {result['map5']:.5f} | "
            f"{result['map5'] - base_result['map5']:+.5f}"
        )
    for name in ("train1", "train2"):
        l1_labels = label_data[name]["L1"][0]
        l2_labels = label_data[name]["L2"][0]
        l2_weights = label_data[name]["L2"][1]
        print(
            f"{name} labels: L1 positives {int(l1_labels.sum()):,}; "
            f"L2 positives {int(l2_labels.sum()):,}; "
            f"soft positives {int((l2_weights == 0.5).sum()):,}"
        )
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )

    better = max(
        [l1_result, l2_result], key=lambda result: result["map5"]
    )
    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        better["aps"] - base_result["aps"],
        n_resamples=2000,
        seed=0,
    )
    print(
        f"better {better['name']} vs base: mean diff "
        f"{mean_diff:+.5f}, 95% CI "
        f"[{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "better label scheme beats base with CI above zero; "
            "building final submission"
        )
        scheme = "L1" if better is l1_result else "L2"
        final_started = time.perf_counter()
        fold4 = build_fold(FOLD4_SPEC)
        final_fold = build_fold(FINAL_SPEC)
        four_folds = [train1, train2, eval_fold, fold4]
        all_rows = pd.concat(
            [rerank2.training_frame(fold) for fold in four_folds],
            ignore_index=True,
        )
        all_labels = []
        all_weights = []
        for fold, spec in zip(
            four_folds,
            BASE_SPECS + [EVAL_SPEC, FOLD4_SPEC],
        ):
            fractions = build_target_fractions(
                occurrences, spec[2], users, fold["pool"]
            )
            labels, weights = make_labels(
                rerank2.training_frame(fold),
                fractions,
                scheme,
            )
            all_labels.append(labels)
            all_weights.append(weights)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights)
        final_rows = final_fold["features"]
        classifier_scores, _, classifier_seconds = (
            fit_classifier(
                all_rows,
                final_rows,
                rerank2.ALL_FEATURES,
                all_labels,
                all_weights,
            )
        )
        ranker_scores, _, ranker_seconds = fit_ranker(
            all_rows,
            final_rows,
            rerank2.ALL_FEATURES,
            all_labels,
            all_weights,
        )
        final_scores = 0.5 * (
            rerank2.minmax_by_user(
                final_rows, classifier_scores
            )
            + rerank2.minmax_by_user(
                final_rows, ranker_scores
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
            f"final scheme {scheme}; training rows "
            f"{len(all_rows):,}; classifier fit "
            f"{classifier_seconds:.2f}s; ranker fit "
            f"{ranker_seconds:.2f}s"
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
        print("no rerank8 submission: CI condition not met")
    print(
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
