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

REFERENCE = rerank2.REFERENCE
SUBMISSION = Path("submissions/rerank_bag.csv")
SEEDS = (0, 1, 2, 3, 4)


def fit_classifier_seed(train_frame, eval_frame, features, seed):
    started = time.perf_counter()
    x_train = train_frame[features].to_numpy(dtype=np.float64)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    x_eval = eval_frame[features].to_numpy(dtype=np.float64)
    model = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_eval)[:, 1]
    return scores, model, time.perf_counter() - started


def fit_ranker_seed(train_frame, eval_frame, features, seed):
    started = time.perf_counter()
    frame = train_frame.sort_values(
        ["user_id", "tv_show_id"], kind="mergesort"
    ).reset_index(drop=True)
    groups = frame.groupby("user_id", sort=False).size().to_numpy()
    x_train = frame[features].to_numpy(dtype=np.float64)
    y_train = frame["label"].to_numpy(dtype=np.int8)
    x_eval = eval_frame[features].to_numpy(dtype=np.float64)
    model = LGBMRanker(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        min_data_in_leaf=50,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, group=groups)
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
    train1 = rerank2.add_slots_to_fold(
        train1_base, slot_data["train"], target_slots["train"]
    )
    train2 = rerank2.add_slots_to_fold(
        train2_base, slot_data["train2"], target_slots["train2"]
    )
    eval_slot = rerank2.add_slots_to_fold(
        eval_base, slot_data["eval"], target_slots["eval"]
    )

    train1_rows = rerank2.training_frame(train1)
    train2_rows = rerank2.training_frame(train2)
    eval_rows = eval_slot["features"]
    train_rows = pd.concat(
        [train1_rows, train2_rows], ignore_index=True
    )

    model_started = time.perf_counter()
    c_clf_scores, _, c_clf_seconds = rerank2.fit_classifier(
        train_rows, eval_rows, rerank2.ALL_FEATURES
    )
    c_rank_scores, _, c_rank_seconds = rerank2.fit_ranker(
        train_rows, eval_rows, rerank2.ALL_FEATURES
    )
    c_scores = 0.5 * (
        rerank2.minmax_by_user(eval_rows, c_clf_scores)
        + rerank2.minmax_by_user(eval_rows, c_rank_scores)
    )
    c_result = rerank2.evaluate_scores(
        "C",
        eval_rows,
        c_scores,
        users,
        eval_base["truth"],
        eval_base["fill"],
    )

    bag_scores = []
    seed_fit_seconds = []
    for seed in SEEDS:
        scores, _, fit_seconds = fit_classifier_seed(
            train_rows, eval_rows, rerank2.ALL_FEATURES, seed
        )
        bag_scores.append(
            rerank2.minmax_by_user(eval_rows, scores)
        )
        seed_fit_seconds.append(fit_seconds)
    for seed in SEEDS:
        scores, _, fit_seconds = fit_ranker_seed(
            train_rows, eval_rows, rerank2.ALL_FEATURES, seed
        )
        bag_scores.append(
            rerank2.minmax_by_user(eval_rows, scores)
        )
        seed_fit_seconds.append(fit_seconds)
    bagged_scores = np.mean(np.vstack(bag_scores), axis=0)
    bagged_result = rerank2.evaluate_scores(
        "seed-bagged C",
        eval_rows,
        bagged_scores,
        users,
        eval_base["truth"],
        eval_base["fill"],
    )
    model_elapsed = time.perf_counter() - model_started

    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        bagged_result["aps"] - c_result["aps"]
    )
    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs C"
    )
    print(
        f"C | {c_result['map5']:.5f} | "
        f"{c_result['map5'] - c_result['map5']:+.5f}"
    )
    print(
        f"seed-bagged C | {bagged_result['map5']:.5f} | "
        f"{bagged_result['map5'] - c_result['map5']:+.5f}"
    )
    print(
        f"seed-bagged C vs C: mean diff {mean_diff:+.5f}, "
        f"95% CI [{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s; "
        f"C classifier {c_clf_seconds:.2f}s; "
        f"C ranker {c_rank_seconds:.2f}s"
    )
    print(
        "bag model fit times: "
        + ", ".join(f"{value:.2f}s" for value in seed_fit_seconds)
    )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "seed-bagged C beats C with CI above zero; "
            "building final submission"
        )
        final_started = time.perf_counter()
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
        slot_data["fold4"] = rerank2.build_slot_data(
            raw_views, rerank2.FOLD4_HISTORY
        )
        slot_data["final"] = rerank2.build_slot_data(
            raw_views, rerank2.FINAL_HISTORY
        )
        target_slots["fold4"] = rerank2.load_item_slots(
            rerank2.EVAL_SCHEDULE, rerank2.FOLD4_TARGET
        )
        target_slots["final"] = rerank2.load_item_slots(
            rerank2.FINAL_SCHEDULE, rerank2.FINAL_TARGET
        )
        fold4 = rerank2.add_slots_to_fold(
            fold4_base,
            slot_data["fold4"],
            target_slots["fold4"],
        )
        final_fold = rerank2.add_slots_to_fold(
            final_base,
            slot_data["final"],
            target_slots["final"],
        )
        train4_rows = rerank2.training_frame(fold4)
        final_rows = final_fold["features"]
        all_train = pd.concat(
            [
                train1_rows,
                train2_rows,
                rerank2.training_frame(eval_slot),
                train4_rows,
            ],
            ignore_index=True,
        )
        final_scores = []
        for seed in SEEDS:
            scores, _, _ = fit_classifier_seed(
                all_train, final_rows, rerank2.ALL_FEATURES, seed
            )
            final_scores.append(
                rerank2.minmax_by_user(final_rows, scores)
            )
        for seed in SEEDS:
            scores, _, _ = fit_ranker_seed(
                all_train, final_rows, rerank2.ALL_FEATURES, seed
            )
            final_scores.append(
                rerank2.minmax_by_user(final_rows, scores)
            )
        final_scores = np.mean(np.vstack(final_scores), axis=0)
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
        print(
            "no rerank_bag submission: CI condition not met"
        )
    print(
        f"total runtime: {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
