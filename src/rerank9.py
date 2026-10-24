import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker, early_stopping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rerank
import rerank2
import rerank6

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_SCHEDULE = "export_arh_11-20-final.csv"
EVAL_SCHEDULE = "export_arh_21-30-final.csv"
FINAL_SCHEDULE = "export_arh_31-42-final.csv"
REFERENCE = Path("submissions/mf_blend.csv")
SUBMISSION = Path("submissions/rerank9.csv")

BASE_SPECS = [
    ("train1", TRAIN_SCHEDULE, (16, 20), (11, 15)),
    ("train2", TRAIN_SCHEDULE, (18, 20), (11, 17)),
]
EVAL_SPEC = ("eval", EVAL_SCHEDULE, (21, 30), (11, 20))
FOLD4_SPEC = ("fold4", EVAL_SCHEDULE, (26, 30), (11, 25))
FINAL_SPEC = ("final", FINAL_SCHEDULE, (31, 42), (11, 30))
INTERACTION_FEATURES = [
    "repeat_ar_interaction",
    "expected_future_qualified_airings",
    "median_programme_duration",
]
COMBINED_FEATURES = (
    rerank2.ALL_FEATURES
    + rerank6.AR_FEATURES
    + INTERACTION_FEATURES
)
WEIGHTS = (0.1, 0.2, 0.35, 0.35)
SEED = 0


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


def rank_frame(frame, scores=None, columns=None):
    if columns is not None:
        ordered = frame[["user_id", "tv_show_id"] + columns].copy()
        ordered.sort_values(
            ["user_id"] + columns,
            ascending=[True] + [False] * (len(columns) - 1) + [True],
            inplace=True,
        )
    else:
        ordered = frame[["user_id", "tv_show_id"]].copy()
        ordered["score"] = np.asarray(scores, dtype=np.float64)
        ordered.sort_values(
            ["user_id", "score", "tv_show_id"],
            ascending=[True, False, True],
            inplace=True,
        )
    ordered["rank"] = (
        ordered.groupby("user_id", sort=False).cumcount() + 1
    )
    return ordered[["user_id", "tv_show_id", "rank"]]


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
    histories = {}
    target_airings_cache = {}
    duration_cache = {}

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

    def get_history_airings(schedule, history):
        key = (schedule, tuple(history))
        if key not in histories:
            frame = pd.read_csv(
                rerank.DATA / schedule,
                usecols=["start_time", "tv_show_id"],
                parse_dates=["start_time"],
                dayfirst=True,
                dtype={"tv_show_id": "int64"},
            )
            week = frame["start_time"].dt.isocalendar().week
            frame = frame[
                (frame["tv_show_id"] > 0)
                & week.between(*history)
            ]
            histories[key] = (
                frame.groupby("tv_show_id").size().to_dict()
            )
        return histories[key]

    def get_target_airings(schedule, target):
        key = (schedule, tuple(target))
        if key not in target_airings_cache:
            frame = pd.read_csv(
                rerank.DATA / schedule,
                usecols=["start_time", "tv_show_id"],
                parse_dates=["start_time"],
                dayfirst=True,
                dtype={"tv_show_id": "int64"},
            )
            week = frame["start_time"].dt.isocalendar().week
            frame = frame[
                (frame["tv_show_id"] > 0)
                & week.between(*target)
            ]
            target_airings_cache[key] = (
                frame.groupby("tv_show_id").size().to_dict()
            )
        return target_airings_cache[key]

    def get_median_duration(schedule, target):
        key = (schedule, tuple(target))
        if key not in duration_cache:
            frame = pd.read_csv(
                rerank.DATA / schedule,
                usecols=["start_time", "tv_show_id", "duration"],
                parse_dates=["start_time"],
                dayfirst=True,
                dtype={
                    "tv_show_id": "int64",
                    "duration": "float64",
                },
            )
            week = frame["start_time"].dt.isocalendar().week
            frame = frame[
                (frame["tv_show_id"] > 0)
                & week.between(*target)
            ]
            duration_cache[key] = (
                frame.groupby("tv_show_id")["duration"]
                .median()
                .to_dict()
            )
        return duration_cache[key]

    def build_base_fold(spec):
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

    def build_combined_fold(spec):
        name, schedule, target, history = spec
        old = (
            rerank.REPEAT_DEPTH,
            rerank.POP_DEPTH,
            rerank.MF_DEPTH,
        )
        rerank.REPEAT_DEPTH = 100
        rerank.POP_DEPTH = 40
        rerank.MF_DEPTH = 20
        try:
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
        finally:
            (
                rerank.REPEAT_DEPTH,
                rerank.POP_DEPTH,
                rerank.MF_DEPTH,
            ) = old
        base = rerank2.add_slots_to_fold(
            base,
            get_slot_data(history),
            get_target_slots(schedule, target),
        )
        rules, baskets, _ = rerank6.build_ar_rules(
            occurrences, history, users
        )
        state = rerank6.build_ar_state(
            users, rules, baskets, base["pool"]
        )
        rerank6.AR_TOP = 100
        ar_candidates = rerank6.build_ar_candidates(state, users)
        candidates = pd.concat(
            [base["candidates"], ar_candidates],
            ignore_index=True,
        ).drop_duplicates(["user_id", "tv_show_id"])
        features = rerank.build_features(
            candidates,
            base["occurrence_stats"],
            base["repeat_rankings"],
            base["popularity_frame"],
            base["schedule_stats"],
            base["mf_scores"],
            base["mf_ranks"],
            base["mf_valid"],
            base["user_programmes"],
            base["user_hours"],
            users,
            base["pool"],
            base["show_channels"],
            base["channel_shares"],
        )
        features = rerank6.add_ar_features(features, state)
        history_airings = get_history_airings(schedule, history)
        target_airings = get_target_airings(schedule, target)
        durations = get_median_duration(schedule, target)
        features["repeat_ar_interaction"] = (
            features["repeat_count_ge_030"]
            * features["ar_max_confidence"]
        )
        expected = np.zeros(len(features), dtype=np.float64)
        medians = np.zeros(len(features), dtype=np.float64)
        for row, (item, count) in enumerate(
            zip(
                features["tv_show_id"],
                features["repeat_count_ge_030"],
            )
        ):
            item = int(item)
            hist = history_airings.get(item, 0)
            target_count = target_airings.get(item, 0)
            if hist > 0:
                expected[row] = count * target_count / hist
            medians[row] = durations.get(item, 0.0)
        features["expected_future_qualified_airings"] = expected
        features["median_programme_duration"] = medians
        features = rerank.add_labels(features, base["truth"])
        updated = dict(base)
        updated["candidates"] = candidates
        updated["features"] = features
        return rerank2.add_slots_to_fold(
            updated,
            get_slot_data(history),
            get_target_slots(schedule, target),
        )

    fold_started = time.perf_counter()
    base_train1 = build_base_fold(BASE_SPECS[0])
    base_train2 = build_base_fold(BASE_SPECS[1])
    base_eval = build_base_fold(EVAL_SPEC)
    train1 = build_combined_fold(BASE_SPECS[0])
    train2 = build_combined_fold(BASE_SPECS[1])
    combined_eval = build_combined_fold(EVAL_SPEC)
    fold_elapsed = time.perf_counter() - fold_started

    base_rows = pd.concat(
        [
            rerank2.training_frame(base_train1),
            rerank2.training_frame(base_train2),
        ],
        ignore_index=True,
    )
    base_eval_rows = base_eval["features"]
    base_truth = base_eval["truth"]
    base_fill = base_eval["fill"]
    combined_rows = pd.concat(
        [
            rerank2.training_frame(train1),
            rerank2.training_frame(train2),
        ],
        ignore_index=True,
    )
    combined_eval_rows = combined_eval["features"]
    combined_truth = combined_eval["truth"]
    combined_fill = combined_eval["fill"]

    model_started = time.perf_counter()
    base_classifier_scores, _, _ = rerank2.fit_classifier(
        base_rows, base_eval_rows, rerank2.ALL_FEATURES
    )
    base_ranker_scores, _, _ = rerank2.fit_ranker(
        base_rows, base_eval_rows, rerank2.ALL_FEATURES
    )
    base_scores = 0.5 * (
        rerank2.minmax_by_user(
            base_eval_rows, base_classifier_scores
        )
        + rerank2.minmax_by_user(
            base_eval_rows, base_ranker_scores
        )
    )
    base_result = rerank2.evaluate_scores(
        "base C",
        base_eval_rows,
        base_scores,
        users,
        base_truth,
        base_fill,
    )

    unique_users = np.array(
        sorted(combined_rows["user_id"].unique())
    )
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(unique_users)
    holdout_count = max(1, int(round(0.2 * len(shuffled))))
    holdout_users = set(
        int(user) for user in shuffled[:holdout_count]
    )
    fit_mask = ~combined_rows["user_id"].isin(holdout_users)
    fit_rows = combined_rows[fit_mask].reset_index(drop=True)
    valid_rows = combined_rows[~fit_mask].reset_index(drop=True)

    x_fit = fit_rows[COMBINED_FEATURES].to_numpy(dtype=np.float64)
    y_fit = fit_rows["label"].to_numpy(dtype=np.int8)
    x_valid = valid_rows[COMBINED_FEATURES].to_numpy(
        dtype=np.float64
    )
    y_valid = valid_rows["label"].to_numpy(dtype=np.int8)
    x_eval = combined_eval_rows[COMBINED_FEATURES].to_numpy(
        dtype=np.float64
    )

    classifier = LGBMClassifier(
        n_estimators=1000,
        num_leaves=31,
        learning_rate=0.03,
        lambda_l2=10,
        min_data_in_leaf=50,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    classifier.fit(
        x_fit,
        y_fit,
        eval_set=[(x_valid, y_valid)],
        callbacks=[early_stopping(50, verbose=False)],
    )
    classifier_scores = classifier.predict_proba(
        x_eval, num_iteration=classifier.best_iteration_ or None
    )[:, 1]

    rank_fit = fit_rows.sort_values(
        ["user_id", "tv_show_id"], kind="mergesort"
    ).reset_index(drop=True)
    rank_valid = valid_rows.sort_values(
        ["user_id", "tv_show_id"], kind="mergesort"
    ).reset_index(drop=True)
    fit_groups = (
        rank_fit.groupby("user_id", sort=False).size().to_numpy()
    )
    valid_groups = (
        rank_valid.groupby("user_id", sort=False).size().to_numpy()
    )
    ranker = LGBMRanker(
        n_estimators=1000,
        num_leaves=31,
        learning_rate=0.03,
        lambda_l2=10,
        min_data_in_leaf=50,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    ranker.fit(
        rank_fit[COMBINED_FEATURES].to_numpy(dtype=np.float64),
        rank_fit["label"].to_numpy(dtype=np.int8),
        group=fit_groups,
        eval_set=[
            (
                rank_valid[COMBINED_FEATURES].to_numpy(
                    dtype=np.float64
                ),
                rank_valid["label"].to_numpy(dtype=np.int8),
            )
        ],
        eval_group=[valid_groups],
        callbacks=[early_stopping(50, verbose=False)],
    )
    ranker_scores = ranker.predict(
        x_eval, num_iteration=ranker.best_iteration_ or None
    )
    logistic_scores, _, _, _ = rerank.fit_and_predict(
        base_rows,
        combined_eval_rows,
        "logistic",
        c_value=1.0,
    )
    repeat_ranks = rank_frame(
        combined_eval_rows,
        columns=[
            "repeat_count_ge_030",
            "repeat_count_ge_050",
            "days_since_last_qualifying",
        ],
    )
    logistic_ranks = rank_frame(
        combined_eval_rows, logistic_scores
    )
    classifier_ranks = rank_frame(
        combined_eval_rows, classifier_scores
    )
    ranker_ranks = rank_frame(
        combined_eval_rows, ranker_scores
    )
    ranked = combined_eval_rows[["user_id", "tv_show_id"]].copy()
    for name, ranks in (
        ("repeat", repeat_ranks),
        ("logistic", logistic_ranks),
        ("classifier", classifier_ranks),
        ("ranker", ranker_ranks),
    ):
        ranked = ranked.merge(
            ranks.rename(columns={"rank": f"{name}_rank"}),
            on=["user_id", "tv_show_id"],
            how="left",
        )
    fused = (
        WEIGHTS[0] * ranked["repeat_rank"].to_numpy()
        + WEIGHTS[1] * ranked["logistic_rank"].to_numpy()
        + WEIGHTS[2] * ranked["classifier_rank"].to_numpy()
        + WEIGHTS[3] * ranked["ranker_rank"].to_numpy()
    )
    recommendations = rerank.recommendations_from_probabilities(
        combined_eval_rows, -fused
    )
    combined_result = rerank.evaluate_recommendations(
        recommendations,
        users,
        combined_truth,
        combined_fill,
    )
    combined_result["name"] = "combined"
    combined_result["aps"] = combined_result["aps"]
    model_elapsed = time.perf_counter() - model_started

    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs base"
    )
    for result in (base_result, combined_result):
        print(
            f"{result['name']} | {result['map5']:.5f} | "
            f"{result['map5'] - base_result['map5']:+.5f}"
        )
    print(
        f"combined training rows: {len(combined_rows):,}; "
        f"candidates per user: "
        f"{len(combined_eval_rows) / len(users):.2f}; "
        f"classifier best iter {classifier.best_iteration_ or None}; "
        f"ranker best iter {ranker.best_iteration_ or None}"
    )
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )
    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        combined_result["aps"] - base_result["aps"],
        n_resamples=2000,
        seed=0,
    )
    print(
        f"combined vs base: mean diff {mean_diff:+.5f}, "
        f"95% CI [{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "combined beats base with CI above zero; "
            "building final submission"
        )
        final_started = time.perf_counter()
        base_fold4 = build_base_fold(FOLD4_SPEC)
        fold4 = build_combined_fold(FOLD4_SPEC)
        final_fold = build_combined_fold(FINAL_SPEC)
        four_base = [
            base_train1,
            base_train2,
            base_eval,
            base_fold4,
        ]
        four_combined = [train1, train2, combined_eval, fold4]
        base_four_rows = pd.concat(
            [rerank2.training_frame(fold) for fold in four_base],
            ignore_index=True,
        )
        combined_four_rows = pd.concat(
            [
                rerank2.training_frame(fold)
                for fold in four_combined
            ],
            ignore_index=True,
        )
        final_frame = final_fold["features"]
        unique_final = np.array(
            sorted(combined_four_rows["user_id"].unique())
        )
        rng_final = np.random.default_rng(SEED)
        shuffled_final = rng_final.permutation(unique_final)
        final_holdout = max(
            1, int(round(0.2 * len(shuffled_final)))
        )
        final_users = set(
            int(user)
            for user in shuffled_final[:final_holdout]
        )
        final_fit_mask = ~combined_four_rows["user_id"].isin(
            final_users
        )
        final_fit = combined_four_rows[
            final_fit_mask
        ].reset_index(drop=True)
        final_valid = combined_four_rows[
            ~final_fit_mask
        ].reset_index(drop=True)
        classifier_final = LGBMClassifier(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.03,
            lambda_l2=10,
            min_data_in_leaf=50,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        classifier_final.fit(
            final_fit[COMBINED_FEATURES].to_numpy(
                dtype=np.float64
            ),
            final_fit["label"].to_numpy(dtype=np.int8),
            eval_set=[
                (
                    final_valid[COMBINED_FEATURES].to_numpy(
                        dtype=np.float64
                    ),
                    final_valid["label"].to_numpy(dtype=np.int8),
                )
            ],
            callbacks=[early_stopping(50, verbose=False)],
        )
        final_classifier_scores = classifier_final.predict_proba(
            final_frame[COMBINED_FEATURES].to_numpy(
                dtype=np.float64
            ),
            num_iteration=classifier_final.best_iteration_ or None,
        )[:, 1]
        rank_final = final_fit.sort_values(
            ["user_id", "tv_show_id"], kind="mergesort"
        ).reset_index(drop=True)
        valid_final = final_valid.sort_values(
            ["user_id", "tv_show_id"], kind="mergesort"
        ).reset_index(drop=True)
        final_groups = (
            rank_final.groupby("user_id", sort=False)
            .size()
            .to_numpy()
        )
        valid_final_groups = (
            valid_final.groupby("user_id", sort=False)
            .size()
            .to_numpy()
        )
        ranker_final = LGBMRanker(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.03,
            lambda_l2=10,
            min_data_in_leaf=50,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        ranker_final.fit(
            rank_final[COMBINED_FEATURES].to_numpy(
                dtype=np.float64
            ),
            rank_final["label"].to_numpy(dtype=np.int8),
            group=final_groups,
            eval_set=[
                (
                    valid_final[COMBINED_FEATURES].to_numpy(
                        dtype=np.float64
                    ),
                    valid_final["label"].to_numpy(dtype=np.int8),
                )
            ],
            eval_group=[valid_final_groups],
            callbacks=[early_stopping(50, verbose=False)],
        )
        final_ranker_scores = ranker_final.predict(
            final_frame[COMBINED_FEATURES].to_numpy(
                dtype=np.float64
            ),
            num_iteration=ranker_final.best_iteration_ or None,
        )
        final_logistic_scores, _, _, _ = (
            rerank.fit_and_predict(
                base_four_rows,
                final_frame,
                "logistic",
                c_value=1.0,
            )
        )
        final_repeat_ranks = rank_frame(
            final_frame,
            columns=[
                "repeat_count_ge_030",
                "repeat_count_ge_050",
                "days_since_last_qualifying",
            ],
        )
        final_logistic_ranks = rank_frame(
            final_frame, final_logistic_scores
        )
        final_classifier_ranks = rank_frame(
            final_frame, final_classifier_scores
        )
        final_ranker_ranks = rank_frame(
            final_frame, final_ranker_scores
        )
        final_ranked = final_frame[
            ["user_id", "tv_show_id"]
        ].copy()
        for name, ranks in (
            ("repeat", final_repeat_ranks),
            ("logistic", final_logistic_ranks),
            ("classifier", final_classifier_ranks),
            ("ranker", final_ranker_ranks),
        ):
            final_ranked = final_ranked.merge(
                ranks.rename(columns={"rank": f"{name}_rank"}),
                on=["user_id", "tv_show_id"],
                how="left",
            )
        final_fused = (
            WEIGHTS[0] * final_ranked["repeat_rank"].to_numpy()
            + WEIGHTS[1]
            * final_ranked["logistic_rank"].to_numpy()
            + WEIGHTS[2]
            * final_ranked["classifier_rank"].to_numpy()
            + WEIGHTS[3]
            * final_ranked["ranker_rank"].to_numpy()
        )
        final_recommendations = (
            rerank.recommendations_from_probabilities(
                final_frame, -final_fused
            )
        )
        checks = write_submission(
            final_recommendations,
            users,
            reference,
            final_fold["pool"],
            final_fold["fill"],
        )
        final_elapsed = time.perf_counter() - final_started
        print(
            f"final combined training rows: "
            f"{len(combined_four_rows):,}; "
            f"classifier best iter "
            f"{classifier_final.best_iteration_ or None}; ranker best iter "
            f"{ranker_final.best_iteration_ or None}"
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
        print("no rerank9 submission: CI condition not met")
    print(
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
