import sys
import time
from collections import Counter, defaultdict
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
SUBMISSION = Path("submissions/rerank6.csv")

BASE_SPECS = [
    ("train1", TRAIN_SCHEDULE, (16, 20), (11, 15)),
    ("train2", TRAIN_SCHEDULE, (18, 20), (11, 17)),
]
EVAL_SPEC = ("eval", EVAL_SCHEDULE, (21, 30), (11, 20))
FOLD4_SPEC = ("fold4", EVAL_SCHEDULE, (26, 30), (11, 25))
FINAL_SPEC = ("final", FINAL_SCHEDULE, (31, 42), (11, 30))
AR_TOP = 50
AR_MIN_SUPPORT = 5
AR_MAX_CONSEQUENTS = 100
AR_BASKET_CAP = 50
AR_FRACTION = 0.30
AR_FEATURES = [
    "ar_max_confidence",
    "ar_sum_confidence",
    "ar_matched_antecedents",
    "ar_rank",
]
WEIGHTS = [
    (0.1, 0.4, 0.5),
    (0.0, 0.3, 0.7),
    (0.1, 0.2, 0.7),
    (0.2, 0.3, 0.5),
]


def build_ar_rules(occurrences, history, users):
    subset = occurrences[
        occurrences["iso_week"].between(*history)
        & (occurrences["fraction"] >= AR_FRACTION)
        & occurrences["user_id"].isin(users)
    ]
    grouped = (
        subset.groupby(["user_id", "tv_show_id"], sort=False)
        .agg(
            n_airings=("airing_start", "size"),
            fraction_sum=("fraction", "sum"),
        )
        .reset_index()
    )
    grouped.sort_values(
        [
            "user_id",
            "n_airings",
            "fraction_sum",
            "tv_show_id",
        ],
        ascending=[True, False, False, True],
        inplace=True,
    )
    grouped = grouped.groupby("user_id", sort=False).head(
        AR_BASKET_CAP
    )
    baskets = {
        int(user): [int(item) for item in group["tv_show_id"]]
        for user, group in grouped.groupby("user_id", sort=False)
    }
    item_support = Counter()
    pair_support = Counter()
    for items in baskets.values():
        for item in items:
            item_support[item] += 1
        for antecedent in items:
            for consequent in items:
                if antecedent != consequent:
                    pair_support[(antecedent, consequent)] += 1
    by_antecedent = defaultdict(list)
    for (antecedent, consequent), support in pair_support.items():
        if support < AR_MIN_SUPPORT:
            continue
        confidence = support / item_support[antecedent]
        by_antecedent[antecedent].append(
            (consequent, confidence, support)
        )
    rules = {}
    for antecedent, rows in by_antecedent.items():
        rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
        rules[antecedent] = rows[:AR_MAX_CONSEQUENTS]
    return rules, baskets, len(pair_support)


def build_ar_state(users, rules, baskets, pool):
    pool_set = set(pool)
    large_rank = len(pool) + 1
    scores = {}
    ranks = {}
    for user in users:
        user_scores = {}
        for antecedent in baskets.get(user, []):
            for consequent, confidence, support in rules.get(
                antecedent, ()
            ):
                if consequent not in pool_set:
                    continue
                current = user_scores.get(consequent)
                if current is None:
                    user_scores[consequent] = [
                        confidence,
                        confidence,
                        1,
                    ]
                else:
                    if confidence > current[0]:
                        current[0] = confidence
                    current[1] += confidence
                    current[2] += 1
        ordered = sorted(
            user_scores.items(),
            key=lambda row: (-row[1][0], -row[1][1], row[0]),
        )
        scores[user] = user_scores
        ranks[user] = {
            item: rank for rank, (item, _) in enumerate(ordered, 1)
        }
    return {
        "scores": scores,
        "ranks": ranks,
        "large_rank": large_rank,
    }


def build_ar_candidates(state, users):
    rows = []
    for user in users:
        ordered = sorted(
            state["scores"].get(user, {}).items(),
            key=lambda row: (-row[1][0], -row[1][1], row[0]),
        )
        for item, _ in ordered[:AR_TOP]:
            rows.append((user, item))
    return pd.DataFrame(
        {
            "user_id": np.asarray(
                [row[0] for row in rows], dtype=np.uint64
            ),
            "tv_show_id": np.asarray(
                [row[1] for row in rows], dtype=np.int64
            ),
        }
    )


def add_ar_features(frame, state):
    max_values = np.zeros(len(frame), dtype=np.float64)
    sum_values = np.zeros(len(frame), dtype=np.float64)
    count_values = np.zeros(len(frame), dtype=np.int16)
    rank_values = np.full(
        len(frame), state["large_rank"], dtype=np.int32
    )
    for index, (user, item) in enumerate(
        zip(frame["user_id"], frame["tv_show_id"])
    ):
        user = int(user)
        item = int(item)
        values = state["scores"].get(user, {}).get(item)
        if values is None:
            continue
        max_values[index] = values[0]
        sum_values[index] = values[1]
        count_values[index] = values[2]
        rank_values[index] = state["ranks"].get(user, {}).get(
            item, state["large_rank"]
        )
    frame = frame.copy()
    frame["ar_max_confidence"] = max_values
    frame["ar_sum_confidence"] = sum_values
    frame["ar_matched_antecedents"] = count_values
    frame["ar_rank"] = rank_values
    return frame


def augment_fold(fold, state, slot_data, target_slots):
    ar_candidates = build_ar_candidates(state, list(fold["truth"]))
    candidates = pd.concat(
        [fold["candidates"], ar_candidates], ignore_index=True
    ).drop_duplicates(["user_id", "tv_show_id"])
    features = rerank.build_features(
        candidates,
        fold["occurrence_stats"],
        fold["repeat_rankings"],
        fold["popularity_frame"],
        fold["schedule_stats"],
        fold["mf_scores"],
        fold["mf_ranks"],
        fold["mf_valid"],
        fold["user_programmes"],
        fold["user_hours"],
        list(fold["truth"]),
        fold["pool"],
        fold["show_channels"],
        fold["channel_shares"],
    )
    features = add_ar_features(features, state)
    features = rerank.add_labels(features, fold["truth"])
    updated = dict(fold)
    updated["candidates"] = candidates
    updated["features"] = features
    return rerank2.add_slots_to_fold(
        updated, slot_data, target_slots
    )


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
            ascending=[True] + [False, False, True],
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

    ar_cache = {}

    def get_ar(history):
        key = tuple(history)
        if key not in ar_cache:
            ar_cache[key] = build_ar_rules(
                occurrences, key, users
            )
        return ar_cache[key]

    def build_ar_fold(base_fold, spec):
        _, schedule, target, history = spec
        rules, baskets, _ = get_ar(history)
        state = build_ar_state(
            users, rules, baskets, base_fold["pool"]
        )
        return augment_fold(
            base_fold,
            state,
            get_slot_data(history),
            get_target_slots(schedule, target),
        )

    fold_started = time.perf_counter()
    base_folds = {
        spec[0]: build_fold(spec) for spec in BASE_SPECS
    }
    eval_base = build_fold(EVAL_SPEC)
    ar_folds = {
        spec[0]: build_ar_fold(base_folds[spec[0]], spec)
        for spec in BASE_SPECS
    }
    eval_ar = build_ar_fold(eval_base, EVAL_SPEC)
    fold_elapsed = time.perf_counter() - fold_started

    base_rows = pd.concat(
        [
            rerank2.training_frame(base_folds["train1"]),
            rerank2.training_frame(base_folds["train2"]),
        ],
        ignore_index=True,
    )
    ar_rows = pd.concat(
        [
            rerank2.training_frame(ar_folds["train1"]),
            rerank2.training_frame(ar_folds["train2"]),
        ],
        ignore_index=True,
    )
    eval_base_rows = eval_base["features"]
    eval_ar_rows = eval_ar["features"]
    eval_truth = eval_base["truth"]
    eval_fill = eval_base["fill"]

    model_started = time.perf_counter()

    def fit_average(name, train_rows, eval_frame, features):
        classifier_scores, classifier, classifier_seconds = (
            rerank2.fit_classifier(
                train_rows, eval_frame, features
            )
        )
        ranker_scores, ranker, ranker_seconds = (
            rerank2.fit_ranker(train_rows, eval_frame, features)
        )
        scores = 0.5 * (
            rerank2.minmax_by_user(eval_frame, classifier_scores)
            + rerank2.minmax_by_user(eval_frame, ranker_scores)
        )
        result = rerank2.evaluate_scores(
            name,
            eval_frame,
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
        return result

    base_result = fit_average(
        "base C", base_rows, eval_base_rows, rerank2.ALL_FEATURES
    )
    ar_result = fit_average(
        "base + AR",
        ar_rows,
        eval_ar_rows,
        rerank2.ALL_FEATURES + AR_FEATURES,
    )
    best_lgbm = max(
        [base_result, ar_result], key=lambda result: result["map5"]
    )
    if best_lgbm is base_result:
        best_frame = eval_base_rows
        best_fill = eval_base["fill"]
        best_config = "base"
    else:
        best_frame = eval_ar_rows
        best_fill = eval_ar["fill"]
        best_config = "AR"
    best_scores = best_lgbm["scores"]

    logistic_scores, _, _, logistic_seconds = (
        rerank.fit_and_predict(
            base_rows,
            best_frame,
            "logistic",
            c_value=1.0,
        )
    )
    logistic_result = rerank2.evaluate_scores(
        "logistic C=1.0",
        best_frame,
        logistic_scores,
        users,
        eval_truth,
        best_fill,
    )
    repeat_ranks = rank_frame(
        best_frame,
        columns=[
            "repeat_count_ge_030",
            "repeat_count_ge_050",
            "days_since_last_qualifying",
        ],
    )
    repeat_recommendations = {
        int(user): [
            int(item) for item in group["tv_show_id"].head(5)
        ]
        for user, group in repeat_ranks.groupby(
            "user_id", sort=False
        )
    }
    repeat_result = rerank.evaluate_recommendations(
        repeat_recommendations,
        users,
        eval_truth,
        best_fill,
    )
    logistic_ranks = rank_frame(best_frame, logistic_scores)
    lgbm_ranks = rank_frame(best_frame, best_scores)
    ranked = best_frame[["user_id", "tv_show_id"]].copy()
    for name, ranks in (
        ("repeat", repeat_ranks),
        ("logistic", logistic_ranks),
        ("lgbm", lgbm_ranks),
    ):
        ranked = ranked.merge(
            ranks.rename(columns={"rank": f"{name}_rank"}),
            on=["user_id", "tv_show_id"],
            how="left",
        )

    fusion_results = []
    for weights in WEIGHTS:
        fused = (
            weights[0] * ranked["repeat_rank"].to_numpy()
            + weights[1] * ranked["logistic_rank"].to_numpy()
            + weights[2] * ranked["lgbm_rank"].to_numpy()
        )
        recommendations = (
            rerank.recommendations_from_probabilities(
                best_frame, -fused
            )
        )
        result = rerank.evaluate_recommendations(
            recommendations,
            users,
            eval_truth,
            best_fill,
        )
        result["name"] = (
            "fusion "
            f"({weights[0]:.1f},{weights[1]:.1f},"
            f"{weights[2]:.1f})"
        )
        result["weights"] = weights
        result["scores"] = fused
        fusion_results.append(result)

    model_elapsed = time.perf_counter() - model_started

    print(
        "association rules: "
        f"{len(ar_cache[tuple(BASE_SPECS[0][3])][0]):,} "
        "antecedents in train1 cache"
    )
    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs base"
    )
    all_results = [base_result, ar_result] + fusion_results
    for result in all_results:
        print(
            f"{result['name']} | {result['map5']:.5f} | "
            f"{result['map5'] - base_result['map5']:+.5f}"
        )
    print(
        "components: repeat "
        f"{repeat_result['map5']:.5f}; logistic "
        f"{logistic_result['map5']:.5f}; lightgbm "
        f"{best_lgbm['map5']:.5f} ({best_config})"
    )
    print(
        f"training rows: base {len(base_rows):,}; "
        f"AR {len(ar_rows):,}; best frame "
        f"{len(best_frame):,}"
    )
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )

    best = max(all_results, key=lambda result: result["map5"])
    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        best["aps"] - base_result["aps"],
        n_resamples=2000,
        seed=0,
    )
    print(
        f"best {best['name']} vs base: mean diff "
        f"{mean_diff:+.5f}, 95% CI "
        f"[{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "best beats base with CI above zero; "
            "building final submission"
        )
        final_started = time.perf_counter()
        fold4_base = build_fold(FOLD4_SPEC)
        fold4_ar = build_ar_fold(fold4_base, FOLD4_SPEC)
        final_base = build_fold(FINAL_SPEC)
        final_ar = build_ar_fold(final_base, FINAL_SPEC)
        if best_config == "AR":
            final_frame = final_ar["features"]
            final_fill = final_ar["fill"]
            final_pool = final_ar["pool"]
        else:
            final_frame = final_base["features"]
            final_fill = final_base["fill"]
            final_pool = final_base["pool"]

        if best["name"] == "base + AR":
            four_folds = [
                ar_folds["train1"],
                ar_folds["train2"],
                eval_ar,
                fold4_ar,
            ]
            final_train = pd.concat(
                [rerank2.training_frame(fold) for fold in four_folds],
                ignore_index=True,
            )
            final_scores = None
            classifier_scores, _, classifier_seconds = (
                rerank2.fit_classifier(
                    final_train,
                    final_frame,
                    rerank2.ALL_FEATURES + AR_FEATURES,
                )
            )
            ranker_scores, _, ranker_seconds = (
                rerank2.fit_ranker(
                    final_train,
                    final_frame,
                    rerank2.ALL_FEATURES + AR_FEATURES,
                )
            )
            final_scores = 0.5 * (
                rerank2.minmax_by_user(
                    final_frame, classifier_scores
                )
                + rerank2.minmax_by_user(
                    final_frame, ranker_scores
                )
            )
            recommendations = (
                rerank.recommendations_from_probabilities(
                    final_frame, final_scores
                )
            )
            print(
                f"final AR classifier fit {classifier_seconds:.2f}s; "
                f"ranker fit {ranker_seconds:.2f}s"
            )
        else:
            base_four = [
                base_folds["train1"],
                base_folds["train2"],
                eval_base,
                fold4_base,
            ]
            final_base_train = pd.concat(
                [
                    rerank2.training_frame(fold)
                    for fold in base_four
                ],
                ignore_index=True,
            )
            if best_config == "AR":
                lgbm_four = [
                    ar_folds["train1"],
                    ar_folds["train2"],
                    eval_ar,
                    fold4_ar,
                ]
            else:
                lgbm_four = base_four
            lgbm_train = pd.concat(
                [rerank2.training_frame(fold) for fold in lgbm_four],
                ignore_index=True,
            )
            lgbm_features = (
                rerank2.ALL_FEATURES + AR_FEATURES
                if best_config == "AR"
                else rerank2.ALL_FEATURES
            )
            classifier_scores, _, classifier_seconds = (
                rerank2.fit_classifier(
                    lgbm_train, final_frame, lgbm_features
                )
            )
            ranker_scores, _, ranker_seconds = (
                rerank2.fit_ranker(
                    lgbm_train, final_frame, lgbm_features
                )
            )
            lightgbm_scores = 0.5 * (
                rerank2.minmax_by_user(
                    final_frame, classifier_scores
                )
                + rerank2.minmax_by_user(
                    final_frame, ranker_scores
                )
            )
            final_logistic_scores, _, _, logistic_seconds = (
                rerank.fit_and_predict(
                    final_base_train,
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
            final_lgbm_ranks = rank_frame(
                final_frame, lightgbm_scores
            )
            final_ranked = final_frame[
                ["user_id", "tv_show_id"]
            ].copy()
            for name, ranks in (
                ("repeat", final_repeat_ranks),
                ("logistic", final_logistic_ranks),
                ("lgbm", final_lgbm_ranks),
            ):
                final_ranked = final_ranked.merge(
                    ranks.rename(columns={"rank": f"{name}_rank"}),
                    on=["user_id", "tv_show_id"],
                    how="left",
                )
            weights = best["weights"]
            fused = (
                weights[0] * final_ranked["repeat_rank"].to_numpy()
                + weights[1]
                * final_ranked["logistic_rank"].to_numpy()
                + weights[2] * final_ranked["lgbm_rank"].to_numpy()
            )
            recommendations = (
                rerank.recommendations_from_probabilities(
                    final_frame, -fused
                )
            )
            print(
                f"final logistic fit {logistic_seconds:.2f}s; "
                f"LGBM classifier fit {classifier_seconds:.2f}s; "
                f"ranker fit {ranker_seconds:.2f}s"
            )
        checks = write_submission(
            recommendations,
            users,
            reference,
            final_pool,
            final_fill,
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
        print("no rerank6 submission: CI condition not met")
    print(
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
