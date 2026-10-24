import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import (
    DATA,
    POPULARITY_WEEKS,
    SEED,
    TOP_K,
    TRAIN_WEEKS,
    VALID_WEEKS,
    ap_at_5,
    build_distinct_days_features,
    build_truth,
    fill_with_popularity,
    load_candidate_pool,
    load_interactions,
    popularity_ranking,
    rankings_from_features,
)
from validate_methods import (
    build_history_share,
    build_mf_matrix,
    build_mf_scores,
    build_popularity_share,
    minmax_rows,
    paired_bootstrap,
    topk_recs,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "validation_strict.csv"
STRICT_FRACTION = 0.8
T_GRID = (0.3, 0.5, 0.8)
FIT_WINDOWS = ((11, 20), (16, 20))
FILLS = ("a", "b")
HOLDOUT_FIT = (11, 15)
HOLDOUT_TRUTH = (16, 20)
HOLDOUT_SCHEDULE = "export_arh_11-20-final.csv"
BLEND_A = 0.30
BLEND_B = 0.02


def build_strict_truth(occurrences, weeks, users, pool):
    subset = occurrences[
        occurrences["iso_week"].between(*weeks)
        & occurrences["tv_show_id"].isin(pool)
        & (occurrences["fraction"] > STRICT_FRACTION)
    ]
    grouped = subset.groupby("user_id")["tv_show_id"].agg(set)
    all_truth = {
        int(user): {int(item) for item in items}
        for user, items in grouped.items()
    }
    return {int(user): all_truth.get(int(user), set()) for user in users}


def build_repeat_rankings(occurrences, weeks, threshold, pool, users):
    pool_set = set(pool)
    subset = occurrences[
        occurrences["iso_week"].between(*weeks)
        & occurrences["tv_show_id"].isin(pool_set)
        & (occurrences["fraction"] >= threshold)
        & occurrences["user_id"].isin(users)
    ]
    if subset.empty:
        return {}
    features = (
        subset.groupby(["user_id", "tv_show_id"], sort=False)
        .agg(
            n_airings=("airing_start", "size"),
            latest=("airing_start", "max"),
        )
        .reset_index()
    )
    features.sort_values(
        ["user_id", "n_airings", "latest", "tv_show_id"],
        ascending=[True, False, False, True],
        inplace=True,
    )
    features = features.groupby("user_id", sort=False).head(TOP_K)
    return {
        int(user): [int(item) for item in group["tv_show_id"]]
        for user, group in features.groupby("user_id", sort=False)
    }


def build_strict_popularity(occurrences, weeks, pool):
    pool_set = set(pool)
    subset = occurrences[
        occurrences["iso_week"].between(*weeks)
        & occurrences["tv_show_id"].isin(pool_set)
        & (occurrences["fraction"] > STRICT_FRACTION)
    ]
    counts = subset.groupby("tv_show_id")["user_id"].nunique()
    ranking = pd.DataFrame({"tv_show_id": sorted(pool)})
    ranking = ranking.merge(
        counts.rename("n_users"),
        left_on="tv_show_id",
        right_index=True,
        how="left",
    )
    ranking["n_users"] = ranking["n_users"].fillna(0).astype("int64")
    ranking.sort_values(
        ["n_users", "tv_show_id"],
        ascending=[False, True],
        inplace=True,
    )
    return [int(item) for item in ranking["tv_show_id"].head(TOP_K)]


def load_week_pool(file_name, weeks):
    schedule = pd.read_csv(
        DATA / file_name,
        usecols=["start_time", "tv_show_id"],
        parse_dates=["start_time"],
        dayfirst=True,
    )
    week = schedule["start_time"].dt.isocalendar().week
    labelled = schedule[
        (schedule["tv_show_id"] != 0) & week.between(*weeks)
    ]
    return sorted({int(item) for item in labelled["tv_show_id"].unique()})


def evaluate_pair(
    method,
    recommendations,
    old_truth,
    strict_truth,
    users,
    popularity,
):
    old_aps = []
    strict_aps = []
    filled = 0
    for user, recommendation in zip(users, recommendations):
        if len(recommendation) < TOP_K:
            filled += 1
        recommendation = fill_with_popularity(recommendation, popularity)
        old_aps.append(ap_at_5(recommendation, old_truth.get(user, set())))
        strict_aps.append(
            ap_at_5(recommendation, strict_truth.get(user, set()))
        )
    return {
        "method": method,
        "old_map": float(np.mean(old_aps)) if old_aps else 0.0,
        "strict_map": float(np.mean(strict_aps)) if strict_aps else 0.0,
        "old_aps": np.asarray(old_aps, dtype=np.float64),
        "strict_aps": np.asarray(strict_aps, dtype=np.float64),
        "fill_share": filled / len(users) if users else 0.0,
    }


def print_table(results, title):
    print(title)
    print(f"{'method':<38} {'old MAP@5':>10} {'strict MAP@5':>13} {'fill':>7}")
    print("-" * 72)
    for result in results:
        print(
            f"{result['method']:<38} {result['old_map']:>10.5f} "
            f"{result['strict_map']:>13.5f} "
            f"{result['fill_share']:>7.2%}"
        )


def main():
    total_started = time.perf_counter()

    interactions = load_interactions()
    occurrences = pd.read_parquet(DATA / "occurrences.parquet")
    submission = pd.read_csv(DATA / "submission.csv", usecols=["user_id"])
    users = [int(user) for user in submission["user_id"]]
    pool = sorted(load_candidate_pool("export_arh_21-30-final.csv"))
    pool_index = {item: i for i, item in enumerate(pool)}

    old_truth_all = build_truth(
        interactions, VALID_WEEKS, users, pool
    )
    eval_users = [user for user in users if user in old_truth_all]
    old_truth = {user: old_truth_all[user] for user in eval_users}
    strict_truth = build_strict_truth(
        occurrences, VALID_WEEKS, eval_users, pool
    )
    strict_users = [user for user in eval_users if strict_truth[user]]
    strict_pairs = sum(len(items) for items in strict_truth.values())
    eval_user_index = {
        user: i for i, user in enumerate(eval_users)
    }

    print("strict validation setup:")
    print(
        f"  train: ISO weeks {TRAIN_WEEKS[0]}-{TRAIN_WEEKS[1]}; "
        f"truth: ISO weeks {VALID_WEEKS[0]}-{VALID_WEEKS[1]}"
    )
    print(
        f"  candidate pool: {len(pool):,} labelled programmes "
        "(truth restricted to pool)"
    )
    print(
        f"  evaluation users: {len(eval_users):,} / {len(users):,}; "
        f"users with strict truth: {len(strict_users):,}"
    )
    if strict_users:
        print(
            f"  strict truth pairs: {strict_pairs:,}; "
            f"mean per strict user: "
            f"{strict_pairs / len(strict_users):.1f}"
        )
    else:
        print(f"  strict truth pairs: {strict_pairs:,}")
    print(f"  seed: {SEED}; fill a = existing popularity weeks 16-20")
    print()

    method_started = time.perf_counter()
    popularity, _ = popularity_ranking(
        interactions, POPULARITY_WEEKS, pool
    )
    popularity_share = build_popularity_share(
        interactions, pool, POPULARITY_WEEKS
    )
    history_features = build_distinct_days_features(
        interactions, TRAIN_WEEKS, pool
    )
    history_rankings = rankings_from_features(history_features)
    history_share = build_history_share(
        history_features,
        eval_users,
        pool,
        eval_user_index,
        pool_index,
    )
    mf_matrix, _, mf_user_index, mf_item_index = build_mf_matrix(
        interactions, TRAIN_WEEKS
    )
    mf_scores, mf_valid, _, _ = build_mf_scores(
        mf_matrix,
        mf_user_index,
        mf_item_index,
        eval_users,
        pool,
        16,
    )
    mf_normalized = minmax_rows(mf_scores)
    blend_scores = (
        BLEND_A * history_share
        + BLEND_B * mf_normalized
        + (1.0 - BLEND_A - BLEND_B) * popularity_share[None, :]
    )
    method_elapsed = time.perf_counter() - method_started

    old_results = []
    old_results.append(
        evaluate_pair(
            "popularity",
            [popularity for _ in eval_users],
            old_truth,
            strict_truth,
            eval_users,
            popularity,
        )
    )
    old_results.append(
        evaluate_pair(
            "history_days",
            [history_rankings.get(user, []) for user in eval_users],
            old_truth,
            strict_truth,
            eval_users,
            popularity,
        )
    )
    old_results.append(
        evaluate_pair(
            "mf_k16",
            topk_recs(mf_scores, pool, mf_valid),
            old_truth,
            strict_truth,
            eval_users,
            popularity,
        )
    )
    old_results.append(
        evaluate_pair(
            "mf_blend_a0.30_b0.02",
            topk_recs(blend_scores, pool),
            old_truth,
            strict_truth,
            eval_users,
            popularity,
        )
    )

    repeat_started = time.perf_counter()
    repeat_results = []
    fill_old = popularity
    fill_strict = build_strict_popularity(
        occurrences, (16, 20), pool
    )
    for fit_weeks in FIT_WINDOWS:
        for threshold in T_GRID:
            rankings = build_repeat_rankings(
                occurrences,
                fit_weeks,
                threshold,
                pool,
                eval_users,
            )
            recommendations = [
                rankings.get(user, []) for user in eval_users
            ]
            for fill_label, fill in (
                ("a", fill_old),
                ("b", fill_strict),
            ):
                method = (
                    f"repeat_t{threshold:.2f}_fit"
                    f"{fit_weeks[0]}-{fit_weeks[1]}_fill{fill_label}"
                )
                repeat_results.append(
                    evaluate_pair(
                        method,
                        recommendations,
                        old_truth,
                        strict_truth,
                        eval_users,
                        fill,
                    )
                )
    repeat_elapsed = time.perf_counter() - repeat_started

    all_results = old_results + repeat_results
    print_table(
        all_results,
        "validation: old truth vs strict truth "
        f"({len(eval_users):,} users):",
    )
    print()

    best_repeat = max(
        repeat_results, key=lambda result: result["strict_map"]
    )
    final_blend = next(
        result
        for result in old_results
        if result["method"] == "mf_blend_a0.30_b0.02"
    )
    bootstrap_started = time.perf_counter()
    mean_diff, ci_lower, ci_upper = paired_bootstrap(
        best_repeat["strict_aps"] - final_blend["strict_aps"]
    )
    bootstrap_elapsed = time.perf_counter() - bootstrap_started
    print("strict-truth paired bootstrap (2,000 resamples, seed 0):")
    print(
        f"  {best_repeat['method']} vs mf_blend_a0.30_b0.02: "
        f"mean diff {mean_diff:+.5f}, "
        f"95% CI [{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )
    print()

    top_three = sorted(
        repeat_results,
        key=lambda result: result["strict_map"],
        reverse=True,
    )[:3]
    holdout_pool = load_week_pool(HOLDOUT_SCHEDULE, HOLDOUT_TRUTH)
    holdout_old_all = build_truth(
        interactions, HOLDOUT_TRUTH, users, holdout_pool
    )
    holdout_users = [
        user for user in users if user in holdout_old_all
    ]
    holdout_old = {
        user: holdout_old_all[user] for user in holdout_users
    }
    holdout_strict = build_strict_truth(
        occurrences, HOLDOUT_TRUTH, holdout_users, holdout_pool
    )
    holdout_fill_old = popularity_ranking(
        interactions, HOLDOUT_FIT, holdout_pool
    )[0]
    holdout_fill_strict = build_strict_popularity(
        occurrences, HOLDOUT_FIT, holdout_pool
    )
    holdout_started = time.perf_counter()
    holdout_results = []
    holdout_cache = {}
    for result in top_three:
        method = result["method"]
        parts = method.split("_")
        threshold = float(parts[1][1:])
        fill_label = parts[-1][-1]
        if threshold not in holdout_cache:
            holdout_cache[threshold] = build_repeat_rankings(
                occurrences,
                HOLDOUT_FIT,
                threshold,
                holdout_pool,
                holdout_users,
            )
        rankings = holdout_cache[threshold]
        fill = (
            holdout_fill_old
            if fill_label == "a"
            else holdout_fill_strict
        )
        holdout_results.append(
            evaluate_pair(
                f"holdout_{method}",
                [rankings.get(user, []) for user in holdout_users],
                holdout_old,
                holdout_strict,
                holdout_users,
                fill,
            )
        )
    holdout_elapsed = time.perf_counter() - holdout_started

    print(
        f"earlier hold-out: fit weeks {HOLDOUT_FIT[0]}-{HOLDOUT_FIT[1]}, "
        f"truth weeks {HOLDOUT_TRUTH[0]}-{HOLDOUT_TRUTH[1]}, "
        f"pool {len(holdout_pool):,}, users {len(holdout_users):,}:"
    )
    print_table(
        holdout_results,
        "holdout results for the 3 best strict-validation configurations:",
    )
    print()
    main_order = [result["method"] for result in top_three]
    holdout_order = [
        result["method"].replace("holdout_", "", 1)
        for result in sorted(
            holdout_results,
            key=lambda result: result["strict_map"],
            reverse=True,
        )
    ]
    print(
        "holdout strict-order stable: "
        f"{'yes' if holdout_order == main_order else 'no'} "
        f"(main {main_order}; holdout {holdout_order})"
    )
    print()

    RESULTS_DIR.mkdir(exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "method": result["method"],
                "map5_strict": result["strict_map"],
                "map5_old_truth": result["old_map"],
            }
            for result in all_results
        ]
    )
    frame.to_csv(
        OUT_PATH,
        index=False,
        lineterminator="\n",
        float_format="%.5f",
    )
    print(f"saved {OUT_PATH} ({len(frame):,} methods)")
    print(
        f"runtimes: existing methods {method_elapsed:.2f}s; "
        f"repeat grid {repeat_elapsed:.2f}s; "
        f"bootstrap {bootstrap_elapsed:.2f}s; "
        f"holdout {holdout_elapsed:.2f}s; "
        f"total {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
