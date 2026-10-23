import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data")
TRAIN_WEEKS = (11, 20)
VALID_WEEKS = (21, 30)
POPULARITY_WEEKS = (16, 20)
RECENT_WEEKS = (17, 20)
BLEND_WEIGHTS = (0.2, 0.4, 0.6, 0.8)
TOP_K = 5
SEED = 0


def load_interactions():
    parquet_path = DATA / "interactions.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(DATA / "interactions.csv")


def load_candidate_pool(file_name):
    sched = pd.read_csv(DATA / file_name, usecols=["tv_show_id"])
    return {int(x) for x in sched.loc[sched["tv_show_id"] != 0, "tv_show_id"].unique()}


def build_truth(interactions, weeks, users, pool):
    val = interactions[interactions["iso_week"].between(*weeks)]
    val = val[val["tv_show_id"].isin(pool)]
    grouped = val.groupby("user_id")["tv_show_id"].agg(set)
    all_truth = {
        int(user): {int(item) for item in items} for user, items in grouped.items()
    }
    return {user: all_truth[user] for user in users if user in all_truth}


def build_user_programme_sets(interactions, weeks):
    subset = interactions[interactions["iso_week"].between(*weeks)]
    return {
        int(user): {int(item) for item in group["tv_show_id"]}
        for user, group in subset.groupby("user_id")
    }


def popularity_ranking(interactions, weeks, pool, k=TOP_K):
    recent = interactions[interactions["iso_week"].between(*weeks)]
    counts = recent.groupby("tv_show_id")["user_id"].nunique().rename("n_users")
    ranking = pd.DataFrame({"tv_show_id": sorted(pool)})
    ranking = ranking.merge(
        counts, left_on="tv_show_id", right_index=True, how="left"
    )
    ranking["n_users"] = ranking["n_users"].fillna(0).astype("int64")
    ranking.sort_values(
        ["n_users", "tv_show_id"], ascending=[False, True], inplace=True
    )
    ranking.reset_index(drop=True, inplace=True)
    return [int(x) for x in ranking["tv_show_id"].head(k)], ranking


def build_total_seconds_rankings(interactions, weeks, pool):
    train = interactions[interactions["iso_week"].between(*weeks)]
    train = train[train["tv_show_id"].isin(pool)]
    features = (
        train.groupby(["user_id", "tv_show_id"], sort=False)["overlap_seconds"]
        .sum()
        .reset_index()
    )
    features.sort_values(
        ["user_id", "overlap_seconds", "tv_show_id"],
        ascending=[True, False, True],
        inplace=True,
    )
    return rankings_from_features(features)


def build_distinct_days_features(interactions, weeks, pool):
    # The interaction table stores airing_start, so days are airing days.
    train = interactions[interactions["iso_week"].between(*weeks)]
    train = train[train["tv_show_id"].isin(pool)].copy()
    train["airing_day"] = train["airing_start"].dt.normalize()
    features = (
        train.groupby(["user_id", "tv_show_id"], sort=False)
        .agg(
            distinct_days=("airing_day", "nunique"),
            total_seconds=("overlap_seconds", "sum"),
        )
        .reset_index()
    )
    features.sort_values(
        ["user_id", "distinct_days", "total_seconds", "tv_show_id"],
        ascending=[True, False, False, True],
        inplace=True,
    )
    return features


def rankings_from_features(features):
    return {
        int(user): [int(item) for item in group["tv_show_id"]]
        for user, group in features.groupby("user_id", sort=False)
    }


def build_blend_rankings(
    user_features, interactions, popularity_weeks, pool, users, weights
):
    features = user_features[user_features["user_id"].isin(users)]
    pool_sorted = np.array(sorted(pool), dtype=np.int64)
    pool_index = {int(item): i for i, item in enumerate(pool_sorted)}
    user_index = {int(user): i for i, user in enumerate(users)}

    user_share = np.zeros((len(users), len(pool_sorted)), dtype=np.float32)
    for row in features.itertuples(index=False):
        user_share[
            user_index[int(row.user_id)], pool_index[int(row.tv_show_id)]
        ] = row.distinct_days
    row_sums = user_share.sum(axis=1, keepdims=True)
    np.divide(user_share, row_sums, out=user_share, where=row_sums > 0)

    recent = interactions[interactions["iso_week"].between(*popularity_weeks)]
    counts = recent.groupby("tv_show_id")["user_id"].nunique()
    pop_counts = counts.reindex(pool_sorted).fillna(0).to_numpy(dtype=np.float64)
    pop_share = pop_counts / pop_counts.sum()

    rankings = {}
    for weight in weights:
        scores = weight * user_share + (1.0 - weight) * pop_share[None, :]
        order = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K]
        rankings[weight] = {
            int(user): [int(pool_sorted[j]) for j in order[i]]
            for i, user in enumerate(users)
        }
    return rankings


def fill_with_popularity(items, popularity, k=TOP_K):
    result = list(dict.fromkeys(int(x) for x in items))
    if len(result) < k:
        result.extend(item for item in popularity if item not in result)
    result = result[:k]
    if len(result) < k:
        raise ValueError(f"cannot fill recommendation list to {k} items")
    return result


def ap_at_5(recommended, relevant):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended[:TOP_K], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), TOP_K)


def evaluate(name, users, truth, recommend_fn, popularity):
    started = time.perf_counter()
    aps = []
    filled = 0
    for user in users:
        recommendation = recommend_fn(user)
        if len(recommendation) < TOP_K:
            filled += 1
        recommendation = fill_with_popularity(recommendation, popularity)
        aps.append(ap_at_5(recommendation, truth[user]))
    elapsed = time.perf_counter() - started
    return {
        "method": name,
        "map5": float(np.mean(aps)) if aps else 0.0,
        "users": len(users),
        "fill_share": filled / len(users) if users else 0.0,
        "runtime": elapsed,
    }


def print_ap_tests():
    letter_ids = {
        "A": 101,
        "B": 102,
        "C": 103,
        "D": 104,
        "E": 105,
        "F": 106,
        "G": 107,
        "H": 108,
        "I": 109,
    }
    tests = [
        ("X", ["A", "B", "D", "C", "E"], {"A", "B", "C"}, (1 + 1 + 0.75) / 3),
        ("Y", ["D", "A", "E", "B", "C"], {"A", "B", "C"}, (0.5 + 0.5 + 0.6) / 3),
        ("Z", ["F", "G", "A", "H", "I"], {"A", "B", "C", "D", "E", "F"}, (1 + 2 / 3) / 5),
    ]
    print("AP@5 unit tests:")
    for label, recommendation, relevant, expected in tests:
        rec_ids = [letter_ids[letter] for letter in recommendation]
        truth_ids = {letter_ids[letter] for letter in relevant}
        got = ap_at_5(rec_ids, truth_ids)
        status = "OK" if abs(got - expected) < 1e-12 else "FAIL"
        print(
            f"  {label} {recommendation}: got {got:.6f} "
            f"expected {expected:.6f} {status}"
        )
        if status != "OK":
            raise AssertionError(f"AP@5 test {label} failed")


def print_diagnostics(truth, train_programmes, eval_users):
    sizes = np.array([len(truth[user]) for user in eval_users], dtype=np.int64)
    repeats = np.array(
        [
            len(truth[user] & train_programmes.get(user, set()))
            for user in eval_users
        ],
        dtype=np.int64,
    )
    total_truth = int(sizes.sum())
    total_repeats = int(repeats.sum())
    per_user_repeat = repeats / sizes
    print(
        f"truth per user: mean {sizes.mean():.1f}, "
        f"median {np.median(sizes):.0f}"
    )
    print(
        f"truth repeats: {total_repeats:,} / {total_truth:,} "
        f"({total_repeats / total_truth:.2%}) vs new "
        f"{1 - total_repeats / total_truth:.2%}; per-user mean "
        f"{per_user_repeat.mean():.2%}, median "
        f"{np.median(per_user_repeat):.2%}"
    )


def main():
    np.random.seed(SEED)
    started = time.perf_counter()

    interactions = load_interactions()
    submission = pd.read_csv(DATA / "submission.csv", usecols=["user_id"])
    users = [int(user) for user in submission["user_id"]]
    pool = load_candidate_pool("export_arh_21-30-final.csv")
    truth = build_truth(interactions, VALID_WEEKS, users, pool)
    eval_users = [user for user in users if user in truth]
    train_programmes = build_user_programme_sets(interactions, TRAIN_WEEKS)

    print(
        f"validation candidate pool: {len(pool):,} labelled programmes "
        "(truth restricted to pool)"
    )
    print(
        f"evaluation users: {len(eval_users):,} / {len(users):,} submission users "
        "with validation truth"
    )
    print(
        f"validation truth pairs (eval users): "
        f"{sum(len(items) for items in truth.values()):,}"
    )
    print(f"seed: {SEED} (methods are deterministic)")
    print()
    print_ap_tests()
    print()
    print_diagnostics(truth, train_programmes, eval_users)
    print()

    pop_started = time.perf_counter()
    popularity, _ = popularity_ranking(interactions, POPULARITY_WEEKS, pool)
    pop_build = time.perf_counter() - pop_started

    b1_started = time.perf_counter()
    b1_rankings = build_total_seconds_rankings(interactions, TRAIN_WEEKS, pool)
    b1_build = time.perf_counter() - b1_started

    b2_started = time.perf_counter()
    b2_features = build_distinct_days_features(interactions, TRAIN_WEEKS, pool)
    b2_rankings = rankings_from_features(b2_features)
    b2_build = time.perf_counter() - b2_started

    b3_started = time.perf_counter()
    b3_features = build_distinct_days_features(interactions, RECENT_WEEKS, pool)
    b3_rankings = rankings_from_features(b3_features)
    b3_build = time.perf_counter() - b3_started

    # Blend: b2 user distinct-days share (weeks 11-20) plus popularity share
    # in weeks 16-20, both normalized over the validation pool.
    blend_started = time.perf_counter()
    blend_rankings = build_blend_rankings(
        b2_features, interactions, POPULARITY_WEEKS, pool, eval_users, BLEND_WEIGHTS
    )
    blend_build = time.perf_counter() - blend_started

    results = [
        evaluate(
            "popularity",
            eval_users,
            truth,
            lambda user: popularity,
            popularity,
        ),
        evaluate(
            "history_seconds",
            eval_users,
            truth,
            lambda user: b1_rankings.get(user, []),
            popularity,
        ),
        evaluate(
            "history_days",
            eval_users,
            truth,
            lambda user: b2_rankings.get(user, []),
            popularity,
        ),
        evaluate(
            "history_days_17_20",
            eval_users,
            truth,
            lambda user: b3_rankings.get(user, []),
            popularity,
        ),
    ]
    for weight in BLEND_WEIGHTS:
        results.append(
            evaluate(
                f"blend_w{weight:.1f}",
                eval_users,
                truth,
                lambda user, rankings=blend_rankings[weight]: rankings.get(user, []),
                popularity,
            )
        )

    print(f"{'method':<20} {'MAP@5':>8} {'users':>6} {'fill':>8} {'runtime':>9}")
    for result in results:
        print(
            f"{result['method']:<20} {result['map5']:>8.5f} "
            f"{result['users']:>6,} {result['fill_share']:>7.2%} "
            f"{result['runtime']:>8.3f}s"
        )
    print(
        f"build: popularity {pop_build:.2f}s, history_seconds {b1_build:.2f}s, "
        f"history_days {b2_build:.2f}s, history_days_17_20 {b3_build:.2f}s, "
        f"blend {blend_build:.2f}s"
    )
    print(f"total runtime: {time.perf_counter() - started:.2f}s")
    best = max(
        (result for result in results if result["method"].startswith("blend_")),
        key=lambda result: result["map5"],
    )
    print(f"best blend: {best['method']} (MAP@5={best['map5']:.5f})")


if __name__ == "__main__":
    main()
