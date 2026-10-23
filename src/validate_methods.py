import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import (
    DATA,
    POPULARITY_WEEKS,
    RECENT_WEEKS,
    SEED,
    TOP_K,
    TRAIN_WEEKS,
    VALID_WEEKS,
    ap_at_5,
    build_distinct_days_features,
    build_total_seconds_rankings,
    build_truth,
    build_user_programme_sets,
    fill_with_popularity,
    load_candidate_pool,
    load_interactions,
    popularity_ranking,
    print_ap_tests,
    print_diagnostics,
    rankings_from_features,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIST_POP_WEIGHTS = (0.05, 0.1, 0.15, 0.2, 0.3)
BLEND_A_GRID = (0.2, 0.4, 0.6)
BLEND_B_GRID = (0.05, 0.1, 0.2)
MF_KS = (16, 32, 64)
ATTRIBUTE_FILES = ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv")
CORE_ATTRIBUTES = {"category", "genre_1", "genre_2", "genre_3", "year"}


def load_item_attributes():
    frames = []
    for file_name in ATTRIBUTE_FILES:
        frame = pd.read_csv(
            DATA / file_name,
            usecols=[
                "tv_show_id",
                "tv_show_category",
                "tv_show_genre_1",
                "tv_show_genre_2",
                "tv_show_genre_3",
                "year_of_production",
                "director",
                "actors",
            ],
        )
        frame = frame[frame["tv_show_id"] != 0].copy()
        frames.append(frame)
    labelled = pd.concat(frames, ignore_index=True)
    labelled["year_parsed"] = pd.to_numeric(
        labelled["year_of_production"]
        .astype("string")
        .str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
    return labelled


def attribute_coverage(labelled):
    total_items = labelled["tv_show_id"].nunique()
    attributes = [
        ("tv_show_category", "category"),
        ("tv_show_genre_1", "genre_1"),
        ("tv_show_genre_2", "genre_2"),
        ("tv_show_genre_3", "genre_3"),
        ("year_parsed", "year"),
        ("director", "director"),
        ("actors", "actors"),
    ]
    rows = []
    for column, label in attributes:
        row_coverage = labelled[column].notna().mean()
        item_coverage = (
            labelled.loc[labelled[column].notna(), "tv_show_id"].nunique()
            / total_items
        )
        rows.append((label, row_coverage, item_coverage))
    return pd.DataFrame(
        rows, columns=["attribute", "row_coverage", "item_coverage"]
    )


def most_frequent(labelled, column):
    values = labelled.loc[labelled[column].notna(), ["tv_show_id", column]]
    counts = (
        values.groupby(["tv_show_id", column], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    counts.sort_values(
        ["tv_show_id", "n", column],
        ascending=[True, False, True],
        inplace=True,
    )
    return (
        counts.drop_duplicates("tv_show_id")
        .set_index("tv_show_id")[column]
    )


def most_frequent_actors(labelled, top_n=3):
    actor_rows = labelled.loc[
        labelled["actors"].notna(), ["tv_show_id", "actors"]
    ].copy()
    actor_rows["actor"] = actor_rows["actors"].str.split(",")
    actor_rows = actor_rows.explode("actor")
    actor_rows["actor"] = actor_rows["actor"].str.strip()
    actor_rows = actor_rows[actor_rows["actor"] != ""]
    counts = (
        actor_rows.groupby(["tv_show_id", "actor"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    counts.sort_values(
        ["tv_show_id", "n", "actor"],
        ascending=[True, False, True],
        inplace=True,
    )
    counts = counts.groupby("tv_show_id", sort=False).head(top_n)
    return (
        counts.groupby("tv_show_id", sort=False)["actor"]
        .agg(list)
        .to_dict()
    )


def build_item_table(labelled, use_director):
    item_ids = np.sort(labelled["tv_show_id"].unique())
    table = pd.DataFrame(index=pd.Index(item_ids, name="tv_show_id"))
    table["category"] = most_frequent(
        labelled, "tv_show_category"
    ).reindex(table.index)
    table["genre_1"] = most_frequent(
        labelled, "tv_show_genre_1"
    ).reindex(table.index)
    table["genre_2"] = most_frequent(
        labelled, "tv_show_genre_2"
    ).reindex(table.index)
    table["genre_3"] = most_frequent(
        labelled, "tv_show_genre_3"
    ).reindex(table.index)
    table["year"] = most_frequent(
        labelled, "year_parsed"
    ).reindex(table.index)
    if use_director:
        table["director"] = most_frequent(
            labelled, "director"
        ).reindex(table.index)
    else:
        table["director"] = np.nan
    return table


def item_tokens(row, actor_tokens):
    tokens = set()
    for prefix, value in (
        ("cat", row.category),
        ("g1", row.genre_1),
        ("g2", row.genre_2),
        ("g3", row.genre_3),
        ("dir", row.director),
    ):
        if pd.notna(value):
            value = str(value).strip()
            if value:
                tokens.add(f"{prefix}:{value}")
    if pd.notna(row.year):
        tokens.add(f"decade:{int(row.year) // 10 * 10}")
    for actor in actor_tokens.get(int(row.Index), []):
        tokens.add(f"act:{actor}")
    return tokens


def build_tfidf(item_table, actor_tokens):
    token_to_index = {}
    rows = []
    cols = []
    data = []
    for row_index, row in enumerate(item_table.itertuples(index=True)):
        for token in sorted(item_tokens(row, actor_tokens)):
            column = token_to_index.setdefault(token, len(token_to_index))
            rows.append(row_index)
            cols.append(column)
            data.append(1.0)
    n_items = len(item_table)
    n_tokens = len(token_to_index)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    document_frequency = np.bincount(cols, minlength=n_tokens)
    idf = np.log((1.0 + n_items) / (1.0 + document_frequency)) + 1.0
    data = idf[cols]
    vectors = csr_matrix(
        (data, (rows, cols)),
        shape=(n_items, n_tokens),
        dtype=np.float64,
    )
    norms = np.sqrt(
        np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel()
    )
    inverse = np.zeros_like(norms)
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    vectors = (diags(inverse) @ vectors).tocsr()
    return vectors, token_to_index


def build_content_scores(
    interactions, eval_users, pool, item_table, vectors
):
    n_eval_users = len(eval_users)
    n_items = len(item_table)
    item_index = {
        int(item): i for i, item in enumerate(item_table.index)
    }
    user_index = {user: i for i, user in enumerate(eval_users)}
    pool_rows = np.asarray(
        [item_index[int(item)] for item in pool], dtype=np.int32
    )

    train = interactions[
        interactions["iso_week"].between(*TRAIN_WEEKS)
        & interactions["user_id"].isin(eval_users)
    ].copy()
    train["airing_day"] = train["airing_start"].dt.normalize()
    train_days = (
        train.groupby(["user_id", "tv_show_id"], sort=False)["airing_day"]
        .nunique()
        .rename("days")
        .reset_index()
    )
    train_days["item_index"] = train_days["tv_show_id"].map(item_index)
    missing_items = int(train_days["item_index"].isna().sum())
    train_days = train_days[train_days["item_index"].notna()].copy()
    train_days["item_index"] = train_days["item_index"].astype(np.int32)
    train_days["user_index"] = train_days["user_id"].map(user_index)
    train_days = train_days[train_days["user_index"].notna()].copy()
    train_days["user_index"] = train_days["user_index"].astype(np.int32)

    weights = csr_matrix(
        (
            train_days["days"].to_numpy(dtype=np.float64),
            (
                train_days["user_index"].to_numpy(),
                train_days["item_index"].to_numpy(),
            ),
        ),
        shape=(n_eval_users, n_items),
    )
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    profiles = (weights @ vectors).tocsr()
    inverse = np.zeros_like(row_sums)
    np.divide(1.0, row_sums, out=inverse, where=row_sums > 0)
    profiles = (diags(inverse) @ profiles).tocsr()
    profile_norms = np.sqrt(
        np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel()
    )
    inverse = np.zeros_like(profile_norms)
    np.divide(
        1.0, profile_norms, out=inverse, where=profile_norms > 0
    )
    profiles = (diags(inverse) @ profiles).tocsr()

    pool_vectors = vectors[pool_rows]
    scores = (profiles @ pool_vectors.T).toarray()
    valid = np.any(scores > 0, axis=1)
    return scores, valid, missing_items


def build_mf_matrix(interactions):
    train = interactions[
        interactions["iso_week"].between(*TRAIN_WEEKS)
    ].copy()
    train["airing_day"] = train["airing_start"].dt.normalize()
    train_days = (
        train.groupby(["user_id", "tv_show_id"], sort=False)["airing_day"]
        .nunique()
        .rename("days")
        .reset_index()
    )
    user_ids = np.sort(train_days["user_id"].unique())
    item_ids = np.sort(train_days["tv_show_id"].unique())
    user_index = {
        int(user): i for i, user in enumerate(user_ids)
    }
    item_index = {
        int(item): i for i, item in enumerate(item_ids)
    }
    rows = train_days["user_id"].map(user_index).to_numpy(dtype=np.int32)
    cols = train_days["tv_show_id"].map(item_index).to_numpy(
        dtype=np.int32
    )
    values = np.log1p(train_days["days"].to_numpy(dtype=np.float64))
    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float64,
    )
    return matrix, train_days, user_index, item_index


def build_mf_scores(
    matrix, user_index, item_index, eval_users, pool, k
):
    started = time.perf_counter()
    left, singular_values, right = svds(
        matrix, k=k, random_state=SEED
    )
    order = np.argsort(-singular_values)
    left = left[:, order]
    singular_values = singular_values[order]
    right = right[order, :]
    user_factors = left * singular_values
    item_factors = right.T

    eval_rows = np.asarray(
        [user_index.get(int(user), -1) for user in eval_users],
        dtype=np.int64,
    )
    valid_users = eval_rows >= 0
    eval_factors = np.zeros((len(eval_users), k), dtype=np.float64)
    eval_factors[valid_users] = user_factors[eval_rows[valid_users]]

    pool_rows = np.asarray(
        [item_index.get(int(item), -1) for item in pool],
        dtype=np.int64,
    )
    valid_pool = pool_rows >= 0
    pool_factors = np.zeros((len(pool), k), dtype=np.float64)
    pool_factors[valid_pool] = item_factors[pool_rows[valid_pool]]

    scores = eval_factors @ pool_factors.T
    valid = valid_users & np.any(scores != 0, axis=1)
    elapsed = time.perf_counter() - started
    return scores, valid, int(valid_pool.sum()), elapsed


def build_history_share(
    features, eval_users, pool, eval_user_index, pool_index
):
    features = features[features["user_id"].isin(eval_users)]
    share = np.zeros(
        (len(eval_users), len(pool)), dtype=np.float64
    )
    for row in features.itertuples(index=False):
        share[
            eval_user_index[int(row.user_id)],
            pool_index[int(row.tv_show_id)],
        ] = row.distinct_days
    row_sums = share.sum(axis=1, keepdims=True)
    np.divide(share, row_sums, out=share, where=row_sums > 0)
    return share


def build_popularity_share(interactions, pool):
    recent = interactions[
        interactions["iso_week"].between(*POPULARITY_WEEKS)
    ]
    counts = recent.groupby("tv_show_id")["user_id"].nunique()
    pop_counts = counts.reindex(pool).fillna(0).to_numpy(
        dtype=np.float64
    )
    return pop_counts / pop_counts.sum()


def minmax_rows(scores):
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    denominator = maximum - minimum
    normalized = np.zeros_like(scores, dtype=np.float64)
    np.divide(
        scores - minimum,
        denominator,
        out=normalized,
        where=denominator > 0,
    )
    return normalized


def topk_recs(scores, pool, valid=None):
    order = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K]
    recommendations = []
    for i in range(scores.shape[0]):
        if valid is not None and not valid[i]:
            recommendations.append([])
        else:
            recommendations.append(
                [int(pool[j]) for j in order[i]]
            )
    return recommendations


def evaluate_recs(
    name, params, recommendations, truth, eval_users, popularity,
    build_time
):
    started = time.perf_counter()
    aps = []
    filled = 0
    for user, recommendation in zip(eval_users, recommendations):
        if len(recommendation) < TOP_K:
            filled += 1
        recommendation = fill_with_popularity(
            recommendation, popularity
        )
        aps.append(ap_at_5(recommendation, truth[user]))
    elapsed = time.perf_counter() - started
    return {
        "method": name,
        "params": params,
        "map5": float(np.mean(aps)) if aps else 0.0,
        "fill_share": filled / len(eval_users) if eval_users else 0.0,
        "build": build_time,
        "eval": elapsed,
        "runtime": build_time + elapsed,
    }


def print_coverage(coverage, use_director, use_actors):
    print(
        "attribute coverage (labelled schedule rows, weeks 11-20 + 21-30):"
    )
    print(
        f"{'attribute':<12} {'rows':>8} {'items':>8} {'used':>5}"
    )
    for row in coverage.itertuples(index=False):
        used = (
            row.attribute in CORE_ATTRIBUTES
            or row.row_coverage > 0.30
        )
        print(
            f"{row.attribute:<12} {row.row_coverage:>7.2%} "
            f"{row.item_coverage:>7.2%} "
            f"{'yes' if used else 'no':>5}"
        )
    print(
        f"director included: {'yes' if use_director else 'no'} "
        "(threshold: row coverage > 30%)"
    )
    print(
        f"actors included: {'yes' if use_actors else 'no'} "
        "(threshold: row coverage > 30%; top 3 actors per item)"
    )


def print_results(results):
    header = (
        f"{'method':<26} {'params':<34} {'MAP@5':>8} "
        f"{'fill':>7} {'build':>7} {'eval':>7}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['method']:<26} {result['params']:<34} "
            f"{result['map5']:>8.5f} "
            f"{result['fill_share']:>7.2%} "
            f"{result['build']:>6.2f}s "
            f"{result['eval']:>6.2f}s"
        )


def best_of(results, prefix):
    candidates = [
        result
        for result in results
        if result["method"].startswith(prefix)
    ]
    return max(candidates, key=lambda result: result["map5"])


def main():
    total_started = time.perf_counter()

    interactions = load_interactions()
    submission = pd.read_csv(DATA / "submission.csv", usecols=["user_id"])
    users = [int(user) for user in submission["user_id"]]
    pool = sorted(load_candidate_pool("export_arh_21-30-final.csv"))
    truth = build_truth(interactions, VALID_WEEKS, users, pool)
    eval_users = [user for user in users if user in truth]
    train_programmes = build_user_programme_sets(
        interactions, TRAIN_WEEKS
    )
    eval_user_index = {
        user: i for i, user in enumerate(eval_users)
    }
    pool_index = {
        int(item): i for i, item in enumerate(pool)
    }

    print("validation setup:")
    print(
        f"  train: ISO weeks {TRAIN_WEEKS[0]}-{TRAIN_WEEKS[1]}; "
        f"truth: ISO weeks {VALID_WEEKS[0]}-{VALID_WEEKS[1]}"
    )
    print(
        f"  candidate pool: {len(pool):,} labelled programmes "
        "(truth restricted to pool)"
    )
    print(
        f"  evaluation users: {len(eval_users):,} / "
        f"{len(users):,} submission users with truth"
    )
    print(
        f"  truth pairs: "
        f"{sum(len(items) for items in truth.values()):,}"
    )
    print(
        "  repeat rule: truth sets; repeats allowed; "
        "tie-break by tv_show_id ascending"
    )
    print(
        "  fill / cold-start: popularity weeks 16-20 over pool"
    )
    print(
        f"  seed: {SEED} (numpy; svds random_state={SEED})"
    )
    print()
    print_ap_tests()
    print()
    print_diagnostics(truth, train_programmes, eval_users)
    print()

    content_started = time.perf_counter()
    labelled = load_item_attributes()
    coverage = attribute_coverage(labelled)
    use_director = bool(
        coverage.loc[
            coverage["attribute"] == "director", "row_coverage"
        ].iloc[0]
        > 0.30
    )
    use_actors = bool(
        coverage.loc[
            coverage["attribute"] == "actors", "row_coverage"
        ].iloc[0]
        > 0.30
    )
    actor_tokens = (
        most_frequent_actors(labelled) if use_actors else {}
    )
    item_table = build_item_table(labelled, use_director)
    item_vectors, token_to_index = build_tfidf(
        item_table, actor_tokens
    )
    print_coverage(coverage, use_director, use_actors)
    print(
        f"content vectors: {len(item_table):,} items x "
        f"{len(token_to_index):,} tokens; "
        "TF-IDF, L2-normalized; profile = distinct-days-weighted mean; "
        "cosine"
    )
    print(
        "attribute source: labelled rows from weeks 11-20 and 21-30; "
        "per-item mode (top 3 actors)"
    )
    print()

    results = []

    popularity_started = time.perf_counter()
    popularity, _ = popularity_ranking(
        interactions, POPULARITY_WEEKS, pool
    )
    popularity_build = time.perf_counter() - popularity_started
    results.append(
        evaluate_recs(
            "popularity",
            "weeks=16-20",
            [popularity for _ in eval_users],
            truth,
            eval_users,
            popularity,
            popularity_build,
        )
    )

    b1_started = time.perf_counter()
    b1_rankings = build_total_seconds_rankings(
        interactions, TRAIN_WEEKS, pool
    )
    b1_build = time.perf_counter() - b1_started
    results.append(
        evaluate_recs(
            "history_seconds",
            "weeks=11-20",
            [b1_rankings.get(user, []) for user in eval_users],
            truth,
            eval_users,
            popularity,
            b1_build,
        )
    )

    b2_started = time.perf_counter()
    b2_features = build_distinct_days_features(
        interactions, TRAIN_WEEKS, pool
    )
    b2_rankings = rankings_from_features(b2_features)
    b2_build = time.perf_counter() - b2_started
    results.append(
        evaluate_recs(
            "history_days",
            "weeks=11-20",
            [b2_rankings.get(user, []) for user in eval_users],
            truth,
            eval_users,
            popularity,
            b2_build,
        )
    )

    b3_started = time.perf_counter()
    b3_features = build_distinct_days_features(
        interactions, RECENT_WEEKS, pool
    )
    b3_rankings = rankings_from_features(b3_features)
    b3_build = time.perf_counter() - b3_started
    results.append(
        evaluate_recs(
            "history_days_17_20",
            "weeks=17-20",
            [b3_rankings.get(user, []) for user in eval_users],
            truth,
            eval_users,
            popularity,
            b3_build,
        )
    )

    share_started = time.perf_counter()
    history_share = build_history_share(
        b2_features, eval_users, pool, eval_user_index, pool_index
    )
    popularity_share = build_popularity_share(interactions, pool)
    share_build = time.perf_counter() - share_started
    print(
        f"shared blend inputs: history_days + popularity shares "
        f"built in {share_build:.2f}s"
    )

    for weight in HIST_POP_WEIGHTS:
        blend_started = time.perf_counter()
        scores = (
            weight * history_share
            + (1.0 - weight) * popularity_share[None, :]
        )
        recommendations = topk_recs(scores, pool)
        blend_build = time.perf_counter() - blend_started
        results.append(
            evaluate_recs(
                f"hist_pop_w{weight:.2f}",
                f"w={weight:.2f}; pop=16-20",
                recommendations,
                truth,
                eval_users,
                popularity,
                blend_build,
            )
        )

    content_scores, content_valid, content_missing = (
        build_content_scores(
            interactions,
            eval_users,
            pool,
            item_table,
            item_vectors,
        )
    )
    content_build = time.perf_counter() - content_started
    content_recommendations = topk_recs(
        content_scores, pool, content_valid
    )
    results.append(
        evaluate_recs(
            "content_tfidf",
            "tfidf; cat+g1-3+dec+dir+act",
            content_recommendations,
            truth,
            eval_users,
            popularity,
            content_build,
        )
    )
    print(
        f"content: {int((~content_valid).sum()):,} cold-start users; "
        f"{content_missing:,} training items missing attributes"
    )

    mf_matrix_started = time.perf_counter()
    mf_matrix, mf_train_days, mf_user_index, mf_item_index = (
        build_mf_matrix(interactions)
    )
    mf_matrix_build = time.perf_counter() - mf_matrix_started
    print(
        f"MF matrix: {mf_matrix.shape[0]:,} users x "
        f"{mf_matrix.shape[1]:,} items; "
        f"{len(mf_train_days):,} user-programme pairs; "
        f"value=log1p(distinct days); built in "
        f"{mf_matrix_build:.2f}s"
    )

    mf_scores = {}
    mf_results = {}
    mf_pool_coverage = None
    mf_users_in_training = None
    for k in MF_KS:
        scores, valid, pool_coverage, mf_elapsed = build_mf_scores(
            mf_matrix,
            mf_user_index,
            mf_item_index,
            eval_users,
            pool,
            k,
        )
        mf_scores[k] = scores
        mf_pool_coverage = pool_coverage
        if mf_users_in_training is None:
            mf_users_in_training = int(
                sum(1 for user in eval_users if user in mf_user_index)
            )
        recommendations = topk_recs(scores, pool, valid)
        result = evaluate_recs(
            f"mf_k{k}",
            f"k={k}; log1p(days); seed={SEED}",
            recommendations,
            truth,
            eval_users,
            popularity,
            mf_matrix_build + mf_elapsed,
        )
        mf_results[k] = result
        results.append(result)

    best_mf_k = max(
        MF_KS, key=lambda k: mf_results[k]["map5"]
    )
    print(
        f"MF: {mf_users_in_training:,} / {len(eval_users):,} eval users "
        f"in training; {mf_pool_coverage:,} / {len(pool):,} pool items "
        "seen in training (unseen items score 0); best k="
        f"{best_mf_k}"
    )
    print()

    normalization_started = time.perf_counter()
    content_normalized = minmax_rows(content_scores)
    mf_normalized = minmax_rows(mf_scores[best_mf_k])
    normalization_build = time.perf_counter() - normalization_started
    print(
        f"min-max normalization for three-part blends: "
        f"{normalization_build:.2f}s"
    )

    blend_grid = [
        (a, b)
        for a in BLEND_A_GRID
        for b in BLEND_B_GRID
        if a + b <= 0.8
    ]
    for method_name, method_normalized in (
        ("content", content_normalized),
        (f"mf_k{best_mf_k}", mf_normalized),
    ):
        for a, b in blend_grid:
            blend_started = time.perf_counter()
            scores = (
                a * history_share
                + b * method_normalized
                + (1.0 - a - b) * popularity_share[None, :]
            )
            recommendations = topk_recs(scores, pool)
            blend_build = time.perf_counter() - blend_started
            results.append(
                evaluate_recs(
                    f"blend_{method_name}_a{a:.2f}_b{b:.2f}",
                    f"a={a:.2f} b={b:.2f}",
                    recommendations,
                    truth,
                    eval_users,
                    popularity,
                    blend_build,
                )
            )

    print()
    print_results(results)
    print()
    print(f"total runtime: {time.perf_counter() - total_started:.2f}s")

    best_hist_pop = best_of(results, "hist_pop_")
    best_content_blend = best_of(results, "blend_content_")
    best_mf_blend = best_of(results, f"blend_mf_k{best_mf_k}_")
    best_overall = max(results, key=lambda result: result["map5"])
    print(
        f"best hist+pop: {best_hist_pop['method']} "
        f"(MAP@5={best_hist_pop['map5']:.5f})"
    )
    print(
        f"best content blend: {best_content_blend['method']} "
        f"(MAP@5={best_content_blend['map5']:.5f})"
    )
    print(
        f"best MF blend: {best_mf_blend['method']} "
        f"(MAP@5={best_mf_blend['map5']:.5f})"
    )
    print(
        f"best overall: {best_overall['method']} "
        f"(MAP@5={best_overall['map5']:.5f})"
    )


if __name__ == "__main__":
    main()
