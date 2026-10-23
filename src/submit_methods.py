import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import (
    DATA,
    TOP_K,
    build_distinct_days_features,
    fill_with_popularity,
    load_candidate_pool,
    load_interactions,
    popularity_ranking,
)
from validate_methods import (
    attribute_coverage,
    build_content_scores,
    build_history_share,
    build_item_table,
    build_mf_matrix,
    build_mf_scores,
    build_popularity_share,
    build_tfidf,
    load_item_attributes,
    minmax_rows,
    most_frequent_actors,
    topk_recs,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("submissions")
HISTORY_WEEKS = (21, 30)
POPULARITY_WEEKS = (26, 30)
CANDIDATE_FILE = "export_arh_31-42-final.csv"
ATTRIBUTE_FILES = (
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
)
MF_K = 16
MF_BLEND_A = 0.30
MF_BLEND_B = 0.02


def validate_file(path, pool):
    original_bytes = (DATA / "submission.csv").read_bytes()
    produced_bytes = path.read_bytes()
    original_header = original_bytes.split(b"\n")[0].decode("utf-8")
    output_header = produced_bytes.split(b"\n")[0].decode("utf-8")
    original = pd.read_csv(
        DATA / "submission.csv", dtype={"tv_show_id": str}
    )
    produced = pd.read_csv(path, dtype={"tv_show_id": str})
    checks = {
        "header": output_header == original_header == "user_id,tv_show_id",
        "row_count": len(produced) == len(original),
        "user_order": np.array_equal(
            produced["user_id"].values, original["user_id"].values
        ),
        "five_integer_ids": bool(
            produced["tv_show_id"].str.fullmatch(r"\d+( \d+){4}").all()
        ),
        "five_unique_ids": bool(
            produced["tv_show_id"]
            .str.split()
            .map(lambda ids: len(set(ids)) == TOP_K)
            .all()
        ),
        "candidates_only": set(
            int(item)
            for ids in produced["tv_show_id"].str.split()
            for item in ids
        ).issubset(set(pool)),
        "line_endings": b"\r\n" not in produced_bytes
        and produced_bytes.count(b"\n") == original_bytes.count(b"\n"),
        "no_trailing_newline": not produced_bytes.endswith(b"\n")
        and not original_bytes.endswith(b"\n"),
    }
    return checks


def write_submission(
    file_name, users, recommendations, popularity, pool
):
    started = time.perf_counter()
    filled = 0
    rows = []
    for recommendation in recommendations:
        if len(recommendation) < TOP_K:
            filled += 1
        recommendation = fill_with_popularity(
            recommendation, popularity
        )
        rows.append(" ".join(map(str, recommendation)))
    output = pd.DataFrame(
        {"user_id": users, "tv_show_id": rows}
    )
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / file_name
    csv_text = output.to_csv(index=False, lineterminator="\n")
    path.write_bytes(csv_text.rstrip("\n").encode("utf-8"))
    checks = validate_file(path, pool)
    elapsed = time.perf_counter() - started
    fill_share = filled / len(users) if users else 0.0
    print(
        f"wrote {path} ({len(output):,} rows); "
        f"fill={fill_share:.2%}; time={elapsed:.2f}s"
    )
    print(
        "  validation: "
        + ", ".join(
            f"{key}={'OK' if value else 'FAIL'}"
            for key, value in checks.items()
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)
    print(
        "  example: "
        f"{path.open(encoding='utf-8').readlines()[1].strip()!r}"
    )
    return fill_share, elapsed


def main():
    total_started = time.perf_counter()

    interactions = load_interactions()
    submission = pd.read_csv(DATA / "submission.csv", usecols=["user_id"])
    users = [int(user) for user in submission["user_id"]]
    pool = sorted(load_candidate_pool(CANDIDATE_FILE))
    user_index = {user: i for i, user in enumerate(users)}
    pool_index = {int(item): i for i, item in enumerate(pool)}

    print(
        f"real split: history/MF weeks {HISTORY_WEEKS[0]}-"
        f"{HISTORY_WEEKS[1]}; popularity weeks "
        f"{POPULARITY_WEEKS[0]}-{POPULARITY_WEEKS[1]}; "
        f"candidates weeks 31-42"
    )
    print(
        f"candidate pool: {len(pool):,}; submission users: "
        f"{len(users):,}"
    )

    popularity_started = time.perf_counter()
    popularity, _ = popularity_ranking(
        interactions, POPULARITY_WEEKS, pool
    )
    popularity_share = build_popularity_share(
        interactions, pool, POPULARITY_WEEKS
    )
    popularity_build = time.perf_counter() - popularity_started

    content_started = time.perf_counter()
    labelled = load_item_attributes(ATTRIBUTE_FILES)
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
    content_scores, content_valid, content_missing = (
        build_content_scores(
            interactions,
            users,
            pool,
            item_table,
            item_vectors,
            HISTORY_WEEKS,
        )
    )
    content_recommendations = topk_recs(
        content_scores, pool, content_valid
    )
    content_fill, _ = write_submission(
        "content.csv",
        users,
        content_recommendations,
        popularity,
        pool,
    )
    content_elapsed = time.perf_counter() - content_started
    print(
        f"content: {len(item_table):,} items x "
        f"{len(token_to_index):,} tokens; "
        f"director={'yes' if use_director else 'no'} "
        f"actors={'yes' if use_actors else 'no'}; "
        f"cold-start={int((~content_valid).sum()):,}; "
        f"missing training items={content_missing:,}; "
        f"fill={content_fill:.2%}; total={content_elapsed:.2f}s"
    )
    print()

    mf_started = time.perf_counter()
    mf_matrix, mf_train_days, mf_user_index, mf_item_index = (
        build_mf_matrix(interactions, HISTORY_WEEKS)
    )
    mf_scores, mf_valid, mf_pool_coverage, mf_svd_time = (
        build_mf_scores(
            mf_matrix,
            mf_user_index,
            mf_item_index,
            users,
            pool,
            MF_K,
        )
    )
    mf_recommendations = topk_recs(mf_scores, pool, mf_valid)
    mf_fill, _ = write_submission(
        "mf_k16.csv",
        users,
        mf_recommendations,
        popularity,
        pool,
    )
    mf_elapsed = time.perf_counter() - mf_started
    mf_users_in_training = sum(
        1 for user in users if user in mf_user_index
    )
    mf_seen_items = set(mf_item_index)
    mf_recommended_items = [
        item for rec in mf_recommendations for item in rec
    ]
    mf_unseen_share = (
        sum(
            1
            for item in mf_recommended_items
            if item not in mf_seen_items
        )
        / (len(users) * TOP_K)
        if users
        else 0.0
    )
    print(
        f"MF: {mf_matrix.shape[0]:,} users x "
        f"{mf_matrix.shape[1]:,} items; "
        f"{len(mf_train_days):,} pairs; k={MF_K}; "
        f"users in training={mf_users_in_training:,}/"
        f"{len(users):,}; "
        f"pool items seen={mf_pool_coverage:,}/{len(pool):,}; "
        f"unseen share of slots={mf_unseen_share:.2%}; "
        f"fill={mf_fill:.2%}; svd+score={mf_svd_time:.2f}s; "
        f"total={mf_elapsed:.2f}s"
    )
    print()

    blend_started = time.perf_counter()
    history_features = build_distinct_days_features(
        interactions, HISTORY_WEEKS, pool
    )
    history_share = build_history_share(
        history_features,
        users,
        pool,
        user_index,
        pool_index,
    )
    mf_normalized = minmax_rows(mf_scores)
    blend_scores = (
        MF_BLEND_A * history_share
        + MF_BLEND_B * mf_normalized
        + (1.0 - MF_BLEND_A - MF_BLEND_B)
        * popularity_share[None, :]
    )
    blend_recommendations = topk_recs(blend_scores, pool)
    blend_fill, _ = write_submission(
        "mf_blend.csv",
        users,
        blend_recommendations,
        popularity,
        pool,
    )
    blend_elapsed = time.perf_counter() - blend_started
    print(
        f"MF blend: a={MF_BLEND_A:.2f}, b={MF_BLEND_B:.2f}, "
        f"popularity={1.0 - MF_BLEND_A - MF_BLEND_B:.2f}; "
        f"fill={blend_fill:.2%}; total={blend_elapsed:.2f}s"
    )
    print(
        f"popularity/share build: {popularity_build:.2f}s; "
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
