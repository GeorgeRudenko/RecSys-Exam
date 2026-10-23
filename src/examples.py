import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import (
    DATA,
    POPULARITY_WEEKS,
    TOP_K,
    TRAIN_WEEKS,
    VALID_WEEKS,
    build_distinct_days_features,
    build_truth,
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

RESULTS_DIR = Path("results")
METHOD_ORDER = (
    "popularity",
    "content_tfidf",
    "mf_k16",
    "hist_pop_w0.30",
    "final_mf_blend",
)
METHOD_LABELS = {
    "popularity": "popularity",
    "content_tfidf": "content_tfidf",
    "mf_k16": "mf_k16",
    "hist_pop_w0.30": "hist_pop_w0.30",
    "final_mf_blend": "final mf_blend (a=0.30, b=0.02, pop=0.68)",
}


def load_metadata():
    frames = []
    for file_name in (
        "export_arh_11-20-final.csv",
        "export_arh_21-30-final.csv",
    ):
        frame = pd.read_csv(
            DATA / file_name,
            usecols=[
                "tv_show_id",
                "tv_show_title",
                "tv_show_category",
            ],
            encoding="utf-8",
        )
        frame = frame[frame["tv_show_id"] != 0].copy()
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True)
    metadata["tv_show_id"] = metadata["tv_show_id"].astype(int)
    metadata["tv_show_title"] = (
        metadata["tv_show_title"].fillna("UNKNOWN").astype(str).str.strip()
    )
    metadata["tv_show_category"] = (
        metadata["tv_show_category"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )
    metadata.sort_values("tv_show_id", inplace=True)
    metadata.drop_duplicates("tv_show_id", keep="first", inplace=True)
    return metadata.set_index("tv_show_id")


def build_user_stats(interactions, users, truth):
    train = interactions[
        interactions["iso_week"].between(*TRAIN_WEEKS)
    ]
    counts = (
        train.groupby("user_id")["tv_show_id"]
        .nunique()
        .rename("train_programmes")
    )
    eligible = pd.DataFrame(
        {"user_id": [user for user in users if user in truth]}
    )
    eligible = eligible.merge(
        counts, left_on="user_id", right_index=True, how="left"
    )
    eligible["train_programmes"] = (
        eligible["train_programmes"].fillna(0).astype(int)
    )
    eligible["truth_size"] = (
        eligible["user_id"].map(lambda user: len(truth[user])).astype(int)
    )
    return eligible


def select_users(eligible):
    heavy_threshold = eligible["train_programmes"].quantile(0.90)
    light_threshold = eligible["train_programmes"].quantile(0.10)
    heavy_row = (
        eligible[eligible["train_programmes"] >= heavy_threshold]
        .sort_values(
            ["train_programmes", "truth_size", "user_id"],
            ascending=[False, False, True],
        )
        .iloc[0]
    )
    light_pool = eligible[
        (eligible["train_programmes"] <= light_threshold)
        & (eligible["train_programmes"] >= 3)
    ]
    light_five = light_pool[light_pool["train_programmes"] >= 5]
    if not light_five.empty:
        light_pool = light_five
    light_row = (
        light_pool.sort_values(
            ["train_programmes", "truth_size", "user_id"],
            ascending=[True, True, True],
        )
        .iloc[0]
    )
    heavy_user = int(eligible.at[heavy_row.name, "user_id"])
    light_user = int(eligible.at[light_row.name, "user_id"])
    heavy_programmes = int(
        eligible.at[heavy_row.name, "train_programmes"]
    )
    light_programmes = int(
        eligible.at[light_row.name, "train_programmes"]
    )
    heavy_truth = int(eligible.at[heavy_row.name, "truth_size"])
    light_truth = int(eligible.at[light_row.name, "truth_size"])
    selected = [
        (
            "heavy",
            heavy_user,
            f"top decile (threshold {heavy_threshold:.0f}); "
            f"selected {heavy_programmes} training programmes; "
            f"truth size {heavy_truth}",
            heavy_programmes,
            heavy_truth,
        ),
        (
            "light",
            light_user,
            f"bottom decile (threshold {light_threshold:.0f}); "
            f"selected {light_programmes} training programmes "
            f"(>=3); truth size {light_truth}",
            light_programmes,
            light_truth,
        ),
    ]
    return selected


def get_metadata(metadata, item):
    if item in metadata.index:
        row = metadata.loc[item]
        return str(row["tv_show_title"]), str(row["tv_show_category"])
    return "UNKNOWN", "UNKNOWN"


def training_top(interactions, user, metadata, top_n=5):
    train = interactions[
        (interactions["user_id"] == user)
        & interactions["iso_week"].between(*TRAIN_WEEKS)
    ].copy()
    train["airing_day"] = train["airing_start"].dt.normalize()
    days = (
        train.groupby("tv_show_id")["airing_day"]
        .nunique()
        .rename("distinct_days")
        .reset_index()
    )
    days.sort_values(
        ["distinct_days", "tv_show_id"],
        ascending=[False, True],
        inplace=True,
    )
    days = days.head(top_n).reset_index(drop=True)
    days["title"] = days["tv_show_id"].map(
        metadata["tv_show_title"]
    ).fillna("UNKNOWN")
    days["category"] = days["tv_show_id"].map(
        metadata["tv_show_category"]
    ).fillna("UNKNOWN")
    return days


def md_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def format_bool(value):
    if pd.isna(value):
        return ""
    return "yes" if value else "no"


def main():
    interactions = load_interactions()
    submission = pd.read_csv(
        DATA / "submission.csv", usecols=["user_id"]
    )
    users = [int(user) for user in submission["user_id"]]
    pool = sorted(load_candidate_pool("export_arh_21-30-final.csv"))
    truth = build_truth(interactions, VALID_WEEKS, users, pool)
    metadata = load_metadata()
    eligible = build_user_stats(interactions, users, truth)
    selected = select_users(eligible)
    selected_users = [user for _, user, _, _, _ in selected]
    user_index = {
        user: i for i, user in enumerate(selected_users)
    }
    pool_index = {
        int(item): i for i, item in enumerate(pool)
    }
    train_sets = {
        user: set(
            interactions.loc[
                (interactions["user_id"] == user)
                & interactions["iso_week"].between(*TRAIN_WEEKS),
                "tv_show_id",
            ]
        )
        for user in selected_users
    }

    popularity, _ = popularity_ranking(
        interactions, POPULARITY_WEEKS, pool
    )
    popularity_recommendations = [
        list(popularity) for _ in selected_users
    ]

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
    item_vectors, _ = build_tfidf(item_table, actor_tokens)
    content_scores, content_valid, _ = build_content_scores(
        interactions,
        selected_users,
        pool,
        item_table,
        item_vectors,
    )
    content_recommendations = [
        fill_with_popularity(recommendation, popularity)
        for recommendation in topk_recs(
            content_scores, pool, content_valid
        )
    ]

    mf_matrix, _, mf_user_index, mf_item_index = build_mf_matrix(
        interactions
    )
    mf_scores, mf_valid, _, _ = build_mf_scores(
        mf_matrix,
        mf_user_index,
        mf_item_index,
        selected_users,
        pool,
        16,
    )
    mf_recommendations = [
        fill_with_popularity(recommendation, popularity)
        for recommendation in topk_recs(mf_scores, pool, mf_valid)
    ]

    history_features = build_distinct_days_features(
        interactions, TRAIN_WEEKS, pool
    )
    history_share = build_history_share(
        history_features,
        selected_users,
        pool,
        user_index,
        pool_index,
    )
    popularity_share = build_popularity_share(interactions, pool)

    hist_pop_scores = (
        0.30 * history_share
        + 0.70 * popularity_share[None, :]
    )
    hist_pop_recommendations = [
        fill_with_popularity(recommendation, popularity)
        for recommendation in topk_recs(hist_pop_scores, pool)
    ]

    mf_normalized = minmax_rows(mf_scores)
    final_scores = (
        0.30 * history_share
        + 0.02 * mf_normalized
        + 0.68 * popularity_share[None, :]
    )
    final_recommendations = [
        fill_with_popularity(recommendation, popularity)
        for recommendation in topk_recs(final_scores, pool)
    ]

    recommendations = {
        "popularity": popularity_recommendations,
        "content_tfidf": content_recommendations,
        "mf_k16": mf_recommendations,
        "hist_pop_w0.30": hist_pop_recommendations,
        "final_mf_blend": final_recommendations,
    }

    records = []
    for i, (
        group,
        user,
        reason,
        train_programmes,
        truth_size,
    ) in enumerate(selected):
        train_top = training_top(interactions, user, metadata)
        records.append(
            {
                "group": group,
                "user_id": user,
                "reason": reason,
                "train_programmes": train_programmes,
                "truth_size": truth_size,
                "train_top": train_top,
                "train_set": train_sets[user],
                "truth": truth[user],
                "recommendations": {
                    method: recommendations[method][i]
                    for method in METHOD_ORDER
                },
            }
        )

    csv_rows = []
    md_lines = [
        "# Validation recommendation examples",
        "",
        "Validation split: training weeks 11-20; truth weeks 21-30. "
        "`hit` means the item is in the user's validation truth; "
        "`repeat` means the user watched it in training weeks 11-20.",
        "",
    ]
    for record in records:
        user = record["user_id"]
        md_lines.extend(
            [
                f"## {record['group'].title()} viewer",
                "",
                f"- user_id: `{user}`",
                f"- why chosen: {record['reason']}",
                f"- training programmes: {record['train_programmes']}",
                f"- validation truth size: {record['truth_size']}",
                "",
                "### Training: 5 most-watched programmes by distinct days",
                "",
                "| rank | tv_show_id | title | category | distinct_days |",
                "|---:|---:|---|---|---:|",
            ]
        )
        for rank, row in enumerate(
            record["train_top"].itertuples(index=False), start=1
        ):
            title = md_escape(row.title)
            category = md_escape(row.category)
            md_lines.append(
                f"| {rank} | {int(row.tv_show_id)} | {title} | "
                f"{category} | {int(row.distinct_days)} |"
            )
            csv_rows.append(
                {
                    "user_id": user,
                    "user_group": record["group"],
                    "selection_reason": record["reason"],
                    "train_programmes": record["train_programmes"],
                    "truth_size": record["truth_size"],
                    "table": "training_top5",
                    "method_label": "",
                    "rank": rank,
                    "tv_show_id": int(row.tv_show_id),
                    "title": str(row.title),
                    "category": str(row.category),
                    "hit": "",
                    "repeat": "",
                    "distinct_days": int(row.distinct_days),
                }
            )
        md_lines.append("")

        for method in METHOD_ORDER:
            label = METHOD_LABELS[method]
            method_rows = []
            for rank, item in enumerate(
                record["recommendations"][method], start=1
            ):
                title, category = get_metadata(metadata, item)
                hit = item in record["truth"]
                repeat = item in record["train_set"]
                method_rows.append(
                    {
                        "rank": rank,
                        "tv_show_id": item,
                        "title": title,
                        "category": category,
                        "hit": hit,
                        "repeat": repeat,
                    }
                )
                csv_rows.append(
                    {
                        "user_id": user,
                        "user_group": record["group"],
                        "selection_reason": record["reason"],
                        "train_programmes": record["train_programmes"],
                        "truth_size": record["truth_size"],
                        "table": method,
                        "method_label": label,
                        "rank": rank,
                        "tv_show_id": item,
                        "title": title,
                        "category": category,
                        "hit": hit,
                        "repeat": repeat,
                        "distinct_days": "",
                    }
                )
            hits = sum(row["hit"] for row in method_rows)
            repeats = sum(row["repeat"] for row in method_rows)
            md_lines.extend(
                [
                    f"### {label}",
                    "",
                    f"Hits@5: {hits}/5; repeats in Top-5: {repeats}/5",
                    "",
                    "| rank | tv_show_id | title | category | hit | repeat |",
                    "|---:|---:|---|---|:---:|:---:|",
                ]
            )
            for row in method_rows:
                md_lines.append(
                    f"| {row['rank']} | {row['tv_show_id']} | "
                    f"{md_escape(row['title'])} | "
                    f"{md_escape(row['category'])} | "
                    f"{format_bool(row['hit'])} | "
                    f"{format_bool(row['repeat'])} |"
                )
            md_lines.append("")

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "examples.csv"
    md_path = RESULTS_DIR / "examples.md"
    csv_columns = [
        "user_id",
        "user_group",
        "selection_reason",
        "train_programmes",
        "truth_size",
        "table",
        "method_label",
        "rank",
        "tv_show_id",
        "title",
        "category",
        "hit",
        "repeat",
        "distinct_days",
    ]
    pd.DataFrame(csv_rows, columns=csv_columns).to_csv(
        csv_path,
        index=False,
        lineterminator="\n",
        encoding="utf-8",
    )
    md_text = "\n".join(md_lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    print(md_text)
    print(f"saved {csv_path}")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
