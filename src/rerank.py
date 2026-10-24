import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import (
    DATA,
    TOP_K,
    ap_at_5,
    fill_with_popularity,
    load_interactions,
    popularity_ranking,
)
from validate_methods import (
    build_mf_matrix,
    build_mf_scores,
    paired_bootstrap,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

T = 0.30
STRICT_FRACTION = 0.8
REPEAT_DEPTH = 50
POP_DEPTH = 20
MF_DEPTH = 20
MF_K = 16
SEED = 0
TRAIN_HISTORY = (11, 15)
TRAIN_TARGET = (16, 20)
EVAL_HISTORY = (11, 20)
EVAL_TARGET = (21, 30)
FINAL_HISTORY = (11, 30)
FINAL_TARGET = (31, 42)
TRAIN_SCHEDULE = "export_arh_11-20-final.csv"
EVAL_SCHEDULE = "export_arh_21-30-final.csv"
FINAL_SCHEDULE = "export_arh_31-42-final.csv"
REFERENCE = Path("submissions/mf_blend.csv")
RERANK_SUBMISSION = Path("submissions/rerank.csv")

FEATURES = [
    "repeat_count_ge_030",
    "repeat_count_ge_050",
    "repeat_count_gt_080",
    "mean_fraction",
    "max_fraction",
    "days_since_last_qualifying",
    "distinct_weeks_watched",
    "watched_last_2_weeks",
    "repeat_rank",
    "old_pop_score",
    "old_pop_rank",
    "strict_pop_score",
    "strict_pop_rank",
    "mf_score",
    "mf_rank",
    "schedule_airings",
    "schedule_channels",
    "user_channel_share_max",
    "user_distinct_programmes",
    "user_view_hours",
    "cold_start",
]


def load_target(file_name, weeks):
    schedule = pd.read_csv(
        DATA / file_name,
        usecols=["channel_id", "start_time", "tv_show_id"],
        parse_dates=["start_time"],
        dayfirst=True,
        dtype={"channel_id": "int32", "tv_show_id": "int64"},
    )
    week = schedule["start_time"].dt.isocalendar().week
    labelled = schedule[
        (schedule["tv_show_id"] > 0) & week.between(*weeks)
    ].copy()
    pool = sorted({int(item) for item in labelled["tv_show_id"].unique()})
    stats = (
        labelled.groupby("tv_show_id", sort=False)
        .agg(
            schedule_airings=("start_time", "size"),
            schedule_channels=("channel_id", "nunique"),
        )
        .reset_index()
    )
    channels = {
        int(item): tuple(sorted(set(int(x) for x in group["channel_id"])))
        for item, group in labelled.groupby("tv_show_id", sort=False)
    }
    return pool, stats, channels, labelled["start_time"].min()


def load_view_aggregates():
    views = pd.read_csv(
        DATA / "dataset11-30.csv",
        usecols=["user_id", "vsetv_id", "start_time", "duraton"],
        parse_dates=["start_time"],
        dtype={
            "user_id": "uint64",
            "vsetv_id": "int32",
            "duraton": "float64",
        },
    )
    views["iso_week"] = (
        views["start_time"].dt.isocalendar().week.astype("int16")
    )
    views = views[views["iso_week"].between(11, 30)]
    windows = {
        "train": TRAIN_HISTORY,
        "eval": EVAL_HISTORY,
        "final": FINAL_HISTORY,
    }
    aggregates = {}
    for name, weeks in windows.items():
        subset = views[views["iso_week"].between(*weeks)]
        grouped = (
            subset.groupby(["user_id", "vsetv_id"], sort=False)["duraton"]
            .sum()
            .reset_index()
        )
        totals = grouped.groupby("user_id")["duraton"].sum()
        grouped["share"] = grouped["duraton"] / grouped["user_id"].map(
            totals
        )
        shares = {
            (int(user), int(channel)): float(share)
            for user, channel, share in zip(
                grouped["user_id"],
                grouped["vsetv_id"],
                grouped["share"],
            )
        }
        hours = {
            int(user): float(seconds / 3600.0)
            for user, seconds in totals.items()
        }
        aggregates[name] = (shares, hours)
    return aggregates


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


def build_repeat_rankings(occurrences, weeks, pool, users, depth):
    pool_set = set(pool)
    subset = occurrences[
        occurrences["iso_week"].between(*weeks)
        & occurrences["tv_show_id"].isin(pool_set)
        & (occurrences["fraction"] >= T)
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
    features = features.groupby("user_id", sort=False).head(depth)
    return {
        int(user): [int(item) for item in group["tv_show_id"]]
        for user, group in features.groupby("user_id", sort=False)
    }


def build_occurrence_stats(occurrences, weeks, target_start):
    history = occurrences[
        occurrences["iso_week"].between(*weeks)
    ].copy()
    history["ge030"] = history["fraction"] >= 0.30
    history["ge050"] = history["fraction"] >= 0.50
    history["gt080"] = history["fraction"] > STRICT_FRACTION
    stats = (
        history.groupby(["user_id", "tv_show_id"], sort=False)
        .agg(
            repeat_count_ge_030=("ge030", "sum"),
            repeat_count_ge_050=("ge050", "sum"),
            repeat_count_gt_080=("gt080", "sum"),
            mean_fraction=("fraction", "mean"),
            max_fraction=("fraction", "max"),
        )
        .reset_index()
    )
    qualifying = history[history["fraction"] >= T]
    qstats = (
        qualifying.groupby(["user_id", "tv_show_id"], sort=False)
        .agg(
            last_qualifying=("airing_start", "max"),
            last_qualifying_week=("iso_week", "max"),
            distinct_weeks_watched=("iso_week", "nunique"),
        )
        .reset_index()
    )
    stats = stats.merge(
        qstats, on=["user_id", "tv_show_id"], how="left"
    )
    stats["days_since_last_qualifying"] = (
        target_start - stats["last_qualifying"]
    ).dt.days
    stats["days_since_last_qualifying"] = stats[
        "days_since_last_qualifying"
    ].fillna(9999).astype("int32")
    stats["watched_last_2_weeks"] = (
        stats["last_qualifying_week"] >= weeks[1] - 1
    ).fillna(False).astype("int8")
    stats["distinct_weeks_watched"] = stats[
        "distinct_weeks_watched"
    ].fillna(0).astype("int16")
    user_programmes = (
        history.groupby("user_id", sort=False)["tv_show_id"]
        .nunique()
        .rename("user_distinct_programmes")
        .reset_index()
    )
    return stats, user_programmes


def build_popularity_features(
    interactions, occurrences, weeks, pool
):
    last_five = (weeks[1] - 4, weeks[1])
    old_counts = (
        interactions[
            interactions["iso_week"].between(*last_five)
        ]
        .groupby("tv_show_id")["user_id"]
        .nunique()
    )
    strict = occurrences[
        occurrences["iso_week"].between(*last_five)
        & (occurrences["fraction"] > STRICT_FRACTION)
    ]
    strict_counts = strict.groupby("tv_show_id")["user_id"].nunique()
    frame = pd.DataFrame({"tv_show_id": sorted(pool)})
    frame["old_pop_score"] = (
        frame["tv_show_id"].map(old_counts).fillna(0).astype("int32")
    )
    frame["strict_pop_score"] = (
        frame["tv_show_id"].map(strict_counts).fillna(0).astype("int32")
    )
    for prefix in ("old", "strict"):
        column = f"{prefix}_pop_score"
        order = frame.sort_values(
            [column, "tv_show_id"], ascending=[False, True]
        ).index
        ranks = np.empty(len(frame), dtype=np.int32)
        ranks[order] = np.arange(1, len(frame) + 1, dtype=np.int32)
        frame[f"{prefix}_pop_rank"] = ranks
    top20 = [
        int(item)
        for item in frame.sort_values("strict_pop_rank")["tv_show_id"].head(
            POP_DEPTH
        )
    ]
    fill5 = [
        int(item)
        for item in frame.sort_values("strict_pop_rank")["tv_show_id"].head(
            TOP_K
        )
    ]
    return frame, top20, fill5


def build_mf(
    interactions, weeks, users, pool
):
    matrix, _, user_index, item_index = build_mf_matrix(
        interactions, weeks
    )
    started = time.perf_counter()
    scores, valid, _, _ = build_mf_scores(
        matrix,
        user_index,
        item_index,
        users,
        pool,
        MF_K,
    )
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(users))[:, None], order] = np.arange(
        1, len(pool) + 1, dtype=np.int32
    )
    elapsed = time.perf_counter() - started
    return scores, ranks, valid, order, elapsed


def build_candidates(
    users, repeat_rankings, strict_top20, mf_order, mf_valid, pool
):
    user_values = []
    item_values = []
    for index, user in enumerate(users):
        items = set(repeat_rankings.get(user, []))
        items.update(strict_top20)
        if mf_valid[index]:
            items.update(
                int(pool[j]) for j in mf_order[index, :MF_DEPTH]
            )
        for item in items:
            user_values.append(user)
            item_values.append(item)
    return pd.DataFrame(
        {
            "user_id": np.asarray(user_values, dtype=np.uint64),
            "tv_show_id": np.asarray(item_values, dtype=np.int64),
        }
    )


def build_features(
    candidates,
    occurrence_stats,
    repeat_rankings,
    popularity_frame,
    schedule_stats,
    mf_scores,
    mf_ranks,
    mf_valid,
    user_programmes,
    user_hours,
    users,
    pool,
    show_channels,
    channel_shares,
):
    frame = candidates.merge(
        occurrence_stats, on=["user_id", "tv_show_id"], how="left"
    )
    for column in (
        "repeat_count_ge_030",
        "repeat_count_ge_050",
        "repeat_count_gt_080",
        "mean_fraction",
        "max_fraction",
        "distinct_weeks_watched",
        "watched_last_2_weeks",
    ):
        frame[column] = frame[column].fillna(0)
    frame["days_since_last_qualifying"] = frame[
        "days_since_last_qualifying"
    ].fillna(9999)

    repeat_rows = [
        (user, item, rank)
        for user, items in repeat_rankings.items()
        for rank, item in enumerate(items, start=1)
    ]
    repeat_frame = pd.DataFrame(
        repeat_rows, columns=["user_id", "tv_show_id", "repeat_rank"]
    )
    repeat_frame["user_id"] = repeat_frame["user_id"].astype("uint64")
    repeat_frame["tv_show_id"] = repeat_frame["tv_show_id"].astype("int64")
    frame = frame.merge(
        repeat_frame, on=["user_id", "tv_show_id"], how="left"
    )
    frame["repeat_rank"] = frame["repeat_rank"].fillna(
        REPEAT_DEPTH + 1
    ).astype("int16")

    frame = frame.merge(
        popularity_frame, on="tv_show_id", how="left"
    )
    frame = frame.merge(
        schedule_stats, on="tv_show_id", how="left"
    )
    frame = frame.merge(
        user_programmes, on="user_id", how="left"
    )
    frame["user_distinct_programmes"] = frame[
        "user_distinct_programmes"
    ].fillna(0).astype("int32")
    frame["user_view_hours"] = (
        frame["user_id"].map(user_hours).fillna(0.0).astype("float64")
    )
    frame["cold_start"] = (
        frame["user_distinct_programmes"] == 0
    ).astype("int8")

    user_index = {int(user): i for i, user in enumerate(users)}
    item_index = {int(item): i for i, item in enumerate(pool)}
    user_rows = np.asarray(
        [user_index[int(user)] for user in frame["user_id"]],
        dtype=np.int32,
    )
    item_rows = np.asarray(
        [item_index[int(item)] for item in frame["tv_show_id"]],
        dtype=np.int32,
    )
    frame["mf_score"] = mf_scores[user_rows, item_rows]
    frame["mf_rank"] = np.where(
        mf_valid[user_rows],
        mf_ranks[user_rows, item_rows],
        len(pool) + 1,
    )

    share_values = []
    for user, item in zip(frame["user_id"], frame["tv_show_id"]):
        channels = show_channels.get(int(item), ())
        best = 0.0
        for channel in channels:
            share = channel_shares.get((int(user), channel), 0.0)
            if share > best:
                best = share
        share_values.append(best)
    frame["user_channel_share_max"] = share_values

    for column in FEATURES:
        if column not in frame:
            raise ValueError(f"missing feature {column}")
    return frame[["user_id", "tv_show_id"] + FEATURES].copy()


def add_labels(frame, truth):
    labels = np.fromiter(
        (
            1 if int(item) in truth.get(int(user), set()) else 0
            for user, item in zip(frame["user_id"], frame["tv_show_id"])
        ),
        dtype=np.int8,
        count=len(frame),
    )
    frame["label"] = labels
    return frame


def candidate_metrics(candidates, truth, users):
    grouped = {
        int(user): set(int(item) for item in group["tv_show_id"])
        for user, group in candidates.groupby("user_id", sort=False)
    }
    recalls = []
    oracle = []
    truth_users = 0
    for user in users:
        relevant = truth.get(user, set())
        if not relevant:
            oracle.append(0.0)
            continue
        truth_users += 1
        selected = grouped.get(user, set())
        hits = len(relevant & selected)
        recalls.append(hits / len(relevant))
        oracle.append(min(hits, TOP_K) / min(len(relevant), TOP_K))
    return {
        "candidate_recall": float(np.mean(recalls)) if recalls else 0.0,
        "candidate_micro_recall": (
            sum(
                len(truth.get(user, set()) & grouped.get(user, set()))
                for user in users
            )
            / sum(len(truth.get(user, set())) for user in users)
            if users
            else 0.0
        ),
        "oracle_map_all": float(np.mean(oracle)) if oracle else 0.0,
        "oracle_map_truth_users": (
            float(
                np.mean(
                    [
                        oracle[i]
                        for i, user in enumerate(users)
                        if truth.get(user)
                    ]
                )
            )
            if truth_users
            else 0.0
        ),
        "truth_users": truth_users,
        "avg_candidates": len(candidates) / len(users) if users else 0.0,
    }


def recommendations_from_probabilities(frame, probabilities):
    scored = frame[["user_id", "tv_show_id"]].copy()
    scored["probability"] = probabilities
    recommendations = {}
    for user, group in scored.groupby("user_id", sort=False):
        group = group.sort_values(
            ["probability", "tv_show_id"],
            ascending=[False, True],
        )
        recommendations[int(user)] = [
            int(item) for item in group["tv_show_id"].head(TOP_K)
        ]
    return recommendations


def evaluate_recommendations(recommendations, users, truth, fill):
    aps = []
    filled = 0
    for user in users:
        recommendation = recommendations.get(user, [])
        if len(recommendation) < TOP_K:
            filled += 1
        recommendation = fill_with_popularity(recommendation, fill)
        aps.append(ap_at_5(recommendation, truth.get(user, set())))
    return {
        "map5": float(np.mean(aps)) if aps else 0.0,
        "aps": np.asarray(aps, dtype=np.float64),
        "fill_share": filled / len(users) if users else 0.0,
    }


def fit_and_predict(train_frame, eval_frame, model_name, c_value=None):
    x_train = train_frame[FEATURES].to_numpy(dtype=np.float64)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    x_eval = eval_frame[FEATURES].to_numpy(dtype=np.float64)
    started = time.perf_counter()
    scaler = None
    if model_name == "logistic":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_eval = scaler.transform(x_eval)
        model = LogisticRegression(
            C=c_value, max_iter=2000, random_state=SEED
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_eval)[:, 1]
    elif model_name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=300,
            num_leaves=15,
            learning_rate=0.05,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_eval)[:, 1]
    else:
        raise ValueError(model_name)
    return probabilities, model, scaler, time.perf_counter() - started


def build_fold(
    interactions,
    occurrences,
    view_aggregates,
    schedule_file,
    target_weeks,
    history_weeks,
    users,
    use_view_key,
):
    pool, schedule_stats, show_channels, target_start = load_target(
        schedule_file, target_weeks
    )
    truth = build_strict_truth(
        occurrences, target_weeks, users, pool
    )
    occurrence_stats, user_programmes = build_occurrence_stats(
        occurrences, history_weeks, target_start
    )
    popularity_frame, strict_top20, fill = build_popularity_features(
        interactions, occurrences, history_weeks, pool
    )
    repeat_rankings = build_repeat_rankings(
        occurrences, history_weeks, pool, users, REPEAT_DEPTH
    )
    mf_scores, mf_ranks, mf_valid, mf_order, mf_elapsed = build_mf(
        interactions, history_weeks, users, pool
    )
    candidates = build_candidates(
        users,
        repeat_rankings,
        strict_top20,
        mf_order,
        mf_valid,
        pool,
    )
    channel_shares, user_hours = view_aggregates[use_view_key]
    user_hours = {
        user: user_hours.get(int(user), 0.0) for user in users
    }
    features = build_features(
        candidates,
        occurrence_stats,
        repeat_rankings,
        popularity_frame,
        schedule_stats,
        mf_scores,
        mf_ranks,
        mf_valid,
        user_programmes,
        user_hours,
        users,
        pool,
        show_channels,
        channel_shares,
    )
    features = add_labels(features, truth)
    return {
        "pool": pool,
        "truth": truth,
        "schedule_stats": schedule_stats,
        "show_channels": show_channels,
        "target_start": target_start,
        "occurrence_stats": occurrence_stats,
        "user_programmes": user_programmes,
        "popularity_frame": popularity_frame,
        "strict_top20": strict_top20,
        "fill": fill,
        "repeat_rankings": repeat_rankings,
        "mf_scores": mf_scores,
        "mf_ranks": mf_ranks,
        "mf_valid": mf_valid,
        "mf_order": mf_order,
        "candidates": candidates,
        "features": features,
        "mf_elapsed": mf_elapsed,
        "channel_shares": channel_shares,
        "user_hours": user_hours,
    }


def write_submission(recommendations, users, reference, pool, fill):
    rows = []
    for user in users:
        recommendation = recommendations.get(user, [])
        recommendation = fill_with_popularity(recommendation, fill)
        rows.append(" ".join(map(str, recommendation)))
    output = pd.DataFrame(
        {"user_id": reference["user_id"].values, "tv_show_id": rows}
    )
    RERANK_SUBMISSION.parent.mkdir(exist_ok=True)
    csv_text = output.to_csv(index=False, lineterminator="\n")
    RERANK_SUBMISSION.write_bytes(
        csv_text.rstrip("\n").encode("utf-8")
    )

    reference_bytes = REFERENCE.read_bytes()
    produced_bytes = RERANK_SUBMISSION.read_bytes()
    produced = pd.read_csv(
        RERANK_SUBMISSION,
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
            produced["tv_show_id"].str.fullmatch(r"\d+( \d+){4}").all()
        ),
        "five_unique_ids": all(
            len(set(row)) == TOP_K for row in produced_rows
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
    interactions = load_interactions()
    occurrences = pd.read_parquet(DATA / "occurrences.parquet")
    reference = pd.read_csv(
        REFERENCE, dtype={"user_id": "uint64"}
    )
    users = [int(user) for user in reference["user_id"]]
    view_started = time.perf_counter()
    view_aggregates = load_view_aggregates()
    view_elapsed = time.perf_counter() - view_started

    fold_started = time.perf_counter()
    train_fold = build_fold(
        interactions,
        occurrences,
        view_aggregates,
        TRAIN_SCHEDULE,
        TRAIN_TARGET,
        TRAIN_HISTORY,
        users,
        "train",
    )
    eval_fold = build_fold(
        interactions,
        occurrences,
        view_aggregates,
        EVAL_SCHEDULE,
        EVAL_TARGET,
        EVAL_HISTORY,
        users,
        "eval",
    )
    fold_elapsed = time.perf_counter() - fold_started

    train_truth = train_fold["truth"]
    eval_truth = eval_fold["truth"]

    train_features = train_fold["features"]
    train_users = set(
        int(user)
        for user in train_fold["user_programmes"]["user_id"]
    )
    train_features = train_features[
        train_features["user_id"].isin(train_users)
    ].reset_index(drop=True)
    eval_features = eval_fold["features"]
    eval_candidates = eval_fold["candidates"]
    eval_metrics = candidate_metrics(
        eval_candidates, eval_truth, users
    )

    print(
        f"candidate generation: repeat top{REPEAT_DEPTH} "
        f"t={T:.2f}, strict-pop top{POP_DEPTH}, MF k={MF_K} "
        f"top{MF_DEPTH}"
    )
    print(
        f"  train rows: {len(train_features):,}; train users: "
        f"{train_features['user_id'].nunique():,}; "
        f"positives: {int(train_features['label'].sum()):,}"
    )
    print(
        f"  eval rows: {len(eval_features):,}; avg candidates/user: "
        f"{eval_metrics['avg_candidates']:.1f}"
    )
    print(
        f"  strict-truth candidate recall: "
        f"{eval_metrics['candidate_recall']:.4f} macro, "
        f"{eval_metrics['candidate_micro_recall']:.4f} micro; "
        f"oracle MAP@5: {eval_metrics['oracle_map_all']:.5f} "
        f"(all users), "
        f"{eval_metrics['oracle_map_truth_users']:.5f} "
        f"({eval_metrics['truth_users']:,} truth users)"
    )
    print()

    repeat_recommendations = build_repeat_rankings(
        occurrences, EVAL_HISTORY, eval_fold["pool"], users, TOP_K
    )
    repeat_result = evaluate_recommendations(
        repeat_recommendations,
        users,
        eval_truth,
        eval_fold["fill"],
    )
    model_results = []
    models = [
        ("logistic_c0.1", "logistic", 0.1),
        ("logistic_c1.0", "logistic", 1.0),
        ("lightgbm", "lightgbm", None),
    ]
    for name, kind, c_value in models:
        probabilities, model, scaler, fit_elapsed = fit_and_predict(
            train_features,
            eval_features,
            kind,
            c_value,
        )
        recommendations = recommendations_from_probabilities(
            eval_features, probabilities
        )
        result = evaluate_recommendations(
            recommendations,
            users,
            eval_truth,
            eval_fold["fill"],
        )
        result.update(
            {
                "name": name,
                "model": model,
                "scaler": scaler,
                "fit_seconds": fit_elapsed,
            }
        )
        model_results.append(result)
        print(
            f"trained {name}: strict MAP@5={result['map5']:.5f}; "
            f"fill={result['fill_share']:.2%}; "
            f"fit={fit_elapsed:.2f}s"
        )

    print()
    print(
        f"repeat baseline: strict MAP@5={repeat_result['map5']:.5f}; "
        f"fill={repeat_result['fill_share']:.2%}"
    )
    best_result = max(model_results, key=lambda result: result["map5"])
    mean_diff, ci_lower, ci_upper = paired_bootstrap(
        best_result["aps"] - repeat_result["aps"]
    )
    print(
        f"best reranker {best_result['name']} vs repeat: "
        f"mean diff {mean_diff:+.5f}, 95% CI "
        f"[{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )
    print()

    if best_result["name"].startswith("logistic"):
        coefficients = best_result["model"].coef_[0]
        order = np.argsort(-np.abs(coefficients))[:10]
        print("logistic coefficients (standardized, top 10 by |value|):")
        for index in order:
            print(
                f"  {FEATURES[index]:<30} {coefficients[index]:+.5f}"
            )
    else:
        importances = best_result["model"].feature_importances_
        order = np.argsort(-importances)[:10]
        print("LightGBM importances (top 10):")
        for index in order:
            print(
                f"  {FEATURES[index]:<30} {importances[index]:.5f}"
            )
    print()

    pre_final_elapsed = time.perf_counter() - total_started
    print(
        f"runtimes: views {view_elapsed:.2f}s; folds {fold_elapsed:.2f}s; "
        f"train/eval {pre_final_elapsed:.2f}s"
    )

    if mean_diff > 0 and ci_lower > 0:
        print("reranker beats repeat with CI above zero; building final submission")
        final_started = time.perf_counter()
        final_fold = build_fold(
            interactions,
            occurrences,
            view_aggregates,
            FINAL_SCHEDULE,
            FINAL_TARGET,
            FINAL_HISTORY,
            users,
            "final",
        )
        final_features = final_fold["features"]
        final_probabilities = []
        for train_frame, label in (
            (train_features, "fold 11-15 -> 16-20"),
            (eval_features, "fold 11-20 -> 21-30"),
        ):
            probabilities, _, _, fit_elapsed = fit_and_predict(
                train_frame,
                final_features,
                best_result["name"].split("_")[0],
                0.1 if best_result["name"] == "logistic_c0.1" else 1.0,
            )
            final_probabilities.append(probabilities)
            print(f"  final model {label}: fit {fit_elapsed:.2f}s")
        probabilities = np.mean(final_probabilities, axis=0)
        recommendations = recommendations_from_probabilities(
            final_features, probabilities
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
            f"wrote {RERANK_SUBMISSION} ({len(reference):,} rows); "
            f"final time {final_elapsed:.2f}s"
        )
        if not all(checks.values()):
            raise SystemExit(1)
    else:
        print("no rerank submission: CI condition not met")

    print(
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
