import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rerank

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_FEATURES = list(rerank.FEATURES)
SLOT_FEATURES = [
    "slot_score_sum",
    "slot_score_max",
    "slot_top10_airings",
    "user_channel_share_sum",
]
ALL_FEATURES = BASE_FEATURES + SLOT_FEATURES

TRAIN2_HISTORY = (11, 17)
TRAIN2_TARGET = (18, 20)
FOLD4_HISTORY = (11, 25)
FOLD4_TARGET = (26, 30)
FINAL_HISTORY = (11, 30)
FINAL_TARGET = (31, 42)
TRAIN_SCHEDULE = "export_arh_11-20-final.csv"
EVAL_SCHEDULE = "export_arh_21-30-final.csv"
FINAL_SCHEDULE = "export_arh_31-42-final.csv"
REFERENCE = Path("submissions/mf_blend.csv")
SUBMISSION = Path("submissions/rerank2.csv")
PREVIOUS_BEST = 0.58578
SEED = 0
MAX_VIEW_SECONDS = 3 * 3600


def load_raw_views():
    views = pd.read_csv(
        rerank.DATA / "dataset11-30.csv",
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
    return views[views["iso_week"].between(11, 30)].copy()


def build_view_aggregate(views, weeks):
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
    return shares, hours


def build_slot_data(views, weeks):
    subset = views[views["iso_week"].between(*weeks)].copy()
    subset["seconds"] = subset["duraton"].clip(upper=MAX_VIEW_SECONDS)
    subset["dow"] = subset["start_time"].dt.dayofweek.astype("int8")
    subset["hour"] = subset["start_time"].dt.hour.astype("int8")
    totals = subset.groupby("user_id")["seconds"].sum()
    grouped = (
        subset.groupby(
            ["user_id", "vsetv_id", "dow", "hour"], sort=False
        )["seconds"]
        .sum()
        .reset_index()
    )
    grouped["share"] = grouped["seconds"] / grouped["user_id"].map(
        totals
    )
    shares_by_user = {}
    for row in grouped.itertuples(index=False):
        shares_by_user.setdefault(int(row.user_id), {})[
            (int(row.vsetv_id), int(row.dow), int(row.hour))
        ] = float(row.share)
    ordered = grouped.sort_values(
        ["user_id", "share", "vsetv_id", "dow", "hour"],
        ascending=[True, False, True, True, True],
    )
    top = ordered.groupby("user_id", sort=False).head(10)
    top10_by_user = {
        int(user): {
            (int(channel), int(dow), int(hour))
            for channel, dow, hour in zip(
                group["vsetv_id"], group["dow"], group["hour"]
            )
        }
        for user, group in top.groupby("user_id", sort=False)
    }
    return shares_by_user, top10_by_user


def load_item_slots(file_name, weeks):
    schedule = pd.read_csv(
        rerank.DATA / file_name,
        usecols=["channel_id", "start_time", "tv_show_id"],
        parse_dates=["start_time"],
        dayfirst=True,
        dtype={"channel_id": "int32", "tv_show_id": "int64"},
    )
    week = schedule["start_time"].dt.isocalendar().week
    labelled = schedule[
        (schedule["tv_show_id"] > 0) & week.between(*weeks)
    ].copy()
    labelled["dow"] = labelled["start_time"].dt.dayofweek.astype("int8")
    labelled["hour"] = labelled["start_time"].dt.hour.astype("int8")
    return {
        int(item): tuple(
            (int(channel), int(dow), int(hour))
            for channel, dow, hour in zip(
                group["channel_id"], group["dow"], group["hour"]
            )
        )
        for item, group in labelled.groupby("tv_show_id", sort=False)
    }


def add_slot_features(
    frame,
    slot_data,
    item_slots,
    show_channels,
    channel_shares,
):
    frame = frame.copy()
    shares_by_user, top10_by_user = slot_data
    user_values = frame["user_id"].to_numpy()
    item_values = frame["tv_show_id"].to_numpy()
    size = len(frame)
    score_sums = np.zeros(size, dtype=np.float64)
    score_maxes = np.zeros(size, dtype=np.float64)
    top10_counts = np.zeros(size, dtype=np.int16)
    channel_sums = np.zeros(size, dtype=np.float64)
    for index in range(size):
        user = int(user_values[index])
        item = int(item_values[index])
        shares = shares_by_user.get(user)
        slots = item_slots.get(item, ())
        if shares and slots:
            top10 = top10_by_user.get(user, set())
            score_sum = 0.0
            score_max = 0.0
            top10_count = 0
            for slot in slots:
                share = shares.get(slot, 0.0)
                score_sum += share
                if share > score_max:
                    score_max = share
                if slot in top10:
                    top10_count += 1
            score_sums[index] = score_sum
            score_maxes[index] = score_max
            top10_counts[index] = top10_count
        channel_sum = 0.0
        for channel in show_channels.get(item, ()):
            channel_sum += channel_shares.get((user, channel), 0.0)
        channel_sums[index] = min(channel_sum, 1.0)
    frame["slot_score_sum"] = score_sums
    frame["slot_score_max"] = score_maxes
    frame["slot_top10_airings"] = top10_counts
    frame["user_channel_share_sum"] = channel_sums
    return frame


def add_slots_to_fold(fold, slot_data, item_slots):
    updated = dict(fold)
    updated["features"] = add_slot_features(
        fold["features"],
        slot_data,
        item_slots,
        fold["show_channels"],
        fold["channel_shares"],
    )
    return updated


def training_frame(fold, features=None):
    user_set = {
        int(user) for user in fold["user_programmes"]["user_id"]
    }
    frame = fold["features"] if features is None else features
    return frame[frame["user_id"].isin(user_set)].reset_index(drop=True)


def fit_classifier(train_frame, eval_frame, features):
    started = time.perf_counter()
    x_train = train_frame[features].to_numpy(dtype=np.float64)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    x_eval = eval_frame[features].to_numpy(dtype=np.float64)
    model = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_eval)[:, 1]
    return scores, model, time.perf_counter() - started


def fit_ranker(train_frame, eval_frame, features):
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
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, group=groups)
    scores = model.predict(x_eval)
    return scores, model, time.perf_counter() - started


def minmax_by_user(frame, scores):
    values = np.asarray(scores, dtype=np.float64)
    scored = pd.DataFrame(
        {"user_id": frame["user_id"].to_numpy(), "score": values}
    )
    grouped = scored.groupby("user_id", sort=False)["score"]
    minimum = grouped.transform("min").to_numpy(dtype=np.float64)
    maximum = grouped.transform("max").to_numpy(dtype=np.float64)
    span = maximum - minimum
    normalized = np.full_like(values, 0.5, dtype=np.float64)
    positive = span > 0
    normalized[positive] = (
        values[positive] - minimum[positive]
    ) / span[positive]
    return normalized


def evaluate_scores(name, frame, scores, users, truth, fill):
    recommendations = rerank.recommendations_from_probabilities(
        frame, scores
    )
    result = rerank.evaluate_recommendations(
        recommendations, users, truth, fill
    )
    result["name"] = name
    result["scores"] = np.asarray(scores, dtype=np.float64)
    return result


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
    raw_views = load_raw_views()
    view_aggregates = rerank.load_view_aggregates()
    view_aggregates["train2"] = build_view_aggregate(
        raw_views, TRAIN2_HISTORY
    )
    view_aggregates["fold4"] = build_view_aggregate(
        raw_views, FOLD4_HISTORY
    )
    slot_data = {
        "train": build_slot_data(raw_views, rerank.TRAIN_HISTORY),
        "train2": build_slot_data(raw_views, TRAIN2_HISTORY),
        "eval": build_slot_data(raw_views, rerank.EVAL_HISTORY),
        "fold4": build_slot_data(raw_views, FOLD4_HISTORY),
        "final": build_slot_data(raw_views, FINAL_HISTORY),
    }
    target_slots = {
        "train": load_item_slots(
            TRAIN_SCHEDULE, rerank.TRAIN_TARGET
        ),
        "train2": load_item_slots(TRAIN_SCHEDULE, TRAIN2_TARGET),
        "eval": load_item_slots(
            EVAL_SCHEDULE, rerank.EVAL_TARGET
        ),
        "fold4": load_item_slots(EVAL_SCHEDULE, FOLD4_TARGET),
        "final": load_item_slots(FINAL_SCHEDULE, FINAL_TARGET),
    }

    fold_started = time.perf_counter()
    train1_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        TRAIN_SCHEDULE,
        rerank.TRAIN_TARGET,
        rerank.TRAIN_HISTORY,
        users,
        "train",
    )
    eval_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        EVAL_SCHEDULE,
        rerank.EVAL_TARGET,
        rerank.EVAL_HISTORY,
        users,
        "eval",
    )
    train2_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        TRAIN_SCHEDULE,
        TRAIN2_TARGET,
        TRAIN2_HISTORY,
        users,
        "train2",
    )
    fold4_base = rerank.build_fold(
        interactions,
        occurrences,
        view_aggregates,
        EVAL_SCHEDULE,
        FOLD4_TARGET,
        FOLD4_HISTORY,
        users,
        "fold4",
    )
    train1 = add_slots_to_fold(
        train1_base, slot_data["train"], target_slots["train"]
    )
    eval_all = add_slots_to_fold(
        eval_base, slot_data["eval"], target_slots["eval"]
    )
    train2 = add_slots_to_fold(
        train2_base, slot_data["train2"], target_slots["train2"]
    )
    fold4 = add_slots_to_fold(
        fold4_base, slot_data["fold4"], target_slots["fold4"]
    )
    fold_elapsed = time.perf_counter() - fold_started

    train1_base_rows = training_frame(train1_base)
    train1_rows = training_frame(train1)
    train2_rows = training_frame(train2)
    eval_rows = eval_all["features"]
    eval_truth = eval_base["truth"]
    eval_fill = eval_base["fill"]

    model_started = time.perf_counter()
    baseline_scores, baseline_model, baseline_seconds = fit_classifier(
        train1_base_rows, eval_base["features"], BASE_FEATURES
    )
    baseline = evaluate_scores(
        "previous best reranker",
        eval_base["features"],
        baseline_scores,
        users,
        eval_truth,
        eval_fill,
    )
    baseline["features"] = BASE_FEATURES
    baseline["model"] = baseline_model
    baseline["fit_seconds"] = baseline_seconds

    a_scores, a_model, a_seconds = fit_classifier(
        train1_rows, eval_rows, ALL_FEATURES
    )
    a_result = evaluate_scores(
        "A: slot affinity",
        eval_rows,
        a_scores,
        users,
        eval_truth,
        eval_fill,
    )
    a_result["features"] = ALL_FEATURES
    a_result["model"] = a_model
    a_result["fit_seconds"] = a_seconds

    b_rows = pd.concat(
        [train1_rows, train2_rows], ignore_index=True
    )
    b_scores, b_model, b_seconds = fit_classifier(
        b_rows, eval_rows, ALL_FEATURES
    )
    b_result = evaluate_scores(
        "B: A + second fold",
        eval_rows,
        b_scores,
        users,
        eval_truth,
        eval_fill,
    )
    b_result["features"] = ALL_FEATURES
    b_result["model"] = b_model
    b_result["fit_seconds"] = b_seconds

    prior_results = [baseline, a_result, b_result]
    best_prior = max(
        prior_results, key=lambda result: result["map5"]
    )
    rank_scores, rank_model, rank_seconds = fit_ranker(
        b_rows, eval_rows, ALL_FEATURES
    )
    c_rank_result = evaluate_scores(
        "C: lambdarank",
        eval_rows,
        rank_scores,
        users,
        eval_truth,
        eval_fill,
    )
    c_rank_result["features"] = ALL_FEATURES
    c_rank_result["model"] = rank_model
    c_rank_result["fit_seconds"] = rank_seconds

    normalized_prior = minmax_by_user(
        eval_rows, best_prior["scores"]
    )
    normalized_rank = minmax_by_user(eval_rows, rank_scores)
    average_scores = 0.5 * (normalized_prior + normalized_rank)
    c_avg_result = evaluate_scores(
        "C: average",
        eval_rows,
        average_scores,
        users,
        eval_truth,
        eval_fill,
    )
    c_avg_result["features"] = best_prior["features"]
    c_avg_result["model"] = None
    c_avg_result["fit_seconds"] = (
        best_prior["fit_seconds"] + rank_seconds
    )
    model_elapsed = time.perf_counter() - model_started

    new_results = [a_result, b_result, c_rank_result, c_avg_result]
    table_results = [baseline] + new_results
    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs previous best reranker"
    )
    for result in table_results:
        print(
            f"{result['name']} | {result['map5']:.5f} | "
            f"{result['map5'] - PREVIOUS_BEST:+.5f}"
        )
    print(
        f"baseline rerun delta vs saved previous best: "
        f"{baseline['map5'] - PREVIOUS_BEST:+.5f}"
    )
    print(
        f"best prior before C: {best_prior['name']} "
        f"({best_prior['map5']:.5f})"
    )
    print()

    best_new = max(
        new_results, key=lambda result: result["map5"]
    )
    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        best_new["aps"] - baseline["aps"]
    )
    print(
        f"best new {best_new['name']} vs previous best: "
        f"mean diff {mean_diff:+.5f}, 95% CI "
        f"[{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "best new configuration beats previous best with CI "
            "above zero; building final submission"
        )
        final_started = time.perf_counter()
        final_base = rerank.build_fold(
            interactions,
            occurrences,
            view_aggregates,
            FINAL_SCHEDULE,
            FINAL_TARGET,
            FINAL_HISTORY,
            users,
            "final",
        )
        final_fold = add_slots_to_fold(
            final_base, slot_data["final"], target_slots["final"]
        )
        final_rows = final_fold["features"]
        train3_rows = training_frame(eval_all)
        train4_rows = training_frame(fold4)
        all_train = pd.concat(
            [
                train1_rows,
                train2_rows,
                train3_rows,
                train4_rows,
            ],
            ignore_index=True,
        )
        final_scores = None
        if best_new["name"].startswith("A:") or best_new[
            "name"
        ].startswith("B:"):
            final_scores, _, fit_seconds = fit_classifier(
                all_train, final_rows, ALL_FEATURES
            )
            print(f"  final classifier fit: {fit_seconds:.2f}s")
        elif best_new["name"] == "C: lambdarank":
            final_scores, _, fit_seconds = fit_ranker(
                all_train, final_rows, ALL_FEATURES
            )
            print(f"  final ranker fit: {fit_seconds:.2f}s")
        else:
            final_prior_scores, _, prior_seconds = fit_classifier(
                all_train, final_rows, best_prior["features"]
            )
            final_rank_scores, _, rank_seconds_final = fit_ranker(
                all_train, final_rows, ALL_FEATURES
            )
            final_scores = 0.5 * (
                minmax_by_user(final_rows, final_prior_scores)
                + minmax_by_user(final_rows, final_rank_scores)
            )
            print(
                f"  final classifier fit: {prior_seconds:.2f}s; "
                f"final ranker fit: {rank_seconds_final:.2f}s"
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
            "no rerank2 submission: CI condition not met"
        )
    print(
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
