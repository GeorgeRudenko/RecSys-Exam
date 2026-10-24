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
SUBMISSION = Path("submissions/rerank4.csv")
BASE_FEATURES = list(rerank2.ALL_FEATURES)
D_FEATURES = [
    "programme_median_duration",
    "programme_completion_rate",
    "strict_popularity_trend",
]
E_FEATURES = [
    "user_programme_watch_rate",
    "user_programme_strict_watch_rate",
    "target_history_airing_ratio",
]
SEED = 0
STRONG_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": SEED,
    "bagging_seed": SEED,
    "feature_fraction_seed": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def load_schedule_all():
    frames = []
    for file_name in (
        rerank2.TRAIN_SCHEDULE,
        rerank2.EVAL_SCHEDULE,
        rerank2.FINAL_SCHEDULE,
    ):
        frame = pd.read_csv(
            rerank.DATA / file_name,
            usecols=[
                "channel_id",
                "start_time",
                "duration",
                "tv_show_id",
            ],
            parse_dates=["start_time"],
            dayfirst=True,
            dtype={
                "channel_id": "int32",
                "duration": "int32",
                "tv_show_id": "int64",
            },
        )
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True)
    schedule = schedule[
        (schedule["tv_show_id"] > 0) & (schedule["duration"] > 0)
    ].copy()
    schedule["iso_week"] = (
        schedule["start_time"].dt.isocalendar().week.astype("int16")
    )
    return schedule


def build_programme_data(
    occurrences,
    schedules,
    history_weeks,
    target_weeks,
):
    history_schedule = schedules[
        schedules["iso_week"].between(*history_weeks)
    ]
    target_schedule = schedules[
        schedules["iso_week"].between(*target_weeks)
    ]
    history_occ = occurrences[
        occurrences["iso_week"].between(*history_weeks)
    ]
    strict_occ = history_occ[history_occ["fraction"] > 0.8]
    history_airings = (
        history_schedule.groupby("tv_show_id").size()
    )
    target_airings = target_schedule.groupby("tv_show_id").size()
    median_duration = (
        history_schedule.groupby("tv_show_id")["duration"].median()
    )
    any_users = (
        history_occ.groupby("tv_show_id")["user_id"].nunique()
    )
    strict_users = (
        strict_occ.groupby("tv_show_id")["user_id"].nunique()
    )
    last_two_weeks = strict_occ[
        strict_occ["iso_week"] >= history_weeks[1] - 1
    ]
    strict_last_two = (
        last_two_weeks.groupby("tv_show_id")["user_id"].nunique()
    )
    completion = (
        strict_users / any_users
    ).replace([np.inf, -np.inf], 0).fillna(0)
    trend = (
        (strict_last_two + 1.0) / (strict_users + 1.0)
    ).replace([np.inf, -np.inf], 0).fillna(1.0)
    ratio = (
        target_airings / history_airings
    ).replace([np.inf, -np.inf], 0).fillna(0)
    item_features = pd.DataFrame(
        {
            "tv_show_id": sorted(
                set(history_airings.index)
                | set(target_airings.index)
                | set(any_users.index)
            )
        }
    )
    for series, column in (
        (median_duration, "programme_median_duration"),
        (completion, "programme_completion_rate"),
        (trend, "strict_popularity_trend"),
        (ratio, "target_history_airing_ratio"),
    ):
        item_features[column] = (
            item_features["tv_show_id"]
            .map(series)
            .fillna(0.0)
            .astype("float64")
        )

    qualifying = history_occ[
        history_occ["fraction"] >= 0.3
    ]
    qualifying_counts = (
        qualifying.groupby(["user_id", "tv_show_id"])
        .size()
        .rename("qualifying_airings")
    )
    strict_counts = (
        strict_occ.groupby(["user_id", "tv_show_id"])
        .size()
        .rename("strict_airings")
    )
    user_features = pd.concat(
        [qualifying_counts, strict_counts], axis=1
    ).reset_index()
    user_features["history_airings"] = (
        user_features["tv_show_id"].map(history_airings).fillna(0)
    )
    user_features["qualifying_airings"] = (
        user_features["qualifying_airings"].fillna(0)
    )
    user_features["strict_airings"] = (
        user_features["strict_airings"].fillna(0)
    )
    user_features["user_programme_watch_rate"] = np.where(
        user_features["history_airings"] > 0,
        user_features["qualifying_airings"]
        / user_features["history_airings"],
        0.0,
    )
    user_features["user_programme_strict_watch_rate"] = np.where(
        user_features["history_airings"] > 0,
        user_features["strict_airings"]
        / user_features["history_airings"],
        0.0,
    )
    user_features = user_features[
        [
            "user_id",
            "tv_show_id",
            "user_programme_watch_rate",
            "user_programme_strict_watch_rate",
        ]
    ]
    return {
        "items": item_features,
        "users": user_features,
    }


def add_programme_features(frame, data, include_d, include_e):
    frame = frame.copy()
    if include_d:
        frame = frame.merge(
            data["items"][
                ["tv_show_id"] + D_FEATURES
            ],
            on="tv_show_id",
            how="left",
        )
        for column in D_FEATURES:
            frame[column] = frame[column].fillna(0.0)
    if include_e:
        frame = frame.merge(
            data["items"][
                ["tv_show_id", "target_history_airing_ratio"]
            ],
            on="tv_show_id",
            how="left",
        )
        frame = frame.merge(
            data["users"],
            on=["user_id", "tv_show_id"],
            how="left",
        )
        for column in E_FEATURES:
            frame[column] = frame[column].fillna(0.0)
    return frame


def add_slots_and_programmes(
    fold,
    slot_data,
    target_slots,
    programme_data,
):
    fold = rerank2.add_slots_to_fold(fold, slot_data, target_slots)
    fold = dict(fold)
    fold["features"] = add_programme_features(
        fold["features"], programme_data, True, True
    )
    return fold


def build_fold_deep(
    interactions,
    occurrences,
    view_aggregates,
    schedule_file,
    target_weeks,
    history_weeks,
    users,
    use_view_key,
):
    old_repeat = rerank.REPEAT_DEPTH
    old_pop = rerank.POP_DEPTH
    old_mf = rerank.MF_DEPTH
    rerank.REPEAT_DEPTH = 100
    rerank.POP_DEPTH = 50
    rerank.MF_DEPTH = 50
    try:
        return rerank.build_fold(
            interactions,
            occurrences,
            view_aggregates,
            schedule_file,
            target_weeks,
            history_weeks,
            users,
            use_view_key,
        )
    finally:
        rerank.REPEAT_DEPTH = old_repeat
        rerank.POP_DEPTH = old_pop
        rerank.MF_DEPTH = old_mf


def fit_classifier(train_frame, eval_frame, features, strong):
    started = time.perf_counter()
    x_train = train_frame[features].to_numpy(dtype=np.float64)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    x_eval = eval_frame[features].to_numpy(dtype=np.float64)
    parameters = (
        STRONG_PARAMS
        if strong
        else {
            "n_estimators": 300,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )
    model = LGBMClassifier(**parameters)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_eval)[:, 1]
    return scores, model, time.perf_counter() - started


def fit_ranker(train_frame, eval_frame, features, strong):
    started = time.perf_counter()
    frame = train_frame.sort_values(
        ["user_id", "tv_show_id"], kind="mergesort"
    ).reset_index(drop=True)
    groups = frame.groupby("user_id", sort=False).size().to_numpy()
    x_train = frame[features].to_numpy(dtype=np.float64)
    y_train = frame["label"].to_numpy(dtype=np.int8)
    x_eval = eval_frame[features].to_numpy(dtype=np.float64)
    parameters = (
        STRONG_PARAMS
        if strong
        else {
            "n_estimators": 300,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "min_data_in_leaf": 50,
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )
    model = LGBMRanker(**parameters)
    model.fit(x_train, y_train, group=groups)
    scores = model.predict(x_eval)
    return scores, model, time.perf_counter() - started


def evaluate_config(
    name,
    frames,
    features,
    users,
    truth,
    fill,
    strong=False,
):
    train_rows = pd.concat(
        [frames["train1_rows"], frames["train2_rows"]],
        ignore_index=True,
    )
    eval_rows = frames["eval_rows"]
    clf_scores, clf_model, clf_seconds = fit_classifier(
        train_rows, eval_rows, features, strong
    )
    rank_scores, rank_model, rank_seconds = fit_ranker(
        train_rows, eval_rows, features, strong
    )
    scores = 0.5 * (
        rerank2.minmax_by_user(eval_rows, clf_scores)
        + rerank2.minmax_by_user(eval_rows, rank_scores)
    )
    result = rerank2.evaluate_scores(
        name, eval_rows, scores, users, truth, fill
    )
    result["features"] = list(features)
    result["classifier"] = clf_model
    result["ranker"] = rank_model
    result["clf_seconds"] = clf_seconds
    result["rank_seconds"] = rank_seconds
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
    started = time.perf_counter()
    interactions = rerank.load_interactions()
    occurrences = pd.read_parquet(
        rerank.DATA / "occurrences.parquet"
    )
    schedules = load_schedule_all()
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
    programme_data = {
        "train": build_programme_data(
            occurrences,
            schedules,
            rerank.TRAIN_HISTORY,
            rerank.TRAIN_TARGET,
        ),
        "train2": build_programme_data(
            occurrences,
            schedules,
            rerank2.TRAIN2_HISTORY,
            rerank2.TRAIN2_TARGET,
        ),
        "eval": build_programme_data(
            occurrences,
            schedules,
            rerank.EVAL_HISTORY,
            rerank.EVAL_TARGET,
        ),
    }
    base_folds = {
        "train1": add_slots_and_programmes(
            train1_base,
            slot_data["train"],
            target_slots["train"],
            programme_data["train"],
        ),
        "eval": add_slots_and_programmes(
            eval_base,
            slot_data["eval"],
            target_slots["eval"],
            programme_data["eval"],
        ),
        "train2": add_slots_and_programmes(
            train2_base,
            slot_data["train2"],
            target_slots["train2"],
            programme_data["train2"],
        ),
    }
    base_frames = {
        "train1_rows": rerank2.training_frame(base_folds["train1"]),
        "train2_rows": rerank2.training_frame(base_folds["train2"]),
        "eval_rows": base_folds["eval"]["features"],
        "eval_truth": eval_base["truth"],
        "eval_fill": eval_base["fill"],
    }
    fold_elapsed = time.perf_counter() - fold_started

    model_started = time.perf_counter()
    base_result = evaluate_config(
        "base C + slot features",
        base_frames,
        BASE_FEATURES,
        users,
        base_frames["eval_truth"],
        base_frames["eval_fill"],
    )
    results = [base_result]
    best = base_result
    best_frames = base_frames
    best_features = list(BASE_FEATURES)
    best_candidate_mode = "base"
    best_strong = False
    kept_d = False
    kept_e = False

    d_result = evaluate_config(
        "base + D",
        base_frames,
        BASE_FEATURES + D_FEATURES,
        users,
        base_frames["eval_truth"],
        base_frames["eval_fill"],
    )
    results.append(d_result)
    if d_result["map5"] > best["map5"]:
        best = d_result
        best_features = BASE_FEATURES + D_FEATURES
        kept_d = True

    e_result = evaluate_config(
        "base + "
        + ("D + " if kept_d else "")
        + "E",
        base_frames,
        BASE_FEATURES
        + (D_FEATURES if kept_d else [])
        + E_FEATURES,
        users,
        base_frames["eval_truth"],
        base_frames["eval_fill"],
    )
    results.append(e_result)
    if e_result["map5"] > best["map5"]:
        best = e_result
        best_features = (
            BASE_FEATURES
            + (D_FEATURES if kept_d else [])
            + E_FEATURES
        )
        kept_e = True

    deep_started = time.perf_counter()
    train1_deep = build_fold_deep(
        interactions,
        occurrences,
        view_aggregates,
        rerank2.TRAIN_SCHEDULE,
        rerank.TRAIN_TARGET,
        rerank.TRAIN_HISTORY,
        users,
        "train",
    )
    eval_deep = build_fold_deep(
        interactions,
        occurrences,
        view_aggregates,
        rerank2.EVAL_SCHEDULE,
        rerank.EVAL_TARGET,
        rerank.EVAL_HISTORY,
        users,
        "eval",
    )
    train2_deep = build_fold_deep(
        interactions,
        occurrences,
        view_aggregates,
        rerank2.TRAIN_SCHEDULE,
        rerank2.TRAIN2_TARGET,
        rerank2.TRAIN2_HISTORY,
        users,
        "train2",
    )
    deep_folds = {
        "train1": add_slots_and_programmes(
            train1_deep,
            slot_data["train"],
            target_slots["train"],
            programme_data["train"],
        ),
        "eval": add_slots_and_programmes(
            eval_deep,
            slot_data["eval"],
            target_slots["eval"],
            programme_data["eval"],
        ),
        "train2": add_slots_and_programmes(
            train2_deep,
            slot_data["train2"],
            target_slots["train2"],
            programme_data["train2"],
        ),
    }
    deep_frames = {
        "train1_rows": rerank2.training_frame(deep_folds["train1"]),
        "train2_rows": rerank2.training_frame(deep_folds["train2"]),
        "eval_rows": deep_folds["eval"]["features"],
        "eval_truth": eval_base["truth"],
        "eval_fill": eval_base["fill"],
    }
    deep_metrics = rerank.candidate_metrics(
        eval_deep["candidates"],
        eval_base["truth"],
        users,
    )
    deep_elapsed = time.perf_counter() - deep_started

    f_features = (
        BASE_FEATURES
        + (D_FEATURES if kept_d else [])
        + (E_FEATURES if kept_e else [])
    )
    f_result = evaluate_config(
        "base + "
        + ("D + " if kept_d else "")
        + ("E + " if kept_e else "")
        + "F",
        deep_frames,
        f_features,
        users,
        deep_frames["eval_truth"],
        deep_frames["eval_fill"],
    )
    results.append(f_result)
    if f_result["map5"] > best["map5"]:
        best = f_result
        best_frames = deep_frames
        best_features = f_features
        best_candidate_mode = "deep"

    g_result = evaluate_config(
        "best features/candidates + G",
        best_frames,
        best_features,
        users,
        best_frames["eval_truth"],
        best_frames["eval_fill"],
        strong=True,
    )
    results.append(g_result)
    if g_result["map5"] > best["map5"]:
        best = g_result
        best_strong = True
    model_elapsed = time.perf_counter() - model_started

    print(
        "configuration | strict MAP@5 | "
        "difference vs base"
    )
    for result in results:
        print(
            f"{result['name']} | {result['map5']:.5f} | "
            f"{result['map5'] - base_result['map5']:+.5f}"
        )
    print(
        f"deep candidates: recall "
        f"{deep_metrics['candidate_recall']:.4f} macro, "
        f"{deep_metrics['candidate_micro_recall']:.4f} micro; "
        f"oracle MAP@5 {deep_metrics['oracle_map_all']:.5f} "
        f"all / {deep_metrics['oracle_map_truth_users']:.5f} "
        f"truth users"
    )
    print(
        f"kept blocks: D={'yes' if kept_d else 'no'}, "
        f"E={'yes' if kept_e else 'no'}, "
        f"F={'yes' if best_candidate_mode == 'deep' else 'no'}, "
        f"G={'yes' if best_strong else 'no'}"
    )

    mean_diff, ci_lower, ci_upper = rerank.paired_bootstrap(
        best["aps"] - base_result["aps"]
    )
    print(
        f"best {best['name']} vs base: mean diff "
        f"{mean_diff:+.5f}, 95% CI "
        f"[{ci_lower:+.5f}, {ci_upper:+.5f}]"
    )
    print(
        f"runtimes: base folds {fold_elapsed:.2f}s; "
        f"deep folds {deep_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )

    importances = best["classifier"].feature_importances_
    order = np.argsort(-importances)[:15]
    print("best classifier importances (top 15):")
    for index in order:
        print(
            f"  {best_features[index]:<36} "
            f"{importances[index]:.5f}"
        )

    if mean_diff > 0 and ci_lower > 0:
        print(
            "best beats base with CI above zero; "
            "building final submission"
        )
        final_started = time.perf_counter()
        if best_candidate_mode == "deep":
            fold4_base = build_fold_deep(
                interactions,
                occurrences,
                view_aggregates,
                rerank2.EVAL_SCHEDULE,
                rerank2.FOLD4_TARGET,
                rerank2.FOLD4_HISTORY,
                users,
                "fold4",
            )
            final_base = build_fold_deep(
                interactions,
                occurrences,
                view_aggregates,
                rerank2.FINAL_SCHEDULE,
                rerank2.FINAL_TARGET,
                rerank2.FINAL_HISTORY,
                users,
                "final",
            )
        else:
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
        programme_data["fold4"] = build_programme_data(
            occurrences,
            schedules,
            rerank2.FOLD4_HISTORY,
            rerank2.FOLD4_TARGET,
        )
        programme_data["final"] = build_programme_data(
            occurrences,
            schedules,
            rerank2.FINAL_HISTORY,
            rerank2.FINAL_TARGET,
        )
        fold4 = add_slots_and_programmes(
            fold4_base,
            slot_data["fold4"],
            target_slots["fold4"],
            programme_data["fold4"],
        )
        final_fold = add_slots_and_programmes(
            final_base,
            slot_data["final"],
            target_slots["final"],
            programme_data["final"],
        )
        all_train = pd.concat(
            [
                best_frames["train1_rows"],
                best_frames["train2_rows"],
                rerank2.training_frame(
                    base_folds["eval"]
                    if best_candidate_mode == "base"
                    else deep_folds["eval"]
                ),
                rerank2.training_frame(fold4),
            ],
            ignore_index=True,
        )
        final_rows = final_fold["features"]
        final_clf_scores, _, final_clf_seconds = (
            fit_classifier(
                all_train,
                final_rows,
                best_features,
                best_strong,
            )
        )
        final_rank_scores, _, final_rank_seconds = (
            fit_ranker(
                all_train,
                final_rows,
                best_features,
                best_strong,
            )
        )
        final_scores = 0.5 * (
            rerank2.minmax_by_user(
                final_rows, final_clf_scores
            )
            + rerank2.minmax_by_user(
                final_rows, final_rank_scores
            )
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
            f"final time {final_elapsed:.2f}s; "
            f"classifier {final_clf_seconds:.2f}s; "
            f"ranker {final_rank_seconds:.2f}s"
        )
        if not all(checks.values()):
            raise SystemExit(1)
    else:
        print(
            "no rerank4 submission: CI condition not met"
        )
    print(
        f"total runtime: {time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
