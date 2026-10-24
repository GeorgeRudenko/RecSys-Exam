import sys
import time
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
SUBMISSION = Path("submissions/rerank7.csv")

BASE_SPECS = [
    ("train1", TRAIN_SCHEDULE, (16, 20), (11, 15)),
    ("train2", TRAIN_SCHEDULE, (18, 20), (11, 17)),
]
EVAL_SPEC = ("eval", EVAL_SCHEDULE, (21, 30), (11, 20))
FOLD4_SPEC = ("fold4", EVAL_SCHEDULE, (26, 30), (11, 25))
FINAL_SPEC = ("final", FINAL_SCHEDULE, (31, 42), (11, 30))

TARGET_COUNTS = [
    "schedule_airings",
    "slot_score_sum",
    "slot_top10_airings",
]
HISTORY_COUNTS = [
    "repeat_count_ge_030",
    "repeat_count_ge_050",
    "repeat_count_gt_080",
    "distinct_weeks_watched",
    "user_view_hours",
    "user_distinct_programmes",
]
NORMALIZED_MAP = {
    column: column + "_per_week"
    for column in TARGET_COUNTS + HISTORY_COUNTS
}
N_FEATURES = [
    NORMALIZED_MAP.get(column, column)
    for column in rerank2.ALL_FEATURES
]
T_FEATURES = [
    "time_hour_share_mean",
    "time_hour_share_max",
    "time_weekend_affinity",
]
SEED = 0
MAX_VIEW_SECONDS = 3 * 3600


def build_time_profiles(raw_views, users):
    user_index = {int(user): index for index, user in enumerate(users)}
    weeks = list(range(11, 31))
    week_index = {week: index for index, week in enumerate(weeks)}
    n_users = len(users)
    hour_counts = np.zeros(
        (len(weeks), n_users, 24), dtype=np.float64
    )
    weekend_counts = np.zeros(
        (len(weeks), n_users), dtype=np.float64
    )
    chunk_size = 500_000
    for start_row in range(0, len(raw_views), chunk_size):
        chunk = raw_views.iloc[start_row : start_row + chunk_size]
        chunk = chunk[
            chunk["user_id"].isin(user_index)
            & chunk["iso_week"].between(11, 30)
        ]
        if chunk.empty:
            continue
        user_values = chunk["user_id"].map(user_index).to_numpy()
        week_values = (
            chunk["iso_week"].to_numpy(dtype=np.int64) - 11
        )
        seconds = np.minimum(
            chunk["duraton"].to_numpy(dtype=np.float64),
            MAX_VIEW_SECONDS,
        )
        seconds = np.clip(seconds, 0.0, None)
        epoch = (
            chunk["start_time"].astype("int64").to_numpy()
            // 1_000_000_000
        )
        hour0 = ((epoch // 3600) % 24).astype(np.int64)
        offset = epoch % 3600
        first_length = 3600 - offset
        lengths = [
            np.minimum(seconds, first_length),
            np.clip(seconds - first_length, 0, 3600),
            np.clip(seconds - first_length - 3600, 0, 3600),
            np.clip(seconds - first_length - 7200, 0, 3600),
        ]
        offsets = [
            np.zeros_like(epoch),
            first_length,
            first_length + 3600,
            first_length + 7200,
        ]
        for segment in range(4):
            length = lengths[segment]
            active = length > 0
            if not active.any():
                continue
            hour = (hour0[active] + segment) % 24
            np.add.at(
                hour_counts,
                (
                    week_values[active],
                    user_values[active],
                    hour,
                ),
                length[active],
            )
            day = (
                epoch[active] + offsets[segment][active]
            ) // 86400
            weekend = ((day + 3) % 7) >= 5
            np.add.at(
                weekend_counts,
                (week_values[active], user_values[active]),
                length[active] * weekend,
            )
    return hour_counts, weekend_counts, week_index


def profile_for_window(hour_counts, weekend_counts, weeks):
    start = weeks[0] - 11
    end = weeks[1] - 11 + 1
    counts = hour_counts[start:end].sum(axis=0)
    weekend = weekend_counts[start:end].sum(axis=0)
    totals = counts.sum(axis=1)
    shares = np.zeros_like(counts)
    positive = totals > 0
    shares[positive] = counts[positive] / totals[positive, None]
    weekend_shares = np.zeros_like(totals)
    weekend_shares[positive] = weekend[positive] / totals[positive]
    return shares, weekend_shares


def load_item_airings(schedule_file, target_weeks):
    schedule = pd.read_csv(
        rerank.DATA / schedule_file,
        usecols=["start_time", "tv_show_id"],
        parse_dates=["start_time"],
        dayfirst=True,
        dtype={"tv_show_id": "int64"},
    )
    week = schedule["start_time"].dt.isocalendar().week
    labelled = schedule[
        (schedule["tv_show_id"] > 0)
        & week.between(*target_weeks)
    ].copy()
    labelled["hour"] = labelled["start_time"].dt.hour.astype(
        np.int16
    )
    labelled["weekend"] = (
        labelled["start_time"].dt.dayofweek >= 5
    ).astype(np.int8)
    return {
        int(item): (
            group["hour"].to_numpy(dtype=np.int64),
            group["weekend"].to_numpy(dtype=np.float64),
        )
        for item, group in labelled.groupby(
            "tv_show_id", sort=False
        )
    }


def add_time_features(
    frame,
    users,
    hour_shares,
    weekend_shares,
    item_airings,
):
    user_index = {int(user): index for index, user in enumerate(users)}
    means = np.zeros(len(frame), dtype=np.float64)
    maxes = np.zeros(len(frame), dtype=np.float64)
    affinities = np.zeros(len(frame), dtype=np.float64)
    for row, (user, item) in enumerate(
        zip(frame["user_id"], frame["tv_show_id"])
    ):
        user = int(user)
        item = int(item)
        airings = item_airings.get(item)
        if airings is None:
            continue
        hours, weekend = airings
        values = hour_shares[user_index[user], hours]
        means[row] = float(values.mean())
        maxes[row] = float(values.max())
        affinities[row] = (
            float(weekend.mean()) * weekend_shares[user_index[user]]
        )
    frame = frame.copy()
    frame["time_hour_share_mean"] = means
    frame["time_hour_share_max"] = maxes
    frame["time_weekend_affinity"] = affinities
    return frame


def add_normalization(frame, target_weeks, history_weeks):
    frame = frame.copy()
    target_length = target_weeks[1] - target_weeks[0] + 1
    history_length = history_weeks[1] - history_weeks[0] + 1
    for column in TARGET_COUNTS:
        frame[NORMALIZED_MAP[column]] = (
            frame[column] / target_length
        )
    for column in HISTORY_COUNTS:
        frame[NORMALIZED_MAP[column]] = (
            frame[column] / history_length
        )
    return frame


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


def print_airing_shift(name, frame):
    raw = frame["schedule_airings"]
    normalized = frame["schedule_airings_per_week"]
    print(
        f"{name}: raw min/median/max "
        f"{raw.min():.5f}/{raw.median():.5f}/{raw.max():.5f}; "
        "normalized min/median/max "
        f"{normalized.min():.5f}/{normalized.median():.5f}/"
        f"{normalized.max():.5f}"
    )


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
    time_profiles = {}
    item_airings = {}

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

    def get_profile(weeks):
        key = tuple(weeks)
        if key not in time_profiles:
            if not time_profiles:
                time_profiles["_all"] = build_time_profiles(
                    raw_views, users
                )
            hour_counts, weekend_counts, _ = time_profiles["_all"]
            time_profiles[key] = profile_for_window(
                hour_counts, weekend_counts, key
            )
        return time_profiles[key]

    def get_airings(schedule, target):
        key = (schedule, tuple(target))
        if key not in item_airings:
            item_airings[key] = load_item_airings(
                schedule, target
            )
        return item_airings[key]

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
        base = rerank2.add_slots_to_fold(
            base,
            get_slot_data(history),
            get_target_slots(schedule, target),
        )
        features = add_normalization(
            base["features"], target, history
        )
        hour_shares, weekend_shares = get_profile(history)
        features = add_time_features(
            features,
            users,
            hour_shares,
            weekend_shares,
            get_airings(schedule, target),
        )
        updated = dict(base)
        updated["features"] = features
        return updated

    fold_started = time.perf_counter()
    folds = {
        spec[0]: build_fold(spec)
        for spec in BASE_SPECS + [EVAL_SPEC, FOLD4_SPEC, FINAL_SPEC]
    }
    fold_elapsed = time.perf_counter() - fold_started

    train1 = folds["train1"]
    train2 = folds["train2"]
    eval_fold = folds["eval"]
    fold4 = folds["fold4"]
    final_fold = folds["final"]
    train1_rows = rerank2.training_frame(train1)
    train2_rows = rerank2.training_frame(train2)
    base_rows = pd.concat(
        [train1_rows, train2_rows], ignore_index=True
    )
    eval_rows = eval_fold["features"]
    eval_truth = eval_fold["truth"]
    eval_fill = eval_fold["fill"]

    model_started = time.perf_counter()

    def evaluate(name, train_rows, features):
        classifier_scores, classifier, classifier_seconds = (
            rerank2.fit_classifier(
                train_rows, eval_rows, features
            )
        )
        ranker_scores, ranker, ranker_seconds = (
            rerank2.fit_ranker(train_rows, eval_rows, features)
        )
        scores = 0.5 * (
            rerank2.minmax_by_user(eval_rows, classifier_scores)
            + rerank2.minmax_by_user(eval_rows, ranker_scores)
        )
        result = rerank2.evaluate_scores(
            name,
            eval_rows,
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

    base_result = evaluate(
        "base C", base_rows, rerank2.ALL_FEATURES
    )
    n_result = evaluate("base + N", base_rows, N_FEATURES)
    t_result = evaluate(
        "base + T",
        base_rows,
        rerank2.ALL_FEATURES + T_FEATURES,
    )
    nt_result = evaluate(
        "base + N + T",
        base_rows,
        N_FEATURES + T_FEATURES,
    )
    model_elapsed = time.perf_counter() - model_started

    print(
        "configuration | strict MAP@5 (21-30) | "
        "difference vs base"
    )
    results = [base_result, n_result, t_result, nt_result]
    for result in results:
        print(
            f"{result['name']} | {result['map5']:.5f} | "
            f"{result['map5'] - base_result['map5']:+.5f}"
        )
    print("schedule_airings shift:")
    for name in ("train1", "train2", "eval", "fold4", "final"):
        print_airing_shift(name, folds[name]["features"])
    print(
        f"runtimes: folds {fold_elapsed:.2f}s; "
        f"models {model_elapsed:.2f}s"
    )

    best = max(results, key=lambda result: result["map5"])
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
        if best["name"] == "base + N":
            final_features = N_FEATURES
        elif best["name"] == "base + T":
            final_features = list(
                rerank2.ALL_FEATURES + T_FEATURES
            )
        elif best["name"] == "base + N + T":
            final_features = N_FEATURES + T_FEATURES
        else:
            final_features = list(rerank2.ALL_FEATURES)
        all_train = pd.concat(
            [
                train1_rows,
                train2_rows,
                rerank2.training_frame(eval_fold),
                rerank2.training_frame(fold4),
            ],
            ignore_index=True,
        )
        final_rows = final_fold["features"]
        classifier_scores, _, classifier_seconds = (
            rerank2.fit_classifier(
                all_train, final_rows, final_features
            )
        )
        ranker_scores, _, ranker_seconds = (
            rerank2.fit_ranker(
                all_train, final_rows, final_features
            )
        )
        final_scores = 0.5 * (
            rerank2.minmax_by_user(
                final_rows, classifier_scores
            )
            + rerank2.minmax_by_user(
                final_rows, ranker_scores
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
            f"final training rows: {len(all_train):,}; "
            f"classifier fit {classifier_seconds:.2f}s; "
            f"ranker fit {ranker_seconds:.2f}s"
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
        print("no rerank7 submission: CI condition not met")
    print(
        f"total runtime: {time.perf_counter() - total_started:.2f}s"
    )


if __name__ == "__main__":
    main()
