import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data")
SCHEDULE_FILES = [
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
]
MIN_OVERLAP_SECONDS = 300.0
OVERLAP_FRACTION = 0.5


def iso_summary(dt):
    iso = dt.dt.isocalendar()
    return {
        "min_date": dt.min(),
        "max_date": dt.max(),
        "week_min": int(iso.week.min()),
        "week_max": int(iso.week.max()),
        "years": sorted(int(y) for y in iso.year.unique()),
    }


def named_week_check(name, weeks):
    match = re.search(r"(\d+)-(\d+)", name)
    if not match:
        return None
    lo, hi = map(int, match.groups())
    inside = weeks.between(lo, hi)
    outside = sorted(int(w) for w in weeks[~inside].unique())
    return lo, hi, float(inside.mean()), outside


def main():
    started = time.perf_counter()

    sched_frames = []
    file_summaries = []
    for name in SCHEDULE_FILES:
        sched = pd.read_csv(
            DATA / name,
            usecols=["channel_id", "start_time", "duration", "tv_show_id"],
            parse_dates=["start_time"],
            dayfirst=True,
        )
        weeks = sched["start_time"].dt.isocalendar().week
        summary = iso_summary(sched["start_time"])
        summary["name"] = name
        summary["named"] = named_week_check(name, weeks)
        file_summaries.append(summary)
        sched["end"] = sched["start_time"] + pd.to_timedelta(
            sched["duration"], unit="s"
        )
        sched_frames.append(sched)

    sched = pd.concat(sched_frames, ignore_index=True)
    del sched_frames
    sched.drop_duplicates(
        ["channel_id", "start_time", "duration", "tv_show_id"], inplace=True
    )
    sched.sort_values(["channel_id", "start_time"], inplace=True, kind="mergesort")
    sched.reset_index(drop=True, inplace=True)

    views = pd.read_csv(
        DATA / "dataset11-30.csv",
        usecols=["user_id", "vsetv_id", "start_time", "stop_time"],
        parse_dates=["start_time", "stop_time"],
    )
    views.reset_index(drop=True, inplace=True)
    view_summary = iso_summary(views["start_time"])
    view_summary["name"] = "dataset11-30.csv"
    view_summary["named"] = named_week_check(
        "dataset11-30.csv", views["start_time"].dt.isocalendar().week
    )

    print("=== FILE -> DATE / ISO WEEK ===")
    for summary in [view_summary] + file_summaries:
        line = (
            f"{summary['name']}: dates {summary['min_date'].date()}.."
            f"{summary['max_date'].date()}; ISO weeks {summary['week_min']}.."
            f"{summary['week_max']} (years {summary['years']})"
        )
        if summary["named"] is not None:
            lo, hi, share, outside = summary["named"]
            line += f"; rows in weeks {lo}-{hi}: {share:.2%}; boundary weeks {outside}"
        print(line)
    checks = []
    for date in ["2020-03-09", "2020-05-18", "2020-07-27", "2020-10-19"]:
        checks.append(f"{date}=W{pd.Timestamp(date).isocalendar().week}")
    print("ISO-2020 check: " + ", ".join(checks) + " -> file names match")

    print("\n=== JOIN ===")
    index = {}
    for channel, group in sched.groupby("channel_id", sort=False):
        starts = group["start_time"].values
        ends = group["end"].values
        if not np.all(ends[1:] >= ends[:-1]):
            raise ValueError(f"non-monotone programme ends on channel {channel}")
        index[channel] = (starts, ends, group.index.values)

    view_parts = []
    sched_parts = []
    overlap_parts = []
    for channel, group in views.groupby("vsetv_id", sort=False):
        if channel not in index:
            continue
        starts, ends, row_ids = index[channel]
        a = group["start_time"].values
        b = group["stop_time"].values
        lo = np.searchsorted(ends, a, side="right")
        hi = np.searchsorted(starts, b, side="left")
        counts = hi - lo
        positive = counts > 0
        if not positive.any():
            continue
        c = counts[positive]
        total = int(c.sum())
        local_views = np.repeat(np.flatnonzero(positive), c)
        offsets = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
        local_sched = np.repeat(lo[positive], c) + offsets
        overlap = (
            np.minimum(b[local_views], ends[local_sched])
            - np.maximum(a[local_views], starts[local_sched])
        ).astype("timedelta64[s]").astype(np.int64)
        view_parts.append(group.index.values[local_views])
        sched_parts.append(row_ids[local_sched])
        overlap_parts.append(overlap)

    view_ids = np.concatenate(view_parts)
    sched_ids = np.concatenate(sched_parts)
    overlap_seconds = np.concatenate(overlap_parts)
    del view_parts, sched_parts, overlap_parts

    # Measured duration percentiles (seconds):
    #   view duraton: p25=133, p50=396, p75=1452
    #   programme duration: p25=600, p50=1500, p75=3000, p95=7200
    # min(300s, 0.5*duration) means 5 minutes for the 75%+ of airings lasting
    # >=10 minutes, while short airings require half their length.  This tolerates
    # zapping but rejects incidental overlaps; 300s is also ~76% of the median view.
    sched_show = sched["tv_show_id"].values
    sched_duration = sched["duration"].values
    threshold = np.minimum(
        MIN_OVERLAP_SECONDS, OVERLAP_FRACTION * sched_duration[sched_ids]
    )
    qualifies = overlap_seconds >= threshold
    labelled = qualifies & (sched_show[sched_ids] != 0)

    matched_any = np.zeros(len(views), dtype=bool)
    matched_any[view_ids[qualifies]] = True
    matched_labelled = np.zeros(len(views), dtype=bool)
    matched_labelled[view_ids[labelled]] = True
    unlabelled_only = matched_any & ~matched_labelled
    unmatched = ~matched_any

    print(f"view rows: {len(views):,}")
    print(f"qualifying view-programme pairs (incl. unlabelled): {int(qualifies.sum()):,}")
    print(
        f"views matched to labelled programme: {int(matched_labelled.sum()):,} "
        f"({matched_labelled.mean():.2%})"
    )
    print(
        f"views matched only unlabelled airtime: {int(unlabelled_only.sum()):,} "
        f"({unlabelled_only.mean():.2%})"
    )
    print(
        f"views with no qualifying overlap: {int(unmatched.sum()):,} "
        f"({unmatched.mean():.2%})"
    )

    v_lab = view_ids[labelled]
    s_lab = sched_ids[labelled]
    inter = pd.DataFrame(
        {
            "user_id": views["user_id"].values[v_lab],
            "tv_show_id": sched_show[s_lab],
            "channel_id": sched["channel_id"].values[s_lab],
            "airing_start": sched["start_time"].values[s_lab],
            "overlap_seconds": overlap_seconds[labelled],
        }
    )
    # ISO week number of the airing; all airings in this dataset are in 2020.
    inter["iso_week"] = inter["airing_start"].dt.isocalendar().week.astype("int16")
    inter.sort_values(
        ["user_id", "airing_start", "tv_show_id"], inplace=True, kind="mergesort"
    )
    inter.reset_index(drop=True, inplace=True)

    print(
        f"interactions: {len(inter):,} pairs; distinct users "
        f"{inter['user_id'].nunique():,}; distinct programmes "
        f"{inter['tv_show_id'].nunique():,}"
    )

    submission = pd.read_csv(DATA / "submission.csv", usecols=["user_id"])
    submission_users = submission["user_id"]
    interaction_users = inter["user_id"].unique()
    with_interactions = int(submission_users.isin(interaction_users).sum())
    print(
        f"submission users with >=1 labelled interaction: {with_interactions:,} / "
        f"{len(submission_users):,}; cold-start (none): "
        f"{len(submission_users) - with_interactions:,}"
    )

    out_path = DATA / "interactions.parquet"
    try:
        inter.to_parquet(out_path, index=False)
    except Exception as exc:
        out_path = DATA / "interactions.csv"
        inter.to_csv(out_path, index=False)
        print(f"parquet unavailable ({exc}); wrote CSV instead")
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
