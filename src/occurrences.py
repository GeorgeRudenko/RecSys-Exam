import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data")
SCHEDULE_FILES = (
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
)
WEEK_LO = 11
WEEK_HI = 30


def load_schedule():
    frames = []
    for name in SCHEDULE_FILES:
        frame = pd.read_csv(
            DATA / name,
            usecols=["channel_id", "start_time", "duration", "tv_show_id"],
            parse_dates=["start_time"],
            dayfirst=True,
            dtype={"channel_id": "int32", "tv_show_id": "int64", "duration": "int32"},
        )
        frames.append(frame)
    sched = pd.concat(frames, ignore_index=True)
    del frames
    sched.drop_duplicates(
        ["channel_id", "start_time", "duration", "tv_show_id"], inplace=True
    )
    sched = sched[(sched["tv_show_id"] > 0) & (sched["duration"] > 0)].copy()
    sched["iso_week"] = sched["start_time"].dt.isocalendar().week.astype("int16")
    sched["end"] = sched["start_time"] + pd.to_timedelta(
        sched["duration"], unit="s"
    )
    sched.sort_values(["channel_id", "start_time"], inplace=True, kind="mergesort")
    sched.reset_index(drop=True, inplace=True)
    return sched


def build_index(sched):
    index = {}
    for channel, group in sched.groupby("channel_id", sort=False):
        starts = group["start_time"].values
        ends = group["end"].values
        if not np.all(ends[1:] >= ends[:-1]):
            raise ValueError(f"non-monotone programme ends on channel {channel}")
        index[channel] = (starts, ends, group.index.values)
    return index


def main():
    started = time.perf_counter()
    sched = load_schedule()
    print(
        f"labelled schedule airings loaded: {len(sched):,}; "
        f"channels {sched['channel_id'].nunique():,}"
    )
    index = build_index(sched)
    sched_show = sched["tv_show_id"].values
    sched_channel = sched["channel_id"].values
    sched_start = sched["start_time"].values
    sched_week = sched["iso_week"].values
    sched_duration = sched["duration"].values.astype("float64")

    views = pd.read_csv(
        DATA / "dataset11-30.csv",
        usecols=["user_id", "vsetv_id", "start_time", "stop_time"],
        parse_dates=["start_time", "stop_time"],
        dtype={"user_id": "uint64", "vsetv_id": "int32"},
    )
    views = views[views["stop_time"] > views["start_time"]].copy()
    views["iso_week"] = views["start_time"].dt.isocalendar().week.astype("int16")
    views = views[views["iso_week"].between(WEEK_LO, WEEK_HI)].copy()
    views.reset_index(drop=True, inplace=True)
    print(f"views in weeks {WEEK_LO}-{WEEK_HI}: {len(views):,}")

    parts = []
    for week, week_views in views.groupby("iso_week", sort=True):
        week_started = time.perf_counter()
        week_parts = []
        for channel, group in week_views.groupby("vsetv_id", sort=False):
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
            keep = overlap > 0
            if not keep.any():
                continue
            view_ids = group.index.values[local_views[keep]]
            sched_ids = row_ids[local_sched[keep]]
            fraction = overlap[keep] / sched_duration[sched_ids]
            week_parts.append(
                pd.DataFrame(
                    {
                        "user_id": views["user_id"].values[view_ids],
                        "tv_show_id": sched_show[sched_ids],
                        "channel_id": sched_channel[sched_ids],
                        "airing_start": sched_start[sched_ids],
                        "iso_week": sched_week[sched_ids],
                        "fraction": fraction,
                    }
                )
            )
        if week_parts:
            week_occ = pd.concat(week_parts, ignore_index=True)
            del week_parts
            week_occ = week_occ.groupby(
                ["user_id", "tv_show_id", "channel_id", "airing_start", "iso_week"],
                as_index=False,
                sort=False,
            )["fraction"].max()
            parts.append(week_occ)
            print(
                f"week {week}: {len(week_occ):,} occurrences; "
                f"{time.perf_counter() - week_started:.1f}s"
            )
        del week_views

    occurrences = pd.concat(parts, ignore_index=True)
    del parts
    occurrences = occurrences.groupby(
        ["user_id", "tv_show_id", "channel_id", "airing_start", "iso_week"],
        as_index=False,
        sort=False,
    )["fraction"].max()
    occurrences.sort_values(
        ["user_id", "airing_start", "tv_show_id"],
        inplace=True,
        kind="mergesort",
    )
    occurrences.reset_index(drop=True, inplace=True)

    out_path = DATA / "occurrences.parquet"
    occurrences.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - started
    strict = occurrences["fraction"] > 0.8
    print(f"occurrences: {len(occurrences):,}")
    print(f"distinct users: {occurrences['user_id'].nunique():,}")
    print(f"distinct programmes: {occurrences['tv_show_id'].nunique():,}")
    print(f"fraction > 0.8: {int(strict.sum()):,} ({strict.mean():.2%})")
    print(f"runtime: {elapsed:.2f}s")
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
