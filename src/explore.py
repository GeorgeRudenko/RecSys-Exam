import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data")
PCTS = [0, 1, 5, 25, 50, 75, 95, 99, 100]
PCT_LABEL = "/".join(f"p{p}" for p in PCTS)
SCHEDULE_FILES = [
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
]


def pct(values):
    return [int(round(v)) for v in np.percentile(np.asarray(values), PCTS)]


def report(df, name):
    print(f"\n{name}: rows={len(df):,}  mem={df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("  columns: " + ", ".join(map(str, df.columns)))
    print("  dtypes: " + ", ".join(f"{c}:{t}" for c, t in df.dtypes.items()))
    print("  sample:")
    print(df.head(3).to_string(index=False, max_colwidth=24))


def join_views(views, index):
    view_parts = []
    sched_parts = []
    for channel, group in views.groupby("vsetv_id", sort=False):
        if channel not in index:
            continue
        starts, ends, _ = index[channel]
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
        positions = np.repeat(np.flatnonzero(positive), c)
        offsets = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
        view_parts.append(group.index.values[positions])
        sched_parts.append(np.repeat(lo[positive], c) + offsets)
    if not view_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(view_parts), np.concatenate(sched_parts)


print("=== 1. FILES ===")

views_path = DATA / "dataset11-30.csv"
views = pd.read_csv(views_path, low_memory=False)
report(views, views_path.name)
views["start_time"] = pd.to_datetime(views["start_time"], format="%Y-%m-%d %H:%M:%S")
views["stop_time"] = pd.to_datetime(views["stop_time"], format="%Y-%m-%d %H:%M:%S")

schedules = []
sched_stats = []
sched_channels = set()
for name in SCHEDULE_FILES:
    sched = pd.read_csv(DATA / name, low_memory=False)
    report(sched, name)
    sched_stats.append(
        (name, (sched["tv_show_id"] != 0).mean(), pct(sched["duration"]), len(sched))
    )
    sched["start_time"] = pd.to_datetime(sched["start_time"], format="%d.%m.%Y %H:%M:%S")
    sched["end"] = sched["start_time"] + pd.to_timedelta(sched["duration"], unit="s")
    sched_channels.update(sched["channel_id"].unique())
    schedules.append(
        sched[["channel_id", "start_time", "end", "duration", "tv_show_id"]].copy()
    )
    del sched

sub_path = DATA / "submission.csv"
sub = pd.read_csv(sub_path, low_memory=False)
report(sub, sub_path.name)

sched_all = pd.concat(schedules, ignore_index=True).drop_duplicates(
    ["channel_id", "start_time", "tv_show_id"]
)

print("\n=== 2. JOIN KEYS / TIME ===")
view_channels = set(views["vsetv_id"].unique())
overlap = view_channels & sched_channels
print(
    f"view vsetv_id: {len(view_channels)} unique ({views.vsetv_id.min()}..{views.vsetv_id.max()}); "
    f"schedule channel_id: {len(sched_channels)} unique "
    f"({sched_all.channel_id.min()}..{sched_all.channel_id.max()}); overlap {len(overlap)}"
)
print(
    f"view-only ids: {sorted(int(x) for x in view_channels - sched_channels)}; "
    f"schedule-only: {len(sched_channels - view_channels)}"
)
print(f"view rows on schedule channels: {views.vsetv_id.isin(sched_channels).mean():.2%}")
print("formats: view '%Y-%m-%d %H:%M:%S'; schedule '%d.%m.%Y %H:%M:%S' (both naive, no tz)")
print(
    f"ranges: view {views.start_time.min()} .. {views.start_time.max()}; "
    f"schedule {sched_all.start_time.min()} .. {sched_all.start_time.max()}"
)
view_dst = (
    (views.start_time >= "2020-03-29 03:00") & (views.start_time < "2020-03-29 04:00")
).sum()
sched_dst = (
    (sched_all.start_time >= "2020-03-29 03:00")
    & (sched_all.start_time < "2020-03-29 04:00")
).sum()
print(
    f"DST check 2020-03-29 03:xx starts: view={view_dst}, schedule={sched_dst} "
    "-> both local Kyiv time (EET/EEST), comparable"
)

print("\n=== 3. SUBMISSION ===")
lengths = sub["tv_show_id"].astype(str).str.split().str.len()
length_counts = {int(k): int(v) for k, v in lengths.value_counts().items()}
in_views = sub["user_id"].isin(views["user_id"]).sum()
with sub_path.open(encoding="utf-8") as handle:
    header = handle.readline().strip()
    first_row = handle.readline().strip()
all_zero = sub["tv_show_id"].astype(str).str.fullmatch(r"0 0 0 0 0").all()
print(
    f"rows={len(sub):,}; unique users={sub.user_id.nunique():,}; "
    f"users in view log={in_views:,} ({in_views / len(sub):.1%})"
)
print(
    f"format: header='{header}'; 5 space-separated tv_show_ids per row; "
    f"lengths={length_counts}; all-zero baseline={all_zero}"
)
print(f"example row: {first_row!r}")

print("\n=== 4. SHOW COVERAGE / DURATIONS ===")
for name, share, duration_pct, n in sched_stats:
    print(
        f"{name}: tv_show_id!=0 {share:.2%} (n={n:,}); "
        f"duration pct (s) {PCT_LABEL}={duration_pct}"
    )
print(f"all schedule duration pct (s) {PCT_LABEL}: {pct(sched_all['duration'])}")
print(f"view duraton pct (s) {PCT_LABEL}: {pct(views['duraton'])}")
same = (views["duraton"] == (views["stop_time"] - views["start_time"]).dt.total_seconds())
print(f"view duraton == stop_time-start_time: {same.mean():.4%} (typo in column name; unit=seconds)")

print("\n=== 5. JOIN BENCHMARK ===")
print(
    "method: per-channel sort by start; for each view "
    "lo=searchsorted(ends,start,'right'), hi=searchsorted(starts,stop,'left'); "
    "overlap pairs are [lo,hi)"
)
view_counts = views["vsetv_id"].value_counts()
sched_counts = sched_all["channel_id"].value_counts()
common = view_counts.index.intersection(sched_counts.index)
naive_pairs = (view_counts[common] * sched_counts[common]).sum()
print(f"naive same-channel cross product: {naive_pairs:.2e} candidate pairs")

t0 = time.perf_counter()
sched_all.sort_values(["channel_id", "start_time"], inplace=True, kind="mergesort")
index = {
    channel: (group["start_time"].values, group["end"].values, group["tv_show_id"].values)
    for channel, group in sched_all.groupby("channel_id", sort=False)
}
index_time = time.perf_counter() - t0
index_bytes = sum(array.nbytes for arrays in index.values() for array in arrays)
print(
    f"index: {len(sched_all):,} rows / {len(index)} channels; "
    f"build {index_time:.2f}s after load; arrays {index_bytes / 1e6:.1f} MB"
)

rng = np.random.default_rng(0)
sample_size = max(1, len(views) // 100)
sample = views.iloc[rng.choice(len(views), sample_size, replace=False)].reset_index(drop=True)

tracemalloc.start()
t0 = time.perf_counter()
view_ids, sched_ids = join_views(sample, index)
join_time = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

pairs = len(view_ids)
matched = len(np.unique(view_ids))
scale = len(views) / len(sample)
print(
    f"1% sample: {len(sample):,} views; join {join_time:.3f}s (kernel, no CSV I/O); "
    f"peak {peak / 1e6:.2f} MB; pairs {pairs:,}; "
    f"matched views {matched:,} ({matched / len(sample):.1%})"
)
print(
    f"extrapolated full: join ~{join_time * scale:.1f}s; pairs ~{pairs * scale / 1e6:.2f}M; "
    f"output arrays ~{pairs * scale * 16 / 1e6:.0f} MB (2x int64); "
    f"linear peak bound ~{peak * scale / 1e6:.0f} MB"
)
