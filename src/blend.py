import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data")
OUT_DIR = Path("submissions")
OUT_PATH = OUT_DIR / "blend_w02.csv"
TRAIN_WEEKS = (21, 30)
POPULARITY_WEEKS = (26, 30)
WEIGHT = 0.2
TOP_K = 5


def load_interactions():
    parquet_path = DATA / "interactions.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(DATA / "interactions.csv")


def main():
    interactions = load_interactions()
    submission = pd.read_csv(DATA / "submission.csv")
    users = [int(user) for user in submission["user_id"]]

    candidates = pd.read_csv(
        DATA / "export_arh_31-42-final.csv", usecols=["tv_show_id"]
    )
    pool = sorted(
        {
            int(item)
            for item in candidates.loc[
                candidates["tv_show_id"] != 0, "tv_show_id"
            ].unique()
        }
    )

    recent = interactions[interactions["iso_week"].between(*POPULARITY_WEEKS)]
    counts = recent.groupby("tv_show_id")["user_id"].nunique()
    pop_counts = counts.reindex(pool).fillna(0).to_numpy(dtype=np.float64)
    if pop_counts.sum() == 0:
        raise ValueError("popularity window has no candidates")
    pop_share = pop_counts / pop_counts.sum()

    history = interactions[
        interactions["iso_week"].between(*TRAIN_WEEKS)
        & interactions["tv_show_id"].isin(pool)
        & interactions["user_id"].isin(users)
    ].copy()
    history["airing_day"] = history["airing_start"].dt.normalize()
    features = (
        history.groupby(["user_id", "tv_show_id"], sort=False)["airing_day"]
        .nunique()
        .rename("distinct_days")
        .reset_index()
    )

    user_index = {user: i for i, user in enumerate(users)}
    pool_index = {item: i for i, item in enumerate(pool)}
    user_share = np.zeros((len(users), len(pool)), dtype=np.float32)
    for row in features.itertuples(index=False):
        user_share[
            user_index[int(row.user_id)], pool_index[int(row.tv_show_id)]
        ] = row.distinct_days
    row_sums = user_share.sum(axis=1, keepdims=True)
    np.divide(user_share, row_sums, out=user_share, where=row_sums > 0)

    scores = WEIGHT * user_share + (1.0 - WEIGHT) * pop_share[None, :]
    order = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K]
    recommendations = [
        [int(pool[j]) for j in order[i]] for i in range(len(users))
    ]

    rec_strings = [" ".join(map(str, row)) for row in recommendations]
    output = pd.DataFrame(
        {
            "user_id": submission["user_id"].values,
            "tv_show_id": rec_strings,
        }
    )
    OUT_DIR.mkdir(exist_ok=True)
    csv_text = output.to_csv(index=False, lineterminator="\n")
    OUT_PATH.write_bytes(csv_text.rstrip("\n").encode("utf-8"))

    original_bytes = (DATA / "submission.csv").read_bytes()
    produced_bytes = OUT_PATH.read_bytes()
    original_header = original_bytes.split(b"\n")[0].decode("utf-8")
    output_header = produced_bytes.split(b"\n")[0].decode("utf-8")
    original = pd.read_csv(DATA / "submission.csv", dtype={"tv_show_id": str})
    produced = pd.read_csv(OUT_PATH, dtype={"tv_show_id": str})

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

    history_users = set(int(user) for user in features["user_id"].unique())
    cold_users = [user for user in users if user not in history_users]
    top_pop = np.argsort(-pop_share, kind="stable")[:TOP_K]

    print(f"candidate pool: {len(pool):,} labelled programmes in 31-42")
    print(
        f"history window: ISO weeks {TRAIN_WEEKS[0]}-{TRAIN_WEEKS[1]} "
        f"({len(history):,} pool interactions, {len(history_users):,} users)"
    )
    print(
        f"popularity window: ISO weeks {POPULARITY_WEEKS[0]}-{POPULARITY_WEEKS[1]} "
        f"({len(recent):,} interactions, {recent['user_id'].nunique():,} users)"
    )
    print(
        f"users with pool history: {len(users) - len(cold_users):,} / "
        f"{len(users):,}; cold-start: {len(cold_users):,} "
        "(popularity fallback)"
    )
    print("popularity top-5:")
    for index in top_pop:
        print(f"  {pool[index]}: {int(pop_counts[index]):,} users")
    print(f"wrote {OUT_PATH} ({len(output):,} rows, weight={WEIGHT})")
    print(
        "validation: "
        + ", ".join(
            f"{key}={'OK' if value else 'FAIL'}"
            for key, value in checks.items()
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)
    print(
        f"example row: "
        f"{OUT_PATH.open(encoding='utf-8').readlines()[1].strip()!r}"
    )


if __name__ == "__main__":
    main()
