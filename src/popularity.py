import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("data")
OUT_DIR = Path("submissions")
OUT_PATH = OUT_DIR / "popularity.csv"
TRAIN_WEEKS = (21, 30)
TOP_K = 5


def load_interactions():
    parquet_path = DATA / "interactions.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(DATA / "interactions.csv")


def main():
    interactions = load_interactions()
    submission = pd.read_csv(DATA / "submission.csv")

    candidates = pd.read_csv(
        DATA / "export_arh_31-42-final.csv", usecols=["tv_show_id"]
    )
    pool = np.sort(
        candidates.loc[candidates["tv_show_id"] != 0, "tv_show_id"].unique()
    )

    recent = interactions[
        interactions["iso_week"].between(TRAIN_WEEKS[0], TRAIN_WEEKS[1])
    ]
    counts = recent.groupby("tv_show_id")["user_id"].nunique().rename("n_users")

    ranking = pd.DataFrame({"tv_show_id": pool})
    ranking = ranking.merge(
        counts, left_on="tv_show_id", right_index=True, how="left"
    )
    ranking["n_users"] = ranking["n_users"].fillna(0).astype("int64")
    ranking.sort_values(
        ["n_users", "tv_show_id"], ascending=[False, True], inplace=True
    )
    ranking.reset_index(drop=True, inplace=True)

    top = ranking.head(TOP_K)
    top_ids = [int(value) for value in top["tv_show_id"]]
    top_string = " ".join(map(str, top_ids))
    if len(top_ids) != TOP_K:
        raise ValueError(f"candidate pool has only {len(top_ids)} programmes")

    OUT_DIR.mkdir(exist_ok=True)
    output = pd.DataFrame(
        {
            "user_id": submission["user_id"].values,
            "tv_show_id": top_string,
        }
    )
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
        "same_top5_all_rows": produced["tv_show_id"].nunique() == 1,
        "candidates_only": set(top_ids).issubset(set(int(x) for x in pool)),
        "line_endings": b"\r\n" not in produced_bytes
        and produced_bytes.count(b"\n") == original_bytes.count(b"\n"),
        "no_trailing_newline": not produced_bytes.endswith(b"\n")
        and not original_bytes.endswith(b"\n"),
    }

    print(f"candidate pool: {len(pool):,} labelled programmes in 31-42")
    print(f"training window: ISO weeks {TRAIN_WEEKS[0]}-{TRAIN_WEEKS[1]} "
          f"({len(recent):,} interactions, {recent['user_id'].nunique():,} users)")
    print("top-5 by distinct users:")
    for _, row in top.iterrows():
        print(f"  {int(row['tv_show_id'])}: {int(row['n_users']):,} users")
    print(f"wrote {OUT_PATH} ({len(output):,} rows)")
    print("validation: " + ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()))
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"example row: {OUT_PATH.open(encoding='utf-8').readlines()[1].strip()!r}")


if __name__ == "__main__":
    main()
