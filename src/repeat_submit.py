import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import (
    DATA,
    TOP_K,
    fill_with_popularity,
    load_candidate_pool,
)
from repeat import build_repeat_rankings, build_strict_popularity

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("submissions/repeat.csv")
REFERENCE = Path("submissions/mf_blend.csv")
FIT_WEEKS = (11, 30)
FILL_WEEKS = (26, 30)
THRESHOLD = 0.30
CANDIDATE_FILE = "export_arh_31-42-final.csv"


def validate_output(path, reference_path, pool):
    produced_bytes = path.read_bytes()
    reference_bytes = reference_path.read_bytes()
    produced = pd.read_csv(
        path, dtype={"user_id": "uint64", "tv_show_id": str}
    )
    reference = pd.read_csv(
        reference_path, dtype={"user_id": "uint64", "tv_show_id": str}
    )
    produced_rows = produced["tv_show_id"].str.split()
    produced_ids = [
        [int(item) for item in row] for row in produced_rows
    ]
    checks = {
        "header": produced_bytes.split(b"\n")[0]
        == reference_bytes.split(b"\n")[0]
        == b"user_id,tv_show_id",
        "row_count": len(produced) == len(reference) == 1260,
        "user_order": np.array_equal(
            produced["user_id"].values,
            reference["user_id"].values,
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
    return checks, produced


def main():
    started = time.perf_counter()
    occurrences = pd.read_parquet(DATA / "occurrences.parquet")
    reference = pd.read_csv(
        REFERENCE, dtype={"user_id": "uint64"}
    )
    users = [int(user) for user in reference["user_id"]]
    pool = sorted(load_candidate_pool(CANDIDATE_FILE))

    rankings = build_repeat_rankings(
        occurrences, FIT_WEEKS, THRESHOLD, pool, users
    )
    fill = build_strict_popularity(occurrences, FILL_WEEKS, pool)

    rows = []
    filled = 0
    for user in users:
        recommendation = rankings.get(user, [])
        if len(recommendation) < TOP_K:
            filled += 1
        recommendation = fill_with_popularity(recommendation, fill)
        rows.append(" ".join(map(str, recommendation)))

    output = pd.DataFrame(
        {"user_id": reference["user_id"].values, "tv_show_id": rows}
    )
    OUT.parent.mkdir(exist_ok=True)
    csv_text = output.to_csv(index=False, lineterminator="\n")
    OUT.write_bytes(csv_text.rstrip("\n").encode("utf-8"))

    checks, produced = validate_output(OUT, REFERENCE, pool)
    elapsed = time.perf_counter() - started
    print(
        f"config: t={THRESHOLD:.2f}, fit weeks {FIT_WEEKS[0]}-"
        f"{FIT_WEEKS[1]}, strict-popularity fill weeks "
        f"{FILL_WEEKS[0]}-{FILL_WEEKS[1]}"
    )
    print(
        f"users: {len(users):,}; candidate pool: {len(pool):,}; "
        f"users with repeat evidence: {len(rankings):,}; "
        f"filled: {filled:,} ({filled / len(users):.2%})"
    )
    print(f"strict-popularity fill: {' '.join(map(str, fill))}")
    print(f"wrote {OUT} ({len(output):,} rows)")
    print(
        "checks: "
        + ", ".join(
            f"{name}={'OK' if passed else 'FAIL'}"
            for name, passed in checks.items()
        )
    )
    print(
        f"first row: {OUT.open(encoding='utf-8').readlines()[1].strip()!r}"
    )
    print(f"runtime: {elapsed:.2f}s")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
