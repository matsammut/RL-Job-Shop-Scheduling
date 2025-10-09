import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Best-known makespan for Taillard PFSP instance ta072
BKS_TA072 = 5349  # sources: MDPI 2018 table; Sci. Reports 2025 table

POSSIBLE_MAKESPAN_COL_PATTERNS = [
    r"(^|/)eval[_/ -]?makespan$",
    r"(^|/)test[_/ -]?makespan$",
    r"(^|/)val[_/ -]?makespan$",
    r"(^|/)makespan$",
    r"(^|/)metrics[_/ -]?makespan$",
    r"makespan",  # fallback: any column with 'makespan'
]

def find_makespan_columns(columns):
    cols = []
    for c in columns:
        for pat in POSSIBLE_MAKESPAN_COL_PATTERNS:
            if re.search(pat, c, flags=re.IGNORECASE):
                cols.append(c)
                break
    # de-dup preserving order
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    return cols

def best_makespan_from_csv(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    # Try common W&B progress.csv column names
    makespan_cols = find_makespan_columns(df.columns)
    if not makespan_cols:
        raise RuntimeError(
            "Could not find a makespan column in progress.csv.\n"
            f"Available columns include:\n{list(df.columns)[:30]} ...\n"
            "Tip: ensure your logs contain a column with 'makespan' in its name "
            "(e.g., 'eval/makespan')."
        )

    # Keep numeric cols only (coerce errors to NaN)
    best = None
    for col in makespan_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            candidate = s.min()  # lower makespan is better
            best = candidate if best is None else min(best, candidate)

    if best is None:
        raise RuntimeError(
            "Found makespan-like columns, but none contained numeric values."
        )
    return float(best)

def main():
    ap = argparse.ArgumentParser(description="Compute makespan and opportunity gap for Taillard ta072.")
    ap.add_argument("progress_csv", type=Path, help="Path to W&B progress.csv (from training on ta072)")
    args = ap.parse_args()

    if not args.progress_csv.exists():
        print(f"File not found: {args.progress_csv}", file=sys.stderr)
        sys.exit(1)

    best_makespan = best_makespan_from_csv(args.progress_csv)
    gap_abs = best_makespan - BKS_TA072
    gap_pct = (gap_abs / BKS_TA072) * 100.0

    print("=== ta072 Results ===")
    print(f"Best-known makespan (BKS): {BKS_TA072}")
    print(f"Your best logged makespan: {best_makespan:.0f}")
    print(f"Opportunity gap (absolute): {gap_abs:.0f}")
    print(f"Opportunity gap (% over BKS): {gap_pct:.3f}%")

if __name__ == "__main__":
    main()
