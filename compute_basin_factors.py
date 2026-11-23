"""Run this ONCE (or re-run yearly alongside fit_statistical.py / TCN
retraining -- it almost never changes) to compute each basin's
FNF_daily_mm -> raw-volume (TAF-like) conversion factor, and cache it to
basin_factors.json so apply_tcn.py never has to re-scan the full historical
record on every forecast.

Needs two aligned per-basin time series (same Date index):
  - history_dir/<basin>.csv   -- the model's own training feature file,
    with an FNF_daily_mm column (this is the TCN's lookback-history feed too)
  - combined_dir/<basin>_monthly.csv -- the observed monthly record, with a
    raw-volume FNF column, used only to derive the conversion factor

Usage:
    python compute_basin_factors.py --history-dir .../converted/monthly \
        --combined-dir .../out.zipped/combined --basins AMF,CSN,... \
        --out basin_factors.json
"""
import argparse
import json

import pandas as pd


def compute_basin_factors(basin_ids, history_dir, combined_dir):
    factors = {}
    for b in basin_ids:
        t = pd.read_csv(f"{history_dir}/{b}.csv", parse_dates=["Date"])
        c = pd.read_csv(f"{combined_dir}/{b}_monthly.csv", parse_dates=["Date"])
        m = t.merge(c, on="Date", suffixes=("_t", "_c"))
        days = m["Date"].dt.days_in_month
        mask = m["FNF_daily_mm"] > 0.05
        ratio = (m["FNF"] / (m["FNF_daily_mm"] * days))[mask]
        factors[b] = float(ratio.median())
    return factors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True)
    ap.add_argument("--combined-dir", required=True)
    ap.add_argument("--basins", required=True, help="comma-separated basin ids")
    ap.add_argument("--out", default="basin_factors.json")
    args = ap.parse_args()

    basin_ids = args.basins.split(",")
    factors = compute_basin_factors(basin_ids, args.history_dir, args.combined_dir)
    with open(args.out, "w") as f:
        json.dump(factors, f, indent=2)
    print(f"wrote {args.out}: {len(factors)} basins")


if __name__ == "__main__":
    main()
