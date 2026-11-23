"""Run this EVERY TIME a new raw ESP ensemble forecast is issued (same
cadence as apply_statistical.py). Loads the trained TCN checkpoint for the
current issuance month and produces a per-member Apr-Jul total for every
basin -- each ensemble member's own forecasted drivers feed the decoder, so
the corrected totals genuinely vary across members.

The model only needs retraining yearly (train_encdec_tcn.py, in
lstm.new.dev.jul.2026/scr/ -- not duplicated here since it's a slow training
job, not a daily operation); this script only ever loads the already-trained
checkpoint and runs inference.

Needs:
  - runs_dir/tcn_<month-name>/{config.json, norm_stats.npz, model.pt} --
    the trained checkpoint for this issuance month (jan/feb/mar/apr/may)
  - history_dir/<basin>.csv -- observed feature history up to (not
    including) the issuance date, used as the lookback window (the model's
    own training-data convention: SWE/SMTOT/T2D/SWDOWN/LWDOWN/Q2D/
    PREC_daily_mm/Qsim_daily_mm/FNF_daily_mm)
  - archive_dir/<member>/<basin>_monthly.csv -- this issuance's live raw
    ensemble (same files apply_statistical.py's load_live_ensemble reads),
    each member's own forecasted RAW SWE/SMTOT/T2D/SWDOWN/LWDOWN/Q2D/PREC/Qsim
  - basin_factors.json -- from compute_basin_factors.py
  - for a May-1 issuance only: combined_dir/<basin>_monthly.csv, to splice
    in April's already-elapsed real nowcast Qsim (the April month has
    already happened by May 1, so it isn't part of any member's "forecast")

Usage:
    python apply_tcn.py \
        --runs-dir .../lstm.new.dev.jul.2026/runs \
        --history-dir .../converted/monthly \
        --archive-dir /path/to/init20270401_update20270401/basins/averaged \
        --combined-dir .../out.zipped/combined \
        --basin-factors basin_factors.json \
        --year 2027 --month 4 \
        --out forecast_2027-04_tcn
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tcn_features import FEATURE_COLS, raw_features, inverse_transform_target
from tcn_model import EncDecTCNForecaster

ANCHOR_RUN = {1: "tcn_jan", 2: "tcn_feb", 3: "tcn_mar", 4: "tcn_apr", 5: "tcn_may"}
LOOKBACK = 30


def load_run(runs_dir, anchor):
    run_dir = Path(runs_dir) / ANCHOR_RUN[anchor]
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    stats_npz = np.load(run_dir / "norm_stats.npz")
    stats = {k: stats_npz[k] for k in stats_npz.files}
    model = EncDecTCNForecaster(
        n_features=len(FEATURE_COLS), n_basins=len(config["basin_ids"]),
        embed_dim=config["embed_dim"], hidden_size=config["hidden_size"],
        n_levels=config["n_levels"], kernel_size=config["kernel_size"], dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    return config, stats, model


def build_lookback_feats(history_dir, basin, anchor_date, prec_mean, stats):
    train_df = pd.read_csv(Path(history_dir) / f"{basin}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    idx = train_df.index.get_indexer([anchor_date])[0]
    if idx < LOOKBACK:
        return None
    window = train_df.iloc[idx - LOOKBACK:idx]
    raw = raw_features(window, prec_mean)
    return (raw - stats["feat_mean"]) / stats["feat_std"]


def build_horizon_feats_members(archive_dir, basin, year, horizon, prec_mean, stats, basin_factor):
    """Returns ([n_members, horizon, F] z-scored features, [n_members,
    horizon] PREC_daily_mm denominators) from each live ensemble member's
    own forecast file, or (None, None) if unavailable."""
    archive_dir = Path(archive_dir)
    member_dirs = sorted(d.name for d in archive_dir.iterdir() if d.is_dir())
    feats_list, denom_list = [], []
    for m in member_dirs:
        f = archive_dir / m / f"{basin}_monthly.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        if len(df) != horizon:
            continue
        days = df["Date"].dt.days_in_month.values
        df["PREC_daily_mm"] = df["PREC"].values / days
        df["Qsim_daily_mm"] = df["Qsim"].values / (days * basin_factor)
        raw = raw_features(df, prec_mean)
        feats_list.append((raw - stats["feat_mean"]) / stats["feat_std"])
        denom_list.append(df["PREC_daily_mm"].where(df["PREC_daily_mm"] > 0, prec_mean).values)
    if not feats_list:
        return None, None
    return np.stack(feats_list), np.stack(denom_list)


def april_nowcast_addend(combined_dir, basin, year):
    f = Path(combined_dir) / f"{basin}_monthly.csv"
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
    ts = pd.Timestamp(f"{year}-04-01")
    if ts not in df.index:
        return None
    return float(df.loc[ts, "Qsim"])


def run_tcn(runs_dir, history_dir, archive_dir, basin_factors, year, month, combined_dir=None):
    config, stats, model = load_run(runs_dir, month)
    prec_means = config["prec_means"]
    horizon = config["horizon"]
    basin_ids = config["basin_ids"]

    rows = []
    for basin in basin_ids:
        if basin not in basin_factors:
            print(f"no basin factor for {basin}, skipping")
            continue
        anchor_date = pd.Timestamp(f"{year}-{month:02d}-01")
        X_lb = build_lookback_feats(history_dir, basin, anchor_date, prec_means[basin], stats)
        if X_lb is None:
            print(f"{basin}: not enough lookback history, skipping")
            continue
        Xh, denom = build_horizon_feats_members(archive_dir, basin, year, horizon, prec_means[basin], stats, basin_factors[basin])
        if Xh is None:
            print(f"{basin}: no live ensemble files found, skipping")
            continue
        n_mem = Xh.shape[0]

        with torch.no_grad():
            Xlb_t = torch.tensor(np.repeat(X_lb[None], n_mem, axis=0), dtype=torch.float32)
            Xh_t = torch.tensor(Xh, dtype=torch.float32)
            b_t = torch.full((n_mem,), basin_ids.index(basin), dtype=torch.long)
            pred_norm = model(Xlb_t, Xh_t, b_t).numpy()

        pred_daily_mm = inverse_transform_target(pred_norm, stats, denom)
        month_dates = [anchor_date + pd.DateOffset(months=i) for i in range(horizon)]
        days_in_month = np.array([d.days_in_month for d in month_dates])
        pred_volume = pred_daily_mm * days_in_month[None, :] * basin_factors[basin]  # [n_mem, horizon]

        if month <= 4:
            apr_jul = pred_volume[:, -4:].sum(axis=1)
        else:
            if combined_dir is None:
                print(f"{basin}: month 5 issuance needs --combined-dir for the April nowcast addend, skipping")
                continue
            apr_addend = april_nowcast_addend(combined_dir, basin, year)
            if apr_addend is None:
                print(f"{basin}: no April nowcast value found, skipping")
                continue
            apr_jul = pred_volume.sum(axis=1) + apr_addend

        rows.append({"basin": basin, "month": month, "year": year, "n_members": n_mem, "tcn_members": apr_jul.tolist()})

    result = pd.DataFrame(rows)
    if len(result):
        result["p10"] = result["tcn_members"].apply(lambda m: np.percentile(m, 10))
        result["p50"] = result["tcn_members"].apply(lambda m: np.percentile(m, 50))
        result["p90"] = result["tcn_members"].apply(lambda m: np.percentile(m, 90))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--history-dir", required=True)
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--combined-dir", default=None, help="required only for a May-1 (month=5) issuance")
    ap.add_argument("--basin-factors", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--out", required=True, help="output path stem; writes <out>.pkl and <out>.csv")
    args = ap.parse_args()

    with open(args.basin_factors) as f:
        basin_factors = json.load(f)

    result = run_tcn(args.runs_dir, args.history_dir, args.archive_dir, basin_factors, args.year, args.month, args.combined_dir)

    out = Path(args.out)
    result.to_pickle(out.with_suffix(".pkl"))
    result.drop(columns=["tcn_members"]).to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out.with_suffix('.pkl')} and {out.with_suffix('.csv')}: {len(result)} basins")


if __name__ == "__main__":
    main()
