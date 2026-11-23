"""Run this ONCE PER YEAR (after the current season's observed Apr-Jul flow
is final, so it can be folded into the training record), not daily.

Fits additive-shift, EMOS-full, EMOS-constrained (all three per basin, per
issuance month, on that basin-month's own full historical record) and CMAL
(one model pooled across every basin+month) on the historical ensemble
panel, and saves every fitted parameter to a single file --
`statistical_fits.pkl` -- that apply_statistical.py loads at forecast time.
Splitting fit (slow, yearly) from apply (fast, run as often as new ensemble
forecasts arrive) means a daily cron job never re-fits anything.

Input: a historical panel pickle (DataFrame) with columns
    basin, month, year, obs_fnf, snow_pct_normal, raw_members
one row per basin x issuance-month x historical year, raw_members = that
year's N-member raw ensemble (list of Apr-Jul totals, one per member).

Usage:
    python fit_statistical.py --panel historical_ensemble_panel.pkl \
        --out statistical_fits.pkl --min-train 8
"""
import argparse
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import add_ensemble_stats, fit_slope, fit_emos_full_params, fit_emos_sigma_only_params
from cmal import NUM_COLS, fit_cmal


def fit_additive_emos(panel, min_train):
    additive, emos_full, emos_con = {}, {}, {}
    panel = panel.dropna(subset=["snow_pct_normal"]).reset_index(drop=True)
    panel["raw_logbias"] = np.log(panel["raw_median"] / panel["obs_fnf"])

    for (basin, month), bdf in panel.groupby(["basin", "month"]):
        if len(bdf) < min_train:
            print(f"skip {basin} month {month}: only {len(bdf)} historical years")
            continue
        key = (basin, int(month))

        slope, snow_mean, my = fit_slope(bdf, "snow_pct_normal", "raw_logbias")
        additive[key] = {"slope": float(slope), "mean_y": float(my), "snow_mean": float(snow_mean), "n_train": len(bdf)}

        emos_full[key] = {**fit_emos_full_params(bdf), "n_train": len(bdf)}

        fixed_c0, fixed_c1 = -my, -slope
        con_params = fit_emos_sigma_only_params(bdf, fixed_c0, fixed_c1, snow_mean)
        emos_con[key] = {"c0": float(fixed_c0), "c1": float(fixed_c1), "snow_mean": float(snow_mean),
                          **con_params, "n_train": len(bdf)}

    return additive, emos_full, emos_con


def fit_cmal_pooled(panel, n_components=3, epochs=400, lr=5e-3, seed=0):
    hist = panel.dropna(subset=["snow_pct_normal"]).reset_index(drop=True).copy()
    hist["log_raw_median"] = np.log(hist["raw_median"])
    hist["log_raw_spread"] = np.log(np.maximum(hist["raw_p90"] - hist["raw_p10"], 1e-6))
    hist["log_bias"] = np.log(hist["obs_fnf"] / hist["raw_median"])

    onehot = pd.get_dummies(hist["basin"], prefix="b").astype(float)
    onehot_cols = onehot.columns.tolist()
    X_num = hist[NUM_COLS].values.astype(float)
    mu_x, sd_x = X_num.mean(axis=0), X_num.std(axis=0) + 1e-8
    X_num_std = (X_num - mu_x) / sd_x
    X_all = np.concatenate([X_num_std, onehot.values], axis=1)
    y_all = hist["log_bias"].values.astype(float)
    n_features = X_all.shape[1]

    print(f"fitting pooled CMAL on {len(hist)} historical basin-month-years ({n_features} features)...")
    model = fit_cmal(X_all, y_all, n_features, n_components=n_components, epochs=epochs, lr=lr, seed=seed)

    return {
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "n_components": n_components,
        "mu_x": mu_x, "sd_x": sd_x,
        "onehot_cols": onehot_cols,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="historical ensemble panel pickle")
    ap.add_argument("--out", default="statistical_fits.pkl")
    ap.add_argument("--min-train", type=int, default=8)
    args = ap.parse_args()

    panel = pd.read_pickle(args.panel)
    panel = add_ensemble_stats(panel)

    additive, emos_full, emos_con = fit_additive_emos(panel, args.min_train)
    cmal_fit = fit_cmal_pooled(panel)

    fits = {
        "additive": additive, "emos_full": emos_full, "emos_constrained": emos_con,
        "cmal": cmal_fit,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_basin_months": len(additive),
        "historical_years": sorted(panel["year"].unique().tolist()),
    }
    with open(args.out, "wb") as f:
        pickle.dump(fits, f)
    print(f"wrote {args.out}: {len(additive)} basin-months fit (additive/EMOS-full/EMOS-constrained), "
          f"CMAL pooled over {cmal_fit['n_features']} features")


if __name__ == "__main__":
    main()
