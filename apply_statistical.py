"""Run this EVERY TIME a new raw ESP ensemble forecast is issued (daily/as
often as the ensemble updates during Jan-May). Loads the yearly fit
(statistical_fits.pkl, from fit_statistical.py) and applies additive-shift,
EMOS-full, EMOS-constrained, and CMAL to the live raw ensemble -- no
refitting, so this is cheap (arithmetic + one CMAL forward pass).

Needs, for the current issuance:
  - the live raw ensemble archive directory for this issuance month, laid
    out as <archive_dir>/<member_name>/<basin>_monthly.csv (same convention
    the ESP system already writes), each file with Date/Qsim columns
  - a snow-course percent-of-normal CSV for this issuance, columns:
    basin, month, snow_pct_normal

Does NOT need observed flow -- that's unknown at forecast time and is only
used later, for verification.

Usage:
    python apply_statistical.py \
        --fits statistical_fits.pkl \
        --archive-dir /path/to/init20270401_update20270401/basins/averaged \
        --snow-csv snow_pct_normal.csv \
        --year 2027 --month 4 \
        --out forecast_2027-04_statistical.csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm
import torch

from common import add_ensemble_stats, logspread
from cmal import CMALNet, build_features, mixture_quantiles, sample_mixture

MIN_MEMBERS = 40


def load_live_ensemble(archive_dir, year, min_members=MIN_MEMBERS):
    """Walks <archive_dir>/<member>/<basin>_monthly.csv and sums each
    member's own Apr-Jul Qsim into one raw-ensemble total per basin.
    Returns DataFrame[basin, raw_members, n_members]."""
    archive_dir = Path(archive_dir)
    member_dirs = sorted(d.name for d in archive_dir.iterdir() if d.is_dir())
    basins = sorted({f.stem.replace("_monthly", "") for m in member_dirs for f in (archive_dir / m).glob("*_monthly.csv")})

    rows = []
    for basin in basins:
        raw_tot = []
        for m in member_dirs:
            f = archive_dir / m / f"{basin}_monthly.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, parse_dates=["Date"])
            aj = df[(df["Date"].dt.year == year) & (df["Date"].dt.month >= 4) & (df["Date"].dt.month <= 7)]
            if len(aj) != 4:
                continue
            raw_tot.append(float(aj["Qsim"].sum()))
        if len(raw_tot) < min_members:
            print(f"basin {basin}: only {len(raw_tot)} members, skipping")
            continue
        rows.append({"basin": basin, "raw_members": raw_tot, "n_members": len(raw_tot)})
    return pd.DataFrame(rows)


def apply_additive(target, p):
    snow_dm = target["snow_pct_normal"] - p["snow_mean"]
    pred_lb = p["mean_y"] + p["slope"] * snow_dm
    factor = np.exp(-pred_lb)
    members = np.array(target["raw_members"])
    median = np.median(members)
    shift = median - median * factor
    corrected = np.clip(members - shift, 0, None)
    return {"p10": np.percentile(corrected, 10), "p50": np.percentile(corrected, 50),
            "p90": np.percentile(corrected, 90), "members": corrected.tolist()}


def apply_emos_full(target, p):
    snow_dm = target["snow_pct_normal"] - p["snow_mean"]
    ls_dm = float(logspread(pd.DataFrame([target])).iloc[0]) - p["logspread_mean"]
    mu = np.log(target["raw_median"]) + p["c0"] + p["c1"] * snow_dm
    sigma = np.clip(np.exp(p["d0"] + p["d1"] * ls_dm), 1e-4, None)
    p10, p50, p90 = lognorm.ppf([0.1, 0.5, 0.9], s=sigma, scale=np.exp(mu))
    return {"p10": p10, "p50": p50, "p90": p90, "mu": float(mu), "sigma": float(sigma)}


def apply_emos_constrained(target, p):
    snow_dm = target["snow_pct_normal"] - p["snow_mean"]
    ls_dm = float(logspread(pd.DataFrame([target])).iloc[0]) - p["logspread_mean"]
    mu = np.log(target["raw_median"]) + p["c0"] + p["c1"] * snow_dm
    sigma = np.clip(np.exp(p["d0"] + p["d1"] * ls_dm), 1e-4, None)
    p10, p50, p90 = lognorm.ppf([0.1, 0.5, 0.9], s=sigma, scale=np.exp(mu))
    return {"p10": p10, "p50": p50, "p90": p90, "mu": float(mu), "sigma": float(sigma)}


def apply_cmal(df, cmal_fit, n_members=45, seed=0):
    model = CMALNet(cmal_fit["n_features"], cmal_fit["n_components"])
    model.load_state_dict(cmal_fit["state_dict"])
    model.eval()
    X = build_features(df, cmal_fit["mu_x"], cmal_fit["sd_x"], cmal_fit["onehot_cols"])
    with torch.no_grad():
        alpha, mu, s, tau = model.forward(torch.tensor(X, dtype=torch.float32))
        alpha, mu, s, tau = alpha.numpy(), mu.numpy(), s.numpy(), tau.numpy()

    rng = np.random.default_rng(seed)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        p10, p50, p90 = mixture_quantiles(alpha[i], mu[i], s[i], tau[i], rng=rng)
        members = sample_mixture(alpha[i], mu[i], s[i], tau[i], n_members, rng)
        raw_median = r["raw_median"]
        rows.append({
            "basin": r["basin"], "month": r["month"],
            "p10": raw_median * np.exp(p10), "p50": raw_median * np.exp(p50), "p90": raw_median * np.exp(p90),
            "members": (raw_median * np.exp(members)).tolist(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--snow-csv", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.fits, "rb") as f:
        fits = pickle.load(f)

    live = load_live_ensemble(args.archive_dir, args.year)
    live = add_ensemble_stats(live)
    live["month"] = args.month

    snow = pd.read_csv(args.snow_csv)
    if "snow_pct_normal" not in snow.columns:
        alt = [c for c in snow.columns if c.startswith("snow_pct_normal")]
        if not alt:
            raise ValueError(f"{args.snow_csv} has no snow_pct_normal(-like) column: {snow.columns.tolist()}")
        snow = snow.rename(columns={alt[0]: "snow_pct_normal"})
    snow_this_month = snow[snow["month"] == args.month][["basin", "snow_pct_normal"]]
    live = live.merge(snow_this_month, on="basin", how="inner")
    print(f"{len(live)} / basins with both ensemble + snow data for month {args.month}")

    rows = []
    for _, target in live.iterrows():
        key = (target["basin"], args.month)
        out = {"basin": target["basin"], "month": args.month, "year": args.year, "n_members": target["n_members"]}

        if key in fits["additive"]:
            a = apply_additive(target, fits["additive"][key])
            out.update({f"additive_{k}": v for k, v in a.items()})
        if key in fits["emos_full"]:
            e = apply_emos_full(target, fits["emos_full"][key])
            out.update({f"emos_full_{k}": v for k, v in e.items()})
        if key in fits["emos_constrained"]:
            c = apply_emos_constrained(target, fits["emos_constrained"][key])
            out.update({f"emos_con_{k}": v for k, v in c.items()})
        if key not in fits["additive"]:
            print(f"no historical fit for {key}, additive/EMOS skipped (raw ensemble only)")
        rows.append(out)

    result = pd.DataFrame(rows)
    cmal_result = apply_cmal(live, fits["cmal"]).rename(columns=lambda c: f"cmal_{c}" if c not in ("basin", "month") else c)
    result = result.merge(cmal_result, on=["basin", "month"], how="left")

    out = Path(args.out)
    pkl_path = out.with_suffix(".pkl")
    result.to_pickle(pkl_path)  # full result, members lists included (needed for CRPS later)
    member_cols = [c for c in result.columns if c.endswith("_members")]
    result.drop(columns=member_cols).to_csv(out.with_suffix(".csv"), index=False)  # quantiles only, for quick review
    print(f"wrote {pkl_path} and {out.with_suffix('.csv')}: {len(result)} basins")


if __name__ == "__main__":
    main()
