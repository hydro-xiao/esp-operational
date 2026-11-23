"""Shared math helpers for the statistical ESP post-processors (additive
shift, EMOS-full, EMOS-constrained, CMAL). Self-contained -- no dependency on
the research/analysis scripts -- so this package can be copied to the
forecast server on its own.

All three per-basin-month methods share the same predictive shape:
    mu(log space)    = log(raw_ensemble_median) + c0 + c1 * snow_dm
    sigma(log space) = exp(d0 + d1 * logspread_dm)
where snow_dm / logspread_dm are the new forecast's snow-course
percent-of-normal / log ensemble spread, each centered on that basin-month's
own historical mean (computed once, at fit time, from the full historical
record -- see fit_statistical.py).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize


def pct_err(pred, obs):
    return 100 * (pred - obs) / obs


def empirical_crps(members, obs):
    """Empirical CRPS over a finite ensemble: mean|x_i - y| - 0.5*mean|x_i - x_j|."""
    m = np.asarray(members, dtype=float)
    term1 = np.mean(np.abs(m - obs))
    term2 = np.mean(np.abs(m[:, None] - m[None, :]))
    return term1 - 0.5 * term2


def crps_lnorm(mu, sigma, y):
    """Closed-form CRPS of a LogNormal(mu, sigma) against observation y."""
    sigma = np.asarray(sigma)
    u = (np.log(y) - mu) / sigma
    return y * (2 * norm.cdf(u) - 1) - 2 * np.exp(mu + sigma ** 2 / 2) * (
        norm.cdf(u - sigma) + norm.cdf(sigma / np.sqrt(2)) - 1
    )


def fit_slope(df, x_col, y_col):
    """OLS slope of y on x, centered (single-basin df, so this is just a
    centered least-squares slope); returns (slope, mean_x, mean_y)."""
    mean_x, mean_y = df[x_col].mean(), df[y_col].mean()
    dm_x, dm_y = df[x_col] - mean_x, df[y_col] - mean_y
    denom = np.sum(dm_x ** 2)
    slope = np.sum(dm_x * dm_y) / denom if denom > 0 else 0.0
    return slope, mean_x, mean_y


def logspread(df):
    return np.log(np.maximum(df["raw_p90"] - df["raw_p10"], 1e-6))


def add_ensemble_stats(df):
    """df needs a raw_members column (list/array per row). Adds
    raw_median/raw_p10/raw_p90, used by every method below."""
    df = df.copy()
    df["raw_median"] = df["raw_members"].apply(np.median)
    df["raw_p10"] = df["raw_members"].apply(lambda m: np.percentile(m, 10))
    df["raw_p90"] = df["raw_members"].apply(lambda m: np.percentile(m, 90))
    return df


def fit_emos_full_params(bdf):
    """Full-sample fit of (c0, c1, d0, d1) on one basin-month's historical
    rows (bdf), minimizing summed CRPS. Returns a dict of params plus the
    snow/logspread centering means used -- apply-time code must re-use these
    exact means, not recompute from the new forecast alone."""
    snow_mean = bdf["snow_pct_normal"].mean()
    ls = logspread(bdf)
    ls_mean = ls.mean()
    snow_dm = (bdf["snow_pct_normal"] - snow_mean).values
    ls_dm = (ls - ls_mean).values
    log_raw_median = np.log(bdf["raw_median"].values)
    obs = bdf["obs_fnf"].values

    slope, _, my = fit_slope(bdf, "snow_pct_normal", "raw_logbias")
    x0 = [-my, -slope, np.log(max(np.exp(ls_dm).std() if len(ls_dm) > 1 else 1.0, 0.05)), 0.0]

    def neg_obj(params):
        c0, c1, d0, d1 = params
        mu = log_raw_median + c0 + c1 * snow_dm
        sigma = np.clip(np.exp(d0 + d1 * ls_dm), 1e-4, None)
        return np.mean(crps_lnorm(mu, sigma, obs))

    res = minimize(neg_obj, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    c0, c1, d0, d1 = res.x
    return {"c0": c0, "c1": c1, "d0": d0, "d1": d1, "snow_mean": snow_mean, "logspread_mean": ls_mean}


def fit_emos_sigma_only_params(bdf, fixed_c0, fixed_c1, snow_mean):
    """(d0, d1) only, CRPS-minimized, with (c0, c1) fixed (from additive's
    own fit on the same bdf) -- the EMOS-constrained method."""
    ls = logspread(bdf)
    ls_mean = ls.mean()
    ls_dm = (ls - ls_mean).values
    snow_dm = (bdf["snow_pct_normal"] - snow_mean).values
    log_raw_median = np.log(bdf["raw_median"].values)
    obs = bdf["obs_fnf"].values
    mu = log_raw_median + fixed_c0 + fixed_c1 * snow_dm

    x0 = [np.log(max(np.exp(ls_dm).std() if len(ls_dm) > 1 else 1.0, 0.05)), 0.0]

    def neg_obj(params):
        d0, d1 = params
        sigma = np.clip(np.exp(d0 + d1 * ls_dm), 1e-4, None)
        return np.mean(crps_lnorm(mu, sigma, obs))

    res = minimize(neg_obj, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 1000})
    d0, d1 = res.x
    return {"d0": d0, "d1": d1, "logspread_mean": ls_mean}
