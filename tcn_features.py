"""Minimal feature-transform helpers the TCN needs at inference time, ported
from lstm.new.dev.jul.2026/scr/dataset.py (that module has many
training-only functions this package doesn't need)."""
import numpy as np

FEATURE_COLS = ["SWE", "SMTOT", "T2D", "SWDOWN", "LWDOWN", "Q2D", "PREC_daily_mm", "Qsim_daily_mm"]
PREC_COL = "PREC_daily_mm"
QSIM_COL = "Qsim_daily_mm"
LOG_EPS = 0.1


def _denom(df, prec_mean):
    """PREC_daily_mm, falling back to the basin's long-term mean when
    PREC_daily_mm <= 0 (dry months, avoids division by zero)."""
    return df[PREC_COL].where(df[PREC_COL] > 0, prec_mean).values


def raw_features(df, prec_mean):
    """Feature matrix with Qsim_daily_mm replaced by the same
    log10(sqrt(x)+LOG_EPS) transform used for the model's target."""
    feats = df[FEATURE_COLS].copy()
    denom = _denom(df, prec_mean)
    qsim_scaled = df[QSIM_COL].values / denom
    feats[QSIM_COL] = np.log10(np.sqrt(qsim_scaled) + LOG_EPS)
    return feats.values


def inverse_transform_target(y_norm, stats, denom):
    """Invert z-score -> log10(sqrt(x)+LOG_EPS) -> scaled flow -> FNF_daily_mm."""
    flow_log = y_norm * stats["target_std"] + stats["target_mean"]
    sqrt_val = np.clip(10 ** flow_log - LOG_EPS, a_min=0.0, a_max=None)
    flow_scaled = sqrt_val ** 2
    return flow_scaled * denom
