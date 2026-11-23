# ESP operational post-processing

Deployable scripts for the 5 non-LSTM post-processing methods compared in the
ESP dashboard: additive shift, EMOS-full, EMOS-constrained, CMAL, and TCN
ensemble. (Raw ensemble needs no script -- it's the forecast system's own
output, unmodified.) Everything here is self-contained: no imports from the
research/analysis directories, so this folder can be copied to the server as-is.

## Two kinds of job: fit (yearly) vs apply (daily/per-issuance)

The 4 statistical methods are fit from the full historical record, then
applied to each new ensemble forecast without refitting. Keep these on
separate schedules:

| Script | Cadence | What it does |
|---|---|---|
| `fit_statistical.py` | **once a year**, after the season's observed Apr-Jul flow is final | fits additive/EMOS-full/EMOS-constrained (per basin+month) and the pooled CMAL model on the historical ensemble panel; writes `statistical_fits.pkl` |
| `apply_statistical.py` | **every new ensemble issuance** (as often as the raw ESP ensemble updates) | loads `statistical_fits.pkl`, applies all 4 methods to the live ensemble, writes a forecast file -- no refitting, just arithmetic + one CMAL forward pass |
| `compute_basin_factors.py` | once (re-run yearly alongside the above; the factor almost never changes) | derives each basin's FNF_daily_mm -> raw-volume conversion factor; writes `basin_factors.json` |
| TCN retraining (`train_encdec_tcn.py`, in `lstm.new.dev.jul.2026/scr/` -- not duplicated here, it's a training job not a daily op) | once a year | retrains the 5 per-issuance-month TCN checkpoints on the updated historical record |
| `apply_tcn.py` | **every new ensemble issuance** | loads the already-trained TCN checkpoint (`runs/tcn_<month>/`) and runs inference on the live ensemble -- no training |

## Files

- `common.py`, `cmal.py` -- shared math (additive/EMOS CRPS fitting, CMAL model) used by `fit_statistical.py` and `apply_statistical.py`.
- `fit_statistical.py` -- yearly fit job. Input: a historical ensemble panel pickle (columns `basin, month, year, obs_fnf, snow_pct_normal, raw_members`). Output: `statistical_fits.pkl`.
- `apply_statistical.py` -- per-issuance apply job. Reads the live raw-ensemble archive directory + a snow percent-of-normal CSV (`basin, month, snow_pct_normal`), applies all 4 statistical methods, writes `<out>.pkl` (full, with member lists) and `<out>.csv` (quantiles only).
- `tcn_model.py`, `tcn_features.py` -- TCN model definition + feature transforms, no training code.
- `compute_basin_factors.py` -- one-time/yearly, writes `basin_factors.json`.
- `apply_tcn.py` -- per-issuance TCN inference. Writes `<out>.pkl` / `<out>.csv` in the same shape as `apply_statistical.py`.
- `runs/tcn_{jan,feb,mar,apr,may}/` -- the 5 trained checkpoints (`config.json`, `norm_stats.npz`, `model.pt`), one per issuance month. Retrain yearly with `train_encdec_tcn.py`; just drop the new files in here afterward.

## What the server needs to provide at apply time

- The live raw ESP ensemble for the current issuance, laid out as
  `<archive_dir>/<member_name>/<basin>_monthly.csv` (Date + Qsim columns for
  the statistical methods; Date + SWE/SMTOT/T2D/SWDOWN/LWDOWN/Q2D/PREC/Qsim
  for TCN) -- the same files the forecast system already writes per ensemble
  member.
- A snow-course percent-of-normal CSV for this issuance month (`basin,
  month, snow_pct_normal`), for the statistical methods and CMAL.
- For TCN only: each basin's observed feature history file
  (`<history_dir>/<basin>.csv`, with SWE/SMTOT/T2D/SWDOWN/LWDOWN/Q2D/
  PREC_daily_mm/Qsim_daily_mm/FNF_daily_mm columns) as the lookback window,
  and -- for a May-1 issuance only -- the observed combined monthly record
  (`<combined_dir>/<basin>_monthly.csv`) to splice in April's already-elapsed
  real nowcast.
- No observed flow is needed at apply time for any method -- `obs_fnf` is
  only used during the yearly fit, from already-final historical years.

## Example

```bash
# yearly maintenance
python fit_statistical.py --panel historical_panel.pkl --out statistical_fits.pkl
python compute_basin_factors.py --history-dir .../converted/monthly \
    --combined-dir .../out.zipped/combined --basins AMF,CSN,... --out basin_factors.json
# (retrain TCN separately, then refresh runs/tcn_*/)

# every new issuance (e.g. April 1)
python apply_statistical.py --fits statistical_fits.pkl \
    --archive-dir /live/init20270401_update20270401/basins/averaged \
    --snow-csv snow_pct_normal.csv --year 2027 --month 4 \
    --out forecast_2027-04_statistical

python apply_tcn.py --runs-dir runs --history-dir .../converted/monthly \
    --archive-dir /live/init20270401_update20270401/basins/averaged \
    --basin-factors basin_factors.json --year 2027 --month 4 \
    --out forecast_2027-04_tcn
```

## Dependencies

`numpy`, `pandas`, `scipy`, `torch` (CPU is fine -- inference and the CMAL/TCN
forward passes are both small).

## Verified against the research results

Both apply scripts were checked against the already-published research
output before being handed off: `apply_statistical.py`'s additive/EMOS-full/
EMOS-constrained medians match `additive_result_2026.pkl` /
`emos_full_result_2026.pkl` / `emos_constrained_result_2026.pkl` exactly
(CMAL matches to within Monte Carlo sampling noise -- its quantiles are
themselves a stochastic sample-based estimate); `apply_tcn.py`'s member-level
output matches `tcn_ensemble_hindcast.pkl` exactly (max abs diff 0.0) for
both a normal issuance (month 4) and the May-1 nowcast-addend path (month 5).
