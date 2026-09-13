# Current Pipeline

## Status

The repository now contains a causally aligned one-step forecasting design, but it must be run from restored raw data in the pinned environment before any empirical claim is made. Existing figures, PDF/DOCX reports, IMF arrays, and LaTeX tables are legacy artifacts and must not be presented as outputs of this pipeline.

## Required inputs and environment

Install [`requirements.txt`](requirements.txt), retain the two provided
workbooks in the bundled folder (or `code/data/`), then run
`python -m data_sources.download_public_inputs` from `code/`. It downloads
the Yahoo macro series, SF Fed sentiment workbook, Google Trends series, and
regional Fama--French factor files into `code/data/`, alongside a
checksum-backed `source_manifest.json`. North America, Europe, and Developed
ex-US factors are daily; the public Emerging five-factor series is monthly,
so China strategy returns are compounded to monthly frequency before alpha
estimation. A custom factor file can override these through
`CRITICAL_MINERALS_FACTOR_FILE`.

The program validates all required raw inputs and regional factor files before it creates derived data. It has no machine-specific fallback paths and does not substitute old `outputs/` files for raw data.

## One-step causal timeline

For each target date `t`:

```text
available at t-1: target history, lagged exogenous features, prior calibration errors
                         ↓
fit / forecast P̂t, form interval and position
                         ↓
realized t: Pt, score forecast and realize the t-1 → t trading return
```

The final 20% of observations is the test period. A 60-observation calibration period immediately before it produces one-step forecasts solely from historical information. It is used for conformal residuals and threshold tuning; it is never included in final-test metrics.

## Model and selection design

- Fixed models are forecast on every calibration and test origin: random walk, drift, ARIMA, LASSO, Elastic Net, Random Forest, histogram gradient boosting, XGBoost, LightGBM, LSTM, GRU, selected ARX, VMD-LSTM, VMD-GRU, CEEMDAN-LSTM, and VMD-LASSO-ARX/LSTM.
- Parameters and full VMD/CEEMDAN decompositions are re-estimated every 20 trading observations (approximately monthly). Between re-estimation dates, each model receives only the history and exogenous values observed at that new daily origin; it neither re-fits nor uses a later observation. Decomposition component histories carry each newly observed close forward by allocating its innovation to the smoothest level/residual component, preserving the exact observed-price sum until the next full causal decomposition. The fit-origin date is stored alongside every forecast for auditability.
- Models with missing dependencies are not relabelled as another algorithm. Full runs fail during preflight until the pinned dependencies are installed.
- The routed proposed model decomposes the historical training slice only; it routes low-ApEn modes to LASSO-selected ARX and high-ApEn modes to LSTM. Its exogenous data and selected features are therefore consumed downstream.
- Dynamic selection chooses the current fixed-model forecast using only each model’s preceding calibration/final-test residuals. The currently selected forecast is exported as `Dynamic Selected`.

## Intervals and trading

- Each model’s conformal half-width is calculated only from that same model’s earlier out-of-sample residuals, with a finite-sample quantile correction.
- Prediction intervals are evaluated with coverage, calibration error, sharpness, and Winkler score.
- The signal is the forecasted return from the stored origin price divided by return-scale interval uncertainty. Thresholds are tuned separately for each model on calibration rows only.
- A position is applied to the corresponding next realized return. The outputs compare uncertainty-adjusted and directional/no-uncertainty versions under 0, 10, 25, and 50 bp costs.
- CAPM, FF3, and FF5 HAC alpha estimates are added from the documented company-region factor file. Missing factor data is explicitly marked as not evaluated, never treated as zero alpha.

## Entrypoints

From `code/`:

```powershell
python -m pip install -r ..\requirements.txt
python -m data_sources.download_public_inputs
python run_all.py
```

`run_all.py` rebuilds the master data, runs the causal experiment, exports fixed-model artifacts, and runs SPA/MCS tests. `run_experiment.py` is the forecasting runner only; `run_fixed_models.py` is an extractor for the canonical fixed-model records.

## Live run progress

Every new experiment creates `outputs/run_logs/<run-id>/run_context.json` and
one JSONL progress file per company. Workers append a flushed event every 25
origins, including calibration/test completion, observed origins per second,
remaining ETA, and estimated UTC finish. To inspect a run without relying on
buffered process stdout, run from `code/`:

```powershell
python watch_run_progress.py --follow --refresh-seconds 20
```

The ETA is an empirical estimate based only on the completed origins for that
specific company; it is intentionally not treated as a promise or research
result.

Before its first rebuild, `run_all.py` copies the existing generated files to `outputs/legacy/pre_causal_refactor/`. This archive is retained only for audit comparison and is excluded from every current report/export hook.

## Outputs

- `outputs/data/walk_forward_forecasts_all.csv`: calibration and final-test long-format forecasts.
- `outputs/data/walk_forward_forecasts.csv` and `outputs/forecasts/forecasts_canonical.csv`: final-test canonical forecasts with origin dates/prices.
- `outputs/data/walk_forward_metrics.csv`, `interval_summary.csv`, `dm_test_results.csv`, `trading_results.csv`, and `feature_stability.csv`.
- `outputs/data/fixed_model_forecasts_all.csv`, `fixed_model_metrics.csv`, `spa_pvalues_*.json`, and `mcs_set_*.json`.
- `seed_registry.json` and `experiment_registry.json` record the actual seed, data paths, split design, re-estimation cadence, model set, and calibration thresholds.
- `outputs/run_logs/<run-id>/`: durable per-company execution progress and observed ETAs for the run.
- `outputs/released_paper_shadow/<run-id>/`: a separately labelled, noncausal forensic reconstruction of the Liu et al. (2025) released code. Its static-holdout metrics, including any strong result, are excluded from canonical outputs and publication claims. The registry records each archive defect repaired solely to make the source executable.

## Publication gate

Do not treat a run as publishable unless all eight assets complete, dependencies use their real labelled implementations, raw-data provenance is archived, test results are regenerated from scratch, factor evaluation is available where claimed, and the report has been regenerated from the new canonical forecasts.
