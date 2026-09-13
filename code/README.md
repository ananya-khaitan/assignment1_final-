# Critical Mineral Causal Forecasting Code

This folder implements the reproducible one-step walk-forward pipeline described in [`../CURRENT_PIPELINE.md`](../CURRENT_PIPELINE.md).

## Before running

1. Create a clean Python environment and install `../requirements.txt`.
2. Retain the two project-provided workbooks in the bundled folder (or place copies in `code/data/`).
3. Download the documented public raw inputs and their provenance manifest:

   ```powershell
   python -m data_sources.download_public_inputs
   ```

   The downloader obtains the Yahoo macro series, SF Fed sentiment, Google
   Trends series, and regional Fama-French factors. `source_manifest.json`
   records source URLs, retrieval time, and SHA-256 checksums.
4. Optionally set `CRITICAL_MINERALS_FACTOR_FILE` to override the documented
   regional factor file used for CAPM/FF3/FF5 alpha.

Run from this directory:

```powershell
python run_all.py
```

Long runs expose live, file-backed progress instead of waiting for worker
stdout to flush. In a second terminal, from `code/`, use:

```powershell
python watch_run_progress.py --follow --refresh-seconds 20
```

Each company records completed origins, observed origins/second, a per-company
ETA, and an estimated UTC finish in `outputs/run_logs/<run-id>/`. The ETA is
calculated from completed work only and updates every 25 origins.

The command fails before creating outputs if a required input (including the
regional factor files) or provenance
record, or model dependency is absent. It does not silently fall back to an
approximation, machine-local path, or partial company sample.

## Main modules

- `build_master.py`: validates raw inputs and creates company/master daily data without future backfilling.
- `run_experiment.py`: calibration, final test, scheduled causal re-estimation, intervals, trading, and canonical forecast output. It forecasts every one-step origin, re-fitting parameters and full decompositions every 20 trading observations; between fits, component histories receive only causal observed-price innovation updates. Each fit origin is recorded.
- `run_paper_style_pilot.py`: an isolated Glencore pilot inspired by Liu et al. (2025). It uses causal scheduled VMD, two BIC-selected ARIMA modes, and LASSO-selected multivariate LSTMs for the remaining modes. It writes only under `outputs/paper_style_causal_pilot/` and never replaces canonical forecasts.
- `run_paper_style_ablation.py`: a development-only, 2024-capped Glencore screen comparing factor inputs, routing, and the LSTM tensor shape. It writes only under `outputs/paper_style_causal_ablation/`.
- `run_released_paper_shadow.py`: an executable forensic reconstruction of the released Liu et al. (2025) archive. It preserves full-sample VMD, globally scaled LASSO selection, the static 80:20 split, and the source's observed-test ARIMA filtering path; it writes only under `outputs/released_paper_shadow/` and is explicitly noncausal. Run it with `python run_released_paper_shadow.py --company Glencore`. Its JSONL log estimates completion from completed VMD/mode-fit steps.
- `data_sources/collect_lithium_unstructured.py`: imports an official five-query Google Trends comparison and a licensed historical lithium-headline export. It refuses to relabel the generic SF Fed sentiment or rare-earth query as lithium-specific data.
- `data_sources/prepare_lithium_inputs.py`: creates segregated model panels for COMEX lithium hydroxide and SMM lithium carbonate using only Albemarle and Ganfeng as equity signals, plus lithium attention/news, S&P 500, VIX, the US dollar index, and MVIS controls.
- `run_liu_lithium.py`: runs the deliberately noncausal Liu-style static protocol on the two lithium targets and six pre-specified feature-block ablations. Run `python run_liu_lithium.py --preflight-only` before launching the full experiment. Outputs are retrospective static predictions and cannot be described as real-time causal forecasts.
- `run_lithium_paper_pipeline.py`: runs the complete lithium-paper experiment. It retains the fixed nine-mode VMD comparator, searches VMD/CEEMDAN specifications and ARIMA/LSTM/SVR/RF component routes on a development segment, estimates convex stacking weights there, and then locks all choices for the final 20% test segment. Each completed candidate prints elapsed time and an estimated finish time. Full-sample decomposition and observed-test ARIMA filtering still make this a retrospective, noncausal protocol.
- `models/adaptive_routing.py`: implements development/test separation, component-route selection, candidate stacking, and deterministic per-fit seeds for the proposed adaptive model.
- `evaluation/validate_adaptive_lithium_run.py`: audits the completed run for model/date alignment, required comparators, route ledgers, locked-test documentation, RMSE/MAE gates, and DM wins/losses before manuscript claims are generated.
- `generate_lithium_paper_assets.py` and `build_lithium_paper.py`: create the twelve publication tables, thirteen figures, complete appendix ledgers, and the evidence-led manuscript. Publication-facing outputs omit Random Walk to match the requested comparison set; it remains in the empirical robustness ledger.
- `run_fixed_models.py`: exports fixed-model records from that canonical run.
- `watch_run_progress.py`: reads live per-company run logs and renders observed-throughput ETAs.
- `evaluation/run_model_selection_tests.py`: SPA and MCS inference using `arch`.
- `trading/factors.py`: optional CAPM/FF3/FF5 alpha estimation.

All artifacts predating a fresh successful run are legacy and are not evidence for the refactored design.
