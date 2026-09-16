# Critical Minerals Forecasting — Assignment 1

Adaptive multimodal decomposition-routing ensemble for lithium price forecasting, benchmarked against a causally-corrected re-implementation of Liu et al. (2024).

---

## Repository Structure

```
assignment1_pratham/
│
├── code/                            ← All source code
│   ├── models/
│   │   ├── adaptive_routing.py      ← OUR MODEL: adaptive K/α grid search,
│   │   │                              per-mode model competition (ARIMA/LSTM/SVR/RF),
│   │   │                              convex stacking ensemble
│   │   └── released_paper_shadow.py ← REFERENCE MODEL: causal-corrected replica of
│   │                                  Liu et al. (full-sample VMD fixed → train-only,
│   │                                  global scaler fixed → causal LASSO, ARIMA filter
│   │                                  fixed → true forecast)
│   │
│   ├── run_lithium_paper_pipeline.py ← MAIN ENTRY POINT: runs both the fixed reference
│   │                                   models AND our adaptive ensemble, writes all
│   │                                   outputs to outputs/lithium_paper/<run-id>/
│   ├── run_experiment.py             ← Walk-forward causal pipeline (separate experiment)
│   │
│   ├── decomposition/
│   │   └── causal.py                ← Train-only VMD & CEEMDAN (used by our model)
│   ├── features/
│   │   └── selection.py             ← Causal LASSO feature selection (TimeSeriesSplit)
│   ├── evaluation/
│   │   └── metrics.py               ← RMSE, MAE, MAPE, DM test, forecast metrics
│   ├── trading/
│   │   └── signals.py               ← Trading signal evaluation
│   ├── uncertainty/
│   │   └── intervals.py             ← Conformal prediction intervals
│   │
│   ├── data/lithium/processed/      ← INPUT DATA: cleaned lithium carbonate &
│   │                                  hydroxide panels (Liu et al. feature set)
│   ├── lithium_config.py            ← All hyperparameters (K, α grid, lags, epochs etc.)
│   └── requirements.txt             ← Python dependencies
│
├── outputs/
│   ├── lithium_paper/
│   │   └── lithium_prices_20260913T070749Z/   ← KEY RESULTS: our fixed run
│   │       ├── forecast_metrics.csv            ← RMSE/MAE/MAPE for all models
│   │       ├── dm_tests.csv                    ← Diebold-Mariano significance tests
│   │       ├── trading_metrics.csv             ← Trading strategy performance
│   │       ├── interval_metrics.csv            ← Conformal interval coverage/width
│   │       ├── forecast_ledger.csv             ← Full date-by-date predictions
│   │       ├── adaptive_candidate_search.csv   ← Dev-set scores for all 14 K/α configs
│   │       ├── adaptive_component_routes.csv   ← Which model won each mode
│   │       └── mode_feature_selection.csv      ← LASSO-selected features per mode
│   │
│   ├── tables/                      ← LaTeX tables for the paper (table1–15)
│   ├── figures/                     ← Generated plots and figures
│   └── papers/                      ← Rendered PDF page images
│
├── tests/                           ← Pytest unit tests (CI enforced)
│   ├── test_adaptive_routing.py     ← Tests for adaptive model causal correctness
│   ├── test_causal_design.py        ← Tests for no data leakage in decomposition
│   └── test_released_paper_shadow.py← Tests for reference model behaviour
│
├── Critical_Mineral_Final_Report.pdf ← Final submitted paper
├── requirements.txt                  ← Install dependencies
└── .github/workflows/causal-design.yml ← CI: runs causal design tests on every push
```

---

## The Two Pipelines

### Reference (Fixed Liu et al.)
**Entry point:** `run_lithium_paper_pipeline.py`

A causally-corrected replica of Liu et al. (2024). The original paper's code had three data leakage issues which artificially inflated performance. These are fixed here so the comparison against our model is fair:

| Issue | Original (Leaky) | Fixed |
|:---|:---|:---|
| VMD decomposition | Applied to full train+test series | Train window only |
| Feature scaling | `StandardScaler` fit on all data | Fit on train only |
| ARIMA evaluation | `.apply(y_test)` filter (observes test targets) | `.forecast(steps=N)` true forecast |

Architecture is otherwise identical to the paper: fixed K=9 modes, α=3000, LASSO feature selection, ARIMA for low-frequency modes, LSTM for high-frequency modes, simple summation.

### Our Model (Adaptive Routing Ensemble)
**Entry point:** `run_lithium_paper_pipeline.py` → calls `models/adaptive_routing.py`

Extends the reference architecture with three improvements:
1. **Adaptive decomposition search** — grid searches K ∈ {7,8,9,10} and α ∈ {1500, 3000, 4500} for VMD, plus CEEMDAN, on a held-out development set
2. **Per-mode model competition** — for every frequency mode, competes ARIMA, LSTM, SVR and RF; routes each mode to the best performer
3. **Convex stacking** — fits optimal non-negative weights across route predictions on development data instead of simple summation

---

## Key Results

From `outputs/lithium_paper/lithium_prices_20260913T070749Z/forecast_metrics.csv`:

| Model | Carbonate RMSE | Hydroxide RMSE |
|:---|---:|---:|
| **Adaptive Ensemble (Ours)** | **2,916** | **0.277** |
| VMD-ARIMA (fixed reference) | 41,142 | 6.022 |
| VMD-LSTM (fixed reference) | 45,803 | 5.826 |
| Fixed VMD-LASSO-ARIMA/LSTM | 42,075 | 5.997 |

Our adaptive ensemble achieves **~14× lower RMSE** on carbonate and **~21× lower RMSE** on hydroxide vs. the best fixed reference.

---

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything (reference + our model) for lithium prices
cd code
python run_lithium_paper_pipeline.py --target-set prices

# Results written to outputs/lithium_paper/<timestamp>/
```

> ⚠️ Full run takes ~18 hours on CPU due to the adaptive grid search (14 configs × ~8 modes × 4 model competitions). Checkpoints are saved so runs are resumable.
