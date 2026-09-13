"""Causal one-step walk-forward experiment runner.

The final test period is never used for calibration, model selection, or
threshold tuning.  Every forecast row stores its forecast origin explicitly,
so trading and accuracy evaluation can be audited without inferring timing.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    AE_M,
    AE_R_COEF,
    ALLOW_APPROXIMATE_DECOMPOSITION,
    ANNUALIZATION,
    CALIBRATION_SIZE,
    CEEMDAN_TRIALS,
    CONFORMAL_ALPHA,
    EXOG_COLS,
    FACTOR_DIR,
    FACTOR_FILE,
    FACTOR_REGION_BY_COMPANY,
    MODEL_REFIT_INTERVAL,
    MODEL_THREADS,
    MAX_PARALLEL_COMPANIES,
    PROGRESS_EVENT_INTERVAL,
    MIN_TRAIN_SIZE,
    ORDER,
    OUT_DATA,
    OUT_FORECASTS,
    OUT_ROOT,
    RANDOM_SEED,
    REQUIRED_MODEL_PACKAGES,
    RNN_EPOCHS,
    SLUG,
    TEST_FRAC,
    THRESHOLD_GRID,
    TRANSACTION_COSTS,
    VMD_ALPHA,
    VMD_DC,
    VMD_INIT,
    VMD_K,
    VMD_TAU,
    VMD_TOL,
    TORCH_THREADS,
    required_factor_inputs,
    validate_raw_inputs,
)
from decomposition.causal import CausalComponentHistory, approximate_entropy, ceemdan_decompose_train_only, vmd_decompose_train_only
from evaluation.metrics import dm_test, forecast_metrics
from features.selection import SelectedARXForecaster
from models.registry import RecurrentForecaster, stateful_candidate_models
from trading.signals import evaluate_trading
from trading.factors import factor_adjusted_alphas, load_factor_data
from uncertainty.intervals import conformal_width_from_residuals, interval_metrics
from utils.causal_pipeline import WalkForwardSplit, calibration_and_test_splits
from utils.progress import CompanyProgressLogger, utc_timestamp
from utils.reproducibility import require_packages, set_global_seed


def _load_series(company: str) -> tuple[pd.Series, pd.DataFrame]:
    """Load a company target and a strictly backward-filled exogenous frame."""
    price_path = Path(OUT_DATA) / f"{SLUG[company]}_daily.csv"
    master_path = Path(OUT_DATA) / "master_daily_prices.csv"
    if not price_path.exists() or not master_path.exists():
        raise FileNotFoundError(
            f"Derived inputs are missing for {company}. Run build_master.py after raw-input validation."
        )
    price = pd.read_csv(price_path, index_col=0, parse_dates=True)["Close"].astype(float).dropna().sort_index()
    price = price[~price.index.duplicated(keep="last")]
    master = pd.read_csv(master_path, index_col=0, parse_dates=True).sort_index()
    missing_columns = [column for column in EXOG_COLS if column not in master.columns]
    if missing_columns:
        raise ValueError(f"Master data lacks required exogenous columns: {missing_columns}")
    # Forward filling only carries the last information already observed.  It
    # deliberately never backfills an earlier date with a later release.
    exog = master.reindex(price.index)[EXOG_COLS].astype(float).ffill()
    if exog.iloc[-1].isna().any():
        missing = exog.columns[exog.iloc[-1].isna()].tolist()
        raise ValueError(f"Latest exogenous values are unavailable for {company}: {missing}")
    return price, exog


class _DecompositionState:
    """Scheduled-fit decomposition models with daily causal state updates."""

    def __init__(self, rnn_epochs: int) -> None:
        self.rnn_epochs = rnn_epochs
        self.vmd_lstm: list[RecurrentForecaster] = []
        self.vmd_gru: list[RecurrentForecaster] = []
        self.ceemdan_lstm: list[RecurrentForecaster] = []
        self.routes: list[tuple[str, RecurrentForecaster | SelectedARXForecaster]] = []
        self.metadata: dict[str, Any] = {}
        self.vmd_history: CausalComponentHistory | None = None
        self.ceemdan_history: CausalComponentHistory | None = None

    @staticmethod
    def _vmd(values: np.ndarray) -> np.ndarray:
        return vmd_decompose_train_only(
            values, VMD_ALPHA, VMD_TAU, VMD_K, VMD_DC, VMD_INIT, VMD_TOL, allow_approximation=ALLOW_APPROXIMATE_DECOMPOSITION
        )

    @staticmethod
    def _ceemdan(values: np.ndarray) -> np.ndarray:
        return ceemdan_decompose_train_only(values, trials=CEEMDAN_TRIALS, seed=RANDOM_SEED)

    def fit(self, train_y: pd.Series, exog_train: pd.DataFrame) -> "_DecompositionState":
        values = train_y.values.astype(float)
        vmd_modes = self._vmd(values)
        ceemdan_modes = self._ceemdan(values)
        self.vmd_history = CausalComponentHistory(vmd_modes, values)
        self.ceemdan_history = CausalComponentHistory(ceemdan_modes, values)
        self.vmd_lstm = [RecurrentForecaster(cell="lstm", epochs=self.rnn_epochs).fit(mode) for mode in vmd_modes]
        self.vmd_gru = [RecurrentForecaster(cell="gru", epochs=self.rnn_epochs).fit(mode) for mode in vmd_modes]
        self.ceemdan_lstm = [RecurrentForecaster(cell="lstm", epochs=self.rnn_epochs).fit(mode) for mode in ceemdan_modes]

        entropies = [approximate_entropy(mode, m=AE_M, r_coef=AE_R_COEF) for mode in vmd_modes]
        route_cutoff = float(np.nanmedian(entropies))
        routes: list[dict[str, Any]] = []
        self.routes = []
        for mode_index, (mode, entropy) in enumerate(zip(vmd_modes, entropies), start=1):
            mode_series = pd.Series(mode, index=train_y.index, dtype=float)
            if np.isfinite(entropy) and entropy <= route_cutoff:
                state = SelectedARXForecaster(lags=5).fit(mode_series, exog_train)
                self.routes.append(("LASSO-selected ARX", state))
                routes.append({"mode": mode_index, "ApEn": float(entropy), "route": "LASSO-selected ARX", "selected_features": state.selected_columns, "alpha": state.alpha})
            else:
                state = RecurrentForecaster(cell="lstm", epochs=self.rnn_epochs).fit(mode)
                self.routes.append(("LSTM", state))
                routes.append({"mode": mode_index, "ApEn": float(entropy), "route": "LSTM", "selected_features": [], "alpha": None})
        self.metadata = {
            "vmd_apen": entropies,
            "route_cutoff": route_cutoff,
            "routes": routes,
            "decomposition_fit_observations": len(values),
            "between_refit_component_update": "causal level-component innovation carry-forward",
        }
        return self

    def predict(self, train_y: pd.Series, exog_train: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
        if self.vmd_history is None or self.ceemdan_history is None:
            raise RuntimeError("Decomposition state has not been fit")
        # A full VMD/CEEMDAN is intentionally performed only by ``fit`` at the
        # scheduled re-estimation origin.  Between those origins this consumes
        # only closes that are already in ``train_y`` and never reuses a target.
        vmd_modes = self.vmd_history.advance(train_y.values.astype(float))
        ceemdan_modes = self.ceemdan_history.advance(train_y.values.astype(float))
        if len(vmd_modes) != len(self.vmd_lstm) or len(vmd_modes) != len(self.routes):
            raise RuntimeError("VMD component count changed unexpectedly")
        if len(ceemdan_modes) != len(self.ceemdan_lstm):
            raise RuntimeError("CEEMDAN component count changed unexpectedly")
        routed = 0.0
        for mode, (route, state) in zip(vmd_modes, self.routes):
            if route == "LASSO-selected ARX":
                routed += float(state.predict(pd.Series(mode, index=train_y.index), exog_train, steps=1)[0])
            else:
                routed += float(state.predict(mode, steps=1)[0])
        return (
            {
                "VMD-LSTM": float(sum(state.predict(mode, steps=1)[0] for state, mode in zip(self.vmd_lstm, vmd_modes))),
                "VMD-GRU": float(sum(state.predict(mode, steps=1)[0] for state, mode in zip(self.vmd_gru, vmd_modes))),
                "CEEMDAN-LSTM": float(sum(state.predict(mode, steps=1)[0] for state, mode in zip(self.ceemdan_lstm, ceemdan_modes))),
                "VMD-LASSO-ARX/LSTM": float(routed),
            },
            self.metadata,
        )


class _CandidateEngine:
    """Daily one-step forecaster with a causal scheduled re-fit policy."""

    def __init__(self, refit_interval: int = MODEL_REFIT_INTERVAL, rnn_epochs: int = RNN_EPOCHS) -> None:
        self.refit_interval = refit_interval
        self.rnn_epochs = rnn_epochs
        self.fixed: dict[str, Any] = {}
        self.selected: SelectedARXForecaster | None = None
        self.decomposition: _DecompositionState | None = None
        self.last_fit_train_size: int | None = None
        self.refit_this_origin = False

    def _fit(self, train_y: pd.Series, exog_train: pd.DataFrame) -> None:
        self.fixed = stateful_candidate_models(rnn_epochs=self.rnn_epochs)
        for state in self.fixed.values():
            state.fit(train_y.values)
        self.selected = SelectedARXForecaster(lags=5).fit(train_y, exog_train)
        self.decomposition = _DecompositionState(self.rnn_epochs).fit(train_y, exog_train)
        self.last_fit_train_size = len(train_y)

    def forecast(self, train_y: pd.Series, exog_train: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
        due = self.last_fit_train_size is None or len(train_y) - self.last_fit_train_size >= self.refit_interval
        self.refit_this_origin = bool(due)
        if due:
            self._fit(train_y, exog_train)
        assert self.selected is not None and self.decomposition is not None
        predictions = {name: float(state.predict(train_y.values, steps=1)[0]) for name, state in self.fixed.items()}
        predictions["Selected ARX"] = float(self.selected.predict(train_y, exog_train, steps=1)[0])
        try:
            decomposed, decomp_metadata = self.decomposition.predict(train_y, exog_train)
        except RuntimeError as error:
            if "component count changed" not in str(error):
                raise
            self._fit(train_y, exog_train)
            self.refit_this_origin = True
            assert self.decomposition is not None and self.selected is not None
            predictions = {name: float(state.predict(train_y.values, steps=1)[0]) for name, state in self.fixed.items()}
            predictions["Selected ARX"] = float(self.selected.predict(train_y, exog_train, steps=1)[0])
            decomposed, decomp_metadata = self.decomposition.predict(train_y, exog_train)
        predictions.update(decomposed)
        return predictions, {"Selected ARX": {"selected_features": self.selected.selected_columns, "alpha": self.selected.alpha}, "VMD-LASSO-ARX/LSTM": decomp_metadata}


def _decomposition_forecasts(train_y: pd.Series, exog_train: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
    """One-shot compatibility wrapper for the same scheduled-fit model code."""
    state = _DecompositionState(RNN_EPOCHS).fit(train_y, exog_train)
    return state.predict(train_y, exog_train)


def _forecast_all_candidates(train_y: pd.Series, exog_train: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
    """One-shot compatibility wrapper for the production candidate engine."""
    return _CandidateEngine(refit_interval=1, rnn_epochs=RNN_EPOCHS).forecast(train_y, exog_train)


def _record_predictions(
    company: str,
    partition: str,
    price: pd.Series,
    exog: pd.DataFrame,
    splits: list[WalkForwardSplit],
    engine: _CandidateEngine,
    residual_history: dict[str, list[float]] | None = None,
    progress: CompanyProgressLogger | None = None,
    completed_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[dict[str, Any]]]:
    """Generate aligned one-step predictions and sequential conformal widths."""
    records: list[dict[str, Any]] = []
    history: dict[str, list[float]] = defaultdict(list)
    if residual_history is not None:
        history.update({model: list(errors) for model, errors in residual_history.items()})
    feature_log: list[dict[str, Any]] = []
    for number, split in enumerate(splits, start=1):
        train_y = price.iloc[: split.train_end]
        exog_train = exog.iloc[: split.train_end]
        origin_date = price.index[split.train_end - 1]
        target_date = price.index[split.train_end]
        origin_price = float(train_y.iloc[-1])
        actual = float(price.iloc[split.train_end])
        predictions, metadata = engine.forecast(train_y, exog_train)
        if engine.last_fit_train_size is None:
            raise RuntimeError("Forecast engine did not retain its fit origin")
        fit_origin_date = price.index[engine.last_fit_train_size - 1]
        for model, prediction in predictions.items():
            errors = history[model]
            width = conformal_width_from_residuals(errors, alpha=CONFORMAL_ALPHA) if len(errors) >= 2 else np.nan
            predicted_return = float((prediction - origin_price) / origin_price)
            records.append(
                {
                    "Company": company,
                    "Partition": partition,
                    "Origin Date": origin_date,
                    "Model Fit Origin Date": fit_origin_date,
                    "Date": target_date,
                    "Origin Price": origin_price,
                    "Actual": actual,
                    "Model": model,
                    "Forecast": float(prediction),
                    "Interval Width": width,
                    "Lower": float(prediction - width) if np.isfinite(width) else np.nan,
                    "Upper": float(prediction + width) if np.isfinite(width) else np.nan,
                    "Predicted Return": predicted_return,
                }
            )
            errors.append(actual - prediction)
        if engine.refit_this_origin and "Selected ARX" in metadata:
            feature_log.append({"company": company, "date": str(target_date.date()), **metadata["Selected ARX"]})
        if number % PROGRESS_EVENT_INTERVAL == 0 or number == len(splits):
            print(f"  {company}: {partition} forecast {number}/{len(splits)}", flush=True)
            if progress is not None:
                progress.record(
                    "running",
                    partition=partition,
                    partition_completed=number,
                    partition_total=len(splits),
                    overall_completed=completed_offset + number,
                    target_date=str(target_date.date()),
                    refit=engine.refit_this_origin,
                )
    return records, history, feature_log


def _tune_thresholds(calibration: pd.DataFrame) -> dict[str, float]:
    """Tune each model's uncertainty threshold using only calibration rows."""
    thresholds: dict[str, float] = {}
    for model, group in calibration.groupby("Model"):
        valid = group.dropna(subset=["Interval Width"])
        if len(valid) < 10:
            raise RuntimeError(f"Insufficient conformal calibration rows to tune {model}")
        candidates: list[tuple[float, float]] = []
        for threshold in THRESHOLD_GRID:
            result = evaluate_trading(
                valid["Origin Price"],
                valid["Actual"],
                valid["Forecast"],
                valid["Interval Width"],
                threshold=threshold,
                cost=TRANSACTION_COSTS[1],
                annualization=ANNUALIZATION,
            )
            score = result.sharpe if np.isfinite(result.sharpe) else -np.inf
            candidates.append((score, float(threshold)))
        # Stable tie-breaking selects the lower threshold.
        thresholds[model] = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
    return thresholds


def _append_dynamic_selection(
    test_records: list[dict[str, Any]],
    calibration_residuals: dict[str, list[float]],
    threshold_by_model: dict[str, float],
) -> list[dict[str, Any]]:
    """Choose a current forecast using only known preceding OOS residuals."""
    grouped: dict[tuple[str, pd.Timestamp], list[dict[str, Any]]] = defaultdict(list)
    for record in test_records:
        grouped[(record["Company"], record["Date"])].append(record)
    history = {model: list(errors) for model, errors in calibration_residuals.items()}
    dynamic_records: list[dict[str, Any]] = []
    for (_, _), rows in sorted(grouped.items(), key=lambda item: item[0][1]):
        available = [row["Model"] for row in rows if len(history.get(row["Model"], [])) >= 2]
        if not available:
            raise RuntimeError("Dynamic selection has no historical calibration residuals")
        selected_model = min(
            available,
            key=lambda model: (float(np.sqrt(np.mean(np.square(history[model][-CALIBRATION_SIZE:])))), model),
        )
        selected_row = next(row for row in rows if row["Model"] == selected_model)
        dynamic = dict(selected_row)
        dynamic["Model"] = "Dynamic Selected"
        dynamic["Selected Model"] = selected_model
        dynamic["Threshold"] = threshold_by_model[selected_model]
        dynamic_records.append(dynamic)
        # The outcome becomes available only after the signal/selection above.
        for row in rows:
            history[row["Model"]].append(float(row["Actual"] - row["Forecast"]))
    return dynamic_records


def run_company(
    company: str,
    *,
    run_id: str | None = None,
    progress_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    price, exog = _load_series(company)
    test_size = max(1, int(len(price) * TEST_FRAC))
    calibration_splits, test_splits = calibration_and_test_splits(
        len(price), MIN_TRAIN_SIZE, CALIBRATION_SIZE, test_size
    )
    print(f"Processing {company}: {len(calibration_splits)} calibration + {len(test_splits)} final-test forecasts", flush=True)
    progress = (
        CompanyProgressLogger(
            progress_path,
            run_id=run_id or "ad-hoc",
            company=company,
            total_origins=len(calibration_splits) + len(test_splits),
        )
        if progress_path is not None
        else None
    )
    if progress is not None:
        progress.started(calibration_total=len(calibration_splits), test_total=len(test_splits))
    engine = _CandidateEngine()
    try:
        calibration_records, calibration_residuals, feature_log = _record_predictions(
            company, "calibration", price, exog, calibration_splits, engine, progress=progress
        )
        threshold_by_model = _tune_thresholds(pd.DataFrame(calibration_records))
        test_records, _, test_features = _record_predictions(
            company,
            "test",
            price,
            exog,
            test_splits,
            engine,
            residual_history=calibration_residuals,
            progress=progress,
            completed_offset=len(calibration_splits),
        )
        dynamic_records = _append_dynamic_selection(test_records, calibration_residuals, threshold_by_model)

        test_frame = pd.DataFrame(test_records + dynamic_records)
        # Fixed-model rows receive their calibration-chosen threshold; dynamic rows
        # retain the threshold of the selected fixed model on that date.
        test_frame["Threshold"] = test_frame.apply(
            lambda row: row["Threshold"] if pd.notna(row.get("Threshold", np.nan)) else threshold_by_model[row["Model"]], axis=1
        )
        calibration_frame = pd.DataFrame(calibration_records)
        calibration_frame["Threshold"] = calibration_frame["Model"].map(threshold_by_model)
        all_frame = pd.concat([calibration_frame, test_frame], ignore_index=True)
        if progress is not None:
            progress.record(
                "completed",
                partition="test",
                partition_completed=len(test_splits),
                partition_total=len(test_splits),
                overall_completed=len(calibration_splits) + len(test_splits),
                target_date=str(price.index[-1].date()),
                detail="Forecast, calibration, and dynamic-selection records completed.",
            )
        return all_frame, feature_log + test_features, threshold_by_model
    except Exception as exc:
        if progress is not None:
            progress.failed(f"{type(exc).__name__}: {exc}")
        raise


def _run_company_worker(
    company: str,
    seed: int,
    run_id: str | None = None,
    progress_path: str | Path | None = None,
) -> tuple[str, pd.DataFrame, list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Run one independent asset in a spawned process with a recorded seed."""
    os.environ.setdefault("PIPELINE_MODEL_THREADS", str(MODEL_THREADS))
    os.environ.setdefault("PIPELINE_TORCH_THREADS", str(TORCH_THREADS))
    # These are inherited by spawned workers before their numerical libraries
    # load. They prevent nested BLAS/OpenMP pools from multiplying the process
    # pool's CPU demand and do not change any model parameter or data input.
    numeric_threads = os.environ.setdefault("PIPELINE_NUMERIC_THREADS", "1")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, numeric_threads)
    worker_seed_registry = set_global_seed(seed)
    frame, features, thresholds = run_company(company, run_id=run_id, progress_path=progress_path)
    return company, frame, features, thresholds, worker_seed_registry


def _feature_stability(feature_log: list[dict[str, Any]]) -> pd.DataFrame:
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for item in feature_log:
        company = str(item["company"])
        totals[company] += 1
        for feature in item.get("selected_features", []):
            counts[(company, feature)] += 1
    rows = [
        {"Company": company, "Feature": feature, "Selection Count": count, "Selection Rate (%)": count / totals[company] * 100.0}
        for (company, feature), count in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def _summaries(test_forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []
    for (company, model), group in test_forecasts.groupby(["Company", "Model"]):
        metrics_rows.append({"Company": company, "Model": model, **forecast_metrics(group["Actual"], group["Forecast"])})
        interval_rows.append({"Company": company, "Model": model, **interval_metrics(group["Actual"], group["Forecast"], group["Interval Width"], CONFORMAL_ALPHA)})
    for company, group in test_forecasts.groupby("Company"):
        proposed = group[group["Model"] == "VMD-LASSO-ARX/LSTM"].set_index("Date")
        if proposed.empty:
            continue
        for model, benchmark in group[group["Model"] != "VMD-LASSO-ARX/LSTM"].groupby("Model"):
            merged = proposed[["Actual", "Forecast"]].join(benchmark.set_index("Date")[["Forecast"]], how="inner", lsuffix="_proposed", rsuffix="_benchmark")
            statistic, pvalue = dm_test(
                (merged["Forecast_proposed"] - merged["Actual"]) ** 2 - (merged["Forecast_benchmark"] - merged["Actual"]) ** 2,
                horizon=1,
            )
            dm_rows.append({"Company": company, "Proposed Model": "VMD-LASSO-ARX/LSTM", "Benchmark": model, "DM Statistic": statistic, "p-value": pvalue})
    return pd.DataFrame(metrics_rows), pd.DataFrame(interval_rows), pd.DataFrame(dm_rows)


def _trading_summary(test_forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (company, model), group in test_forecasts.groupby(["Company", "Model"]):
        group = group.sort_values("Date")
        if FACTOR_FILE is not None:
            if not FACTOR_FILE.exists():
                raise FileNotFoundError(f"Configured factor file does not exist: {FACTOR_FILE}")
            factor_path = FACTOR_FILE
        else:
            region = FACTOR_REGION_BY_COMPANY.get(company)
            factor_path = FACTOR_DIR / f"{region}_ff5.csv" if region is not None else None
        if factor_path is not None and factor_path.exists():
            factor_data = load_factor_data(factor_path)
            factor_status = f"Evaluated from {factor_path.name} ({factor_data.attrs.get('frequency', 'unknown')})"
        else:
            factor_data = None
            factor_status = "Not evaluated: documented regional factor file is unavailable"
        for cost in TRANSACTION_COSTS:
            for use_filter, label in ((True, "Uncertainty-adjusted"), (False, "Directional (no uncertainty)")):
                result = evaluate_trading(
                    group["Origin Price"],
                    group["Actual"],
                    group["Forecast"],
                    group["Interval Width"],
                    threshold=group["Threshold"].values,
                    cost=cost,
                    use_uncertainty_filter=use_filter,
                    annualization=ANNUALIZATION,
                )
                row: dict[str, Any] = {
                        "Company": company,
                        "Model": model,
                        "Strategy": label,
                        "Cost": cost,
                        "Annualized Return": result.annualized_return,
                        "Sharpe": result.sharpe,
                        "Sortino": result.sortino,
                        "Calmar": result.calmar,
                        "Max Drawdown (%)": result.max_drawdown,
                        "Turnover": result.turnover,
                        "Hit Rate (%)": result.hit_rate,
                        "Final Equity": float(result.equity_curve[-1]) if len(result.equity_curve) else np.nan,
                        "Factor Status": factor_status,
                    }
                if factor_data is not None:
                    row.update(factor_adjusted_alphas(group["Date"], result.returns, factor_data, ANNUALIZATION))
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 72)
    print("CAUSAL ONE-STEP WALK-FORWARD EXPERIMENT")
    print("=" * 72)
    validate_raw_inputs()
    require_packages(REQUIRED_MODEL_PACKAGES)
    os.environ.setdefault("PIPELINE_MODEL_THREADS", str(MODEL_THREADS))
    os.environ.setdefault("PIPELINE_TORCH_THREADS", str(TORCH_THREADS))
    # Set native-kernel limits before spawning. Spawned workers inherit these
    # values before NumPy/PyTorch load their OpenMP/BLAS backends.
    numeric_threads = os.environ.setdefault("PIPELINE_NUMERIC_THREADS", "1")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, numeric_threads)
    seed_registry = set_global_seed(RANDOM_SEED)
    worker_count = min(MAX_PARALLEL_COMPANIES, len(ORDER))

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_log_dir = Path(OUT_ROOT) / "run_logs" / run_id
    run_log_dir.mkdir(parents=True, exist_ok=False)
    (run_log_dir / "run_context.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_utc": utc_timestamp(),
                "companies": ORDER,
                "calibration_size": CALIBRATION_SIZE,
                "test_fraction": TEST_FRAC,
                "model_refit_interval": MODEL_REFIT_INTERVAL,
                "progress_event_interval_origins": PROGRESS_EVENT_INTERVAL,
                "max_parallel_companies": worker_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Live progress logs: {run_log_dir}", flush=True)

    company_frames: dict[str, pd.DataFrame] = {}
    feature_log: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, float]] = {}
    company_seed_registry: dict[str, dict[str, Any]] = {}
    if worker_count == 1:
        for position, company in enumerate(ORDER):
            progress_path = run_log_dir / f"{SLUG[company]}.jsonl"
            company, frame, features, company_thresholds, company_seed = _run_company_worker(
                company, RANDOM_SEED + position, run_id, progress_path
            )
            company_frames[company] = frame
            feature_log.extend(features)
            thresholds[company] = company_thresholds
            company_seed_registry[company] = company_seed
    else:
        print(f"Running {len(ORDER)} independent company experiments with {worker_count} WSL workers", flush=True)
        context = get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _run_company_worker,
                    company,
                    RANDOM_SEED + position,
                    run_id,
                    run_log_dir / f"{SLUG[company]}.jsonl",
                ): company
                for position, company in enumerate(ORDER)
            }
            for future in as_completed(futures):
                company, frame, features, company_thresholds, company_seed = future.result()
                company_frames[company] = frame
                feature_log.extend(features)
                thresholds[company] = company_thresholds
                company_seed_registry[company] = company_seed
                print(f"Completed {company}", flush=True)

    all_frames = [company_frames[company] for company in ORDER]

    forecasts_all = pd.concat(all_frames, ignore_index=True)
    forecasts_test = forecasts_all[forecasts_all["Partition"] == "test"].copy()
    if forecasts_test.empty:
        raise RuntimeError("No final-test forecasts were generated")
    metrics, intervals, dm_results = _summaries(forecasts_test)
    trading = _trading_summary(forecasts_test)
    stability = _feature_stability(feature_log)

    # Outputs are written only after every company succeeds, preventing an
    # incomplete cross-section from being presented as a completed experiment.
    forecasts_all.to_csv(Path(OUT_DATA) / "walk_forward_forecasts_all.csv", index=False)
    forecasts_test.to_csv(Path(OUT_DATA) / "walk_forward_forecasts.csv", index=False)
    forecasts_test.rename(
        columns={
            "Date": "date",
            "Company": "asset",
            "Actual": "actual",
            "Forecast": "forecast",
            "Lower": "lower",
            "Upper": "upper",
            "Model": "model",
        }
    ).to_csv(Path(OUT_FORECASTS) / "forecasts_canonical.csv", index=False)
    metrics.to_csv(Path(OUT_DATA) / "walk_forward_metrics.csv", index=False)
    metrics.to_csv(Path(OUT_DATA) / "model_comparison.csv", index=False)
    intervals.to_csv(Path(OUT_DATA) / "interval_summary.csv", index=False)
    dm_results.to_csv(Path(OUT_DATA) / "dm_test_results.csv", index=False)
    trading.to_csv(Path(OUT_DATA) / "trading_results.csv", index=False)
    stability.to_csv(Path(OUT_DATA) / "feature_stability.csv", index=False)
    with open(Path(OUT_DATA) / "seed_registry.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "orchestrator": seed_registry,
                "company_workers": company_seed_registry,
                "run_id": run_id,
                "progress_log_directory": str(run_log_dir),
            },
            handle,
            indent=2,
        )
    with open(Path(OUT_DATA) / "experiment_registry.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "forecast_horizon": 1,
                "test_fraction": TEST_FRAC,
                "calibration_size": CALIBRATION_SIZE,
                "model_refit_interval": MODEL_REFIT_INTERVAL,
                "decomposition_refit_policy": "full VMD/CEEMDAN at each scheduled model refit",
                "between_refit_decomposition_state": "causal level-component innovation carry-forward",
                "rnn_epochs": RNN_EPOCHS,
                "model_threads": MODEL_THREADS,
                "torch_threads": TORCH_THREADS,
                "run_id": run_id,
                "progress_log_directory": str(run_log_dir),
                "conformal_alpha": CONFORMAL_ALPHA,
                "threshold_grid": list(THRESHOLD_GRID),
                "thresholds": thresholds,
                "models": sorted(forecasts_test["Model"].unique().tolist()),
                "raw_inputs": {label: str(path) for label, path in __import__("config").required_raw_inputs().items()},
                "factor_inputs": {label: str(path) for label, path in required_factor_inputs().items()},
            },
            handle,
            indent=2,
        )
    print("Saved causal forecasts, accuracy tests, interval metrics, and aligned trading results.")


if __name__ == "__main__":
    main()
