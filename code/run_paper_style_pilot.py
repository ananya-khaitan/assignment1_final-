"""Run one causal paper-style VMD-ARIMA/LSTM pilot without touching canonical outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ANNUALIZATION,
    CONFORMAL_ALPHA,
    MIN_TRAIN_SIZE,
    OUT_ROOT,
    PAPER_STYLE_PILOT_CALIBRATION_SIZE,
    PAPER_STYLE_PILOT_COMPANY,
    PAPER_STYLE_PILOT_LOW_MODES,
    PAPER_STYLE_PILOT_LSTM_EPOCHS,
    PAPER_STYLE_PILOT_LSTM_HIDDEN,
    PAPER_STYLE_PILOT_LSTM_WINDOW,
    PAPER_STYLE_PILOT_REFIT_INTERVAL,
    PAPER_STYLE_PILOT_TEST_SIZE,
    PAPER_STYLE_PILOT_VMD_K,
    RANDOM_SEED,
    THRESHOLD_GRID,
    TRANSACTION_COSTS,
    VMD_ALPHA,
    VMD_DC,
    VMD_INIT,
    VMD_TAU,
    VMD_TOL,
    validate_raw_inputs,
)
from evaluation.metrics import dm_test, forecast_metrics
from models.paper_style import CausalPaperStyleEngine
from run_experiment import _load_series
from trading.signals import evaluate_trading
from uncertainty.intervals import conformal_width_from_residuals, interval_metrics
from utils.causal_pipeline import WalkForwardSplit, calibration_and_test_splits
from utils.progress import CompanyProgressLogger, utc_timestamp
from utils.reproducibility import require_packages, set_global_seed


def _record_partition(
    *,
    company: str,
    partition: str,
    price: pd.Series,
    exog: pd.DataFrame,
    splits: list[WalkForwardSplit],
    engine: CausalPaperStyleEngine,
    residual_history: dict[str, list[float]] | None,
    progress: CompanyProgressLogger,
    completed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, list[float]], list[dict[str, Any]]]:
    """Emit one-step records with sequential conformal widths."""

    history: dict[str, list[float]] = defaultdict(list)
    if residual_history is not None:
        history.update({model: list(errors) for model, errors in residual_history.items()})
    records: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    for number, split in enumerate(splits, start=1):
        train_y = price.iloc[: split.train_end]
        exog_train = exog.iloc[: split.train_end]
        origin_date = price.index[split.train_end - 1]
        target_date = price.index[split.train_end]
        predictions, metadata = engine.forecast(train_y, exog_train)
        for model, forecast in predictions.items():
            errors = history[model]
            width = conformal_width_from_residuals(errors, alpha=CONFORMAL_ALPHA) if len(errors) >= 2 else np.nan
            records.append(
                {
                    "Company": company,
                    "Partition": partition,
                    "Origin Date": origin_date,
                    "Model Fit Origin Date": price.index[engine.last_fit_train_size - 1],
                    "Date": target_date,
                    "Origin Price": float(train_y.iloc[-1]),
                    "Actual": float(price.iloc[split.train_end]),
                    "Model": model,
                    "Forecast": float(forecast),
                    "Interval Width": float(width),
                    "Lower": float(forecast - width) if np.isfinite(width) else np.nan,
                    "Upper": float(forecast + width) if np.isfinite(width) else np.nan,
                    "Predicted Return": float((forecast - train_y.iloc[-1]) / train_y.iloc[-1]),
                }
            )
            errors.append(float(price.iloc[split.train_end] - forecast))
        if engine.refit_this_origin:
            routes.append({"target_date": str(target_date.date()), **metadata[engine.proposed_label]})
        if number % 5 == 0 or number == len(splits):
            print(f"  {company}: {partition} {number}/{len(splits)} origins; fit count {engine.fit_count}", flush=True)
            progress.record(
                "running",
                partition=partition,
                partition_completed=number,
                partition_total=len(splits),
                overall_completed=completed_offset + number,
                target_date=str(target_date.date()),
                refit=engine.refit_this_origin,
                detail=f"scheduled VMD fits completed: {engine.fit_count}",
            )
    return records, history, routes


def _tune_thresholds(calibration: pd.DataFrame) -> dict[str, float]:
    """Tune only uncertainty thresholds on the historical calibration slice."""

    thresholds: dict[str, float] = {}
    for model, group in calibration.groupby("Model"):
        usable = group.dropna(subset=["Interval Width"])
        if len(usable) < 10:
            raise RuntimeError(f"Insufficient calibration residuals to tune threshold for {model}")
        candidates: list[tuple[float, float]] = []
        for threshold in THRESHOLD_GRID:
            result = evaluate_trading(
                usable["Origin Price"],
                usable["Actual"],
                usable["Forecast"],
                usable["Interval Width"],
                threshold=threshold,
                cost=TRANSACTION_COSTS[1],
                annualization=ANNUALIZATION,
            )
            score = result.sharpe if np.isfinite(result.sharpe) else -np.inf
            candidates.append((float(score), float(threshold)))
        thresholds[model] = sorted(candidates, key=lambda value: (-value[0], value[1]))[0][1]
    return thresholds


def _summaries(test_frame: pd.DataFrame, thresholds: dict[str, float], proposed_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    trading_rows: list[dict[str, Any]] = []
    for model, group in test_frame.groupby("Model"):
        metric_rows.append({"Company": group["Company"].iloc[0], "Model": model, **forecast_metrics(group["Actual"], group["Forecast"])})
        usable = group.dropna(subset=["Interval Width"])
        if not usable.empty:
            interval_rows.append(
                {
                    "Company": group["Company"].iloc[0],
                    "Model": model,
                    **interval_metrics(usable["Actual"], usable["Forecast"], usable["Interval Width"], alpha=CONFORMAL_ALPHA),
                }
            )
        for cost in (0.0, TRANSACTION_COSTS[1]):
            for uncertainty_filter in (False, True):
                result = evaluate_trading(
                    group["Origin Price"],
                    group["Actual"],
                    group["Forecast"],
                    group["Interval Width"],
                    threshold=thresholds[model],
                    cost=cost,
                    use_uncertainty_filter=uncertainty_filter,
                    annualization=ANNUALIZATION,
                )
                trading_rows.append(
                    {
                        "Company": group["Company"].iloc[0],
                        "Model": model,
                        "Cost (bps)": cost * 10_000,
                        "Uncertainty Filter": uncertainty_filter,
                        "Threshold": thresholds[model],
                        "Sharpe": result.sharpe,
                        "Annualized Return": result.annualized_return,
                        "Max Drawdown (%)": result.max_drawdown,
                        "Turnover": result.turnover,
                        "Hit Rate (%)": result.hit_rate,
                    }
                )

    proposed = test_frame[test_frame["Model"] == proposed_label].set_index("Date")
    dm_rows: list[dict[str, Any]] = []
    for model, group in test_frame[test_frame["Model"] != proposed_label].groupby("Model"):
        aligned = proposed.join(group.set_index("Date")[["Actual", "Forecast"]], lsuffix="_proposed", rsuffix="_benchmark", how="inner")
        losses = (aligned["Forecast_proposed"] - aligned["Actual_proposed"]) ** 2 - (aligned["Forecast_benchmark"] - aligned["Actual_benchmark"]) ** 2
        statistic, pvalue = dm_test(losses)
        dm_rows.append(
            {
                "Company": proposed["Company"].iloc[0],
                "Proposed Model": proposed_label,
                "Benchmark": model,
                "DM Statistic": statistic,
                "p-value": pvalue,
                "Loss Difference Definition": "proposed squared loss - benchmark squared loss",
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(interval_rows), pd.DataFrame(trading_rows), pd.DataFrame(dm_rows)


def run_pilot(
    *,
    company: str,
    calibration_size: int,
    test_size: int,
    vmd_k: int,
    low_modes: int,
    refit_interval: int,
    lstm_epochs: int,
    lstm_hidden: int,
    lstm_window: int,
) -> Path:
    """Execute and save an isolated one-company causal paper-style pilot."""

    validate_raw_inputs()
    require_packages({"statsmodels": "ARIMA", "torch": "LSTM", "vmdpy": "VMD"})
    os.environ.setdefault("PIPELINE_MODEL_THREADS", "1")
    os.environ.setdefault("PIPELINE_TORCH_THREADS", "1")
    seed_registry = set_global_seed(RANDOM_SEED)
    price, exog = _load_series(company)
    calibration_splits, test_splits = calibration_and_test_splits(
        len(price), MIN_TRAIN_SIZE, calibration_size, test_size
    )

    run_id = datetime.now(timezone.utc).strftime(f"paper_style_causal_{company.lower().replace(' ', '_')}_%Y%m%dT%H%M%SZ")
    run_dir = Path(OUT_ROOT) / "paper_style_causal_pilot" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = CompanyProgressLogger(
        run_dir / "logs" / f"{company.lower().replace(' ', '_')}.jsonl",
        run_id=run_id,
        company=company,
        total_origins=len(calibration_splits) + len(test_splits),
    )
    logger.started(calibration_total=len(calibration_splits), test_total=len(test_splits))
    engine = CausalPaperStyleEngine(
        vmd_k=vmd_k,
        low_modes=low_modes,
        refit_interval=refit_interval,
        vmd_alpha=VMD_ALPHA,
        vmd_tau=VMD_TAU,
        vmd_dc=VMD_DC,
        vmd_init=VMD_INIT,
        vmd_tol=VMD_TOL,
        lstm_window=lstm_window,
        lstm_hidden=lstm_hidden,
        lstm_epochs=lstm_epochs,
    )
    print(
        f"Paper-style causal pilot: {company}; {len(calibration_splits)} calibration + {len(test_splits)} final-test origins",
        flush=True,
    )
    try:
        calibration_records, calibration_residuals, routes = _record_partition(
            company=company,
            partition="calibration",
            price=price,
            exog=exog,
            splits=calibration_splits,
            engine=engine,
            residual_history=None,
            progress=logger,
            completed_offset=0,
        )
        calibration = pd.DataFrame(calibration_records)
        thresholds = _tune_thresholds(calibration)
        test_records, _, test_routes = _record_partition(
            company=company,
            partition="test",
            price=price,
            exog=exog,
            splits=test_splits,
            engine=engine,
            residual_history=calibration_residuals,
            progress=logger,
            completed_offset=len(calibration_splits),
        )
        test = pd.DataFrame(test_records)
        all_forecasts = pd.concat([calibration, test], ignore_index=True)
        all_forecasts["Threshold"] = all_forecasts["Model"].map(thresholds)
        metrics, intervals, trading, dm_results = _summaries(test, thresholds, engine.proposed_label)
        all_forecasts.to_csv(run_dir / "forecasts_all.csv", index=False)
        test.to_csv(run_dir / "forecasts_test.csv", index=False)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        intervals.to_csv(run_dir / "interval_metrics.csv", index=False)
        trading.to_csv(run_dir / "trading_results.csv", index=False)
        dm_results.to_csv(run_dir / "dm_test_results.csv", index=False)
        (run_dir / "route_fits.json").write_text(json.dumps(routes + test_routes, indent=2, default=str), encoding="utf-8")
        (run_dir / "experiment_registry.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "started_and_completed_utc": utc_timestamp(),
                    "company": company,
                    "architecture": "causal VMD (8 modes) -> first two BIC-selected ARIMA, remaining LASSO-selected multivariate LSTM",
                    "paper_inspiration": "Liu et al. (2025); released full-sample VMD/global-scaler implementation deliberately not reused",
                    "forecast_horizon": 1,
                    "calibration_origins": len(calibration_splits),
                    "final_test_origins": len(test_splits),
                    "refit_interval": refit_interval,
                    "vmd_k": vmd_k,
                    "low_modes": low_modes,
                    "lstm": {"hidden": lstm_hidden, "window": lstm_window, "epochs": lstm_epochs},
                    "scaling_policy": "all LASSO and LSTM scalers fit only on the scheduled historical fit slice",
                    "decomposition_policy": "full VMD only at scheduled historical fit origins; between fits, causal component innovation carry-forward",
                    "thresholds": thresholds,
                    "seed_registry": seed_registry,
                    "progress_log": str(logger.path),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.record(
            "completed",
            partition="test",
            partition_completed=len(test_splits),
            partition_total=len(test_splits),
            overall_completed=len(calibration_splits) + len(test_splits),
            target_date=str(test["Date"].max().date()),
            detail="Forecast, interval, trading, and comparison outputs saved.",
        )
    except Exception as exc:
        logger.failed(f"{type(exc).__name__}: {exc}")
        raise
    return run_dir


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default=PAPER_STYLE_PILOT_COMPANY)
    parser.add_argument("--calibration-size", type=int, default=PAPER_STYLE_PILOT_CALIBRATION_SIZE)
    parser.add_argument("--test-size", type=int, default=PAPER_STYLE_PILOT_TEST_SIZE)
    parser.add_argument("--vmd-k", type=int, default=PAPER_STYLE_PILOT_VMD_K)
    parser.add_argument("--low-modes", type=int, default=PAPER_STYLE_PILOT_LOW_MODES)
    parser.add_argument("--refit-interval", type=int, default=PAPER_STYLE_PILOT_REFIT_INTERVAL)
    parser.add_argument("--epochs", type=int, default=PAPER_STYLE_PILOT_LSTM_EPOCHS)
    parser.add_argument("--hidden", type=int, default=PAPER_STYLE_PILOT_LSTM_HIDDEN)
    parser.add_argument("--window", type=int, default=PAPER_STYLE_PILOT_LSTM_WINDOW)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.calibration_size < 10 or args.test_size < 10:
        raise ValueError("The pilot needs at least 10 calibration and 10 final-test origins")
    if args.refit_interval < 1 or args.epochs < 1 or args.hidden < 1 or args.window < 2:
        raise ValueError("Refit interval, epochs, hidden size, and window must be positive")
    run_dir = run_pilot(
        company=args.company,
        calibration_size=args.calibration_size,
        test_size=args.test_size,
        vmd_k=args.vmd_k,
        low_modes=args.low_modes,
        refit_interval=args.refit_interval,
        lstm_epochs=args.epochs,
        lstm_hidden=args.hidden,
        lstm_window=args.window,
    )
    print(f"Saved paper-style causal pilot to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
