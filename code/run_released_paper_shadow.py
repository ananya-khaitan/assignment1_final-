"""Run a noncausal, release-faithful shadow of Liu et al. (2025) on one company.

This runner exists to diagnose why the released carbon-market implementation
can achieve strong static-holdout scores.  It is deliberately separate from
the causal critical-minerals experiment and writes only under
``outputs/released_paper_shadow``.  Its outputs are not valid real-time
forecasting or trading evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both ``python code/...`` and package imports.
    from config import OUT_ROOT, RANDOM_SEED, validate_raw_inputs
    from evaluation.metrics import dm_test, forecast_metrics
    from models.released_paper_shadow import (
        full_sample_vmd,
        released_arima_predictions,
        released_feature_frame,
        released_lasso_selection,
        released_lstm_predictions,
        released_pre_process,
    )
    from run_experiment import _load_series
    from utils.progress import CompanyProgressLogger, utc_timestamp
    from utils.reproducibility import require_packages, set_global_seed
except ImportError:  # pragma: no cover - package execution convenience.
    from .config import OUT_ROOT, RANDOM_SEED, validate_raw_inputs
    from .evaluation.metrics import dm_test, forecast_metrics
    from .models.released_paper_shadow import (
        full_sample_vmd,
        released_arima_predictions,
        released_feature_frame,
        released_lasso_selection,
        released_lstm_predictions,
        released_pre_process,
    )
    from .run_experiment import _load_series
    from .utils.progress import CompanyProgressLogger, utc_timestamp
    from .utils.reproducibility import require_packages, set_global_seed


PROTOCOL_LABEL = "Released-paper-style shadow (noncausal static holdout)"


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def _prepare_source_sample(price: pd.Series, exogenous: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, bool]:
    """Mirror ``load_y``'s unconditional removal of the first odd observation."""

    prepared_price = pd.Series(price, dtype=float).copy()
    prepared_exog = pd.DataFrame(exogenous).reindex(prepared_price.index).copy()
    removed_first_observation = bool(len(prepared_price) % 2)
    if removed_first_observation:
        prepared_price = prepared_price.iloc[1:]
        prepared_exog = prepared_exog.iloc[1:]
    if prepared_price.empty:
        raise ValueError("No observations remain after source-compatible parity adjustment")
    return prepared_price, prepared_exog, removed_first_observation


def _component_record(
    *,
    mode: int,
    route: str,
    dates: pd.DatetimeIndex,
    actual_component: np.ndarray,
    forecast_component: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "Mode": mode,
            "Route": route,
            "Date": date,
            "Actual Component": float(actual),
            "Forecast Component": float(forecast),
        }
        for date, actual, forecast in zip(dates, actual_component, forecast_component)
    ]


def run_shadow(
    *,
    company: str,
    modes: int = 9,
    horizon: int = 1,
    lags: int = 5,
    test_size: float = 0.20,
    epochs: int = 100,
    hidden_size: int = 40,
    batch_size: int = 128,
) -> Path:
    """Execute the archived implementation's available path on one company."""

    if modes != 9:
        raise ValueError("The released paper shadow is locked to the archive's nine VMD modes")
    if horizon != 1 or lags != 5 or test_size != 0.20:
        raise ValueError("The released paper shadow is locked to h=1, lag=5, and a 20% static holdout")
    if min(epochs, hidden_size, batch_size) < 1:
        raise ValueError("epochs, hidden_size, and batch_size must be positive")
    validate_raw_inputs()
    require_packages({"pmdarima": "released auto-ARIMA", "statsmodels": "released ARIMA", "torch": "released LSTM", "vmdpy": "released VMD"})
    os.environ.setdefault("PIPELINE_TORCH_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(max(1, int(os.environ["PIPELINE_TORCH_THREADS"])))
    except ImportError:  # require_packages above turns this into a clear runtime error.
        pass
    seed_registry = set_global_seed(RANDOM_SEED)
    price, exogenous = _load_series(company)
    price, exogenous, parity_adjusted = _prepare_source_sample(price, exogenous)

    run_id = datetime.now(timezone.utc).strftime(f"released_paper_shadow_{_slug(company)}_%Y%m%dT%H%M%SZ")
    run_dir = Path(OUT_ROOT) / "released_paper_shadow" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = CompanyProgressLogger(
        run_dir / "logs" / f"{_slug(company)}.jsonl",
        run_id=run_id,
        company=company,
        total_origins=modes + 2,
    )
    logger.record(
        "started",
        detail=(
            f"release-faithful static shadow: full-sample VMD + {modes} mode fits; "
            "ETA is based on completed mode fits, not forecast origins"
        ),
    )
    print(
        f"Released-paper shadow: {company}; full-sample VMD, static 80:20 holdout, {modes} modes. "
        "This is intentionally noncausal.",
        flush=True,
    )

    completed = 0
    try:
        components = full_sample_vmd(price.to_numpy(), modes=modes)
        completed += 1
        logger.record(
            "running",
            partition="setup",
            partition_completed=1,
            partition_total=1,
            overall_completed=completed,
            detail="Completed full-sample VMD (uses observations from the final static holdout).",
        )

        expected_dates: pd.DatetimeIndex | None = None
        component_forecasts: list[np.ndarray] = []
        selection_rows: list[dict[str, Any]] = []
        route_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        for mode_index, component_values in enumerate(components, start=1):
            component = pd.Series(component_values, index=price.index, name="target", dtype=float)
            frame = released_feature_frame(component, exogenous, horizon=horizon, lags=lags)
            selection = released_lasso_selection(frame)
            selected_frame = frame[["target", *selection.selected_columns]]
            x_train, y_train, x_test, y_test, target_scaler, split = released_pre_process(selected_frame, test_size=test_size)
            test_dates = pd.DatetimeIndex(selected_frame.index[split.test_start :])
            if expected_dates is None:
                expected_dates = test_dates
            elif not expected_dates.equals(test_dates):
                raise RuntimeError("Component static-holdout dates are not aligned")

            if mode_index <= 2:
                route = "ARIMA (auto-selected order; observed-test apply path)"
                component_prediction, arima_order = released_arima_predictions(x_train, y_train, y_test, target_scaler)
            else:
                route = "LSTM (source sequence-major layout)"
                component_prediction = released_lstm_predictions(
                    x_train,
                    y_train,
                    x_test,
                    target_scaler,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                arima_order = None
            if len(component_prediction) != len(test_dates):
                raise RuntimeError(f"Mode {mode_index} prediction length does not match static-holdout dates")
            component_forecasts.append(component_prediction)
            component_rows.extend(
                _component_record(
                    mode=mode_index,
                    route=route,
                    dates=test_dates,
                    actual_component=component.loc[test_dates].to_numpy(),
                    forecast_component=component_prediction,
                )
            )
            selection_rows.append(
                {
                    "Mode": mode_index,
                    "Route": route,
                    "LASSO Alpha": selection.alpha,
                    "LASSO Validation MSE": selection.validation_mse,
                    "Selected Feature Count": len(selection.selected_columns),
                    "Selected Features": " | ".join(selection.selected_columns),
                    "Empty-selection Fallback": selection.used_empty_selection_fallback,
                    "ARIMA Order": None if arima_order is None else str(arima_order),
                }
            )
            route_rows.append(
                {
                    "mode": mode_index,
                    "route": route,
                    "selected_features": list(selection.selected_columns),
                    "lasso_alpha": selection.alpha,
                    "lasso_validation_mse": selection.validation_mse,
                    "arima_order": arima_order,
                    "input_train_rows": int(len(x_train)),
                    "input_test_rows": int(len(x_test)),
                }
            )
            completed += 1
            print(f"  {company}: completed mode {mode_index}/{modes} ({route})", flush=True)
            logger.record(
                "running",
                partition="mode_fit",
                partition_completed=mode_index,
                partition_total=modes,
                overall_completed=completed,
                target_date=str(test_dates[-1].date()),
                detail=f"Mode {mode_index}: {route}",
            )

        if expected_dates is None:
            raise RuntimeError("No component forecasts were produced")
        aggregate = np.sum(np.vstack(component_forecasts), axis=0)
        # The raw archive explicitly discards the first aggregate row.  Its
        # ``true_y`` is missing, so we align that repaired target to the same
        # remaining feature-frame dates.
        scored_dates = expected_dates[1:]
        actual = price.reindex(scored_dates).to_numpy(dtype=float)
        forecast = aggregate[1:]
        origin_price = price.shift(1).reindex(scored_dates).to_numpy(dtype=float)
        random_walk = origin_price.copy()
        forecast_rows = pd.DataFrame(
            {
                "Company": company,
                "Protocol": PROTOCOL_LABEL,
                "Date": scored_dates,
                "Origin Date": pd.DatetimeIndex(scored_dates) - pd.Timedelta(days=0),
                "Origin Price": origin_price,
                "Actual": actual,
                "Released Shadow Forecast": forecast,
                "Static Random Walk Diagnostic": random_walk,
                "Valid Real-time Forecast": False,
            }
        )
        forecast_rows["Origin Date"] = expected_dates[:-1].to_numpy()
        shadow_metrics = forecast_metrics(actual, forecast)
        random_walk_metrics = forecast_metrics(actual, random_walk)
        losses = (forecast - actual) ** 2 - (random_walk - actual) ** 2
        dm_statistic, dm_pvalue = dm_test(losses)
        metrics = pd.DataFrame(
            [
                {
                    "Company": company,
                    "Model": "Released-paper-style shadow",
                    "Evaluation": "Noncausal static 80:20 holdout; full-sample VMD; ARIMA filters observed test targets",
                    **shadow_metrics,
                },
                {
                    "Company": company,
                    "Model": "Static random walk diagnostic",
                    "Evaluation": "Causal one-day baseline calculated only for diagnostic comparison",
                    **random_walk_metrics,
                },
            ]
        )
        comparisons = pd.DataFrame(
            [
                {
                    "Company": company,
                    "Model A": "Released-paper-style shadow",
                    "Model B": "Static random walk diagnostic",
                    "DM Statistic": dm_statistic,
                    "p-value": dm_pvalue,
                    "Loss Difference Definition": "shadow squared loss - random-walk squared loss",
                    "Interpretation": "Descriptive only; invalid for forecasting inference because Model A is noncausal",
                }
            ]
        )
        forecast_rows.to_csv(run_dir / "static_holdout_forecasts.csv", index=False)
        pd.DataFrame(component_rows).to_csv(run_dir / "component_static_holdout_forecasts.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(run_dir / "mode_selection_and_routes.csv", index=False)
        metrics.to_csv(run_dir / "static_holdout_metrics.csv", index=False)
        comparisons.to_csv(run_dir / "descriptive_dm_comparison.csv", index=False)
        (run_dir / "route_metadata.json").write_text(json.dumps(route_rows, indent=2, default=str), encoding="utf-8")
        registry: dict[str, Any] = {
            "run_id": run_id,
            "completed_utc": utc_timestamp(),
            "company": company,
            "classification": "noncausal forensic shadow; never a reportable forecasting result",
            "source_protocol": {
                "VMD": {"scope": "full input sample", "K": 9, "alpha": 3000, "tau": 0, "DC": 0, "init": 1, "tol": 1e-6},
                "features": {"horizon": 1, "lags": 5, "external_features": list(exogenous.columns), "external_feature_shift": 1},
                "feature_selection": "global StandardScaler before static 80:20 outer split; inner chronological 80:20 LASSO validation",
                "static_holdout": 0.20,
                "routing": "modes 1-2 ARIMA; modes 3-9 LSTM",
                "LSTM": {"epochs": epochs, "batch_size": batch_size, "hidden_size": hidden_size, "dropout": 0.5, "layout": "(N, 1, features), batch_first=False"},
                "ARIMA": "pmdarima auto_arima with selected features to select order; statsmodels ARIMA on target only; apply(observed test targets).fittedvalues",
                "aggregate_first_row": "discarded, as in the raw archive",
            },
            "repairs_required_to_execute_archive": [
                "Set y_name='target', h=1, lag=5, test_size=0.20, K=9 from the repaired local copy.",
                "Construct df_vmd directly from VMD components because it is undefined in raw main_forecast.py.",
                "Implement missing pre_process using the repaired local copy: train-fitted StandardScaler for X and y, transformed static test set.",
                "Replace undefined arima_orders with the raw code's immediately preceding auto_arima-selected order.",
                "Flatten one-column target arrays for current pmdarima/statsmodels API compatibility.",
                "Use raw archive's test-target ARIMA apply/fittedvalues route, rather than the repaired copy's recursive forecast route.",
                "Resolve raw LSTM's missing drop argument and list hidden-size mismatch with the repaired copy's effective hidden_size=40 and dropout=0.5.",
                "Keep the strongest own lag only if LASSO selects no features; raw archive would otherwise fail to create an LSTM input.",
                "Do not write the archive's shared ./net.pth; in-memory evaluation is numerically equivalent and avoids overwriting a project file.",
                "Align undefined true_y to the retained static-holdout feature dates after the raw archive's first-row drop.",
            ],
            "known_noncausal_properties": [
                "VMD decomposes the entire sample before the static split, so every component includes future test-period information.",
                "LASSO's first scaler is fitted on the entire component sample before selection and the static split.",
                "The LSTM receives test-period own-component lag values; those components were obtained from full-sample VMD.",
                "ARIMA's apply path filters observed test targets rather than forecasting an unobserved target sequence.",
            ],
            "source_compatibility": {"dropped_first_odd_observation": parity_adjusted, "scored_rows": int(len(forecast_rows))},
            "seed_registry": seed_registry,
            "progress_log": str(logger.path),
        }
        (run_dir / "experiment_registry.json").write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
        completed += 1
        logger.record(
            "completed",
            partition="finalize",
            partition_completed=1,
            partition_total=1,
            overall_completed=completed,
            target_date=str(scored_dates[-1].date()),
            detail="Saved noncausal release-faithful shadow outputs and provenance registry.",
        )
    except Exception as exc:
        logger.failed(f"{type(exc).__name__}: {exc}", overall_completed=completed)
        raise
    return run_dir


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="Glencore")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output = run_shadow(
        company=args.company,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
    )
    print(f"Saved released-paper shadow to {output}", flush=True)


if __name__ == "__main__":
    main()
