"""Run the Liu-style noncausal static protocol on two lithium prices.

The protocol intentionally preserves full-sample VMD, globally scaled LASSO
selection, a chronological 80:20 holdout, two ARIMA-routed low modes, and
sequence-major LSTMs for the remaining modes.  It is a retrospective static
prediction experiment, not a causal real-time forecast.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data_sources.prepare_lithium_inputs import build_panels
from evaluation.metrics import dm_test, forecast_metrics
from lithium_config import (
    ABLATIONS,
    FEATURE_BLOCKS,
    LIU_HORIZON,
    LIU_LAGS,
    LIU_LOW_COMPLEXITY_MODES,
    LIU_LSTM_BATCH_SIZE,
    LIU_LSTM_EPOCHS,
    LIU_LSTM_HIDDEN_SIZE,
    LIU_RANDOM_SEED,
    LIU_TEST_FRACTION,
    LITHIUM_MANIFEST,
    TARGETS,
)
from models.released_paper_shadow import (
    full_sample_vmd,
    released_arima_predictions,
    released_feature_frame,
    released_lasso_selection,
    released_lstm_predictions,
    released_pre_process,
)
from utils.progress import CompanyProgressLogger
from utils.reproducibility import require_packages, set_global_seed


OUTPUT_ROOT = CODE_ROOT.parent / "outputs" / "liu_lithium"
PROTOCOL_LABEL = "Liu-style full-sample VMD static 80:20 protocol (noncausal)"


def ablation_columns(ablation: str) -> tuple[str, ...]:
    if ablation not in ABLATIONS:
        raise KeyError(f"Unknown ablation {ablation!r}")
    columns: list[str] = []
    for block in ABLATIONS[ablation]:
        for column in FEATURE_BLOCKS[block]:
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def _load_panel(target: str) -> tuple[pd.Series, pd.DataFrame]:
    if target not in TARGETS:
        raise KeyError(f"Unknown lithium target {target!r}")
    path = Path(TARGETS[target]["processed"])
    if not path.exists():
        raise FileNotFoundError(f"Prepared lithium panel is missing: {path}")
    panel = pd.read_csv(path, index_col="Date", parse_dates=["Date"]).sort_index()
    if "Target_Close" not in panel:
        raise ValueError(f"{path} has no Target_Close column")
    price = pd.to_numeric(panel.pop("Target_Close"), errors="coerce").dropna()
    price = price[~price.index.duplicated(keep="last")]
    return price, panel.reindex(price.index)


def preflight(
    *,
    targets: list[str] | None = None,
    ablations: list[str] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    selected_targets = list(TARGETS) if targets is None else targets
    selected_ablations = list(ABLATIONS) if ablations is None else ablations
    build_panels(require_complete=require_complete, score_finbert=False)
    require_packages(
        {
            "pmdarima": "Liu-style auto-ARIMA",
            "statsmodels": "Liu-style ARIMA filtering",
            "torch": "Liu-style LSTM",
            "vmdpy": "full-sample VMD",
        }
    )
    report: dict[str, Any] = {"manifest": str(LITHIUM_MANIFEST), "targets": {}}
    for target in selected_targets:
        price, exogenous = _load_panel(target)
        target_report: dict[str, Any] = {
            "observations": int(len(price)),
            "start": str(price.index.min().date()),
            "end": str(price.index.max().date()),
            "modes": int(TARGETS[target]["modes"]),
            "ablations": {},
        }
        for ablation in selected_ablations:
            columns = ablation_columns(ablation)
            missing = [column for column in columns if column not in exogenous]
            if missing:
                raise ValueError(f"{target}/{ablation} lacks columns: {missing}")
            selected = exogenous.loc[:, list(columns)] if columns else pd.DataFrame(index=price.index)
            complete = pd.concat([price.rename("target"), selected], axis=1).dropna()
            if len(complete) < 320:
                raise ValueError(
                    f"{target}/{ablation} has only {len(complete)} complete rows; "
                    "at least 320 are required for the published LSTM batch/split design"
                )
            target_report["ablations"][ablation] = {
                "external_columns": list(columns),
                "complete_rows_before_component_lags": int(len(complete)),
                "first_complete_date": str(complete.index.min().date()),
                "last_complete_date": str(complete.index.max().date()),
            }
        report["targets"][target] = target_report
    return report


def _fit_target_ablation(
    *,
    target_name: str,
    ablation: str,
    price: pd.Series,
    exogenous: pd.DataFrame,
    components: np.ndarray,
    epochs: int,
    logger: CompanyProgressLogger,
    overall_completed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    expected_dates: pd.DatetimeIndex | None = None
    component_forecasts: list[np.ndarray] = []
    selection_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for mode_index, component_values in enumerate(components, start=1):
        component = pd.Series(component_values, index=price.index, name="target", dtype=float)
        frame = released_feature_frame(component, exogenous, horizon=LIU_HORIZON, lags=LIU_LAGS)
        selection = released_lasso_selection(frame)
        selected_frame = frame[["target", *selection.selected_columns]]
        x_train, y_train, x_test, y_test, target_scaler, split = released_pre_process(
            selected_frame, test_size=LIU_TEST_FRACTION
        )
        test_dates = pd.DatetimeIndex(selected_frame.index[split.test_start :])
        if expected_dates is None:
            expected_dates = test_dates
        elif not expected_dates.equals(test_dates):
            raise RuntimeError(f"{target_name}/{ablation} component holdout dates are not aligned")

        if mode_index <= LIU_LOW_COMPLEXITY_MODES:
            route = "ARIMA observed-test apply path"
            component_prediction, arima_order = released_arima_predictions(x_train, y_train, y_test, target_scaler)
        else:
            route = "LSTM source sequence-major layout"
            component_prediction = released_lstm_predictions(
                x_train,
                y_train,
                x_test,
                target_scaler,
                hidden_size=LIU_LSTM_HIDDEN_SIZE,
                epochs=epochs,
                batch_size=LIU_LSTM_BATCH_SIZE,
            )
            arima_order = None
        component_forecasts.append(component_prediction)
        component_rows.extend(
            {
                "Target": target_name,
                "Ablation": ablation,
                "Mode": mode_index,
                "Route": route,
                "Date": date,
                "Actual Component": float(actual),
                "Predicted Component": float(predicted),
            }
            for date, actual, predicted in zip(
                test_dates,
                component.loc[test_dates].to_numpy(dtype=float),
                component_prediction,
            )
        )
        selection_rows.append(
            {
                "Target": target_name,
                "Ablation": ablation,
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
        overall_completed += 1
        logger.record(
            "running",
            partition=ablation,
            partition_completed=mode_index,
            partition_total=len(components),
            overall_completed=overall_completed,
            target_date=str(test_dates[-1].date()),
            detail=f"{ablation}: mode {mode_index}/{len(components)} via {route}",
        )
        print(f"  {target_name}/{ablation}: mode {mode_index}/{len(components)} complete", flush=True)

    if expected_dates is None:
        raise RuntimeError(f"{target_name}/{ablation} produced no component forecasts")
    aggregate = np.sum(np.vstack(component_forecasts), axis=0)
    scored_dates = expected_dates[1:]
    actual = price.reindex(scored_dates).to_numpy(dtype=float)
    forecast = aggregate[1:]
    random_walk = price.shift(1).reindex(scored_dates).to_numpy(dtype=float)
    origin_dates = expected_dates[:-1]
    forecast_rows = [
        {
            "Target": target_name,
            "Ablation": ablation,
            "Protocol": PROTOCOL_LABEL,
            "Origin Date": origin_date,
            "Date": date,
            "Origin Price": float(origin),
            "Actual": float(observed),
            "Forecast": float(predicted),
            "Random Walk": float(baseline),
            "Valid Real-time Forecast": False,
        }
        for origin_date, date, origin, observed, predicted, baseline in zip(
            origin_dates,
            scored_dates,
            random_walk,
            actual,
            forecast,
            random_walk,
        )
    ]
    model_metrics = forecast_metrics(actual, forecast)
    random_walk_metrics = forecast_metrics(actual, random_walk)
    losses = (forecast - actual) ** 2 - (random_walk - actual) ** 2
    dm_statistic, dm_pvalue = dm_test(losses)
    summary = {
        "Target": target_name,
        "Ablation": ablation,
        "Protocol": PROTOCOL_LABEL,
        "Scored Rows": int(len(scored_dates)),
        **{f"Model {key}": value for key, value in model_metrics.items()},
        **{f"Random Walk {key}": value for key, value in random_walk_metrics.items()},
        "DM Statistic vs Random Walk": dm_statistic,
        "DM p-value vs Random Walk": dm_pvalue,
    }
    return forecast_rows, selection_rows + component_rows, summary, overall_completed


def run(
    *,
    targets: list[str],
    ablations: list[str],
    epochs: int = LIU_LSTM_EPOCHS,
    require_complete: bool = True,
) -> Path:
    readiness = preflight(targets=targets, ablations=ablations, require_complete=require_complete)
    os.environ.setdefault("PIPELINE_TORCH_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(max(1, int(os.environ["PIPELINE_TORCH_THREADS"])))
    except ImportError:
        pass
    seed_registry = set_global_seed(LIU_RANDOM_SEED)
    run_id = datetime.now(timezone.utc).strftime("liu_lithium_%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "preflight.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")

    all_forecasts: list[dict[str, Any]] = []
    all_selection: list[dict[str, Any]] = []
    all_components: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target_name in targets:
        price, full_exogenous = _load_panel(target_name)
        if len(price) % 2:
            price = price.iloc[1:]
            full_exogenous = full_exogenous.iloc[1:]
        modes = int(TARGETS[target_name]["modes"])
        target_total_steps = modes * len(ablations) + 1
        target_completed = 0
        logger = CompanyProgressLogger(
            run_dir / "logs" / f"{target_name}.jsonl",
            run_id=run_id,
            company=target_name,
            total_origins=target_total_steps,
        )
        logger.record("started", detail=f"{PROTOCOL_LABEL}; ETA updates after each completed mode")
        components = full_sample_vmd(price.to_numpy(dtype=float), modes=modes)
        target_completed += 1
        logger.record(
            "running",
            partition="full_sample_vmd",
            partition_completed=1,
            partition_total=1,
            overall_completed=target_completed,
            detail=f"Completed full-sample {modes}-mode VMD",
        )

        for ablation in ablations:
            columns = ablation_columns(ablation)
            exogenous = full_exogenous.loc[:, list(columns)] if columns else pd.DataFrame(index=price.index)
            forecasts, records, summary, target_completed = _fit_target_ablation(
                target_name=target_name,
                ablation=ablation,
                price=price,
                exogenous=exogenous,
                components=components,
                epochs=epochs,
                logger=logger,
                overall_completed=target_completed,
            )
            all_forecasts.extend(forecasts)
            summaries.append(summary)
            for record in records:
                if "Actual Component" in record:
                    all_components.append(record)
                else:
                    all_selection.append(record)
        logger.record("completed", overall_completed=target_completed, detail=f"Completed {target_name}")

    pd.DataFrame(all_forecasts).to_csv(run_dir / "static_holdout_forecasts.csv", index=False)
    pd.DataFrame(all_selection).to_csv(run_dir / "mode_feature_selection.csv", index=False)
    pd.DataFrame(all_components).to_csv(run_dir / "component_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(run_dir / "ablation_metrics.csv", index=False)
    registry = {
        "run_id": run_id,
        "protocol": PROTOCOL_LABEL,
        "classification": "noncausal static retrospective prediction; not real-time forecasting evidence",
        "targets": targets,
        "ablations": ablations,
        "parameters": {
            "test_fraction": LIU_TEST_FRACTION,
            "horizon": LIU_HORIZON,
            "lags": LIU_LAGS,
            "low_complexity_arima_modes": LIU_LOW_COMPLEXITY_MODES,
            "target_modes": {target: TARGETS[target]["modes"] for target in targets},
            "lstm_epochs": epochs,
            "lstm_hidden_size": LIU_LSTM_HIDDEN_SIZE,
            "lstm_batch_size": LIU_LSTM_BATCH_SIZE,
        },
        "known_noncausal_properties": [
            "VMD is fitted to the complete target series before the static split.",
            "LASSO standardization is fitted on the complete component feature frame.",
            "ARIMA filters observed test targets through apply(...).fittedvalues.",
        ],
        "seed_registry": seed_registry,
        "input_manifest": str(LITHIUM_MANIFEST),
        "publication_input_gate_enforced": require_complete,
    }
    (run_dir / "experiment_registry.json").write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    return run_dir


def _parse_selection(value: str, available: dict[str, Any], label: str) -> list[str]:
    if value == "all":
        return list(available)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="all", help="all or comma-separated target keys")
    parser.add_argument("--ablations", default="all", help="all or comma-separated ablation keys")
    parser.add_argument("--epochs", type=int, default=LIU_LSTM_EPOCHS)
    parser.add_argument("--preflight-only", action="store_true", help="Validate complete inputs and runtime without fitting models")
    parser.add_argument(
        "--allow-incomplete-inputs",
        action="store_true",
        help="Development smoke tests only; full publication runs reject missing Trends/news",
    )
    args = parser.parse_args()
    targets = _parse_selection(args.targets, TARGETS, "targets")
    ablations = _parse_selection(args.ablations, ABLATIONS, "ablations")
    if args.preflight_only:
        print(
            json.dumps(
                preflight(
                    targets=targets,
                    ablations=ablations,
                    require_complete=not args.allow_incomplete_inputs,
                ),
                indent=2,
            )
        )
        return
    output = run(
        targets=targets,
        ablations=ablations,
        epochs=args.epochs,
        require_complete=not args.allow_incomplete_inputs,
    )
    print(f"Saved Liu-style lithium run to {output}")


if __name__ == "__main__":
    main()
