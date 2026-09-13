"""Development-only ablation of causal paper-style VMD forecasting choices.

The four variants use exactly the same historical Glencore development dates.
The final date is capped at 2024-12-31 so the 2025-26 final-test period is not
used to choose a model direction.  The design isolates three choices:

* full exogenous inputs versus own-mode lag features only;
* two low-frequency ARIMA modes versus an all-LSTM route;
* a true 20-step temporal LSTM versus the source archive's length-one LSTM
  tensor shape.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    MIN_TRAIN_SIZE,
    OUT_ROOT,
    PAPER_STYLE_PILOT_LSTM_EPOCHS,
    PAPER_STYLE_PILOT_LSTM_HIDDEN,
    PAPER_STYLE_PILOT_LSTM_WINDOW,
    PAPER_STYLE_PILOT_REFIT_INTERVAL,
    PAPER_STYLE_PILOT_VMD_K,
    RANDOM_SEED,
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
from utils.progress import CompanyProgressLogger, utc_timestamp
from utils.reproducibility import require_packages, set_global_seed


@dataclass(frozen=True)
class AblationVariant:
    label: str
    low_modes: int
    use_exogenous: bool
    sequence_style: str
    purpose: str


VARIANTS = (
    AblationVariant(
        label="Hybrid / Temporal LSTM / Full Exog",
        low_modes=2,
        use_exogenous=True,
        sequence_style="temporal",
        purpose="reference causal paper-style architecture",
    ),
    AblationVariant(
        label="Hybrid / Temporal LSTM / Lag Only",
        low_modes=2,
        use_exogenous=False,
        sequence_style="temporal",
        purpose="isolates the marginal value of external factors",
    ),
    AblationVariant(
        label="All LSTM / Temporal / Full Exog",
        low_modes=0,
        use_exogenous=True,
        sequence_style="temporal",
        purpose="isolates the fixed two-ARIMA routing decision",
    ),
    AblationVariant(
        label="Hybrid / One-Step LSTM / Full Exog",
        low_modes=2,
        use_exogenous=True,
        sequence_style="single_step",
        purpose="tests the released source archive's length-one LSTM tensor shape",
    ),
)


def _development_slice(price: pd.Series, exog: pd.DataFrame, *, end: str, origins: int) -> tuple[pd.Series, pd.DataFrame, int, int]:
    end_timestamp = pd.Timestamp(end)
    eligible_positions = np.flatnonzero(price.index <= end_timestamp)
    if len(eligible_positions) == 0:
        raise ValueError(f"No observations exist on or before development end {end}")
    final_target_position = int(eligible_positions[-1])
    first_target_position = final_target_position - origins + 1
    if first_target_position < MIN_TRAIN_SIZE:
        raise ValueError("Not enough history before the requested development window")
    return price.iloc[: final_target_position + 1], exog.iloc[: final_target_position + 1], first_target_position, final_target_position


def run_ablation(
    *,
    company: str,
    development_end: str,
    development_origins: int,
    refit_interval: int,
    epochs: int,
    hidden: int,
    window: int,
) -> Path:
    validate_raw_inputs()
    require_packages({"statsmodels": "ARIMA", "torch": "LSTM", "vmdpy": "VMD"})
    os.environ.setdefault("PIPELINE_MODEL_THREADS", "1")
    os.environ.setdefault("PIPELINE_TORCH_THREADS", "1")
    seed_registry = set_global_seed(RANDOM_SEED)
    price_all, exog_all = _load_series(company)
    price, exog, first_target, final_target = _development_slice(
        price_all, exog_all, end=development_end, origins=development_origins
    )
    run_id = datetime.now(timezone.utc).strftime(f"paper_style_ablation_{company.lower().replace(' ', '_')}_%Y%m%dT%H%M%SZ")
    run_dir = Path(OUT_ROOT) / "paper_style_causal_ablation" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = CompanyProgressLogger(
        run_dir / "logs" / f"{company.lower().replace(' ', '_')}.jsonl",
        run_id=run_id,
        company=company,
        total_origins=development_origins * len(VARIANTS),
    )
    logger.record(
        "started",
        detail=(
            f"{development_origins} date origins x {len(VARIANTS)} variants; "
            f"development data ends {price.index[final_target].date()}"
        ),
    )
    engines = {
        variant.label: CausalPaperStyleEngine(
            vmd_k=PAPER_STYLE_PILOT_VMD_K,
            low_modes=variant.low_modes,
            refit_interval=refit_interval,
            vmd_alpha=VMD_ALPHA,
            vmd_tau=VMD_TAU,
            vmd_dc=VMD_DC,
            vmd_init=VMD_INIT,
            vmd_tol=VMD_TOL,
            lstm_window=window,
            lstm_hidden=hidden,
            lstm_epochs=epochs,
            lstm_sequence_style=variant.sequence_style,
            proposed_label=variant.label,
        )
        for variant in VARIANTS
    }
    records: list[dict[str, Any]] = []
    route_fits: dict[str, list[dict[str, Any]]] = {variant.label: [] for variant in VARIANTS}
    print(
        f"Development ablation: {company}; {development_origins} origins ending {price.index[final_target].date()}; "
        f"{len(VARIANTS)} variants",
        flush=True,
    )
    try:
        for number, target_position in enumerate(range(first_target, final_target + 1), start=1):
            train_y = price.iloc[:target_position]
            full_exog_train = exog.iloc[:target_position]
            origin_date = price.index[target_position - 1]
            target_date = price.index[target_position]
            actual = float(price.iloc[target_position])
            for variant in VARIANTS:
                engine = engines[variant.label]
                variant_exog = full_exog_train if variant.use_exogenous else pd.DataFrame(index=train_y.index)
                predictions, metadata = engine.forecast(train_y, variant_exog)
                prediction = float(predictions[variant.label])
                records.append(
                    {
                        "Company": company,
                        "Model": variant.label,
                        "Purpose": variant.purpose,
                        "Use Exogenous": variant.use_exogenous,
                        "Low ARIMA Modes": variant.low_modes,
                        "LSTM Sequence Style": variant.sequence_style,
                        "Origin Date": origin_date,
                        "Model Fit Origin Date": price.index[engine.last_fit_train_size - 1],
                        "Date": target_date,
                        "Origin Price": float(train_y.iloc[-1]),
                        "Actual": actual,
                        "Forecast": prediction,
                        "Error": prediction - actual,
                        "Scheduled Refit": engine.refit_this_origin,
                    }
                )
                if engine.refit_this_origin:
                    route_fits[variant.label].append({"target_date": str(target_date.date()), **metadata[variant.label]})
            if number % 5 == 0 or number == development_origins:
                completed = number * len(VARIANTS)
                refit_states = ", ".join(f"{engine.fit_count}" for engine in engines.values())
                print(f"  {company}: development {number}/{development_origins}; fit counts [{refit_states}]", flush=True)
                logger.record(
                    "running",
                    partition="development",
                    partition_completed=number,
                    partition_total=development_origins,
                    overall_completed=completed,
                    target_date=str(target_date.date()),
                    refit=any(engine.refit_this_origin for engine in engines.values()),
                    detail=f"variant fit counts: [{refit_states}]",
                )
        forecasts = pd.DataFrame(records)
        metrics_rows: list[dict[str, Any]] = []
        for model, group in forecasts.groupby("Model", sort=False):
            metrics_rows.append(
                {
                    "Company": company,
                    "Model": model,
                    "Purpose": group["Purpose"].iloc[0],
                    "Use Exogenous": group["Use Exogenous"].iloc[0],
                    "Low ARIMA Modes": group["Low ARIMA Modes"].iloc[0],
                    "LSTM Sequence Style": group["LSTM Sequence Style"].iloc[0],
                    **forecast_metrics(group["Actual"], group["Forecast"]),
                    "Mean Error": float(group["Error"].mean()),
                    "Scheduled Fits": int(group["Scheduled Refit"].sum()),
                }
            )
        metrics = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)
        reference = forecasts[forecasts["Model"] == VARIANTS[0].label].set_index("Date")
        comparison_rows: list[dict[str, Any]] = []
        for variant in VARIANTS[1:]:
            candidate = forecasts[forecasts["Model"] == variant.label].set_index("Date")
            aligned = reference.join(candidate[["Actual", "Forecast"]], lsuffix="_reference", rsuffix="_candidate", how="inner")
            losses = (aligned["Forecast_candidate"] - aligned["Actual_candidate"]) ** 2 - (
                aligned["Forecast_reference"] - aligned["Actual_reference"]
            ) ** 2
            statistic, pvalue = dm_test(losses)
            comparison_rows.append(
                {
                    "Reference": VARIANTS[0].label,
                    "Candidate": variant.label,
                    "DM Statistic": statistic,
                    "p-value": pvalue,
                    "Loss Difference Definition": "candidate squared loss - reference squared loss",
                }
            )
        comparisons = pd.DataFrame(comparison_rows)
        forecasts.to_csv(run_dir / "development_forecasts.csv", index=False)
        metrics.to_csv(run_dir / "development_metrics.csv", index=False)
        comparisons.to_csv(run_dir / "development_dm_comparisons.csv", index=False)
        (run_dir / "route_fits.json").write_text(json.dumps(route_fits, indent=2, default=str), encoding="utf-8")
        (run_dir / "experiment_registry.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "completed_utc": utc_timestamp(),
                    "company": company,
                    "purpose": "development-only architecture/input/routing ablation; not a final-test result",
                    "development_start_target_date": str(price.index[first_target].date()),
                    "development_end_target_date": str(price.index[final_target].date()),
                    "development_origins": development_origins,
                    "vmd_k": PAPER_STYLE_PILOT_VMD_K,
                    "refit_interval": refit_interval,
                    "lstm": {"hidden": hidden, "window": window, "epochs": epochs},
                    "causal_guarantees": {
                        "scaling": "selection and LSTM scalers fit only on the scheduled historical fit slice",
                        "decomposition": "full VMD only at scheduled historical fit origins; innovation carry-forward between refits",
                        "development_cap": str(pd.Timestamp(development_end).date()),
                    },
                    "variants": [variant.__dict__ for variant in VARIANTS],
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
            partition="development",
            partition_completed=development_origins,
            partition_total=development_origins,
            overall_completed=development_origins * len(VARIANTS),
            target_date=str(price.index[final_target].date()),
            detail="All development ablation outputs saved.",
        )
    except Exception as exc:
        logger.failed(f"{type(exc).__name__}: {exc}")
        raise
    return run_dir


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="Glencore")
    parser.add_argument("--development-end", default="2024-12-31")
    parser.add_argument("--development-origins", type=int, default=120)
    parser.add_argument("--refit-interval", type=int, default=PAPER_STYLE_PILOT_REFIT_INTERVAL)
    parser.add_argument("--epochs", type=int, default=PAPER_STYLE_PILOT_LSTM_EPOCHS)
    parser.add_argument("--hidden", type=int, default=PAPER_STYLE_PILOT_LSTM_HIDDEN)
    parser.add_argument("--window", type=int, default=PAPER_STYLE_PILOT_LSTM_WINDOW)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.development_origins < 20:
        raise ValueError("Development ablation requires at least 20 forecast origins")
    if min(args.refit_interval, args.epochs, args.hidden, args.window) < 1:
        raise ValueError("Refit interval, epochs, hidden size, and window must be positive")
    run_dir = run_ablation(
        company=args.company,
        development_end=args.development_end,
        development_origins=args.development_origins,
        refit_interval=args.refit_interval,
        epochs=args.epochs,
        hidden=args.hidden,
        window=args.window,
    )
    print(f"Saved development ablation to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
