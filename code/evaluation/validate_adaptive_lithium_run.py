"""Validate an adaptive lithium-paper run and write an evidence-based verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROPOSED = "Proposed Adaptive Decomposition-Routing Ensemble"
FIXED = "Fixed VMD-LASSO-ARIMA/LSTM"
REQUIRED_COMPARATORS = {
    "CEEMDAN-ARIMA",
    "VMD-ARIMA",
    "CEEMDAN-LSTM",
    "VMD-LSTM",
    FIXED,
}
SINGLE_COMPARATORS = {"ES", "ARIMA", "SVR", "RF", "MLP", "ELM", "LSTM"}
PUBLICATION_COMPARATORS = REQUIRED_COMPARATORS | SINGLE_COMPARATORS


def validate(run_dir: Path, *, write: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    metrics = pd.read_csv(run_dir / "forecast_metrics.csv")
    forecasts = pd.read_csv(run_dir / "forecast_ledger.csv", parse_dates=["Date", "Origin Date"])
    dm = pd.read_csv(run_dir / "dm_tests.csv")
    routes = pd.read_csv(run_dir / "adaptive_component_routes.csv")
    candidates = pd.read_csv(run_dir / "adaptive_candidate_search.csv")
    registry = json.loads((run_dir / "experiment_registry.json").read_text(encoding="utf-8"))

    targets = sorted(metrics["Target"].unique())
    if set(targets) != {"lithium_carbonate", "lithium_hydroxide"}:
        raise AssertionError(f"Unexpected targets: {targets}")
    required = PUBLICATION_COMPARATORS | {PROPOSED}
    for target in targets:
        models = set(metrics.loc[metrics["Target"] == target, "Model"])
        missing = required - models
        if missing:
            raise AssertionError(f"{target} lacks required models: {sorted(missing)}")
    if forecasts.duplicated(["Target", "Model", "Date"]).any():
        raise AssertionError("Duplicate target/model/date rows in forecast ledger")
    counts = forecasts.groupby(["Target", "Model"]).size()
    if counts.groupby(level=0).nunique().max() != 1:
        raise AssertionError("Models do not share a common scored date count within target")
    if routes.empty or candidates.empty:
        raise AssertionError("Adaptive route or candidate ledger is empty")
    if set(routes["Selected Route"]) - {"ARIMA", "LSTM", "SVR", "RF"}:
        raise AssertionError("Unexpected adaptive component route")
    if registry.get("adaptive_protocol", {}).get("locked_test") != "final 20% of the complete aligned sample":
        raise AssertionError("Registry does not document the locked final test")

    comparisons: list[dict[str, Any]] = []
    strict_metric_wins = []
    for target in targets:
        target_metrics = metrics[metrics["Target"] == target].set_index("Model")
        proposed = target_metrics.loc[PROPOSED]
        comparator_metrics = target_metrics.loc[sorted(PUBLICATION_COMPARATORS)]
        best_rmse_model = str(comparator_metrics["RMSE"].idxmin())
        best_mae_model = str(comparator_metrics["MAE"].idxmin())
        best_rmse = float(comparator_metrics.loc[best_rmse_model, "RMSE"])
        best_mae = float(comparator_metrics.loc[best_mae_model, "MAE"])
        rmse_improvement = 100.0 * (best_rmse - float(proposed["RMSE"])) / best_rmse
        mae_improvement = 100.0 * (best_mae - float(proposed["MAE"])) / best_mae
        strict = bool(float(proposed["RMSE"]) < best_rmse and float(proposed["MAE"]) < best_mae)
        strict_metric_wins.append(strict)
        comparisons.append(
            {
                "target": target,
                "proposed_rmse": float(proposed["RMSE"]),
                "best_comparator_rmse": best_rmse,
                "best_comparator_rmse_model": best_rmse_model,
                "rmse_improvement_percent": rmse_improvement,
                "proposed_mae": float(proposed["MAE"]),
                "best_comparator_mae": best_mae,
                "best_comparator_mae_model": best_mae_model,
                "mae_improvement_percent": mae_improvement,
                "beats_all_required_comparators_on_rmse_and_mae": strict,
            }
        )

    proposed_dm = dm[(dm["Row Model"] == PROPOSED) | (dm["Column Model"] == PROPOSED)].copy()
    if proposed_dm.empty:
        raise AssertionError("No proposed-model DM comparisons")
    significant_wins = 0
    significant_losses = 0
    for row in proposed_dm.itertuples(index=False):
        left = getattr(row, "_1")
        statistic = float(getattr(row, "_3"))
        p_value = float(getattr(row, "_4"))
        proposed_difference = statistic if left == PROPOSED else -statistic
        if p_value < 0.05 and proposed_difference < 0:
            significant_wins += 1
        elif p_value < 0.05 and proposed_difference > 0:
            significant_losses += 1

    verdict = {
        "run_id": registry["run_id"],
        "protocol_classification": registry["classification"],
        "targets": comparisons,
        "strict_dual_target_accuracy_gate": bool(all(strict_metric_wins)),
        "proposed_dm_significant_wins": significant_wins,
        "proposed_dm_significant_losses": significant_losses,
        "claim": (
            "The adaptive model clears the pre-specified dual-target RMSE-and-MAE gate."
            if all(strict_metric_wins)
            else "The adaptive model does not clear the pre-specified dual-target RMSE-and-MAE gate; claims must remain target- and metric-specific."
        ),
    }
    if write:
        (run_dir / "adaptive_validation_report.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
