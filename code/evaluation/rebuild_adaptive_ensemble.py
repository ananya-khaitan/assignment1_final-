"""Rebuild adaptive ensemble weights and downstream ledgers from checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from ..models.adaptive_routing import _convex_weights
    from ..run_lithium_paper_pipeline import _inferential_and_trading_ledgers, _metric_row
except ImportError:  # pragma: no cover - direct execution path.
    from models.adaptive_routing import _convex_weights
    from run_lithium_paper_pipeline import _inferential_and_trading_ledgers, _metric_row


PROPOSED = "Proposed Adaptive Decomposition-Routing Ensemble"


def rebuild(run_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    forecasts = pd.read_csv(run_dir / "forecast_ledger.csv", parse_dates=["Date", "Origin Date"])
    registry_path = run_dir / "experiment_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selection_by_target = {entry["target"]: entry for entry in registry["adaptive_selection"]}

    for target in sorted(forecasts["Target"].unique()):
        checkpoint_root = run_dir / "adaptive_checkpoints" / target
        labels = list(selection_by_target[target]["stacking_weights"])
        development, test = [], []
        development_dates = test_dates = None
        for label in labels:
            arrays = np.load(checkpoint_root / f"{label}.npz")
            metadata = json.loads((checkpoint_root / f"{label}.json").read_text(encoding="utf-8"))
            development.append(np.asarray(arrays["development"], dtype=float))
            test.append(np.asarray(arrays["test"], dtype=float))
            candidate_dev_dates = pd.DatetimeIndex(pd.to_datetime(metadata["development_dates"]))
            candidate_test_dates = pd.DatetimeIndex(pd.to_datetime(metadata["test_dates"]))
            if development_dates is None:
                development_dates, test_dates = candidate_dev_dates, candidate_test_dates
            elif not development_dates.equals(candidate_dev_dates) or not test_dates.equals(candidate_test_dates):
                raise AssertionError(f"Checkpoint dates are not aligned for {target}")
        target_rows = forecasts[(forecasts["Target"] == target) & (forecasts["Model"] == PROPOSED)].sort_values("Date")
        if not pd.DatetimeIndex(target_rows["Date"]).equals(test_dates[1:]):
            raise AssertionError(f"Forecast ledger and checkpoint dates differ for {target}")
        panel = pd.read_csv(
            Path(__file__).resolve().parents[1] / "data" / "lithium" / "processed" / f"{target}_liu_panel.csv",
            index_col="Date",
            parse_dates=["Date"],
        )
        actual_development = panel["Target_Close"].reindex(development_dates).to_numpy(float)
        weights = _convex_weights(actual_development, np.vstack(development))
        ensemble = weights @ np.vstack(test)
        mask = (forecasts["Target"] == target) & (forecasts["Model"] == PROPOSED)
        ordered_index = forecasts.loc[mask].sort_values("Date").index
        forecasts.loc[ordered_index, "Forecast"] = ensemble[1:]
        selection_by_target[target]["stacking_weights"] = {
            label: float(weight) for label, weight in zip(labels, weights)
        }

    forecasts.to_csv(run_dir / "forecast_ledger.csv", index=False)
    metrics = pd.DataFrame(
        [
            _metric_row(target, model, str(frame["Model Class"].iloc[0]), frame["Actual"].to_numpy(), frame["Forecast"].to_numpy())
            for (target, model), frame in forecasts.groupby(["Target", "Model"])
        ]
    )
    dm, intervals, interval_summary, trading = _inferential_and_trading_ledgers(forecasts)
    metrics.to_csv(run_dir / "forecast_metrics.csv", index=False)
    dm.to_csv(run_dir / "dm_tests.csv", index=False)
    intervals.to_csv(run_dir / "interval_ledger.csv", index=False)
    interval_summary.to_csv(run_dir / "interval_metrics.csv", index=False)
    trading.to_csv(run_dir / "trading_metrics.csv", index=False)
    registry["adaptive_selection"] = [selection_by_target[target] for target in registry["targets"]]
    registry.setdefault("numerical_repairs", []).append(
        "Convex stacking loss was normalized by target scale; candidate predictions and development/test boundaries were unchanged."
    )
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(rebuild(args.run_dir))


if __name__ == "__main__":
    main()
