"""Run SPA and MCS tests on causal fixed-model forecasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import OUT_DATA
from evaluation.stat_tests import model_confidence_set, spa_test


def run_tests(forecast_csv: str | Path | None = None, out_dir: str | Path | None = None) -> None:
    out_dir = Path(out_dir or OUT_DATA)
    forecast_csv = Path(forecast_csv or out_dir / "fixed_model_forecasts_all.csv")
    if not forecast_csv.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_csv}")
    forecasts = pd.read_csv(forecast_csv, parse_dates=["Date"])
    for company, group in forecasts.groupby("Company"):
        actual = group.groupby("Date")["Actual"].first()
        wide = group.pivot(index="Date", columns="Model", values="Forecast").join(actual.rename("Actual"), how="inner").dropna()
        if "Random Walk" not in wide.columns:
            raise ValueError("SPA requires the pre-specified Random Walk benchmark")
        model_names = [name for name in wide.columns if name != "Actual"]
        losses = (wide[model_names].sub(wide["Actual"], axis=0) ** 2).to_numpy()
        benchmark_index = model_names.index("Random Walk")
        alternatives = [name for name in model_names if name != "Random Walk"]
        alternative_losses = np.column_stack([losses[:, model_names.index(name)] for name in alternatives])
        spa_pvalues = spa_test(losses[:, benchmark_index], alternative_losses)
        included, mcs_pvalues = model_confidence_set(losses)
        safe_name = company.replace(" ", "_")
        with open(out_dir / f"spa_pvalues_{safe_name}.json", "w", encoding="utf-8") as handle:
            json.dump(
                {"benchmark": "Random Walk", "alternatives": alternatives, "pvalues": spa_pvalues}, handle, indent=2
            )
        with open(out_dir / f"mcs_set_{safe_name}.json", "w", encoding="utf-8") as handle:
            named_pvalues = {model_names[int(index)]: value for index, value in mcs_pvalues.items()}
            json.dump(
                {
                    "size": 0.05,
                    "mcs_models": [model_names[index] for index in included],
                    "pvalues": named_pvalues,
                },
                handle,
                indent=2,
            )
    print("SPA and MCS tests completed.")


if __name__ == "__main__":
    run_tests()
