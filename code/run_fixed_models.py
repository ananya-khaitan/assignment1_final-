"""Export fixed-model outputs from the causal walk-forward experiment.

`run_experiment.py` now creates every fixed-model forecast on the identical
one-step origins. This script intentionally does not rerun a second,
misaligned experiment; it turns those canonical records into the fixed-model
artifacts used by SPA/MCS analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import OUT_DATA
from evaluation.metrics import forecast_metrics


def run_fixed_models() -> None:
    source = Path(OUT_DATA) / "walk_forward_forecasts.csv"
    if not source.exists():
        raise FileNotFoundError("Run run_experiment.py before exporting fixed-model forecasts")
    forecasts = pd.read_csv(source, parse_dates=["Date", "Origin Date"])
    fixed = forecasts[forecasts["Model"] != "Dynamic Selected"].copy()
    if fixed.empty:
        raise RuntimeError("Canonical experiment output contains no fixed-model forecasts")
    fixed.to_csv(Path(OUT_DATA) / "fixed_model_forecasts_all.csv", index=False)
    for company, group in fixed.groupby("Company"):
        group.to_csv(Path(OUT_DATA) / f"fixed_model_forecasts_{company.replace(' ', '_')}.csv", index=False)
    metrics = [
        {"Company": company, "Model": model, **forecast_metrics(group["Actual"], group["Forecast"])}
        for (company, model), group in fixed.groupby(["Company", "Model"])
    ]
    pd.DataFrame(metrics).to_csv(Path(OUT_DATA) / "fixed_model_metrics.csv", index=False)
    print("Exported fixed-model forecasts from the canonical one-step run.")


if __name__ == "__main__":
    run_fixed_models()
