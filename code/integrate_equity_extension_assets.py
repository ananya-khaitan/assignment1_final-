"""Add validated ALB and Ganfeng extension evidence to reviewed paper assets."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROPOSED = "Proposed Adaptive Decomposition-Routing Ensemble"
LABELS = {"albemarle_equity": "Albemarle (ALB)", "ganfeng_equity": "Ganfeng Lithium (002460.SZ)"}
COLORS = {"actual": "#0B2545", "proposed": "#9B1C1C", "random_walk": "#2E74B5"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shutil.copytree(args.base, args.output)
    metrics = pd.read_csv(args.run / "forecast_metrics.csv")
    dm = pd.read_csv(args.run / "dm_tests.csv")
    forecasts = pd.read_csv(args.run / "forecast_ledger.csv", parse_dates=["Date"])
    rows = []
    for target, label in LABELS.items():
        block = metrics[metrics["Target"] == target].set_index("Model")
        proposed, vmd, rw = block.loc[PROPOSED], block.loc["VMD-ARIMA"], block.loc["Random Walk"]
        test = dm[(dm["Target"] == target) & (dm["Row Model"] == PROPOSED) & (dm["Column Model"] == "Random Walk")].iloc[0]
        rows.append({
            "Equity target": label,
            "AMDR-Li RMSE": proposed["RMSE"],
            "VMD-ARIMA RMSE": vmd["RMSE"],
            "Random Walk RMSE": rw["RMSE"],
            "RMSE improvement vs Random Walk (%)": 100 * (rw["RMSE"] - proposed["RMSE"]) / rw["RMSE"],
            "AMDR-Li MAE": proposed["MAE"],
            "AMDR-Li directional accuracy (%)": proposed["Directional Accuracy"],
            "DM p-value vs Random Walk": test["p-value"],
        })
    table = pd.DataFrame(rows)
    for column in table.columns[1:]:
        if "p-value" in column:
            table[column] = table[column].map(lambda x: f"{x:.6f}")
        else:
            table[column] = table[column].map(lambda x: f"{x:.3f}")
    table.to_csv(args.output / "tables" / "table13.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.5))
    for axis, (target, label) in zip(axes, LABELS.items()):
        data = forecasts[(forecasts["Target"] == target) & (forecasts["Model"].isin([PROPOSED, "Random Walk"]))].copy()
        dates = sorted(data["Date"].unique())[-90:]
        data = data[data["Date"].isin(dates)]
        actual = data.drop_duplicates("Date").sort_values("Date")
        axis.plot(actual["Date"], actual["Actual"], color=COLORS["actual"], lw=1.5, label="Actual")
        for model, color, name in [(PROPOSED, COLORS["proposed"], "AMDR-Li"), ("Random Walk", COLORS["random_walk"], "Random Walk")]:
            piece = data[data["Model"] == model].sort_values("Date")
            axis.plot(piece["Date"], piece["Forecast"], color=color, lw=1.0, label=name)
        metric = metrics[(metrics["Target"] == target) & (metrics["Model"] == PROPOSED)].iloc[0]
        axis.set_title(f"{label}: final 90 evaluation observations (AMDR-Li RMSE={metric['RMSE']:.3f})", fontsize=10)
        axis.grid(alpha=.18); axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "figures" / "fig14_equity_forecasts.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    ledger = args.output / "appendix_ledgers" / "equity_extension"
    ledger.mkdir()
    for source in args.run.glob("*.csv"):
        shutil.copy2(source, ledger / source.name)
    shutil.copy2(args.run / "experiment_registry.json", ledger / "experiment_registry.json")
    manifest_path = args.output / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["equity_extension"] = {"run_id": json.loads((args.run / "experiment_registry.json").read_text())["run_id"], "targets": list(LABELS.values())}
    manifest["tables"]["Table 13"] = "Static 80:20 forecasting evidence for the lithium-producer equity extension."
    manifest["figures"]["Figure 14"] = "Testing-set equity forecast paths for AMDR-Li and the Random Walk benchmark."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
