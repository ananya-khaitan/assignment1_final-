"""Create the 12 tables and 13 figures for the causal critical-mineral paper.

All empirical artifacts are derived from the most recent completed canonical
run under ``outputs/data``.  The script deliberately does not read the legacy
LaTex tables or July figures.  It also keeps the released-paper shadow out of
the paper's empirical results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.signal import find_peaks
from vmdpy import VMD

from config import EXOG_COLS, ORDER, OUT_DATA, RANDOM_SEED, VMD_ALPHA, VMD_DC, VMD_INIT, VMD_K, VMD_TAU, VMD_TOL
from decomposition.causal import approximate_entropy
from run_experiment import _load_series


PAPER_TITLE = "Causal Multimodal Forecasting and Trading Evaluation for Critical-Mineral Equities"
PROPOSED = "VMD-LASSO-ARX/LSTM"
FIGURE_CAPTIONS = {
    1: "Publication-record trend for a narrowly defined Crossref title query. This is a descriptive search snapshot, not a systematic review.",
    2: "Conceptual structure of the LSTM cell used as one candidate forecaster.",
    3: "Causal one-step framework. Every operation at a forecast origin has access only to information available before the target close.",
    4: "Normalized closing-price paths for the eight critical-mineral equities.",
    5: "Word cloud of predictors repeatedly selected by the causal ARX procedure. It is not a news-headline word cloud because raw headlines are not an input to this study.",
    6: "SF Fed news-sentiment index during the sample and its empirical distribution.",
    7: "Illustrative VMD components fitted only to Glencore's pre-test training segment; the display is not a full-sample decomposition.",
    8: "Approximate-entropy values for the Figure 7 training-only VMD components.",
    9: "Illustrative final-test forecasts for Glencore using single-model candidates.",
    10: "Cross-company mean final-test RMSE by model, normalized to the random-walk mean RMSE.",
    11: "Illustrative conformal interval forecasts for the proposed causal model on Glencore's final test set.",
    12: "Timing of the causal one-step trading evaluation.",
    13: "Cross-company mean Sharpe ratios for directional strategies at 10 bp transaction costs.",
}
TABLE_CAPTIONS = {
    1: "Representative research and methodological references informing the study.",
    2: "Actual parameter settings used in the causal empirical run.",
    3: "Predictors used in the causal multimodal feature frame.",
    4: "Descriptive statistics for the eight company closing-price series.",
    5: "Descriptive characteristics of training-only Glencore VMD components.",
    6: "Most frequently selected ARX predictors across scheduled causal refits.",
    7: "Final-test forecasting errors for single-model candidates, averaged across companies.",
    8: "Final-test forecasting errors for decomposition-based candidates, averaged across companies.",
    9: "Diebold-Mariano comparisons of the proposed model with each benchmark across companies.",
    10: "Final-test conformal interval quality, averaged across companies.",
    11: "Trading performance of decomposition-model forecasts under calibrated strategies, averaged across companies.",
    12: "Trading performance of single-model forecasts under calibrated strategies, averaged across companies.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _draw_predictor_word_cloud(axis: plt.Axes, frequency: dict[str, float]) -> None:
    """Draw a deterministic text word cloud without Pillow/font-version coupling."""

    ranked = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:30]
    if not ranked:
        axis.text(.5, .5, "No selected predictors recorded", ha="center", va="center")
        axis.axis("off")
        return
    maximum = max(value for _, value in ranked)
    # Three columns give multi-word predictors enough room to remain legible
    # in both DOCX and browser-print layouts.
    columns = 3
    rows = int(np.ceil(len(ranked) / columns))
    palette = ["#0B2545", "#1F4D78", "#2E74B5", "#5E92C4", "#6AA84F"]
    for rank, (word, count) in enumerate(ranked):
        row, column = divmod(rank, columns)
        x = (column + .5) / columns
        y = 1 - (row + .53) / rows
        size = 9 + 9 * np.sqrt(count / maximum)
        axis.text(x, y, textwrap.fill(word.replace("_", " "), width=16), ha="center", va="center", fontsize=size,
                  color=palette[rank % len(palette)], fontweight="bold" if rank < 5 else "normal")
    axis.set(xlim=(0, 1), ylim=(0, 1), title="Repeatedly selected causal predictors")
    axis.axis("off")


def _format_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return result


def _crossref_publication_trend(output: Path) -> pd.DataFrame:
    """Save a reproducible, explicitly non-systematic literature-query snapshot."""

    records: list[dict[str, Any]] = []
    for year in range(2011, datetime.now(timezone.utc).year + 1):
        params = {
            "query.title": "critical mineral forecasting",
            "filter": f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31",
            "rows": 0,
        }
        try:
            response = requests.get("https://api.crossref.org/works", params=params, timeout=30)
            response.raise_for_status()
            count = int(response.json()["message"]["total-results"])
            status = "ok"
        except Exception as exc:  # Network is auxiliary; preserve a visible record on failure.
            count = 0
            status = f"unavailable: {type(exc).__name__}"
        records.append({"Year": year, "Candidate records": count, "Query status": status})
    trend = pd.DataFrame(records)
    trend.to_csv(output / "literature_query_snapshot.csv", index=False)
    return trend


def _training_vmd() -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """Produce a descriptive VMD display with no final-test observations."""

    glencore, _ = _load_series("Glencore")
    pretest_stop = len(glencore) - int(len(glencore) * 0.20)
    train = glencore.iloc[:pretest_stop]
    components, _hat, _omega = VMD(
        train.to_numpy(dtype=float), alpha=VMD_ALPHA, tau=VMD_TAU, K=VMD_K, DC=VMD_DC, init=VMD_INIT, tol=VMD_TOL
    )
    # The VMD reference implementation drops one observation for an odd-length
    # signal.  Align the descriptive price display and summary statistics to
    # the component length rather than silently plotting mismatched arrays.
    train = train.iloc[-components.shape[1]:]
    component_rows: list[dict[str, Any]] = []
    entropies = [approximate_entropy(component) for component in components]
    cutoff = float(np.nanmedian(entropies))
    for number, (component, entropy) in enumerate(zip(components, entropies), start=1):
        peaks, _ = find_peaks(component)
        average_period = float(len(component) / len(peaks)) if len(peaks) else float("nan")
        component_rows.append(
            {
                "Mode": f"VMD {number}",
                "Variance Share (%)": float(np.var(component, ddof=1) / np.var(train.to_numpy(), ddof=1) * 100.0),
                "Average Peak Period (obs.)": average_period,
                "Approximate Entropy": float(entropy),
                "Illustrative Route": "LASSO-ARX" if entropy <= cutoff else "LSTM",
                "Training Observations": len(train),
            }
        )
    return train, components, pd.DataFrame(component_rows)


def _write_tables(output: Path, registry: dict[str, Any], metrics: pd.DataFrame, interval: pd.DataFrame, trading: pd.DataFrame, dm: pd.DataFrame, features: pd.DataFrame, vmd_table: pd.DataFrame) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    tables[1] = pd.DataFrame(
        [
            ["Dragomiretskiy and Zosso (2014)", "Variational mode decomposition", "Signal decomposition", "Introduces VMD as a variational signal-decomposition method."],
            ["Tibshirani (1996)", "Least absolute shrinkage and selection operator", "Feature selection", "Provides the LASSO regularization used for sparse predictor selection."],
            ["Hochreiter and Schmidhuber (1997)", "Long short-term memory", "Sequence modelling", "Introduces the recurrent architecture used by the LSTM candidates."],
            ["Diebold and Mariano (1995)", "Comparing predictive accuracy", "Forecast comparison", "Provides the forecast-loss comparison framework."],
            ["Liu et al. (2025)", "Multimodal carbon-price forecasting", "VMD, LASSO, ARIMA/LSTM and trading", "Methodological reference; this study changes the target domain and enforces a causal walk-forward evaluation."],
            ["This study", "Critical-mineral equities", "Causal one-step forecasting and trading", "Uses scheduled train-only decomposition, historical calibration, conformal intervals and transaction-cost evaluation."],
        ],
        columns=["Study", "Focus", "Methodological Contribution", "Relevance to This Paper"],
    )
    run_id = registry.get("run_id", "not recorded")
    tables[2] = pd.DataFrame(
        [
            ["Forecast horizon", registry.get("forecast_horizon"), "One trading day"],
            ["Final-test share", registry.get("test_fraction"), "Chronological final holdout"],
            ["Calibration observations", registry.get("calibration_size"), "Pre-test residual and threshold calibration"],
            ["Model refit interval", registry.get("model_refit_interval"), "Trading observations"],
            ["VMD components", VMD_K, "Train-only VMD at scheduled refits"],
            ["VMD alpha", VMD_ALPHA, "VMD penalty parameter"],
            ["RNN epochs", registry.get("rnn_epochs"), "LSTM/GRU candidate fitting"],
            ["Conformal alpha", registry.get("conformal_alpha"), "Two-sided interval nominal error"],
            ["Run identifier", run_id, "Provenance registry"],
        ],
        columns=["Parameter", "Value", "Role"],
    )
    indicator_roles = {
        "mvis_critical_minerals": ("MVIS Global Rare Earth / Strategic Metals index", "Sector benchmark"),
        "SP500": ("S&P 500", "Global equity-market condition"),
        "Shanghai_Index": ("Shanghai Composite", "China equity-market condition"),
        "Crude_Oil": ("Crude-oil futures close", "Energy-market condition"),
        "VIX": ("CBOE VIX", "Market uncertainty"),
        "US_Dollar_Index": ("US Dollar Index", "Foreign-exchange condition"),
        "Search_Index": ("Google Trends: rare earth", "Public attention"),
        "News_Sentiment": ("SF Fed news-sentiment index", "News sentiment"),
    }
    tables[3] = pd.DataFrame(
        [[name, *indicator_roles[name], "Lagged one trading day before use"] for name in EXOG_COLS], columns=["Variable", "Construct", "Feature Group", "Availability Treatment"]
    )
    desc_rows: list[dict[str, Any]] = []
    for company in ORDER:
        price, _ = _load_series(company)
        desc_rows.append(
            {
                "Company": company,
                "Observations": len(price),
                "Start": str(price.index.min().date()),
                "End": str(price.index.max().date()),
                "Mean Close": price.mean(),
                "Std. Dev.": price.std(ddof=1),
                "Minimum": price.min(),
                "Maximum": price.max(),
            }
        )
    tables[4] = pd.DataFrame(desc_rows)
    tables[5] = vmd_table
    top_features = (
        features.sort_values(["Company", "Selection Rate (%)", "Feature"], ascending=[True, False, True])
        .groupby("Company", as_index=False)
        .head(5)
        .rename(columns={"Selection Rate (%)": "Selection Rate (%)"})
    )
    tables[6] = top_features[["Company", "Feature", "Selection Count", "Selection Rate (%)"]]
    decomposition = metrics[metrics["Model"].str.contains("VMD|CEEMDAN", regex=True)].copy()
    single = metrics[~metrics["Model"].str.contains("VMD|CEEMDAN|Dynamic Selected", regex=True)].copy()
    for subset in (single, decomposition):
        subset["Within-Company RMSE Rank"] = subset.groupby("Company")["RMSE"].rank(method="average")
    aggregation = {
        "RMSE": "mean",
        "MAE": "mean",
        "MASE": "mean",
        "MAPE": "mean",
        "Directional Accuracy": "mean",
        "Within-Company RMSE Rank": "mean",
    }
    tables[7] = single.groupby("Model", as_index=False).agg(aggregation).sort_values("RMSE")
    tables[8] = decomposition.groupby("Model", as_index=False).agg(aggregation).sort_values("RMSE")
    summary_rows: list[dict[str, Any]] = []
    for benchmark, group in dm.groupby("Benchmark"):
        significant = group[group["p-value"] < 0.05]
        summary_rows.append(
            {
                "Benchmark": benchmark,
                "Companies": len(group),
                "Proposed Better (5%)": int(((significant["DM Statistic"] < 0)).sum()),
                "Proposed Worse (5%)": int(((significant["DM Statistic"] > 0)).sum()),
                "Median DM Statistic": group["DM Statistic"].median(),
                "Median p-value": group["p-value"].median(),
            }
        )
    tables[9] = pd.DataFrame(summary_rows).sort_values(["Proposed Better (5%)", "Proposed Worse (5%)"], ascending=[False, True])
    tables[10] = interval.groupby("Model", as_index=False).agg(
        {"Coverage (%)": "mean", "Avg Width": "mean", "Sharpness": "mean", "Calibration Error": "mean", "Winkler": "mean"}
    ).sort_values("Winkler")
    decomp_trading = trading[trading["Model"].str.contains("VMD|CEEMDAN", regex=True)].copy()
    single_trading = trading[~trading["Model"].str.contains("VMD|CEEMDAN|Dynamic Selected", regex=True)].copy()
    trading_cols = ["Annualized Return", "Sharpe", "Sortino", "Calmar", "Max Drawdown (%)", "Turnover", "Hit Rate (%)", "Final Equity"]
    tables[11] = decomp_trading.groupby(["Model", "Strategy", "Cost"], as_index=False)[trading_cols].mean().sort_values(["Model", "Strategy", "Cost"])
    # Keep the printed table readable at the empirically relevant 10 bp cost.
    # The complete four-cost ledger is embedded in the paper appendix and saved
    # beside the manuscript.
    tables[12] = single_trading[np.isclose(single_trading["Cost"], 0.001)].groupby(["Model", "Strategy", "Cost"], as_index=False)[trading_cols].mean().sort_values(["Model", "Strategy"])
    for number, table in tables.items():
        table.to_csv(output / "tables" / f"table{number}.csv", index=False)
    return tables


def _write_figures(output: Path, tables: dict[int, pd.DataFrame], metrics: pd.DataFrame, trading: pd.DataFrame, forecasts: pd.DataFrame, training: pd.Series, components: np.ndarray, features: pd.DataFrame) -> None:
    figure_dir = output / "figures"
    # Fig. 1
    trend = pd.read_csv(output / "literature_query_snapshot.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(trend["Year"], trend["Candidate records"], marker="o", color="#1F4D78", linewidth=2)
    ax.set(title="Crossref candidate records: 'critical mineral forecasting'", xlabel="Publication year", ylabel="Candidate records")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_dir / "fig1_publication_trend.png")
    # Fig. 2
    fig, ax = plt.subplots(figsize=(8.2, 3.7)); ax.axis("off")
    labels = [(0.05, "Input features\n$X_t$", "#E8EEF5"), (0.31, "Forget / input / output\ngates", "#D9EAF7"), (0.57, "Cell state\n$c_t$", "#FFF2CC"), (0.79, "Hidden state\n$h_t$ and forecast", "#E2F0D9")]
    for x, label, color in labels:
        ax.add_patch(FancyBboxPatch((x, 0.31), 0.16, 0.33, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#1F4D78", linewidth=1.4))
        ax.text(x + 0.08, 0.475, label, ha="center", va="center", fontsize=10)
    for x in (0.21, 0.47, 0.73): ax.add_patch(FancyArrowPatch((x, 0.475), (x + 0.08, 0.475), arrowstyle="->", mutation_scale=16, color="#1F4D78", linewidth=1.4))
    ax.text(0.5, 0.84, "Conceptual LSTM recurrent-regression candidate", ha="center", fontsize=14, fontweight="bold", color="#0B2545")
    _save_figure(fig, figure_dir / "fig2_lstm_structure.png")
    # Fig. 3
    fig, ax = plt.subplots(figsize=(10, 4.3)); ax.axis("off")
    stages = [(0.04, "Observed history\nup to t-1"), (0.23, "Train-only\nfeature selection"), (0.42, "Scheduled VMD /\nmodel re-fit"), (0.61, "One-step forecast\nand interval"), (0.80, "Target close /\ntrading score at t")]
    for x, label in stages:
        ax.add_patch(FancyBboxPatch((x, .34), .14, .26, boxstyle="round,pad=.02", facecolor="#E8EEF5", edgecolor="#1F4D78"))
        ax.text(x+.07,.47,label,ha="center",va="center",fontsize=9)
    for x in (.18,.37,.56,.75): ax.add_patch(FancyArrowPatch((x,.47),(x+.05,.47),arrowstyle="->",mutation_scale=15,color="#1F4D78"))
    ax.text(.5,.78,"Causal one-step forecasting and trading framework",ha="center",fontsize=14,fontweight="bold",color="#0B2545")
    ax.text(.5,.15,"Calibration residuals and thresholds are fixed before the final test; no test target enters training, selection, decomposition, or tuning.",ha="center",fontsize=9,color="#4D4D4D")
    _save_figure(fig, figure_dir / "fig3_forecasting_framework.png")
    # Fig. 4
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for company in ORDER:
        price, _ = _load_series(company)
        ax.plot(price.index, price / price.iloc[0] * 100, linewidth=1.1, label=company)
    ax.set(title="Normalized critical-mineral equity closing prices", ylabel="Index (first observation = 100)")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.22)
    _save_figure(fig, figure_dir / "fig4_price_series.png")
    # Fig. 5
    frequency = features.groupby("Feature")["Selection Count"].sum().to_dict()
    fig, ax = plt.subplots(figsize=(9.5, 4.1)); _draw_predictor_word_cloud(ax, frequency)
    _save_figure(fig, figure_dir / "fig5_selected_predictor_wordcloud.png")
    # Fig. 6
    sentiment = pd.read_excel(Path("code/data/news_sentiment_data.xlsx"), sheet_name="Data")
    sentiment["date"] = pd.to_datetime(sentiment["date"]); sentiment = sentiment.set_index("date")["News Sentiment"].loc["2011-05-01":]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7)); sentiment.resample("MS").mean().plot(ax=axes[0], color="#1F4D78", linewidth=1); axes[0].set(title="Monthly mean news sentiment", ylabel="Index"); axes[0].grid(alpha=.2)
    axes[1].hist(sentiment.dropna(), bins=45, color="#2E74B5", alpha=.82); axes[1].set(title="Distribution of daily news sentiment", xlabel="Index", ylabel="Frequency")
    _save_figure(fig, figure_dir / "fig6_sentiment.png")
    # Fig. 7
    display_start = max(0, len(training) - 350); display_index = training.index[display_start:]
    fig, axes = plt.subplots(VMD_K + 1, 1, figsize=(9.5, 7.3), sharex=True); axes[0].plot(display_index, training.iloc[display_start:], color="#0B2545", linewidth=1); axes[0].set_ylabel("Close"); axes[0].set_title("Training-only VMD illustration: Glencore")
    for mode, ax in enumerate(axes[1:], start=1): ax.plot(display_index, components[mode - 1, display_start:], color="#2E74B5", linewidth=.85); ax.set_ylabel(f"V{mode}")
    _save_figure(fig, figure_dir / "fig7_vmd_components.png")
    # Fig. 8
    table5 = tables[5]
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); colors = ["#1F4D78" if value == "LASSO-ARX" else "#77A6C8" for value in table5["Illustrative Route"]]; ax.bar(table5["Mode"], table5["Approximate Entropy"], color=colors); ax.set(title="Approximate entropy of training-only VMD components", ylabel="Approximate entropy"); ax.grid(axis="y",alpha=.2)
    _save_figure(fig, figure_dir / "fig8_approximate_entropy.png")
    # Fig. 9 and Fig. 11 use current final test, not calibration.
    glencore = forecasts[(forecasts["Company"] == "Glencore") & (forecasts["Partition"] == "test")].copy(); end_dates = sorted(glencore["Date"].unique())[-100:]; window = glencore[glencore["Date"].isin(end_dates)]
    fig, ax = plt.subplots(figsize=(9.5, 4.2)); actual = window.drop_duplicates("Date").sort_values("Date"); ax.plot(actual["Date"], actual["Actual"], color="#0B2545", label="Actual", linewidth=1.6)
    for model, color in [("Random Walk", "#999999"), ("ARIMA", "#2E74B5"), ("Selected ARX", "#6AA84F")]:
        group = window[window["Model"] == model].sort_values("Date"); ax.plot(group["Date"], group["Forecast"], label=model, color=color, linewidth=1)
    ax.set(title="Glencore final-test forecasts: single-model candidates", ylabel="Close"); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=.2)
    _save_figure(fig, figure_dir / "fig9_single_model_forecasts.png")
    # Fig. 10
    rank = metrics.groupby("Model")["RMSE"].mean().sort_values(); random_walk = rank.get("Random Walk", np.nan); rank = (rank / random_walk).head(14)
    fig, ax = plt.subplots(figsize=(9.5, 4.4)); ax.barh(rank.index[::-1], rank.values[::-1], color=["#9B1C1C" if item == PROPOSED else "#2E74B5" for item in rank.index[::-1]]); ax.axvline(1, color="#555555", linestyle="--"); ax.set(title="Cross-company mean RMSE relative to random walk", xlabel="Relative mean RMSE")
    _save_figure(fig, figure_dir / "fig10_error_barplots.png")
    # Fig. 11
    proposed = window[window["Model"] == PROPOSED].sort_values("Date"); fig, ax = plt.subplots(figsize=(9.5, 4.2)); ax.plot(actual["Date"], actual["Actual"], color="#0B2545", label="Actual",linewidth=1.5); ax.plot(proposed["Date"], proposed["Forecast"], color="#9B1C1C", label="Proposed forecast",linewidth=1.1); ax.fill_between(proposed["Date"], proposed["Lower"], proposed["Upper"], color="#9B1C1C",alpha=.18,label="90% conformal interval"); ax.set(title="Glencore proposed-model conformal intervals",ylabel="Close"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    _save_figure(fig, figure_dir / "fig11_interval_forecasts.png")
    # Fig. 12
    fig, ax = plt.subplots(figsize=(9.5, 3.8)); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.plot([.06,.94],[.45,.45],color="#1F4D78",linewidth=2)
    milestones=[(.12,"t-1 close\nand features"),(.36,"Forecast, interval\nand position"),(.61,"t close\nrealized"),(.84,"Return, cost\nand score")]
    for x,label in milestones: ax.scatter([x],[.45],s=130,color="#2E74B5",zorder=3); ax.text(x,.69,label,ha="center",va="center",fontsize=10)
    ax.add_patch(FancyArrowPatch((.15,.45),(.33,.45),arrowstyle="->",mutation_scale=15,color="#1F4D78")); ax.add_patch(FancyArrowPatch((.39,.45),(.58,.45),arrowstyle="->",mutation_scale=15,color="#1F4D78")); ax.add_patch(FancyArrowPatch((.64,.45),(.81,.45),arrowstyle="->",mutation_scale=15,color="#1F4D78")); ax.text(.5,.13,"A signal is formed before the realized next-period return; costs are deducted when position changes.",ha="center",fontsize=9,color="#4D4D4D")
    _save_figure(fig, figure_dir / "fig12_trading_strategy.png")
    # Fig. 13
    selected = trading[(np.isclose(trading["Cost"], .001)) & (trading["Strategy"] == "Directional (no uncertainty)")].groupby("Model")["Sharpe"].mean().sort_values(ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(9.5, 4.3)); ax.barh(selected.index[::-1], selected.values[::-1], color=["#9B1C1C" if item == PROPOSED else "#2E74B5" for item in selected.index[::-1]]); ax.axvline(0,color="#555555",linewidth=.8); ax.set(title="Mean directional-strategy Sharpe ratio at 10 bp",xlabel="Sharpe ratio")
    _save_figure(fig, figure_dir / "fig13_trading_evaluation.png")


def generate(output: Path) -> Path:
    output = Path(output).resolve()
    (output / "tables").mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir(parents=True, exist_ok=False)
    required = {name: Path(OUT_DATA) / name for name in ("experiment_registry.json", "walk_forward_metrics.csv", "interval_summary.csv", "trading_results.csv", "dm_test_results.csv", "feature_stability.csv", "walk_forward_forecasts.csv")}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing: raise FileNotFoundError("Complete the full causal run before generating paper assets: " + "; ".join(missing))
    registry = json.loads(required["experiment_registry.json"].read_text(encoding="utf-8"))
    metrics = pd.read_csv(required["walk_forward_metrics.csv"])
    interval = pd.read_csv(required["interval_summary.csv"])
    trading = pd.read_csv(required["trading_results.csv"])
    dm = pd.read_csv(required["dm_test_results.csv"])
    features = pd.read_csv(required["feature_stability.csv"])
    forecasts = pd.read_csv(required["walk_forward_forecasts.csv"], parse_dates=["Origin Date", "Model Fit Origin Date", "Date"])
    trend = _crossref_publication_trend(output)
    training, components, vmd_table = _training_vmd()
    tables = _write_tables(output, registry, metrics, interval, trading, dm, features, vmd_table)
    _write_figures(output, tables, metrics, trading, forecasts, training, components, features)
    # Full ledgers satisfy the complete-metrics requirement without forcing
    # 100+ row machine tables into the main results narrative.
    appendix = output / "appendix_ledgers"; appendix.mkdir()
    for source, target in ((metrics, "complete_forecast_metrics.csv"), (interval, "complete_interval_metrics.csv"), (trading, "complete_trading_metrics.csv"), (dm, "complete_dm_tests.csv"), (features, "complete_feature_stability.csv")):
        source.to_csv(appendix / target, index=False)
    figure_manifest = {f"Figure {number}": {"caption": FIGURE_CAPTIONS[number], "file": f"figures/fig{number}_{['publication_trend','lstm_structure','forecasting_framework','price_series','selected_predictor_wordcloud','sentiment','vmd_components','approximate_entropy','single_model_forecasts','error_barplots','interval_forecasts','trading_strategy','trading_evaluation'][number-1]}.png"} for number in range(1,14)}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "paper_title": PAPER_TITLE, "canonical_run_id": registry.get("run_id"),
        "classification": "causal empirical paper assets; released-paper shadow excluded", "tables": {f"Table {number}": TABLE_CAPTIONS[number] for number in range(1,13)}, "figures": figure_manifest,
        "literature_query": {"provider": "Crossref Works API", "query_title": "critical mineral forecasting", "purpose": "descriptive candidate-record trend; not a systematic review"},
        "source_checksums": {path.name: _sha256(path) for path in required.values()},
    }
    (output / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.output))


if __name__ == "__main__":
    main()
