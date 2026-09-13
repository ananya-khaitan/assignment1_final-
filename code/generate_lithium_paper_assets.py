"""Generate the twelve tables and thirteen figures for the lithium paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lithium_config import ADAPTIVE_FINAL_EPOCHS, ADAPTIVE_SEARCH_EPOCHS, FEATURE_BLOCKS, LIU_LAGS, LIU_LSTM_BATCH_SIZE, LIU_LSTM_EPOCHS, LIU_LSTM_HIDDEN_SIZE, TARGETS
from models.released_paper_shadow import full_sample_vmd

PROPOSED = "Proposed Adaptive Decomposition-Routing Ensemble"
PROPOSED_DISPLAY = "AMDR-Li"
FIXED = "Fixed VMD-LASSO-ARIMA/LSTM"
SINGLES = ["ES", "ARIMA", "SVR", "RF", "MLP", "ELM", "LSTM"]
DECOMPOSITION = ["CEEMDAN-ARIMA", "VMD-ARIMA", "CEEMDAN-LSTM", "VMD-LSTM", FIXED, PROPOSED]
TARGET_LABELS = {
    "lithium_hydroxide": "COMEX lithium hydroxide CIF CJK futures",
    "lithium_carbonate": "SMM battery-grade lithium carbonate spot",
}
TABLE_CAPTIONS = {
    1: "Representative research works in lithium and energy-price forecasting.",
    2: "Parameter settings used in this study.",
    3: "Indicators used in this study.",
    4: "Statistical information for the forecasting targets and multimodal indicators.",
    5: "Information on the decomposed lithium-price modes.",
    6: "Indicators selected for the decomposed modes using LASSO regression.",
    7: "Forecasting errors on the testing sets for single models.",
    8: "Forecasting errors on the testing sets for decomposition models.",
    9: "Diebold-Mariano tests comparing AMDR-Li with forecasting benchmarks.",
    10: "Interval forecasting errors on the evaluation sets.",
    11: "Performance of two trading schemes and optimized versions based on decomposition-model forecasts.",
    12: "Performance of two trading schemes and optimized versions based on single-model forecasts.",
}
FIGURE_CAPTIONS = {
    1: "Time trend of candidate publications in lithium-price forecasting; Crossref title-query counts provide a descriptive measure of publication activity.",
    2: "Conceptual structure of the LSTM model.",
    3: "Architecture of the proposed AMDR-Li forecasting framework.",
    4: "Lithium-price time series used as forecasting targets.",
    5: "Word cloud of collected lithium-related news headlines.",
    6: "Lithium-news sentiment scores and distributions.",
    7: "Reference-parameter VMD modes of the lithium-price series.",
    8: "Approximate-entropy values of the reference-parameter VMD modes.",
    9: "Forecasting results using single forecasting models.",
    10: "Forecasting-error metrics across models.",
    11: "Interval forecasting results for AMDR-Li.",
    12: "Graphical illustration of the trading strategies.",
    13: "Trading-evaluation metrics across forecasting models.",
}
COLORS = {"navy": "#0B2545", "blue": "#2E74B5", "light": "#D9EAF7", "red": "#9B1C1C", "green": "#4F7F3B", "grey": "#777777"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _format(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(include=[np.number]).columns:
        if "Coverage" in column or "%" in column or "Accuracy" in column or "MAPE" in column:
            out[column] = out[column].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        else:
            out[column] = out[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out


def _trend(output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(2011, 2027):
        try:
            response = requests.get(
                "https://api.crossref.org/works",
                params={"query.title": "lithium price forecasting", "filter": f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31", "rows": 0},
                headers={"User-Agent": "LithiumForecastStudy/1.0 (mailto:research@example.com)"},
                timeout=25,
            )
            response.raise_for_status()
            count = int(response.json()["message"]["total-results"])
            status = "ok"
        except Exception as exc:
            count, status = 0, f"unavailable:{type(exc).__name__}"
        rows.append({"Year": year, "Candidate records": count, "Status": status})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "literature_query_snapshot.csv", index=False)
    return frame


def _targets() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(spec["processed"], parse_dates=["Date"]).sort_values("Date") for name, spec in TARGETS.items()}


def _write_tables(output: Path, registry: dict[str, Any], metrics: pd.DataFrame, dm: pd.DataFrame, intervals: pd.DataFrame,
                  trading: pd.DataFrame, selections: pd.DataFrame, components: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    tables: dict[int, pd.DataFrame] = {}
    tables[1] = pd.DataFrame([
        ["Hochreiter and Schmidhuber (1997)", "Sequence learning", "LSTM", "Recurrent nonlinear forecasting"],
        ["Tibshirani (1996)", "Sparse regression", "LASSO", "Multimodal indicator selection"],
        ["Dragomiretskiy and Zosso (2014)", "Signal decomposition", "VMD", "Adaptive narrow-band modes"],
        ["Torres et al. (2011)", "Noise-assisted decomposition", "CEEMDAN", "Decomposition comparator"],
        ["Diebold and Mariano (1995)", "Predictive-accuracy testing", "DM test", "Pairwise loss comparison"],
        ["Box et al. (2015)", "Time-series forecasting", "ARIMA", "Linear forecasting of smooth decomposed modes"],
        ["This study", "One-step-ahead daily price forecasting", "AMDR-Li", "Development-locked decomposition, route and stacking selection"],
    ], columns=["Study", "Research focus", "Method", "Role in this study"])
    tables[2] = pd.DataFrame([
        ["Chronological split", "80:20", "Static training and testing sets"],
        ["Lag order", LIU_LAGS, "Lagged target and indicator features"],
        ["Reference-parity VMD modes", "; ".join(f"{key}: {value['modes']}" for key, value in TARGETS.items()), "Fixed-routing comparator"],
        ["Adaptive VMD search", "K ∈ {7, 8, 9, 10}; α ∈ {1500, 3000, 4500}", "Development-only specification selection"],
        ["Adaptive CEEMDAN search", "maximum IMF ∈ {7, 9}", "Development-only specification selection"],
        ["Adaptive route candidates", "ARIMA, LSTM, SVR, RF", "Mode-level development selection"],
        ["LSTM hidden units", LIU_LSTM_HIDDEN_SIZE, "One recurrent layer"],
        ["Comparator LSTM epochs", LIU_LSTM_EPOCHS, "Homogeneous/fixed model fitting"],
        ["Adaptive LSTM epochs", f"search={ADAPTIVE_SEARCH_EPOCHS}; final={ADAPTIVE_FINAL_EPOCHS}", "Development selection and locked-test refit"],
        ["LSTM batch size", LIU_LSTM_BATCH_SIZE, "Mini-batch size"],
        ["Single-model RF estimators", 100, "Undecomposed benchmark"],
        ["Adaptive-route RF", "300 trees; leaf>=2; max_features=0.8", "Component-route candidate"],
        ["Single-model SVR", "RBF; C=8.1; gamma=0.1", "Undecomposed benchmark"],
        ["Adaptive-route SVR", "RBF; C=8.1; gamma=scale; epsilon=0.05", "Component-route candidate"],
        ["Interval levels", "80%, 90%, 95%", "Residual-quantile intervals"],
        ["Random seed", 42, "Reproducibility"],
    ], columns=["Parameter", "Setting", "Function"])
    block_map = {column: block for block, columns in FEATURE_BLOCKS.items() for column in columns}
    descriptions = {
        "Cross_Lithium_Close": "Price of the other lithium contract/spot target",
        "Albemarle_Close": "Albemarle equity close",
        "Ganfeng_Lithium_Close": "Ganfeng Lithium equity close",
        "News_Article_Count": "Lithium-headline volume",
        "News_TextBlob_Polarity": "Lexicon sentiment of lithium headlines",
        "News_FinBERT_Score": "Financial-language sentiment of lithium headlines",
        "SP500": "S&P 500 close", "VIX": "CBOE VIX close", "US_Dollar_Index": "US dollar index close",
        "MVIS_Critical_Minerals": "MVIS Global Rare Earth/Strategic Metals index",
    }
    indicators = []
    for column, block in block_map.items():
        descriptions.setdefault(column, column.replace("Trends_", "Google Trends: ").replace("_", " "))
        indicators.append([column, descriptions[column], block.replace("_", " ").title(), "Aligned to target trading date; lagged in model frame"])
    tables[3] = pd.DataFrame(indicators, columns=["Indicator", "Definition", "Data modality", "Treatment"])
    stats = []
    for target, panel in panels.items():
        numeric = panel.drop(columns=["Date"]).apply(pd.to_numeric, errors="coerce")
        for column in ["Target_Close", *list(block_map)]:
            series = numeric[column].dropna()
            stats.append([TARGET_LABELS[target] if column == "Target_Close" else column, target, len(series), series.mean(), series.std(), series.min(), series.max()])
    tables[4] = _format(pd.DataFrame(stats, columns=["Series", "Panel", "N", "Mean", "Std. dev.", "Minimum", "Maximum"]))
    tables[5] = _format(components.rename(columns={"Target": "Panel", "Variance Ratio": "Variance share", "Average Period": "Average period"})[["Panel", "Method", "Mode", "Frequency", "Average period", "Variance share", "Correlation", "Approximate Entropy"]])
    proposed_selection = selections[selections["Model"] == PROPOSED].copy()
    adaptive_weights = {
        (entry["target"], candidate): float(weight)
        for entry in registry.get("adaptive_selection", [])
        for candidate, weight in entry.get("stacking_weights", {}).items()
    }
    proposed_selection["Candidate"] = proposed_selection["Mode"].astype(str).str.split(" / ").str[0]
    proposed_selection["Stacking Weight"] = [
        adaptive_weights.get((target, candidate), 0.0)
        for target, candidate in zip(proposed_selection["Target"], proposed_selection["Candidate"])
    ]
    proposed_selection = proposed_selection[proposed_selection["Stacking Weight"] > 1e-6]
    proposed_selection["Selected Features"] = proposed_selection["Selected Features"].fillna("").map(lambda value: textwrap.shorten(value.replace(" | ", ", "), width=115, placeholder=" …"))
    tables[6] = proposed_selection[["Target", "Mode", "Route", "Stacking Weight", "Selected Features"]].rename(columns={"Target": "Panel", "Selected Features": "Selected indicators", "Stacking Weight": "Candidate weight"})
    tables[6] = _format(tables[6])
    metric_cols = ["Target", "Model", "RMSE", "MAE", "MASE", "MAPE", "sMAPE", "Directional Accuracy"]
    tables[7] = _format(metrics[metrics["Model"].isin(SINGLES)][metric_cols].sort_values(["Target", "RMSE"]))
    tables[8] = _format(metrics[metrics["Model"].isin(DECOMPOSITION)][metric_cols].sort_values(["Target", "RMSE"]))
    tables[8]["Model"] = tables[8]["Model"].replace({PROPOSED: PROPOSED_DISPLAY})
    dm_proposed = []
    for row in dm.itertuples(index=False):
        left, right = getattr(row, "_1"), getattr(row, "_2")
        if PROPOSED not in (left, right):
            continue
        benchmark = right if left == PROPOSED else left
        if benchmark == "Random Walk":
            continue
        statistic = row[3] if left == PROPOSED else -row[3]
        dm_proposed.append([row.Target, benchmark, statistic, row[4], "Proposed better" if statistic < 0 else "Benchmark better"])
    tables[9] = _format(pd.DataFrame(dm_proposed, columns=["Target", "Benchmark", "DM statistic", "p-value", "Direction"]).sort_values(["Target", "p-value"]))
    tables[10] = _format(intervals[intervals["Model"] == PROPOSED][["Target", "Nominal Coverage (%)", "Coverage (%)", "Avg Width", "Calibration Error", "Winkler"]].sort_values(["Target", "Nominal Coverage (%)"]))
    trade_cols = ["Model", "Scheme", "Cumulative Return (%)", "Maximum Drawdown (%)", "Sharpe Ratio", "Number of Transactions", "Hit Rate (%)"]
    tables[11] = _format(trading[trading["Model"].isin(DECOMPOSITION)].groupby(["Model", "Scheme"], as_index=False)[trade_cols[2:]].mean().sort_values(["Model", "Scheme"]))
    tables[11]["Model"] = tables[11]["Model"].replace({PROPOSED: PROPOSED_DISPLAY})
    tables[12] = _format(trading[trading["Model"].isin(SINGLES)].groupby(["Model", "Scheme"], as_index=False)[trade_cols[2:]].mean().sort_values(["Model", "Scheme"]))
    for number, frame in tables.items():
        frame.to_csv(output / "tables" / f"table{number}.csv", index=False)
    return tables


def _draw_wordcloud(ax: plt.Axes, news: pd.DataFrame) -> None:
    stop = {
        "lithium", "the", "and", "for", "with", "from", "that", "this", "are", "its", "into", "after",
        "price", "prices", "market", "stock", "stocks", "new", "says", "why", "how", "china", "chinese",
    }
    words = Counter(
        token
        for title in news["title"].fillna("")
        for token in re.findall(r"[A-Za-z]{3,}", title.lower())
        if token not in stop
    )
    ranked = words.most_common(48)
    canvas = Image.new("RGB", (1500, 620), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = findfont(FontProperties(family="DejaVu Sans"))
    rng = np.random.default_rng(42)
    placed: list[tuple[int, int, int, int]] = []
    maximum = max((count for _, count in ranked), default=1)
    palette = ["#0B2545", "#2E74B5", "#4F7F3B", "#557A95"]
    for rank, (word, count) in enumerate(ranked):
        size = int(22 + 54 * np.sqrt(count / maximum))
        fitted = False
        while size >= 18 and not fitted:
            font = ImageFont.truetype(font_path, size=size)
            left, top, right, bottom = draw.textbbox((0, 0), word, font=font)
            width, height = right - left, bottom - top
            for _ in range(1200):
                x = int(rng.integers(14, max(15, 1500 - width - 14)))
                y = int(rng.integers(14, max(15, 620 - height - 14)))
                box = (x - 8, y - 6, x + width + 8, y + height + 6)
                if all(box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3] for old in placed):
                    draw.text((x, y), word, font=font, fill=palette[rank % len(palette)])
                    placed.append(box)
                    fitted = True
                    break
            size -= 2
    ax.imshow(canvas, interpolation="bilinear")
    ax.axis("off")


def _write_equations(output: Path) -> None:
    directory = output / "equations"

    fig, ax = plt.subplots(figsize=(10.0, 3.0))
    ax.axis("off")
    equations = [
        (0.84, r"$\widehat{\Delta P}_t=\widehat{P}_t-P_{t-1},\qquad q_{0.90}=Q_{0.90}(|e|)$"),
        (0.64, r"$s_t=\operatorname{sign}(\widehat{\Delta P}_t)$"),
        (0.44, r"$s^{g}_t=+1\ \mathrm{if}\ \widehat{\Delta P}_t>q_{0.90};\quad s^{g}_t=-1\ \mathrm{if}\ \widehat{\Delta P}_t<-q_{0.90};\quad s^{g}_t=0\ \mathrm{otherwise}$"),
        (0.22, r"$s^{g}_t\ne0\quad\Longleftrightarrow\quad P_{t-1}\notin[\widehat{P}_t-q_{0.90},\widehat{P}_t+q_{0.90}]$"),
    ]
    for y, equation in equations:
        ax.text(.5, y, equation, ha="center", va="center", fontsize=17, color=COLORS["navy"])
    fig.savefig(directory / "trading_rule.png", dpi=260, bbox_inches="tight", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, .9))
    ax.axis("off")
    ax.text(.5, .5, r"$r^{\mathrm{strategy}}_t=s_t\left(\frac{P_t}{P_{t-1}}-1\right)-c\,|s_t-s_{t-1}|$", ha="center", va="center", fontsize=18, color=COLORS["navy"])
    fig.savefig(directory / "strategy_return.png", dpi=260, bbox_inches="tight", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, .9))
    ax.axis("off")
    ax.text(.5, .5, r"$K\in\{7,8,9,10\},\qquad \alpha\in\{1500,3000,4500\},\qquad M_{\mathrm{CEEMDAN}}\in\{7,9\}$", ha="center", va="center", fontsize=18, color=COLORS["navy"])
    fig.savefig(directory / "search_space.png", dpi=260, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _write_figures(output: Path, trend: pd.DataFrame, metrics: pd.DataFrame, intervals_ledger: pd.DataFrame, trading: pd.DataFrame,
                   components: pd.DataFrame, panels: dict[str, pd.DataFrame], forecasts: pd.DataFrame) -> None:
    directory = output / "figures"
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); ax.plot(trend["Year"], trend["Candidate records"], marker="o", color=COLORS["blue"], lw=2); ax.set(title="Crossref candidate records: lithium price forecasting", xlabel="Publication year", ylabel="Candidate records"); ax.grid(axis="y", alpha=.25); _save(fig, directory / "fig1_publication_trend.png")
    fig, ax = plt.subplots(figsize=(11.4, 3.8)); ax.axis("off")
    boxes = [(0.03, "Input vector\n$x_t$"), (.28, "LSTM gates"), (.53, "Cell state\n$c_t$"), (.78, "Hidden state $h_t$\n" + r"and forecast $\widehat{P}_t$")]
    box_w, box_h, box_y = .18, .30, .34
    for x, label in boxes:
        ax.add_patch(FancyBboxPatch((x,box_y),box_w,box_h,boxstyle="round,pad=.018",facecolor=COLORS["light"],edgecolor=COLORS["blue"],linewidth=1.4))
        ax.text(x+box_w/2,box_y+box_h/2,label,ha="center",va="center",fontsize=10)
    for left, right in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+box_w+.012,.49),(right[0]-.012,.49),arrowstyle="->",mutation_scale=16,color=COLORS["blue"],linewidth=1.3))
    ax.text(.5,.84,"LSTM recurrent forecasting block",ha="center",fontsize=15,fontweight="bold",color=COLORS["navy"])
    ax.set(xlim=(0,1),ylim=(0,1)); _save(fig, directory / "fig2_lstm_structure.png")

    fig, ax = plt.subplots(figsize=(12.2,4.4)); ax.axis("off")
    stages=[(.02,"Multimodal\ninformation"),(.215,"VMD / CEEMDAN\ncandidates"),(.41,"Sparse features +\nroute selection"),(.605,"Convex\nstacking"),(.80,"Forecasts, intervals\nand trading")]
    box_w, box_h, box_y = .16, .30, .34
    for x,label in stages:
        ax.add_patch(FancyBboxPatch((x,box_y),box_w,box_h,boxstyle="round,pad=.018",facecolor=COLORS["light"],edgecolor=COLORS["blue"],linewidth=1.4))
        ax.text(x+box_w/2,box_y+box_h/2,label,ha="center",va="center",fontsize=9.5)
    for left,right in zip(stages,stages[1:]):
        ax.add_patch(FancyArrowPatch((left[0]+box_w+.008,.49),(right[0]-.008,.49),arrowstyle="->",mutation_scale=16,color=COLORS["blue"],linewidth=1.3))
    ax.text(.5,.84,"AMDR-Li forecasting architecture",ha="center",fontsize=15,fontweight="bold",color=COLORS["navy"])
    ax.text(.5,.15,"Development selection  •  locked test evaluation  •  reproducible forecast and decision ledgers",ha="center",fontsize=9.5,color=COLORS["grey"])
    ax.set(xlim=(0,1),ylim=(0,1)); _save(fig, directory / "fig3_framework.png")
    fig, axes = plt.subplots(2,1,figsize=(9.4,6.2),sharex=False)
    for ax,(target,panel) in zip(axes,panels.items()): ax.plot(panel["Date"],panel["Target_Close"],color=COLORS["blue"],lw=1.2); ax.set(title=TARGET_LABELS[target],ylabel="Price"); ax.grid(alpha=.2)
    _save(fig, directory / "fig4_price_series.png")
    news = pd.read_csv(CODE_ROOT / "data/lithium/external/lithium_news_headlines.csv")
    fig, ax = plt.subplots(figsize=(9.4,4.4)); _draw_wordcloud(ax,news); ax.set_title("Lithium-specific headline vocabulary",fontsize=14,fontweight="bold",color=COLORS["navy"]); _save(fig,directory / "fig5_wordcloud.png")
    news["published_at_utc"] = pd.to_datetime(news["published_at_utc"],utc=True)
    fig,axes=plt.subplots(2,2,figsize=(11.6,7.2)); daily=news.set_index("published_at_utc")[["textblob_polarity","finbert_score"]].resample("W").mean(); axes[0,0].plot(daily.index,daily["textblob_polarity"],color=COLORS["blue"],lw=.9); axes[0,0].set_title("Weekly TextBlob polarity",pad=10); axes[0,1].hist(news["textblob_polarity"].dropna(),bins=35,color=COLORS["blue"]); axes[0,1].set_title("TextBlob distribution",pad=10); axes[1,0].plot(daily.index,daily["finbert_score"],color=COLORS["green"],lw=.9); axes[1,0].set_title("Weekly FinBERT score",pad=10); axes[1,1].hist(news["finbert_score"].dropna(),bins=35,color=COLORS["green"]); axes[1,1].set_title("FinBERT distribution",pad=10)
    locator = mdates.MonthLocator(interval=4)
    formatter = mdates.DateFormatter("%Y-%m")
    for ax in (axes[0,0],axes[1,0]):
        ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter); ax.tick_params(axis="x",labelrotation=28,labelsize=8,pad=5)
    for ax in axes.flat: ax.grid(alpha=.15); ax.tick_params(axis="y",labelsize=8)
    fig.subplots_adjust(hspace=.48,wspace=.30); _save(fig,directory / "fig6_sentiment.png")
    fig,axes=plt.subplots(2,1,figsize=(9.4,7.2))
    for ax,(target,panel) in zip(axes,panels.items()):
        values=panel[["Date","Target_Close",*list(dict.fromkeys(c for cols in FEATURE_BLOCKS.values() for c in cols))]].dropna()["Target_Close"].to_numpy(float); values=values[1:] if len(values)%2 else values; modes=full_sample_vmd(values,modes=int(TARGETS[target]["modes"])); start=max(0,len(values)-220); ax.plot(values[start:],color=COLORS["navy"],lw=1.2,label="Price");
        for index,mode in enumerate(modes,start=1): ax.plot(mode[start:],lw=.65,alpha=.8,label=f"V{index}")
        ax.set_title(TARGET_LABELS[target]); ax.legend(ncol=5,fontsize=7); ax.grid(alpha=.15)
    _save(fig,directory / "fig7_vmd_modes.png")
    vmd=components[components["Method"]=="VMD"]; fig,ax=plt.subplots(figsize=(8.8,4.2));
    for index,(target,group) in enumerate(vmd.groupby("Target")): ax.bar(np.arange(len(group))+(index-.5)*.36,group["Approximate Entropy"],width=.36,label=TARGET_LABELS[target])
    ax.set_xticks(np.arange(max(vmd.groupby("Target").size()))); ax.set_xticklabels([f"V{i}" for i in range(1,max(vmd.groupby("Target").size())+1)]); ax.set(title="Approximate entropy of VMD modes",ylabel="Approximate entropy"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.2); _save(fig,directory / "fig8_entropy.png")
    fig,axes=plt.subplots(2,1,figsize=(9.5,6.4))
    for ax,(target,group) in zip(axes,forecasts[forecasts["Model"].isin(SINGLES)].groupby("Target")):
        dates=sorted(group["Date"].unique())[-80:]; window=group[group["Date"].isin(dates)]; actual=window.drop_duplicates("Date").sort_values("Date"); ax.plot(actual["Date"],actual["Actual"],color=COLORS["navy"],lw=1.5,label="Actual")
        for model,color in [("ARIMA",COLORS["blue"]),("RF",COLORS["green"]),("LSTM",COLORS["red"])]: part=window[window["Model"]==model].sort_values("Date"); ax.plot(part["Date"],part["Forecast"],lw=.9,label=model,color=color)
        ax.set_title(TARGET_LABELS[target]); ax.legend(ncol=4,fontsize=8); ax.grid(alpha=.18)
    _save(fig,directory / "fig9_single_forecasts.png")
    ranked=metrics[metrics["Model"]!="Random Walk"].copy(); ranked["nRMSE"]=ranked.groupby("Target")["RMSE"].transform(lambda x:x/x.min()); summary=ranked.groupby("Model")["nRMSE"].mean().sort_values(); summary.index=summary.index.to_series().replace({PROPOSED:PROPOSED_DISPLAY}); fig,ax=plt.subplots(figsize=(9.2,5.0)); ax.barh(summary.index[::-1],summary.values[::-1],color=[COLORS["red"] if m==PROPOSED_DISPLAY else COLORS["blue"] for m in summary.index[::-1]]); ax.set(title="Mean target-normalized RMSE",xlabel="RMSE / best RMSE within target"); ax.grid(axis="x",alpha=.2); _save(fig,directory / "fig10_errors.png")
    interval=intervals_ledger[intervals_ledger["Model"]==PROPOSED].copy(); interval["Date"]=pd.to_datetime(interval["Date"]); fig,axes=plt.subplots(2,1,figsize=(9.4,6.2))
    for ax,(target,group) in zip(axes,interval.groupby("Target")): group=group.sort_values("Date").tail(90); ax.plot(group["Date"],group["Actual"],color=COLORS["navy"],label="Actual"); ax.plot(group["Date"],group["Forecast"],color=COLORS["red"],label="Forecast"); ax.fill_between(group["Date"],group["Lower"],group["Upper"],color=COLORS["red"],alpha=.18,label="90% interval"); ax.set_title(TARGET_LABELS[target]); ax.legend(fontsize=8); ax.grid(alpha=.18)
    _save(fig,directory / "fig11_intervals.png")
    fig,ax=plt.subplots(figsize=(9.4,3.7)); ax.axis("off"); milestones=[(.08,"Forecast\nchange"),(.29,"Scheme 1:\ndirectional"),(.51,"Optimized:\ninterval filter"),(.73,"Scheme 2:\ncost-aware"),(.92,"Return and\nrisk metrics")]
    for x,label in milestones: ax.scatter([x],[.48],s=130,color=COLORS["blue"]); ax.text(x,.72,label,ha="center",va="center",fontsize=9)
    for left,right in zip(milestones,milestones[1:]): ax.add_patch(FancyArrowPatch((left[0]+.025,.48),(right[0]-.025,.48),arrowstyle="->",mutation_scale=14,color=COLORS["blue"]))
    ax.set(xlim=(0,1),ylim=(0,1)); ax.text(.5,.16,"Signals compare forecast and origin prices; optimized variants abstain when uncertainty is large.",ha="center",fontsize=9); _save(fig,directory / "fig12_trading.png")
    trade=trading[(trading["Scheme"]=="Scheme 2")&(trading["Model"]!="Random Walk")].groupby("Model")["Sharpe Ratio"].mean().sort_values(); trade.index=trade.index.to_series().replace({PROPOSED:PROPOSED_DISPLAY}); fig,ax=plt.subplots(figsize=(9.2,4.8)); ax.barh(trade.index,trade.values,color=[COLORS["red"] if m==PROPOSED_DISPLAY else COLORS["blue"] for m in trade.index]); ax.axvline(0,color=COLORS["grey"],lw=.8); ax.set(title="Mean Sharpe ratio: Scheme 2 (10 bp)",xlabel="Sharpe ratio"); ax.grid(axis="x",alpha=.2); _save(fig,directory / "fig13_trading.png")


def generate(run_dir: Path, output: Path) -> Path:
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    (output / "tables").mkdir(parents=True, exist_ok=False); (output / "figures").mkdir(); (output / "equations").mkdir()
    required=[
        "experiment_registry.json",
        "forecast_ledger.csv",
        "forecast_metrics.csv",
        "dm_tests.csv",
        "interval_ledger.csv",
        "interval_metrics.csv",
        "trading_metrics.csv",
        "mode_feature_selection.csv",
        "component_summary.csv",
        "adaptive_candidate_search.csv",
        "adaptive_component_routes.csv",
        "adaptive_validation_report.json",
    ]
    missing=[name for name in required if not (run_dir/name).exists()]
    if missing: raise FileNotFoundError(f"Incomplete empirical run: {missing}")
    registry=json.loads((run_dir/"experiment_registry.json").read_text(encoding="utf-8")); metrics=pd.read_csv(run_dir/"forecast_metrics.csv"); dm=pd.read_csv(run_dir/"dm_tests.csv"); interval_ledger=pd.read_csv(run_dir/"interval_ledger.csv",parse_dates=["Date"]); intervals=pd.read_csv(run_dir/"interval_metrics.csv"); trading=pd.read_csv(run_dir/"trading_metrics.csv"); selections=pd.read_csv(run_dir/"mode_feature_selection.csv"); components=pd.read_csv(run_dir/"component_summary.csv"); forecasts=pd.read_csv(run_dir/"forecast_ledger.csv",parse_dates=["Date","Origin Date"]); panels=_targets(); trend=_trend(output)
    tables=_write_tables(output,registry,metrics,dm,intervals,trading,selections,components,panels); _write_figures(output,trend,metrics,interval_ledger,trading,components,panels,forecasts); _write_equations(output)
    appendix=output/"appendix_ledgers"; appendix.mkdir()
    for name in required[1:]:
        if not name.endswith(".csv"):
            (appendix / name).write_bytes((run_dir / name).read_bytes())
            continue
        ledger=pd.read_csv(run_dir/name)
        if "Model" in ledger.columns:
            ledger=ledger[ledger["Model"]!="Random Walk"]
        if {"Row Model","Column Model"}.issubset(ledger.columns):
            ledger=ledger[(ledger["Row Model"]!="Random Walk")&(ledger["Column Model"]!="Random Walk")]
        ledger.to_csv(appendix/name,index=False)
    # Keep the reference paper's twelve-table main-text architecture while
    # preserving the strongest naive benchmark as an explicit robustness
    # check.  The comparison is written separately so Random Walk cannot be
    # mistaken for one of the architecture benchmarks in Tables 7--9.
    rw = metrics[metrics["Model"].isin([PROPOSED, "Random Walk"])][
        ["Target", "Model", "RMSE", "MAE", "MASE", "MAPE", "sMAPE", "Directional Accuracy"]
    ].copy()
    proposed_rw = rw[rw["Model"] == PROPOSED].set_index("Target")
    naive_rw = rw[rw["Model"] == "Random Walk"].set_index("Target")
    robustness_rows = []
    for target in proposed_rw.index:
        proposed_row, naive_row = proposed_rw.loc[target], naive_rw.loc[target]
        robustness_rows.append(
            {
                "Target": target,
                "Proposed RMSE": proposed_row["RMSE"],
                "Random Walk RMSE": naive_row["RMSE"],
                "RMSE improvement (%)": 100.0 * (naive_row["RMSE"] - proposed_row["RMSE"]) / naive_row["RMSE"],
                "Proposed MAE": proposed_row["MAE"],
                "Random Walk MAE": naive_row["MAE"],
                "MAE improvement (%)": 100.0 * (naive_row["MAE"] - proposed_row["MAE"]) / naive_row["MAE"],
                "Proposed MAPE": proposed_row["MAPE"],
                "Random Walk MAPE": naive_row["MAPE"],
                "Proposed directional accuracy": proposed_row["Directional Accuracy"],
                "Random Walk directional accuracy": naive_row["Directional Accuracy"],
            }
        )
    _format(pd.DataFrame(robustness_rows)).to_csv(appendix / "random_walk_robustness.csv", index=False)
    rw_dm = []
    for row in dm.itertuples(index=False):
        left, right = getattr(row, "_1"), getattr(row, "_2")
        if set((left, right)) != {PROPOSED, "Random Walk"}:
            continue
        statistic = row[3] if left == PROPOSED else -row[3]
        rw_dm.append(
            {
                "Target": row.Target,
                "DM statistic (proposed minus Random Walk loss)": statistic,
                "p-value": row[4],
                "Direction": "Proposed better" if statistic < 0 else "Random Walk better",
            }
        )
    _format(pd.DataFrame(rw_dm)).to_csv(appendix / "random_walk_dm_tests.csv", index=False)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"run_id":registry["run_id"],"protocol":"development-locked adaptive decomposition-routing static retrospective lithium experiment","classification":registry["classification"],"adaptive_selection":registry.get("adaptive_selection",[]),"adaptive_protocol":registry.get("adaptive_protocol",{}),"tables":{f"Table {n}":TABLE_CAPTIONS[n] for n in range(1,13)},"figures":{f"Figure {n}":FIGURE_CAPTIONS[n] for n in range(1,14)},"source_checksums":{name:_sha256(run_dir/name) for name in required},"literature_query":"Crossref Works API query.title=lithium price forecasting; descriptive, not systematic"}
    (output/"asset_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); print(generate(args.run,args.output))


if __name__ == "__main__": main()
