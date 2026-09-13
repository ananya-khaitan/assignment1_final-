"""Build segregated Liu-style lithium model panels from documented inputs.

The preparation stage does not run VMD or fit a forecasting model.  It keeps
the two lithium targets, two lithium-exposed equities, lithium-specific
unstructured inputs, and broad controls in separately named columns so every
feature-block ablation is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lithium_config import (
    CARBONATE_WORKBOOK,
    EQUITY_TICKERS,
    EQUITY_WORKBOOK,
    GOOGLE_TRENDS_TERMS,
    HYDROXIDE_WORKBOOK,
    LITHIUM_EXTERNAL_DIR,
    LITHIUM_MANIFEST,
    LITHIUM_NEWS_FILE,
    LITHIUM_PROCESSED_DIR,
    LITHIUM_TRENDS_FILE,
    MVIS_WORKBOOK,
    NEWS_HEADLINE_FILTER_TERMS,
    NEWS_QUERY_TERMS,
    TARGETS,
    YAHOO_WORKBOOK,
)


UTC = timezone.utc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_series(series: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    result.index = pd.to_datetime(result.index, errors="coerce").tz_localize(None)
    result = result[~result.index.isna()].dropna()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result.name = name
    if result.empty:
        raise ValueError(f"{name} has no usable dated observations")
    return result.astype(float)


def read_carbonate() -> pd.Series:
    raw = pd.read_excel(CARBONATE_WORKBOOK, sheet_name="Table Data", header=None)
    data = raw.iloc[2:, :2].copy()
    data.columns = ["Date", "Close"]
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return _clean_series(data.dropna(subset=["Date"]).set_index("Date")["Close"], "Lithium_Carbonate_Close")


def _find_header_row(raw: pd.DataFrame, first_label: str) -> int:
    expected = first_label.strip().lower()
    for row_number, value in raw.iloc[:, 0].items():
        if isinstance(value, str) and value.strip().lower() == expected:
            return int(row_number)
    raise ValueError(f"Could not find {first_label!r} header row")


def read_hydroxide() -> pd.Series:
    raw = pd.read_excel(HYDROXIDE_WORKBOOK, sheet_name="Sheet 1", header=None)
    header_row = _find_header_row(raw, "Exchange Date")
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = [str(value).strip() for value in raw.iloc[header_row]]
    if not {"Exchange Date", "Close"}.issubset(data.columns):
        raise ValueError("Hydroxide workbook must contain Exchange Date and Close")
    data["Exchange Date"] = pd.to_datetime(data["Exchange Date"], errors="coerce")
    return _clean_series(
        data.dropna(subset=["Exchange Date"]).set_index("Exchange Date")["Close"],
        "Lithium_Hydroxide_Close",
    )


def read_lithium_equities() -> pd.DataFrame:
    raw = pd.read_excel(EQUITY_WORKBOOK, sheet_name="Stock Price", skiprows=[1])
    raw.columns = [str(column).replace(" (TRDPRC_1)", "").strip() for column in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
    missing = [ticker for ticker in EQUITY_TICKERS if ticker not in raw.columns]
    if missing:
        raise ValueError(f"Equity workbook is missing lithium tickers: {missing}")
    output = pd.DataFrame(index=raw.index)
    for ticker, label in EQUITY_TICKERS.items():
        output[f"{label}_Close"] = pd.to_numeric(raw[ticker], errors="coerce")
    return output.loc[:, ["Albemarle_Close", "Ganfeng_Lithium_Close"]]


def read_mvis() -> pd.Series:
    raw = pd.read_excel(MVIS_WORKBOOK, sheet_name="Sheet 1", header=None)
    header_row = _find_header_row(raw, "Exchange Date")
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = [str(value).strip() for value in raw.iloc[header_row]]
    if not {"Exchange Date", "Close"}.issubset(data.columns):
        raise ValueError("MVIS workbook must contain Exchange Date and Close")
    data["Exchange Date"] = pd.to_datetime(data["Exchange Date"], errors="coerce")
    return _clean_series(
        data.dropna(subset=["Exchange Date"]).set_index("Exchange Date")["Close"],
        "MVIS_Critical_Minerals",
    )


def read_market_controls() -> pd.DataFrame:
    raw = pd.read_csv(YAHOO_WORKBOOK, header=[0, 1], index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    mapping = {
        ("^GSPC", "Close"): "SP500",
        ("^VIX", "Close"): "VIX",
        ("DX-Y.NYB", "Close"): "US_Dollar_Index",
    }
    missing = [key for key in mapping if key not in raw.columns]
    if missing:
        raise ValueError(f"Yahoo input is missing market-control fields: {missing}")
    output = pd.DataFrame(index=raw.index)
    for key, label in mapping.items():
        output[label] = pd.to_numeric(raw[key], errors="coerce")
    return output.sort_index()


def _match_trends_column(columns: list[str], term: str) -> str:
    normalized_term = re.sub(r"\s+", " ", term.strip().lower())
    for column in columns:
        normalized_column = re.sub(r"\s+", " ", str(column).strip().lower())
        normalized_column = normalized_column.split(":", 1)[0]
        if normalized_column == normalized_term:
            return column
    raise ValueError(f"Google Trends file is missing the jointly retrieved term {term!r}")


def read_lithium_trends(path: Path = LITHIUM_TRENDS_FILE) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_column = next((column for column in raw.columns if str(column).strip().lower() in {"date", "week", "time"}), None)
    if date_column is None:
        raise ValueError("Lithium Trends file needs a Date, Week, or time column")
    raw[date_column] = pd.to_datetime(raw[date_column], errors="coerce")
    raw = raw.dropna(subset=[date_column]).sort_values(date_column)
    output = pd.DataFrame(index=raw[date_column])
    for term in GOOGLE_TRENDS_TERMS:
        source_column = _match_trends_column(list(raw.columns), term)
        label = f"Trends_{term.replace(' ', '_')}"
        output[label] = pd.to_numeric(raw[source_column], errors="coerce").to_numpy()
    if output.empty or output.dropna(how="all").empty:
        raise ValueError("Lithium Trends file contains no numeric observations")
    # For multi-year exports Google returns a weekly score labelled by the
    # start of the measurement week.  It is not observable until that week is
    # complete, so its as-of availability date is six days later.
    median_gap = output.index.to_series().diff().dt.days.median()
    if pd.notna(median_gap) and median_gap >= 6:
        output.index = output.index + pd.Timedelta(days=6)
    output.index.name = "Available_Date"
    return output[~output.index.duplicated(keep="last")].sort_index()


def _finbert_scores(titles: list[str]) -> list[float]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("FinBERT scoring requires transformers; install requirements.txt") from exc
    classifier = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
    scores: list[float] = []
    for result in classifier(titles, batch_size=32):
        label = str(result["label"]).lower()
        probability = float(result["score"])
        scores.append(probability if label == "positive" else -probability if label == "negative" else 0.0)
    return scores


def read_lithium_news(path: Path = LITHIUM_NEWS_FILE, *, score_finbert: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    required = {"published_at_utc", "title", "source", "url"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Lithium news file is missing columns: {sorted(missing)}")
    raw["published_at_utc"] = pd.to_datetime(raw["published_at_utc"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["published_at_utc", "title"])
    raw["title"] = raw["title"].astype(str).str.strip()
    pattern = "|".join(re.escape(term) for term in NEWS_HEADLINE_FILTER_TERMS)
    raw = raw[raw["title"].str.contains(pattern, case=False, regex=True, na=False)]
    raw = raw.drop_duplicates(subset=["url"], keep="first")
    raw = raw.drop_duplicates(subset=["published_at_utc", "title"], keep="first")
    if raw.empty:
        raise ValueError("No lithium-specific headlines remain after validation")

    if "textblob_polarity" not in raw or raw["textblob_polarity"].isna().any():
        from textblob import TextBlob

        raw["textblob_polarity"] = [float(TextBlob(title).sentiment.polarity) for title in raw["title"]]
    else:
        raw["textblob_polarity"] = pd.to_numeric(raw["textblob_polarity"], errors="coerce")

    if "finbert_score" not in raw:
        raw["finbert_score"] = np.nan
    raw["finbert_score"] = pd.to_numeric(raw["finbert_score"], errors="coerce")
    if score_finbert and raw["finbert_score"].isna().any():
        missing_mask = raw["finbert_score"].isna()
        raw.loc[missing_mask, "finbert_score"] = _finbert_scores(raw.loc[missing_mask, "title"].tolist())

    raw["Date"] = raw["published_at_utc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    daily = raw.groupby("Date", sort=True).agg(
        News_Article_Count=("title", "size"),
        News_TextBlob_Polarity=("textblob_polarity", "mean"),
        News_FinBERT_Score=("finbert_score", "mean"),
    )
    full_calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_calendar)
    # A day with no matching headline is an observed zero-attention/neutral
    # news day, not a missing value to be carried forward from an older story.
    daily["News_Article_Count"] = daily["News_Article_Count"].fillna(0.0)
    daily["News_TextBlob_Polarity"] = daily["News_TextBlob_Polarity"].fillna(0.0)
    no_news = daily["News_Article_Count"].eq(0.0)
    daily.loc[no_news, "News_FinBERT_Score"] = 0.0
    # On a day containing headlines, a missing FinBERT value remains missing
    # so the publication preflight cannot mistake an unscored article for a
    # neutral result.
    daily.index.name = "Available_Date"
    if score_finbert:
        persisted = raw.drop(columns=["Date"], errors="ignore")
        persisted.to_csv(path, index=False)
    return raw.sort_values("published_at_utc"), daily


def _asof_columns(calendar: pd.DatetimeIndex, frame: pd.DataFrame | pd.Series) -> pd.DataFrame:
    source = frame.to_frame() if isinstance(frame, pd.Series) else frame.copy()
    source.index = pd.to_datetime(source.index, errors="coerce").tz_localize(None)
    source = source[~source.index.isna()]
    source = source[~source.index.duplicated(keep="last")].sort_index().ffill().dropna(how="all")
    source.index.name = "Available_Date"
    left = pd.DataFrame({"Date": pd.DatetimeIndex(calendar).sort_values()})
    right = source.reset_index().sort_values("Available_Date")
    merged = pd.merge_asof(left, right, left_on="Date", right_on="Available_Date", direction="backward")
    return merged.drop(columns="Available_Date").set_index("Date")


def _coverage(frame: pd.DataFrame | pd.Series) -> dict[str, Any]:
    data = frame.to_frame() if isinstance(frame, pd.Series) else frame
    valid = data.dropna(how="all")
    return {
        "rows": int(len(data)),
        "first_valid_date": None if valid.empty else str(valid.index.min().date()),
        "last_valid_date": None if valid.empty else str(valid.index.max().date()),
        "missing_by_column": {str(column): int(value) for column, value in data.isna().sum().items()},
    }


def _required_paths() -> dict[str, Path]:
    return {
        "lithium_carbonate_workbook": CARBONATE_WORKBOOK,
        "lithium_hydroxide_workbook": HYDROXIDE_WORKBOOK,
        "lithium_equity_workbook": EQUITY_WORKBOOK,
        "mvis_workbook": MVIS_WORKBOOK,
        "yahoo_market_controls": YAHOO_WORKBOOK,
    }


def build_panels(*, require_complete: bool = True, score_finbert: bool = False) -> dict[str, Path]:
    missing_market = {name: path for name, path in _required_paths().items() if not path.exists()}
    if missing_market:
        raise FileNotFoundError("Missing lithium market inputs: " + "; ".join(f"{k}={v}" for k, v in missing_market.items()))

    carbonate = read_carbonate()
    hydroxide = read_hydroxide()
    equities = read_lithium_equities()
    mvis = read_mvis()
    controls = read_market_controls()

    # All ablations use one common information window.  Otherwise the
    # history-only hydroxide model would be scored from 2021 while the full
    # model begins with carbonate availability in 2023, invalidating the
    # incremental feature comparison.
    common_start = max(
        carbonate.index.min(),
        hydroxide.index.min(),
        equities.dropna().index.min(),
        mvis.index.min(),
        controls.dropna().index.min(),
    )
    common_end = min(
        carbonate.index.max(),
        hydroxide.index.max(),
        equities.dropna().index.max(),
        mvis.index.max(),
        controls.dropna().index.max(),
    )
    if common_end <= common_start:
        raise ValueError(f"Structured lithium inputs have no common sample: {common_start} to {common_end}")
    carbonate = carbonate.loc[common_start:common_end]
    hydroxide = hydroxide.loc[common_start:common_end]

    trends = None
    news_raw = None
    news_daily = None
    if LITHIUM_TRENDS_FILE.exists():
        trends = read_lithium_trends()
    if LITHIUM_NEWS_FILE.exists():
        news_raw, news_daily = read_lithium_news(score_finbert=score_finbert)

    if require_complete:
        missing_external = [
            str(path)
            for path, loaded in ((LITHIUM_TRENDS_FILE, trends), (LITHIUM_NEWS_FILE, news_daily))
            if loaded is None
        ]
        if missing_external:
            raise FileNotFoundError(
                "Publication run requires lithium-specific unstructured inputs: " + "; ".join(missing_external)
            )
        assert news_daily is not None
        if news_daily["News_FinBERT_Score"].isna().any():
            raise ValueError("Publication run requires complete FinBERT scores in lithium_news_headlines.csv")
        assert trends is not None and news_raw is not None
        if trends.index.min() > common_start + pd.Timedelta(days=14) or trends.index.max() < common_end - pd.Timedelta(days=14):
            raise ValueError(
                f"Lithium Trends coverage ({trends.index.min().date()} to {trends.index.max().date()}) "
                f"does not span the common study window ({common_start.date()} to {common_end.date()})"
            )
        news_dates = news_raw["published_at_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        if news_dates.min().normalize() > common_start or news_dates.max().normalize() < common_end:
            raise ValueError(
                f"Lithium headline coverage ({news_dates.min().date()} to {news_dates.max().date()}) "
                f"does not span the common study window ({common_start.date()} to {common_end.date()})"
            )

    LITHIUM_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sources: dict[str, pd.DataFrame | pd.Series] = {
        "equities": equities,
        "mvis": mvis,
        "market_controls": controls,
    }
    if trends is not None:
        sources["lithium_attention"] = trends
    if news_daily is not None:
        sources["lithium_news"] = news_daily

    outputs: dict[str, Path] = {}
    target_series = {
        "lithium_hydroxide": hydroxide,
        "lithium_carbonate": carbonate,
    }
    cross_series = {
        "lithium_hydroxide": carbonate.rename("Cross_Lithium_Close"),
        "lithium_carbonate": hydroxide.rename("Cross_Lithium_Close"),
    }
    for target_name, target in target_series.items():
        panel = target.rename("Target_Close").to_frame()
        panel = panel.join(_asof_columns(panel.index, cross_series[target_name]))
        for source in sources.values():
            panel = panel.join(_asof_columns(panel.index, source))
        # Preserve a stable schema even before external files arrive.  This
        # makes missing acquisition explicit instead of substituting generic
        # attention or sentiment.
        for term in GOOGLE_TRENDS_TERMS:
            column = f"Trends_{term.replace(' ', '_')}"
            if column not in panel:
                panel[column] = np.nan
        for column in ("News_Article_Count", "News_TextBlob_Polarity", "News_FinBERT_Score"):
            if column not in panel:
                panel[column] = np.nan
        ordered = [
            "Target_Close",
            "Cross_Lithium_Close",
            "Albemarle_Close",
            "Ganfeng_Lithium_Close",
            *(f"Trends_{term.replace(' ', '_')}" for term in GOOGLE_TRENDS_TERMS),
            "News_Article_Count",
            "News_TextBlob_Polarity",
            "News_FinBERT_Score",
            "SP500",
            "VIX",
            "US_Dollar_Index",
            "MVIS_Critical_Minerals",
        ]
        panel = panel[ordered]
        panel.index.name = "Date"
        output_path = Path(TARGETS[target_name]["processed"])
        panel.to_csv(output_path)
        outputs[target_name] = output_path

    dictionary = pd.DataFrame(
        [
            ("Target_Close", "forecast_target", "native price units", "Current target close"),
            ("Cross_Lithium_Close", "cross_lithium", "native price units", "Other lithium market's latest available close"),
            ("Albemarle_Close", "lithium_equities", "native listed currency", "ALB close"),
            ("Ganfeng_Lithium_Close", "lithium_equities", "native listed currency", "002460.SZ close"),
            *[(f"Trends_{term.replace(' ', '_')}", "lithium_attention", "Google index 0-100", term) for term in GOOGLE_TRENDS_TERMS],
            ("News_Article_Count", "lithium_news", "headlines/day", "Validated lithium headline count"),
            ("News_TextBlob_Polarity", "lithium_news", "[-1, 1]", "Mean headline TextBlob polarity"),
            ("News_FinBERT_Score", "lithium_news", "[-1, 1]", "Signed mean FinBERT confidence"),
            ("SP500", "market_controls", "index points", "S&P 500 close"),
            ("VIX", "market_controls", "index points", "CBOE volatility index close"),
            ("US_Dollar_Index", "market_controls", "index points", "US dollar index close"),
            ("MVIS_Critical_Minerals", "market_controls", "index points", "MVIS strategic-metals benchmark close"),
        ],
        columns=["column", "feature_block", "unit", "definition"],
    )
    dictionary_path = LITHIUM_PROCESSED_DIR / "data_dictionary.csv"
    dictionary.to_csv(dictionary_path, index=False)
    outputs["data_dictionary"] = dictionary_path

    manifest: dict[str, Any] = {
        "study": "Liu-style static lithium-price forecasting",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_classification": "noncausal static retrospective prediction",
        "common_sample": {"start": str(common_start.date()), "end": str(common_end.date())},
        "targets": {name: {**TARGETS[name], "workbook": str(TARGETS[name]["workbook"]), "processed": str(path), "coverage": _coverage(target_series[name])} for name, path in outputs.items() if name in TARGETS},
        "feature_blocks": {
            "lithium_equities": ["Albemarle", "Ganfeng Lithium"],
            "lithium_attention": list(GOOGLE_TRENDS_TERMS),
            "lithium_news_query_terms": list(NEWS_QUERY_TERMS),
            "market_controls": ["SP500", "VIX", "US_Dollar_Index", "MVIS_Critical_Minerals"],
        },
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in _required_paths().items()
        },
        "external_inputs": {
            "google_trends": None if trends is None else {"path": str(LITHIUM_TRENDS_FILE.resolve()), "sha256": _sha256(LITHIUM_TRENDS_FILE), "coverage": _coverage(trends)},
            "lithium_news": None if news_raw is None else {"path": str(LITHIUM_NEWS_FILE.resolve()), "sha256": _sha256(LITHIUM_NEWS_FILE), "headline_rows": int(len(news_raw)), "daily_coverage": _coverage(news_daily)},
        },
        "alignment": "Latest available observation on each target calendar; Liu runner applies the one-observation external-feature shift.",
        "publication_ready": bool(trends is not None and news_daily is not None and news_daily["News_FinBERT_Score"].notna().all()),
    }
    LITHIUM_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    outputs["manifest"] = LITHIUM_MANIFEST
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="Build structured market panels before Trends/news acquisition is complete")
    parser.add_argument("--score-finbert", action="store_true", help="Fill missing FinBERT scores using ProsusAI/finbert")
    args = parser.parse_args()
    outputs = build_panels(require_complete=not args.allow_incomplete, score_finbert=args.score_finbert)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
