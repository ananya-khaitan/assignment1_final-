"""Build auditable static-study panels for the ALB and Ganfeng extension.

The extension keeps the physical lithium-price study intact.  It treats the
two listed producers as separate forecast targets, with the other producer,
both lithium prices, lithium-specific attention/news, and equity-market
controls as lagged external inputs in the released-style feature frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from lithium_config import (
    EQUITY_TARGETS,
    LITHIUM_MANIFEST,
    LITHIUM_PROCESSED_DIR,
    USD_CNY_FILE,
)
from data_sources.prepare_lithium_inputs import (
    _asof_columns,
    _coverage,
    _sha256,
    read_carbonate,
    read_hydroxide,
    read_lithium_equities,
    read_lithium_news,
    read_lithium_trends,
    read_market_controls,
    read_mvis,
)


UTC = timezone.utc


def read_usd_cny() -> pd.Series:
    raw = pd.read_csv(USD_CNY_FILE)
    required = {"observation_date", "DEXCHUS"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"USD/CNY file is missing columns: {sorted(missing)}")
    raw["observation_date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw["DEXCHUS"] = pd.to_numeric(raw["DEXCHUS"], errors="coerce")
    series = raw.dropna(subset=["observation_date", "DEXCHUS"]).set_index("observation_date")["DEXCHUS"]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "USD_CNY"
    if series.empty:
        raise ValueError("USD/CNY file has no usable observations")
    return series.astype(float)


def read_china_equity_control() -> pd.Series:
    controls = pd.read_csv(CODE_ROOT / "data" / "Bulk_Yahoo_Historical_Data.csv", header=[0, 1], index_col=0, parse_dates=True)
    key = ("000001.SS", "Close")
    if key not in controls.columns:
        raise ValueError("Bulk Yahoo file is missing the Shanghai Composite close")
    series = pd.to_numeric(controls[key], errors="coerce")
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[~series.index.isna()].dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "Shanghai_Composite"
    return series.astype(float)


def _source_end(source: pd.Series | pd.DataFrame) -> pd.Timestamp:
    valid = source.dropna(how="all") if isinstance(source, pd.DataFrame) else source.dropna()
    return pd.Timestamp(valid.index.max())


def build_equity_panels(*, require_complete: bool = True) -> dict[str, Path]:
    if not USD_CNY_FILE.exists():
        raise FileNotFoundError(f"Missing USD/CNY input: {USD_CNY_FILE}")

    equities = read_lithium_equities()
    carbonate = read_carbonate().rename("Lithium_Carbonate_Close")
    hydroxide = read_hydroxide().rename("Lithium_Hydroxide_Close")
    market = read_market_controls()
    mvis = read_mvis()
    china = read_china_equity_control()
    usd_cny = read_usd_cny()
    trends = read_lithium_trends()
    news_raw, news = read_lithium_news()
    if require_complete and news["News_FinBERT_Score"].isna().any():
        raise ValueError("Equity extension requires complete FinBERT scores")

    sources: dict[str, pd.Series | pd.DataFrame] = {
        "Lithium_Carbonate_Close": carbonate,
        "Lithium_Hydroxide_Close": hydroxide,
        "market_controls": market,
        "MVIS_Critical_Minerals": mvis,
        "Shanghai_Composite": china,
        "USD_CNY": usd_cny,
        "lithium_attention": trends,
        "lithium_news": news,
    }
    common_start = max(pd.Timestamp(source.dropna(how="all").index.min()) if isinstance(source, pd.DataFrame) else pd.Timestamp(source.dropna().index.min()) for source in sources.values())
    common_end = min(_source_end(source) for source in sources.values())
    common_start = max(common_start, pd.Timestamp(equities.dropna().index.min()))
    common_end = min(common_end, pd.Timestamp(equities.dropna().index.max()))
    if common_end <= common_start:
        raise ValueError(f"Equity inputs have no common sample: {common_start} to {common_end}")

    target_specs = {
        "albemarle_equity": ("Albemarle_Close", "Ganfeng_Lithium_Close"),
        "ganfeng_equity": ("Ganfeng_Lithium_Close", "Albemarle_Close"),
    }
    LITHIUM_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for target, (target_column, other_equity) in target_specs.items():
        calendar = equities.loc[common_start:common_end, target_column].dropna().index
        panel = equities.loc[calendar, target_column].rename("Target_Close").to_frame()
        panel = panel.join(_asof_columns(calendar, carbonate))
        panel = panel.join(_asof_columns(calendar, hydroxide))
        panel = panel.join(_asof_columns(calendar, equities[[other_equity]]))
        panel = panel.join(_asof_columns(calendar, trends))
        panel = panel.join(_asof_columns(calendar, news))
        panel = panel.join(_asof_columns(calendar, market))
        panel = panel.join(_asof_columns(calendar, mvis))
        panel = panel.join(_asof_columns(calendar, china))
        panel = panel.join(_asof_columns(calendar, usd_cny))
        panel.index.name = "Date"
        output = Path(EQUITY_TARGETS[target]["processed"])
        panel.to_csv(output)
        outputs[target] = output

    equity_manifest: dict[str, Any] = {
        "study": "AMDR-Li lithium-producer equity extension",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_classification": "noncausal static 80:20 retrospective prediction",
        "common_sample": {"start": str(common_start.date()), "end": str(common_end.date())},
        "targets": {name: {**EQUITY_TARGETS[name], "processed": str(path), "coverage": _coverage(pd.read_csv(path, index_col="Date", parse_dates=["Date"]))} for name, path in outputs.items()},
        "feature_design": {
            "price_links": ["Lithium_Carbonate_Close", "Lithium_Hydroxide_Close"],
            "producer_equity": "other producer close only; own historical information enters through target lags",
            "equity_controls": ["SP500", "VIX", "US_Dollar_Index", "MVIS_Critical_Minerals", "Shanghai_Composite", "USD_CNY"],
            "unstructured": ["five Google Trends series", "three lithium-news aggregates"],
        },
        "source_files": {"usd_cny_fred_dexchus": {"path": str(USD_CNY_FILE.resolve()), "sha256": _sha256(USD_CNY_FILE)}},
        "alignment": "Latest available values are aligned to each target calendar; the model then shifts all external predictors by one observation.",
    }
    manifest_path = LITHIUM_MANIFEST.with_name("equity_extension_source_manifest.json")
    manifest_path.write_text(json.dumps(equity_manifest, indent=2, default=str), encoding="utf-8")
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    for name, path in build_equity_panels(require_complete=not args.allow_incomplete).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
