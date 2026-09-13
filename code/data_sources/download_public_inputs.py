"""Acquire, validate, and document public raw inputs for a reproducible run.

Sources are fixed, explicit, and stored with SHA-256 hashes. The script does
not call the forecasting pipeline and never writes model output.

Run from ``code/`` after installing requirements:
    python -m data_sources.download_public_inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import requests

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config import MVIS_FILE, RAW_DATA_DIR, REFINITIV_FILE


# ``datetime.UTC`` was added in Python 3.11.  The pinned stack also supports
# WSL Ubuntu's Python 3.10, so keep the provenance timestamp explicitly UTC
# without narrowing the supported interpreter unnecessarily.
UTC = timezone.utc


SF_FED_SENTIMENT_URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
YAHOO_SYMBOLS = {
    "^GSPC": "SP500",
    "000001.SS": "Shanghai_Index",
    "CL=F": "Crude_Oil",
    "^VIX": "VIX",
    "DX-Y.NYB": "US_Dollar_Index",
}
GOOGLE_TRENDS_QUERY = {"term": "rare earth", "geo": "", "category": 0, "property": ""}
KEN_FRENCH_URLS = {
    "north_america": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/North_America_5_Factors_Daily_CSV.zip",
    "europe": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_Daily_CSV.zip",
    "developed_ex_us": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_ex_US_5_Factors_Daily_CSV.zip",
    # Ken French publishes the public emerging five-factor series monthly.
    "emerging_monthly": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Emerging_5_Factors_CSV.zip",
}
FACTOR_REGION_BY_COMPANY = {
    "Freeport-McMoRan": "north_america",
    "Albemarle": "north_america",
    "Glencore": "europe",
    "Anglo American": "europe",
    "BHP Group": "developed_ex_us",
    "Rio Tinto": "developed_ex_us",
    "Zijin Mining": "emerging_monthly",
    "Ganfeng Lithium": "emerging_monthly",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(manifest: dict[str, Any], name: str, path: Path, **metadata: Any) -> None:
    manifest[name] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        **metadata,
    }


def _sample_end_date() -> pd.Timestamp:
    frame = pd.read_excel(REFINITIV_FILE, sheet_name="Stock Price", skiprows=[1], usecols=[0])
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Could not determine the Refinitiv sample end date")
    return pd.Timestamp(dates.max()).normalize()


def _copy_workbook(source: Path, output_dir: Path, manifest: dict[str, Any], label: str) -> None:
    target = output_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    _record(manifest, label, target, source="Project-provided workbook")


def _download_yahoo(output_dir: Path, start: pd.Timestamp, end: pd.Timestamp, manifest: dict[str, Any]) -> None:
    import yfinance as yf

    raw = yf.download(
        list(YAHOO_SYMBOLS),
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=False,
        progress=False,
    )
    series: dict[str, pd.Series] = {}
    for symbol in YAHOO_SYMBOLS:
        try:
            close = pd.to_numeric(raw[(symbol, "Close")], errors="coerce").dropna()
        except KeyError as exc:
            raise ValueError(f"Yahoo response has no Close field for {symbol}") from exc
        if close.empty:
            raise ValueError(f"Yahoo response has no observations for {symbol}")
        series[symbol] = close
    data = pd.concat(series, axis=1).sort_index()
    data.index.name = "Date"
    data.columns = pd.MultiIndex.from_tuples([(symbol, "Close") for symbol in data.columns])
    bulk_path = output_dir / "Bulk_Yahoo_Historical_Data.csv"
    data.to_csv(bulk_path)
    _record(
        manifest,
        "yahoo_macro",
        bulk_path,
        source="Yahoo Finance queried through yfinance",
        symbols=YAHOO_SYMBOLS,
        field="Close, unadjusted",
        sample_start=str(start.date()),
        sample_end=str(end.date()),
    )
    aligned = pd.DataFrame(
        {
            "Date": data.index,
            "VIX": data[("^VIX", "Close")].to_numpy(),
            "US_Dollar_Index": data[("DX-Y.NYB", "Close")].to_numpy(),
        }
    )
    aligned_path = output_dir / "aligned_dataset.csv"
    aligned.to_csv(aligned_path, index=False)
    _record(
        manifest,
        "aligned_macro",
        aligned_path,
        source="Derived solely from yahoo_macro",
        fields={"VIX": "^VIX Close", "US_Dollar_Index": "DX-Y.NYB Close"},
    )


def _download_sentiment(output_dir: Path, manifest: dict[str, Any]) -> None:
    response = requests.get(SF_FED_SENTIMENT_URL, timeout=60)
    response.raise_for_status()
    path = output_dir / "news_sentiment_data.xlsx"
    path.write_bytes(response.content)
    source = pd.read_excel(path, sheet_name="Data")
    if not {"date", "News Sentiment"}.issubset(source.columns):
        raise ValueError("SF Fed workbook has an unexpected schema")
    _record(
        manifest,
        "sf_fed_sentiment",
        path,
        source=SF_FED_SENTIMENT_URL,
        sheet="Data",
        expected_columns=["date", "News Sentiment"],
    )


def _download_trends(output_dir: Path, start: pd.Timestamp, end: pd.Timestamp, manifest: dict[str, Any]) -> None:
    from pytrends.request import TrendReq

    client = TrendReq(hl="en-US", tz=0, timeout=(10, 60), retries=3, backoff_factor=0.2)
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    client.build_payload([GOOGLE_TRENDS_QUERY["term"]], cat=0, timeframe=timeframe, geo="", gprop="")
    interest = client.interest_over_time()
    term = GOOGLE_TRENDS_QUERY["term"]
    if interest.empty or term not in interest.columns:
        raise ValueError("Google Trends returned no interest-over-time observations for 'rare earth'")
    output = interest[[term]].rename_axis("time").reset_index()
    output["time"] = pd.to_datetime(output["time"], errors="coerce").dt.tz_localize(None)
    if output["time"].isna().any():
        raise ValueError("Google Trends response has invalid time observations")
    path = output_dir / "rare_earth_trends.csv"
    output.to_csv(path, index=False)
    _record(
        manifest,
        "google_trends",
        path,
        source="https://trends.google.com/trends/explore",
        query=GOOGLE_TRENDS_QUERY,
        timeframe=timeframe,
        notes="Worldwide Google web-search indexed interest at the frequency returned by Google Trends.",
    )


def _parse_ken_french_zip(content: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(content)) as archive:
        name = next(member for member in archive.namelist() if member.lower().endswith(".csv"))
        lines = archive.read(name).decode("utf-8", errors="replace").splitlines()
    header = next(index for index, line in enumerate(lines) if line.strip().startswith(",") and "Mkt-RF" in line)
    rows = []
    for line in lines[header:]:
        if not line.strip():
            break
        rows.append(line)
    frame = pd.read_csv(StringIO("\n".join(rows)))
    frame = frame.rename(columns={frame.columns[0]: "Date"})
    date_tokens = frame["Date"].astype(str).str.strip()
    # pandas permits a six-digit ``YYYYMM`` token with an eight-digit format
    # and interprets it as consecutive January days (e.g. 198902 -> 1989-01-02).
    # Determine Ken French frequency from the raw token width *before* parsing
    # so monthly emerging-market factors retain their real monthly dates.
    widths = set(date_tokens.str.len().unique())
    if widths == {8}:
        frequency = "daily"
        frame["Date"] = pd.to_datetime(date_tokens, format="%Y%m%d", errors="coerce")
    elif widths == {6}:
        frequency = "monthly"
        frame["Date"] = pd.to_datetime(date_tokens, format="%Y%m", errors="coerce")
    else:
        raise ValueError(f"Ken French factor file has unsupported or mixed date token widths: {sorted(widths)}")
    frame = frame.dropna(subset=["Date"])
    frame.columns = [str(column).strip() for column in frame.columns]
    required = ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    if any(column not in frame for column in required):
        raise ValueError("Ken French factor file has an unexpected schema")
    for column in required[1:]:
        values = pd.to_numeric(frame[column], errors="coerce")
        # Ken French uses values such as -99.99 to mark unavailable factors.
        # Retain them as missing rather than treating them as a -99.99% return.
        frame[column] = values.where(values > -99.0) / 100.0
    output = frame[required].dropna().sort_values("Date")
    output.attrs["frequency"] = frequency
    return output


def _download_factors(output_dir: Path, manifest: dict[str, Any]) -> None:
    factor_dir = output_dir / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)
    for region, url in KEN_FRENCH_URLS.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = _parse_ken_french_zip(response.content)
        path = factor_dir / f"{region}_ff5.csv"
        data.to_csv(path, index=False)
        _record(
            manifest,
            f"ken_french_{region}",
            path,
            source=url,
            frequency=data.attrs["frequency"],
            companies=[company for company, mapped in FACTOR_REGION_BY_COMPANY.items() if mapped == region],
        )
    map_path = factor_dir / "company_factor_regions.json"
    map_path.write_text(json.dumps(FACTOR_REGION_BY_COMPANY, indent=2), encoding="utf-8")
    _record(manifest, "factor_region_map", map_path, source="Research-design mapping")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download documented public research inputs.")
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--factors-only",
        action="store_true",
        help="Refresh only Ken French factor files and their provenance records in an existing manifest.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_manifest.json"
    if args.factors_only:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cannot refresh factors without the existing raw-data manifest: {manifest_path}. "
                "Run the full downloader first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _download_factors(output_dir, manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Refreshed documented factor inputs and provenance manifest: {manifest_path}")
        return
    start, end = pd.Timestamp("2011-05-01"), _sample_end_date()
    manifest: dict[str, Any] = {
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "retrieval_script": str(Path(__file__).resolve()),
    }
    _copy_workbook(REFINITIV_FILE, output_dir, manifest, "refinitiv_prices")
    _copy_workbook(MVIS_FILE, output_dir, manifest, "mvis_benchmark")
    _download_yahoo(output_dir, start, end, manifest)
    _download_sentiment(output_dir, manifest)
    _download_trends(output_dir, start, end, manifest)
    _download_factors(output_dir, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote documented raw inputs and provenance manifest: {manifest_path}")


if __name__ == "__main__":
    main()
