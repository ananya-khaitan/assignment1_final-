"""Collect or import lithium-specific Google Trends and news headlines.

Google Trends has no generally available official API for arbitrary terms.
The automatic route uses the same unofficial client already pinned by the
project, while ``--trends-csv`` supports the official Explore-page CSV export.
Historical headline archives require an entitled provider export; the script
does not mislabel the generic SF Fed index or a recent-only API sample as a
2021-present lithium news history.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data_sources.prepare_lithium_inputs import read_lithium_news, read_lithium_trends
from lithium_config import (
    GOOGLE_TRENDS_TERMS,
    LITHIUM_EXTERNAL_DIR,
    LITHIUM_NEWS_FILE,
    LITHIUM_TRENDS_FILE,
    NEWS_QUERY_TERMS,
)


def _import_csv(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _import_news_csv(source: Path, destination: Path) -> dict[str, object]:
    """Normalize a canonical or Media Cloud content export for the pipeline."""
    if not source.exists():
        raise FileNotFoundError(source)
    raw = pd.read_csv(source)
    input_rows = len(raw)
    media_cloud_columns = {"id", "language", "media_name", "publish_date", "title", "url"}
    if media_cloud_columns.issubset(raw.columns):
        raw = raw.loc[raw["language"].astype(str).str.lower().eq("en")].copy()
        raw = raw.rename(
            columns={
                "publish_date": "published_at_utc",
                "media_name": "source",
                "id": "provider_record_id",
            }
        )
        keep = [
            "published_at_utc",
            "title",
            "source",
            "url",
            "provider_record_id",
            "indexed_date",
            "language",
            "media_url",
        ]
        raw = raw.loc[:, [column for column in keep if column in raw.columns]]
        provider = "Media Cloud Online News"
    else:
        required = {"published_at_utc", "title", "source", "url"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"News CSV is missing columns: {sorted(missing)}")
        provider = "canonical headline CSV"
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(destination, index=False)
    return {"provider": provider, "input_rows": input_rows, "normalized_rows": len(raw)}


def download_google_trends(start: str, end: str) -> Path:
    try:
        from pytrends.request import TrendReq
    except ImportError as exc:
        raise ImportError("Install pytrends from requirements.txt or provide --trends-csv") from exc
    client = TrendReq(hl="en-US", tz=0, timeout=(10, 60), retries=0)
    timeframe = f"{start} {end}"
    try:
        client.build_payload(list(GOOGLE_TRENDS_TERMS), cat=0, timeframe=timeframe, geo="", gprop="")
        interest = client.interest_over_time()
    except Exception as exc:
        raise RuntimeError(
            "Google Trends did not allow automated retrieval. Export the five-term comparison from "
            "https://trends.google.com/trends/explore and rerun with --trends-csv PATH. "
            f"Underlying response: {type(exc).__name__}: {exc}"
        ) from exc
    if interest.empty:
        raise ValueError("Google Trends returned an empty comparison")
    missing = [term for term in GOOGLE_TRENDS_TERMS if term not in interest]
    if missing:
        raise ValueError(f"Google Trends response omitted terms: {missing}")
    output = interest.loc[:, list(GOOGLE_TRENDS_TERMS)].rename_axis("Date").reset_index()
    LITHIUM_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(LITHIUM_TRENDS_FILE, index=False)
    return LITHIUM_TRENDS_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-05-04")
    parser.add_argument("--end", default="2026-03-12")
    parser.add_argument("--trends-csv", type=Path, help="Official Google Trends comparison CSV")
    parser.add_argument("--news-csv", type=Path, help="Entitled historical lithium-headline export")
    parser.add_argument("--automatic-trends", action="store_true", help="Attempt retrieval through the pinned pytrends client")
    args = parser.parse_args()
    if not any((args.trends_csv, args.news_csv, args.automatic_trends)):
        parser.error("provide --trends-csv, --news-csv, or --automatic-trends")

    acquisition_path = LITHIUM_EXTERNAL_DIR / "acquisition_metadata.json"
    if acquisition_path.exists():
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    else:
        acquisition = {}
    acquisition["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.trends_csv:
        _import_csv(args.trends_csv, LITHIUM_TRENDS_FILE)
        read_lithium_trends(LITHIUM_TRENDS_FILE)
        acquisition["google_trends"] = {
            "method": "official Explore-page CSV import",
            "source": "https://trends.google.com/trends/explore",
            "input": str(args.trends_csv.resolve()),
            "queries": list(GOOGLE_TRENDS_TERMS),
            "geo": "worldwide",
        }
    elif args.automatic_trends:
        path = download_google_trends(args.start, args.end)
        read_lithium_trends(path)
        acquisition["google_trends"] = {
            "method": "pytrends retrieval",
            "source": "https://trends.google.com/trends/explore",
            "queries": list(GOOGLE_TRENDS_TERMS),
            "geo": "worldwide",
            "timeframe": f"{args.start} {args.end}",
        }

    if args.news_csv:
        import_details = _import_news_csv(args.news_csv, LITHIUM_NEWS_FILE)
        raw, daily = read_lithium_news(LITHIUM_NEWS_FILE, score_finbert=False)
        # Store the validated, headline-relevant subset plus its inexpensive
        # TextBlob score.  The raw provider export remains alongside it for
        # reproducibility and future filtering changes.
        raw.drop(columns=["Date"], errors="ignore").to_csv(LITHIUM_NEWS_FILE, index=False)
        acquisition["lithium_news"] = {
            "method": "exported headline CSV import",
            "input": str(args.news_csv.resolve()),
            "queries": list(NEWS_QUERY_TERMS),
            **import_details,
            "headline_rows": int(len(raw)),
            "first_date": str(daily.index.min().date()),
            "last_date": str(daily.index.max().date()),
        }

    LITHIUM_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    acquisition_path.write_text(json.dumps(acquisition, indent=2), encoding="utf-8")
    print(f"Validated lithium unstructured inputs; metadata: {acquisition_path}")


if __name__ == "__main__":
    main()
