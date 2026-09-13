"""Build the causally safe daily master dataset from versioned raw inputs."""

from __future__ import annotations

import os

import pandas as pd

from config import (
    ALIGNED_FILE,
    BULK_YAHOO_FILE,
    EXOG_COLS,
    MVIS_FILE,
    OUT_DATA,
    REFINITIV_FILE,
    SENTIMENT_FILE,
    SLUG,
    TICKER_TO_COMPANY,
    TRENDS_FILE,
    validate_raw_inputs,
)


START_DATE = "2011-05-19"


def _read_refinitiv_prices() -> pd.DataFrame:
    source = pd.read_excel(REFINITIV_FILE, sheet_name="Stock Price", skiprows=[1])
    source.columns = [str(column).replace(" (TRDPRC_1)", "").strip() for column in source.columns]
    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")
    source = source.dropna(subset=["Date"]).set_index("Date").sort_index()
    missing = [ticker for ticker in TICKER_TO_COMPANY if ticker not in source.columns]
    if missing:
        raise ValueError(f"Refinitiv workbook is missing required tickers: {missing}")
    result = pd.DataFrame(index=source.index)
    for ticker, company in TICKER_TO_COMPANY.items():
        result[company] = pd.to_numeric(source[ticker], errors="coerce")
    return result


def _read_mvis_close() -> pd.Series:
    """Read the labelled Close column rather than guessing an OHLC field."""
    raw = pd.read_excel(MVIS_FILE, sheet_name="Sheet 1", header=None)
    header_row = next(
        (
            row_index
            for row_index, value in raw.iloc[:, 0].items()
            if isinstance(value, str) and value.strip().lower() == "exchange date"
        ),
        None,
    )
    if header_row is None:
        raise ValueError("MVIS workbook does not contain an 'Exchange Date' header row")
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = [str(value).strip() for value in raw.iloc[header_row]]
    if not {"Exchange Date", "Close"}.issubset(data.columns):
        raise ValueError("MVIS workbook must contain labelled 'Exchange Date' and 'Close' columns")
    data["Exchange Date"] = pd.to_datetime(data["Exchange Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    series = data.dropna(subset=["Exchange Date", "Close"]).set_index("Exchange Date")["Close"]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "mvis_critical_minerals"
    if series.empty:
        raise ValueError("MVIS Close series is empty after parsing")
    return series


def _yahoo_close(frame: pd.DataFrame, ticker: str, label: str) -> pd.Series:
    key = (ticker, "Close")
    if key not in frame.columns:
        raise ValueError(f"Yahoo source lacks required column {key}")
    series = pd.to_numeric(frame[key], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Yahoo source has no usable observations for {ticker}")
    series.name = label
    return series


def _read_public_predictors() -> list[pd.Series]:
    yahoo = pd.read_csv(BULK_YAHOO_FILE, header=[0, 1], index_col=0, parse_dates=True)
    yahoo.index.name = "Date"
    sp500 = _yahoo_close(yahoo, "^GSPC", "SP500")
    shanghai = _yahoo_close(yahoo, "000001.SS", "Shanghai_Index")
    crude = _yahoo_close(yahoo, "CL=F", "Crude_Oil")

    aligned = pd.read_csv(ALIGNED_FILE, parse_dates=["Date"]).set_index("Date")
    required_aligned = {"VIX", "US_Dollar_Index"}
    missing_aligned = required_aligned.difference(aligned.columns)
    if missing_aligned:
        raise ValueError(f"aligned_dataset.csv is missing {sorted(missing_aligned)}")
    vix = pd.to_numeric(aligned["VIX"], errors="coerce").dropna().rename("VIX")
    dollar = pd.to_numeric(aligned["US_Dollar_Index"], errors="coerce").dropna().rename("US_Dollar_Index")

    trends = pd.read_csv(TRENDS_FILE)
    trends.columns = [str(column).strip().lower() for column in trends.columns]
    if not {"time", "rare earth"}.issubset(trends.columns):
        raise ValueError("rare_earth_trends.csv must contain 'time' and 'rare earth' columns")
    trends["time"] = pd.to_datetime(trends["time"], errors="coerce")
    search = pd.to_numeric(trends.set_index("time")["rare earth"], errors="coerce").dropna().sort_index()
    search = search.resample("D").ffill().rename("Search_Index")

    sentiment = pd.read_excel(SENTIMENT_FILE, sheet_name="Data")
    if not {"date", "News Sentiment"}.issubset(sentiment.columns):
        raise ValueError("SF Fed workbook must contain 'date' and 'News Sentiment' columns")
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    news = pd.to_numeric(sentiment.set_index("date")["News Sentiment"], errors="coerce").dropna().sort_index()
    news = news[~news.index.duplicated(keep="last")].rename("News_Sentiment")
    return [sp500, shanghai, crude, vix, dollar, search, news]


def main() -> None:
    validate_raw_inputs()
    print("=" * 60)
    print("BUILD MASTER: CAUSAL CRITICAL-MINERAL INPUTS")
    print("=" * 60)
    company_close = _read_refinitiv_prices()
    print(f"Refinitiv: {company_close.shape}; {company_close.index.min().date()} to {company_close.index.max().date()}")

    for company in company_close.columns:
        prices = company_close[company].dropna().rename("Close").to_frame()
        prices["High"] = prices["Close"]
        prices["Low"] = prices["Close"]
        target = os.path.join(OUT_DATA, f"{SLUG[company]}_daily.csv")
        prices.to_csv(target)

    master = company_close.copy()
    for series in [_read_mvis_close(), *_read_public_predictors()]:
        master = master.join(series, how="outer")
    master = master.sort_index().loc[START_DATE : company_close.index.max()]
    master.index.name = "Date"
    # Only propagate values forward from information that was already observed.
    # We do not impose a cap: trading-calendar closures can legitimately last
    # longer than five days, and source availability is recorded in the manifest.
    master = master.ffill()
    required = list(TICKER_TO_COMPANY.values()) + EXOG_COLS
    missing_latest = [column for column in required if column not in master or pd.isna(master[column].iloc[-1])]
    if missing_latest:
        raise ValueError(f"Master dataset has no current usable value for {missing_latest}")
    master.to_csv(os.path.join(OUT_DATA, "master_daily_prices.csv"))
    print(f"Saved master_daily_prices.csv: {master.shape}; {master.index.min().date()} to {master.index.max().date()}")
    print(master[EXOG_COLS].isna().sum().to_string())


if __name__ == "__main__":
    main()
