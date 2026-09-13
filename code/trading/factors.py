"""CAPM, FF3, and FF5 alpha estimation from a user-supplied factor file."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


_ALIASES = {
    "date": "Date",
    "rf": "RF",
    "mkt-rf": "Mkt-RF",
    "mktrf": "Mkt-RF",
    "mkt_rf": "Mkt-RF",
    "smb": "SMB",
    "hml": "HML",
    "rmw": "RMW",
    "cma": "CMA",
}


def load_factor_data(path: str | Path) -> pd.DataFrame:
    """Load daily or monthly Fama--French-style factors and normalize units."""
    factors = pd.read_csv(path)
    factors = factors.rename(columns={column: _ALIASES.get(column.strip().lower(), column) for column in factors.columns})
    required = {"Date", "RF", "Mkt-RF"}
    missing = required.difference(factors.columns)
    if missing:
        raise ValueError(f"Factor file is missing required columns: {sorted(missing)}")
    factors["Date"] = pd.to_datetime(factors["Date"], errors="coerce")
    factors = factors.dropna(subset=["Date"]).set_index("Date").sort_index()
    numeric_columns = [column for column in factors.columns if column in {"RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"}]
    factors[numeric_columns] = factors[numeric_columns].apply(pd.to_numeric, errors="coerce")
    # Standard Fama--French downloads are in percent; preserve decimal files.
    if factors[numeric_columns].abs().stack().median() > 0.2:
        factors[numeric_columns] = factors[numeric_columns] / 100.0
    if len(factors) < 2:
        raise ValueError("Factor file needs at least two dated observations")
    median_gap = factors.index.to_series().diff().dt.days.median()
    factors.attrs["frequency"] = "monthly" if median_gap and median_gap > 10 else "daily"
    return factors


def _fit_alpha(excess_returns: pd.Series, factors: pd.DataFrame, columns: list[str], annualization: int) -> dict[str, float]:
    import statsmodels.api as sm

    frame = pd.concat([excess_returns.rename("excess"), factors[columns]], axis=1).dropna()
    # The final test has roughly three years of daily data.  Emerging-market
    # factors are correctly matched at monthly frequency, yielding about
    # 37 observations: enough for the six FF5 regression coefficients and a
    # HAC estimate, but not enough for an arbitrary 40-observation rule.
    if len(frame) < 30:
        return {"alpha_annualized": float("nan"), "alpha_pvalue": float("nan"), "n_obs": float(len(frame))}
    fitted = sm.OLS(frame["excess"], sm.add_constant(frame[columns])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "alpha_annualized": float(fitted.params["const"] * annualization),
        "alpha_pvalue": float(fitted.pvalues["const"]),
        "n_obs": float(len(frame)),
    }


def _align_returns_to_factor_frequency(returns: pd.Series, factors: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, int]:
    frequency = factors.attrs.get("frequency", "daily")
    if frequency == "daily":
        return returns, factors, 252
    # Emerging-market FF5 is public at monthly frequency. Compound the
    # causally generated daily strategy returns to the same observation unit.
    monthly_returns = returns.groupby(returns.index.to_period("M")).apply(lambda values: (1.0 + values).prod() - 1.0)
    monthly_returns.index = monthly_returns.index.to_timestamp()
    normalized = factors.copy()
    normalized.index = normalized.index.to_period("M").to_timestamp()
    return monthly_returns, normalized, 12


def factor_adjusted_alphas(
    dates: Sequence[pd.Timestamp], strategy_returns: Sequence[float], factor_data: pd.DataFrame, annualization: int = 252
) -> dict[str, float]:
    """Estimate CAPM/FF3/FF5 annualized alpha at compatible data frequency."""
    returns = pd.Series(np.asarray(strategy_returns, dtype=float), index=pd.to_datetime(dates), name="strategy")
    returns, factor_data, frequency_annualization = _align_returns_to_factor_frequency(returns, factor_data)
    merged = returns.to_frame().join(factor_data, how="inner").dropna(subset=["strategy", "RF", "Mkt-RF"])
    excess = merged["strategy"] - merged["RF"]
    result: dict[str, float] = {"Factor Frequency": factor_data.attrs.get("frequency", "daily")}
    for name, columns in (("CAPM", ["Mkt-RF"]), ("FF3", ["Mkt-RF", "SMB", "HML"]), ("FF5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"])):
        if not set(columns).issubset(merged.columns):
            result[f"{name} alpha"] = float("nan")
            result[f"{name} alpha p-value"] = float("nan")
            continue
        fitted = _fit_alpha(excess, merged, columns, frequency_annualization)
        result[f"{name} alpha"] = fitted["alpha_annualized"]
        result[f"{name} alpha p-value"] = fitted["alpha_pvalue"]
    return result
