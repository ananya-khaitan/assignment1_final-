"""Configuration for the causal critical-mineral forecasting pipeline.

Paths are deliberately explicit.  Set ``CRITICAL_MINERALS_GROUP1_DATA_DIR``
to the directory containing the four macro/sentiment source files; the
pipeline never guesses a machine-specific location or silently substitutes
generated outputs for raw inputs.
"""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

OUT_ROOT = PROJECT_ROOT / "outputs"
OUT_DATA = OUT_ROOT / "data"
OUT_FIG = OUT_ROOT / "figures"
OUT_TAB = OUT_ROOT / "tables"
OUT_FORECASTS = OUT_ROOT / "forecasts"

for directory in (OUT_DATA, OUT_FIG, OUT_TAB, OUT_FORECASTS):
    directory.mkdir(parents=True, exist_ok=True)

# The project-owned directory is the canonical location.  The bundled
# workbooks remain a temporary compatibility source until they are moved.
RAW_DATA_DIR = Path(os.environ.get("CRITICAL_MINERALS_RAW_DATA_DIR", HERE / "data")).expanduser()
GROUP1_RAW = Path(os.environ.get("CRITICAL_MINERALS_GROUP1_DATA_DIR", RAW_DATA_DIR)).expanduser()


def _resolve_project_input(filename: str) -> Path:
    """Return a project-owned source path without falling back to user folders."""
    candidates = (
        RAW_DATA_DIR / filename,
        HERE / "CriticalMinerals_Pipeline_20260328_022036" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return the canonical expected path so validation can give a useful error.
    return candidates[0]


REFINITIV_FILE = _resolve_project_input("Top Critical Mineral Companies & Stock Price 2026.xlsx")
MVIS_FILE = _resolve_project_input("MVIS Global Rare Earth OR Strategic Metals (Pr) Index_Benchmark Index.xlsx")
BULK_YAHOO_FILE = GROUP1_RAW / "Bulk_Yahoo_Historical_Data.csv"
ALIGNED_FILE = GROUP1_RAW / "aligned_dataset.csv"
SENTIMENT_FILE = GROUP1_RAW / "news_sentiment_data.xlsx"
TRENDS_FILE = GROUP1_RAW / "rare_earth_trends.csv"
# Optional, but required before factor-adjusted trading claims are made.
FACTOR_FILE = Path(os.environ["CRITICAL_MINERALS_FACTOR_FILE"]).expanduser() if "CRITICAL_MINERALS_FACTOR_FILE" in os.environ else None
FACTOR_DIR = Path(os.environ.get("CRITICAL_MINERALS_FACTOR_DIR", RAW_DATA_DIR / "factors")).expanduser()
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


def required_raw_inputs() -> dict[str, Path]:
    return {
        "Refinitiv prices": REFINITIV_FILE,
        "MVIS benchmark": MVIS_FILE,
        "Yahoo macro series": BULK_YAHOO_FILE,
        "aligned macro series": ALIGNED_FILE,
        "news sentiment": SENTIMENT_FILE,
        "Google Trends": TRENDS_FILE,
    }


def required_factor_inputs() -> dict[str, Path]:
    """Regional factor sources used by the published trading evaluation."""
    return {
        "North America FF5 factors": FACTOR_DIR / "north_america_ff5.csv",
        "Europe FF5 factors": FACTOR_DIR / "europe_ff5.csv",
        "Developed ex-US FF5 factors": FACTOR_DIR / "developed_ex_us_ff5.csv",
        "Emerging FF5 factors": FACTOR_DIR / "emerging_monthly_ff5.csv",
        "company factor-region mapping": FACTOR_DIR / "company_factor_regions.json",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_inputs(require_manifest: bool = True) -> None:
    all_inputs = {**required_raw_inputs(), **required_factor_inputs()}
    missing = {label: path for label, path in all_inputs.items() if not path.exists()}
    if missing:
        detail = "\n".join(f"  - {label}: {path}" for label, path in missing.items())
        raise FileNotFoundError(
            "Raw inputs are incomplete. Restore the files below (or set "
            "CRITICAL_MINERALS_GROUP1_DATA_DIR) before rebuilding data:\n" + detail
        )
    manifest_path = RAW_DATA_DIR / "source_manifest.json"
    if not require_manifest:
        return
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Raw-data provenance manifest is required for a run: {manifest_path}. "
            "Run `python -m data_sources.download_public_inputs` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "Refinitiv prices": "refinitiv_prices",
        "MVIS benchmark": "mvis_benchmark",
        "Yahoo macro series": "yahoo_macro",
        "aligned macro series": "aligned_macro",
        "news sentiment": "sf_fed_sentiment",
        "Google Trends": "google_trends",
        "North America FF5 factors": "ken_french_north_america",
        "Europe FF5 factors": "ken_french_europe",
        "Developed ex-US FF5 factors": "ken_french_developed_ex_us",
        "Emerging FF5 factors": "ken_french_emerging_monthly",
        "company factor-region mapping": "factor_region_map",
    }
    errors = []
    for label, key in expected.items():
        entry = manifest.get(key)
        if not isinstance(entry, dict) or "sha256" not in entry:
            errors.append(f"manifest record absent: {key}")
            continue
        actual = _file_sha256(all_inputs[label])
        if actual != entry["sha256"]:
            errors.append(f"checksum mismatch: {label}")
    if errors:
        raise ValueError("Raw-data provenance verification failed: " + "; ".join(errors))


TICKER_TO_COMPANY = {
    "GLEN.L": "Glencore",
    "BHP.AX": "BHP Group",
    "RIO.AX": "Rio Tinto",
    "FCX": "Freeport-McMoRan",
    "ALB": "Albemarle",
    "2899.HK": "Zijin Mining",
    "002460.SZ": "Ganfeng Lithium",
    "AAL.L": "Anglo American",
}
COMPANY_TO_TICKER = {value: key for key, value in TICKER_TO_COMPANY.items()}
ORDER = list(TICKER_TO_COMPANY.values())
SLUG = {company: company.lower().replace(" ", "_").replace("-", "_") for company in ORDER}

EXOG_COLS = [
    "mvis_critical_minerals",
    "SP500",
    "Shanghai_Index",
    "Crude_Oil",
    "VIX",
    "US_Dollar_Index",
    "Search_Index",
    "News_Sentiment",
]

# Decomposition
VMD_ALPHA = 2000
VMD_TAU = 0
VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7
VMD_K = 5
CEEMDAN_TRIALS = 20
ALLOW_APPROXIMATE_DECOMPOSITION = False  # Never enable for reported results.

# Feature selection / model routing
N_LAGS = 5
CV_FOLDS = 5
AE_M = 2
AE_R_COEF = 0.2

# Operationally realistic causal re-estimation.  Forecasts remain one-step
# and are emitted on every test date; only model parameters are refreshed
# every 20 trading observations (approximately monthly), never with future
# observations.  This policy is recorded in every experiment registry.
MODEL_REFIT_INTERVAL = 20
RNN_EPOCHS = 5
MODEL_THREADS = 1
TORCH_THREADS = 1
# The benchmark/decomposition workers use native numerical kernels in addition
# to Python multiprocessing. Two concurrent companies avoid severe oversub-
# scription on typical laptop CPUs while preserving independent company runs.
MAX_PARALLEL_COMPANIES = int(os.environ.get("CRITICAL_MINERALS_MAX_PARALLEL_COMPANIES", "2"))
# A first scheduled fit is expensive, so expose observed throughput promptly.
PROGRESS_EVENT_INTERVAL = 5

# Paper-style causal pilot.  This is a separately labelled, single-company
# experiment inspired by Liu et al. (2025): VMD, two low-frequency ARIMA
# modes, and LASSO-selected multivariate LSTMs for the remaining modes.  It
# never shares outputs or selection results with the canonical experiment.
PAPER_STYLE_PILOT_COMPANY = "Glencore"
PAPER_STYLE_PILOT_CALIBRATION_SIZE = 150
PAPER_STYLE_PILOT_TEST_SIZE = 250
PAPER_STYLE_PILOT_VMD_K = 8
PAPER_STYLE_PILOT_LOW_MODES = 2
PAPER_STYLE_PILOT_REFIT_INTERVAL = 20
PAPER_STYLE_PILOT_LSTM_EPOCHS = 30
PAPER_STYLE_PILOT_LSTM_HIDDEN = 64
PAPER_STYLE_PILOT_LSTM_WINDOW = 20

# Evaluation design.  The final 20% is never used for calibration or tuning.
TEST_FRAC = 0.20
MIN_TRAIN_SIZE = 120
CALIBRATION_SIZE = 60
CONFORMAL_ALPHA = 0.10
TRANSACTION_COSTS = (0.0, 0.0010, 0.0025, 0.0050)
THRESHOLD_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
ANNUALIZATION = 252
RANDOM_SEED = 42

# Full benchmark labels must correspond to real installed implementations.
REQUIRED_MODEL_PACKAGES = {
    "statsmodels": "ARIMA",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "torch": "LSTM/GRU",
    "vmdpy": "VMD",
    "PyEMD": "CEEMDAN",
}
