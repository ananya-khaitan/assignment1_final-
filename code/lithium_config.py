"""Configuration for the Liu-style lithium-price forecasting study.

This module is intentionally separate from ``config.py`` so the historical
eight-company causal experiment remains reproducible and unchanged.
"""

from __future__ import annotations

from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DATA_ROOT = CODE_ROOT / "data"
LITHIUM_ROOT = DATA_ROOT / "lithium"
LITHIUM_RAW_DIR = LITHIUM_ROOT / "raw"
LITHIUM_EXTERNAL_DIR = LITHIUM_ROOT / "external"
LITHIUM_PROCESSED_DIR = LITHIUM_ROOT / "processed"
LITHIUM_MANIFEST = LITHIUM_ROOT / "source_manifest.json"

CARBONATE_WORKBOOK = DATA_ROOT / "lithium_carbonate_spot_smm.xlsx"
HYDROXIDE_WORKBOOK = DATA_ROOT / "lithium_hydroxide_futures_fastmarkets_comex.xlsx"
EQUITY_WORKBOOK = DATA_ROOT / "Top Critical Mineral Companies & Stock Price 2026.xlsx"
MVIS_WORKBOOK = DATA_ROOT / "MVIS Global Rare Earth OR Strategic Metals (Pr) Index_Benchmark Index.xlsx"
YAHOO_WORKBOOK = DATA_ROOT / "Bulk_Yahoo_Historical_Data.csv"

LITHIUM_TRENDS_FILE = LITHIUM_EXTERNAL_DIR / "lithium_google_trends.csv"
LITHIUM_NEWS_FILE = LITHIUM_EXTERNAL_DIR / "lithium_news_headlines.csv"
USD_CNY_FILE = LITHIUM_EXTERNAL_DIR / "usd_cny_fred_dexchus.csv"

TARGETS = {
    "MSFT_O": {
        "label": "MSFT.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "MSFT_O_panel.csv",
        "modes": 9,
    },
    "GOOGL_O": {
        "label": "GOOGL.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "GOOGL_O_panel.csv",
        "modes": 9,
    },
    "AMZN_O": {
        "label": "AMZN.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "AMZN_O_panel.csv",
        "modes": 9,
    },
    "TSLA_O": {
        "label": "TSLA.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "TSLA_O_panel.csv",
        "modes": 9,
    },
    "NVDA_O": {
        "label": "NVDA.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "NVDA_O_panel.csv",
        "modes": 9,
    },
    "META_O": {
        "label": "META.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "META_O_panel.csv",
        "modes": 9,
    },
    "ORCL_K": {
        "label": "ORCL.K Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "ORCL_K_panel.csv",
        "modes": 9,
    },
    "IBM": {
        "label": "IBM Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "IBM_panel.csv",
        "modes": 9,
    },
}


# The equity extension is deliberately kept separate from the two physical
# lithium-price targets.  It uses the same retrospective 80:20 protocol and
# model suite, but each listed producer is forecast against its own set of
# equity-appropriate inputs and benchmarks.
EQUITY_TARGETS = {
    "MSFT_O": {
        "label": "MSFT.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "MSFT_O_panel.csv",
        "modes": 9,
    },
    "GOOGL_O": {
        "label": "GOOGL.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "GOOGL_O_panel.csv",
        "modes": 9,
    },
    "AMZN_O": {
        "label": "AMZN.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "AMZN_O_panel.csv",
        "modes": 9,
    },
    "TSLA_O": {
        "label": "TSLA.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "TSLA_O_panel.csv",
        "modes": 9,
    },
    "NVDA_O": {
        "label": "NVDA.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "NVDA_O_panel.csv",
        "modes": 9,
    },
    "META_O": {
        "label": "META.O Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "META_O_panel.csv",
        "modes": 9,
    },
    "ORCL_K": {
        "label": "ORCL.K Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "ORCL_K_panel.csv",
        "modes": 9,
    },
    "IBM": {
        "label": "IBM Stock Price",
        "processed": LITHIUM_PROCESSED_DIR / "IBM_panel.csv",
        "modes": 9,
    },
}


EQUITY_TICKERS = {
    "ALB": "Albemarle",
    "002460.SZ": "Ganfeng_Lithium",
}

GOOGLE_TRENDS_TERMS = (
    "lithium",
    "lithium price",
    "lithium carbonate",
    "lithium hydroxide",
    "lithium battery",
)

NEWS_QUERY_TERMS = (
    "lithium price",
    "lithium carbonate",
    "lithium hydroxide",
    "lithium market",
    "lithium supply",
    "lithium demand",
)

# Media Cloud searches full article text, but the exported content contains
# only headline text.  Retain only headlines that explicitly mention lithium
# for sentiment scoring; this prevents company-only and generic mining stories
# from drifting the news signal away from the commodity target.
NEWS_HEADLINE_FILTER_TERMS = ("lithium",)

# Each block is tested incrementally.  The full model never hides which data
# family creates an apparent improvement in the static holdout.
FEATURE_BLOCKS = {
    "history_only": (),
    "cross_lithium": ("Cross_Lithium_Close",),
    "lithium_equities": ("Albemarle_Close", "Ganfeng_Lithium_Close"),
    "lithium_attention": tuple(f"Trends_{term.replace(' ', '_')}" for term in GOOGLE_TRENDS_TERMS),
    "lithium_news": (
        "News_Article_Count",
        "News_TextBlob_Polarity",
        "News_FinBERT_Score",
    ),
    "market_controls": ("SP500", "VIX", "US_Dollar_Index", "MVIS_Critical_Minerals"),
}

ABLATIONS = {
    "history_only": (),
    "history_plus_cross_lithium": ("cross_lithium",),
    "history_plus_equities": ("lithium_equities",),
    "history_plus_attention": ("lithium_attention",),
    "history_plus_news": ("lithium_news",),
    "history_plus_market_controls": ("market_controls",),
    "full_multimodal": (
        "cross_lithium",
        "lithium_equities",
        "lithium_attention",
        "lithium_news",
        "market_controls",
    ),
}

LIU_TEST_FRACTION = 0.20
LIU_HORIZON = 1
LIU_LAGS = 5
LIU_LOW_COMPLEXITY_MODES = 2
LIU_LSTM_EPOCHS = 100
# Defaults in the released forecasting model are hidden=40 and batch=128.
# The earlier lithium reconstruction used 64/64; that run remains preserved
# as an implementation-sensitivity comparator in outputs/.
LIU_LSTM_HIDDEN_SIZE = 40
LIU_LSTM_BATCH_SIZE = 128
LIU_RANDOM_SEED = 42

# Exact-parity and adaptive-routing settings.  The static experiment remains
# retrospective/noncausal because decomposition uses the complete target path,
# but all adaptive route and stacking decisions are locked on the development
# segment before the final 20% test segment is scored.
PARITY_LSTM_HIDDEN_SIZES = (40, 64)
PARITY_LSTM_BATCH_SIZES = (128, 64)
ADAPTIVE_VMD_K = (9,)
ADAPTIVE_VMD_ALPHA = (3000.0,)
ADAPTIVE_CEEMDAN_MAX_IMF = (9,)
ADAPTIVE_ROUTE_MODELS = ("ARIMA", "LSTM", "SVR", "RF")
ADAPTIVE_DEVELOPMENT_FRACTION = 0.20
ADAPTIVE_SEARCH_EPOCHS = 40
ADAPTIVE_FINAL_EPOCHS = 100
