from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from code.data_sources.prepare_lithium_inputs import read_lithium_news, read_lithium_trends
    from code.lithium_config import ABLATIONS, EQUITY_TICKERS, FEATURE_BLOCKS, GOOGLE_TRENDS_TERMS
    from code.run_liu_lithium import ablation_columns
except ModuleNotFoundError:  # Support running pytest from the code directory.
    from data_sources.prepare_lithium_inputs import read_lithium_news, read_lithium_trends
    from lithium_config import ABLATIONS, EQUITY_TICKERS, FEATURE_BLOCKS, GOOGLE_TRENDS_TERMS
    from run_liu_lithium import ablation_columns


def test_primary_equity_block_contains_only_direct_lithium_exposure():
    assert EQUITY_TICKERS == {"ALB": "Albemarle", "002460.SZ": "Ganfeng_Lithium"}
    assert FEATURE_BLOCKS["lithium_equities"] == ("Albemarle_Close", "Ganfeng_Lithium_Close")


def test_full_ablation_keeps_feature_blocks_segregated_and_includes_mvis():
    columns = ablation_columns("full_multimodal")
    assert len(columns) == len(set(columns))
    assert "MVIS_Critical_Minerals" in columns
    assert "Albemarle_Close" in columns
    assert "Ganfeng_Lithium_Close" in columns
    assert "News_TextBlob_Polarity" in columns
    assert set(ABLATIONS) == {
        "history_only",
        "history_plus_cross_lithium",
        "history_plus_equities",
        "history_plus_attention",
        "history_plus_news",
        "history_plus_market_controls",
        "full_multimodal",
    }


def test_weekly_google_trends_becomes_available_at_week_end(tmp_path: Path):
    data = {"Week": ["2024-01-01", "2024-01-08"]}
    for position, term in enumerate(GOOGLE_TRENDS_TERMS, start=1):
        data[term] = [position, position + 1]
    path = tmp_path / "trends.csv"
    pd.DataFrame(data).to_csv(path, index=False)

    parsed = read_lithium_trends(path)

    assert parsed.index.tolist() == [pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-14")]
    assert parsed.columns.tolist() == [f"Trends_{term.replace(' ', '_')}" for term in GOOGLE_TRENDS_TERMS]


def test_news_input_filters_non_lithium_headlines_and_aggregates_daily(tmp_path: Path):
    path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "published_at_utc": ["2024-01-01T01:00:00Z", "2024-01-01T12:00:00Z", "2024-01-02T01:00:00Z"],
            "title": [
                "Lithium carbonate prices rebound strongly",
                "Albemarle expands lithium production",
                "Unrelated copper market headline",
            ],
            "source": ["A", "B", "C"],
            "url": ["https://a.example/1", "https://b.example/2", "https://c.example/3"],
            "finbert_score": [0.8, 0.4, -0.2],
        }
    ).to_csv(path, index=False)

    raw, daily = read_lithium_news(path)

    assert len(raw) == 2
    assert daily.index.tolist() == [pd.Timestamp("2024-01-01")]
    assert daily.loc[pd.Timestamp("2024-01-01"), "News_Article_Count"] == 2
    assert np.isclose(daily.loc[pd.Timestamp("2024-01-01"), "News_FinBERT_Score"], 0.6)


def test_unscored_news_day_is_not_silently_treated_as_neutral(tmp_path: Path):
    path = tmp_path / "news_without_finbert.csv"
    pd.DataFrame(
        {
            "published_at_utc": ["2024-01-01T01:00:00Z"],
            "title": ["Lithium hydroxide market expands"],
            "source": ["A"],
            "url": ["https://a.example/1"],
        }
    ).to_csv(path, index=False)

    _raw, daily = read_lithium_news(path)

    assert pd.isna(daily.loc[pd.Timestamp("2024-01-01"), "News_FinBERT_Score"])
