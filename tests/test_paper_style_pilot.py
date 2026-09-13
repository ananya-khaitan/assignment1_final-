import numpy as np
import pandas as pd

from code.models.paper_style import CausalPaperStyleEngine, next_origin_feature_row


def test_paper_style_next_origin_row_uses_only_values_available_at_origin():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    target = pd.Series([10, 11, 12, 13, 14, 15], index=index, dtype=float)
    exog = pd.DataFrame({"macro": [100, 101, 102, 103, 104, 105]}, index=index, dtype=float)

    row = next_origin_feature_row(target, exog, ["target_lag_1", "target_lag_3", "macro"], lags=5)

    np.testing.assert_allclose(row.loc[0, ["target_lag_1", "target_lag_3", "macro"]], [15.0, 13.0, 105.0])


def test_paper_style_next_origin_row_is_invariant_to_an_unseen_future_exogenous_value():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    target = pd.Series(np.arange(10, 16, dtype=float), index=index)
    available = pd.DataFrame({"macro": np.arange(100, 106, dtype=float)}, index=index)
    future = pd.concat([available, pd.DataFrame({"macro": [999999.0]}, index=[index[-1] + pd.Timedelta(days=1)])])

    first = next_origin_feature_row(target, available, ["target_lag_1", "macro"], lags=5)
    # The helper must itself discard any row beyond the supplied target history;
    # callers do not need to trim an unavailable future exogenous observation.
    second = next_origin_feature_row(target, future, ["target_lag_1", "macro"], lags=5)

    pd.testing.assert_frame_equal(first, second)


def test_paper_style_lag_only_engine_supports_all_lstm_routing():
    engine = CausalPaperStyleEngine(
        vmd_k=8,
        low_modes=0,
        refit_interval=20,
        vmd_alpha=2000,
        vmd_tau=0,
        vmd_dc=0,
        vmd_init=1,
        vmd_tol=1e-7,
        lstm_sequence_style="single_step",
    )
    assert engine.low_modes == 0
    assert engine.lstm_sequence_style == "single_step"
