import numpy as np
import pandas as pd

from code.models.released_paper_shadow import released_feature_frame, static_holdout_split


def test_released_static_split_matches_sklearn_upward_rounding():
    split = static_holdout_split(101, 0.20)
    assert split.train_stop == 80
    assert split.test_start == 80
    assert split.test_size == 21


def test_released_feature_frame_shifts_exogenous_values_and_uses_own_lags():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    component = pd.Series(np.arange(10, 18, dtype=float), index=index)
    exogenous = pd.DataFrame({"macro": np.arange(100, 108, dtype=float)}, index=index)

    frame = released_feature_frame(component, exogenous, horizon=1, lags=2)

    assert frame.index[0] == index[2]
    assert frame.loc[index[2], "macro"] == 101.0
    assert frame.loc[index[2], "target_1"] == 11.0
    assert frame.loc[index[2], "target_2"] == 10.0
