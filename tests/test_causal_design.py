import numpy as np
import pandas as pd
import unittest
import sys
import types
from io import BytesIO
from zipfile import ZipFile

from code.features.selection import build_feature_frame
from code.decomposition.causal import CausalComponentHistory, approximate_entropy
from code.trading.signals import evaluate_trading
from code.uncertainty.intervals import conformal_width_from_residuals, interval_metrics
from code.evaluation.stat_tests import model_confidence_set
from code.data_sources.download_public_inputs import _parse_ken_french_zip
from code.utils.causal_pipeline import calibration_and_test_splits, expanding_walk_forward, make_exogenous_frame


def test_one_step_split_scores_the_immediate_next_observation():
    splits = expanding_walk_forward(n_obs=20, min_train=10)
    assert splits[0].train_end == 10
    assert splits[0].valid_end == 10
    assert splits[0].test_end == 11
    assert len(splits) == 10


def test_validation_cannot_create_a_gap_before_a_one_step_target():
    try:
        expanding_walk_forward(n_obs=30, min_train=10, valid_size=5)
    except ValueError as error:
        assert "validation" in str(error).lower()
    else:
        raise AssertionError("A validation gap must be rejected")


def test_calibration_is_before_the_final_test_period():
    calibration, final_test = calibration_and_test_splits(100, min_train=20, calibration_size=15, test_size=20)
    assert calibration[-1].test_end == final_test[0].train_end
    assert final_test[0].train_end == 80
    assert final_test[0].test_end == 81


def test_feature_frame_is_unchanged_by_future_exogenous_mutation():
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    target = pd.Series(np.arange(20, dtype=float), index=index)
    exog = pd.DataFrame({"macro": np.arange(20, dtype=float)}, index=index)
    original = build_feature_frame(target, exog, lags=2)
    mutated = exog.copy()
    mutated.loc[index[15]:, "macro"] = 999_999.0
    changed = build_feature_frame(target, mutated, lags=2)
    pd.testing.assert_series_equal(original.loc[:index[15], "macro"], changed.loc[:index[15], "macro"])


def test_compatibility_exogenous_helper_lags_every_external_predictor():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    target = pd.Series(np.arange(8, dtype=float), index=index)
    exog = pd.DataFrame({"macro": np.arange(100, 108, dtype=float)}, index=index)
    features, aligned_target = make_exogenous_frame(target, exog, lags=2)
    assert aligned_target.index[0] == index[2]
    assert features.loc[index[2], "macro"] == exog.loc[index[1], "macro"]


def test_conformal_width_does_not_use_unprovided_future_errors():
    historical = np.array([1.0, 2.0, 3.0, 4.0])
    width = conformal_width_from_residuals(historical, alpha=0.10)
    assert width == 4.0
    assert conformal_width_from_residuals(historical.copy(), alpha=0.10) == width


def test_interval_metrics_include_standard_quality_measures():
    metrics = interval_metrics([10, 11], [10, 10], [1, 1], alpha=0.10)
    assert set(metrics) == {"Coverage (%)", "Avg Width", "Sharpness", "Calibration Error", "Winkler"}
    assert metrics["Coverage (%)"] == 100.0


def test_trade_signal_is_invariant_to_unseen_realized_targets():
    origin = [100.0, 100.0, 100.0]
    forecast = [102.0, 98.0, 101.0]
    width = [1.0, 1.0, 1.0]
    first = evaluate_trading(origin, [101.0, 99.0, 100.5], forecast, width, threshold=0.25)
    second = evaluate_trading(origin, [150.0, 50.0, 120.0], forecast, width, threshold=0.25)
    np.testing.assert_array_equal(first.signal, second.signal)
    assert not np.allclose(first.realized_returns, second.realized_returns)


def test_trading_uses_origin_to_target_return_not_forecast_path_changes():
    result = evaluate_trading([100.0, 100.0], [101.0, 99.0], [102.0, 98.0], [1.0, 1.0], threshold=0.25)
    np.testing.assert_allclose(result.predicted_returns, [0.02, -0.02])
    np.testing.assert_allclose(result.realized_returns, [0.01, -0.01])


def test_between_refit_component_update_uses_only_observed_innovations():
    # The two components reconstruct the fitted history exactly.  The first
    # component is the smoothest and therefore carries newly observed level
    # innovations until the next scheduled full decomposition.
    modes = np.asarray([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0]])
    history = CausalComponentHistory(modes, np.asarray([10.0, 10.0, 10.0]))
    updated = history.advance(np.asarray([10.0, 10.0, 10.0, 12.0]))
    np.testing.assert_allclose(updated.sum(axis=0), [10.0, 10.0, 10.0, 12.0])
    # Values that are not supplied cannot affect the current component state.
    np.testing.assert_allclose(history.modes[:, -1], updated[:, -1])


def test_between_refit_component_update_rejects_rewritten_history():
    state = CausalComponentHistory(np.asarray([[1.0, 1.0], [9.0, 9.0]]), np.asarray([10.0, 10.0]))
    try:
        state.advance(np.asarray([999.0, 10.0, 11.0]))
    except ValueError as error:
        assert "already observed" in str(error)
    else:
        raise AssertionError("A decomposed state cannot be updated from rewritten history")


def test_approximate_entropy_matches_the_dense_chebyshev_definition():
    signal = np.asarray([0.0, 0.3, 0.9, 0.2, 0.7, 0.1, 0.8, 0.4])

    def dense_phi(order: int, tolerance: float) -> float:
        patterns = np.asarray([signal[index : index + order] for index in range(len(signal) - order + 1)])
        distances = np.max(np.abs(patterns[:, None, :] - patterns[None, :, :]), axis=2)
        probabilities = np.mean(distances <= tolerance, axis=1)
        return float(np.mean(np.log(np.maximum(probabilities, 1e-12))))

    tolerance = 0.2 * np.std(signal)
    expected = dense_phi(2, tolerance) - dense_phi(3, tolerance)
    np.testing.assert_allclose(approximate_entropy(signal), expected)


def test_mcs_maps_exact_duplicate_loss_paths_back_to_all_labels(monkeypatch):
    """The bootstrap should receive unique paths but report both duplicates."""
    captured = {}

    class FakeMCS:
        def __init__(self, losses, **_kwargs):
            captured["losses"] = np.asarray(losses)
            self.included = [0]
            self.pvalues = pd.DataFrame({"Pvalue": [0.8, 0.2]}, index=[0, 1])

        def compute(self):
            return None

    fake_bootstrap = types.ModuleType("arch.bootstrap")
    fake_bootstrap.MCS = FakeMCS
    fake_arch = types.ModuleType("arch")
    fake_arch.bootstrap = fake_bootstrap
    monkeypatch.setitem(sys.modules, "arch", fake_arch)
    monkeypatch.setitem(sys.modules, "arch.bootstrap", fake_bootstrap)

    losses = np.asarray([[1.0, 1.0, 2.0], [2.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    included, pvalues = model_confidence_set(losses, reps=5)
    assert captured["losses"].shape == (3, 2)
    assert included == [0, 1]
    assert pvalues == {"0": 0.8, "1": 0.8, "2": 0.2}


def test_ken_french_monthly_tokens_are_not_parsed_as_january_days():
    text = """Monthly factors\n, Mkt-RF, SMB, HML, RMW, CMA, RF\n202401,1,2,3,4,5,0\n202402,1,2,3,4,5,0\n\n"""
    archive = BytesIO()
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("Emerging_5_Factors.csv", text)
    parsed = _parse_ken_french_zip(archive.getvalue())
    assert parsed.attrs["frequency"] == "monthly"
    assert parsed["Date"].dt.day.tolist() == [1, 1]
    assert parsed["Date"].dt.month.tolist() == [1, 2]


class TestCausalDesign(unittest.TestCase):
    """Expose the same checks to the standard-library runner and pytest."""

    def test_one_step_split(self):
        test_one_step_split_scores_the_immediate_next_observation()

    def test_validation_gap(self):
        test_validation_cannot_create_a_gap_before_a_one_step_target()

    def test_calibration_partition(self):
        test_calibration_is_before_the_final_test_period()

    def test_exogenous_future_mutation(self):
        test_feature_frame_is_unchanged_by_future_exogenous_mutation()

    def test_compatibility_exog_lag(self):
        test_compatibility_exogenous_helper_lags_every_external_predictor()

    def test_conformal_history(self):
        test_conformal_width_does_not_use_unprovided_future_errors()

    def test_interval_quality_metrics(self):
        test_interval_metrics_include_standard_quality_measures()

    def test_signal_ignores_actual(self):
        test_trade_signal_is_invariant_to_unseen_realized_targets()

    def test_origin_return_alignment(self):
        test_trading_uses_origin_to_target_return_not_forecast_path_changes()

    def test_between_refit_component_update(self):
        test_between_refit_component_update_uses_only_observed_innovations()

    def test_component_history_rewrite_rejected(self):
        test_between_refit_component_update_rejects_rewritten_history()
