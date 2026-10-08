"""Unit tests for return, risk and performance metrics (hand-computed values)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_data_analysis_lab import metrics

SQRT_252 = float(np.sqrt(252.0))


def _series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    index = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=index, dtype=float)


class TestReturns:
    def test_simple_returns_match_pct_change(self):
        prices = _series([100.0, 110.0, 99.0])
        result = metrics.simple_returns(prices)
        assert list(result.index) == list(prices.index[1:])
        assert result.iloc[0] == pytest.approx(0.10)
        assert result.iloc[1] == pytest.approx(99.0 / 110.0 - 1.0)

    def test_log_returns_are_additive_over_two_steps(self):
        prices = _series([100.0, 110.0, 121.0])
        result = metrics.log_returns(prices)
        assert result.sum() == pytest.approx(np.log(121.0 / 100.0))


class TestAnnualizedStatistics:
    def test_cagr_from_two_observations(self):
        # 100 -> 121 in one daily interval: CAGR = 1.21**252 - 1 (hand-derived).
        prices = _series([100.0, 121.0])
        expected = 1.21 ** (252.0 / 1) - 1.0
        assert metrics.annualized_return(prices) == pytest.approx(expected)

    def test_annualized_volatility_hand_computed(self):
        # Alternating +/-1%: sample variance = 4*1e-4/3, vol = sd*sqrt(252).
        returns = _series([0.01, -0.01, 0.01, -0.01])
        expected = np.sqrt(4 * 0.01**2 / 3) * SQRT_252
        assert metrics.annualized_volatility(returns) == pytest.approx(expected)

    def test_too_few_observations_raise(self):
        with pytest.raises(ValueError, match="two"):
            metrics.annualized_volatility(_series([0.01]))


class TestSharpeSortino:
    def test_sharpe_hand_computed_zero_rf(self):
        # Mean = 0.025, sample variance = 0.0005/3.
        # Sharpe = (0.025*252) / sqrt((0.0005/3)*252) = 6.3 / sqrt(0.042).
        returns = _series([0.01, 0.02, 0.03, 0.04])
        expected = 6.3 / np.sqrt(0.042)
        assert metrics.sharpe_ratio(returns, rf_annual=0.0) == pytest.approx(expected)

    def test_sharpe_subtracts_rf(self):
        returns = _series([0.01, 0.02, 0.03, 0.04])
        expected = (6.3 - 2.0) / np.sqrt(0.042)
        assert metrics.sharpe_ratio(returns, rf_annual=2.0) == pytest.approx(expected)

    def test_sharpe_zero_volatility_is_nan(self):
        returns = _series([0.01, 0.01])
        assert np.isnan(metrics.sharpe_ratio(returns))

    def test_sortino_hand_computed(self):
        # Downside deviation (rf=0): sqrt((1e-4 + 9e-4)/4) * sqrt(252).
        returns = _series([-0.01, 0.02, -0.03, 0.04])
        expected = (0.005 * 252.0) / (np.sqrt(0.001 / 4) * SQRT_252)
        assert metrics.sortino_ratio(returns, rf_annual=0.0) == pytest.approx(expected)

    def test_sortino_no_downside_is_nan(self):
        returns = _series([0.01, 0.02, 0.03, 0.04])
        assert np.isnan(metrics.sortino_ratio(returns, rf_annual=0.0))


class TestDrawdown:
    def test_drawdown_series_hand_computed(self):
        prices = _series([100.0, 120.0, 90.0, 60.0, 110.0])
        expected = pd.Series(
            [0.0, 0.0, -0.25, -0.50, 110.0 / 120.0 - 1.0], index=prices.index
        )
        result = metrics.drawdown_series(prices)
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12)

    def test_max_drawdown_is_deepest_loss(self):
        prices = _series([100.0, 120.0, 90.0, 60.0, 110.0])
        assert metrics.max_drawdown(prices) == pytest.approx(-0.50)

    def test_monotonic_rising_series_has_zero_drawdown(self):
        prices = _series([100.0, 101.0, 102.0, 103.0])
        assert metrics.max_drawdown(prices) == pytest.approx(0.0)

    def test_calmar_hand_computed(self):
        # CAGR = 0.9**(252/2) - 1 over 2 intervals; |maxDD| = 0.25.
        prices = _series([100.0, 120.0, 90.0])
        expected = (0.9 ** (252.0 / 2) - 1.0) / 0.25
        assert metrics.calmar_ratio(prices) == pytest.approx(expected)

    def test_calmar_zero_drawdown_is_nan(self):
        prices = _series([100.0, 101.0, 102.0])
        assert np.isnan(metrics.calmar_ratio(prices))


class TestBetaAlpha:
    def test_beta_two_alpha_zero(self):
        market = _series([0.01, -0.02, 0.015, 0.005, -0.01])
        asset = market * 2.0
        result = metrics.beta_alpha(asset, market)
        assert result["beta"] == pytest.approx(2.0)
        assert result["alpha_annual"] == pytest.approx(0.0, abs=1e-12)

    def test_alpha_is_annualized_residual(self):
        # asset = 2*market + 0.001 daily -> alpha_annual = 0.001*252.
        market = _series([0.01, -0.02, 0.015, 0.005, -0.01])
        asset = market * 2.0 + 0.001
        result = metrics.beta_alpha(asset, market)
        assert result["beta"] == pytest.approx(2.0)
        assert result["alpha_annual"] == pytest.approx(0.001 * 252.0)

    def test_misaligned_dates_are_inner_joined(self):
        market = _series([0.01, -0.02, 0.015, 0.005, -0.01])
        asset = (market * 2.0).iloc[1:].copy()  # drop first date -> 4 aligned rows
        result = metrics.beta_alpha(asset, market)
        assert result["beta"] == pytest.approx(2.0)

    def test_zero_variance_benchmark_raises(self):
        market = _series([0.0, 0.0, 0.0])
        asset = _series([0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="zero variance"):
            metrics.beta_alpha(asset, market)


class TestRollingStatistics:
    def test_rolling_volatility_window_two(self):
        returns = _series([0.01, -0.01, 0.01])
        result = metrics.rolling_volatility(returns, window=2)
        assert np.isnan(result.iloc[0])
        expected_last = np.std([-0.01, 0.01], ddof=1) * SQRT_252  # last window of 2
        assert result.iloc[-1] == pytest.approx(expected_last)

    def test_rolling_sharpe_matches_full_sample_at_end(self):
        returns = _series([0.01, 0.02, 0.03, 0.04])
        result = metrics.rolling_sharpe(returns, window=4)
        expected = metrics.sharpe_ratio(returns)
        assert result.iloc[-1] == pytest.approx(expected)
