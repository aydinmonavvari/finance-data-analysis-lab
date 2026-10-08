"""Unit tests for statistical diagnostics on synthetic series with known properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_data_analysis_lab import stats_tests

# Fixed seed so every assertion below is deterministic across runs.
RNG_SEED = 42
N_OBS = 2000


def _white_noise() -> pd.Series:
    rng = np.random.default_rng(RNG_SEED)
    return pd.Series(rng.normal(loc=0.0, scale=0.01, size=N_OBS))


def _trending_series() -> pd.Series:
    rng = np.random.default_rng(RNG_SEED)
    trend = np.linspace(0.0, 50.0, N_OBS)
    noise = rng.normal(scale=0.25, size=N_OBS)
    return pd.Series(trend + noise)


class TestJarqueBera:
    def test_white_noise_does_not_reject_normality(self):
        result = stats_tests.jarque_bera_test(_white_noise())
        assert result["reject_normality_at_5pct"] is False
        assert result["p_value"] > 0.05
        assert result["n"] == N_OBS

    def test_heavy_tailed_series_rejects_normality(self):
        # Student-t with 3 degrees of freedom has excess kurtosis far above 0.
        rng = np.random.default_rng(RNG_SEED)
        heavy = pd.Series(rng.standard_t(df=3, size=N_OBS))
        result = stats_tests.jarque_bera_test(heavy)
        assert result["reject_normality_at_5pct"] is True
        assert result["p_value"] < 0.05
        assert result["kurtosis"] > 0

    def test_interpretation_string_present(self):
        result = stats_tests.jarque_bera_test(_white_noise())
        assert "normality" in result["interpretation"]


class TestAdf:
    def test_trending_series_is_flagged_non_stationary(self):
        result = stats_tests.adf_test(_trending_series())
        assert result["reject_unit_root_at_5pct"] is False
        assert result["p_value"] > 0.05

    def test_white_noise_is_flagged_stationary(self):
        result = stats_tests.adf_test(_white_noise())
        assert result["reject_unit_root_at_5pct"] is True
        assert result["p_value"] < 0.05

    def test_critical_values_are_reported(self):
        result = stats_tests.adf_test(_white_noise())
        assert {"1%", "5%", "10%"} <= set(result["critical_values"])


class TestLjungBox:
    def test_white_noise_shows_no_autocorrelation(self):
        result = stats_tests.ljung_box_test(_white_noise(), lags=20)
        assert result["reject_no_autocorrelation_at_5pct"] is False
        assert result["p_values"][-1] > 0.05
        assert len(result["p_values"]) == 20

    def test_autoregressive_series_is_detected(self):
        # AR(1) with rho=0.5 has strong, slowly decaying autocorrelation.
        rng = np.random.default_rng(RNG_SEED)
        eps = rng.normal(scale=0.01, size=N_OBS)
        ar = np.zeros(N_OBS)
        for i in range(1, N_OBS):
            ar[i] = 0.5 * ar[i - 1] + eps[i]
        result = stats_tests.ljung_box_test(pd.Series(ar), lags=20)
        assert result["reject_no_autocorrelation_at_5pct"] is True
        assert result["p_values"][-1] < 0.05


class TestSummaryWrapper:
    def test_summarize_contains_all_three_tests(self):
        summary = stats_tests.summarize_series_diagnostics(_white_noise())
        assert set(summary) == {"jarque_bera", "ljung_box", "adf_on_returns"}
        # NaN inputs are dropped by every wrapper; empty input must fail loudly.
        with pytest.raises(ValueError):
            stats_tests.jarque_bera_test(pd.Series([np.nan, np.nan]))
