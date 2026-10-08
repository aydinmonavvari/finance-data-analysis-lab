"""Statistical diagnostics for return and price series.

Wrappers around ``scipy.stats`` and ``statsmodels`` that return **tidy dicts**
so results can be serialized straight to JSON and rendered as tables. Each
docstring states exactly how the result should be interpreted; the returned
dict also carries a short ``interpretation`` string for report generation.

Diagnostics provided
--------------------
* **Jarque-Bera** — tests the null hypothesis that a sample comes from a
  normal distribution, using sample skewness and kurtosis. Relevant because
  many textbook risk rules (e.g. "95% of returns lie within +/- 2 sigma")
  assume normality that daily equity returns typically violate.
* **Augmented Dickey-Fuller (ADF)** — tests the null hypothesis of a unit root
  (non-stationarity). Prices are usually non-stationary (random-walk-like);
  returns are usually stationary. This asymmetry motivates modelling returns
  rather than prices.
* **Ljung-Box** — tests the null hypothesis that a series' autocorrelations up
  to lag ``k`` are jointly zero (no predictable linear structure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

SIGNIFICANCE_LEVEL = 0.05


def jarque_bera_test(returns: pd.Series) -> dict:
    """Jarque-Bera normality test on a return series.

    Interpretation: the null hypothesis is that the sample is drawn from a
    normal distribution. A p-value below 0.05 means the test *rejects
    normality* — the empirical return distribution has significant skewness
    and/or excess kurtosis. For daily equity returns rejection is the norm,
    which warns against naive Gaussian risk summaries.

    Args:
        returns: Daily returns (simple or log; the test is location/scale
            invariant so the choice does not affect the statistic).

    Returns:
        Dict with keys ``test``, ``statistic``, ``p_value``, ``n``,
        ``skew``, ``kurtosis`` (excess), ``reject_normality_at_5pct`` and a
        short ``interpretation`` string.
    """
    clean = pd.Series(returns).dropna()
    if len(clean) < 2:
        raise ValueError("Jarque-Bera test requires at least two non-missing observations.")
    statistic, p_value = stats.jarque_bera(clean)
    reject = bool(p_value < SIGNIFICANCE_LEVEL)
    return {
        "test": "Jarque-Bera",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": int(len(clean)),
        "skew": float(stats.skew(clean)),
        "kurtosis": float(stats.kurtosis(clean)),
        "reject_normality_at_5pct": reject,
        "interpretation": (
            "Rejects normality at the 5% level: fat tails and/or asymmetry present."
            if reject
            else "Does not reject normality at the 5% level: Gaussian assumption not contradicted."
        ),
    }


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test.

    Interpretation: the null hypothesis is that the series contains a unit
    root (is non-stationary). A p-value below 0.05 *rejects the unit root*,
    i.e. the series looks stationary. In this project the test is applied to
    log price levels (usually non-stationary) and to log returns (usually
    stationary), documenting why analysis operates on returns.

    Args:
        series: Level or return series; NaNs are dropped.

    Returns:
        Dict with keys ``test``, ``statistic``, ``p_value``, ``lags``,
        ``nobs``, ``critical_values``, ``reject_unit_root_at_5pct`` and a
        short ``interpretation`` string.
    """
    clean = pd.Series(series).dropna().astype(float)
    statistic, p_value, lags, nobs, critical_values, _ = adfuller(clean, autolag="AIC")
    reject = bool(p_value < SIGNIFICANCE_LEVEL)
    return {
        "test": "Augmented Dickey-Fuller",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "lags": int(lags),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "reject_unit_root_at_5pct": reject,
        "interpretation": (
            "Rejects the unit-root null at the 5% level: series looks stationary."
            if reject
            else "Fails to reject the unit-root null at the 5% level: series looks non-stationary."
        ),
    }


def ljung_box_test(returns: pd.Series, lags: int = 20) -> dict:
    """Ljung-Box test for serial autocorrelation in a return series.

    Interpretation: the null hypothesis is that autocorrelations up to lag
    ``lags`` are jointly zero (no linear predictability). A p-value below 0.05
    rejects that null, indicating some autocorrelation structure. For liquid
    large-cap daily returns any rejection is typically economically small, but
    it still cautions against treating returns as independent draws.

    Args:
        returns: Daily returns; NaNs are dropped.
        lags: Maximum lag to test (default 20, roughly one trading month).

    Returns:
        Dict with keys ``test``, ``lags`` (list), ``statistics`` (list),
        ``p_values`` (list), ``reject_no_autocorrelation_at_5pct`` (bool,
        evaluated at the largest lag) and a short ``interpretation`` string.
    """
    clean = pd.Series(returns).dropna().astype(float)
    result = acorr_ljungbox(clean, lags=list(range(1, lags + 1)), return_df=True)
    p_values = result["lb_pvalue"].to_numpy(dtype=float)
    reject = bool(p_values[-1] < SIGNIFICANCE_LEVEL)
    return {
        "test": "Ljung-Box",
        "lags": [int(lag) for lag in result.index],
        "statistics": [float(x) for x in result["lb_stat"].to_numpy()],
        "p_values": [float(x) for x in p_values],
        "reject_no_autocorrelation_at_5pct": reject,
        "interpretation": (
            "Rejects no-autocorrelation at the 5% level: some serial dependence present."
            if reject
            else "Does not reject no-autocorrelation at the 5% level."
        ),
    }


def summarize_series_diagnostics(returns: pd.Series, lags: int = 20) -> dict:
    """Convenience wrapper: all three diagnostics for one return series.

    Args:
        returns: Daily returns.
        lags: Maximum lag for the Ljung-Box test.

    Returns:
        Dict with ``jarque_bera``, ``ljung_box`` and ``adf_on_returns`` entries.
    """
    return {
        "jarque_bera": jarque_bera_test(returns),
        "ljung_box": ljung_box_test(returns, lags=lags),
        "adf_on_returns": adf_test(np.log1p(pd.Series(returns).dropna())),
    }
