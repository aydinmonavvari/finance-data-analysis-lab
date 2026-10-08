"""Return, risk and performance metrics.

Conventions (documented once, used everywhere)
----------------------------------------------
* **Returns**: both simple and logarithmic returns are provided. Simple returns
  are additive across portfolios; log returns are additive across time. Simple
  returns are used for beta/alpha and drawdown-free statistics; log returns are
  used for distributional diagnostics.
* **Annualization factor**: 252 trading days per year
  (:data:`ANNUALIZATION_FACTOR`), the standard convention for US equity
  markets. Daily statistics are scaled by 252 (mean) and sqrt(252)
  (volatility). This is an approximation, not a market truth.
* **Annualized return**: geometric (CAGR-style), computed from the price path,
  i.e. ``(P_end / P_start) ** (252 / n_periods) - 1``.
* **Sharpe ratio**: annualized arithmetic mean of daily simple returns minus
  the annual risk-free assumption, divided by annualized volatility. With the
  default ``rf_annual = 0.0`` this is the classic "reward to variability"
  ratio of Sharpe (1966); setting a positive rate makes it an excess-return
  ratio. The risk-free assumption is configurable in ``configs/default.yaml``.
* **Sortino ratio**: like Sharpe, but the denominator is the downside
  deviation computed against the daily target return ``rf_annual / 252`` using
  *all* observations in the mean-square (the full-population convention).
* **Maximum drawdown**: most negative peak-to-trough loss of the price path.
* **Beta / alpha**: OLS slope of asset daily returns on benchmark daily
  returns over aligned dates; alpha is the annualized intercept residual
  ``(mean_asset - beta * mean_benchmark) * 252`` (an arithmetic, non-compounded
  approximation to Jensen's alpha).

All functions are pure and side-effect free so they can be unit-tested against
hand-computed values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# US equity markets trade ~252 days per year on average; used for all
# annualization in this project. Change here (or pass periods_per_year) to
# switch conventions.
ANNUALIZATION_FACTOR = 252.0


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute daily simple (arithmetic) returns ``P_t / P_{t-1} - 1``.

    Args:
        prices: Price series or frame sorted by date.

    Returns:
        Same shape as input, first observation dropped.
    """
    return prices.pct_change().dropna(how="any")


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute daily logarithmic returns ``ln(P_t / P_{t-1})``.

    Log returns are additive across time and are the natural input for
    distributional diagnostics (normality tests) and volatility scaling.

    Args:
        prices: Price series or frame sorted by date.

    Returns:
        Same shape as input, first observation dropped.
    """
    return np.log(prices).diff().dropna(how="any")


def annualized_return(prices: pd.Series, periods_per_year: float = ANNUALIZATION_FACTOR) -> float:
    """Geometric annualized return (CAGR) implied by a price path.

    ``CAGR = (P_end / P_start) ** (periods_per_year / n_periods) - 1`` where
    ``n_periods`` is the number of daily return intervals in the sample.

    Args:
        prices: Adjusted-close series (no missing values).
        periods_per_year: Annualization factor (default 252).

    Returns:
        Annualized return as a decimal fraction (e.g. 0.12 for 12%).
    """
    clean = prices.dropna()
    n_periods = len(clean) - 1
    if n_periods <= 0:
        raise ValueError("Need at least two price observations to annualize.")
    growth = float(clean.iloc[-1] / clean.iloc[0])
    return growth ** (periods_per_year / n_periods) - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: float = ANNUALIZATION_FACTOR
) -> float:
    """Annualized standard deviation of daily simple returns.

    Uses the sample standard deviation (``ddof=1``) scaled by
    ``sqrt(periods_per_year)``.

    Args:
        returns: Daily simple returns.
        periods_per_year: Annualization factor (default 252).

    Returns:
        Annualized volatility as a decimal fraction.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations to estimate volatility.")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: float = ANNUALIZATION_FACTOR,
) -> float:
    """Annualized Sharpe ratio: excess mean return over volatility.

    ``(mean(returns) * periods_per_year - rf_annual) / (std(returns, ddof=1)
    * sqrt(periods_per_year))``. The risk-free assumption ``rf_annual`` is a
    configuration choice (0.0 by default), not a measured input.

    Args:
        returns: Daily simple returns.
        rf_annual: Annual risk-free rate assumption.
        periods_per_year: Annualization factor (default 252).

    Returns:
        Sharpe ratio in annualized units. ``np.nan`` if volatility is zero.
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations for the Sharpe ratio.")
    volatility = annualized_volatility(returns, periods_per_year)
    if volatility == 0.0:
        return np.nan
    mean_annual = float(returns.mean() * periods_per_year)
    return (mean_annual - rf_annual) / volatility


def sortino_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: float = ANNUALIZATION_FACTOR,
) -> float:
    """Annualized Sortino ratio: excess mean return over downside deviation.

    Downside deviation uses the full-population convention:
    ``sqrt(mean(min(r - target, 0) ** 2)) * sqrt(periods_per_year)`` with the
    daily target ``rf_annual / periods_per_year``.

    Args:
        returns: Daily simple returns.
        rf_annual: Annual risk-free/target rate assumption.
        periods_per_year: Annualization factor (default 252).

    Returns:
        Sortino ratio in annualized units; ``np.nan`` when the downside
        deviation is zero (no observations at or below target).
    """
    if len(returns) < 2:
        raise ValueError("Need at least two return observations for the Sortino ratio.")
    daily_target = rf_annual / periods_per_year
    downside = np.minimum(returns.to_numpy(dtype=float) - daily_target, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year))
    if downside_deviation == 0.0:
        return np.nan
    mean_annual = float(returns.mean() * periods_per_year)
    return (mean_annual - rf_annual) / downside_deviation


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Drawdown path of a price series relative to its running maximum.

    ``DD_t = P_t / max(P_0..P_t) - 1``, so drawdowns are <= 0 and measured in
    price units (adjusted for splits/dividends by the input series).

    Args:
        prices: Adjusted-close series.

    Returns:
        Series of drawdowns (non-positive floats) with the same index.
    """
    running_max = prices.cummax()
    return prices / running_max - 1.0


def max_drawdown(prices: pd.Series) -> float:
    """Most negative peak-to-trough loss of the price path.

    Args:
        prices: Adjusted-close series.

    Returns:
        Maximum drawdown as a non-positive decimal fraction (e.g. -0.35 for a
        35% peak-to-trough loss). 0.0 for a monotonically rising series.
    """
    return float(drawdown_series(prices).min())


def calmar_ratio(
    prices: pd.Series, periods_per_year: float = ANNUALIZATION_FACTOR
) -> float:
    """Calmar ratio: geometric annualized return divided by |maximum drawdown|.

    Args:
        prices: Adjusted-close series.
        periods_per_year: Annualization factor (default 252).

    Returns:
        Calmar ratio; ``np.nan`` when the maximum drawdown is zero.
    """
    mdd = max_drawdown(prices)
    if mdd == 0.0:
        return np.nan
    return annualized_return(prices, periods_per_year) / abs(mdd)


def beta_alpha(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float = ANNUALIZATION_FACTOR,
) -> dict[str, float]:
    """Benchmark beta and annualized alpha from aligned daily returns.

    Beta is the OLS slope ``cov(r_asset, r_bench) / var(r_bench)`` computed on
    the intersection of trading dates. Alpha is annualized arithmetically as
    ``(mean_asset - beta * mean_benchmark) * periods_per_year`` — an
    approximation to Jensen's alpha that ignores compounding.

    Args:
        asset_returns: Daily simple returns of the asset.
        benchmark_returns: Daily simple returns of the benchmark.
        periods_per_year: Annualization factor (default 252).

    Returns:
        Dict with keys ``beta`` and ``alpha_annual``.

    Raises:
        ValueError: If fewer than two aligned observations exist.
    """
    aligned = pd.concat(
        {"asset": asset_returns, "benchmark": benchmark_returns}, axis=1
    ).dropna(how="any")
    if len(aligned) < 2:
        raise ValueError("Need at least two aligned return observations for beta/alpha.")
    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    if benchmark_variance == 0.0:
        raise ValueError("Benchmark returns have zero variance; beta is undefined.")
    covariance = float(aligned["asset"].cov(aligned["benchmark"]))
    beta = covariance / benchmark_variance
    alpha_annual = (
        float(aligned["asset"].mean()) - beta * float(aligned["benchmark"].mean())
    ) * periods_per_year
    return {"beta": beta, "alpha_annual": alpha_annual}


def rolling_volatility(
    returns: pd.Series, window: int = 63, periods_per_year: float = ANNUALIZATION_FACTOR
) -> pd.Series:
    """Rolling annualized volatility over a trailing window.

    Args:
        returns: Daily simple returns.
        window: Trailing window in trading days (default 63, i.e. ~one quarter).
        periods_per_year: Annualization factor (default 252).

    Returns:
        Series of rolling annualized volatilities (first ``window - 1`` values
        are NaN).
    """
    if window < 2:
        raise ValueError("Rolling volatility window must be at least 2 observations.")
    return returns.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    rf_annual: float = 0.0,
    periods_per_year: float = ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Rolling annualized Sharpe ratio over a trailing window.

    Args:
        returns: Daily simple returns.
        window: Trailing window in trading days (default 252, i.e. one year).
        rf_annual: Annual risk-free rate assumption.
        periods_per_year: Annualization factor (default 252).

    Returns:
        Series of rolling Sharpe ratios (first ``window - 1`` values are NaN).
    """
    if window < 2:
        raise ValueError("Rolling Sharpe window must be at least 2 observations.")
    rolling_mean = returns.rolling(window=window).mean() * periods_per_year
    rolling_std = returns.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (rolling_mean - rf_annual) / rolling_std
    return result.replace([np.inf, -np.inf], np.nan)
