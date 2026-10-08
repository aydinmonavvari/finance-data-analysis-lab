"""End-to-end analysis pipeline.

Orchestrates the whole study in one deterministic pass:

1. load (or download-and-cache) adjusted closes for the universe + benchmark;
2. validate prices and apply the documented missing-value policy;
3. compute per-ticker risk/return metrics and benchmark-relative statistics;
4. run distributional and time-series diagnostics (Jarque-Bera, ADF,
   Ljung-Box);
5. render all figures to ``figures/``;
6. write machine-readable results to ``reports/metrics.json`` and a
   human-readable summary to ``reports/summary.md``.

The pipeline is purely descriptive: it transforms observed prices into summary
statistics and charts. It contains no prediction, no trading logic and no
randomness (a seed is set anyway for reproducibility of any future stochastic
extension).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finance_data_analysis_lab import metrics, plots, stats_tests
from finance_data_analysis_lab.config import PROJECT_ROOT, Config, load_config
from finance_data_analysis_lab.data import load_price_frame, validate_prices

logger = logging.getLogger(__name__)

JSON_SERIALIZE_ROUND = 6


def _json_safe(value):
    """Convert numpy/pandas scalars and paths to plain Python types for JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return round(value, JSON_SERIALIZE_ROUND)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def compute_ticker_metrics(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    ticker: str,
    benchmark: str,
    rf_annual: float,
    periods_per_year: int,
) -> dict:
    """Compute the full descriptive metric block for one ticker.

    Args:
        prices: Validated wide price frame.
        returns: Wide frame of daily simple returns derived from ``prices``.
        ticker: Column of the asset to analyse.
        benchmark: Benchmark column (e.g. ``"SPY"``).
        rf_annual: Annual risk-free rate assumption.
        periods_per_year: Annualization factor (252).

    Returns:
        Dict of metrics including annualized return/volatility, Sharpe,
        Sortino, Calmar, maximum drawdown, benchmark beta/alpha and the three
        statistical diagnostics.
    """
    asset_prices = prices[ticker]
    asset_returns = returns[ticker]
    log_rets = metrics.log_returns(asset_prices)

    benchmark_relative = metrics.beta_alpha(asset_returns, returns[benchmark], periods_per_year)
    jarque_bera = stats_tests.jarque_bera_test(log_rets)
    ljung_box = stats_tests.ljung_box_test(asset_returns, lags=20)
    adf_prices = stats_tests.adf_test(np.log(asset_prices))
    adf_returns = stats_tests.adf_test(log_rets)

    return {
        "ticker": ticker,
        "first_date": str(asset_prices.index.min().date()),
        "last_date": str(asset_prices.index.max().date()),
        "n_observations": int(len(asset_prices)),
        "total_return": float(asset_prices.iloc[-1] / asset_prices.iloc[0] - 1.0),
        "annualized_return": metrics.annualized_return(asset_prices, periods_per_year),
        "annualized_volatility": metrics.annualized_volatility(asset_returns, periods_per_year),
        "sharpe_ratio": metrics.sharpe_ratio(asset_returns, rf_annual, periods_per_year),
        "sortino_ratio": metrics.sortino_ratio(asset_returns, rf_annual, periods_per_year),
        "calmar_ratio": metrics.calmar_ratio(asset_prices, periods_per_year),
        "max_drawdown": metrics.max_drawdown(asset_prices),
        "beta_vs_benchmark": benchmark_relative["beta"],
        "alpha_annual_vs_benchmark": benchmark_relative["alpha_annual"],
        "jarque_bera": {
            "statistic": jarque_bera["statistic"],
            "p_value": jarque_bera["p_value"],
            "skew": jarque_bera["skew"],
            "kurtosis": jarque_bera["kurtosis"],
            "reject_normality_at_5pct": jarque_bera["reject_normality_at_5pct"],
        },
        "adf_log_price": {
            "statistic": adf_prices["statistic"],
            "p_value": adf_prices["p_value"],
            "reject_unit_root_at_5pct": adf_prices["reject_unit_root_at_5pct"],
        },
        "adf_log_return": {
            "statistic": adf_returns["statistic"],
            "p_value": adf_returns["p_value"],
            "reject_unit_root_at_5pct": adf_returns["reject_unit_root_at_5pct"],
        },
        "ljung_box_lag20": {
            "statistic": ljung_box["statistics"][-1],
            "p_value": ljung_box["p_values"][-1],
            "reject_no_autocorrelation_at_5pct": ljung_box[
                "reject_no_autocorrelation_at_5pct"
            ],
        },
    }


def _render_summary(
    config: Config,
    validation: dict,
    metrics_rows: list[dict],
    generated_at: str,
) -> str:
    """Render the human-readable ``reports/summary.md`` from actual results."""
    bench = config.benchmark
    df = pd.DataFrame(metrics_rows).set_index("ticker")
    pct_cols = ["total_return", "annualized_return", "annualized_volatility", "max_drawdown"]
    display = df.copy()
    for column in pct_cols:
        display[column] = (display[column] * 100.0).map("{:.2f}%".format)
    for column in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "beta_vs_benchmark"]:
        display[column] = display[column].map("{:.2f}".format)
    display["alpha_annual_vs_benchmark"] = (
        df["alpha_annual_vs_benchmark"] * 100.0
    ).map("{:.2f}%".format)

    order = df["sharpe_ratio"].sort_values(ascending=False).index.tolist()
    display = display.loc[order]

    table_lines = [
        "| Ticker | Ann. return | Ann. vol | Sharpe | Sortino | "
        "Calmar | Max DD | Beta | Alpha (ann.) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ticker, row in display.iterrows():
        table_lines.append(
            f"| {ticker} | {row['annualized_return']} | {row['annualized_volatility']} | "
            f"{row['sharpe_ratio']} | {row['sortino_ratio']} | {row['calmar_ratio']} | "
            f"{row['max_drawdown']} | {row['beta_vs_benchmark']} | "
            f"{row['alpha_annual_vs_benchmark']} |"
        )

    jb_rejections = [
        row["ticker"] for row in metrics_rows if row["jarque_bera"]["reject_normality_at_5pct"]
    ]
    deepest = min(metrics_rows, key=lambda row: row["max_drawdown"])
    vol_max = max(metrics_rows, key=lambda row: row["annualized_volatility"])
    vol_min = min(metrics_rows, key=lambda row: row["annualized_volatility"])
    sharpe_max = max(metrics_rows, key=lambda row: row["sharpe_ratio"])
    sharpe_min = min(metrics_rows, key=lambda row: row["sharpe_ratio"])
    bench_row = next(row for row in metrics_rows if row["ticker"] == bench)

    lines = [
        "# Descriptive analysis summary (actual run output)",
        "",
        f"*Generated by `pipeline.run_pipeline` on {generated_at}. All figures derive from the "
        f"cached Yahoo Finance adjusted closes listed below. This is a descriptive, educational "
        f"analysis — not investment advice.*",
        "",
        "## Sample",
        "",
        f"- Universe: {', '.join(config.tickers)}; benchmark: {bench}",
        f"- Validated price window: {validation['start']} to {validation['end']} "
        f"({validation['rows_after_cleaning']} trading days, "
        f"{validation['rows_dropped']} rows dropped by the missing-value policy)",
        f"- Risk-free assumption: rf_annual = {config.rf_annual:.2f} (configurable in "
        f"`configs/default.yaml`)",
        f"- Annualization factor: {config.periods_per_year} trading days",
        "",
        "## Per-ticker metrics (sorted by Sharpe ratio)",
        "",
        *table_lines,
        "",
        "## Key observations (computed from this run)",
        "",
        f"- Highest Sharpe (rf={config.rf_annual:.2f}): **{sharpe_max['ticker']}** "
        f"({sharpe_max['sharpe_ratio']:.2f}); lowest: **{sharpe_min['ticker']}** "
        f"({sharpe_min['sharpe_ratio']:.2f}).",
        f"- Most volatile: **{vol_max['ticker']}** ({vol_max['annualized_volatility']:.2%} "
        f"annualized); least volatile: **{vol_min['ticker']}** "
        f"({vol_min['annualized_volatility']:.2%}).",
        f"- Deepest maximum drawdown: **{deepest['ticker']}** ({deepest['max_drawdown']:.2%}); "
        f"benchmark {bench}: {bench_row['max_drawdown']:.2%}.",
        f"- Jarque-Bera rejects normality of daily log returns at 5% for "
        f"{len(jb_rejections)}/{len(metrics_rows)} series"
        + (f": {', '.join(jb_rejections)}" if jb_rejections else "")
        + ".",
        f"- Benchmark {bench}: annualized return {bench_row['annualized_return']:.2%}, "
        f"volatility {bench_row['annualized_volatility']:.2%}, Sharpe "
        f"{bench_row['sharpe_ratio']:.2f}.",
        "",
        "## Reading guide",
        "",
        "Sharpe/Sortino/Calmar are *descriptive summaries of the past sample*; they are not "
        "forecasts and their ranks can change materially with the sample window or the "
        "risk-free assumption. See `docs/research_report.md` for methodology, limitations and "
        "caveats.",
        "",
    ]
    return "\n".join(lines)


def run_pipeline(config: Config | None = None, refresh_data: bool = False) -> dict:
    """Run the full descriptive analysis and write reports + figures.

    Args:
        config: Optional pre-built :class:`Config`; defaults to
            :func:`finance_data_analysis_lab.config.load_config`.
        refresh_data: Force re-download of cached prices when True.

    Returns:
        Dict with keys ``validation``, ``metrics`` (list of per-ticker dicts)
        and ``artifacts`` (paths of written files).
    """
    np.random.seed(42)  # deterministic by construction; seed kept for future stochastic steps
    config = config or load_config()
    config.ensure_output_dirs()

    raw = load_price_frame(
        config.all_symbols, config.start, config.end, config.raw_dir, refresh=refresh_data
    )
    prices, validation = validate_prices(raw)
    logger.info(
        "Validated price frame: %d rows x %d symbols (%s to %s)",
        len(prices),
        prices.shape[1],
        validation["start"],
        validation["end"],
    )

    returns = metrics.simple_returns(prices)

    metrics_rows = [
        compute_ticker_metrics(
            prices, returns, ticker, config.benchmark, config.rf_annual, config.periods_per_year
        )
        for ticker in config.tickers + [config.benchmark]
    ]

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    figures = {
        "price_series_indexed.png": plots.plot_price_series(prices, normalize=True),
        "return_histograms.png": plots.plot_return_histogram_grid(
            metrics.log_returns(prices)
        ),
        "cumulative_returns.png": plots.plot_cumulative_returns(returns, config.benchmark),
        "drawdowns.png": plots.plot_drawdowns(prices),
        "correlation_heatmap.png": plots.plot_correlation_heatmap(returns),
        "rolling_volatility.png": plots.plot_rolling_volatility(returns, window=63),
    }
    artifacts = []
    for filename, figure in figures.items():
        path = config.figures_dir / filename
        plots.save_figure(figure, str(path))
        artifacts.append(str(path))

    # Report paths relative to the project root so generated artifacts stay
    # portable across machines (no absolute sandbox/home paths in the repo).
    config_payload = {
        key: (str(value).replace(str(PROJECT_ROOT) + "/", "") if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }

    payload = _json_safe(
        {
            "generated_at": generated_at,
            "project": "finance-data-analysis-lab",
            "description": "Descriptive financial analytics on real daily market data "
            "(educational research project; not investment advice).",
            "config": config_payload,
            "validation": validation,
            "metrics": metrics_rows,
        }
    )
    metrics_path = config.reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    artifacts.append(str(metrics_path))

    summary = _render_summary(config, validation, metrics_rows, generated_at)
    summary_path = config.reports_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    artifacts.append(str(summary_path))

    logger.info("Pipeline complete: %d figures, %s, %s", len(figures), metrics_path, summary_path)
    return {"validation": validation, "metrics": metrics_rows, "artifacts": artifacts}
