"""Matplotlib visualization layer.

Every figure produced here has an analytical purpose in the research write-up:
price levels and co-movement, return distributions versus a fitted Gaussian,
relative cumulative performance, drawdown paths, cross-correlation and time
variation of volatility. Functions return ``matplotlib.figure.Figure`` objects
(unfigured with any GUI backend) so they can be saved by the pipeline or shown
in notebooks. Axes and titles are always labeled.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are saved to disk, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from finance_data_analysis_lab.metrics import drawdown_series

sns.set_theme(style="whitegrid", context="notebook")
DEFAULT_FIGSIZE = (11, 5.5)
DEFAULT_DPI = 150


def _finalize(fig: Figure, title: str, xlabel: str, ylabel: str, legend: bool = True) -> Figure:
    """Apply shared labeling and layout to a figure."""
    fig.axes[0].set_title(title, fontsize=13, fontweight="bold")
    fig.axes[0].set_xlabel(xlabel)
    fig.axes[0].set_ylabel(ylabel)
    if legend:
        fig.axes[0].legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    return fig


def plot_price_series(
    prices: pd.DataFrame, normalize: bool = True, figsize: tuple = DEFAULT_FIGSIZE
) -> Figure:
    """Line chart of adjusted-close price paths (optionally rebased to 100).

    Args:
        prices: Wide frame of adjusted closes (one column per symbol).
        normalize: When True, rebase each series to 100 at the first common
            observation so levels are comparable across symbols.

    Returns:
        The matplotlib Figure.
    """
    data = prices.dropna(how="any")
    if normalize:
        data = data / data.iloc[0] * 100.0
    fig, ax = plt.subplots(figsize=figsize)
    for column in data.columns:
        ax.plot(data.index, data[column], linewidth=1.3, label=column)
    suffix = " (rebased to 100)" if normalize else ""
    return _finalize(
        fig,
        f"Adjusted closing prices{suffix}",
        "Date",
        "Price (index, first common date = 100)" if normalize else "Price (USD, adjusted)",
    )


def plot_return_histogram(
    returns: pd.Series, label: str, bins: int = 80, figsize: tuple = (8, 5)
) -> Figure:
    """Histogram of daily returns with a fitted normal density overlay.

    The overlay is the Gaussian with the sample mean and standard deviation;
    departures from it (peakedness, fat tails) are what the Jarque-Bera test
    quantifies.

    Args:
        returns: Daily simple or log returns for one symbol.
        label: Symbol name used in the title/legend.
        bins: Histogram bin count.

    Returns:
        The matplotlib Figure.
    """
    clean = pd.Series(returns).dropna()
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(clean, bins=bins, stat="density", kde=False, ax=ax, color="#4c72b0", alpha=0.6)
    grid = np.linspace(clean.min(), clean.max(), 400)
    ax.plot(
        grid,
        1.0 / (clean.std(ddof=1) * np.sqrt(2 * np.pi)) * np.exp(
            -((grid - clean.mean()) ** 2) / (2 * clean.var(ddof=1))
        ),
        color="#c44e52",
        linewidth=2.0,
        label="Fitted normal",
    )
    ax.set_title(f"{label}: daily return distribution vs fitted normal", fontsize=12)
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_return_histogram_grid(
    returns: pd.DataFrame, bins: int = 80, columns: int = 2
) -> Figure:
    """Panel of per-symbol return histograms, each with a fitted normal.

    Args:
        returns: Wide frame of daily returns (one column per symbol).
        bins: Histogram bin count per panel.
        columns: Number of panel columns (rows are derived).

    Returns:
        The matplotlib Figure.
    """
    frame = returns.dropna(how="any")
    n = len(frame.columns)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for position, column in enumerate(frame.columns):
        ax = axes[position]
        clean = frame[column]
        sns.histplot(
            clean, bins=bins, stat="density", kde=False, ax=ax, color="#4c72b0", alpha=0.6
        )
        grid = np.linspace(clean.min(), clean.max(), 300)
        ax.plot(
            grid,
            1.0 / (clean.std(ddof=1) * np.sqrt(2 * np.pi)) * np.exp(
                -((grid - clean.mean()) ** 2) / (2 * clean.var(ddof=1))
            ),
            color="#c44e52",
            linewidth=1.8,
        )
        ax.set_title(column, fontsize=11)
        ax.set_xlabel("Daily return")
        ax.set_ylabel("Density")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Daily return distributions vs fitted normal densities", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_cumulative_returns(
    returns: pd.DataFrame, benchmark: str, figsize: tuple = DEFAULT_FIGSIZE
) -> Figure:
    """Growth of one unit (cumulative simple-return compounding) per symbol.

    Args:
        returns: Wide frame of daily simple returns.
        benchmark: Column name of the benchmark, drawn thicker.

    Returns:
        The matplotlib Figure.
    """
    growth = (1.0 + returns.dropna(how="any")).cumprod()
    fig, ax = plt.subplots(figsize=figsize)
    for column in growth.columns:
        is_benchmark = column == benchmark
        ax.plot(
            growth.index,
            growth[column],
            linewidth=2.2 if is_benchmark else 1.2,
            alpha=1.0 if is_benchmark else 0.85,
            label=f"{column} (benchmark)" if is_benchmark else column,
            zorder=3 if is_benchmark else 2,
        )
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    return _finalize(fig, "Cumulative growth of one unit invested", "Date", "Growth of $1")


def plot_drawdowns(
    prices: pd.DataFrame, figsize: tuple = DEFAULT_FIGSIZE, fill: bool = True
) -> Figure:
    """Drawdown curves (distance to running maximum) for each symbol.

    Args:
        prices: Wide frame of adjusted closes.
        fill: Shade the area under each drawdown curve.

    Returns:
        The matplotlib Figure.
    """
    frame = prices.dropna(how="any")
    fig, ax = plt.subplots(figsize=figsize)
    for column in frame.columns:
        dd = drawdown_series(frame[column])
        ax.plot(dd.index, dd * 100.0, linewidth=1.2, label=column)
        if fill:
            ax.fill_between(dd.index, dd * 100.0, 0.0, alpha=0.08)
    ax.set_ylim(top=2.0)
    return _finalize(fig, "Drawdown from running maximum", "Date", "Drawdown (%)", legend=True)


def plot_correlation_heatmap(
    returns: pd.DataFrame, figsize: tuple = (8.5, 7), annot: bool = True
) -> Figure:
    """Heatmap of Pearson correlations between daily return series.

    Args:
        returns: Wide frame of daily returns.
        annot: Annotate cells with correlation values.

    Returns:
        The matplotlib Figure.
    """
    corr = returns.dropna(how="any").corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap="vlag",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation (daily returns)"},
        ax=ax,
    )
    ax.set_title("Cross-correlation of daily returns", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_rolling_volatility(
    returns: pd.DataFrame, window: int = 63, figsize: tuple = DEFAULT_FIGSIZE
) -> Figure:
    """Rolling annualized volatility per symbol over a trailing window.

    Args:
        returns: Wide frame of daily simple returns.
        window: Trailing window in trading days (default 63, one quarter).

    Returns:
        The matplotlib Figure.
    """
    frame = returns.dropna(how="any")
    rolling = frame.rolling(window=window).std(ddof=1) * np.sqrt(252.0)
    fig, ax = plt.subplots(figsize=figsize)
    for column in rolling.columns:
        ax.plot(rolling.index, rolling[column] * 100.0, linewidth=1.2, label=column)
    return _finalize(
        fig,
        f"Rolling {window}-day annualized volatility",
        "Date",
        "Annualized volatility (%)",
    )


def save_figure(fig: Figure, path: str | Path, dpi: int = DEFAULT_DPI) -> None:
    """Save a figure to ``path`` (creating parent directories as needed)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
