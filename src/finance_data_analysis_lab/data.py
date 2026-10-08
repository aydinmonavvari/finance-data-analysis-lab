"""Data ingestion, caching and validation.

Pipeline design
---------------
1. **Cache first, network on miss.** :func:`load_or_fetch` reads a per-ticker
   CSV from ``data/raw/`` when present and only calls Yahoo Finance through
   ``yfinance`` when the cache is missing (or ``refresh=True`` is requested).
   A cache hit is additionally **coverage-checked**: the cached series must
   span the requested ``start..end`` window (within a small calendar
   tolerance for weekends/holidays), otherwise the loader warns and
   re-downloads instead of silently shortening the study to whatever the
   truncated cache happens to contain. This keeps the pipeline reproducible,
   cheap and polite to the data provider.
2. **Validation.** :func:`validate_prices` enforces an explicit data contract:
   a strictly increasing, unique business-day ``DatetimeIndex``, strictly
   positive prices, and a documented missing-value policy (forward-fill gaps of
   at most 5 consecutive trading days, then drop rows that are still missing,
   e.g. leading rows before a listing date). The function returns both the
   cleaned frame and a machine-readable validation report so that every
   downstream number can be traced to an auditable input state.

Data source note: Yahoo Finance (via ``yfinance``) provides *delayed,
unverified* daily prices. Adjusted close is used throughout so that splits and
dividends do not create artificial jumps in the return series.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Maximum number of consecutive trading days that may be forward-filled when
# repairing isolated missing values. Longer gaps (e.g. listing dates, halts)
# are dropped instead of fabricated.
MAX_FFILL_LIMIT = 5

# Share of missing observations above which a warning is recorded.
MISSING_WARN_FRACTION = 0.01

# Calendar-day slack allowed at either edge of the requested window before a
# cached series is judged not to cover it. Weekends and market holidays around
# the window boundaries (e.g. a Jan 1 start) are legitimate; a truncated cache
# is short by weeks or years and fails the check.
CACHE_COVERAGE_TOLERANCE_DAYS = 10


def cache_path(raw_dir: str | Path, ticker: str) -> Path:
    """Return the CSV cache path for a ticker inside ``raw_dir``."""
    safe = ticker.upper().replace("^", "_").replace("/", "_")
    return Path(raw_dir) / f"{safe}.csv"


def fetch_adjusted_close(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> pd.Series:
    """Download daily adjusted-close prices for one ticker from Yahoo Finance.

    Uses ``yfinance`` with ``auto_adjust=True`` so the returned "Close" column
    is the split- and dividend-adjusted close. Retries transient failures with
    exponential backoff (Yahoo occasionally rate-limits requests).

    Args:
        ticker: Yahoo ticker symbol, e.g. ``"AAPL"`` or ``"SPY"``.
        start: Start date ``YYYY-MM-DD`` (inclusive).
        end: End date ``YYYY-MM-DD`` (exclusive on Yahoo's side).
        max_retries: Total download attempts before giving up.
        backoff_seconds: Base sleep between attempts; doubles each retry.

    Returns:
        A ``pd.Series`` of adjusted closes indexed by normalized calendar dates.

    Raises:
        RuntimeError: If all download attempts fail or return no data.
    """
    import yfinance as yf  # lazy import: keeps import cost and test footprint low

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if frame is None or frame.empty:
                raise ValueError(f"Yahoo returned no rows for {ticker!r}")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            close = frame["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance may wrap a single ticker
                close = close.squeeze("columns")
            series = close
            series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
            series = series.astype(float).sort_index()
            series.name = ticker.upper()
            if series.isna().all():
                raise ValueError(f"Yahoo returned only missing values for {ticker!r}")
            return series
        except Exception as exc:  # noqa: BLE001 - yfinance raises a variety of errors
            last_error = exc
            if attempt < max_retries:
                sleep_for = backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Download attempt %d/%d for %s failed (%s); retrying in %.1fs",
                    attempt,
                    max_retries,
                    ticker,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
    raise RuntimeError(
        f"Could not download adjusted close for {ticker!r} "
        f"after {max_retries} attempts: {last_error}"
    ) from last_error


def _cache_covers_window(series: pd.Series, start: str, end: str) -> bool:
    """Return True when the cached series spans the requested ``start..end`` window.

    ``end`` is exclusive (Yahoo's download convention), so the last expected
    trading day is the calendar day before ``end``. A tolerance of
    :data:`CACHE_COVERAGE_TOLERANCE_DAYS` calendar days at each edge absorbs
    weekends and market holidays; a cache truncated at either end by more than
    that fails the check instead of silently shortening the study.
    """
    tolerance = pd.Timedelta(days=CACHE_COVERAGE_TOLERANCE_DAYS)
    first, last = series.index.min(), series.index.max()
    starts_early_enough = first <= pd.Timestamp(start) + tolerance
    expected_last = pd.Timestamp(end) - pd.Timedelta(days=1)  # end is exclusive
    ends_late_enough = last >= expected_last - tolerance
    return bool(starts_early_enough and ends_late_enough)


def load_or_fetch(
    ticker: str,
    start: str,
    end: str,
    raw_dir: str | Path,
    refresh: bool = False,
) -> pd.Series:
    """Load a ticker's adjusted close from the local cache, else download it.

    Cache-first policy with a coverage check: on a cache hit the CSV is read
    and validated against the requested ``start..end`` window, so runs are
    reproducible without network access — and a cache that only covers part of
    the configured window (e.g. written by an earlier run with different
    dates) is not silently accepted. An insufficient cache triggers a warning
    and a re-download; if that re-download fails, a clear error is raised
    rather than returning a short series. Pass ``refresh=True`` (or use
    ``scripts/download_data.py --refresh``) to force a fresh download.

    Args:
        ticker: Yahoo ticker symbol.
        start: Start date ``YYYY-MM-DD``.
        end: End date ``YYYY-MM-DD`` (exclusive, Yahoo convention).
        raw_dir: Directory holding the per-ticker CSV cache.
        refresh: When True, bypass the cache and re-download.

    Returns:
        Adjusted-close ``pd.Series`` named after the ticker.
    """
    path = cache_path(raw_dir, ticker)
    if path.exists() and not refresh:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        series = frame.iloc[:, 0].astype(float)
        series.name = ticker.upper()
        if _cache_covers_window(series, start, end):
            logger.info(
                "Cache hit for %s: %d rows (%s to %s)",
                ticker,
                len(series),
                series.index.min(),
                series.index.max(),
            )
            return series
        logger.warning(
            "Cache for %s covers %s to %s, which does NOT cover the requested "
            "window %s to %s; re-downloading",
            ticker,
            series.index.min().date(),
            series.index.max().date(),
            start,
            end,
        )
    try:
        series = fetch_adjusted_close(ticker, start, end)
    except Exception as exc:  # noqa: BLE001 - yfinance/urllib raise a variety of errors
        if path.exists() and not refresh:
            raise RuntimeError(
                f"Cached data for {ticker!r} at {path} does not cover the requested "
                f"window {start} to {end}, and the re-download failed ({exc}). "
                "Refresh or delete the stale cache and retry with network access."
            ) from exc
        raise
    series.to_frame().to_csv(path, index=True, index_label="Date")
    logger.info("Downloaded and cached %s: %d rows", ticker, len(series))
    return series


def load_price_frame(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Assemble a wide adjusted-close frame (one column per symbol).

    Args:
        tickers: Symbols to load (universe + benchmark).
        start: Start date ``YYYY-MM-DD``.
        end: End date ``YYYY-MM-DD``.
        raw_dir: CSV cache directory.
        refresh: Force re-download when True.

    Returns:
        DataFrame indexed by date with one float column per symbol.
    """
    series_by_symbol = {
        ticker: load_or_fetch(ticker, start, end, raw_dir, refresh=refresh)
        for ticker in tickers
    }
    frame = pd.concat(series_by_symbol, axis=1).sort_index()
    frame.index.name = "Date"
    return frame.astype(float)


def validate_prices(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean an adjusted-price frame.

    Contract enforced (any violation of the first two rules is a hard error):

    * The index must be a ``DatetimeIndex`` that is **strictly increasing and
      unique**; otherwise a ``ValueError`` is raised.
    * All prices must be strictly positive; negative or zero prices raise a
      ``ValueError``.
    * *Missing-value policy*: isolated gaps of up to
      :data:`MAX_FFILL_LIMIT` (5) consecutive trading days are forward-filled;
      rows still containing missing values afterwards (e.g. leading rows
      before a symbol's listing date) are dropped. Columns with more than
      1% missing inputs generate a recorded warning.

    Args:
        frame: Wide price frame (one column per symbol).

    Returns:
        Tuple ``(clean_frame, report)`` where ``report`` is a JSON-serializable
        dict documenting row counts, date coverage, per-column missing-value
        counts before/after cleaning, and any warnings.

    Raises:
        ValueError: On a non-monotonic/duplicated index, a non-datetime index,
            or non-positive prices.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Price index must be a pandas DatetimeIndex.")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(
            "Price index must be strictly increasing and unique "
            "(sort the frame and drop duplicated dates before validating)."
        )
    if (frame <= 0).any().any():
        bad = frame[frame <= 0].stack().index.tolist()
        raise ValueError(f"Prices must be strictly positive; violations at {bad}.")

    original_rows = len(frame)
    missing_before = frame.isna().sum().astype(int)

    repaired = frame.ffill(limit=MAX_FFILL_LIMIT)
    clean = repaired.dropna(how="any")
    missing_after = clean.isna().sum().astype(int)

    warnings: list[str] = []
    for column, count in missing_before.items():
        if count > 0:
            fraction = count / original_rows
            if fraction > MISSING_WARN_FRACTION:
                warnings.append(
                    f"{column}: {count} missing prices ({fraction:.2%} of rows) before cleaning."
                )

    report = {
        "rows_input": original_rows,
        "rows_after_cleaning": len(clean),
        "rows_dropped": original_rows - len(clean),
        "start": str(clean.index.min().date()) if len(clean) else None,
        "end": str(clean.index.max().date()) if len(clean) else None,
        "symbols": list(clean.columns),
        "missing_before": {k: int(v) for k, v in missing_before.items()},
        "missing_after": {k: int(v) for k, v in missing_after.items()},
        "max_ffill_limit": MAX_FFILL_LIMIT,
        "warnings": warnings,
    }
    clean = clean.copy()
    clean.index.name = "Date"
    return clean, report
