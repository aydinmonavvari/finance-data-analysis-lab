#!/usr/bin/env python3
"""CLI: download and cache daily adjusted closes for all configured symbols.

Cache-first policy: symbols already present in ``data/raw/`` are skipped unless
``--refresh`` is passed. On cache miss the symbol is fetched from Yahoo Finance
(via ``yfinance``) with retries, then written to the cache as CSV.

Examples
--------
    python scripts/download_data.py                 # fetch only what is missing
    python scripts/download_data.py --refresh       # force re-download of all
    python scripts/download_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

from finance_data_analysis_lab.config import load_config
from finance_data_analysis_lab.data import cache_path, load_or_fetch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("download_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download every symbol even when a cached CSV exists.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.ensure_output_dirs()
    logger.info(
        "Universe: %s | benchmark: %s | window: %s..%s",
        ", ".join(config.tickers),
        config.benchmark,
        config.start,
        config.end,
    )

    failures: list[str] = []
    for symbol in config.all_symbols:
        path = cache_path(config.raw_dir, symbol)
        try:
            series = load_or_fetch(
                symbol, config.start, config.end, config.raw_dir, refresh=args.refresh
            )
        except RuntimeError as exc:
            failures.append(symbol)
            logger.error("%s", exc)
            continue
        logger.info(
            "%s: %d rows, %s to %s -> %s",
            symbol,
            len(series),
            series.index.min().date(),
            series.index.max().date(),
            path,
        )

    if failures:
        logger.error("Failed symbols: %s", ", ".join(failures))
        return 1
    logger.info("All symbols cached under %s", config.raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
