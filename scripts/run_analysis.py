#!/usr/bin/env python3
"""CLI: run the full descriptive analysis pipeline.

Loads (or downloads and caches) the configured universe, validates prices,
computes metrics and statistical diagnostics, renders all figures and writes
``reports/metrics.json`` and ``reports/summary.md``.

Examples
--------
    python scripts/run_analysis.py                 # cache-first, offline if cached
    python scripts/run_analysis.py --refresh-data  # re-download prices first
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_data_analysis_lab.config import load_config  # noqa: E402
from finance_data_analysis_lab.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_analysis")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download cached prices before analysing.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_pipeline(config, refresh_data=args.refresh_data)

    validation = result["validation"]
    logger.info(
        "Validated window %s..%s (%d trading days, %d rows dropped)",
        validation["start"],
        validation["end"],
        validation["rows_after_cleaning"],
        validation["rows_dropped"],
    )
    for artifact in result["artifacts"]:
        logger.info("Wrote %s", artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
