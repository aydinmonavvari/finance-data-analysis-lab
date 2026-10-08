"""Project configuration.

A small :class:`Config` dataclass holds the universe, sample window, rate
assumptions and filesystem paths. Values are loaded from
``configs/default.yaml`` when available; otherwise the same defaults are used
inline, so the package works even without a config file present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Repository root (src/finance_data_analysis_lab/config.py -> parents[2]).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Config:
    """Immutable analysis configuration.

    Attributes:
        tickers: Equity tickers included in the study universe.
        benchmark: Benchmark ticker used for beta/alpha and relative charts.
        start: Sample start date (YYYY-MM-DD, inclusive).
        end: Sample end date (YYYY-MM-DD, inclusive; Yahoo treats it as
            exclusive for downloads, so the last trading day strictly before
            this date is the effective end of the cached sample).
        rf_annual: Annual risk-free rate assumption used by Sharpe/Sortino and
            the excess-return definitions. 0.0 by default and configurable in
            ``configs/default.yaml``.
        periods_per_year: Annualization factor. 252 = approximate number of US
            equity trading days per year.
        raw_dir / processed_dir / figures_dir / reports_dir: Output locations,
            resolved against the repository root.
    """

    tickers: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "AMZN"]
    )
    benchmark: str = "SPY"
    start: str = "2018-01-01"
    end: str = "2026-10-08"
    rf_annual: float = 0.0
    periods_per_year: int = 252
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    figures_dir: Path = PROJECT_ROOT / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @property
    def all_symbols(self) -> list[str]:
        """Universe tickers plus the benchmark, in deterministic order."""
        return [*self.tickers, self.benchmark]

    def ensure_output_dirs(self) -> None:
        """Create the output directories if they do not exist."""
        for path in (self.raw_dir, self.processed_dir, self.figures_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def _apply_yaml(cfg: Config, payload: dict) -> Config:
    """Overlay a parsed YAML payload onto a Config, returning a new Config."""
    universe = payload.get("universe", {})
    dates = payload.get("dates", {})
    rates = payload.get("rates", {})
    paths = payload.get("paths", {})

    def _resolve(key: str, default: Path) -> Path:
        value = paths.get(key)
        if value is None:
            return default
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    return Config(
        tickers=list(universe.get("tickers", cfg.tickers)),
        benchmark=str(universe.get("benchmark", cfg.benchmark)),
        start=str(dates.get("start", cfg.start)),
        end=str(dates.get("end", cfg.end)),
        rf_annual=float(rates.get("rf_annual", cfg.rf_annual)),
        periods_per_year=int(cfg.periods_per_year),
        raw_dir=_resolve("raw_dir", cfg.raw_dir),
        processed_dir=_resolve("processed_dir", cfg.processed_dir),
        figures_dir=_resolve("figures_dir", cfg.figures_dir),
        reports_dir=_resolve("reports_dir", cfg.reports_dir),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load the analysis configuration.

    Args:
        path: Optional path to a YAML config file. When ``None`` the bundled
            ``configs/default.yaml`` is used if it exists; otherwise built-in
            defaults apply.

    Returns:
        A fully-resolved :class:`Config`.
    """
    base = Config()
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return _apply_yaml(base, payload)
    # No config file found: fall back to the built-in defaults so the package
    # remains usable outside a repository checkout.
    return base
