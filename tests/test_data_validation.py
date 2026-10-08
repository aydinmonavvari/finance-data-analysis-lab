"""Unit tests for price validation and the cache-first ingestion policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_data_analysis_lab.data import (
    cache_path,
    load_or_fetch,
    validate_prices,
)


def _frame(data: dict[str, list[float | None]], start: str = "2024-01-01") -> pd.DataFrame:
    n = len(next(iter(data.values())))
    index = pd.bdate_range(start=start, periods=n)
    frame = pd.DataFrame(data, index=index, dtype=float)
    frame.index.name = "Date"
    return frame


class TestValidationContract:
    def test_rejects_non_datetime_index(self):
        frame = pd.DataFrame({"AAPL": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_prices(frame)

    def test_rejects_unsorted_index(self):
        frame = _frame({"AAPL": [100.0, 102.0, 101.0]})
        unsorted_frame = frame.iloc[::-1]
        assert not unsorted_frame.index.is_monotonic_increasing
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_prices(unsorted_frame)

    def test_rejects_duplicate_dates(self):
        frame = _frame({"AAPL": [100.0, 101.0, 102.0]})
        duplicated = pd.concat([frame, frame.iloc[[0]]]).sort_index()
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_prices(duplicated)

    def test_rejects_negative_prices(self):
        frame = _frame({"AAPL": [100.0, -1.0, 101.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            validate_prices(frame)

    def test_rejects_zero_prices(self):
        frame = _frame({"AAPL": [100.0, 0.0, 101.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            validate_prices(frame)

    def test_valid_frame_passes_unchanged(self):
        frame = _frame({"AAPL": [100.0, 101.0, 99.5]})
        clean, report = validate_prices(frame)
        pd.testing.assert_frame_equal(clean, frame)
        assert report["rows_dropped"] == 0
        assert report["warnings"] == []


class TestMissingValuePolicy:
    def test_interior_gap_forward_filled_within_limit(self):
        # One missing day inside a 5-day tolerance: forward-filled, nothing dropped.
        frame = _frame({"AAPL": [100.0, np.nan, 102.0]})
        clean, report = validate_prices(frame)
        assert len(clean) == 3
        assert clean["AAPL"].iloc[1] == pytest.approx(100.0)  # carried forward
        assert report["missing_before"]["AAPL"] == 1
        assert report["missing_after"]["AAPL"] == 0
        assert report["rows_dropped"] == 0

    def test_long_leading_gap_is_dropped_not_fabricated(self):
        # Leading NaNs cannot be forward-filled, so those rows are dropped.
        frame = _frame({"AAPL": [np.nan, np.nan, 100.0, 101.0, 102.0]})
        clean, report = validate_prices(frame)
        assert len(clean) == 3
        assert report["rows_dropped"] == 2
        assert report["start"] == str(frame.index[2].date())

    def test_gap_beyond_limit_is_dropped(self):
        # Six consecutive NaNs: the first five are forward-filled (the documented
        # limit), the sixth cannot be repaired and its row is dropped.
        values = [100.0] + [np.nan] * 6 + [110.0]
        frame = _frame({"AAPL": values})
        clean, report = validate_prices(frame)
        assert len(clean) == 7
        assert report["rows_dropped"] == 1
        assert clean["AAPL"].isna().sum() == 0
        assert clean["AAPL"].iloc[5] == pytest.approx(100.0)  # last carried-forward value
        assert clean["AAPL"].iloc[6] == pytest.approx(110.0)

    def test_warning_recorded_for_heavily_missing_column(self):
        frame = _frame(
            {"GOOD": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
             "GAPPY": [100.0, np.nan, np.nan, np.nan, np.nan, 105.0]}
        )
        _, report = validate_prices(frame)
        assert any("GAPPY" in warning for warning in report["warnings"])
        assert not any("GOOD" in warning for warning in report["warnings"])


class TestCachePolicy:
    def test_cache_path_is_deterministic_and_sanitized(self):
        assert cache_path("data/raw", "aapl").name == "AAPL.csv"
        assert cache_path("data/raw", "^GSPC").name == "_GSPC.csv"

    def test_load_or_fetch_reads_cache_without_network(self, tmp_path):
        index = pd.bdate_range("2024-01-01", periods=3)
        index.name = "Date"
        index.freq = None  # a CSV round-trip does not preserve frequency metadata
        cached = pd.Series([100.0, 101.0, 102.0], index=index, name="AAPL")
        cached.to_frame().to_csv(tmp_path / "AAPL.csv", index_label="Date")

        result = load_or_fetch("AAPL", "2024-01-01", "2024-01-05", tmp_path, refresh=False)
        pd.testing.assert_series_equal(result, cached)

    def test_cache_miss_would_attempt_download(self, tmp_path, monkeypatch):
        # On cache miss the loader must call the network fetcher exactly once.
        calls = []

        def fake_fetch(ticker, start, end, **kwargs):
            calls.append(ticker)
            return pd.Series(
                [1.0, 2.0], index=pd.bdate_range("2024-01-01", periods=2), name=ticker
            )

        monkeypatch.setattr("finance_data_analysis_lab.data.fetch_adjusted_close", fake_fetch)
        result = load_or_fetch("FAKE", "2024-01-01", "2024-01-05", tmp_path)
        assert calls == ["FAKE"]
        assert len(result) == 2
        assert (tmp_path / "FAKE.csv").exists()  # downloaded frame was cached
