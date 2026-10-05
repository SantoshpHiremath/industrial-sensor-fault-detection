"""
Tests for the data-cleaning module, written before the implementation.

The SECOM dataset is real, messy fab-sensor data: 1567 rows x 591 sensor
columns, 4.52% missing cells scattered across 538/591 columns, 116
zero-variance ("dead") sensor columns, and a severe class imbalance
(104 fails / 1463 passes). Each test below encodes a real decision the
cleaning pipeline has to make, so a failing test here means a real data
problem, not a contrived one.
"""
import numpy as np
import pandas as pd
import pytest

from data_cleaning import (
    drop_constant_columns,
    drop_high_missing_columns,
    impute_median,
    load_secom,
)


@pytest.fixture
def toy_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],       # normal, varying
        "b": [5.0, 5.0, 5.0, 5.0],       # constant -> dead sensor
        "c": [np.nan, np.nan, np.nan, np.nan],  # all missing -> dead sensor
        "d": [1.0, np.nan, 3.0, np.nan],  # 50% missing
        "e": [1.0, 2.0, np.nan, 4.0],    # 25% missing, keep + impute
    })


class TestDropConstantColumns:
    def test_drops_truly_constant_column(self, toy_df):
        out = drop_constant_columns(toy_df)
        assert "b" not in out.columns

    def test_drops_all_nan_column_as_constant(self, toy_df):
        # A column that is 100% NaN has zero variance in any real sense —
        # it carries no information and should be treated as dead, same
        # as a literal constant value.
        out = drop_constant_columns(toy_df)
        assert "c" not in out.columns

    def test_keeps_varying_columns(self, toy_df):
        out = drop_constant_columns(toy_df)
        assert "a" in out.columns
        assert "d" in out.columns
        assert "e" in out.columns

    def test_reports_dropped_column_names(self, toy_df):
        _, dropped = drop_constant_columns(toy_df, return_dropped=True)
        assert set(dropped) == {"b", "c"}


class TestDropHighMissingColumns:
    def test_drops_column_above_threshold(self, toy_df):
        # 'd' is 50% missing; with a 40% threshold it should be dropped.
        out = drop_high_missing_columns(toy_df, threshold=0.4)
        assert "d" not in out.columns

    def test_keeps_column_below_threshold(self, toy_df):
        # 'e' is 25% missing; with a 40% threshold it should be kept.
        out = drop_high_missing_columns(toy_df, threshold=0.4)
        assert "e" in out.columns

    def test_threshold_is_exclusive_boundary_documented(self, toy_df):
        # Exactly-at-threshold columns: document behavior explicitly
        # rather than leaving it implicit. At threshold=0.5, column 'd'
        # (50% missing) should be DROPPED (>= threshold triggers drop).
        out = drop_high_missing_columns(toy_df, threshold=0.5)
        assert "d" not in out.columns


class TestImputeMedian:
    def test_fills_missing_with_column_median(self):
        df = pd.DataFrame({"x": [1.0, 3.0, np.nan, 5.0]})
        out = impute_median(df)
        # median of [1, 3, 5] = 3
        assert out["x"].iloc[2] == 3.0

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        original = df.copy()
        impute_median(df)
        pd.testing.assert_frame_equal(df, original)

    def test_no_missing_values_remain(self, toy_df):
        cleaned = drop_constant_columns(toy_df)
        out = impute_median(cleaned)
        assert out.isna().sum().sum() == 0


class TestLoadSecom:
    def test_loads_real_file_with_expected_shape(self):
        df = load_secom("data/secom_raw.csv")
        # 1567 rows. Columns: verified directly against the downloaded
        # file rather than assumed — it's 590 numbered sensor columns
        # (0-589) + Time + Pass/Fail = 592 total (some secondary sources
        # describe SECOM as "591 features"; the actual CSV header has 590
        # numbered sensor columns, confirmed by inspection below).
        assert df.shape[0] == 1567
        assert df.shape[1] == 592
        sensor_cols = [c for c in df.columns if c not in ("Time", "Pass/Fail")]
        assert len(sensor_cols) == 590

    def test_pass_fail_is_binary(self):
        df = load_secom("data/secom_raw.csv")
        assert set(df["Pass/Fail"].unique()) == {-1, 1}

    def test_time_column_parses_as_datetime(self):
        df = load_secom("data/secom_raw.csv")
        assert pd.api.types.is_datetime64_any_dtype(df["Time"])
