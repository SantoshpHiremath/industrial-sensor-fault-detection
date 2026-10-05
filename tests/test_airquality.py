"""
Tests for the Air Quality module (De Vito et al., 2008, UCI ML
Repository — real gas multisensor device deployed roadside in an Italian
city, CC BY 4.0, non-commercial research use).

Real messiness this file has to handle, verified by direct inspection of
the downloaded CSV (not assumed):
- 114 fully-blank trailing rows (a spreadsheet-export artifact) and two
  unnamed trailing columns that must be dropped.
- Missing sensor readings are encoded as the sentinel value -200, not
  NaN, and must be converted before any statistics are computed on them.
- NMHC(GT) is missing (-200) in ~90% of rows (8443/9357) — a dead
  reference channel, not a usable feature at this missingness rate.
- CO(GT), the regression target used here, is missing in ~18% of rows
  (1683/9357) and those rows must be dropped rather than imputed, since
  imputing the prediction target itself would be circular.
"""
import numpy as np
import pandas as pd
import pytest

from airquality import clean_airquality, load_airquality


class TestLoadAirquality:
    def test_drops_fully_blank_trailing_rows(self):
        df = load_airquality("data/airquality_raw.csv")
        assert df[["Date", "Time"]].isna().all(axis=1).sum() == 0

    def test_drops_unnamed_trailing_columns(self):
        df = load_airquality("data/airquality_raw.csv")
        assert not any(c.startswith("Unnamed") for c in df.columns)

    def test_expected_row_count(self):
        df = load_airquality("data/airquality_raw.csv")
        # Verified directly against the downloaded file: 9357 real hourly
        # records after dropping the 114 blank trailing rows.
        assert len(df) == 9357


class TestCleanAirquality:
    def test_sentinel_minus_200_becomes_nan_before_target_rows_are_dropped(self):
        # clean_airquality() drops rows where the TARGET is missing (see
        # test_drops_rows_missing_target below), so to observe the -200
        # sentinel actually being converted to NaN -- rather than the row
        # simply vanishing -- this checks a NON-target column, where a
        # -200 reading is converted but the row is kept.
        df = pd.DataFrame({
            "CO(GT)": [2.6, 3.0, 3.1],
            "T": [13.6, -200.0, 11.0],
        })
        out = clean_airquality(df, target="CO(GT)")
        assert len(out) == 3
        assert out["T"].isna().sum() == 1
        assert out["T"].iloc[1] != -200.0

    def test_drops_rows_missing_target(self):
        df = pd.DataFrame({
            "CO(GT)": [2.6, -200.0, 3.1],
            "T": [13.6, 12.0, 11.0],
        })
        out = clean_airquality(df, target="CO(GT)")
        assert len(out) == 2
        assert out["CO(GT)"].isna().sum() == 0

    def test_drops_nmhc_gt_as_unusably_sparse(self):
        # NMHC(GT) is -200 (missing) in ~90% of real rows -- verified by
        # direct inspection, not assumed. A feature missing 9 times out
        # of 10 cannot be meaningfully imputed and is dropped rather than
        # silently filled, which would mostly be inventing data.
        df = pd.DataFrame({
            "CO(GT)": [2.6, 3.1, 2.0, 1.8],
            "NMHC(GT)": [-200.0, -200.0, -200.0, 150.0],
            "T": [13.6, 12.0, 11.0, 10.0],
        })
        out = clean_airquality(df, target="CO(GT)")
        assert "NMHC(GT)" not in out.columns
