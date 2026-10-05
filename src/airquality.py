"""
Cleaning utilities for the UCI Air Quality dataset (De Vito et al., 2008;
CC BY 4.0, non-commercial research use). Real gas multisensor device data
from an Italian city, used here for a regression task: predict a
reference-grade CO concentration reading from the cheaper co-located
metal-oxide sensor array (PT08.S1..S5) plus temperature/humidity — the
same "cheap sensor estimates an expensive/slow measurement" problem
pattern common in industrial process monitoring.
"""
import pandas as pd

MISSING_SENTINEL = -200.0
NMHC_MISSING_FRACTION_THRESHOLD = 0.5


def load_airquality(path: str) -> pd.DataFrame:
    """Load the raw semicolon-delimited, comma-decimal CSV, dropping the
    fully-blank trailing rows and unnamed trailing columns that are
    artifacts of how the original file was exported from a spreadsheet.
    """
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["Date", "Time"], how="all").reset_index(drop=True)
    return df


def clean_airquality(df: pd.DataFrame, target: str = "CO(GT)") -> pd.DataFrame:
    """Replace the -200 missing-value sentinel with NaN, drop rows where
    the regression target itself is missing (imputing the target would be
    circular), and drop NMHC(GT) specifically since it is missing in
    roughly 90% of real rows -- too sparse to impute meaningfully.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].replace(MISSING_SENTINEL, pd.NA)
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if "NMHC(GT)" in out.columns:
        out = out.drop(columns=["NMHC(GT)"])

    out = out.dropna(subset=[target]).reset_index(drop=True)
    return out
