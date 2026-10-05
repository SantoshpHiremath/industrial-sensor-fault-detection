"""
Cleaning utilities for the SECOM industrial sensor dataset.

The dataset is real fab sensor data (McCann & Johnston, 2008, UCI ML
Repository, CC BY 4.0): 1567 production runs x 591 anonymized sensor
readings, plus a Time column and a Pass/Fail label. It is genuinely messy
in two ways that matter for how it must be handled:

1. Scattered missingness (4.52% of cells, across 538/591 columns, 28
   columns >50% missing) from sensors that occasionally failed to report.
2. 116 zero-variance ("dead") sensor columns — sensors that recorded the
   exact same value (or were entirely NaN) for all 1567 runs. These carry
   no information for any model and are a very real artifact of large
   real-world sensor arrays, not a synthetic-data quirk.
"""
import pandas as pd


def load_secom(path: str) -> pd.DataFrame:
    """Load the raw SECOM CSV and parse the Time column as a datetime."""
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def drop_constant_columns(df: pd.DataFrame, return_dropped: bool = False):
    """Drop columns with zero variance (a single distinct non-null value,
    or entirely NaN). A column that is 100% missing is treated as
    constant/dead rather than as a candidate for imputation: there is no
    observed value anywhere to impute from, so a median fill would just
    invent a number with no evidential basis.
    """
    nunique = df.nunique(dropna=True)
    all_nan = df.isna().all()
    dead = nunique.le(1) | all_nan
    dropped = df.columns[dead].tolist()
    out = df.drop(columns=dropped)
    if return_dropped:
        return out, dropped
    return out


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns whose fraction of missing values is >= threshold.

    Boundary behavior is explicit and tested: a column exactly at the
    threshold is dropped (>=), not kept, so threshold=0.5 means "drop
    anything half-missing or worse."
    """
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac >= threshold].index.tolist()
    return df.drop(columns=to_drop)


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining missing values with each column's median.

    Median rather than mean because sensor readings can have skewed
    distributions and occasional extreme outliers (e.g. a sensor glitch
    reading far outside normal range); a median is more robust to that
    than a mean.
    """
    return df.fillna(df.median(numeric_only=True))
