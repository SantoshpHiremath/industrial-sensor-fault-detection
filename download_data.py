"""
Downloads the two real datasets this project uses. Run this once before
running the demos or tests.

Both datasets originate from the UCI Machine Learning Repository. They are
fetched here from GitHub mirrors of the same files, which match the
original UCI files (same row counts, same columns, same values).
"""
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

SOURCES = {
    "secom_raw.csv": (
        "https://raw.githubusercontent.com/sharmaroshan/SECOM-Detecting-Defected-Items/master/uci-secom.csv"
    ),
    "airquality_raw.csv": (
        "https://raw.githubusercontent.com/asharvi1/UCI-Air-Quality-Data/master/AirQualityUCI.csv"
    ),
}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for filename, url in SOURCES.items():
        dest = DATA_DIR / filename
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"  saved {dest} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
