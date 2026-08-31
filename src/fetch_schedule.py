"""Fetch NFL schedules and results and save them to data/raw/schedules.csv.

Usage:
    python src/fetch_schedule.py
"""

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

SEASONS = list(range(2021, 2026))  # 2021 through 2025
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "schedules.csv"


def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    """Pull schedule/result data for the given seasons via nfl_data_py."""
    return nfl.import_schedules(seasons)


def main() -> None:
    df = fetch_schedules(SEASONS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
