# nfl-model

A Python project for NFL prediction modeling.

## Project Structure

```
nfl-model/
├── data/
│   ├── raw/          # Raw, unprocessed data (e.g. pulled from nfl_data_py)
│   └── processed/    # Cleaned/feature-engineered data ready for modeling
├── src/               # Source code (data loading, feature engineering, models)
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

> **Note:** This machine's default `python` is 3.14, which is too new for `nfl_data_py`'s
> pinned `numpy<2.0`/`pandas<2.0` requirement (no prebuilt wheels exist for 3.14, and there's
> no C compiler here to build from source). The project venv was created with **Python 3.12**
> (via `py -3.12`, found under the Anaconda install) instead, using newer `pandas`/`numpy` that
> `nfl_data_py` actually works fine with despite its stale metadata pins.

Create a virtual environment and install dependencies:

```bash
py -3.12 -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

If you don't have Python 3.12 available, `pip install nfl_data_py` may fail while resolving
`numpy`/`pandas`. In that case install `pandas`, `numpy`, and `scikit-learn` first, then install
`nfl_data_py` with `--no-deps` (it works fine with modern pandas/numpy in practice).

## Dependencies

- [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) — NFL play-by-play, schedule, and roster data
- `pandas` — data manipulation
- `numpy` — numerical computing
- `scikit-learn` — machine learning models and utilities
