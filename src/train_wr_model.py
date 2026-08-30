"""Train a WR receiving-yards prop model (linear regression + XGBoost).

Mirrors train_rb_model.py exactly, for receiving instead of rushing.
Loads data/processed/wr_receiving_features.csv (which already includes
is_dome, rest_advantage, spread_line, total_line, and wr_injury_status from
wr_features.py), drops rows with null features, trains on seasons 2022-2024
and evaluates on season 2025 (same season-based split as train_model.py).

Usage:
    python src/train_wr_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
WR_FEATURES_PATH = BASE_DIR / "data" / "processed" / "wr_receiving_features.csv"

FEATURE_COLS = [
    "wr_rec_yards_last5",
    "wr_receptions_last5",
    "wr_targets_last5",
    "target_share_last5",
    "opponent_pass_defense_rank",
    "spread_line",
    "total_line",
    "is_dome",
    "rest_advantage",
    "wr_injury_status",
]
TARGET_COL = "receiving_yards"

TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025

RANDOM_STATE = 42
N_SAMPLE_PREDICTIONS = 10


def load_data() -> pd.DataFrame:
    df = pd.read_csv(WR_FEATURES_PATH)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"] == TEST_SEASON]
    return train, test


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f} yards, RMSE={rmse:.2f} yards")
    return {"model": name, "MAE": mae, "RMSE": rmse}


def main() -> None:
    df = load_data()
    print(f"Rows after dropping nulls in feature/target columns: {len(df)}")

    train, test = split_train_test(df)
    print(f"Train rows (seasons {TRAIN_SEASONS}): {len(train)}")
    print(f"Test rows (season {TEST_SEASON}): {len(test)}")

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    print()
    print("=== Linear Regression ===")
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    linreg_pred = linreg.predict(X_test)
    linreg_metrics = evaluate("Linear Regression", y_test, linreg_pred)

    coefs = pd.DataFrame({"feature": FEATURE_COLS, "coefficient": linreg.coef_}).sort_values(
        "coefficient", key=np.abs, ascending=False
    )
    print()
    print("Coefficients (raw scale, yards per 1 unit of feature):")
    print(coefs.to_string(index=False))
    print(f"Intercept: {linreg.intercept_:.2f}")

    print()
    print("=== XGBoost Regressor ===")
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = evaluate("XGBoost", y_test, xgb_pred)

    importances = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": xgb.feature_importances_}
    ).sort_values("importance", ascending=False)
    print()
    print("Feature importances (gain-based):")
    print(importances.to_string(index=False))

    print()
    print("=== Model comparison, test season 2025 ===")
    print(pd.DataFrame([linreg_metrics, xgb_metrics]).set_index("model").to_string())

    print()
    print(f"=== {N_SAMPLE_PREDICTIONS} random test-set predictions vs. actual ===")
    sample = test.sample(N_SAMPLE_PREDICTIONS, random_state=RANDOM_STATE)
    comparison = pd.DataFrame(
        {
            "player_name": sample["player_name"],
            "team": sample["team"],
            "opponent": sample["opponent"],
            "week": sample["week"],
            "actual_yards": sample[TARGET_COL],
            "linreg_pred": linreg.predict(sample[FEATURE_COLS]),
            "xgb_pred": xgb.predict(sample[FEATURE_COLS]),
        }
    )
    comparison["linreg_error"] = comparison["linreg_pred"] - comparison["actual_yards"]
    comparison["xgb_error"] = comparison["xgb_pred"] - comparison["actual_yards"]
    print(comparison.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
