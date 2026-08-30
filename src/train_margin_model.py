"""Train a game-margin regression model (linear regression + XGBoost).

Same data source and train/test split as train_model.py (2022-2024 train,
2025 test), and the same final 15-feature set (13 original +
home_starters_out + away_starters_out), but predicting margin =
home_score - away_score instead of a binary winner.

Compares both models' MAE/RMSE against the betting market's own margin
prediction: spread_line is already the market's expected margin (positive =
home favored by that many points, per the sign convention verified in
train_model.py/train_player_model.py), so it's a natural regression
baseline the same way moneyline-implied probability was for the win/loss
task.

Usage:
    python src/train_margin_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

FEATURE_COLS = [
    "home_pts_scored_last5",
    "home_pts_allowed_last5",
    "away_pts_scored_last5",
    "away_pts_allowed_last5",
    "home_win_streak",
    "away_win_streak",
    "rest_advantage",
    "is_dome",
    "home_qb_change",
    "away_qb_change",
    "home_qb_rating_last5",
    "away_qb_rating_last5",
    "div_game",
    "home_starters_out",
    "away_starters_out",
]
TARGET_COL = "margin"

TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=FEATURE_COLS + ["spread_line"]).reset_index(drop=True)
    df[TARGET_COL] = df["home_score"] - df["away_score"]
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"] == TEST_SEASON]
    return train, test


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f} points, RMSE={rmse:.2f} points")
    return {"model": name, "MAE": mae, "RMSE": rmse}


def main() -> None:
    df = load_data()
    print(f"Rows after dropping nulls in feature columns (and spread_line): {len(df)}")

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
    print("Coefficients (raw scale, margin points per 1 unit of feature):")
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

    # --- Betting-market baseline: spread_line as a margin prediction --------
    # spread_line is positive when the home team is favored by that many
    # points, i.e. it already IS the market's predicted margin -- no
    # transformation needed, same role moneyline-implied probability played
    # for the win/loss task in train_model.py.
    market_pred = test["spread_line"].to_numpy()
    market_metrics = evaluate("Market (spread_line)", y_test, market_pred)

    print()
    print("=== Model comparison, test season 2025 ===")
    print(
        pd.DataFrame([linreg_metrics, xgb_metrics, market_metrics])
        .set_index("model")
        .to_string()
    )


if __name__ == "__main__":
    main()
