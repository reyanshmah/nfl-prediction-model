"""Train a QB passing-yards prop model (linear regression + XGBoost).

Loads data/processed/qb_passing_features.csv (which already includes
qb_injury_status and top_target_availability from player_features.py),
joins is_dome, a team-perspective rest_advantage, a team-perspective
spread_line, and total_line in from data/processed/games_with_features.csv,
drops rows with null features, trains on seasons 2022-2024 and evaluates on
season 2025 (same season-based split as train_model.py).

Usage:
    python src/train_player_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
QB_FEATURES_PATH = BASE_DIR / "data" / "processed" / "qb_passing_features.csv"
GAME_FEATURES_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

FEATURE_COLS = [
    "qb_pass_yards_last5",
    "qb_pass_attempts_last5",
    "opponent_pass_defense_rank",
    "is_dome",
    "rest_advantage",
    "spread_line",
    "total_line",
    "qb_injury_status",
    "top_target_availability",
]
TARGET_COL = "passing_yards"

TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025

RANDOM_STATE = 42
N_SAMPLE_PREDICTIONS = 10


def team_game_context(games: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) -> is_dome, rest_advantage, spread_line, and
    total_line, all from the QB's team's own perspective.

    games_with_features.csv's rest_advantage (home_rest - away_rest) and
    spread_line are both home-perspective and get sign-flipped for the away
    side. spread_line in this data is positive when the home team is
    favored (verified against home_moneyline/away_moneyline: positive
    spread_line always paired with a negative/favorite home_moneyline) --
    so flipping its sign for the away team gives a "this team's own spread"
    reading where positive = this team is favored, negative = underdog.
    total_line (the game's combined over/under) isn't team-specific, so it's
    joined unchanged for both sides.
    """
    home = games[
        ["season", "week", "home_team", "is_dome", "rest_advantage", "spread_line", "total_line"]
    ].rename(columns={"home_team": "team"})
    away = games[
        ["season", "week", "away_team", "is_dome", "rest_advantage", "spread_line", "total_line"]
    ].rename(columns={"away_team": "team"})
    away["rest_advantage"] = -away["rest_advantage"]
    away["spread_line"] = -away["spread_line"]
    return pd.concat([home, away], ignore_index=True)


def load_data() -> pd.DataFrame:
    qb_df = pd.read_csv(QB_FEATURES_PATH)
    games = pd.read_csv(GAME_FEATURES_PATH)

    context = team_game_context(games)
    df = qb_df.merge(context, on=["season", "week", "team"], how="left")

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


def over_under_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, line: np.ndarray) -> dict:
    """% of games where the model correctly calls over/under against `line`.

    A "hit" is the model's predicted side (over/under the line) matching the
    actual side. Exact ties (actual_yards == line) are pushes and excluded
    from the denominator, same as a real sportsbook would void them.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    line = np.asarray(line)

    push = y_true == line
    actual_over = y_true[~push] > line[~push]
    pred_over = y_pred[~push] > line[~push]
    hits = actual_over == pred_over

    return {
        "n_games": len(y_true),
        "n_pushes": int(push.sum()),
        "n_graded": len(hits),
        "hit_rate": hits.mean() if len(hits) else float("nan"),
    }


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
    sample_idx = sample.index
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

    # --- Over/under hit rate --------------------------------------------------
    # We don't have real historical prop lines, so this uses each QB's own
    # qb_pass_yards_last5 (already a model feature) as a stand-in "line" --
    # it's a plausible number a sportsbook might post, but it's not an actual
    # market line, and a real one would already price in more than a simple
    # rolling average does. Treat this as a rough self-consistency check on
    # the model's directional calls, not a claim about real betting edge.
    line = test["qb_pass_yards_last5"].to_numpy()
    ou_result = over_under_hit_rate(y_test.to_numpy(), linreg_pred, line)

    print()
    print("=== Over/under hit rate (linear regression), test season 2025 ===")
    print("Line used: qb_pass_yards_last5 (proxy -- no real historical prop odds available)")
    print(f"Games: {ou_result['n_games']}, pushes: {ou_result['n_pushes']}, graded: {ou_result['n_graded']}")
    print(f"Hit rate: {ou_result['hit_rate']:.4f}")


if __name__ == "__main__":
    main()
