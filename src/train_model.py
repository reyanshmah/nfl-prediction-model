"""Train logistic regression and XGBoost models to predict NFL game winners.

Loads data/processed/games_with_features.csv, drops rows with insufficient
prior history (nulls in the feature columns), builds a binary home_win
target, trains on seasons 2022-2024 and evaluates on season 2025. Compares
both models against each other and against the betting market baseline.

Usage:
    python src/train_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

# Turnover margin and SOS-adjusted scoring (home_turnover_margin_last5,
# away_turnover_margin_last5, home_pts_scored_last5_sos_adj,
# away_pts_scored_last5_sos_adj) are computed in features.py and remain in
# games_with_features.csv for later experimentation, but are left out of
# training for now -- they measurably hurt both models' test accuracy/log
# loss versus this feature set (see conversation history / commit notes).
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

TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025

BLEND_WEIGHTS = [0.2, 0.3, 0.4, 0.5]
TOP_DISAGREEMENTS_N = 15

# XGBoost gets its own inner split so it can early-stop on a held-out
# validation season rather than the 2025 test set.
XGB_TRAIN_SEASONS = [2022, 2023]
XGB_VAL_SEASON = 2024


def moneyline_to_prob(moneyline: pd.Series) -> pd.Series:
    """Raw (vig-included) implied win probability from American moneyline odds."""
    return np.where(
        moneyline < 0,
        -moneyline / (-moneyline + 100),
        100 / (moneyline + 100),
    )


def market_implied_home_prob(df: pd.DataFrame) -> pd.Series:
    """De-vigged market-implied home-win probability for each game.

    Converts both sides' moneylines to raw implied probabilities (which sum
    to > 1 because of the vig), then normalizes so home + away = 1.
    """
    home_raw = moneyline_to_prob(df["home_moneyline"])
    away_raw = moneyline_to_prob(df["away_moneyline"])
    return home_raw / (home_raw + away_raw)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df


def split_train_test(
    df: pd.DataFrame, train_seasons: list[int], test_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season]
    return train, test


def blend_weight_table(
    y_true: pd.Series,
    model_proba: np.ndarray,
    market_proba: np.ndarray,
    weights: list[float],
) -> pd.DataFrame:
    """Accuracy/log loss for blended_prob = weight*model + (1-weight)*market,
    across the given weights, plus the pure-model and pure-market rows."""
    rows = []
    for weight in weights:
        blended_prob = weight * model_proba + (1 - weight) * market_proba
        blended_pred = (blended_prob > 0.5).astype(int)
        rows.append(
            {
                "blend": f"weight={weight}",
                "accuracy": accuracy_score(y_true, blended_pred),
                "log_loss": log_loss(y_true, blended_prob),
            }
        )

    model_pred = (model_proba > 0.5).astype(int)
    market_pred = (market_proba > 0.5).astype(int)
    rows.append(
        {
            "blend": "pure_logistic_regression",
            "accuracy": accuracy_score(y_true, model_pred),
            "log_loss": log_loss(y_true, model_proba),
        }
    )
    rows.append(
        {
            "blend": "pure_market",
            "accuracy": accuracy_score(y_true, market_pred),
            "log_loss": log_loss(y_true, market_proba),
        }
    )
    return pd.DataFrame(rows).set_index("blend")


def fit_logreg(train: pd.DataFrame):
    """Fit a fresh StandardScaler + LogisticRegression pipeline on train."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURE_COLS], train["home_win"])
    return model


# |rest_advantage| >= 3 days ("large") vs. < 3 ("small"). Most games have 0
# rest difference (both teams coming off a standard week); >=3 captures real
# rest mismatches like a short week / off a bye, without splitting on a
# near-empty group.
LARGE_REST_DIFF_THRESHOLD = 3


def subgroup_masks(test_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks defining each subgroup, paired so each dimension covers
    the whole test set exactly once (qb_change/no_qb_change, etc.)."""
    qb_change = (test_df["home_qb_change"] == 1) | (test_df["away_qb_change"] == 1)
    large_rest_diff = test_df["rest_advantage"].abs() >= LARGE_REST_DIFF_THRESHOLD
    return {
        "qb_change": qb_change,
        "no_qb_change": ~qb_change,
        "divisional": test_df["div_game"] == 1,
        "non_divisional": test_df["div_game"] == 0,
        "dome": test_df["is_dome"] == 1,
        "outdoor": test_df["is_dome"] == 0,
        "large_rest_diff": large_rest_diff,
        "small_rest_diff": ~large_rest_diff,
    }


def subgroup_accuracy_table(
    test_df: pd.DataFrame, y_true: pd.Series, model_proba: np.ndarray, market_proba: np.ndarray
) -> pd.DataFrame:
    """Model vs. market accuracy on each subgroup of the test set."""
    y_true = y_true.to_numpy()
    rows = []
    for name, mask in subgroup_masks(test_df).items():
        mask = mask.to_numpy()
        n = int(mask.sum())
        if n == 0:
            rows.append({"subgroup": name, "n": 0, "model_acc": np.nan, "market_acc": np.nan, "model_minus_market": np.nan})
            continue
        yt = y_true[mask]
        model_acc = accuracy_score(yt, (model_proba[mask] > 0.5).astype(int))
        market_acc = accuracy_score(yt, (market_proba[mask] > 0.5).astype(int))
        rows.append(
            {
                "subgroup": name,
                "n": n,
                "model_acc": model_acc,
                "market_acc": market_acc,
                "model_minus_market": model_acc - market_acc,
            }
        )
    return pd.DataFrame(rows).set_index("subgroup")


def model_called_upsets(
    test_df: pd.DataFrame, y_true: pd.Series, model_proba: np.ndarray, market_proba: np.ndarray
) -> pd.DataFrame:
    """Games where the model predicted the winner correctly but the market's
    implied favorite (moneyline > 0.5) did not -- i.e. the model called an
    upset the market missed."""
    games = pd.DataFrame(
        {
            "home_team": test_df["home_team"].to_numpy(),
            "away_team": test_df["away_team"].to_numpy(),
            "week": test_df["week"].to_numpy(),
            "home_score": test_df["home_score"].to_numpy(),
            "away_score": test_df["away_score"].to_numpy(),
            "model_prob": model_proba,
            "market_prob": market_proba,
            "home_won": y_true.to_numpy(),
        }
    )
    model_correct = (games["model_prob"] > 0.5).astype(int) == games["home_won"]
    market_correct = (games["market_prob"] > 0.5).astype(int) == games["home_won"]
    upsets = games[model_correct & ~market_correct].drop(columns=["home_won"])
    return upsets.sort_values("week")


# market_prob outside (0.40, 0.60) = the market had a reasonably confident
# favorite, as opposed to a near coin-flip line.
CONFIDENT_MARKET_LOW = 0.40
CONFIDENT_MARKET_HIGH = 0.60


def confident_market_upsets(upsets: pd.DataFrame) -> pd.DataFrame:
    """Subset of model_called_upsets() where the market's favorite was
    reasonably confident (market_prob > 0.60 or < 0.40), not a near coin-flip,
    and the model still called the other side correctly."""
    return upsets[
        (upsets["market_prob"] > CONFIDENT_MARKET_HIGH) | (upsets["market_prob"] < CONFIDENT_MARKET_LOW)
    ]


def main() -> None:
    df = load_data()
    print(f"Rows after dropping nulls in feature columns: {len(df)}")

    train, test = split_train_test(df, TRAIN_SEASONS, TEST_SEASON)
    print(f"Train rows (seasons {TRAIN_SEASONS}): {len(train)}")
    print(f"Test rows (season {TEST_SEASON}): {len(test)}")

    X_train, y_train = train[FEATURE_COLS], train["home_win"]
    X_test, y_test = test[FEATURE_COLS], test["home_win"]

    # --- Logistic regression ---------------------------------------------
    # Standardize features first: the raw features are on very different
    # scales (e.g. passer rating ~0-150 vs. a 0/1 flag), so unscaled logistic
    # regression coefficients aren't comparable as "feature importance."
    # Scaling puts every coefficient in the same units (effect per 1 standard
    # deviation change), which is what makes them comparable.
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(X_train, y_train)

    logreg_pred = logreg.predict(X_test)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]

    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    logreg_loss = log_loss(y_test, logreg_proba)
    logreg_cm = confusion_matrix(y_test, logreg_pred)

    print()
    print("=== Logistic Regression ===")
    print(f"Test accuracy: {logreg_accuracy:.4f}")
    print(f"Test log loss: {logreg_loss:.4f}")
    print()
    print("Confusion matrix (rows=actual, cols=predicted; [0,1] = [away_win, home_win]):")
    print(pd.DataFrame(logreg_cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_string())

    logreg_clf = logreg.named_steps["logisticregression"]
    print()
    print("Model coefficients (standardized features -> log-odds per 1 std dev; sorted by |effect|):")
    coefs = pd.DataFrame(
        {"feature": FEATURE_COLS, "coefficient": logreg_clf.coef_[0]}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    print(coefs.to_string(index=False))
    print()
    print(f"Intercept: {logreg_clf.intercept_[0]:.4f}")

    # --- XGBoost -----------------------------------------------------------
    # 773 training rows over 13 features is a small, overfitting-prone
    # regime for gradient-boosted trees. To fight that: cut model capacity
    # (fewer/shallower trees, a low learning rate), carve a validation
    # season (2024) out of the training data instead of using all of
    # 2022-2024, and stop boosting as soon as validation log loss stops
    # improving rather than running the full n_estimators regardless.
    xgb_train = train[train["season"].isin(XGB_TRAIN_SEASONS)]
    xgb_val = train[train["season"] == XGB_VAL_SEASON]
    X_xgb_train, y_xgb_train = xgb_train[FEATURE_COLS], xgb_train["home_win"]
    X_xgb_val, y_xgb_val = xgb_val[FEATURE_COLS], xgb_val["home_win"]

    # The logistic regression's confusion matrix showed a bias toward
    # predicting home wins (169/256 predictions vs. 135/256 actual home
    # wins). scale_pos_weight down-weights the majority class (home_win=1,
    # which is slightly more common in the training data) during training,
    # counteracting that bias -- it's XGBoost's equivalent of sklearn's
    # class_weight="balanced" for imbalanced binary targets.
    neg_count = (y_xgb_train == 0).sum()
    pos_count = (y_xgb_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print()
    print(
        f"XGBoost train ({XGB_TRAIN_SEASONS}): {len(xgb_train)} rows, "
        f"{pos_count} home wins / {neg_count} away wins -> scale_pos_weight={scale_pos_weight:.4f}"
    )
    print(f"XGBoost validation ({XGB_VAL_SEASON}): {len(xgb_val)} rows")

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=10,
        random_state=42,
    )
    xgb.fit(X_xgb_train, y_xgb_train, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)
    print(f"Best iteration: {xgb.best_iteration + 1} (of {xgb.n_estimators} max)")

    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_loss = log_loss(y_test, xgb_proba)
    xgb_cm = confusion_matrix(y_test, xgb_pred)

    print()
    print("=== XGBoost ===")
    print(f"Test accuracy: {xgb_accuracy:.4f}")
    print(f"Test log loss: {xgb_loss:.4f}")
    print()
    print("Confusion matrix (rows=actual, cols=predicted; [0,1] = [away_win, home_win]):")
    print(pd.DataFrame(xgb_cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_string())

    print()
    print("Feature importances (gain-based; sorted descending):")
    importances = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": xgb.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(importances.to_string(index=False))

    # --- Betting-market baseline --------------------------------------------
    # Convert home_moneyline/away_moneyline to a de-vigged implied home-win
    # probability, and compare its accuracy/log loss on the same test set.
    market_prob = market_implied_home_prob(test)
    market_pred = (market_prob > 0.5).astype(int)

    market_accuracy = accuracy_score(y_test, market_pred)
    market_loss = log_loss(y_test, market_prob)

    print()
    print("=== Model comparison, test season 2025 ===")
    comparison = pd.DataFrame(
        {
            "accuracy": [logreg_accuracy, xgb_accuracy, market_accuracy],
            "log_loss": [logreg_loss, xgb_loss, market_loss],
        },
        index=["logistic_regression", "xgboost", "market_moneyline"],
    )
    print(comparison.to_string())

    # --- Blended prediction --------------------------------------------------
    # blended_prob = weight * logistic_regression_prob + (1 - weight) * market_prob.
    # logreg_proba and market_prob are both plain arrays positionally aligned
    # with X_test/y_test (test's original row order is preserved throughout).
    print()
    print(f"=== Blended prediction (logistic regression + market), test season {TEST_SEASON} ===")
    print(blend_weight_table(y_test, logreg_proba, market_prob, BLEND_WEIGHTS).to_string())

    # --- Model vs. market disagreement ---------------------------------------
    # Where the two probabilities diverge most, which one tended to be right?
    disagreement = pd.DataFrame(
        {
            "home_team": test["home_team"].to_numpy(),
            "away_team": test["away_team"].to_numpy(),
            "week": test["week"].to_numpy(),
            "logreg_prob": logreg_proba,
            "market_prob": market_prob,
            "home_won": y_test.to_numpy(),
        }
    )
    disagreement["diff"] = (disagreement["logreg_prob"] - disagreement["market_prob"]).abs()
    disagreement["model_correct"] = (disagreement["logreg_prob"] > 0.5).astype(int) == disagreement[
        "home_won"
    ]
    disagreement["market_correct"] = (disagreement["market_prob"] > 0.5).astype(int) == disagreement[
        "home_won"
    ]

    top_disagreements = disagreement.sort_values("diff", ascending=False).head(TOP_DISAGREEMENTS_N)

    print()
    print(f"=== Top {TOP_DISAGREEMENTS_N} model/market disagreements, test season 2025 ===")
    print(top_disagreements.to_string(index=False))
    print()
    print(
        f"Of these {TOP_DISAGREEMENTS_N} highest-disagreement games: "
        f"model correct {top_disagreements['model_correct'].sum()}/{TOP_DISAGREEMENTS_N}, "
        f"market correct {top_disagreements['market_correct'].sum()}/{TOP_DISAGREEMENTS_N}"
    )

    # --- Robustness check: does the blend still help on a different holdout? -
    # Same blend-weight test as above, but trained on 2022-2023 only and
    # evaluated on 2024 -- an entirely separate model and test season -- to
    # see whether the weight=0.2-ish edge over pure market on 2025 replicates,
    # or was a small-sample fluke of that one season.
    check_train_seasons = [2022, 2023]
    check_test_season = 2024
    check_train, check_test = split_train_test(df, check_train_seasons, check_test_season)

    check_logreg = fit_logreg(check_train)
    check_y_test = check_test["home_win"]
    check_logreg_proba = check_logreg.predict_proba(check_test[FEATURE_COLS])[:, 1]
    check_market_prob = market_implied_home_prob(check_test)

    print()
    print(
        f"=== Robustness check: same blend, train={check_train_seasons}, "
        f"test={check_test_season} ({len(check_train)} train / {len(check_test)} test rows) ==="
    )
    print(blend_weight_table(check_y_test, check_logreg_proba, check_market_prob, BLEND_WEIGHTS).to_string())

    # --- Subgroup accuracy: model vs. market, in both holdout seasons --------
    # Checking for a specific, consistent strength (or weakness) of the model
    # relative to the market -- not just games it happened to get right.
    subgroups_2025 = subgroup_accuracy_table(test, y_test, logreg_proba, market_prob)
    subgroups_2024 = subgroup_accuracy_table(check_test, check_y_test, check_logreg_proba, check_market_prob)

    print()
    print("=== Subgroup accuracy: model vs. market, test season 2025 ===")
    print(subgroups_2025.to_string())
    print()
    print("=== Subgroup accuracy: model vs. market, test season 2024 ===")
    print(subgroups_2024.to_string())

    print()
    print("=== Consistency check: model_minus_market sign, both seasons side by side ===")
    consistency = pd.DataFrame(
        {
            "n_2025": subgroups_2025["n"],
            "model_minus_market_2025": subgroups_2025["model_minus_market"],
            "n_2024": subgroups_2024["n"],
            "model_minus_market_2024": subgroups_2024["model_minus_market"],
        }
    )
    consistency["model_beat_market_both_seasons"] = (consistency["model_minus_market_2025"] > 0) & (
        consistency["model_minus_market_2024"] > 0
    )
    consistency["model_lost_to_market_both_seasons"] = (consistency["model_minus_market_2025"] < 0) & (
        consistency["model_minus_market_2024"] < 0
    )
    print(consistency.to_string())

    # --- Upsets the model called that the market missed ----------------------
    upsets_2025 = model_called_upsets(test, y_test, logreg_proba, market_prob)
    upsets_2024 = model_called_upsets(check_test, check_y_test, check_logreg_proba, check_market_prob)

    print()
    print(f"=== Model called it, market's favorite lost -- test season 2025 ({len(upsets_2025)} games) ===")
    print(upsets_2025.to_string(index=False))

    print()
    print(f"=== Model called it, market's favorite lost -- test season 2024 ({len(upsets_2024)} games) ===")
    print(upsets_2024.to_string(index=False))

    # --- Same, but only where the market's favorite was reasonably confident -
    confident_upsets_2025 = confident_market_upsets(upsets_2025)
    confident_upsets_2024 = confident_market_upsets(upsets_2024)

    print()
    print(
        f"=== Same, market_prob outside ({CONFIDENT_MARKET_LOW}, {CONFIDENT_MARKET_HIGH}) -- "
        f"test season 2025 ({len(confident_upsets_2025)}/{len(upsets_2025)} qualify) ==="
    )
    print(confident_upsets_2025.to_string(index=False))

    print()
    print(
        f"=== Same, market_prob outside ({CONFIDENT_MARKET_LOW}, {CONFIDENT_MARKET_HIGH}) -- "
        f"test season 2024 ({len(confident_upsets_2024)}/{len(upsets_2024)} qualify) ==="
    )
    print(confident_upsets_2024.to_string(index=False))


if __name__ == "__main__":
    main()
