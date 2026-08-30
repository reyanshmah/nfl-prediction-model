"""Train a binary "recorded a sack" classifier for edge pass rushers.

Loads data/processed/pass_rush_features.csv, filters to edge-position
players only (interior rushers have a much lower base rate and would just
dilute this), reframes the target as recorded_sack = 1 if def_sacks > 0 else
0, drops rows with null features, trains on seasons 2022-2024 and evaluates
on season 2025 (same season-based split as the other train_*.py scripts).

Reports accuracy for logistic regression and XGBoost against the "always
predict no sack" baseline -- with a ~70-72% no-sack base rate, that baseline
alone looks deceptively strong, so it's the real bar to beat, not 50%.

Usage:
    python src/train_pass_rush_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
PASS_RUSH_FEATURES_PATH = BASE_DIR / "data" / "processed" / "pass_rush_features.csv"

FEATURE_COLS = [
    "def_sacks_last5",
    "def_pressures_last5",
    "opponent_oline_sacks_allowed_last5",
    "spread_line",
    "total_line",
    "is_dome",
]
TARGET_COL = "recorded_sack"

TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PASS_RUSH_FEATURES_PATH)
    df = df[df["position_group"] == "edge"].copy()
    df[TARGET_COL] = (df["def_sacks"] > 0).astype(int)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"] == TEST_SEASON]
    return train, test


def main() -> None:
    df = load_data()
    print(f"Edge-position rows after dropping nulls in feature/target columns: {len(df)}")

    train, test = split_train_test(df)
    print(f"Train rows (seasons {TRAIN_SEASONS}): {len(train)}")
    print(f"Test rows (season {TEST_SEASON}): {len(test)}")

    base_rate = train[TARGET_COL].mean()
    print(f"Train recorded_sack rate: {base_rate:.4f} (i.e. no-sack rate: {1 - base_rate:.4f})")

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    # --- Baseline: always predict "no sack" ---------------------------------
    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    # --- Logistic regression -------------------------------------------------
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    logreg_cm = confusion_matrix(y_test, logreg_pred)

    logreg_clf = logreg.named_steps["logisticregression"]
    coefs = pd.DataFrame(
        {"feature": FEATURE_COLS, "coefficient": logreg_clf.coef_[0]}
    ).sort_values("coefficient", key=np.abs, ascending=False)

    print()
    print("=== Logistic Regression ===")
    print(f"Test accuracy: {logreg_accuracy:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted; [0,1] = [no_sack, sack]):")
    print(pd.DataFrame(logreg_cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_string())
    print()
    print("Coefficients (standardized features -> log-odds per 1 std dev):")
    print(coefs.to_string(index=False))

    # --- XGBoost ---------------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_cm = confusion_matrix(y_test, xgb_pred)

    importances = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": xgb.feature_importances_}
    ).sort_values("importance", ascending=False)

    print()
    print("=== XGBoost ===")
    print(f"Test accuracy: {xgb_accuracy:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted; [0,1] = [no_sack, sack]):")
    print(pd.DataFrame(xgb_cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_string())
    print()
    print("Feature importances (gain-based):")
    print(importances.to_string(index=False))

    # --- Comparison ------------------------------------------------------------
    print()
    print("=== Accuracy comparison, test season 2025 (edge pass rushers) ===")
    comparison = pd.DataFrame(
        {
            "accuracy": [baseline_accuracy, logreg_accuracy, xgb_accuracy],
            "beats_baseline_by": [
                0.0,
                logreg_accuracy - baseline_accuracy,
                xgb_accuracy - baseline_accuracy,
            ],
        },
        index=["always_predict_no_sack", "logistic_regression", "xgboost"],
    )
    print(comparison.to_string())


if __name__ == "__main__":
    main()
