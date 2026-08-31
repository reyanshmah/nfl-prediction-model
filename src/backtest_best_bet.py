"""Backtest the dashboard's "Best Bet" strategy against the real, completed
2025 season.

Mirrors train_model.py's train/test split exactly: the win/loss model is
fit ONLY on seasons 2022-2024 and never sees any 2025 game (score, feature,
or outcome) during training -- 2025 is held out end-to-end, so every
prediction it produces is genuinely blind, the same way the live weekly
dashboard predicts an upcoming week it hasn't seen yet. It then reimplements
dashboard.html's combinedProbability() / disagreementTrust() / bestBet()
logic in Python (kept numerically identical to the JS -- see src/dashboard.html
for the source of truth on this formula) and simulates flat $100 bets on the
top 5 "Best Bet" picks each week of that season, settled against the real
final scores.

Usage:
    python src/backtest_best_bet.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_model as win_mod

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

TRAIN_SEASONS = [2022, 2023, 2024]
BACKTEST_SEASON = 2025
STAKE = 100.0
TOP_N_PER_WEEK = 5

# Same threshold as dashboard.html's disagreementTrust(): once the market's
# own confidence in the side it disagrees with the model on passes this,
# trust in overriding it is fully zeroed out.
DISAGREEMENT_TRUST_CUTOFF = 0.10


def moneyline_to_payout(ml):
    if pd.isna(ml):
        return None
    return ml / 100 if ml > 0 else 100 / abs(ml)


def expected_value(p, b):
    if p is None or b is None or pd.isna(p):
        return None
    return p * b - (1 - p)


def market_confidence(market_prob):
    return abs(market_prob - 0.5) * 2


def combined_probability(model_prob, market_prob, home_ml, away_ml, home_team, away_team):
    mconf = market_confidence(market_prob)
    model_weight = 0.5 * (1 - mconf)
    market_weight = 1 - model_weight
    combined = model_weight * model_prob + market_weight * market_prob
    side = "home" if combined >= 0.5 else "away"
    return {
        "side": side,
        "team": home_team if side == "home" else away_team,
        "confidence": combined if side == "home" else 1 - combined,
        "moneyline": home_ml if side == "home" else away_ml,
    }


def disagreement_trust(model_prob, market_prob):
    model_home = model_prob >= 0.5
    market_home = market_prob >= 0.5
    if model_home == market_home:
        return 1.0
    mconf = market_confidence(market_prob)
    return max(0.0, 1 - mconf / DISAGREEMENT_TRUST_CUTOFF)


def best_bet(row):
    if pd.isna(row["home_moneyline"]) or pd.isna(row["away_moneyline"]):
        return None
    c = combined_probability(
        row["model_win_prob"], row["market_win_prob"],
        row["home_moneyline"], row["away_moneyline"],
        row["home_team"], row["away_team"],
    )
    b = moneyline_to_payout(c["moneyline"])
    if b is None:
        return None
    raw_ev = expected_value(c["confidence"], b)
    trust = disagreement_trust(row["model_win_prob"], row["market_win_prob"])
    return {
        "team": c["team"],
        "side": c["side"],
        "moneyline": c["moneyline"],
        "win_prob": c["confidence"],
        "payout": b,
        "raw_ev": raw_ev,
        "ev": raw_ev * trust,
        "trust": trust,
    }


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=win_mod.FEATURE_COLS).reset_index(drop=True)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"] == BACKTEST_SEASON].copy()
    print(f"Train seasons {TRAIN_SEASONS}: {len(train)} games")
    print(f"Backtest season {BACKTEST_SEASON}: {len(test)} games (2025 never touches training)")

    # --- Fit the SAME win/loss model as train_model.py / predict_week.py,
    # blind to all of 2025. -------------------------------------------------
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[win_mod.FEATURE_COLS], train["home_win"])

    test["model_win_prob"] = model.predict_proba(test[win_mod.FEATURE_COLS])[:, 1]
    test["market_win_prob"] = win_mod.market_implied_home_prob(test)

    # --- Best Bet per game, weekly top-5, $100 flat stake -------------------
    weekly_rows = []
    bet_log = []
    for week, week_games in test.groupby("week"):
        picks = []
        for _, row in week_games.iterrows():
            bet = best_bet(row)
            if bet is None:
                continue
            bet["week"] = week
            bet["home_team"] = row["home_team"]
            bet["away_team"] = row["away_team"]
            bet["home_won"] = bool(row["home_win"])
            picks.append(bet)

        picks.sort(key=lambda b: b["ev"], reverse=True)
        top_picks = picks[:TOP_N_PER_WEEK]

        week_profit = 0.0
        week_wins = 0
        for bet in top_picks:
            won = (bet["side"] == "home" and bet["home_won"]) or (
                bet["side"] == "away" and not bet["home_won"]
            )
            profit = STAKE * bet["payout"] if won else -STAKE
            week_profit += profit
            week_wins += int(won)
            bet_log.append(
                {
                    "week": week,
                    "matchup": f"{bet['away_team']} @ {bet['home_team']}",
                    "bet": f"{bet['team']} ({bet['moneyline']:+.0f})",
                    "win_prob": bet["win_prob"],
                    "ev": bet["ev"],
                    "won": won,
                    "profit": profit,
                }
            )

        weekly_rows.append(
            {
                "week": week,
                "n_bets": len(top_picks),
                "wins": week_wins,
                "losses": len(top_picks) - week_wins,
                "staked": STAKE * len(top_picks),
                "profit": week_profit,
            }
        )

    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df["cumulative_profit"] = weekly_df["profit"].cumsum()

    bet_log_df = pd.DataFrame(bet_log)

    print()
    print(f"=== Every Best Bet pick, {BACKTEST_SEASON} (top {TOP_N_PER_WEEK}/week, ${STAKE:.0f} flat stake) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(bet_log_df.to_string(index=False))

    print()
    print(f"=== Weekly summary, {BACKTEST_SEASON} ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(weekly_df.to_string(index=False))

    total_staked = weekly_df["staked"].sum()
    total_profit = weekly_df["profit"].sum()
    total_bets = weekly_df["n_bets"].sum()
    total_wins = weekly_df["wins"].sum()

    print()
    print(f"=== {BACKTEST_SEASON} season total ===")
    print(f"Bets placed:     {total_bets}")
    print(f"Record:          {total_wins}-{total_bets - total_wins} ({total_wins / total_bets:.1%} win rate)")
    print(f"Total staked:    ${total_staked:,.2f}")
    print(f"Total profit:    ${total_profit:,.2f}")
    print(f"ROI on staked:   {total_profit / total_staked:.1%}")


if __name__ == "__main__":
    main()
