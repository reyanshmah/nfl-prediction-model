"""Two variations on backtest_best_bet.py's season backtest:

1. Threshold mode: instead of always betting the top 5 EV picks every week
   (even when none of them are great), only bet games whose (trust-
   discounted) EV clears a minimum bar. Swept across a few thresholds to see
   whether being more selective actually helps, or just bets less often on
   the same losing edge.
2. A second season (2024, train=[2022, 2023]) run through both the original
   top-5 strategy and the threshold strategy, to check whether 2025's result
   was a one-season fluke or a repeatable pattern.

Reuses every formula from backtest_best_bet.py untouched (same moneyline /
EV / disagreement-trust math, same Best Bet definition) -- only the bet
*selection* rule (top-N vs. threshold) and the train/test season pair change.

Usage:
    python src/backtest_variants.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_model as win_mod
from backtest_best_bet import STAKE, best_bet

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

THRESHOLDS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]
TOP_N = 5

SEASON_CONFIGS = [
    {"label": "2025 (train 2022-2024)", "train_seasons": [2022, 2023, 2024], "test_season": 2025},
    {"label": "2024 (train 2021-2023)", "train_seasons": [2021, 2022, 2023], "test_season": 2024},
]


def load_predictions(train_seasons, test_season):
    """Fit the win/loss model on train_seasons only, predict test_season
    blind (never trained on it), and compute every Best Bet field per game."""
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=win_mod.FEATURE_COLS).reset_index(drop=True)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season].copy()

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[win_mod.FEATURE_COLS], train["home_win"])

    test["model_win_prob"] = model.predict_proba(test[win_mod.FEATURE_COLS])[:, 1]
    test["market_win_prob"] = win_mod.market_implied_home_prob(test)

    picks_by_week = {}
    for week, week_games in test.groupby("week"):
        picks = []
        for _, row in week_games.iterrows():
            bet = best_bet(row)
            if bet is None:
                continue
            bet["week"] = week
            bet["home_won"] = bool(row["home_win"])
            picks.append(bet)
        picks_by_week[week] = picks
    return picks_by_week


def settle(bet):
    won = (bet["side"] == "home" and bet["home_won"]) or (bet["side"] == "away" and not bet["home_won"])
    profit = STAKE * bet["payout"] if won else -STAKE
    return won, profit


def run_top_n(picks_by_week, n=TOP_N):
    total_bets = total_wins = 0
    total_profit = 0.0
    for week, picks in picks_by_week.items():
        for bet in sorted(picks, key=lambda b: b["ev"], reverse=True)[:n]:
            won, profit = settle(bet)
            total_bets += 1
            total_wins += int(won)
            total_profit += profit
    return total_bets, total_wins, total_profit


def run_threshold(picks_by_week, threshold):
    total_bets = total_wins = 0
    total_profit = 0.0
    for week, picks in picks_by_week.items():
        for bet in picks:
            if bet["ev"] < threshold:
                continue
            won, profit = settle(bet)
            total_bets += 1
            total_wins += int(won)
            total_profit += profit
    return total_bets, total_wins, total_profit


def summarize(label, total_bets, total_wins, total_profit):
    staked = total_bets * STAKE
    win_rate = total_wins / total_bets if total_bets else float("nan")
    roi = total_profit / staked if staked else float("nan")
    return {
        "strategy": label,
        "bets": total_bets,
        "wins": total_wins,
        "losses": total_bets - total_wins,
        "win_rate": win_rate,
        "staked": staked,
        "profit": total_profit,
        "roi": roi,
    }


def main() -> None:
    all_rows = []
    for cfg in SEASON_CONFIGS:
        print(f"\n{'=' * 70}\nSeason: {cfg['label']}\n{'=' * 70}")
        picks_by_week = load_predictions(cfg["train_seasons"], cfg["test_season"])
        total_games = sum(len(v) for v in picks_by_week.values())
        print(f"Games with usable odds+predictions: {total_games}")

        # Top-5-per-week (original strategy), for reference.
        bets, wins, profit = run_top_n(picks_by_week, TOP_N)
        row = summarize(f"{cfg['label']}: top {TOP_N}/week", bets, wins, profit)
        all_rows.append(row)

        # Threshold sweep.
        for t in THRESHOLDS:
            bets, wins, profit = run_threshold(picks_by_week, t)
            row = summarize(f"{cfg['label']}: EV >= {t:.0%}", bets, wins, profit)
            all_rows.append(row)

    result = pd.DataFrame(all_rows)
    result["win_rate"] = (result["win_rate"] * 100).round(1)
    result["roi"] = (result["roi"] * 100).round(1)
    result["profit"] = result["profit"].round(2)

    print(f"\n\n{'=' * 70}\nAll strategies, side by side\n{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
