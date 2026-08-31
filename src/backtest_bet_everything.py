"""No top-5 cap, no threshold: bet $100 on literally every game, two ways --

  - "Model favorite": whichever side the model's own win probability favors,
    at that side's real moneyline.
  - "Market favorite": whichever side the betting market itself favors (i.e.
    just laying the actual Vegas favorite every single game) -- this is the
    "what if you had no model at all and just always bet the favorite"
    baseline, which should land close to break-even minus the vig.

Same held-out methodology as every other backtest in this project: the
win/loss model is trained ONLY on prior seasons and never sees the season
it's betting on. Run across both 2025 and 2024.

Usage:
    python src/backtest_bet_everything.py
"""

import pandas as pd

from backtest_all_sorts import SEASON_CONFIGS, compute_sorts_for_row, load_games
from backtest_best_bet import STAKE, moneyline_to_payout


def bet_every_game(test_df, sort_key, label):
    total_bets = total_wins = 0
    total_profit = 0.0
    rows = []
    for _, row in test_df.iterrows():
        sorts = compute_sorts_for_row(row)
        entry = sorts.get(sort_key)
        if entry is None:
            continue
        b = moneyline_to_payout(entry["moneyline"])
        if b is None:
            continue
        home_won = bool(row["home_win"])
        won = (entry["side"] == "home" and home_won) or (entry["side"] == "away" and not home_won)
        profit = STAKE * b if won else -STAKE
        total_bets += 1
        total_wins += int(won)
        total_profit += profit
        rows.append({"week": row["week"], "won": won, "profit": profit})
    return total_bets, total_wins, total_profit, pd.DataFrame(rows)


def main() -> None:
    strategies = [("confidence", "Bet the MODEL's favorite, every game"), ("market-confidence", "Bet the MARKET's favorite, every game")]

    combined_totals = {key: {"bets": 0, "wins": 0, "profit": 0.0} for key, _ in strategies}
    per_season_rows = []

    for cfg in SEASON_CONFIGS:
        test_df = load_games(cfg["train_seasons"], cfg["test_season"])
        for key, label in strategies:
            bets, wins, profit, _ = bet_every_game(test_df, key, label)
            staked = bets * STAKE
            per_season_rows.append(
                {
                    "season": cfg["label"],
                    "strategy": label,
                    "bets": bets,
                    "wins": wins,
                    "losses": bets - wins,
                    "win_rate": wins / bets * 100 if bets else float("nan"),
                    "staked": staked,
                    "profit": round(profit, 2),
                    "roi": profit / staked * 100 if staked else float("nan"),
                }
            )
            combined_totals[key]["bets"] += bets
            combined_totals[key]["wins"] += wins
            combined_totals[key]["profit"] += profit

    per_season_df = pd.DataFrame(per_season_rows)
    per_season_df["win_rate"] = per_season_df["win_rate"].round(1)
    per_season_df["roi"] = per_season_df["roi"].round(1)

    print("=== Bet EVERY game, no top-N cap, $100 flat stake ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(per_season_df.to_string(index=False))

    print()
    print("=== Combined across 2024+2025 ===")
    combined_rows = []
    for key, label in strategies:
        t = combined_totals[key]
        staked = t["bets"] * STAKE
        combined_rows.append(
            {
                "strategy": label,
                "bets": t["bets"],
                "wins": t["wins"],
                "losses": t["bets"] - t["wins"],
                "win_rate": round(t["wins"] / t["bets"] * 100, 1) if t["bets"] else float("nan"),
                "staked": staked,
                "profit": round(t["profit"], 2),
                "roi": round(t["profit"] / staked * 100, 1) if staked else float("nan"),
            }
        )
    combined_df = pd.DataFrame(combined_rows)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(combined_df.to_string(index=False))


if __name__ == "__main__":
    main()
