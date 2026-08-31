"""Combine every possible subset (size 2+) of the 9 backtested sorts into a
composite ranking, and backtest each combo the same way as backtest_all_sorts.py
(top 5/week, $100 flat stake, both 2024 and 2025 held-out seasons), to see
whether combining sorts beats the best single sort (Combined (Model+Market),
+4.7% ROI).

How a combo works, per game per week:
  1. Every sort in the combo must have a valid pick on that game (has model
     prob / market prob / odds, whatever it individually needs).
  2. Every sort in the combo must agree on which TEAM to bet -- if even one
     sort in the combo favors the other side, the game is skipped for that
     combo. (This is why combos mixing "pick the favorite" sorts with
     "chase underdog value" sorts end up with very few qualifying games --
     they frequently disagree on the side by design.)
  3. Each sort's score is z-scored within that week (so a 0-1 probability
     and a raw EV number are on comparable footing before averaging) and the
     combo's composite score is the mean of those z-scores.
  4. Rank qualifying games by composite score, bet the top 5.

502 possible combos (2^9 - 9 singles - 1 empty). Reports the best ones by
total profit AND by ROI, with bet counts shown throughout -- a combo that
only ever finds 8 qualifying bets across two full seasons and happens to hit
90% isn't a strategy, it's a small sample, so a minimum bet count is used to
separate the "trustworthy" leaderboard from the raw one.

Usage:
    python src/backtest_combos.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_all_sorts import SORT_LABELS, compute_sorts_for_row, load_games
from backtest_best_bet import STAKE, moneyline_to_payout

MIN_BETS_FOR_LEADERBOARD = 30  # across both seasons combined
TOP_N = 5

SEASON_CONFIGS = [
    {"label": "2025", "train_seasons": [2022, 2023, 2024], "test_season": 2025},
    {"label": "2024", "train_seasons": [2021, 2022, 2023], "test_season": 2024},
]

SORT_KEYS = list(SORT_LABELS.keys())


def build_sort_table(test_df):
    """One row per game, with <key>_score / <key>_side / <key>_moneyline
    columns per sort (NaN/None where that sort has no valid pick), plus week
    and home_win."""
    records = []
    for _, row in test_df.iterrows():
        sorts = compute_sorts_for_row(row)
        rec = {"week": row["week"], "home_won": bool(row["home_win"])}
        for key in SORT_KEYS:
            entry = sorts.get(key)
            rec[f"{key}_score"] = entry["score"] if entry else np.nan
            rec[f"{key}_side"] = entry["side"] if entry else None
            rec[f"{key}_moneyline"] = entry["moneyline"] if entry else np.nan
        records.append(rec)
    table = pd.DataFrame(records)

    # Per-week z-score for each sort's raw score, so different sorts'
    # wildly different scales (0-1 probabilities vs. raw EV vs. products of
    # both) are comparable before averaging.
    for key in SORT_KEYS:
        col = f"{key}_score"
        table[f"{key}_z"] = table.groupby("week")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
        )
    return table


def backtest_combo(table, combo):
    """combo: tuple of sort keys, len >= 2. Returns (bets, wins, profit)."""
    score_cols = [f"{k}_z" for k in combo]
    side_cols = [f"{k}_side" for k in combo]

    valid = table[score_cols].notna().all(axis=1)
    sides = table[side_cols]
    agree = sides.eq(sides[side_cols[0]], axis=0).all(axis=1) & sides[side_cols[0]].notna()
    qualifying = table[valid & agree].copy()
    if qualifying.empty:
        return 0, 0, 0.0

    qualifying["composite"] = qualifying[score_cols].mean(axis=1)
    qualifying["side"] = qualifying[side_cols[0]]
    # moneyline is the same regardless of which sort in the combo it came
    # from, since it's determined purely by (game, side) -- take the first.
    qualifying["moneyline"] = qualifying[[f"{k}_moneyline" for k in combo]].bfill(axis=1).iloc[:, 0]

    total_bets = total_wins = 0
    total_profit = 0.0
    for week, week_games in qualifying.groupby("week"):
        top = week_games.sort_values("composite", ascending=False).head(TOP_N)
        for _, pick in top.iterrows():
            b = moneyline_to_payout(pick["moneyline"])
            won = (pick["side"] == "home" and pick["home_won"]) or (
                pick["side"] == "away" and not pick["home_won"]
            )
            profit = STAKE * b if won else -STAKE
            total_bets += 1
            total_wins += int(won)
            total_profit += profit

    return total_bets, total_wins, total_profit


def main() -> None:
    tables = {}
    for cfg in SEASON_CONFIGS:
        test_df = load_games(cfg["train_seasons"], cfg["test_season"])
        tables[cfg["label"]] = build_sort_table(test_df)

    all_combos = []
    for size in range(2, len(SORT_KEYS) + 1):
        all_combos.extend(combinations(SORT_KEYS, size))
    print(f"Testing {len(all_combos)} combos across {len(SEASON_CONFIGS)} seasons...")

    rows = []
    for combo in all_combos:
        total_bets = total_wins = 0
        total_profit = 0.0
        for label, table in tables.items():
            bets, wins, profit = backtest_combo(table, combo)
            total_bets += bets
            total_wins += wins
            total_profit += profit
        if total_bets == 0:
            continue
        staked = total_bets * STAKE
        rows.append(
            {
                "combo": " + ".join(SORT_LABELS[k] for k in combo),
                "n_sorts": len(combo),
                "bets": total_bets,
                "wins": total_wins,
                "losses": total_bets - total_wins,
                "win_rate": total_wins / total_bets * 100,
                "staked": staked,
                "profit": round(total_profit, 2),
                "roi": total_profit / staked * 100,
            }
        )

    result = pd.DataFrame(rows)
    result["win_rate"] = result["win_rate"].round(1)
    result["roi"] = result["roi"].round(1)

    print()
    print(f"=== Top 15 combos by TOTAL PROFIT (any bet count) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(result.sort_values("profit", ascending=False).head(15).to_string(index=False))

    print()
    print(f"=== Top 15 combos by ROI, minimum {MIN_BETS_FOR_LEADERBOARD} bets across both seasons ===")
    trustworthy = result[result["bets"] >= MIN_BETS_FOR_LEADERBOARD]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(trustworthy.sort_values("roi", ascending=False).head(15).to_string(index=False))

    print()
    print(f"=== Top 15 combos by ROI, ANY bet count (includes small-sample noise -- see bets column) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(result.sort_values("roi", ascending=False).head(15).to_string(index=False))

    print()
    print(f"Combos tested: {len(result)} (of {len(all_combos)} possible -- rest had 0 qualifying bets)")
    print(f"Best single sort for comparison: Combined (Model+Market), 204 bets, +$964.63, +4.7% ROI")


if __name__ == "__main__":
    main()
