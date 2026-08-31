"""Same 502 sort combinations as backtest_combos.py, but NO top-5-per-week
cap -- bet every single game that qualifies for a given combo (every sort in
the combo has a valid pick AND all agree on the same team), every week.

This isolates a different kind of selectivity than backtest_combos.py: there,
selectivity came from "only the top 5 highest-scoring qualifying games per
week." Here, selectivity comes purely from "only games where every sort in
the combo agrees at all" -- a combo of many sorts naturally bets fewer games
(agreement is rarer), a combo of few sorts bets close to every game (agreement
is closer to automatic). Comparing the two shows whether agreement-filtering
by itself captures the same edge as rank-filtering did.

Usage:
    python src/backtest_combos_all_games.py
"""

from itertools import combinations

import pandas as pd

from backtest_all_sorts import SORT_LABELS, load_games
from backtest_best_bet import STAKE, moneyline_to_payout
from backtest_combos import SEASON_CONFIGS, SORT_KEYS, build_sort_table

MIN_BETS_FOR_LEADERBOARD = 30


def backtest_combo_all_games(table, combo):
    """Bet every qualifying game -- no per-week cap. Returns (bets, wins, profit)."""
    score_cols = [f"{k}_z" for k in combo]
    side_cols = [f"{k}_side" for k in combo]

    valid = table[score_cols].notna().all(axis=1)
    sides = table[side_cols]
    agree = sides.eq(sides[side_cols[0]], axis=0).all(axis=1) & sides[side_cols[0]].notna()
    qualifying = table[valid & agree].copy()
    if qualifying.empty:
        return 0, 0, 0.0

    qualifying["side"] = qualifying[side_cols[0]]
    qualifying["moneyline"] = qualifying[[f"{k}_moneyline" for k in combo]].bfill(axis=1).iloc[:, 0]

    total_bets = total_wins = 0
    total_profit = 0.0
    for _, pick in qualifying.iterrows():
        b = moneyline_to_payout(pick["moneyline"])
        if b is None:
            continue
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
    for size in range(1, len(SORT_KEYS) + 1):
        all_combos.extend(combinations(SORT_KEYS, size))
    print(f"Testing {len(all_combos)} combos (including single sorts this time) across {len(SEASON_CONFIGS)} seasons, no top-N cap...")

    rows = []
    for combo in all_combos:
        total_bets = total_wins = 0
        total_profit = 0.0
        for label, table in tables.items():
            bets, wins, profit = backtest_combo_all_games(table, combo)
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
    print("=== Top 15 combos by TOTAL PROFIT, betting every qualifying game (no cap) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(result.sort_values("profit", ascending=False).head(15).to_string(index=False))

    print()
    print(f"=== Top 15 combos by ROI, minimum {MIN_BETS_FOR_LEADERBOARD} bets across both seasons ===")
    trustworthy = result[result["bets"] >= MIN_BETS_FOR_LEADERBOARD]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(trustworthy.sort_values("roi", ascending=False).head(15).to_string(index=False))

    print()
    print("=== Bottom 10 combos by ROI (min 30 bets) -- for contrast ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(trustworthy.sort_values("roi", ascending=True).head(10).to_string(index=False))

    print()
    print(f"Combos tested: {len(result)} (of {len(all_combos)} possible)")
    print("For comparison: top-5/week Combined (Model+Market) alone = 204 bets, +$964.63, +4.7% ROI")
    print("For comparison: bet-every-game, model favorite = 500 bets, -$618.85, -1.2% ROI")
    print("For comparison: bet-every-game, market favorite = 500 bets, -$546.99, -1.1% ROI")


if __name__ == "__main__":
    main()
