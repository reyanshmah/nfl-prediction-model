"""Pattern search: slice every "Combined (Model+Market)" pick from both held-
out seasons (2024, 2025 -- 500 games, no top-5 cap so every game is in the
sample) along a battery of dimensions, and for each slice compare:

  - "As picked": betting the side Combined actually favors on that slice.
  - "Faded": betting the OPPOSITE side instead, at that side's own price.

The user's own example ("11 of 15 times the market won when model and
market disagreed -- what if we always faded that") is one specific instance
of this general question: is there a real, systematic subgroup where the
pick is wrong more than half the time, such that betting the other way
would have been better? This checks that one plus a wide net of others,
using the exact same real games/outcomes/odds as every other backtest here.

IMPORTANT CAVEAT (stated again in the printed output): with ~20 slices
tested across one 500-game sample, finding one or two that look great after
the fact is expected by chance alone, even with zero real signal -- this is
the multiple-comparisons trap. A slice is only worth trusting if it's a
large sample, a big edge, AND a pattern with an actual reason behind it
(not just "week 6 was good"). None of this is re-validated on a third,
untouched season, so treat every number here as a lead, not a conclusion.

Usage:
    python src/backtest_pattern_search.py
"""

import pandas as pd

from backtest_all_sorts import SEASON_CONFIGS, compute_sorts_for_row, load_games
from backtest_best_bet import STAKE, market_confidence, moneyline_to_payout

MIN_SLICE_N = 20


def build_dataset():
    rows = []
    for cfg in SEASON_CONFIGS:
        test_df = load_games(cfg["train_seasons"], cfg["test_season"])
        for _, row in test_df.iterrows():
            sorts = compute_sorts_for_row(row)
            c = sorts.get("combined")
            if c is None:
                continue
            m = row["model_win_prob"]
            k = row["market_win_prob"]
            home_won = bool(row["home_win"])
            bet_side = c["side"]
            bet_ml = c["moneyline"]
            fade_ml = row["away_moneyline"] if bet_side == "home" else row["home_moneyline"]
            bet_won = (bet_side == "home" and home_won) or (bet_side == "away" and not home_won)

            b_bet = moneyline_to_payout(bet_ml)
            b_fade = moneyline_to_payout(fade_ml)

            rows.append(
                {
                    "season": cfg["label"],
                    "week": row["week"],
                    "game_type": row.get("game_type"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_side": bet_side,
                    "bet_moneyline": bet_ml,
                    "bet_won": bet_won,
                    "bet_profit": STAKE * b_bet if bet_won else -STAKE,
                    "fade_profit": STAKE * b_fade if not bet_won else -STAKE,
                    "model_prob": m,
                    "market_prob": k,
                    "model_conf": abs(m - 0.5) * 2,
                    "market_conf": market_confidence(k),
                    "disagreement": abs(m - k),
                    "agree_side": (m >= 0.5) == (k >= 0.5),
                    "bet_is_home": bet_side == "home",
                    "bet_is_favorite": bet_ml < 0,
                    "div_game": row.get("div_game") == 1,
                    "is_dome": row.get("is_dome") == 1,
                    "qb_change": (row.get("home_qb_change") == 1) or (row.get("away_qb_change") == 1),
                    "large_rest_diff": abs(row.get("rest_advantage", 0)) >= 3,
                    "starters_out_total": (row.get("home_starters_out") or 0) + (row.get("away_starters_out") or 0),
                    "spread_line": row.get("spread_line"),
                }
            )
    return pd.DataFrame(rows)


def slice_stats(df, mask, label):
    sub = df[mask]
    n = len(sub)
    if n < MIN_SLICE_N:
        return None
    bet_wins = sub["bet_won"].sum()
    bet_profit = sub["bet_profit"].sum()
    fade_profit = sub["fade_profit"].sum()
    staked = n * STAKE
    return {
        "slice": label,
        "n": n,
        "as_picked_win_rate": round(bet_wins / n * 100, 1),
        "as_picked_profit": round(bet_profit, 2),
        "as_picked_roi": round(bet_profit / staked * 100, 1),
        "faded_win_rate": round((n - bet_wins) / n * 100, 1),
        "faded_profit": round(fade_profit, 2),
        "faded_roi": round(fade_profit / staked * 100, 1),
    }


def build_slices(df):
    """Every slice to test: (label, boolean mask). A broad net across the
    dimensions this project already tracks per game, plus a few explicit
    replications of patterns raised earlier in this conversation."""
    slices = []

    # -- The user's literal example: model overrides a market with a real
    # lean the other way (the exact shape of the earlier 11/15 finding). --
    slices.append(("Model/market disagree on winner (any degree)", ~df["agree_side"]))
    slices.append(("Disagree + market lean >=5%", ~df["agree_side"] & (df["market_conf"] >= 0.05)))
    slices.append(("Disagree + market lean >=10%", ~df["agree_side"] & (df["market_conf"] >= 0.10)))
    slices.append(("Disagree + market lean >=15%", ~df["agree_side"] & (df["market_conf"] >= 0.15)))
    slices.append(("Agree on winner (both sources same side)", df["agree_side"]))

    # -- Market conviction buckets, regardless of agreement. --
    slices.append(("Market near coin flip (<10% conf)", df["market_conf"] < 0.10))
    slices.append(("Market modest lean (10-30% conf)", df["market_conf"].between(0.10, 0.30)))
    slices.append(("Market confident (30-60% conf)", df["market_conf"].between(0.30, 0.60)))
    slices.append(("Market very confident (>=60% conf)", df["market_conf"] >= 0.60))

    # -- Model conviction buckets. --
    slices.append(("Model near coin flip (<10% conf)", df["model_conf"] < 0.10))
    slices.append(("Model modest lean (10-30% conf)", df["model_conf"].between(0.10, 0.30)))
    slices.append(("Model confident (30-60% conf)", df["model_conf"].between(0.30, 0.60)))
    slices.append(("Model very confident (>=60% conf)", df["model_conf"] >= 0.60))

    # -- Bet-side shape. --
    slices.append(("Betting the home team", df["bet_is_home"]))
    slices.append(("Betting the away team", ~df["bet_is_home"]))
    slices.append(("Betting a favorite (negative moneyline)", df["bet_is_favorite"]))
    slices.append(("Betting an underdog (positive moneyline)", ~df["bet_is_favorite"]))
    slices.append(("Heavy favorite (moneyline <= -200)", df["bet_moneyline"] <= -200))
    slices.append(("Moderate favorite (-200 < ml <= -110)", df["bet_moneyline"].between(-200, -110, inclusive="left")))
    slices.append(("Near pick'em (-110 < ml < 110)", df["bet_moneyline"].between(-110, 110, inclusive="neither")))
    slices.append(("Underdog (ml >= 110)", df["bet_moneyline"] >= 110))

    # -- Game context. --
    slices.append(("Divisional games", df["div_game"]))
    slices.append(("Non-divisional games", ~df["div_game"]))
    slices.append(("Dome games", df["is_dome"]))
    slices.append(("Outdoor games", ~df["is_dome"]))
    slices.append(("Either team has a QB change", df["qb_change"]))
    slices.append(("No QB change either side", ~df["qb_change"]))
    slices.append(("Large rest mismatch (>=3 days)", df["large_rest_diff"]))
    slices.append(("Small/no rest mismatch", ~df["large_rest_diff"]))
    slices.append(("Any starters out (either team)", df["starters_out_total"] > 0))
    slices.append(("No starters out reported", df["starters_out_total"] == 0))
    slices.append(("Big spread game (|spread| >= 7)", df["spread_line"].abs() >= 7))
    slices.append(("Close spread game (|spread| < 3)", df["spread_line"].abs() < 3))

    # -- Season timing. --
    slices.append(("Weeks 1-6 (early season)", df["week"] <= 6))
    slices.append(("Weeks 7-12 (mid season)", df["week"].between(7, 12)))
    slices.append(("Weeks 13-18 (late regular season)", df["week"].between(13, 18)))
    slices.append(("Playoffs (week 19+)", df["week"] >= 19))

    return slices


def main() -> None:
    df = build_dataset()
    print(f"Base sample: {len(df)} games (2024+2025 combined, every game -- no top-N cap)")
    base_profit = df["bet_profit"].sum()
    base_roi = base_profit / (len(df) * STAKE) * 100
    print(f"Baseline (bet Combined's pick on every game): {df['bet_won'].sum()}/{len(df)} "
          f"({df['bet_won'].mean() * 100:.1f}%), profit ${base_profit:.2f}, ROI {base_roi:.1f}%")

    slices = build_slices(df)
    rows = []
    for label, mask in slices:
        stats = slice_stats(df, mask, label)
        if stats:
            rows.append(stats)

    result = pd.DataFrame(rows)

    print()
    print(f"=== All {len(result)} slices (min {MIN_SLICE_N} games), 'as picked' vs. 'faded' ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(result.sort_values("as_picked_roi").to_string(index=False))

    print()
    print("=== Where FADING would have beaten BETTING AS PICKED, sorted by the improvement ===")
    result["fade_improvement"] = result["faded_roi"] - result["as_picked_roi"]
    beats = result[result["fade_improvement"] > 0].sort_values("fade_improvement", ascending=False)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(beats.to_string(index=False))

    print()
    print("=== Worst 'as picked' slices (candidates worth fading) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(result.sort_values("as_picked_roi").head(8).to_string(index=False))

    print()
    print("CAVEAT: multiple-comparisons trap. ~30 slices tested on one 500-game sample --")
    print("expect a few to look great by chance alone even with zero real signal. A slice is")
    print("only worth trusting if n is large, the edge is large, AND there's an actual causal")
    print("story -- not just because it happened to work in this specific 2-season window.")
    print("None of this has been checked against a third, untouched season.")


if __name__ == "__main__":
    main()
