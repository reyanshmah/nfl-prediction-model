"""The real thing: join our out-of-sample 2025 player-yardage predictions
(trained only on 2022-2024, same as every other backtest here) against REAL
closing prop lines/odds/results pulled from the SportsGameOdds API
(data/processed/real_prop_odds_2025.csv, via fetch_real_prop_odds.py).

For every player-week where both exist: bet over if our predicted yards >
the real closing line, under if less, at the REAL closing odds for that
side. Settle against the REAL final result. Buckets bets into Most
Risky / Medium / Least Risky by TERCILE of the real market's own implied
win probability for the side we bet (not our synthetic dashboard tiers --
those were never real odds; this uses what the actual market charged).

Usage:
    python src/backtest_real_prop_odds.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_best_bet import STAKE, moneyline_to_payout
from backtest_player_props import POSITIONS, load_position
from fetch_real_prop_odds import normalize_name

BASE_DIR = Path(__file__).resolve().parent.parent

STAT_TO_POSITION = {"passing_yards": "QB", "rushing_yards": "RB", "receiving_yards": "WR"}
TOP_N = 5


def implied_prob(american_odds: float) -> float:
    return -american_odds / (-american_odds + 100) if american_odds < 0 else 100 / (american_odds + 100)


def build_joined(season: int, train_seasons: list[int]) -> pd.DataFrame:
    real_path = BASE_DIR / "data" / "processed" / f"real_prop_odds_{season}.csv"
    real = pd.read_csv(real_path)
    real = real.dropna(subset=["line", "over_odds", "under_odds", "actual"])

    model_frames = []
    for pos in POSITIONS:
        df = load_position(pos, train_seasons, season)
        if df.empty:
            continue
        df = df.copy()
        df["player_name_norm"] = df["player_name"].map(normalize_name)
        df["stat"] = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards"}[pos]
        model_frames.append(df[["week", "player_name_norm", "stat", "predicted", "actual", "team"]])
    model_df = pd.concat(model_frames, ignore_index=True)

    joined = real.merge(
        model_df, on=["week", "player_name_norm", "stat"], how="inner", suffixes=("_real", "_model")
    )
    return joined


def settle(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    df["side"] = np.where(df["predicted"] > df["line"], "over", np.where(df["predicted"] < df["line"], "under", "push"))
    df["bet_odds"] = np.where(df["side"] == "over", df["over_odds"], df["under_odds"])
    df["payout"] = df["bet_odds"].apply(moneyline_to_payout)
    df["actual_side"] = np.where(df["actual_real"] > df["line"], "over", np.where(df["actual_real"] < df["line"], "under", "push"))
    df["is_push"] = (df["side"] == "push") | (df["actual_side"] == "push")
    df["bet_won"] = (~df["is_push"]) & (df["side"] == df["actual_side"])
    df["bet_profit"] = np.where(df["is_push"], 0.0, np.where(df["bet_won"], STAKE * df["payout"], -STAKE))
    df["implied_prob"] = df["bet_odds"].apply(implied_prob)
    df["edge"] = (df["predicted"] - df["line"]).abs()
    return df


def report(df: pd.DataFrame, label: str) -> None:
    graded = df[~df["is_push"]]
    n = len(graded)
    if n == 0:
        print(f"{label}: no graded bets")
        return
    wins = int(graded["bet_won"].sum())
    profit = graded["bet_profit"].sum()
    staked = n * STAKE
    print(f"{label}: {n} bets, {wins}-{n-wins} ({wins/n*100:.1f}%), profit ${profit:,.2f}, ROI {profit/staked*100:.1f}%")


SEASONS = [
    {"label": "2025", "season": 2025, "train_seasons": [2022, 2023, 2024]},
    {"label": "2024", "season": 2024, "train_seasons": [2021, 2022, 2023]},
]


def top5_per_week(graded):
    total_n = total_wins = 0
    total_profit = 0.0
    for week, week_rows in graded.groupby("week"):
        top = week_rows.sort_values("edge", ascending=False).head(TOP_N)
        total_n += len(top)
        total_wins += int(top["bet_won"].sum())
        total_profit += top["bet_profit"].sum()
    return total_n, total_wins, total_profit


def main() -> None:
    all_graded = []
    for cfg in SEASONS:
        print(f"\n{'='*70}\nSeason {cfg['label']}\n{'='*70}")
        joined = build_joined(cfg["season"], cfg["train_seasons"])
        print(f"Matched {len(joined)} player-props (by position: {joined['stat'].value_counts().to_dict()})")

        df = settle(joined)
        graded = df[~df["is_push"]].copy()
        graded["season"] = cfg["label"]
        all_graded.append(graded)

        report(df, "Bet every real prop")

        print("\n-- By tier: TERCILE of the real market's own implied probability for our side --")
        graded["risk_tier"] = pd.qcut(
            graded["implied_prob"], 3, labels=["Most Risky", "Medium", "Least Risky"]
        )
        for tier in ["Least Risky", "Medium", "Most Risky"]:
            report(graded[graded["risk_tier"] == tier], tier)

        n, w, p = top5_per_week(graded)
        staked = n * STAKE
        print(f"\nTop 5/week by edge: {n} bets, {w}-{n-w} ({w/n*100:.1f}%), profit ${p:,.2f}, ROI {p/staked*100:.1f}%")

        print("\n-- By position --")
        for pos, stat in [("QB", "passing_yards"), ("RB", "rushing_yards"), ("WR", "receiving_yards")]:
            report(df[df["stat"] == stat], pos)

        print("\n-- Over vs Under --")
        report(df[df["side"] == "over"], "OVER")
        report(df[df["side"] == "under"], "UNDER")
        print(f"\nAvg real implied probability of our side: {graded['implied_prob'].mean()*100:.1f}%")

    print(f"\n\n{'='*70}\nCombined 2024+2025\n{'='*70}")
    combined = pd.concat(all_graded, ignore_index=True)
    report(combined, "Bet every real prop, both seasons")

    print("\n-- By tier (pooled tercile across both seasons) --")
    combined["risk_tier"] = pd.qcut(combined["implied_prob"], 3, labels=["Most Risky", "Medium", "Least Risky"])
    for tier in ["Least Risky", "Medium", "Most Risky"]:
        sub = combined[combined["risk_tier"] == tier]
        report(sub, tier)
        for season_label in ["2025", "2024"]:
            s = sub[sub["season"] == season_label]
            if len(s):
                report(s, f"    {season_label} only")


if __name__ == "__main__":
    main()
