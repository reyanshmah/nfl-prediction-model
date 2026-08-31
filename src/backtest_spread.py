"""Same backtesting exercise as backtest_best_bet.py / backtest_all_sorts.py,
applied to the spread (against-the-spread / "points won or lost by") market
instead of moneyline. Real market odds (home_spread_odds / away_spread_odds)
exist in games_with_features.csv, so -- unlike player props -- this is a
fully legitimate, real-odds backtest, not an approximation.

Mechanics:
  - margin = home_score - away_score (positive = home won by that much).
  - spread_line is home-perspective, positive = home favored by that many
    points (verified in train_margin_model.py).
  - Home covers if margin > spread_line; away covers if margin < spread_line;
    exact equality is a push (void, excluded, same as a real sportsbook).
  - The margin model (same LinearRegression as train_margin_model.py) is
    fit ONLY on prior seasons, predicts margin blind for the test season.
  - Pick: bet home to cover if predicted margin > spread_line, else away.
  - edge = |predicted_margin - spread_line| stands in for "confidence" /
    the ranking metric for top-N-per-week selection (there's no separate
    "market probability" signal here the way moneyline had -- spread_line
    IS the market's number, so edge against it is the only signal).

Runs, same as the moneyline backtests: bet-every-game baseline, top-5/week,
and a pattern search for fade candidates -- across both 2025 and 2024 held
out seasons.

Usage:
    python src/backtest_spread.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import train_margin_model as margin_mod
from backtest_best_bet import STAKE

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

TOP_N = 5
MIN_SLICE_N = 15

SEASON_CONFIGS = [
    {"label": "2025", "train_seasons": [2022, 2023, 2024], "test_season": 2025},
    {"label": "2024", "train_seasons": [2021, 2022, 2023], "test_season": 2024},
]


def spread_payout(odds):
    """American odds -> profit per $1 staked, same convention as
    moneyline_to_payout but spread odds are near -110 either side, not
    strongly +/- like moneylines."""
    if pd.isna(odds):
        return None
    return odds / 100 if odds > 0 else 100 / abs(odds)


def load_predictions(train_seasons, test_season):
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=margin_mod.FEATURE_COLS + ["spread_line", "home_spread_odds", "away_spread_odds"]).reset_index(drop=True)
    df["margin"] = df["home_score"] - df["away_score"]

    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season].copy()

    model = LinearRegression()
    model.fit(train[margin_mod.FEATURE_COLS], train["margin"])
    test["model_margin"] = model.predict(test[margin_mod.FEATURE_COLS])

    test["edge"] = test["model_margin"] - test["spread_line"]
    test["side"] = np.where(test["edge"] > 0, "home", np.where(test["edge"] < 0, "away", "push"))
    test["bet_odds"] = np.where(test["side"] == "home", test["home_spread_odds"], test["away_spread_odds"])
    test["bet_payout"] = test["bet_odds"].apply(spread_payout)

    actual_cover = np.where(
        test["margin"] > test["spread_line"], "home",
        np.where(test["margin"] < test["spread_line"], "away", "push"),
    )
    test["actual_cover"] = actual_cover
    test["bet_won"] = (test["side"] != "push") & (test["actual_cover"] != "push") & (test["side"] == test["actual_cover"])
    test["is_push"] = (test["side"] == "push") | (test["actual_cover"] == "push")
    test["bet_profit"] = np.where(
        test["is_push"], 0.0, np.where(test["bet_won"], STAKE * test["bet_payout"], -STAKE)
    )
    return test


def settle_all(test_df):
    graded = test_df[~test_df["is_push"]]
    bets = len(graded)
    wins = int(graded["bet_won"].sum())
    profit = graded["bet_profit"].sum()
    return bets, wins, profit


def top_n_per_week(test_df, n=TOP_N):
    total_bets = total_wins = 0
    total_profit = 0.0
    for week, week_games in test_df.groupby("week"):
        candidates = week_games[~week_games["is_push"]].copy()
        candidates["abs_edge"] = candidates["edge"].abs()
        top = candidates.sort_values("abs_edge", ascending=False).head(n)
        total_bets += len(top)
        total_wins += int(top["bet_won"].sum())
        total_profit += top["bet_profit"].sum()
    return total_bets, total_wins, total_profit


def main() -> None:
    all_test = []
    print("=== Bet-every-game baseline & top-5/week, spread (ATS) market ===")
    combined_baseline = {"bets": 0, "wins": 0, "profit": 0.0}
    combined_top5 = {"bets": 0, "wins": 0, "profit": 0.0}
    for cfg in SEASON_CONFIGS:
        test_df = load_predictions(cfg["train_seasons"], cfg["test_season"])
        test_df["season_label"] = cfg["label"]
        all_test.append(test_df)

        bets, wins, profit = settle_all(test_df)
        staked = bets * STAKE
        print(f"{cfg['label']} bet-every-game: {bets} bets, {wins}-{bets-wins} "
              f"({wins/bets*100:.1f}%), profit ${profit:,.2f}, ROI {profit/staked*100:.1f}%")
        combined_baseline["bets"] += bets
        combined_baseline["wins"] += wins
        combined_baseline["profit"] += profit

        bets5, wins5, profit5 = top_n_per_week(test_df)
        staked5 = bets5 * STAKE
        print(f"{cfg['label']} top-5/week:      {bets5} bets, {wins5}-{bets5-wins5} "
              f"({wins5/bets5*100:.1f}%), profit ${profit5:,.2f}, ROI {profit5/staked5*100:.1f}%")
        combined_top5["bets"] += bets5
        combined_top5["wins"] += wins5
        combined_top5["profit"] += profit5

    print()
    print("=== Combined across 2024+2025 ===")
    for label, t in [("Bet every game", combined_baseline), ("Top 5/week", combined_top5)]:
        staked = t["bets"] * STAKE
        print(f"{label}: {t['bets']} bets, {t['wins']}-{t['bets']-t['wins']} "
              f"({t['wins']/t['bets']*100:.1f}%), profit ${t['profit']:,.2f}, ROI {t['profit']/staked*100:.1f}%")

    # --- Pattern search / fade check ----------------------------------------
    full = pd.concat(all_test, ignore_index=True)
    full = full[~full["is_push"]].copy()
    full["fade_odds"] = np.where(full["side"] == "home", full["away_spread_odds"], full["home_spread_odds"])
    full["fade_payout"] = full["fade_odds"].apply(spread_payout)
    full["fade_profit"] = np.where(full["bet_won"], -STAKE, STAKE * full["fade_payout"])

    slices = [
        ("Home covers picked", full["side"] == "home"),
        ("Away covers picked", full["side"] == "away"),
        ("Big edge (|edge| >= 3 pts)", full["edge"].abs() >= 3),
        ("Small edge (|edge| < 1.5 pts)", full["edge"].abs() < 1.5),
        ("Favorite predicted to cover (spread_line and edge same sign)",
         np.sign(full["spread_line"]) == np.sign(full["edge"])),
        ("Underdog predicted to cover (spread_line and edge opposite sign)",
         np.sign(full["spread_line"]) == -np.sign(full["edge"])),
        ("Divisional games", full["div_game"] == 1),
        ("Non-divisional games", full["div_game"] == 0),
        ("Dome games", full["is_dome"] == 1),
        ("Outdoor games", full["is_dome"] == 0),
        ("QB change either side", (full["home_qb_change"] == 1) | (full["away_qb_change"] == 1)),
        ("Large rest mismatch (>=3 days)", full["rest_advantage"].abs() >= 3),
        ("Big spread game (|spread_line| >= 7)", full["spread_line"].abs() >= 7),
        ("Close spread game (|spread_line| < 3)", full["spread_line"].abs() < 3),
    ]

    rows = []
    for label, mask in slices:
        sub = full[mask]
        n = len(sub)
        if n < MIN_SLICE_N:
            continue
        staked = n * STAKE
        as_picked_profit = sub["bet_profit"].sum()
        fade_profit = sub["fade_profit"].sum()
        rows.append(
            {
                "slice": label,
                "n": n,
                "as_picked_win_rate": round(sub["bet_won"].mean() * 100, 1),
                "as_picked_roi": round(as_picked_profit / staked * 100, 1),
                "faded_roi": round(fade_profit / staked * 100, 1),
            }
        )
    result = pd.DataFrame(rows).sort_values("as_picked_roi")
    print()
    print(f"=== Pattern search, spread market ({len(full)} graded bets, both seasons) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
