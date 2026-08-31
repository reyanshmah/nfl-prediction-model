"""Backtest every one of dashboard.html's betting-relevant sorts (all except
"Date", which isn't a ranking by confidence/value at all) as a top-5-per-week,
$100-flat-stake strategy, and compare total profit -- same methodology as
backtest_best_bet.py, run across both the 2025 and 2024 held-out seasons so a
single lucky/unlucky season doesn't decide the "winner".

Each sort's implied bet (which team, at which price) mirrors what dashboard.html
actually shows/recommends for that sort:
  - Model Confidence   -> the team the model favors, at that team's own price
  - Market Confidence  -> the team the market favors, at that team's own price
  - Highest Agreement  -> the team BOTH sources favor (excluded if they don't)
  - Highest Disagreement -> the team the model favors (the sort itself doesn't
    recommend a side on the dashboard -- this is the natural default so it can
    be backtested at all; flagged as a caveat in the output)
  - Best Value (EV)    -> bestValueBet(): whichever side has the better EV
    using the model's OWN probability (can pick the underdog the model itself
    didn't favor -- the flaw "Best Bet" was built to fix)
  - Hidden Edge        -> the team the model favors, ranked by
    modelConfidence*(1-marketConfidence)
  - Smart Value        -> same team as Best Value (EV), ranked by
    ev*(1-marketConfidence)
  - Combined (Model+Market) -> the blended-probability favorite, ranked by
    that blended confidence (no EV/payout awareness)
  - Best Bet           -> combined-probability favorite, ranked by
    disagreement-trust-discounted EV (see backtest_best_bet.py)

Usage:
    python src/backtest_all_sorts.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_model as win_mod
from backtest_best_bet import (
    STAKE,
    best_bet,
    combined_probability,
    expected_value,
    market_confidence,
    moneyline_to_payout,
)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

TOP_N = 5

SEASON_CONFIGS = [
    {"label": "2025", "train_seasons": [2022, 2023, 2024], "test_season": 2025},
    {"label": "2024", "train_seasons": [2021, 2022, 2023], "test_season": 2024},
]


def model_confidence(model_prob):
    return abs(model_prob - 0.5) * 2


def compute_sorts_for_row(row):
    """Returns {sort_name: {"score": ..., "side": "home"/"away", "moneyline": ...}}
    for every sort that has a valid pick on this game."""
    m = row["model_win_prob"]
    k = row["market_win_prob"]
    home_ml = row["home_moneyline"]
    away_ml = row["away_moneyline"]
    home_team = row["home_team"]
    away_team = row["away_team"]
    out = {}

    have_m = pd.notna(m)
    have_k = pd.notna(k)
    have_odds = pd.notna(home_ml) and pd.notna(away_ml)

    if have_m and have_odds:
        side = "home" if m >= 0.5 else "away"
        ml = home_ml if side == "home" else away_ml
        out["confidence"] = {"score": model_confidence(m), "side": side, "moneyline": ml}

    if have_k and have_odds:
        side = "home" if k >= 0.5 else "away"
        ml = home_ml if side == "home" else away_ml
        out["market-confidence"] = {"score": market_confidence(k), "side": side, "moneyline": ml}

    if have_m and have_k and have_odds:
        model_home = m >= 0.5
        market_home = k >= 0.5
        if model_home == market_home:
            side = "home" if model_home else "away"
            ml = home_ml if side == "home" else away_ml
            model_c = m if model_home else 1 - m
            market_c = k if market_home else 1 - k
            out["agreement"] = {"score": min(model_c, market_c), "side": side, "moneyline": ml}

        side = "home" if m >= 0.5 else "away"
        ml = home_ml if side == "home" else away_ml
        out["disagreement"] = {"score": abs(m - k), "side": side, "moneyline": ml}

    if have_m and have_odds:
        b_home = moneyline_to_payout(home_ml)
        b_away = moneyline_to_payout(away_ml)
        ev_home = expected_value(m, b_home)
        ev_away = expected_value(1 - m, b_away)
        if ev_home >= ev_away:
            out["value"] = {"score": ev_home, "side": "home", "moneyline": home_ml}
        else:
            out["value"] = {"score": ev_away, "side": "away", "moneyline": away_ml}

    if have_m and have_k and have_odds:
        mc = model_confidence(m)
        kc = market_confidence(k)
        side = "home" if m >= 0.5 else "away"
        ml = home_ml if side == "home" else away_ml
        out["hidden-edge"] = {"score": mc * (1 - kc), "side": side, "moneyline": ml}

    if "value" in out and have_k:
        kc = market_confidence(k)
        out["smart-value"] = {
            "score": out["value"]["score"] * (1 - kc),
            "side": out["value"]["side"],
            "moneyline": out["value"]["moneyline"],
        }

    if have_m and have_k and have_odds:
        c = combined_probability(m, k, home_ml, away_ml, home_team, away_team)
        out["combined"] = {"score": c["confidence"], "side": c["side"], "moneyline": c["moneyline"]}

    bb = best_bet(row)
    if bb is not None:
        out["best-bet"] = {"score": bb["ev"], "side": bb["side"], "moneyline": bb["moneyline"]}

    return out


SORT_LABELS = {
    "confidence": "Model Confidence",
    "market-confidence": "Market Confidence",
    "agreement": "Highest Agreement",
    "disagreement": "Highest Disagreement*",
    "value": "Best Value (EV)",
    "hidden-edge": "Hidden Edge",
    "smart-value": "Smart Value",
    "combined": "Combined (Model+Market)",
    "best-bet": "Best Bet",
}


def load_games(train_seasons, test_season):
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=win_mod.FEATURE_COLS).reset_index(drop=True)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season].copy()

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[win_mod.FEATURE_COLS], train["home_win"])

    test["model_win_prob"] = model.predict_proba(test[win_mod.FEATURE_COLS])[:, 1]
    test["market_win_prob"] = win_mod.market_implied_home_prob(test)
    return test


def backtest_sort(test_df, sort_key):
    """Top-5-per-week, $100 flat stake, for a single sort key."""
    total_bets = total_wins = 0
    total_profit = 0.0
    for week, week_games in test_df.groupby("week"):
        picks = []
        for _, row in week_games.iterrows():
            sorts = compute_sorts_for_row(row)
            entry = sorts.get(sort_key)
            if entry is None:
                continue
            entry = dict(entry)
            entry["home_won"] = bool(row["home_win"])
            picks.append(entry)

        picks.sort(key=lambda p: p["score"], reverse=True)
        for pick in picks[:TOP_N]:
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
    per_season_rows = []
    combined_totals = {key: {"bets": 0, "wins": 0, "profit": 0.0} for key in SORT_LABELS}

    for cfg in SEASON_CONFIGS:
        test_df = load_games(cfg["train_seasons"], cfg["test_season"])
        for key, label in SORT_LABELS.items():
            bets, wins, profit = backtest_sort(test_df, key)
            staked = bets * STAKE
            per_season_rows.append(
                {
                    "season": cfg["label"],
                    "sort": label,
                    "bets": bets,
                    "wins": wins,
                    "losses": bets - wins,
                    "win_rate": (wins / bets * 100) if bets else float("nan"),
                    "staked": staked,
                    "profit": round(profit, 2),
                    "roi": (profit / staked * 100) if staked else float("nan"),
                }
            )
            combined_totals[key]["bets"] += bets
            combined_totals[key]["wins"] += wins
            combined_totals[key]["profit"] += profit

    per_season_df = pd.DataFrame(per_season_rows)
    per_season_df["win_rate"] = per_season_df["win_rate"].round(1)
    per_season_df["roi"] = per_season_df["roi"].round(1)

    print("=== Every sort, both seasons, top 5/week, $100 flat stake ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(per_season_df.to_string(index=False))

    combined_rows = []
    for key, label in SORT_LABELS.items():
        t = combined_totals[key]
        staked = t["bets"] * STAKE
        combined_rows.append(
            {
                "sort": label,
                "total_bets": t["bets"],
                "total_wins": t["wins"],
                "total_losses": t["bets"] - t["wins"],
                "win_rate": (t["wins"] / t["bets"] * 100) if t["bets"] else float("nan"),
                "total_staked": staked,
                "total_profit": round(t["profit"], 2),
                "roi": (t["profit"] / staked * 100) if staked else float("nan"),
            }
        )
    combined_df = pd.DataFrame(combined_rows).sort_values("total_profit", ascending=False)
    combined_df["win_rate"] = combined_df["win_rate"].round(1)
    combined_df["roi"] = combined_df["roi"].round(1)

    print()
    print("=== Combined across 2024+2025, ranked by total profit ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(combined_df.to_string(index=False))

    print()
    print("* Highest Disagreement doesn't recommend a side on the dashboard itself")
    print("  (it's a diagnostic sort) -- backtested here betting the model's own")
    print("  favored side, the natural default, so it can be compared at all.")


if __name__ == "__main__":
    main()
