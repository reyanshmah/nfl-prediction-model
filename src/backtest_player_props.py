"""Same exercise as backtest_spread.py, applied to player-yardage props
(QB passing, RB rushing, WR receiving).

IMPORTANT CAVEAT, stated up front because it changes what these numbers mean:
this project has NO real historical player-prop betting lines or odds --
nfl_data_py's schedule data only covers game-level markets (moneyline,
spread, total). train_player_model.py already documented this same gap and
used each player's own last-5-game rolling average as a stand-in "line" --
this script does the same, and additionally assumes standard -110/-110
odds on every prop (the typical real-world price for player yardage props,
even though the exact historical number isn't available). So unlike
backtest_spread.py and every moneyline backtest before it, THIS IS NOT A
REAL-ODDS BACKTEST -- it's the most honest approximation possible with the
data on hand, not a claim about real, tradeable profit.

Mechanics, per position (QB/RB/WR), mirroring train_player_model.py /
train_rb_model.py / train_wr_model.py exactly:
  - Model (LinearRegression, same features) fit ONLY on prior seasons,
    predicts yardage blind for the test season.
  - Line = that player's own last-5-game rolling average (already a model
    feature) -- a plausible number, not a real market line.
  - Bet over if predicted > line, under if predicted < line; push (excluded)
    if exactly equal.
  - Assumed -110/-110 odds both sides -> payout = 100/110 = 0.909 per $1.

Runs bet-every-prop baseline, top-5/week (ranked by |predicted - line|,
pooled across all 3 positions), and a pattern search -- across both 2025 and
2024 held-out seasons.

Usage:
    python src/backtest_player_props.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from backtest_best_bet import STAKE

BASE_DIR = Path(__file__).resolve().parent.parent
GAME_FEATURES_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

STANDARD_ODDS = -110
PAYOUT = 100 / 110  # same both sides

TOP_N = 5
MIN_SLICE_N = 15

SEASON_CONFIGS = [
    {"label": "2025", "train_seasons": [2022, 2023, 2024], "test_season": 2025},
    {"label": "2024", "train_seasons": [2021, 2022, 2023], "test_season": 2024},
]

POSITIONS = {
    "QB": {
        "path": BASE_DIR / "data" / "processed" / "qb_passing_features.csv",
        "features": [
            "qb_pass_yards_last5", "qb_pass_attempts_last5", "opponent_pass_defense_rank",
            "is_dome", "rest_advantage", "spread_line", "total_line",
            "qb_injury_status", "top_target_availability",
        ],
        "target": "passing_yards",
        "line_col": "qb_pass_yards_last5",
        "needs_context_merge": True,
    },
    "RB": {
        "path": BASE_DIR / "data" / "processed" / "rb_rushing_features.csv",
        "features": [
            "rb_rush_yards_last5", "rb_rush_attempts_last5", "opponent_rush_defense_rank",
            "spread_line", "total_line", "is_dome", "rest_advantage", "rb_injury_status",
        ],
        "target": "rushing_yards",
        "line_col": "rb_rush_yards_last5",
        "needs_context_merge": False,
    },
    "WR": {
        "path": BASE_DIR / "data" / "processed" / "wr_receiving_features.csv",
        "features": [
            "wr_rec_yards_last5", "wr_receptions_last5", "wr_targets_last5",
            "target_share_last5", "opponent_pass_defense_rank", "spread_line",
            "total_line", "is_dome", "rest_advantage", "wr_injury_status",
        ],
        "target": "receiving_yards",
        "line_col": "wr_rec_yards_last5",
        "needs_context_merge": False,
    },
}


def team_game_context(games: pd.DataFrame) -> pd.DataFrame:
    home = games[["season", "week", "home_team", "is_dome", "rest_advantage", "spread_line", "total_line"]].rename(
        columns={"home_team": "team"}
    )
    away = games[["season", "week", "away_team", "is_dome", "rest_advantage", "spread_line", "total_line"]].rename(
        columns={"away_team": "team"}
    )
    away["rest_advantage"] = -away["rest_advantage"]
    away["spread_line"] = -away["spread_line"]
    return pd.concat([home, away], ignore_index=True)


def load_position(position: str, train_seasons, test_season) -> pd.DataFrame:
    cfg = POSITIONS[position]
    # Some position feature files only have QB/RB/WR-specific columns to
    # start with; those need is_dome/rest_advantage/spread_line/total_line
    # merged in from the same team-game context as train_player_model.py.
    if cfg["needs_context_merge"] and "is_dome" not in pd.read_csv(cfg["path"], nrows=1).columns:
        df = pd.read_csv(cfg["path"])
        games = pd.read_csv(GAME_FEATURES_PATH)
        context = team_game_context(games)
        df = df.merge(context, on=["season", "week", "team"], how="left")
    else:
        df = pd.read_csv(cfg["path"])

    df = df.dropna(subset=cfg["features"] + [cfg["target"]]).reset_index(drop=True)

    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season].copy()
    if len(train) == 0 or len(test) == 0:
        return pd.DataFrame()

    model = LinearRegression()
    model.fit(train[cfg["features"]], train[cfg["target"]])
    test["predicted"] = model.predict(test[cfg["features"]])
    test["actual"] = test[cfg["target"]]
    test["line"] = test[cfg["line_col"]]
    test["edge"] = test["predicted"] - test["line"]
    test["side"] = np.where(test["edge"] > 0, "over", np.where(test["edge"] < 0, "under", "push"))
    actual_side = np.where(test["actual"] > test["line"], "over", np.where(test["actual"] < test["line"], "under", "push"))
    test["actual_side"] = actual_side
    test["is_push"] = (test["side"] == "push") | (test["actual_side"] == "push")
    test["bet_won"] = (~test["is_push"]) & (test["side"] == test["actual_side"])
    test["bet_profit"] = np.where(test["is_push"], 0.0, np.where(test["bet_won"], STAKE * PAYOUT, -STAKE))
    test["position"] = position
    return test


def settle_all(df):
    graded = df[~df["is_push"]]
    return len(graded), int(graded["bet_won"].sum()), graded["bet_profit"].sum()


def top_n_per_week(df, n=TOP_N):
    total_bets = total_wins = 0
    total_profit = 0.0
    for week, week_rows in df.groupby("week"):
        candidates = week_rows[~week_rows["is_push"]].copy()
        candidates["abs_edge"] = candidates["edge"].abs()
        top = candidates.sort_values("abs_edge", ascending=False).head(n)
        total_bets += len(top)
        total_wins += int(top["bet_won"].sum())
        total_profit += top["bet_profit"].sum()
    return total_bets, total_wins, total_profit


def main() -> None:
    print("=== Bet-every-prop baseline & top-5/week (pooled QB+RB+WR), assumed -110/-110 odds ===")
    all_rows = []
    combined_baseline = {"bets": 0, "wins": 0, "profit": 0.0}
    combined_top5 = {"bets": 0, "wins": 0, "profit": 0.0}

    for cfg in SEASON_CONFIGS:
        season_frames = []
        for pos in POSITIONS:
            pos_df = load_position(pos, cfg["train_seasons"], cfg["test_season"])
            if not pos_df.empty:
                season_frames.append(pos_df)
        season_df = pd.concat(season_frames, ignore_index=True)
        season_df["season_label"] = cfg["label"]
        all_rows.append(season_df)

        bets, wins, profit = settle_all(season_df)
        staked = bets * STAKE
        print(f"{cfg['label']} bet-every-prop: {bets} bets, {wins}-{bets-wins} "
              f"({wins/bets*100:.1f}%), profit ${profit:,.2f}, ROI {profit/staked*100:.1f}%")
        combined_baseline["bets"] += bets
        combined_baseline["wins"] += wins
        combined_baseline["profit"] += profit

        bets5, wins5, profit5 = top_n_per_week(season_df)
        staked5 = bets5 * STAKE
        print(f"{cfg['label']} top-5/week:     {bets5} bets, {wins5}-{bets5-wins5} "
              f"({wins5/bets5*100:.1f}%), profit ${profit5:,.2f}, ROI {profit5/staked5*100:.1f}%")
        combined_top5["bets"] += bets5
        combined_top5["wins"] += wins5
        combined_top5["profit"] += profit5

        # Per-position breakdown for this season
        for pos in POSITIONS:
            pos_df = season_df[season_df["position"] == pos]
            b, w, p = settle_all(pos_df)
            if b:
                print(f"    {cfg['label']} {pos} only: {b} bets, {w}-{b-w} ({w/b*100:.1f}%), "
                      f"profit ${p:,.2f}, ROI {p/(b*STAKE)*100:.1f}%")

    print()
    print("=== Combined across 2024+2025 ===")
    for label, t in [("Bet every prop", combined_baseline), ("Top 5/week", combined_top5)]:
        staked = t["bets"] * STAKE
        print(f"{label}: {t['bets']} bets, {t['wins']}-{t['bets']-t['wins']} "
              f"({t['wins']/t['bets']*100:.1f}%), profit ${t['profit']:,.2f}, ROI {t['profit']/staked*100:.1f}%")

    full = pd.concat(all_rows, ignore_index=True)
    full = full[~full["is_push"]].copy()
    full["fade_profit"] = np.where(full["bet_won"], -STAKE, STAKE * PAYOUT)

    slices = [
        ("QB props only", full["position"] == "QB"),
        ("RB props only", full["position"] == "RB"),
        ("WR props only", full["position"] == "WR"),
        ("Betting the OVER", full["side"] == "over"),
        ("Betting the UNDER", full["side"] == "under"),
        ("Big edge (|edge| >= 15 yards)", full["edge"].abs() >= 15),
        ("Small edge (|edge| < 5 yards)", full["edge"].abs() < 5),
        ("Dome games", full["is_dome"] == 1),
        ("Outdoor games", full["is_dome"] == 0),
        ("Player is favored team (spread_line > 0)", full["spread_line"] > 0),
        ("Player is underdog team (spread_line < 0)", full["spread_line"] < 0),
        ("High total game (total_line >= 47)", full["total_line"] >= 47),
        ("Low total game (total_line < 42)", full["total_line"] < 42),
        ("Large rest mismatch (>=3 days)", full["rest_advantage"].abs() >= 3),
        ("Weeks 1-6", full["week"] <= 6),
        ("Weeks 7-12", full["week"].between(7, 12)),
        ("Weeks 13-18", full["week"].between(13, 18)),
    ]

    rows = []
    for label, mask in slices:
        sub = full[mask]
        n = len(sub)
        if n < MIN_SLICE_N:
            continue
        staked = n * STAKE
        rows.append(
            {
                "slice": label,
                "n": n,
                "as_picked_win_rate": round(sub["bet_won"].mean() * 100, 1),
                "as_picked_roi": round(sub["bet_profit"].sum() / staked * 100, 1),
                "faded_roi": round(sub["fade_profit"].sum() / staked * 100, 1),
            }
        )
    result = pd.DataFrame(rows).sort_values("as_picked_roi")
    print()
    print(f"=== Pattern search, player props ({len(full)} graded bets, both seasons, assumed -110) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
