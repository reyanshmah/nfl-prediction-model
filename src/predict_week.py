"""Generate predictions for the current/upcoming NFL week.

Pipeline:
  1. Retrain the final win/loss logistic regression (15 features), margin
     linear regression, and QB/RB/WR yardage linear regressions on ALL
     available data (2022-2025) -- no held-out test set, since we're
     predicting into a season with no results yet.
  2. Pull the live schedule via nfl_data_py.import_schedules() and find the
     next unplayed week (min week with home_score still null).
  3. Compute each upcoming game's features by extending the real 2022-2025
     schedule with these future rows and re-running the same
     shift-before-rolling logic as features.py -- so "entering this game"
     values are each team's real trailing stats through their last played
     game, with no leakage (there's nothing to leak; these games haven't
     happened).
  4. For each team, identify the current starting QB/RB/WR1 via the same
     depth-chart approach as get_current_starter.py (latest snapshot,
     pos_rank == 1), with an injury-status cross-check when available.
  5. Look up each identified starter's own trailing rolling stats from the
     existing qb_passing_features.csv / rb_rushing_features.csv /
     wr_receiving_features.csv (their last-up-to-5 games' actual stats --
     mathematically the same shift-before-rolling formula used everywhere
     else, just extended one game past their last recorded one), and run
     the yardage models.
  6. Save data/processed/current_week_predictions.json and print it.

IMPORTANT, real limitations of predicting into an unplayed week (all
confirmed against the live data before writing this, not assumed):
  - The schedule's own home_qb_id/away_qb_id are only ever populated AFTER
    a game is played -- they're null for every upcoming game. features.py's
    normal home_qb_change/qb_rating_last5 logic depends on that field, so
    used as-is it would compute qb_change as spuriously true for every team
    (comparing "unknown" against history always reads as "different") and
    qb_rating_last5 as null for everyone. Both are overwritten using the
    depth-chart-identified starter instead (fix_qb_change_and_rating()).
  - nfl_data_py has no 2026 import_injuries() or import_snap_counts() data
    yet (both 404 -- the season hasn't started, so there ARE no injury
    reports or snap counts to fetch). This means:
      * home/away_starters_out cannot be computed for real and are set to
        0 for every team -- NOT a claim that nobody is hurt, just that the
        data to know either way doesn't exist yet.
      * The starter-identification injury fallback (get_current_starter.py's
        step 2) has nothing to check against, so every team's presumptive
        starter is whatever the latest depth chart says, unconditionally.
      * qb_injury_status / rb_injury_status / wr_injury_status all default
        to 0 for the same reason.
  - top_target_availability (a QB-model feature) needs the same season's
    receiving+roster data to compute; defaults to 0 for 2026 games.
  - spread_line / total_line come from whatever odds the schedule file
    already carries for that week (checked before writing this: fully
    populated for the next several weeks) -- if a future run targets a
    week odds haven't been posted for yet, those features will be null and
    that game's margin/win-prob predictions will be null too.

Usage:
    python src/predict_week.py
"""

import json
import re
import sys
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import features as feat  # noqa: E402
import get_current_starter as gcs  # noqa: E402
import train_margin_model as margin_mod  # noqa: E402
import train_model as win_mod  # noqa: E402
import train_player_model as qb_mod  # noqa: E402
import train_rb_model as rb_mod  # noqa: E402
import train_wr_model as wr_mod  # noqa: E402

GAME_FEATURES_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"
QB_FEATURES_PATH = BASE_DIR / "data" / "processed" / "qb_passing_features.csv"
RB_FEATURES_PATH = BASE_DIR / "data" / "processed" / "rb_rushing_features.csv"
WR_FEATURES_PATH = BASE_DIR / "data" / "processed" / "wr_receiving_features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "current_week_predictions.json"

ALL_SEASONS = [2022, 2023, 2024, 2025]
ROLLING_WINDOW = 5


# --------------------------------------------------------------------------
# Part 1: retrain everything on ALL available data (no test split).
# --------------------------------------------------------------------------


def retrain_win_model() -> tuple:
    df = win_mod.load_data()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(df[win_mod.FEATURE_COLS], df["home_win"])
    print(f"Win/loss model retrained on {len(df)} games (all seasons).")
    return model, win_mod.FEATURE_COLS


def retrain_margin_model() -> tuple:
    df = margin_mod.load_data()
    model = LinearRegression()
    model.fit(df[margin_mod.FEATURE_COLS], df[margin_mod.TARGET_COL])
    print(f"Margin model retrained on {len(df)} games (all seasons).")
    return model, margin_mod.FEATURE_COLS


def retrain_yardage_model(module) -> tuple:
    df = module.load_data()
    model = LinearRegression()
    model.fit(df[module.FEATURE_COLS], df[module.TARGET_COL])
    print(f"{module.TARGET_COL} model retrained on {len(df)} player-games (all seasons).")
    return model, module.FEATURE_COLS


# --------------------------------------------------------------------------
# Part 2: find the target week from the live schedule.
# --------------------------------------------------------------------------


def fetch_target_week_games(season: int) -> tuple[pd.DataFrame, int]:
    schedule = nfl.import_schedules([season])
    unplayed = schedule[schedule["home_score"].isnull()]
    if unplayed.empty:
        raise RuntimeError(f"No unplayed games found in the {season} schedule.")
    target_week = int(unplayed["week"].min())
    games = schedule[schedule["week"] == target_week].copy()
    return games, target_week


# --------------------------------------------------------------------------
# Part 3: team-level game features entering the target week, via the same
# shift-before-rolling logic as features.py, extended with the future rows.
# --------------------------------------------------------------------------


def build_extended_schedule(season: int) -> pd.DataFrame:
    """The real 2022-2025 schedule plus the target season's full schedule
    (played + upcoming), for feeding into features.py's rolling functions."""
    historical = pd.read_csv(BASE_DIR / "data" / "raw" / "schedules.csv")
    current = nfl.import_schedules([season])
    combined = pd.concat([historical, current], ignore_index=True)
    return combined


def compute_entering_game_features(combined_schedule: pd.DataFrame, all_seasons: list[int]) -> pd.DataFrame:
    """Run features.py's exact pipeline against the extended schedule, so
    the future weeks' rows get properly no-leakage "entering this game"
    values computed from real trailing history."""
    df = feat.add_dome_flag(combined_schedule)
    df = feat.add_rolling_team_features(df, all_seasons)

    print("Fetching QB stats for passer-rating feature (future seasons will warn/skip -- expected)...")
    df = feat.add_qb_rating_features(df, all_seasons)

    print("Fetching snap counts/injuries for team-injury features (future season will warn/skip -- expected)...")
    df = feat.add_team_injury_features(df, all_seasons, df)
    df["home_starters_out"] = df["home_starters_out"].fillna(0)
    df["away_starters_out"] = df["away_starters_out"].fillna(0)

    return df


# --------------------------------------------------------------------------
# Part 4: identify current starters via depth charts (get_current_starter.py
# logic, generalized to QB/RB/WR and to "right now" instead of a specific
# past game date).
# --------------------------------------------------------------------------


def fetch_depth_charts(season: int) -> pd.DataFrame:
    """Fetch once and reuse -- import_depth_charts() returns ~500K rows, far
    too expensive to re-fetch per team/position/prediction."""
    dc = nfl.import_depth_charts([season])
    dc["dt"] = pd.to_datetime(dc["dt"]).dt.tz_localize(None)
    return dc


def latest_depth_chart_starter(team: str, position: str, injuries: pd.DataFrame | None, depth_charts: pd.DataFrame) -> dict:
    """The presumptive starter at `position` for `team`, using the latest
    available depth chart snapshot, with an injury-status fallback to
    pos_rank == 2 when injury data is available and shows the #1 as
    Out/Doubtful."""
    dc = depth_charts[(depth_charts["team"] == team) & (depth_charts["pos_abb"] == position)]
    if dc.empty:
        return {"name": None, "note": "no depth chart data"}

    latest_dt = dc["dt"].max()
    snapshot = dc[dc["dt"] == latest_dt].sort_values("pos_rank")

    primary = snapshot[snapshot["pos_rank"] == 1]
    backup = snapshot[snapshot["pos_rank"] == 2]
    if primary.empty:
        return {"name": None, "gsis_id": None, "note": "no #1 on depth chart"}

    primary_name = primary.iloc[0]["player_name"]
    primary_gsis = primary.iloc[0]["gsis_id"]

    if injuries is None or injuries.empty:
        return {"name": primary_name, "gsis_id": primary_gsis, "note": "no injury data available to cross-check"}

    status_row = injuries[injuries["gsis_id"] == primary_gsis]
    status = status_row.iloc[0]["report_status"] if not status_row.empty else None
    if status in gcs.BAD_STATUSES and not backup.empty:
        return {
            "name": backup.iloc[0]["player_name"],
            "gsis_id": backup.iloc[0]["gsis_id"],
            "note": f"{primary_name} listed '{status}', used backup",
        }
    return {"name": primary_name, "gsis_id": primary_gsis, "note": "no fallback triggered"}


def try_fetch_injuries(season: int) -> pd.DataFrame:
    try:
        return nfl.import_injuries([season])
    except Exception as exc:  # noqa: BLE001
        print(f"  [note] import_injuries({season}) unavailable ({exc}) -- using depth chart #1 unconditionally.")
        return pd.DataFrame()


def recent_qb_ids_from_history(games_df: pd.DataFrame, team: str, window: int = feat.QB_HISTORY_WINDOW) -> list:
    """This team's last `window` *actually played* starting-QB ids, from the
    real historical schedule (home_qb_id/away_qb_id, which are only ever
    populated after a game is played) -- used to correctly evaluate
    qb_change against the depth-chart-identified starter for an upcoming
    game, since the schedule's own qb_id fields are null for it.
    """
    played = games_df[games_df["home_score"].notna()].copy()
    home = played[played["home_team"] == team][["season", "week", "home_qb_id"]].rename(
        columns={"home_qb_id": "qb_id"}
    )
    away = played[played["away_team"] == team][["season", "week", "away_qb_id"]].rename(
        columns={"away_qb_id": "qb_id"}
    )
    team_games = pd.concat([home, away], ignore_index=True).sort_values(["season", "week"])
    return team_games["qb_id"].dropna().tail(window).tolist()


def fix_qb_change_and_rating(
    target_rows: pd.DataFrame,
    starters: dict,
    historical_games: pd.DataFrame,
    qb_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Overwrite home/away_qb_change and home/away_qb_rating_last5, which
    features.py's normal pipeline cannot compute correctly here: it derives
    both from the schedule's own home_qb_id/away_qb_id, which are only
    populated after a game is played -- null for every upcoming game, which
    would otherwise make qb_change spuriously true for every team and
    qb_rating_last5 null for every team. Uses the depth-chart-identified
    starter (Part 4) instead.
    """
    target_rows = target_rows.copy()
    ratings_by_id = qb_ratings.groupby("qb_id")["passer_rating"].apply(
        lambda s: s.tail(ROLLING_WINDOW).mean()
    )

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        changes = []
        ratings = []
        for team in target_rows[team_col]:
            starter = starters[team]["QB"]
            history = recent_qb_ids_from_history(historical_games, team)
            if not history or starter["gsis_id"] is None:
                changes.append(np.nan)
            else:
                recent_starter = feat._most_common(history)
                changes.append(float(starter["gsis_id"] != recent_starter))
            ratings.append(ratings_by_id.get(starter["gsis_id"], np.nan))
        target_rows[f"{side}_qb_change"] = changes
        target_rows[f"{side}_qb_rating_last5"] = ratings

    return target_rows


# --------------------------------------------------------------------------
# Part 5: each identified starter's own trailing rolling stats, extended one
# game past their last recorded appearance -- same formula as the
# shift-before-rolling functions elsewhere, evaluated at "now".
# --------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Same normalization used by rb_features.py/wr_features.py/
    player_features.py: strips periods and generational suffixes (e.g.
    "Travis Etienne Jr." -> "travis etienne"), needed here because
    depth-chart names include suffixes that the historical per-game CSVs'
    names sometimes don't."""
    if not isinstance(name, str):
        return name
    name = name.replace(".", "")
    name = re.sub(r"\s+(Jr|Sr|II|III|IV|V)$", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def player_trailing_stats(features_csv: Path, player_name: str, stat_cols: list[str]) -> dict | None:
    df = pd.read_csv(features_csv)
    name_key = _normalize_name(player_name)
    rows = df[df["player_name"].map(_normalize_name) == name_key].sort_values(["season", "week"]).tail(ROLLING_WINDOW)
    if rows.empty:
        return None
    return {col: rows[col].mean() for col in stat_cols}


def compute_defense_ranks_entering(qb_df: pd.DataFrame, rb_df: pd.DataFrame) -> tuple[dict, dict]:
    """Each team's opponent_pass_defense_rank / opponent_rush_defense_rank
    entering their next game, derived by taking their most recent 5 games'
    "allowed" totals from the existing per-player feature files (which
    already tag each row with the opponent faced that week) and ranking
    across the league -- the same method as features.py's SOS-style rank
    features, evaluated one step past the last played game.
    """
    def rank_for(df: pd.DataFrame, group_col: str) -> dict:
        # df has one row per player-game with an 'opponent' column (the
        # defense that gave up this player's yards) and a yards column.
        allowed = df.groupby(["opponent", "season", "week"])[group_col].sum().reset_index()
        allowed = allowed.rename(columns={"opponent": "team"})
        trailing = allowed.sort_values(["team", "season", "week"]).groupby("team").tail(ROLLING_WINDOW)
        trailing_avg = trailing.groupby("team")[group_col].mean()
        ranks = trailing_avg.rank(method="min", ascending=True)
        return ranks.to_dict()

    pass_ranks = rank_for(qb_df, "passing_yards")
    rush_ranks = rank_for(rb_df, "rushing_yards")
    return pass_ranks, rush_ranks


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_qb_prediction(team: str, model, feature_cols, pass_def_ranks, game_row, side_prefix, starters) -> dict:
    starter = starters[team]["QB"]
    name = starter["name"]
    if name is None:
        return {"name": None, "predicted_yards": None, "note": starter["note"]}

    stats = player_trailing_stats(QB_FEATURES_PATH, name, ["passing_yards", "attempts"])
    if stats is None:
        return {"name": name, "predicted_yards": None, "note": "no trailing game history found for this player"}

    row = {
        "qb_pass_yards_last5": stats["passing_yards"],
        "qb_pass_attempts_last5": stats["attempts"],
        "opponent_pass_defense_rank": pass_def_ranks.get(team, np.nan),
        "is_dome": game_row["is_dome"],
        "rest_advantage": game_row[f"{side_prefix}_rest_relative"],
        "spread_line": game_row[f"{side_prefix}_spread_relative"],
        "total_line": game_row["total_line"],
        "qb_injury_status": 0,
        "top_target_availability": 0,
    }
    X = pd.DataFrame([row])[feature_cols]
    if X.isnull().any(axis=None):
        return {"name": name, "predicted_yards": None, "note": "missing feature(s), likely no opponent defense history"}
    pred = float(model.predict(X)[0])
    return {"name": name, "predicted_yards": round(pred, 1)}


def build_rb_prediction(team: str, model, feature_cols, rush_def_ranks, game_row, side_prefix, starters) -> dict:
    starter = starters[team]["RB"]
    name = starter["name"]
    if name is None:
        return {"name": None, "predicted_yards": None, "note": starter["note"]}

    stats = player_trailing_stats(RB_FEATURES_PATH, name, ["rushing_yards", "attempts"])
    if stats is None:
        return {"name": name, "predicted_yards": None, "note": "no trailing game history found for this player"}

    row = {
        "rb_rush_yards_last5": stats["rushing_yards"],
        "rb_rush_attempts_last5": stats["attempts"],
        "opponent_rush_defense_rank": rush_def_ranks.get(team, np.nan),
        "spread_line": game_row[f"{side_prefix}_spread_relative"],
        "total_line": game_row["total_line"],
        "is_dome": game_row["is_dome"],
        "rest_advantage": game_row[f"{side_prefix}_rest_relative"],
        "rb_injury_status": 0,
    }
    X = pd.DataFrame([row])[feature_cols]
    if X.isnull().any(axis=None):
        return {"name": name, "predicted_yards": None, "note": "missing feature(s), likely no opponent defense history"}
    pred = float(model.predict(X)[0])
    return {"name": name, "predicted_yards": round(pred, 1)}


def build_wr_prediction(team: str, model, feature_cols, pass_def_ranks, game_row, side_prefix, starters) -> dict:
    starter = starters[team]["WR"]
    name = starter["name"]
    if name is None:
        return {"name": None, "predicted_yards": None, "note": starter["note"]}

    stats = player_trailing_stats(WR_FEATURES_PATH, name, ["receiving_yards", "receptions", "targets"])
    if stats is None:
        return {"name": name, "predicted_yards": None, "note": "no trailing game history found for this player"}

    df = pd.read_csv(WR_FEATURES_PATH)
    name_key = _normalize_name(name)
    player_rows = df[df["player_name"].map(_normalize_name) == name_key].sort_values(["season", "week"]).tail(ROLLING_WINDOW)
    target_share = player_rows["target_share_last5"].mean() if not player_rows.empty else np.nan

    row = {
        "wr_rec_yards_last5": stats["receiving_yards"],
        "wr_receptions_last5": stats["receptions"],
        "wr_targets_last5": stats["targets"],
        "target_share_last5": target_share,
        "opponent_pass_defense_rank": pass_def_ranks.get(team, np.nan),
        "spread_line": game_row[f"{side_prefix}_spread_relative"],
        "total_line": game_row["total_line"],
        "is_dome": game_row["is_dome"],
        "rest_advantage": game_row[f"{side_prefix}_rest_relative"],
        "wr_injury_status": 0,
    }
    X = pd.DataFrame([row])[feature_cols]
    if X.isnull().any(axis=None):
        return {"name": name, "predicted_yards": None, "note": "missing feature(s), likely no opponent defense history"}
    pred = float(model.predict(X)[0])
    return {"name": name, "predicted_yards": round(pred, 1)}


def main() -> None:
    print("=== Retraining models on all available data (2022-2025) ===")
    win_model, win_cols = retrain_win_model()
    margin_model, margin_cols = retrain_margin_model()
    qb_model, qb_cols = retrain_yardage_model(qb_mod)
    rb_model, rb_cols = retrain_yardage_model(rb_mod)
    wr_model, wr_cols = retrain_yardage_model(wr_mod)

    print()
    print("=== Fetching current schedule ===")
    target_season = 2026
    upcoming_games, target_week = fetch_target_week_games(target_season)
    print(f"Target: season {target_season}, week {target_week} ({len(upcoming_games)} games)")
    teams = sorted(set(upcoming_games["home_team"]) | set(upcoming_games["away_team"]))

    print()
    print("=== Fetching injury data for starter cross-check (2026 -- expected to be unavailable) ===")
    injuries_2026 = try_fetch_injuries(target_season)

    print()
    print("=== Identifying current starters via depth charts ===")
    depth_charts = fetch_depth_charts(target_season)
    starters = {
        team: {
            pos: latest_depth_chart_starter(team, pos, injuries_2026, depth_charts)
            for pos in ["QB", "RB", "WR"]
        }
        for team in teams
    }
    for team in teams:
        print(f"  {team}: QB={starters[team]['QB']['name']}, RB={starters[team]['RB']['name']}, WR={starters[team]['WR']['name']}")

    print()
    print("=== Computing entering-game features ===")
    extended_schedule = build_extended_schedule(target_season)
    all_seasons = ALL_SEASONS + [target_season]
    featured = compute_entering_game_features(extended_schedule, all_seasons)
    target_rows = featured[(featured["season"] == target_season) & (featured["week"] == target_week)].copy()

    # home_qb_change/away_qb_change and home/away_qb_rating_last5 can't be
    # computed correctly by the normal pipeline here -- see
    # fix_qb_change_and_rating()'s docstring -- so overwrite them using the
    # depth-chart-identified starters instead.
    print("Fetching QB rating history for the identified starters...")
    qb_ratings = feat.fetch_qb_weekly_ratings(ALL_SEASONS)
    historical_games = pd.read_csv(BASE_DIR / "data" / "raw" / "schedules.csv")
    target_rows = fix_qb_change_and_rating(target_rows, starters, historical_games, qb_ratings)

    # Team-perspective spread/rest, matching train_model.py/train_player_model.py's convention.
    target_rows["home_rest_relative"] = target_rows["rest_advantage"]
    target_rows["away_rest_relative"] = -target_rows["rest_advantage"]
    target_rows["home_spread_relative"] = target_rows["spread_line"]
    target_rows["away_spread_relative"] = -target_rows["spread_line"]

    print()
    print("=== Computing opponent defense ranks entering this week ===")
    qb_df = pd.read_csv(QB_FEATURES_PATH)
    rb_df = pd.read_csv(RB_FEATURES_PATH)
    pass_def_ranks, rush_def_ranks = compute_defense_ranks_entering(qb_df, rb_df)

    print()
    print("=== Building predictions ===")
    records = []
    for _, game in target_rows.iterrows():
        home, away = game["home_team"], game["away_team"]

        X_win = pd.DataFrame([game[win_cols]])
        model_win_prob = float(win_model.predict_proba(X_win)[:, 1][0]) if not X_win.isnull().any(axis=None) else None

        X_margin = pd.DataFrame([game[margin_cols]])
        model_margin = float(margin_model.predict(X_margin)[0]) if not X_margin.isnull().any(axis=None) else None

        home_moneyline = game.get("home_moneyline")
        away_moneyline = game.get("away_moneyline")
        market_win_prob = None
        if pd.notna(home_moneyline) and pd.notna(away_moneyline):
            market_win_prob = float(win_mod.market_implied_home_prob(pd.DataFrame([game]))[0])
        market_spread = game.get("spread_line")

        record = {
            "home_team": home,
            "away_team": away,
            "week": int(game["week"]),
            "game_date": str(game["gameday"]),
            "model_win_prob": round(model_win_prob, 4) if model_win_prob is not None else None,
            "market_win_prob": round(market_win_prob, 4) if market_win_prob is not None else None,
            "model_margin": round(model_margin, 2) if model_margin is not None else None,
            "market_spread": float(market_spread) if pd.notna(market_spread) else None,
            # Raw American moneyline odds (not just the de-vigged probability
            # above) -- needed to compute a real payout/expected-value figure,
            # since the de-vigged probability alone can't be un-mixed back
            # into the actual price a sportsbook is offering.
            "home_moneyline": float(home_moneyline) if pd.notna(home_moneyline) else None,
            "away_moneyline": float(away_moneyline) if pd.notna(away_moneyline) else None,
            "home_starters_out": float(game["home_starters_out"]),
            "away_starters_out": float(game["away_starters_out"]),
            "home_qb": build_qb_prediction(home, qb_model, qb_cols, pass_def_ranks, game, "home", starters),
            "home_rb": build_rb_prediction(home, rb_model, rb_cols, rush_def_ranks, game, "home", starters),
            "home_wr": build_wr_prediction(home, wr_model, wr_cols, pass_def_ranks, game, "home", starters),
            "away_qb": build_qb_prediction(away, qb_model, qb_cols, pass_def_ranks, game, "away", starters),
            "away_rb": build_rb_prediction(away, rb_model, rb_cols, rush_def_ranks, game, "away", starters),
            "away_wr": build_wr_prediction(away, wr_model, wr_cols, pass_def_ranks, game, "away", starters),
        }
        records.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} game predictions to {OUTPUT_PATH}")
    print()
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
