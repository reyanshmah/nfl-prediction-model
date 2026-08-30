"""Build an RB rushing-yards prop dataset: one row per starting RB per game.

Mirrors player_features.py's QB pipeline exactly, for rushing instead of
passing:
  1. Pull RB rushing stats via nfl_data_py.import_weekly_data() (falling back
     to import_ngs_data('rushing') for seasons not yet published there, e.g.
     the current season mid-year). NGS includes a week == 0 row per player
     holding *season totals*, not a game -- filtered out, or it would be
     treated as an extra, wildly inflated game in every rolling calculation.
  2. Pull snap counts via nfl_data_py.import_snap_counts() and keep only
     starters: offense_pct >= STARTER_SNAP_SHARE in that game.
  3. Add rolling, no-leakage features (shift-before-rolling):
     rb_rush_yards_last5, rb_rush_attempts_last5, and
     opponent_rush_defense_rank (the upcoming opponent's rank in rush yards
     allowed per game, over their own last 5 games -- same SOS-style logic
     as opponent_pass_defense_rank in player_features.py).
  4. Add rb_injury_status via nfl_data_py.import_injuries(), and join
     spread_line, total_line, is_dome, and a team-perspective rest_advantage
     in from data/processed/games_with_features.csv.
  5. Target: rushing_yards (the RB's actual yards in that game).

top_target_availability has no RB-specific equivalent here (a starting RB
mostly *is* the backfield's workload, unlike a WR corps with a clear #2/#3);
team_injury context more broadly (e.g. an OL/WR out affecting run blocking or
box counts) is left for a later pass.

Usage:
    python src/rb_features.py
"""

import re
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEDULE_PATH = BASE_DIR / "data" / "raw" / "schedules.csv"
GAME_FEATURES_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rb_rushing_features.csv"

STARTER_SNAP_SHARE = 0.60
ROLLING_WINDOW = 5
BAD_RB_STATUSES = {"Questionable", "Doubtful", "Out"}


def _normalize_name(name: str) -> str:
    """Normalize a player name for joining across data sources.

    Strips periods (e.g. "A.J. Dillon" -> "AJ Dillon") and trailing
    generational suffixes (e.g. "Gardner Minshew II" -> "Gardner Minshew"),
    which otherwise cause a small number of join misses between
    weekly_data/NGS names and snap_counts names for the same player.
    """
    if not isinstance(name, str):
        return name
    name = name.replace(".", "")
    name = re.sub(r"\s+(Jr|Sr|II|III|IV|V)$", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def _team_week_opponents(schedule: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) -> opponent, derived from the schedule file."""
    home = schedule[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = schedule[["season", "week", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    return pd.concat([home, away], ignore_index=True)


def _rb_stats_from_weekly_data(season: int) -> pd.DataFrame:
    """All RB rusher-games for one season via nfl_data_py.import_weekly_data()."""
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[(weekly["position"] == "RB") & (weekly["carries"] > 0)].copy()
    # weekly_data already has an abbreviated "player_name" column (e.g.
    # "S.Barkley") and a passing "attempts" column -- drop both before
    # renaming, to avoid duplicate-column collisions.
    weekly = weekly.drop(columns=["player_name", "attempts"]).rename(
        columns={
            "player_id": "rb_id",
            "player_display_name": "player_name",
            "recent_team": "team",
            "carries": "attempts",
        }
    )
    return weekly[["rb_id", "player_name", "team", "season", "week", "rushing_yards", "attempts"]]


def _rb_stats_from_ngs_data(season: int) -> pd.DataFrame:
    """All RB rusher-games for one season via nfl_data_py.import_ngs_data().

    Fallback for seasons import_weekly_data() hasn't published yet. Handles
    the same Super Bowl week-23-vs-22 numbering quirk documented in
    features.py, and drops NGS's week == 0 season-total row per player.
    """
    ngs = nfl.import_ngs_data("rushing", [season])
    ngs = ngs[(ngs["player_position"] == "RB") & (ngs["rush_attempts"] > 0) & (ngs["week"] > 0)].copy()
    ngs.loc[(ngs["season_type"] == "POST") & (ngs["week"] == 23), "week"] = 22
    ngs = ngs.rename(
        columns={
            "player_gsis_id": "rb_id",
            "player_display_name": "player_name",
            "team_abbr": "team",
            "rush_yards": "rushing_yards",
            "rush_attempts": "attempts",
        }
    )
    return ngs[["rb_id", "player_name", "team", "season", "week", "rushing_yards", "attempts"]]


def fetch_rb_game_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-RB-game rushing stats across seasons, preferring import_weekly_data()
    and falling back to import_ngs_data() when a season isn't published there."""
    frames = []
    for season in seasons:
        try:
            frames.append(_rb_stats_from_weekly_data(season))
            print(f"  {season}: RB rushing stats from import_weekly_data")
            continue
        except Exception as exc:  # noqa: BLE001 - try the fallback source
            print(f"  {season}: import_weekly_data unavailable ({exc}); trying import_ngs_data")

        try:
            frames.append(_rb_stats_from_ngs_data(season))
            print(f"  {season}: RB rushing stats from import_ngs_data (fallback)")
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch RB stats for {season} from either source: {exc}")

    if not frames:
        return pd.DataFrame(columns=["rb_id", "player_name", "team", "season", "week", "rushing_yards", "attempts"])
    return pd.concat(frames, ignore_index=True)


def fetch_starter_snap_shares(seasons: list[int]) -> pd.DataFrame:
    """Per-RB-game offense snap share via nfl_data_py.import_snap_counts()."""
    frames = []
    for season in seasons:
        try:
            snaps = nfl.import_snap_counts([season])
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch snap counts for {season}: {exc}")
            continue
        snaps = snaps[snaps["position"] == "RB"].copy()
        frames.append(snaps[["season", "week", "team", "player", "offense_pct"]])

    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "player", "offense_pct"])
    return pd.concat(frames, ignore_index=True)


def build_starters(rb_stats: pd.DataFrame, snap_shares: pd.DataFrame) -> pd.DataFrame:
    """Join RB stats to snap shares (via normalized name/team/week) and keep
    only rows meeting the starter threshold."""
    rb_stats = rb_stats.copy()
    snap_shares = snap_shares.copy()
    rb_stats["name_key"] = rb_stats["player_name"].map(_normalize_name)
    snap_shares["name_key"] = snap_shares["player"].map(_normalize_name)

    merged = rb_stats.merge(
        snap_shares[["season", "week", "team", "name_key", "offense_pct"]],
        on=["season", "week", "team", "name_key"],
        how="inner",
    )
    starters = merged[merged["offense_pct"] >= STARTER_SNAP_SHARE].copy()
    return starters.drop(columns=["name_key"])


def add_team_rush_defense_rank(starters: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Add opponent_rush_defense_rank to each RB-game row.

    Team rush yards allowed per game = sum of all opposing RBs' rushing
    yards that week (from the same starters-eligible source data, but
    summed over every rusher, not just the starter, so a backup's mop-up
    yardage still counts against the defense's total).

    Ranked 1 = fewest rush yards allowed (best run defense) using each
    team's own shift-before-rolling last-5-games average -- no leakage --
    then looked up for each RB's specific upcoming opponent that week.
    """
    team_week_opp = _team_week_opponents(schedule)

    yards_by_team_week = starters.groupby(["season", "week", "team"])["rushing_yards"].sum().reset_index()
    yards_by_team_week = yards_by_team_week.merge(team_week_opp, on=["season", "week", "team"], how="left")

    allowed = (
        yards_by_team_week.groupby(["season", "week", "opponent"])["rushing_yards"]
        .sum()
        .reset_index()
        .rename(columns={"opponent": "team", "rushing_yards": "rush_yards_allowed"})
    )
    allowed = allowed.sort_values(["team", "season", "week"]).reset_index(drop=True)

    grouped = allowed.groupby("team", group_keys=False)
    allowed["rush_yards_allowed_last5"] = grouped["rush_yards_allowed"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    allowed["rush_defense_rank"] = allowed.groupby(["season", "week"])["rush_yards_allowed_last5"].rank(
        method="min", ascending=True
    )

    starters = starters.merge(team_week_opp, on=["season", "week", "team"], how="left")
    starters = starters.merge(
        allowed[["season", "week", "team", "rush_defense_rank"]].rename(
            columns={"team": "opponent", "rush_defense_rank": "opponent_rush_defense_rank"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )
    return starters


def add_rb_rolling_features(starters: pd.DataFrame) -> pd.DataFrame:
    """Add rb_rush_yards_last5 / rb_rush_attempts_last5, no-leakage
    shift-before-rolling over each RB's own prior starts."""
    starters = starters.sort_values(["rb_id", "season", "week"]).reset_index(drop=True)
    grouped = starters.groupby("rb_id", group_keys=False)

    starters["rb_rush_yards_last5"] = grouped["rushing_yards"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    starters["rb_rush_attempts_last5"] = grouped["attempts"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return starters


def fetch_injuries(seasons: list[int]) -> pd.DataFrame:
    """Per-player-week injury report status via nfl_data_py.import_injuries()."""
    frames = []
    for season in seasons:
        try:
            inj = nfl.import_injuries([season])
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch injuries for {season}: {exc}")
            continue
        frames.append(inj[["season", "week", "team", "gsis_id", "report_status"]])

    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "report_status"])
    result = pd.concat(frames, ignore_index=True)
    # A null gsis_id must never be used as a join key: pandas merge treats
    # NaN == NaN, which would cross-multiply against every other null key.
    return result.dropna(subset=["gsis_id"])


def add_rb_injury_status(starters: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add rb_injury_status: 1 if the starter's own report_status entering
    this game was Questionable/Doubtful/Out, else 0."""
    starters = starters.merge(
        injuries[["season", "week", "gsis_id", "report_status"]].rename(columns={"gsis_id": "rb_id"}),
        on=["season", "week", "rb_id"],
        how="left",
    )
    starters["rb_injury_status"] = starters["report_status"].isin(BAD_RB_STATUSES).astype(int)
    return starters.drop(columns=["report_status"])


def add_game_context(starters: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Join is_dome, spread_line, total_line, and a team-perspective
    rest_advantage in from games_with_features.csv.

    spread_line and rest_advantage are home-perspective in that file
    (positive spread_line = home favored; rest_advantage = home_rest -
    away_rest) -- both get sign-flipped for the away team so they read as
    "this team's own" spread/rest edge.
    """
    home = games[
        ["season", "week", "home_team", "is_dome", "rest_advantage", "spread_line", "total_line"]
    ].rename(columns={"home_team": "team"})
    away = games[
        ["season", "week", "away_team", "is_dome", "rest_advantage", "spread_line", "total_line"]
    ].rename(columns={"away_team": "team"})
    away["rest_advantage"] = -away["rest_advantage"]
    away["spread_line"] = -away["spread_line"]
    context = pd.concat([home, away], ignore_index=True)

    return starters.merge(context, on=["season", "week", "team"], how="left")


def main() -> None:
    schedule = pd.read_csv(SCHEDULE_PATH)
    games = pd.read_csv(GAME_FEATURES_PATH)
    seasons = sorted(schedule["season"].unique().tolist())

    print("Fetching RB rushing stats...")
    rb_stats = fetch_rb_game_stats(seasons)

    print("Fetching snap counts...")
    snap_shares = fetch_starter_snap_shares(seasons)

    starters = build_starters(rb_stats, snap_shares)
    print(f"Starter RB-games (offense_pct >= {STARTER_SNAP_SHARE}): {len(starters)}")

    starters = add_team_rush_defense_rank(starters, schedule)
    starters = add_rb_rolling_features(starters)

    print("Fetching injury reports...")
    injuries = fetch_injuries(seasons)
    starters = add_rb_injury_status(starters, injuries)

    starters = add_game_context(starters, games)

    starters = starters.sort_values(["season", "week", "team"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    starters.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(starters)} rows to {OUTPUT_PATH}")
    print()
    print(f"Row count: {len(starters)}")
    print()
    print("Null counts:")
    print(
        starters[
            [
                "rushing_yards",
                "attempts",
                "rb_rush_yards_last5",
                "rb_rush_attempts_last5",
                "opponent_rush_defense_rank",
                "rb_injury_status",
                "is_dome",
                "rest_advantage",
                "spread_line",
                "total_line",
            ]
        ]
        .isnull()
        .sum()
        .to_string()
    )
    print()

    known_rb = "Saquon Barkley"
    sample = starters[
        (starters["player_name"] == known_rb) & (starters["season"] == 2024) & (starters["week"].between(6, 15))
    ].sort_values("week")
    print(f"{known_rb}, 2024 weeks 6-15:")
    print(
        sample[
            [
                "season",
                "week",
                "team",
                "opponent",
                "rushing_yards",
                "attempts",
                "rb_rush_yards_last5",
                "rb_rush_attempts_last5",
                "opponent_rush_defense_rank",
                "rb_injury_status",
                "is_dome",
                "rest_advantage",
                "spread_line",
                "total_line",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
