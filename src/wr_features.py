"""Build a WR receiving-yards prop dataset: one row per starting WR per game.

Mirrors rb_features.py's structure exactly, for receiving instead of rushing:
  1. Pull WR receiving stats via nfl_data_py.import_weekly_data() (falling
     back to import_ngs_data('receiving') for seasons not yet published
     there, e.g. the current season mid-year). NGS includes a week == 0 row
     per player holding *season totals*, not a game -- filtered out, or it
     would be treated as an extra, wildly inflated game in every rolling
     calculation.
  2. Pull snap counts via nfl_data_py.import_snap_counts() and keep only
     starters: offense_pct >= STARTER_SNAP_SHARE in that game.
  3. Add rolling, no-leakage features (shift-before-rolling):
     wr_rec_yards_last5, wr_receptions_last5, wr_targets_last5, and
     opponent_pass_defense_rank (the upcoming opponent's rank in pass yards
     allowed per game, over their own last 5 games -- reusing the same
     logic built for QBs in player_features.py, recomputed here from a
     fresh QB-stats fetch since each pipeline script is self-contained).
  4. Add target_share_last5: this player's share of their team's total
     targets over their last 5 games (sum of the player's own targets over
     the window, divided by the team's total targets over the same window
     -- a ratio of sums rather than an average of per-game ratios, which is
     more stable and standard for target-share). Team totals are summed
     across all pass-catchers (WR/TE/RB), not just starters -- except for
     seasons falling back to NGS (2025 as of writing), where NGS has no
     targets data for RBs at all, so the denominator (and thus
     target_share_last5) is a slight undercount there.
  5. Add wr_injury_status via nfl_data_py.import_injuries(), and join
     spread_line, total_line, is_dome, and a team-perspective rest_advantage
     in from data/processed/games_with_features.csv.
  6. Target: receiving_yards (the WR's actual yards in that game).

Usage:
    python src/wr_features.py
"""

import re
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEDULE_PATH = BASE_DIR / "data" / "raw" / "schedules.csv"
GAME_FEATURES_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "wr_receiving_features.csv"

STARTER_SNAP_SHARE = 0.60
ROLLING_WINDOW = 5
BAD_WR_STATUSES = {"Questionable", "Doubtful", "Out"}


def _normalize_name(name: str) -> str:
    """Normalize a player name for joining across data sources.

    Strips periods (e.g. "A.J. Brown" -> "AJ Brown") and trailing
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


def _wr_stats_from_weekly_data(season: int) -> pd.DataFrame:
    """All WR receiver-games for one season via nfl_data_py.import_weekly_data()."""
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[(weekly["position"] == "WR") & (weekly["targets"] > 0)].copy()
    # weekly_data already has an abbreviated "player_name" column (e.g.
    # "J.Jefferson") -- drop it before renaming player_display_name to
    # avoid a duplicate-column collision.
    weekly = weekly.drop(columns=["player_name"]).rename(
        columns={"player_id": "wr_id", "player_display_name": "player_name", "recent_team": "team"}
    )
    return weekly[["wr_id", "player_name", "team", "season", "week", "receiving_yards", "receptions", "targets"]]


def _wr_stats_from_ngs_data(season: int) -> pd.DataFrame:
    """All WR receiver-games for one season via nfl_data_py.import_ngs_data().

    Fallback for seasons import_weekly_data() hasn't published yet. Handles
    the same Super Bowl week-23-vs-22 numbering quirk documented in
    features.py, and drops NGS's week == 0 season-total row per player.
    """
    ngs = nfl.import_ngs_data("receiving", [season])
    ngs = ngs[(ngs["player_position"] == "WR") & (ngs["targets"] > 0) & (ngs["week"] > 0)].copy()
    ngs.loc[(ngs["season_type"] == "POST") & (ngs["week"] == 23), "week"] = 22
    ngs = ngs.rename(
        columns={
            "player_gsis_id": "wr_id",
            "player_display_name": "player_name",
            "team_abbr": "team",
            "yards": "receiving_yards",
        }
    )
    # NGS occasionally reports receiving_yards as NaN on a 0-reception game
    # (e.g. targeted but never caught one) -- 0 receptions unambiguously
    # means 0 receiving yards, so fill rather than leave a stray null.
    ngs.loc[ngs["receptions"] == 0, "receiving_yards"] = ngs.loc[ngs["receptions"] == 0, "receiving_yards"].fillna(0)
    return ngs[["wr_id", "player_name", "team", "season", "week", "receiving_yards", "receptions", "targets"]]


def fetch_wr_game_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-WR-game receiving stats across seasons, preferring import_weekly_data()
    and falling back to import_ngs_data() when a season isn't published there."""
    frames = []
    for season in seasons:
        try:
            frames.append(_wr_stats_from_weekly_data(season))
            print(f"  {season}: WR receiving stats from import_weekly_data")
            continue
        except Exception as exc:  # noqa: BLE001 - try the fallback source
            print(f"  {season}: import_weekly_data unavailable ({exc}); trying import_ngs_data")

        try:
            frames.append(_wr_stats_from_ngs_data(season))
            print(f"  {season}: WR receiving stats from import_ngs_data (fallback)")
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch WR stats for {season} from either source: {exc}")

    if not frames:
        return pd.DataFrame(
            columns=["wr_id", "player_name", "team", "season", "week", "receiving_yards", "receptions", "targets"]
        )
    return pd.concat(frames, ignore_index=True)


def _all_pass_catchers_from_weekly_data(season: int) -> pd.DataFrame:
    """Every pass-catcher (WR/TE/RB) per game, for the team-total-targets
    denominator used by target_share_last5 -- not just starters, and not
    just WRs, since a WR's target share is relative to the whole passing
    offense."""
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[weekly["position"].isin(["WR", "TE", "RB"]) & (weekly["targets"] > 0)].copy()
    return weekly.rename(columns={"recent_team": "team"})[["team", "season", "week", "targets"]]


def _all_pass_catchers_from_ngs_data(season: int) -> pd.DataFrame:
    """Fallback for seasons import_weekly_data() lacks: targets from NGS's
    receiving table.

    NGS's 'receiving' stat type only covers WR/TE -- it has no RB rows at
    all, and NGS's 'rushing' table (which does have RBs) carries no targets
    column. So pass-catching RBs' targets are simply unavailable from NGS,
    meaning the team-total-targets denominator (and therefore
    target_share_last5) is a slight *undercount* for any season using this
    fallback (2025 as of writing) -- a real, if usually modest, limitation
    worth knowing about rather than silently ignoring.
    """
    ngs = nfl.import_ngs_data("receiving", [season])
    ngs = ngs[ngs["player_position"].isin(["WR", "TE"]) & (ngs["week"] > 0) & (ngs["targets"] > 0)].copy()
    return ngs.rename(columns={"team_abbr": "team"})[["team", "season", "week", "targets"]]


def fetch_team_total_targets(seasons: list[int]) -> pd.DataFrame:
    """(season, week, team) -> total targets across every pass-catcher that week."""
    frames = []
    for season in seasons:
        try:
            frames.append(_all_pass_catchers_from_weekly_data(season))
            continue
        except Exception:  # noqa: BLE001 - try the fallback source
            pass
        try:
            frames.append(_all_pass_catchers_from_ngs_data(season))
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch team target totals for {season}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "team_targets"])
    all_targets = pd.concat(frames, ignore_index=True)
    return all_targets.groupby(["season", "week", "team"])["targets"].sum().reset_index(
        name="team_targets"
    )


def _qb_stats_from_weekly_data(season: int) -> pd.DataFrame:
    """All QB passer-games for one season, for opponent_pass_defense_rank."""
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[(weekly["position"] == "QB") & (weekly["attempts"] > 0)].copy()
    weekly = weekly.drop(columns=["player_name"]).rename(
        columns={"player_id": "qb_id", "player_display_name": "player_name", "recent_team": "team"}
    )
    return weekly[["qb_id", "team", "season", "week", "passing_yards"]]


def _qb_stats_from_ngs_data(season: int) -> pd.DataFrame:
    """Fallback QB passer-games via NGS, for opponent_pass_defense_rank."""
    ngs = nfl.import_ngs_data("passing", [season])
    ngs = ngs[(ngs["player_position"] == "QB") & (ngs["attempts"] > 0) & (ngs["week"] > 0)].copy()
    ngs.loc[(ngs["season_type"] == "POST") & (ngs["week"] == 23), "week"] = 22
    ngs = ngs.rename(columns={"player_gsis_id": "qb_id", "team_abbr": "team", "pass_yards": "passing_yards"})
    return ngs[["qb_id", "team", "season", "week", "passing_yards"]]


def fetch_qb_game_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-QB-game passing yards across seasons, for opponent_pass_defense_rank
    -- same source/fallback pattern as player_features.py, recomputed here
    since each pipeline script is self-contained."""
    frames = []
    for season in seasons:
        try:
            frames.append(_qb_stats_from_weekly_data(season))
            continue
        except Exception:  # noqa: BLE001 - try the fallback source
            pass
        try:
            frames.append(_qb_stats_from_ngs_data(season))
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch QB stats for {season} from either source: {exc}")

    if not frames:
        return pd.DataFrame(columns=["qb_id", "team", "season", "week", "passing_yards"])
    return pd.concat(frames, ignore_index=True)


def fetch_starter_snap_shares(seasons: list[int]) -> pd.DataFrame:
    """Per-WR-game offense snap share via nfl_data_py.import_snap_counts()."""
    frames = []
    for season in seasons:
        try:
            snaps = nfl.import_snap_counts([season])
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch snap counts for {season}: {exc}")
            continue
        snaps = snaps[snaps["position"] == "WR"].copy()
        frames.append(snaps[["season", "week", "team", "player", "offense_pct"]])

    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "player", "offense_pct"])
    return pd.concat(frames, ignore_index=True)


def build_starters(wr_stats: pd.DataFrame, snap_shares: pd.DataFrame) -> pd.DataFrame:
    """Join WR stats to snap shares (via normalized name/team/week) and keep
    only rows meeting the starter threshold."""
    wr_stats = wr_stats.copy()
    snap_shares = snap_shares.copy()
    wr_stats["name_key"] = wr_stats["player_name"].map(_normalize_name)
    snap_shares["name_key"] = snap_shares["player"].map(_normalize_name)

    merged = wr_stats.merge(
        snap_shares[["season", "week", "team", "name_key", "offense_pct"]],
        on=["season", "week", "team", "name_key"],
        how="inner",
    )
    starters = merged[merged["offense_pct"] >= STARTER_SNAP_SHARE].copy()
    return starters.drop(columns=["name_key"])


def add_team_pass_defense_rank(starters: pd.DataFrame, qb_stats: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Add opponent_pass_defense_rank to each WR-game row.

    Team pass yards allowed per game = sum of all opposing QBs' passing
    yards that week (same logic as player_features.py's version for QBs).
    Ranked 1 = fewest pass yards allowed (best pass defense) using each
    team's own shift-before-rolling last-5-games average -- no leakage --
    then looked up for each WR's specific upcoming opponent that week.
    """
    team_week_opp = _team_week_opponents(schedule)

    yards_by_team_week = qb_stats.groupby(["season", "week", "team"])["passing_yards"].sum().reset_index()
    yards_by_team_week = yards_by_team_week.merge(team_week_opp, on=["season", "week", "team"], how="left")

    allowed = (
        yards_by_team_week.groupby(["season", "week", "opponent"])["passing_yards"]
        .sum()
        .reset_index()
        .rename(columns={"opponent": "team", "passing_yards": "pass_yards_allowed"})
    )
    allowed = allowed.sort_values(["team", "season", "week"]).reset_index(drop=True)

    grouped = allowed.groupby("team", group_keys=False)
    allowed["pass_yards_allowed_last5"] = grouped["pass_yards_allowed"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    allowed["pass_defense_rank"] = allowed.groupby(["season", "week"])["pass_yards_allowed_last5"].rank(
        method="min", ascending=True
    )

    starters = starters.merge(team_week_opp, on=["season", "week", "team"], how="left")
    starters = starters.merge(
        allowed[["season", "week", "team", "pass_defense_rank"]].rename(
            columns={"team": "opponent", "pass_defense_rank": "opponent_pass_defense_rank"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )
    return starters


def add_wr_rolling_features(starters: pd.DataFrame) -> pd.DataFrame:
    """Add wr_rec_yards_last5 / wr_receptions_last5 / wr_targets_last5,
    no-leakage shift-before-rolling over each WR's own prior starts."""
    starters = starters.sort_values(["wr_id", "season", "week"]).reset_index(drop=True)
    grouped = starters.groupby("wr_id", group_keys=False)

    starters["wr_rec_yards_last5"] = grouped["receiving_yards"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    starters["wr_receptions_last5"] = grouped["receptions"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    starters["wr_targets_last5"] = grouped["targets"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return starters


def add_target_share(starters: pd.DataFrame, team_total_targets: pd.DataFrame) -> pd.DataFrame:
    """Add target_share_last5: this player's share of their team's total
    targets over their last 5 games -- a ratio of sums (player's summed
    targets over the window / team's summed targets over the same window),
    which is more stable than averaging per-game ratios. Both sums use the
    same shift-before-rolling no-leakage pattern.
    """
    starters = starters.sort_values(["wr_id", "season", "week"]).reset_index(drop=True)
    grouped = starters.groupby("wr_id", group_keys=False)
    player_targets_sum_last5 = grouped["targets"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum()
    )

    team_total_targets = team_total_targets.sort_values(["team", "season", "week"]).reset_index(drop=True)
    team_grouped = team_total_targets.groupby("team", group_keys=False)
    team_total_targets["team_targets_sum_last5"] = team_grouped["team_targets"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).sum()
    )

    starters["player_targets_sum_last5"] = player_targets_sum_last5
    starters = starters.merge(
        team_total_targets[["season", "week", "team", "team_targets_sum_last5"]],
        on=["season", "week", "team"],
        how="left",
    )
    starters["target_share_last5"] = np.where(
        starters["team_targets_sum_last5"] > 0,
        starters["player_targets_sum_last5"] / starters["team_targets_sum_last5"],
        np.nan,
    )
    return starters.drop(columns=["player_targets_sum_last5", "team_targets_sum_last5"])


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


def add_wr_injury_status(starters: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add wr_injury_status: 1 if the starter's own report_status entering
    this game was Questionable/Doubtful/Out, else 0."""
    starters = starters.merge(
        injuries[["season", "week", "gsis_id", "report_status"]].rename(columns={"gsis_id": "wr_id"}),
        on=["season", "week", "wr_id"],
        how="left",
    )
    starters["wr_injury_status"] = starters["report_status"].isin(BAD_WR_STATUSES).astype(int)
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

    print("Fetching WR receiving stats...")
    wr_stats = fetch_wr_game_stats(seasons)

    print("Fetching snap counts...")
    snap_shares = fetch_starter_snap_shares(seasons)

    starters = build_starters(wr_stats, snap_shares)
    print(f"Starter WR-games (offense_pct >= {STARTER_SNAP_SHARE}): {len(starters)}")

    print("Fetching QB passing stats for opponent_pass_defense_rank...")
    qb_stats = fetch_qb_game_stats(seasons)
    starters = add_team_pass_defense_rank(starters, qb_stats, schedule)

    starters = add_wr_rolling_features(starters)

    print("Fetching team target totals for target_share_last5...")
    team_total_targets = fetch_team_total_targets(seasons)
    starters = add_target_share(starters, team_total_targets)

    print("Fetching injury reports...")
    injuries = fetch_injuries(seasons)
    starters = add_wr_injury_status(starters, injuries)

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
                "receiving_yards",
                "receptions",
                "targets",
                "wr_rec_yards_last5",
                "wr_receptions_last5",
                "wr_targets_last5",
                "opponent_pass_defense_rank",
                "target_share_last5",
                "wr_injury_status",
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

    known_wr = "Justin Jefferson"
    sample = starters[
        (starters["player_name"] == known_wr) & (starters["season"] == 2023) & (starters["week"].between(6, 15))
    ].sort_values("week")
    print(f"{known_wr}, 2023 weeks 6-15:")
    print(
        sample[
            [
                "season",
                "week",
                "team",
                "opponent",
                "receiving_yards",
                "receptions",
                "targets",
                "wr_rec_yards_last5",
                "wr_receptions_last5",
                "wr_targets_last5",
                "opponent_pass_defense_rank",
                "target_share_last5",
                "wr_injury_status",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
