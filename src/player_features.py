"""Build a QB passing-yards prop dataset: one row per starting QB per game.

Pipeline:
  1. Pull QB passing stats via nfl_data_py.import_weekly_data() (falling back
     to import_ngs_data('passing') for seasons not yet published there, e.g.
     the current season mid-year -- same pattern as features.py).
  2. Pull snap counts via nfl_data_py.import_snap_counts() and keep only
     starters: offense_pct >= STARTER_SNAP_SHARE in that game.
  3. Add rolling, no-leakage features (shift-before-rolling, as in
     features.py): qb_pass_yards_last5, qb_pass_attempts_last5, and
     opponent_pass_defense_rank (the upcoming opponent's rank in pass yards
     allowed per game, over their own last 5 games).
  4. Add availability features via nfl_data_py.import_injuries() and
     import_weekly_rosters(): qb_injury_status (is the starter himself
     banged up entering this game) and top_target_availability (is the
     team's presumed #1 WR/TE, based on recent receiving form, out hurt or
     traded away this week).
  5. Target: passing_yards (the QB's actual yards in that game).

Usage:
    python src/player_features.py
"""

import re
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEDULE_PATH = BASE_DIR / "data" / "raw" / "schedules.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "qb_passing_features.csv"

STARTER_SNAP_SHARE = 0.60
ROLLING_WINDOW = 5


def _normalize_name(name: str) -> str:
    """Normalize a player name for joining across data sources.

    Strips periods (e.g. "A.J. McCarron" -> "AJ McCarron") and trailing
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
    """(season, week, team) -> opponent, derived from the schedule file.

    Used as the single source of truth for "who did this team play" across
    every data source, rather than trusting each stats source's own opponent
    column (which isn't always present, e.g. NGS passing data).
    """
    home = schedule[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = schedule[["season", "week", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    return pd.concat([home, away], ignore_index=True)


def _qb_stats_from_weekly_data(season: int) -> pd.DataFrame:
    """All QB passer-games for one season via nfl_data_py.import_weekly_data()."""
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[(weekly["position"] == "QB") & (weekly["attempts"] > 0)].copy()
    # weekly_data already has an abbreviated "player_name" column (e.g.
    # "J.Allen") -- drop it before renaming player_display_name to avoid a
    # duplicate-column collision.
    weekly = weekly.drop(columns=["player_name"]).rename(
        columns={"player_id": "qb_id", "player_display_name": "player_name", "recent_team": "team"}
    )
    return weekly[["qb_id", "player_name", "team", "season", "week", "passing_yards", "attempts"]]


def _qb_stats_from_ngs_data(season: int) -> pd.DataFrame:
    """All QB passer-games for one season via nfl_data_py.import_ngs_data().

    Fallback for seasons import_weekly_data() hasn't published yet. Handles
    the same Super Bowl week-23-vs-22 numbering quirk documented in
    features.py.
    """
    ngs = nfl.import_ngs_data("passing", [season])
    # NGS includes a week == 0 row per player holding *season totals*, not a
    # game -- must be dropped or it's treated as an extra, wildly inflated game.
    ngs = ngs[(ngs["player_position"] == "QB") & (ngs["attempts"] > 0) & (ngs["week"] > 0)].copy()
    ngs.loc[(ngs["season_type"] == "POST") & (ngs["week"] == 23), "week"] = 22
    ngs = ngs.rename(
        columns={
            "player_gsis_id": "qb_id",
            "player_display_name": "player_name",
            "team_abbr": "team",
            "pass_yards": "passing_yards",
        }
    )
    return ngs[["qb_id", "player_name", "team", "season", "week", "passing_yards", "attempts"]]


def fetch_qb_game_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-QB-game passing stats across seasons, preferring import_weekly_data()
    and falling back to import_ngs_data() when a season isn't published there."""
    frames = []
    for season in seasons:
        try:
            frames.append(_qb_stats_from_weekly_data(season))
            print(f"  {season}: QB passing stats from import_weekly_data")
            continue
        except Exception as exc:  # noqa: BLE001 - try the fallback source
            print(f"  {season}: import_weekly_data unavailable ({exc}); trying import_ngs_data")

        try:
            frames.append(_qb_stats_from_ngs_data(season))
            print(f"  {season}: QB passing stats from import_ngs_data (fallback)")
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch QB stats for {season} from either source: {exc}")

    if not frames:
        return pd.DataFrame(columns=["qb_id", "player_name", "team", "season", "week", "passing_yards", "attempts"])

    return pd.concat(frames, ignore_index=True)


def fetch_starter_snap_shares(seasons: list[int]) -> pd.DataFrame:
    """Per-QB-game offense snap share via nfl_data_py.import_snap_counts()."""
    frames = []
    for season in seasons:
        try:
            snaps = nfl.import_snap_counts([season])
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch snap counts for {season}: {exc}")
            continue
        snaps = snaps[snaps["position"] == "QB"].copy()
        frames.append(snaps[["season", "week", "team", "player", "offense_pct"]])

    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "player", "offense_pct"])
    return pd.concat(frames, ignore_index=True)


def build_starters(qb_stats: pd.DataFrame, snap_shares: pd.DataFrame) -> pd.DataFrame:
    """Join QB stats to snap shares (via normalized name/team/week) and keep
    only rows meeting the starter threshold."""
    qb_stats = qb_stats.copy()
    snap_shares = snap_shares.copy()
    qb_stats["name_key"] = qb_stats["player_name"].map(_normalize_name)
    snap_shares["name_key"] = snap_shares["player"].map(_normalize_name)

    merged = qb_stats.merge(
        snap_shares[["season", "week", "team", "name_key", "offense_pct"]],
        on=["season", "week", "team", "name_key"],
        how="inner",
    )
    starters = merged[merged["offense_pct"] >= STARTER_SNAP_SHARE].copy()
    return starters.drop(columns=["name_key"])


def add_team_pass_defense_rank(starters: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Add opponent_pass_defense_rank to each QB-game row.

    Team pass yards allowed per game = sum of all opposing QBs' passing
    yards that week (from the same starters-eligible source data, but
    summed over every passer, not just the starter, so a backup's mop-up
    yardage still counts against the defense's total).

    Ranked 1 = fewest pass yards allowed (best defense) using each team's
    own shift-before-rolling last-5-games average -- no leakage, computed
    the same way as the SOS adjustment in features.py -- then looked up for
    each QB's specific upcoming opponent that week.
    """
    team_week_opp = _team_week_opponents(schedule)

    # Attribute every passer's yards that week to the opposing team's "allowed" total.
    yards_by_team_week = starters.groupby(["season", "week", "team"])["passing_yards"].sum().reset_index()
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

    # Rank teams against each other within each (season, week) snapshot --
    # 1 = fewest yards allowed entering that week (stingiest defense).
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


def add_qb_rolling_features(starters: pd.DataFrame) -> pd.DataFrame:
    """Add qb_pass_yards_last5 / qb_pass_attempts_last5, no-leakage
    shift-before-rolling over each QB's own prior starts."""
    starters = starters.sort_values(["qb_id", "season", "week"]).reset_index(drop=True)
    grouped = starters.groupby("qb_id", group_keys=False)

    starters["qb_pass_yards_last5"] = grouped["passing_yards"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    starters["qb_pass_attempts_last5"] = grouped["attempts"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return starters


BAD_QB_STATUSES = {"Questionable", "Doubtful", "Out"}
BAD_TARGET_STATUSES = {"Doubtful", "Out"}


def fetch_injuries(seasons: list[int]) -> pd.DataFrame:
    """Per-player-week injury report status via nfl_data_py.import_injuries().

    One row per (season, week, gsis_id) already (verified no Wed/Thu/Fri
    duplicate rows in this data -- it's the single, final pregame report),
    so a direct join reflects the QB's/receiver's status entering that game,
    never anything filed after it.
    """
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
    # NaN == NaN, so a null key here would cross-multiply against every
    # other null key on the other side of any merge using it.
    return result.dropna(subset=["gsis_id"])


def fetch_traded_this_week(seasons: list[int]) -> pd.DataFrame:
    """Per-player-week flag: did this player's roster team change from the
    prior week they appeared, via nfl_data_py.import_weekly_rosters()."""
    frames = []
    for season in seasons:
        try:
            frames.append(nfl.import_weekly_rosters([season]))
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch weekly rosters for {season}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "week", "traded_this_week"])

    rosters = pd.concat(frames, ignore_index=True)
    # A null player_id (e.g. undrafted/practice-squad players without a
    # gsis id) must never be used as a join key: pandas merge treats
    # NaN == NaN, so a null key here would cross-multiply against every
    # other null key on the other side of any merge using it.
    rosters = rosters.dropna(subset=["player_id"])
    rosters = rosters.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    rosters["prev_team"] = rosters.groupby("player_id")["team"].shift(1)
    rosters["traded_this_week"] = rosters["prev_team"].notna() & (rosters["prev_team"] != rosters["team"])
    return rosters[["player_id", "season", "week", "traded_this_week"]]


def add_qb_injury_status(starters: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add qb_injury_status: 1 if the starter's own report_status entering
    this game was Questionable/Doubtful/Out, else 0 (including no report at
    all, i.e. not on the injury list)."""
    starters = starters.merge(
        injuries[["season", "week", "gsis_id", "report_status"]].rename(columns={"gsis_id": "qb_id"}),
        on=["season", "week", "qb_id"],
        how="left",
    )
    starters["qb_injury_status"] = starters["report_status"].isin(BAD_QB_STATUSES).astype(int)
    return starters.drop(columns=["report_status"])


def _receiving_stats_from_weekly_data(season: int) -> pd.DataFrame:
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[weekly["position"].isin(["WR", "TE"])].copy()
    weekly = weekly.drop(columns=["player_name"]).rename(
        columns={"player_id": "receiver_id", "player_display_name": "player_name", "recent_team": "team"}
    )
    return weekly[["receiver_id", "player_name", "team", "season", "week", "receptions"]]


def _receiving_stats_from_ngs_data(season: int) -> pd.DataFrame:
    ngs = nfl.import_ngs_data("receiving", [season])
    # NGS includes a week == 0 row per player holding *season totals*, not a
    # game -- must be dropped or it's treated as an extra, wildly inflated game.
    ngs = ngs[ngs["player_position"].isin(["WR", "TE"]) & (ngs["week"] > 0)].copy()
    ngs.loc[(ngs["season_type"] == "POST") & (ngs["week"] == 23), "week"] = 22
    ngs = ngs.rename(
        columns={"player_gsis_id": "receiver_id", "player_display_name": "player_name", "team_abbr": "team"}
    )
    return ngs[["receiver_id", "player_name", "team", "season", "week", "receptions"]]


def fetch_receiving_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-WR/TE-game receiving stats, preferring import_weekly_data() and
    falling back to import_ngs_data('receiving') for unpublished seasons."""
    frames = []
    for season in seasons:
        try:
            frames.append(_receiving_stats_from_weekly_data(season))
            print(f"  {season}: receiving stats from import_weekly_data")
            continue
        except Exception as exc:  # noqa: BLE001 - try the fallback source
            print(f"  {season}: import_weekly_data unavailable ({exc}); trying import_ngs_data")

        try:
            frames.append(_receiving_stats_from_ngs_data(season))
            print(f"  {season}: receiving stats from import_ngs_data (fallback)")
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch receiving stats for {season} from either source: {exc}")

    if not frames:
        return pd.DataFrame(columns=["receiver_id", "player_name", "team", "season", "week", "receptions"])
    return pd.concat(frames, ignore_index=True)


def add_top_target_availability(
    starters: pd.DataFrame, seasons: list[int], injuries: pd.DataFrame
) -> pd.DataFrame:
    """Add top_target_availability to each QB-game row.

    1. Identify each team's top WR/TE target *for a given game* as whoever
       had the highest receptions_last5 (shift-before-rolling, so already
       using only games before that one) among players who played for that
       team that week.
    2. Shift that by one game per team, so "top target entering this game"
       reflects who led the leaderboard *last time this team played* --
       looking backward like this (rather than at this week's own roster)
       is what lets a traded-away or inactive player still surface as "the
       presumed top target" so the injury/trade check below has someone to
       check.
    3. Flag 1 if that specific player has an Out/Doubtful status this week
       (via import_injuries(), matched on gsis_id regardless of which team
       they're currently listed under) OR was traded away this week (via
       import_weekly_rosters() week-over-week team changes). NaN if the team
       has no prior receiving history yet to identify a top target from.
    """
    receiving = fetch_receiving_stats(seasons)
    receiving = receiving.sort_values(["receiver_id", "season", "week"]).reset_index(drop=True)
    receiving["receptions_last5"] = receiving.groupby("receiver_id", group_keys=False)["receptions"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    # Leader among that team's actual WR/TE participants, for each game they played.
    valid = receiving.dropna(subset=["receptions_last5"])
    leader_idx = valid.groupby(["season", "week", "team"])["receptions_last5"].idxmax()
    top_by_game = valid.loc[leader_idx, ["season", "week", "team", "receiver_id"]].reset_index(drop=True)

    # Shift by one team-game: "presumed top target" reflects the leaderboard
    # as of the team's last game, not this week's (possibly already-changed) roster.
    top_by_game = top_by_game.sort_values(["team", "season", "week"]).reset_index(drop=True)
    top_by_game["top_target_id"] = top_by_game.groupby("team")["receiver_id"].shift(1)
    top_by_game = top_by_game.drop(columns=["receiver_id"])

    traded = fetch_traded_this_week(seasons)

    top_by_game = top_by_game.merge(
        injuries[["season", "week", "gsis_id", "report_status"]].rename(columns={"gsis_id": "top_target_id"}),
        on=["season", "week", "top_target_id"],
        how="left",
    )
    top_by_game = top_by_game.merge(
        traded.rename(columns={"player_id": "top_target_id"}),
        on=["season", "week", "top_target_id"],
        how="left",
    )
    top_by_game["traded_this_week"] = top_by_game["traded_this_week"].fillna(False)

    injury_bad = top_by_game["report_status"].isin(BAD_TARGET_STATUSES)
    top_by_game["top_target_availability"] = np.where(
        top_by_game["top_target_id"].isna(),
        np.nan,
        (injury_bad | top_by_game["traded_this_week"]).astype(float),
    )

    starters = starters.merge(
        top_by_game[["season", "week", "team", "top_target_availability"]],
        on=["season", "week", "team"],
        how="left",
    )
    return starters


def main() -> None:
    schedule = pd.read_csv(SCHEDULE_PATH)
    seasons = sorted(schedule["season"].unique().tolist())

    print("Fetching QB passing stats...")
    qb_stats = fetch_qb_game_stats(seasons)

    print("Fetching snap counts...")
    snap_shares = fetch_starter_snap_shares(seasons)

    starters = build_starters(qb_stats, snap_shares)
    print(f"Starter QB-games (offense_pct >= {STARTER_SNAP_SHARE}): {len(starters)}")

    starters = add_team_pass_defense_rank(starters, schedule)
    starters = add_qb_rolling_features(starters)

    print("Fetching injury reports...")
    injuries = fetch_injuries(seasons)
    starters = add_qb_injury_status(starters, injuries)

    print("Fetching receiving stats and weekly rosters for top-target availability...")
    starters = add_top_target_availability(starters, seasons, injuries)

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
                "passing_yards",
                "attempts",
                "qb_pass_yards_last5",
                "qb_pass_attempts_last5",
                "opponent_pass_defense_rank",
                "qb_injury_status",
                "top_target_availability",
            ]
        ]
        .isnull()
        .sum()
        .to_string()
    )
    print()

    known_qb = "Josh Allen"
    sample = starters[
        (starters["player_name"] == known_qb) & (starters["season"] == 2023) & (starters["week"].between(6, 15))
    ].sort_values("week")
    print(f"{known_qb}, 2023 weeks 6-15:")
    print(
        sample[
            [
                "season",
                "week",
                "team",
                "opponent",
                "passing_yards",
                "attempts",
                "qb_pass_yards_last5",
                "qb_pass_attempts_last5",
                "opponent_pass_defense_rank",
                "qb_injury_status",
                "top_target_availability",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("qb_injury_status value counts:")
    print(starters["qb_injury_status"].value_counts(dropna=False).to_string())
    print()
    print("top_target_availability value counts:")
    print(starters["top_target_availability"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
