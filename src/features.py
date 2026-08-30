"""Build engineered features from the raw schedule data.

Adds:
  - is_dome: 1 when roof is 'dome' or 'closed', else 0.
  - Team-level rolling features (points scored/allowed over last 5 games, win
    streak) computed strictly from each team's *prior* games, plus rest_advantage.
  - QB continuity flags (home_qb_change / away_qb_change) that signal a team
    is starting someone other than its recent starter (injury/benching).
  - QB quality (home_qb_rating_last5 / away_qb_rating_last5): each starter's
    average NFL passer rating over their last 5 games started, pulled from
    nfl_data_py.import_weekly_data() (falling back to import_ngs_data() for
    seasons not yet published there, e.g. the current season mid-year).
  - Turnover margin (home_turnover_margin_last5 / away_turnover_margin_last5):
    each team's average (takeaways - giveaways) over its last 5 games,
    aggregated from nfl_data_py.import_pbp_data() play-by-play.
  - A strength-of-schedule-adjusted scoring feature (*_pts_scored_last5_sos_adj)
    that discounts/boosts pts_scored_last5 based on how good the defenses
    faced in those same 5 games actually were.

Usage:
    python src/features.py
"""

import re
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_data_py as nfl

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "schedules.csv"
DOME_FLAG_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_dome_flag.csv"
FEATURES_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"

ROLLING_WINDOW = 5
QB_HISTORY_WINDOW = 3
QB_RATING_WINDOW = 5


def add_dome_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add an is_dome column based on the roof column.

    is_dome = 1 when roof is 'dome' or 'closed', else 0 (including 'outdoors',
    'open', and any nulls). temp/wind are left as-is.
    """
    df = df.copy()
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    return df


def _build_team_game_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape game-level rows into one row per team per game (long format)."""
    gameday = pd.to_datetime(df["gameday"])

    home = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "gameday": gameday,
            "team": df["home_team"],
            "opponent": df["away_team"],
            "pts_scored": df["home_score"],
            "pts_allowed": df["away_score"],
            "qb_id": df["home_qb_id"],
            "side": "home",
        }
    )
    away = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "gameday": gameday,
            "team": df["away_team"],
            "opponent": df["home_team"],
            "pts_scored": df["away_score"],
            "pts_allowed": df["home_score"],
            "qb_id": df["away_qb_id"],
            "side": "away",
        }
    )

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["outcome"] = np.sign(long_df["pts_scored"] - long_df["pts_allowed"])  # 1/0/-1
    long_df = long_df.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    return long_df


def _add_prior_win_streak(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute each team's win streak entering each game, using only prior games.

    Positive = current win streak, negative = current losing streak, 0 = a tie
    was the most recent result, NaN = the team has no prior games yet.
    """
    long_df = long_df.copy()
    streak_before = np.full(len(long_df), np.nan)

    for _, idx in long_df.groupby("team").groups.items():
        idx = idx.to_numpy()
        running_streak = 0
        for i, row_pos in enumerate(idx):
            streak_before[row_pos] = running_streak if i > 0 else np.nan
            outcome = long_df.loc[row_pos, "outcome"]
            if outcome > 0:
                running_streak = running_streak + 1 if running_streak >= 0 else 1
            elif outcome < 0:
                running_streak = running_streak - 1 if running_streak <= 0 else -1
            else:
                running_streak = 0

    long_df["win_streak"] = streak_before
    return long_df


def _most_common(history: deque) -> str:
    """Mode of a small deque, breaking ties in favor of the most recent entry."""
    counts = Counter(history)
    best_count = max(counts.values())
    for qb in reversed(history):
        if counts[qb] == best_count:
            return qb
    return None  # unreachable, but keeps type-checkers happy


def _add_prior_qb_change(long_df: pd.DataFrame) -> pd.DataFrame:
    """Flag when a team's current starter differs from its recent starter.

    For each game, compares the current qb_id to the most common qb_id over
    up to the team's last QB_HISTORY_WINDOW prior games (ties broken by most
    recent). 1 = different starter (possible injury/benching), 0 = same,
    NaN = the team has no prior QB history yet.
    """
    long_df = long_df.copy()
    qb_change = np.full(len(long_df), np.nan)

    for _, idx in long_df.groupby("team").groups.items():
        idx = idx.to_numpy()
        history: deque = deque(maxlen=QB_HISTORY_WINDOW)
        for row_pos in idx:
            current_qb = long_df.loc[row_pos, "qb_id"]
            if len(history) > 0:
                recent_starter = _most_common(history)
                qb_change[row_pos] = float(current_qb != recent_starter)
            history.append(current_qb)

    long_df["qb_change"] = qb_change
    return long_df


def _add_prior_rolling_avgs(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages of pts_scored/pts_allowed/turnover_margin using
    only prior games.

    Shifting by 1 before rolling excludes the current game; rolling with
    min_periods=1 uses however many prior games are available (< window early
    on), and naturally yields NaN when there are zero prior games.
    """
    long_df = long_df.copy()
    grouped = long_df.groupby("team", group_keys=False)

    long_df["pts_scored_last5"] = grouped["pts_scored"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    long_df["pts_allowed_last5"] = grouped["pts_allowed"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    long_df["turnover_margin_last5"] = grouped["turnover_margin"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return long_df


def fetch_team_turnover_margins(seasons: list[int]) -> pd.DataFrame:
    """Per-team-game turnover margin (takeaways - giveaways) via play-by-play data.

    Aggregated from nfl_data_py.import_pbp_data(): a play is a turnover when
    either interception or fumble_lost is 1. Giveaways are turnovers on plays
    where the team was on offense (posteam); takeaways are turnovers on plays
    where the team was on defense (defteam). This naturally covers special
    teams turnovers (e.g. a muffed punt) too, since posteam/defteam reflect
    who had the ball on every play type.
    """
    frames = []
    for season in seasons:
        try:
            pbp = nfl.import_pbp_data([season], downcast=True)
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch play-by-play turnover data for {season}: {exc}")
            continue

        # nfl_data_py doesn't always raise on a 404 for an unpublished season
        # (e.g. one with no games played yet) -- it prints its own error and
        # returns an empty/malformed frame instead, so check for the columns
        # we need rather than assuming an exception would have caught this.
        if "posteam" not in pbp.columns or "defteam" not in pbp.columns:
            print(f"  [warn] play-by-play data for {season} is missing expected columns -- skipping")
            continue

        pbp = pbp[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()
        pbp["turnover"] = ((pbp["interception"] == 1) | (pbp["fumble_lost"] == 1)).astype(int)

        giveaways = (
            pbp.groupby(["game_id", "posteam"])["turnover"].sum().rename("giveaways").reset_index()
        )
        giveaways = giveaways.rename(columns={"posteam": "team"})
        takeaways = (
            pbp.groupby(["game_id", "defteam"])["turnover"].sum().rename("takeaways").reset_index()
        )
        takeaways = takeaways.rename(columns={"defteam": "team"})

        merged = giveaways.merge(takeaways, on=["game_id", "team"], how="outer").fillna(0)
        frames.append(merged)
        print(f"  {season}: turnover data from import_pbp_data ({len(merged)} team-games)")

    if not frames:
        return pd.DataFrame(columns=["game_id", "team", "turnover_margin"])

    result = pd.concat(frames, ignore_index=True)
    result["turnover_margin"] = result["takeaways"] - result["giveaways"]
    return result[["game_id", "team", "turnover_margin"]]


def _add_prior_sos_adjustment(long_df: pd.DataFrame) -> pd.DataFrame:
    """Add a strength-of-schedule-adjusted version of pts_scored_last5.

    For each of a team's last ROLLING_WINDOW games, looks up how many points
    that specific opponent was allowing entering that same matchup (the
    opponent's own pts_allowed_last5 at that historical game -- already
    computed no-leakage, and safely in the past relative to the *current*
    game since it comes from one of our own prior games). Averaging those
    gives a measure of how tough the schedule faced actually was; scoring
    against weak defenses is discounted, scoring against stingy ones is
    boosted:

        adjusted = pts_scored_last5 * (league_avg_pts_allowed / avg_opponent_def_strength)

    league_avg_pts_allowed is a single dataset-wide constant used purely as a
    fixed rescaling anchor -- it isn't specific to any one game, so it
    doesn't leak game-specific information.
    """
    long_df = long_df.copy()
    lookup = dict(zip(zip(long_df["team"], long_df["game_id"]), long_df["pts_allowed_last5"]))
    league_avg_pts_allowed = long_df["pts_allowed"].mean()

    sos_adj = np.full(len(long_df), np.nan)

    for _, idx in long_df.groupby("team").groups.items():
        idx = idx.to_numpy()
        window: deque = deque(maxlen=ROLLING_WINDOW)
        for row_pos in idx:
            if len(window) > 0:
                opp_def_values = [
                    lookup[key] for key in window if key in lookup and pd.notna(lookup[key])
                ]
                if opp_def_values:
                    avg_opp_def = float(np.mean(opp_def_values))
                    pts_scored_last5_val = long_df.at[row_pos, "pts_scored_last5"]
                    if avg_opp_def > 0 and pd.notna(pts_scored_last5_val):
                        sos_adj[row_pos] = pts_scored_last5_val * (league_avg_pts_allowed / avg_opp_def)

            window.append((long_df.at[row_pos, "opponent"], long_df.at[row_pos, "game_id"]))

    long_df["pts_scored_last5_sos_adj"] = sos_adj
    return long_df


def _passer_rating(completions, attempts, yards, tds, ints) -> float:
    """Standard NFL passer rating from its four components (each clamped 0-2.375)."""
    a = np.clip(((completions / attempts) - 0.3) * 5, 0, 2.375)
    b = np.clip(((yards / attempts) - 3) * 0.25, 0, 2.375)
    c = np.clip((tds / attempts) * 20, 0, 2.375)
    d = np.clip(2.375 - (ints / attempts) * 25, 0, 2.375)
    return ((a + b + c + d) / 6) * 100


def _qb_ratings_from_weekly_data(season: int) -> pd.DataFrame:
    """Per-game QB passer rating for one season via nfl_data_py.import_weekly_data()."""
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[(weekly["position"] == "QB") & (weekly["attempts"] > 0)].copy()

    weekly["passer_rating"] = _passer_rating(
        weekly["completions"],
        weekly["attempts"],
        weekly["passing_yards"],
        weekly["passing_tds"],
        weekly["interceptions"],
    )

    weekly = weekly.rename(columns={"player_id": "qb_id"})
    return weekly[["qb_id", "season", "week", "passer_rating"]]


def _qb_ratings_from_ngs_data(season: int) -> pd.DataFrame:
    """Per-game QB passer rating for one season via nfl_data_py.import_ngs_data().

    Used as a fallback when import_weekly_data() hasn't published the season
    yet (e.g. the current season, mid-year). NGS already provides a computed
    passer_rating column -- verified to exactly match our own formula against
    import_weekly_data on a known player/week.

    NGS numbers the Super Bowl as week 23 (leaving week 22 unused), while
    import_weekly_data / schedules.csv number it as week 22. Remap so both
    sources line up on the same week numbering for postseason games.
    """
    ngs = nfl.import_ngs_data("passing", [season])
    # NGS includes a week == 0 row per player holding *season totals*, not a
    # game -- must be dropped or it gets treated as an extra (and wildly
    # inflated) game in every rolling calculation.
    ngs = ngs[(ngs["player_position"] == "QB") & (ngs["attempts"] > 0) & (ngs["week"] > 0)].copy()

    ngs.loc[(ngs["season_type"] == "POST") & (ngs["week"] == 23), "week"] = 22

    ngs = ngs.rename(columns={"player_gsis_id": "qb_id"})
    return ngs[["qb_id", "season", "week", "passer_rating"]]


def fetch_qb_weekly_ratings(seasons: list[int]) -> pd.DataFrame:
    """Pull per-game passer rating for each QB across the given seasons.

    Fetches season-by-season, preferring import_weekly_data(); if a season
    isn't published there yet (e.g. the most recent season, mid-year), falls
    back to import_ngs_data('passing', ...). A season is only skipped (with a
    warning) if neither source has it.
    """
    frames = []
    for season in seasons:
        try:
            frames.append(_qb_ratings_from_weekly_data(season))
            print(f"  {season}: QB ratings from import_weekly_data")
            continue
        except Exception as exc:  # noqa: BLE001 - try the fallback source
            print(f"  {season}: import_weekly_data unavailable ({exc}); trying import_ngs_data")

        try:
            frames.append(_qb_ratings_from_ngs_data(season))
            print(f"  {season}: QB ratings from import_ngs_data (fallback)")
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch QB ratings for {season} from either source: {exc}")

    if not frames:
        return pd.DataFrame(columns=["qb_id", "season", "week", "passer_rating"])

    return pd.concat(frames, ignore_index=True)


def _add_prior_qb_rating(qb_games: pd.DataFrame) -> pd.DataFrame:
    """Add each QB's rolling passer rating average using only prior starts.

    Shift-before-rolling: excludes the current game, uses however many prior
    starts are available (< QB_RATING_WINDOW early in a QB's log), and is NaN
    with zero prior starts.
    """
    qb_games = qb_games.sort_values(["qb_id", "season", "week"]).reset_index(drop=True)
    grouped = qb_games.groupby("qb_id", group_keys=False)
    qb_games["qb_rating_last5"] = grouped["passer_rating"].apply(
        lambda s: s.shift(1).rolling(QB_RATING_WINDOW, min_periods=1).mean()
    )
    return qb_games


def add_qb_rating_features(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Add home_qb_rating_last5 / away_qb_rating_last5 to the game-level df.

    Joins each game's starter (home_qb_id/away_qb_id) to that QB's rolling
    passer-rating average entering that (season, week) via nfl_data_py weekly
    stats -- a separate data source from the schedule file.
    """
    df = df.copy()

    qb_games = fetch_qb_weekly_ratings(seasons)
    qb_games = _add_prior_qb_rating(qb_games)

    ratings = qb_games[["qb_id", "season", "week", "qb_rating_last5"]]

    df = df.merge(
        ratings.rename(columns={"qb_id": "home_qb_id", "qb_rating_last5": "home_qb_rating_last5"}),
        on=["home_qb_id", "season", "week"],
        how="left",
    )
    df = df.merge(
        ratings.rename(columns={"qb_id": "away_qb_id", "qb_rating_last5": "away_qb_rating_last5"}),
        on=["away_qb_id", "season", "week"],
        how="left",
    )
    return df


def add_rolling_team_features(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Add home/away rolling scoring averages, win streaks, QB continuity
    flags, turnover margin, a strength-of-schedule-adjusted scoring feature,
    and rest_advantage.

    All rolling stats use only each team's games strictly before the current
    gameday (no data leakage). Values are NaN wherever a team has zero prior
    games (e.g. the start of a team's first season in the data).
    """
    df = df.copy()

    long_df = _build_team_game_long(df)

    print("Fetching play-by-play data for turnover-margin feature...")
    turnovers = fetch_team_turnover_margins(seasons)
    long_df = long_df.merge(turnovers, on=["game_id", "team"], how="left")

    long_df = _add_prior_rolling_avgs(long_df)
    long_df = _add_prior_sos_adjustment(long_df)
    long_df = _add_prior_win_streak(long_df)
    long_df = _add_prior_qb_change(long_df)

    feature_cols = [
        "pts_scored_last5",
        "pts_allowed_last5",
        "turnover_margin_last5",
        "pts_scored_last5_sos_adj",
        "win_streak",
        "qb_change",
    ]

    home_feats = long_df.loc[long_df["side"] == "home", ["game_id", *feature_cols]].rename(
        columns={
            "pts_scored_last5": "home_pts_scored_last5",
            "pts_allowed_last5": "home_pts_allowed_last5",
            "turnover_margin_last5": "home_turnover_margin_last5",
            "pts_scored_last5_sos_adj": "home_pts_scored_last5_sos_adj",
            "win_streak": "home_win_streak",
            "qb_change": "home_qb_change",
        }
    )
    away_feats = long_df.loc[long_df["side"] == "away", ["game_id", *feature_cols]].rename(
        columns={
            "pts_scored_last5": "away_pts_scored_last5",
            "pts_allowed_last5": "away_pts_allowed_last5",
            "turnover_margin_last5": "away_turnover_margin_last5",
            "pts_scored_last5_sos_adj": "away_pts_scored_last5_sos_adj",
            "win_streak": "away_win_streak",
            "qb_change": "away_qb_change",
        }
    )

    df = df.merge(home_feats, on="game_id", how="left")
    df = df.merge(away_feats, on="game_id", how="left")
    df["rest_advantage"] = df["home_rest"] - df["away_rest"]

    return df


STARTER_SNAP_SHARE = 0.60
BAD_INJURY_STATUSES = {"Out", "Doubtful"}


def _normalize_player_name(name: str) -> str:
    """Normalize a player name for joining across data sources.

    Strips periods (e.g. "A.J. Brown" -> "AJ Brown") and trailing
    generational suffixes (e.g. "Charles Leno Jr." -> "Charles Leno"), which
    otherwise cause a small number of join misses between snap_counts and
    import_injuries() names for the same player.
    """
    if not isinstance(name, str):
        return name
    name = name.replace(".", "")
    name = re.sub(r"\s+(Jr|Sr|II|III|IV|V)$", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def fetch_all_position_snap_shares(seasons: list[int]) -> pd.DataFrame:
    """Per-player-game snap share (offense or defense, whichever applies) via
    nfl_data_py.import_snap_counts(), across every position.

    A "regular starter" for this feature can play either side of the ball,
    so effective_pct is whichever of offense_pct/defense_pct is nonzero for
    that player (special-teams-only players end up near 0 on both, which
    correctly excludes them from ever counting as a "starter" here).
    """
    frames = []
    for season in seasons:
        try:
            snaps = nfl.import_snap_counts([season])
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch snap counts for {season}: {exc}")
            continue
        snaps = snaps.copy()
        snaps["effective_pct"] = snaps[["offense_pct", "defense_pct"]].fillna(0).max(axis=1)
        snaps["name_key"] = snaps["player"].map(_normalize_player_name)
        snaps["period"] = (snaps["season"] * 100 + snaps["week"]).astype("int64")
        frames.append(snaps[["season", "week", "period", "team", "name_key", "position", "effective_pct"]])

    if not frames:
        return pd.DataFrame(columns=["season", "week", "period", "team", "name_key", "position", "effective_pct"])
    return pd.concat(frames, ignore_index=True)


def _add_is_starter_entering(snap_shares: pd.DataFrame) -> pd.DataFrame:
    """Add is_starter_entering: this player's own shift-before-rolling
    average snap share over up to their last 5 games was >= STARTER_SNAP_SHARE.

    Scoped to (name_key, team) rather than name_key alone. Unlike the
    single-position QB/RB/WR pipelines (small player pools, low collision
    risk), this feature spans every position on every roster -- confirmed a
    real collision here (there are two actual NFL players named "Lamar
    Jackson": the Ravens QB and a Bears CB), which corrupted both players'
    rolling history when grouped by name alone. Scoping by team too avoids
    that, since two same-named players essentially never share a roster,
    at the cost of resetting a traded player's snap-share history for their
    new team (a reasonable trade-off here: "established starter for team X"
    arguably should reflect their tenure with X anyway).
    """
    snap_shares = snap_shares.sort_values(["name_key", "team", "season", "week"]).reset_index(drop=True)
    grouped = snap_shares.groupby(["name_key", "team"], group_keys=False)
    rolling_avg = grouped["effective_pct"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    snap_shares["is_starter_entering"] = rolling_avg >= STARTER_SNAP_SHARE
    return snap_shares


def _team_full_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """(team, season, week, period) for every game every team plays."""
    home = schedule[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    away = schedule[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    full = pd.concat([home, away], ignore_index=True)
    full["period"] = (full["season"] * 100 + full["week"]).astype("int64")
    return full


def build_presumed_starters(snap_shares: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """For every team-game, the roster of players presumed to be starters
    entering it, based on their *last known* appearance for that team
    (looking strictly backward -- never the current game's own data, since
    an injured/inactive player won't have a current-game row at all).

    This is the same "look at the last time this team played" pattern used
    for top_target_availability in player_features.py, generalized from a
    single player to a whole roster via a grouped as-of merge.
    """
    team_schedule = _team_full_schedule(schedule)

    team_players = snap_shares[["team", "name_key"]].drop_duplicates()
    grid = team_players.merge(team_schedule, on="team", how="left")
    # merge_asof requires the "on" column globally sorted (not just within
    # each "by" group), so period must be the primary sort key here.
    grid = grid.sort_values(["period", "team", "name_key"]).reset_index(drop=True)

    source = snap_shares[["team", "name_key", "period", "position", "is_starter_entering"]].copy()
    source["matched_period"] = source["period"]
    source = source.sort_values(["period", "team", "name_key"]).reset_index(drop=True)

    # allow_exact_matches=True (the default): is_starter_entering is already
    # leak-free (shift-before-rolling over that player's own PRIOR games), so
    # when the player has a row for this exact game, use it directly rather
    # than needlessly falling back to a staler one. The backward look-up
    # only matters when there's no current-week row at all -- e.g. the
    # player is inactive/injured this week, so their last known status
    # carries forward.
    presumed = pd.merge_asof(
        grid,
        source,
        on="period",
        by=["team", "name_key"],
        direction="backward",
    )
    presumed = presumed[presumed["is_starter_entering"] == True].copy()  # noqa: E712 - explicit bool filter, not NaN
    return presumed


def fetch_injury_status_by_name(seasons: list[int]) -> pd.DataFrame:
    """(season, week, team, name_key) -> report_status via
    nfl_data_py.import_injuries(), name-normalized for joining to
    snap_counts-derived data (which has no shared ID scheme with injuries)."""
    frames = []
    for season in seasons:
        try:
            inj = nfl.import_injuries([season])
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch injuries for {season}: {exc}")
            continue
        inj = inj.copy()
        inj["name_key"] = inj["full_name"].map(_normalize_player_name)
        frames.append(inj[["season", "week", "team", "name_key", "report_status"]])

    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", "name_key", "report_status"])
    return pd.concat(frames, ignore_index=True)


def add_team_injury_features(df: pd.DataFrame, seasons: list[int], schedule: pd.DataFrame) -> pd.DataFrame:
    """Add home_starters_out / away_starters_out (count of presumed regular
    starters, any position, listed Out/Doubtful that week) and
    home_qb_out / away_qb_out (binary, specifically for the presumed
    starting QB).
    """
    df = df.copy()

    snap_shares = fetch_all_position_snap_shares(seasons)
    snap_shares = _add_is_starter_entering(snap_shares)
    presumed_starters = build_presumed_starters(snap_shares, schedule)

    injuries = fetch_injury_status_by_name(seasons)
    presumed_starters = presumed_starters.merge(
        injuries, on=["season", "week", "team", "name_key"], how="left"
    )
    presumed_starters["is_out_or_doubtful"] = presumed_starters["report_status"].isin(BAD_INJURY_STATUSES)

    starters_out = presumed_starters.groupby(["season", "week", "team"])["is_out_or_doubtful"].sum().reset_index(
        name="starters_out"
    )

    # QB-specific: among presumed-starter QBs for a team-game, the "real"
    # starter is whichever one's last known qualifying appearance is most
    # recent (handles the rare case of two backup-turned-starters both
    # having a recent qualifying stretch).
    qb_candidates = presumed_starters[presumed_starters["position"] == "QB"].sort_values(
        ["season", "week", "team", "matched_period"]
    )
    starting_qb = qb_candidates.groupby(["season", "week", "team"]).tail(1)
    qb_out = starting_qb[["season", "week", "team", "is_out_or_doubtful"]].rename(
        columns={"is_out_or_doubtful": "qb_out"}
    )
    qb_out["qb_out"] = qb_out["qb_out"].astype(int)

    team_features = starters_out.merge(qb_out, on=["season", "week", "team"], how="left")

    for side, col_team in [("home", "home_team"), ("away", "away_team")]:
        side_feats = team_features.rename(
            columns={
                "team": col_team,
                "starters_out": f"{side}_starters_out",
                "qb_out": f"{side}_qb_out",
            }
        )
        df = df.merge(side_feats, on=["season", "week", col_team], how="left")

    return df


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = add_dome_flag(df)

    DOME_FLAG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DOME_FLAG_OUTPUT_PATH, index=False)

    seasons = sorted(df["season"].unique().tolist())
    df = add_rolling_team_features(df, seasons)

    print("Fetching weekly QB stats for passer-rating feature...")
    df = add_qb_rating_features(df, seasons)

    print("Fetching snap counts and injury reports for team-injury features...")
    df = add_team_injury_features(df, seasons, df)

    FEATURES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {FEATURES_OUTPUT_PATH}")
    print()
    new_cols = [
        "home_team",
        "away_team",
        "season",
        "week",
        "home_pts_scored_last5",
        "home_pts_allowed_last5",
        "away_pts_scored_last5",
        "away_pts_allowed_last5",
        "home_win_streak",
        "away_win_streak",
        "rest_advantage",
    ]
    print(df[new_cols].head(10).to_string(index=False))
    print()

    qb_cols = [
        "home_qb_id",
        "home_qb_name",
        "home_qb_change",
        "away_qb_id",
        "away_qb_name",
        "away_qb_change",
    ]
    mid_season = df[(df["season"] == 2022) & (df["week"] >= 6)].sort_values("week")
    print("QB continuity (2022, week 6+):")
    print(mid_season[qb_cols].head(10).to_string(index=False))
    print()

    rating_cols = [
        "home_team",
        "away_team",
        "week",
        "home_qb_name",
        "home_qb_rating_last5",
        "away_qb_name",
        "away_qb_rating_last5",
    ]
    mid_2023 = df[(df["season"] == 2023) & (df["week"].between(6, 10))].sort_values("week")
    print("QB rating (2023, weeks 6-10):")
    print(mid_2023[rating_cols].head(10).to_string(index=False))
    print()

    print("Recognizable QBs, full 2023 season (rolling rating entering each game):")
    known_qbs = ["Josh Allen", "Patrick Mahomes"]
    for name in known_qbs:
        as_home = df[(df["season"] == 2023) & (df["home_qb_name"] == name)][
            ["week", "home_qb_rating_last5"]
        ].rename(columns={"home_qb_rating_last5": "qb_rating_last5"})
        as_away = df[(df["season"] == 2023) & (df["away_qb_name"] == name)][
            ["week", "away_qb_rating_last5"]
        ].rename(columns={"away_qb_rating_last5": "qb_rating_last5"})
        combined = pd.concat([as_home, as_away]).sort_values("week")
        print(f"-- {name} --")
        print(combined.to_string(index=False))
    print()

    print("home_qb_rating_last5 / away_qb_rating_last5 null counts by season:")
    null_counts = df.groupby("season")[["home_qb_rating_last5", "away_qb_rating_last5"]].apply(
        lambda g: g.isnull().sum()
    )
    print(null_counts.to_string())
    print()

    new_feature_cols = [
        "home_turnover_margin_last5",
        "away_turnover_margin_last5",
        "home_pts_scored_last5",
        "home_pts_scored_last5_sos_adj",
        "away_pts_scored_last5",
        "away_pts_scored_last5_sos_adj",
    ]
    mid_2023_full = df[(df["season"] == 2023) & (df["week"].between(6, 10))].sort_values("week")
    print("Turnover margin & SOS-adjusted scoring (2023, weeks 6-10):")
    print(
        mid_2023_full[["home_team", "away_team", "week", *new_feature_cols]]
        .head(10)
        .to_string(index=False)
    )
    print()

    print("Null counts for the new columns:")
    print(df[new_feature_cols].isnull().sum().to_string())
    print()

    injury_cols = ["home_starters_out", "away_starters_out", "home_qb_out", "away_qb_out"]
    print("Null counts for the team-injury columns:")
    print(df[injury_cols].isnull().sum().to_string())
    print()

    high_starters_out = df[
        (df["home_starters_out"] >= 3) | (df["away_starters_out"] >= 3)
    ].sort_values(["season", "week"])
    print(f"Sample of games with 3+ starters out on either side ({len(high_starters_out)} total):")
    print(
        high_starters_out[
            [
                "season",
                "week",
                "home_team",
                "away_team",
                "home_starters_out",
                "away_starters_out",
                "home_qb_out",
                "away_qb_out",
                "home_score",
                "away_score",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
