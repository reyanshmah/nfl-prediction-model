"""Determine a team's currently expected starting QB for an upcoming game.

Design:
  1. Pull the latest nfl_data_py.import_depth_charts() snapshot for the
     team's QB position, strictly before that game's date -- take the
     pos_rank == 1 player as the presumptive starter.
  2. Cross-check that player's report_status via nfl_data_py.import_injuries()
     for that (season, week). If Out or Doubtful, fall back to the depth
     chart's pos_rank == 2 player instead.
  3. Report which player was used, and flag clearly if a fallback fired.

This is inherently a *forward-looking* heuristic (unlike player_features.py's
retrospective snap-count-based starter identification, which only works
after a game has already been played) -- see main() for a live test against
Houston's actual 2025 QB situation, including a case where it gets fooled.

Usage:
    python src/get_current_starter.py
"""

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEDULE_PATH = BASE_DIR / "data" / "raw" / "schedules.csv"

BAD_STATUSES = {"Out", "Doubtful"}


def _game_date(schedule: pd.DataFrame, team: str, season: int, week: int) -> pd.Timestamp:
    """The team's game date for (season, week), from the schedule file."""
    game = schedule[
        (schedule["season"] == season)
        & (schedule["week"] == week)
        & ((schedule["home_team"] == team) | (schedule["away_team"] == team))
    ]
    if game.empty:
        raise ValueError(f"No scheduled game found for {team}, season {season}, week {week}")
    return pd.to_datetime(game.iloc[0]["gameday"])


def _latest_qb_depth_chart(team: str, season: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    """The most recent depth_charts QB snapshot for `team` strictly before `cutoff`."""
    dc = nfl.import_depth_charts([season])
    dc = dc[(dc["team"] == team) & (dc["pos_abb"] == "QB")].copy()
    dc["dt"] = pd.to_datetime(dc["dt"]).dt.tz_localize(None)
    dc = dc[dc["dt"] < cutoff]

    if dc.empty:
        return dc

    latest_dt = dc["dt"].max()
    return dc[dc["dt"] == latest_dt].sort_values("pos_rank")


def get_current_starter(team: str, season: int, week: int, schedule: pd.DataFrame) -> dict:
    """Determine the presumptive starting QB for `team` entering (season, week).

    Returns a dict with the depth chart's #1 and #2 QBs, the #1's injury
    status, which one was ultimately selected, and whether the injury
    fallback fired.
    """
    cutoff = _game_date(schedule, team, season, week)
    depth_chart = _latest_qb_depth_chart(team, season, cutoff)

    if depth_chart.empty:
        return {
            "team": team,
            "season": season,
            "week": week,
            "primary_name": None,
            "primary_status": None,
            "fallback_triggered": False,
            "starter_name": None,
            "note": "No depth chart snapshot available before this game.",
        }

    primary = depth_chart[depth_chart["pos_rank"] == 1]
    backup = depth_chart[depth_chart["pos_rank"] == 2]

    primary_name = primary.iloc[0]["player_name"] if not primary.empty else None
    primary_gsis_id = primary.iloc[0]["gsis_id"] if not primary.empty else None

    injuries = nfl.import_injuries([season])
    status_row = injuries[
        (injuries["season"] == season) & (injuries["week"] == week) & (injuries["gsis_id"] == primary_gsis_id)
    ]
    primary_status = status_row.iloc[0]["report_status"] if not status_row.empty else None

    fallback_triggered = primary_status in BAD_STATUSES
    if fallback_triggered and not backup.empty:
        starter_name = backup.iloc[0]["player_name"]
        note = f"{primary_name} listed '{primary_status}' -- falling back to depth chart #2: {starter_name}"
    elif fallback_triggered:
        starter_name = primary_name
        note = f"{primary_name} listed '{primary_status}' but no #2 QB found on depth chart -- using #1 anyway"
    else:
        starter_name = primary_name
        note = "No fallback triggered (primary QB not listed Out/Doubtful)"

    return {
        "team": team,
        "season": season,
        "week": week,
        "primary_name": primary_name,
        "primary_status": primary_status,
        "fallback_triggered": fallback_triggered,
        "starter_name": starter_name,
        "note": note,
    }


def main() -> None:
    schedule = pd.read_csv(SCHEDULE_PATH)

    team = "HOU"
    season = 2025
    # Actual 2025 HOU starters per qb_passing_features.csv:
    #   wk 1-8: C.J. Stroud | wk 9-12: Davis Mills | wk 13+: C.J. Stroud
    test_weeks = [1, 9, 10, 11, 12, 13]
    actual_starters = {
        1: "C.J. Stroud",
        9: "Davis Mills",
        10: "Davis Mills",
        11: "Davis Mills",
        12: "Davis Mills",
        13: "C.J. Stroud",
    }

    print(f"Testing get_current_starter() against {team}'s actual 2025 QB situation:")
    print()
    for week in test_weeks:
        result = get_current_starter(team, season, week, schedule)
        actual = actual_starters[week]
        predicted = result["starter_name"]
        match = "OK" if predicted == actual else "MISMATCH"

        print(f"Week {week}: assumed starter = {predicted}  [{match}, actual = {actual}]")
        print(f"  Depth chart #1: {result['primary_name']} (status: {result['primary_status']})")
        print(f"  Fallback triggered: {result['fallback_triggered']}")
        print(f"  Note: {result['note']}")
        print()


if __name__ == "__main__":
    main()
