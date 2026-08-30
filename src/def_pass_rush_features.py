"""Build a pass-rusher prop dataset: one row per starting edge/interior
pass rusher per game, target = sacks recorded that game.

Pipeline:
  1. Pull weekly defensive stats via nfl_data_py.import_weekly_pfr('def',
     seasons) -- def_sacks, def_pressures, at (season, week, team, opponent)
     granularity. Works for all of 2022-2025 (unlike import_weekly_data(),
     which 404s for 2025).
  2. Join to nfl_data_py.import_snap_counts() via pfr_player_id to get each
     player's position and defense_pct; keep only starters (defense_pct >=
     STARTER_SNAP_SHARE).
  3. Normalize position into two buckets:
       edge     = DE, OLB, LB, LDE, RDE, LOLB, ROLB, LLB, RLB (and variants)
       interior = DT, LDT, RDT, DL, NT
     Non-front-seven positions (DBs, etc.) are dropped.
  4. Add rolling, no-leakage features (shift-before-rolling):
     def_sacks_last5, def_pressures_last5 (each player's own prior games),
     and opponent_oline_sacks_allowed_last5 -- the upcoming opponent's own
     recent sacks-allowed-per-game, a proxy for "how leaky is this O-line
     lately" (summed across *all* defenders who sacked them, not just
     starters, so it reflects the offense's true total sacks allowed).
  5. Join spread_line, total_line, and is_dome in from
     data/processed/games_with_features.csv.
  6. Target: def_sacks (the player's actual sacks recorded that game).

Usage:
    python src/def_pass_rush_features.py
"""

from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEDULE_PATH = BASE_DIR / "data" / "raw" / "schedules.csv"
GAME_FEATURES_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "pass_rush_features.csv"

STARTER_SNAP_SHARE = 0.60
ROLLING_WINDOW = 5

EDGE_POSITIONS = {"DE", "OLB", "LB", "LDE", "RDE", "LOLB", "ROLB", "LLB", "RLB", "LILB", "RILB"}
INTERIOR_POSITIONS = {"DT", "LDT", "RDT", "DL", "NT"}


def classify_position(position: str) -> str | None:
    """Map a raw position label to 'edge', 'interior', or None (not a
    front-seven pass-rush position, e.g. a DB).

    Combo labels from a mid-season position switch (e.g. "NT-RDE") are
    classified by their first listed position.
    """
    if not isinstance(position, str) or not position:
        return None
    primary = position.upper().split("-")[0].strip()
    if primary in EDGE_POSITIONS:
        return "edge"
    if primary in INTERIOR_POSITIONS:
        return "interior"
    return None


def fetch_weekly_def_stats(seasons: list[int]) -> pd.DataFrame:
    """Per-defender-game sacks/pressures via nfl_data_py.import_weekly_pfr('def', ...)."""
    frames = []
    for season in seasons:
        try:
            frames.append(nfl.import_weekly_pfr("def", [season]))
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch weekly defensive stats for {season}: {exc}")

    if not frames:
        return pd.DataFrame(
            columns=["pfr_player_id", "pfr_player_name", "team", "opponent", "season", "week", "def_sacks", "def_pressures"]
        )

    result = pd.concat(frames, ignore_index=True)
    return result[
        ["pfr_player_id", "pfr_player_name", "team", "opponent", "season", "week", "def_sacks", "def_pressures"]
    ]


def fetch_defense_snap_shares(seasons: list[int]) -> pd.DataFrame:
    """Per-defender-game position and defensive snap share via
    nfl_data_py.import_snap_counts()."""
    frames = []
    for season in seasons:
        try:
            snaps = nfl.import_snap_counts([season])
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  [warn] could not fetch snap counts for {season}: {exc}")
            continue
        frames.append(snaps[["season", "week", "pfr_player_id", "position", "defense_pct"]])

    if not frames:
        return pd.DataFrame(columns=["season", "week", "pfr_player_id", "position", "defense_pct"])
    return pd.concat(frames, ignore_index=True)


def build_starters(def_stats: pd.DataFrame, snap_shares: pd.DataFrame) -> pd.DataFrame:
    """Join weekly def stats to snap shares via pfr_player_id, classify
    position into edge/interior, and keep only starters at those positions."""
    merged = def_stats.merge(
        snap_shares,
        on=["season", "week", "pfr_player_id"],
        how="inner",
    )
    merged["position_group"] = merged["position"].map(classify_position)
    starters = merged[
        (merged["defense_pct"] >= STARTER_SNAP_SHARE) & (merged["position_group"].notna())
    ].copy()
    return starters


def add_rolling_pass_rush_features(starters: pd.DataFrame) -> pd.DataFrame:
    """Add def_sacks_last5 / def_pressures_last5, no-leakage
    shift-before-rolling over each player's own prior starts."""
    starters = starters.sort_values(["pfr_player_id", "season", "week"]).reset_index(drop=True)
    grouped = starters.groupby("pfr_player_id", group_keys=False)

    starters["def_sacks_last5"] = grouped["def_sacks"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    starters["def_pressures_last5"] = grouped["def_pressures"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return starters


def add_opponent_oline_context(starters: pd.DataFrame, all_def_stats: pd.DataFrame) -> pd.DataFrame:
    """Add opponent_oline_sacks_allowed_last5: the upcoming opponent's own
    recent sacks-allowed-per-game.

    Uses *all* weekly def stats (not just starters), summed per (season,
    week, opponent) -- i.e. total sacks that offense allowed that week,
    from every defender who recorded one -- then rolled with the same
    shift-before-rolling no-leakage pattern used everywhere else in this
    project, and looked up for each pass rusher's specific upcoming
    opponent that week.
    """
    sacks_allowed = (
        all_def_stats.groupby(["season", "week", "opponent"])["def_sacks"]
        .sum()
        .reset_index()
        .rename(columns={"opponent": "team", "def_sacks": "sacks_allowed"})
    )
    sacks_allowed = sacks_allowed.sort_values(["team", "season", "week"]).reset_index(drop=True)

    grouped = sacks_allowed.groupby("team", group_keys=False)
    sacks_allowed["sacks_allowed_last5"] = grouped["sacks_allowed"].apply(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )

    starters = starters.merge(
        sacks_allowed[["season", "week", "team", "sacks_allowed_last5"]].rename(
            columns={"team": "opponent", "sacks_allowed_last5": "opponent_oline_sacks_allowed_last5"}
        ),
        on=["season", "week", "opponent"],
        how="left",
    )
    return starters


def add_game_context(starters: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Join is_dome, spread_line, and total_line in from games_with_features.csv.

    spread_line is home-perspective there (positive = home favored) and gets
    sign-flipped for the away team, so it reads as "this team's own" spread.
    total_line (the game's combined over/under) isn't team-specific.
    """
    home = games[["season", "week", "home_team", "is_dome", "spread_line", "total_line"]].rename(
        columns={"home_team": "team"}
    )
    away = games[["season", "week", "away_team", "is_dome", "spread_line", "total_line"]].rename(
        columns={"away_team": "team"}
    )
    away["spread_line"] = -away["spread_line"]
    context = pd.concat([home, away], ignore_index=True)

    return starters.merge(context, on=["season", "week", "team"], how="left")


def main() -> None:
    schedule = pd.read_csv(SCHEDULE_PATH)
    games = pd.read_csv(GAME_FEATURES_PATH)
    seasons = sorted(schedule["season"].unique().tolist())

    print("Fetching weekly defensive stats...")
    def_stats = fetch_weekly_def_stats(seasons)

    print("Fetching snap counts...")
    snap_shares = fetch_defense_snap_shares(seasons)

    starters = build_starters(def_stats, snap_shares)
    print(f"Starter edge/interior pass-rusher games (defense_pct >= {STARTER_SNAP_SHARE}): {len(starters)}")
    print()
    print("Position group counts:")
    print(starters["position_group"].value_counts().to_string())

    starters = add_rolling_pass_rush_features(starters)
    starters = add_opponent_oline_context(starters, def_stats)
    starters = add_game_context(starters, games)

    starters = starters.sort_values(["season", "week", "team"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    starters.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved {len(starters)} rows to {OUTPUT_PATH}")
    print()
    print("Null counts:")
    print(
        starters[
            [
                "def_sacks",
                "def_pressures",
                "def_sacks_last5",
                "def_pressures_last5",
                "opponent_oline_sacks_allowed_last5",
                "is_dome",
                "spread_line",
                "total_line",
            ]
        ]
        .isnull()
        .sum()
        .to_string()
    )
    print()

    print("def_sacks distribution (player-games):")
    sack_buckets = pd.cut(
        starters["def_sacks"], bins=[-0.01, 0.01, 1.01, 2.01, np.inf], labels=["0", "1", "2", "3+"]
    )
    print(sack_buckets.value_counts().sort_index().to_string())
    print(f"Mean def_sacks per game: {starters['def_sacks'].mean():.4f}")
    print(f"% of player-games with 0 sacks: {(starters['def_sacks'] == 0).mean():.2%}")
    print()

    known_rusher = "Myles Garrett"
    sample = starters[
        (starters["pfr_player_name"] == known_rusher)
        & (starters["season"] == 2023)
        & (starters["week"].between(6, 15))
    ].sort_values("week")
    print(f"{known_rusher}, 2023 weeks 6-15:")
    print(
        sample[
            [
                "season",
                "week",
                "team",
                "opponent",
                "position_group",
                "def_sacks",
                "def_pressures",
                "def_sacks_last5",
                "def_pressures_last5",
                "opponent_oline_sacks_allowed_last5",
                "is_dome",
                "spread_line",
                "total_line",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
