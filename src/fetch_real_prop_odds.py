"""Pull REAL NFL player-prop closing lines/odds/results from the
SportsGameOdds API (free tier), one request per regular-season week, and
cache the raw JSON to disk so this only has to hit the API once.

Only extracts passing_yards / rushing_yards / receiving_yards O/U markets
(discards the hundreds of other markets per event -- team spreads, quarter
lines, longest completion, etc.) into a flat CSV:
  week, event_teams, sgo_player_id, player_name_norm, stat,
  over_odds, under_odds, line, actual

Usage:
    python src/fetch_real_prop_odds.py [season]   # defaults to 2025
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://api.sportsgameodds.com/v2/events/"

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

BASE_DIR = Path(__file__).resolve().parent.parent
GAMES_PATH = BASE_DIR / "data" / "processed" / "games_with_features.csv"
CACHE_DIR = BASE_DIR / "data" / "raw" / "sgo_cache" / str(SEASON)
OUT_PATH = BASE_DIR / "data" / "processed" / f"real_prop_odds_{SEASON}.csv"
ENV_PATH = BASE_DIR / ".env"


def load_api_key() -> str:
    """Reads SPORTSGAMEODDS_API_KEY from .env (gitignored -- never commit
    this file or hardcode the key directly in source)."""
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("SPORTSGAMEODDS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"SPORTSGAMEODDS_API_KEY not found in {ENV_PATH}")


API_KEY = load_api_key()

STATS = ["passing_yards", "rushing_yards", "receiving_yards"]


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.replace(".", "")
    name = re.sub(r"\s+(Jr|Sr|II|III|IV|V)$", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def sgo_id_to_name(player_id: str) -> str:
    # e.g. "MICHAEL_PENIX_JR_1_NFL" -> "michael penix" (normalized)
    core = re.sub(r"_\d+_NFL$", "", player_id)
    words = core.split("_")
    return normalize_name(" ".join(w.capitalize() for w in words))


def week_date_ranges():
    games = pd.read_csv(GAMES_PATH)
    wk = games[games["season"] == SEASON].groupby("week")["gameday"].agg(["min", "max"])
    wk = wk[wk.index <= 18]  # regular season only
    return wk


def fetch_week(week: int, starts_after: str, starts_before: str) -> tuple[dict, bool]:
    """Returns (data, was_cached)."""
    cache_file = CACHE_DIR / f"week_{week:02d}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8")), True

    params = {
        "apiKey": API_KEY,
        "leagueID": "NFL",
        "startsAfter": f"{starts_after}T00:00:00Z",
        "startsBefore": f"{starts_before}T23:59:59Z",
        "limit": 20,
    }
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-model-backtest/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data, False


def extract_props(data: dict, week: int) -> list[dict]:
    rows = []
    for ev in data.get("data", []):
        teams = ev.get("teams", {})
        home = teams.get("home", {}).get("names", {}).get("short", "")
        away = teams.get("away", {}).get("names", {}).get("short", "")
        odds = ev.get("odds", {})

        seen = set()
        for odd_id, entry in odds.items():
            stat = entry.get("statID")
            if stat not in STATS or entry.get("periodID") != "game":
                continue
            player_id = entry.get("playerID")
            if not player_id or player_id in seen:
                continue
            seen.add(player_id)

            over_key = f"{stat}-{player_id}-game-ou-over"
            under_key = f"{stat}-{player_id}-game-ou-under"
            over_entry = odds.get(over_key)
            under_entry = odds.get(under_key)
            if not over_entry or not under_entry:
                continue

            # closeBookOverUnder/closeBookOdds (the actual pre-kickoff closing
            # line) is only populated on more recent events -- for older
            # seasons (e.g. 2024) the free tier only retains a "bookOdds"/
            # "bookOverUnder" snapshot instead. Fall back to that rather than
            # dropping the row; it's still real market data, just not
            # guaranteed to be the exact closing moment the way 2025's is.
            line = over_entry.get("closeBookOverUnder", over_entry.get("bookOverUnder"))
            over_odds = over_entry.get("closeBookOdds", over_entry.get("bookOdds"))
            under_odds = under_entry.get("closeBookOdds", under_entry.get("bookOdds"))
            is_close_line = "closeBookOverUnder" in over_entry

            rows.append(
                {
                    "week": week,
                    "matchup": f"{away}@{home}",
                    "sgo_player_id": player_id,
                    "player_name_norm": sgo_id_to_name(player_id),
                    "stat": stat,
                    "line": line,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "actual": over_entry.get("score"),
                    "is_close_line": is_close_line,
                }
            )
    return rows


def main() -> None:
    weeks = week_date_ranges()
    all_rows = []
    for week, r in weeks.iterrows():
        print(f"Week {int(week)}: fetching {r['min']} to {r['max']}...")
        try:
            data, was_cached = fetch_week(int(week), r["min"], r["max"])
        except urllib.error.HTTPError as e:
            print(f"  ERROR: {e} -- stopping here, cached weeks so far are still saved.")
            break
        n_events = len(data.get("data", []))
        rows = extract_props(data, int(week))
        print(f"  {n_events} events, {len(rows)} player-prop lines extracted{' (cached)' if was_cached else ''}")
        all_rows.extend(rows)
        if not was_cached:
            time.sleep(8)  # stay well under the 10 req/min free-tier limit

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} total real prop lines to {OUT_PATH}")
    print(df["stat"].value_counts())


if __name__ == "__main__":
    main()
