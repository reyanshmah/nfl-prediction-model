# NFL Model — Methodology Reference

A complete record of the approach taken and the exact computation behind every statistic in this
project, from raw data through the live prediction dashboard.

## 1. Project setup

- Structure: `data/raw/`, `data/processed/`, `src/`, `notebooks/`, `requirements.txt`, `README.md`.
- **Python version issue**: the machine's default Python (3.14) has no compatible wheels for
  `nfl_data_py`'s pinned `numpy<2.0`/`pandas<2.0` (too new, no C compiler available to build from
  source). Fix: built the venv with **Python 3.12** instead, and installed `nfl_data_py` with
  `--no-deps` alongside modern `pandas`/`numpy` — verified it works fine despite the stale pins
  (pulled real 2023 schedule data as a smoke test).
- Core packages: `pandas`, `numpy`, `scikit-learn`, `nfl_data_py`, `xgboost`.

## 2. Raw data ingestion — [src/fetch_schedule.py](src/fetch_schedule.py)

`nfl_data_py.import_schedules([2022, 2023, 2024, 2025])` → [data/raw/schedules.csv](data/raw/schedules.csv)
(1139 games, 46 columns: teams, scores, dates, `spread_line`, `total_line`, moneylines, `roof`,
`div_game`, rest days, QB IDs, etc.).

## 3. Game-level features — [src/features.py](src/features.py)

Output: [data/processed/games_with_features.csv](data/processed/games_with_features.csv).

**No-leakage convention used everywhere in this project**: every rolling stat is computed via
`shift(1)` before `.rolling(window, min_periods=1).mean()` on a team/player's own chronologically
sorted history, so a value entering game *N* only ever reflects games *1..N-1*. Rolling stats carry
over across season boundaries (no reset each September) since there's no reason to discard a team's
end-of-last-season form entering week 1.

| Statistic | Exact method |
|---|---|
| `is_dome` | `roof.isin(['dome','closed'])` → 1, else 0 (covers `outdoors`, `open`, and nulls) |
| `home/away_pts_scored_last5`, `pts_allowed_last5` | Team-game long format (one row per team per game); `shift(1).rolling(5, min_periods=1).mean()` of points scored/allowed |
| `home/away_win_streak` | Sequential per-team loop: +1 if extending a win streak (reset to 1 if flipping from a loss streak), −1 symmetric for losses, 0 after a tie; NaN with no prior games |
| `rest_advantage` | `home_rest − away_rest`, taken directly from the schedule (already known pregame, no leakage) |
| `home/away_qb_change` | Mode of each team's last 3 games' starting QB ID (ties broken toward the most recent), compared to the current game's QB ID; 1 if different, 0 if same |
| `home/away_qb_rating_last5` | Passer rating (standard NFL formula: 4 components from completions/attempts/yards/TDs/INTs, each clamped 0–2.375) fetched via `import_weekly_data()`, falling back to `import_ngs_data('passing')` for seasons not yet published (2025) — then `shift(1).rolling(5).mean()` per QB |
| `home/away_turnover_margin_last5` | `import_pbp_data()`; a play is a turnover if `interception==1` or `fumble_lost==1`; giveaways = turnovers on `posteam`, takeaways = on `defteam`; margin = takeaways − giveaways, rolled. **Not used in the final model** (measurably hurt test accuracy) |
| `home/away_pts_scored_last5_sos_adj` | For each of a team's last 5 games, look up that opponent's own `pts_allowed_last5` at the time (already leak-free); `adjusted = pts_scored_last5 × (league_avg_pts_allowed / avg_opponent_def_strength)`. **Not used in the final model** (also hurt performance) |
| `div_game` | Direct from schedule |
| `home/away_starters_out` | See below |
| `home/away_qb_out` | See below |

### Team-injury features (`starters_out`, `qb_out`)

1. `import_snap_counts()` for every position; `effective_pct = max(offense_pct, defense_pct)` per
   player-game.
2. `is_starter_entering` = that player's own `shift(1).rolling(5, min_periods=1).mean()` of
   `effective_pct` ≥ 0.60 — grouped by **(normalized name, team)**, not name alone. (Found a real
   bug: two actual NFL players are both named "Lamar Jackson" — the Ravens QB and a Bears CB —
   which corrupted both players' rolling history when grouped by name only.)
3. Since an injured player has no snap-count row for the current week by definition, each team's
   presumed starting roster entering a game is built via `pd.merge_asof` (backward direction,
   `allow_exact_matches=True`) — matching to the player's own row if they played, or their most
   recent prior appearance if they didn't.
4. `starters_out` = count of presumed starters whose `import_injuries()` `report_status` is
   Out/Doubtful that week (joined via normalized player name — no shared ID scheme between
   `snap_counts`, which uses PFR IDs, and `injuries`, which uses gsis IDs).
5. `qb_out` = same logic restricted to `position == 'QB'`; if multiple QBs qualify as "presumed
   starter," the one with the most recent underlying appearance wins the tiebreak.

## 4. Game-winner model — [src/train_model.py](src/train_model.py)

- **Features (final, 15)**: `home/away_pts_scored_last5`, `pts_allowed_last5`, `win_streak`,
  `rest_advantage`, `is_dome`, `qb_change`, `qb_rating_last5`, `div_game`, `starters_out`.
  (`turnover_margin_last5`, `sos_adj`, and `qb_out` were tested and explicitly excluded — see
  Section 3 and the blend/ablation history below.)
- **Logistic Regression**: `StandardScaler` + `LogisticRegression`, trained on 2022–2024, tested on
  2025. Standardized specifically so coefficients are comparable as "feature importance" (raw
  features span wildly different scales).
- **XGBoost**: `n_estimators=100, max_depth=3, learning_rate=0.05`, trained on 2022–2023 only with
  2024 as an early-stopping validation set (`early_stopping_rounds=10`) — an earlier 200-tree,
  no-validation version was badly overfit (0.56 test accuracy vs. 0.63 after fixing).
  `scale_pos_weight` computed from the train set's actual class balance to correct a home-win
  prediction bias found in the logistic regression's confusion matrix.
- **Market baseline**: American moneyline → implied probability
  (`ml<0: -ml/(-ml+100)` else `100/(ml+100)`), then de-vigged by normalizing home+away raw
  probabilities to sum to 1.
- **Blend test**: `weight × model_prob + (1−weight) × market_prob` at weights 0.2/0.3/0.4/0.5.
  Weight=0.2 beat pure market on both accuracy and log loss on the 2025 test season, but a
  robustness check (retrain on 2022–2023, test on 2024) only replicated the accuracy edge, not the
  log-loss one — logged as a real, only-partial finding, not oversold.
- **Subgroup / disagreement analysis**: accuracy broken out by QB-change games, divisional games,
  dome games, and rest mismatches (model never found a specific edge over market in any bucket,
  both seasons); games sorted by `|model_prob − market_prob|` to find the biggest disagreements
  (market won 11 of the top 15 in 2025); "confident-market-only" upset filter using
  `market_prob` outside (0.40, 0.60).

## 5. Margin model — [src/train_margin_model.py](src/train_margin_model.py)

- Same 15-feature set and train/test split, `LinearRegression` + `XGBRegressor` predicting
  `margin = home_score − away_score`.
- **Market baseline**: `spread_line` directly — verified its sign convention (positive = home
  favored) against `home_moneyline`/`away_moneyline` before using it as a margin prediction.
- Signed-error distribution computed (`predicted − actual`) to check for directional bias (small,
  found a mild tendency to under-predict home blowouts specifically).

## 6. Player-prop pipelines

All four mirror the same shape: starters-only filter via `import_snap_counts()`, target from
`import_weekly_data()` (NGS fallback for 2025), rolling own-stat features, an opponent-defense-rank
feature, and game context (`spread_line`/`total_line`/`is_dome`/`rest_advantage`).

### QB passing — [src/player_features.py](src/player_features.py) → [data/processed/qb_passing_features.csv](data/processed/qb_passing_features.csv)
- **Starters**: `offense_pct ≥ 0.60`, joined to weekly stats by normalized name + team + week.
- **Target**: `passing_yards`.
- **`qb_pass_yards_last5` / `qb_pass_attempts_last5`**: own rolling average.
- **`opponent_pass_defense_rank`**: team pass-yards-allowed (summed across *all* opposing QBs that
  week, not just starters), rolled, then ranked 1 (stingiest) to 32 across the league at each
  weekly snapshot.
- **`qb_injury_status`**: 1 if `import_injuries()` status is Questionable/Doubtful/Out.
- **`top_target_availability`**: each team's WR/TE with the highest `receptions_last5` entering a
  *specific* game is looked up by shifting one game back on the team's own game log (so a
  traded-away or injured presumed-#1 receiver still surfaces); flag 1 if that specific player is
  Out/Doubtful (`import_injuries()`) or was traded that week (`import_weekly_rosters()`
  week-over-week team change).

### RB rushing — [src/rb_features.py](src/rb_features.py) → [data/processed/rb_rushing_features.csv](data/processed/rb_rushing_features.csv)
Same shape; target `rushing_yards`; `rb_rush_yards_last5`/`rb_rush_attempts_last5`;
`opponent_rush_defense_rank` (same ranking method, rushing yards allowed); `rb_injury_status`.
No `top_target`-equivalent (a starting RB generally *is* the backfield's workload).

### WR receiving — [src/wr_features.py](src/wr_features.py) → [data/processed/wr_receiving_features.csv](data/processed/wr_receiving_features.csv)
Target `receiving_yards`; `wr_rec_yards_last5`/`wr_receptions_last5`/`wr_targets_last5`;
`opponent_pass_defense_rank` recomputed fresh from QB stats (each pipeline script is
self-contained); **`target_share_last5`** = this player's *summed* targets over their last 5 games
÷ their team's *summed* total targets (all pass-catchers, not just starters) over the same window —
a ratio of sums, more stable than averaging per-game ratios; `wr_injury_status`.

### Pass rush — [src/def_pass_rush_features.py](src/def_pass_rush_features.py) → [data/processed/pass_rush_features.csv](data/processed/pass_rush_features.csv)
- **Stats source**: `import_weekly_pfr('def', ...)` (has `def_sacks`/`def_pressures` at weekly
  granularity — works for 2025, unlike `import_weekly_data()`).
- **Position**: joined to `snap_counts` via `pfr_player_id` (both PFR-based, no name-matching
  needed here), normalized into `edge` (DE/OLB/LB and side variants) vs. `interior` (DT/DL/NT).
- `def_sacks_last5`/`def_pressures_last5`: own rolling average.
- `opponent_oline_sacks_allowed_last5`: summed across *all* defenders who sacked that offense
  (not just starters), rolled.

**Data-quality bugs found and fixed across these pipelines**: a duplicate-column crash from
`weekly_data`'s pre-existing `player_name`/`attempts` columns colliding with renamed fields; a
single NGS row with `receiving_yards = NaN` despite 0 receptions (filled with 0, since that's
unambiguous); NGS's `week == 0` row (season totals, not a game) silently corrupting rolling
averages until explicitly filtered out everywhere NGS is used as a fallback.

## 7. Player-prop models

[src/train_player_model.py](src/train_player_model.py) (QB), [src/train_rb_model.py](src/train_rb_model.py) (RB),
[src/train_wr_model.py](src/train_wr_model.py) (WR), [src/train_pass_rush_model.py](src/train_pass_rush_model.py)
(edge pass-rushers) — all `LinearRegression` + `XGBRegressor` on 2022–2024 train / 2025 test,
except pass-rush, which is reframed as **binary classification** (`recorded_sack = def_sacks > 0`)
since ~70% of edge-rusher games record zero sacks — tested against an "always predict no sack"
baseline (models beat it by only ~1 point of accuracy, honestly reported as weak).

## 8. Live weekly prediction — [src/predict_week.py](src/predict_week.py)

1. Retrains all 5 models (win/loss, margin, QB/RB/WR yardage — linear only, for production
   simplicity) on **all** 2022–2025 data, no holdout.
2. Pulls the live schedule (`import_schedules([2026])`), finds the earliest week with
   `home_score` still null.
3. Extends the real 2022–2025 schedule with that future week and reruns `features.py`'s exact
   rolling-feature logic against the combined data — so the unplayed week's "entering this game"
   values are real trailing stats through each team's last actual game.
4. **Current starters** identified via `import_depth_charts()` (latest snapshot, `pos_rank == 1`
   per position), cross-checked against `import_injuries()` when available (it isn't yet for a
   season that hasn't started — the script degrades gracefully and says so).
5. Each identified starter's rolling stats are read from their own historical row sequence
   (tail-5 mean — the same formula as the shift-before-rolling functions, just evaluated one step
   past their last recorded game).

**Three real bugs found and fixed here**: `import_pbp_data()` silently returns a malformed frame
(prints its own error) instead of raising on a 404, which crashed the turnover-fetch step — fixed
with an explicit column check; the schedule's own `home_qb_id`/`away_qb_id` are null for every
unplayed game (only populated after a game happens), which made `qb_change` spuriously `True` for
every team and `qb_rating_last5` null for everyone — fixed by overwriting both using the
depth-chart-identified starter instead; and player-name suffix mismatches (e.g. "Travis Etienne
Jr." vs. stored "Travis Etienne") caused real veterans to look like rookies with no history — fixed
with the same name-normalization helper used elsewhere.

Output: [data/processed/current_week_predictions.json](data/processed/current_week_predictions.json)
— one object per game with `model_win_prob`, `market_win_prob`, `model_margin`, `market_spread`,
raw `home/away_moneyline`, `starters_out`, and each team's projected QB/RB/WR1 with predicted
yardage.

## 9. Dashboard — [src/dashboard.html](src/dashboard.html)

Single static HTML file (embedded CSS/JS, no build step) reading the JSON above via `fetch()`
(requires an HTTP server — `file://` is blocked by the browser's fetch sandboxing).

### Over/under yardage lines

**Not** a plain standard-deviation/normal-distribution formula — checked and rejected: yardage is
bounded at 0 and right-skewed, so a symmetric std-based cutoff badly overstates real safety margins
(a "90%-confidence" std cutoff for WR yards came out negative). Instead, for each position, the real
historical Nth/(100−N)th percentile gap from the mean is measured and applied to each player's own
prediction: `over = predicted − (mean − Nth percentile)`, `under = predicted + ((100−N)th
percentile − mean)`. Three tiers: **Safe (~80%)**, **Risky (~65%)**, **Most Risky (~55%, one side
only** — whichever of over/under has the smaller gap at that tier wins).

### The 9 sort modes (exact formulas)

| Sort | Formula |
|---|---|
| Date | `game_date` ascending |
| Model Confidence | `\|model_win_prob − 0.5\| × 2` |
| Market Confidence | `\|market_win_prob − 0.5\| × 2` |
| Highest Agreement | `min(modelConf, marketConf)` **only if both favor the same team** (direction mismatch excluded entirely, not just penalized) — the weaker of the two convictions is the bottleneck |
| Highest Disagreement | `\|model_win_prob − market_win_prob\|` (a direction flip between sources produces the largest possible value here) |
| Best Value (EV) | For each side: `EV = p × b − (1−p)`, where `p` is the model's probability for that side and `b` is that side's real payout from the moneyline (`ml>0: ml/100`, else `100/|ml|`); best of home/away EV wins |
| Hidden Edge | `modelConfidence × (1 − marketConfidence)` — rewards a confident model specifically when the market has no opinion |
| Smart Value | `EV × (1 − marketConfidence)` — EV discounted by how confident the market already is, since the biggest raw-EV numbers mostly come from the model overriding a confident market (shown in this project's own disagreement testing to be the model's weak spot) |
| **Combined (Model+Market)** ★ recommended | Model and market probabilities blended into one number — **not** multiplied with the payout, kept as pure reasoning, with the market's real moneyline shown separately as payout info only. Blend weight is dynamic: `modelWeight = 0.5 × (1 − marketConfidence)` (market's weight never drops below 50%, rising to 100% as it gets more confident) |

The last three sorts were built iteratively in direct response to spotting a real flaw in the
previous one: raw EV rewards fighting a confident market (which the model loses more often, per
this project's own disagreement testing); Hidden Edge fixes that but never checks whether a bet is
actually profitable; Smart Value fixes that by multiplying the two together, which then conflates
"how confident should I be" with "what's the payout" into one number; Combined keeps those two
questions separate, as requested, with a weighting scheme designed so the model only gets real say
specifically when the market is uncertain.

## Known limitations (stated honestly throughout, not just here)

- Market beats both the win/loss and margin models on average; the model contributes a small,
  inconsistent edge at best.
- `starters_out`/injury-based features are `0` for the live 2026 predictions purely because
  `import_injuries()`/`import_snap_counts()` don't have 2026 data yet — not a claim nobody's hurt.
- Rolling features use position-wide or team-wide historical variance, not each individual
  player's own consistency.
- Pass-rush sack prediction barely beats a naive "always predict no sack" baseline.
