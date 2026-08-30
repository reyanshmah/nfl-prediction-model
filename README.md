# NFL Prediction Model

A machine learning pipeline that predicts NFL game outcomes, margins, and individual player stats (QB/RB/WR yardage), built and rigorously backtested against real betting market odds across the 2022–2025 seasons.

## What this does
- Predicts game winners (win probability) and point margins
- Predicts QB passing yards, RB rushing yards, and WR receiving yards for starting players
- Compares every prediction against real closing betting lines (moneyline, spread) to honestly evaluate performance — not just accuracy in isolation
- Includes a live weekly prediction script that identifies current starters (handling injuries/backups) and generates upcoming-week predictions

## Key findings
- **Game winner model:** 63.3% accuracy vs. the market's 66.0% — close, but no reliable edge found even after testing across multiple seasons and game subgroups (QB changes, divisional games, rest mismatches, etc.)
- **Margin prediction:** ~9.9 points MAE vs. the market's ~9.5 points MAE
- **Player props:** QB yardage is the most predictable (MAE ~54 yards); RB/WR yardage is noisier (~28 yards MAE, driven by game-script variance); sacks are mostly random (barely beats a naive "no sack" baseline), but pass-rush pressure rate is a real, stable signal
- Several tested features (turnover margin, strength-of-schedule adjustment) were found to add noise rather than signal and were correctly excluded after testing
- Team-level injury context (starters out) adds a small, genuine improvement to the win/loss model

## Pipeline
data/raw/ -> src/features.py (leak-free rolling features)
          -> src/train_model.py, src/train_player_model.py, src/train_rb_model.py, src/train_wr_model.py, src/train_margin_model.py
          -> src/predict_week.py (live weekly predictions, current starter detection)
          -> dashboard

## Setup
Note: this machine's default python is 3.14, which is too new for nfl_data_py's pinned numpy<2.0/pandas<2.0 requirement. The project venv was built with Python 3.12 instead, using newer pandas/numpy that nfl_data_py works fine with despite its stale metadata pins.

py -3.12 -m venv venv
venv\Scripts\activate      (Windows)
pip install -r requirements.txt

If Python 3.12 isn't available, install pandas, numpy, and scikit-learn first, then install nfl_data_py with --no-deps.

## Dependencies
- nfl_data_py - NFL play-by-play, schedule, roster, injury, and depth chart data
- pandas / numpy - data manipulation
- scikit-learn - modeling
- xgboost - gradient-boosted models
