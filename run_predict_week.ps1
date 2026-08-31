# Wrapper script for the daily scheduled task. Runs predict_week.py with the
# project's venv and appends timestamped output to a log file, so a failed
# run (e.g. a network hiccup, an nfl_data_py source going down) is visible
# without needing to babysit the task.

$root = "C:\Users\pmahe\OneDrive\Documents\nfl-model"
$logFile = Join-Path $root "predict_week_log.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Add-Content -Path $logFile -Value "`n=== Run started: $timestamp ==="
try {
    & "$root\venv\Scripts\python.exe" "$root\src\predict_week.py" *>> $logFile
    Add-Content -Path $logFile -Value "=== Run finished OK: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
} catch {
    Add-Content -Path $logFile -Value "=== Run FAILED: $_ ==="
}
