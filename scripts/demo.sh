#!/usr/bin/env bash
# Generate synthetic telemetry, run the SOH brain, and open the dashboard.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

export BATTRYML_SOURCE_ROOT="$ROOT/brain"
export BATTRYML_SOURCE_PYTHON="$PY"
export BATTRYML_AUTH_ENABLED=false
export BATTRYML_CATALOG_PATH="$ROOT/var/battryml.duckdb"
export BATTRYML_JOB_ROOT="$ROOT/var/jobs"
export PYTHONPATH="$ROOT/brain${PYTHONPATH:+:$PYTHONPATH}"

echo "Generating synthetic 1-second telemetry..."
"$PY" "$ROOT/scripts/generate_synthetic_telemetry.py" \
  --output "$ROOT/brain/data/raw_parquet"

echo "Running the brain on the synthetic lab reference and two field devices..."
(
  cd "$ROOT/brain"
  "$PY" -m src.build_events --serial 100000017
  "$PY" -m src.discharge_cycle_builder --serial 100000017
  "$PY" -m src.capacity_ml \
    --serial 100000017 \
    --nominal-capacity-ah 56 \
    --future-days 3650 \
    --step-days 30
  "$PY" -m src.plot_capacity_forecast_report_image \
    --serial 100000017 \
    --title "Synthetic data — SOH forecast"
  "$PY" -m src.build_events --serial 100000001
  "$PY" -m src.build_events --serial 100000020
)

echo "Exporting demo figures..."
"$PY" "$ROOT/scripts/export_readme_figures.py" \
  --output "$ROOT/docs/images/generated" \
  || echo "Figure export skipped. The dashboard still starts."

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "$ROOT/frontend" && npm ci)
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then kill "$API_PID" 2>/dev/null || true; fi
  if [[ -n "${UI_PID:-}" ]]; then kill "$UI_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

echo "Starting API on http://127.0.0.1:8000"
"$PY" -m battryml.cli serve --host 127.0.0.1 --port 8000 &
API_PID=$!

echo "Starting dashboard on http://127.0.0.1:5173"
(cd "$ROOT/frontend" && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort) &
UI_PID=$!

echo
echo "Synthetic demo is up."
echo "  Dashboard  http://127.0.0.1:5173"
echo "  API        http://127.0.0.1:8000/api/health"
echo "Press Ctrl+C to stop."
wait
