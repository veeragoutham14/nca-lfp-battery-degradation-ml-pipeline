# BattryML

For installation on another computer, environment recreation, data transfer,
and startup instructions, see [docs/architecture.md](docs/architecture.md).

BattryML is a local modular platform for battery analytics. It imports raw
telemetry through `Veera_SOH`, catalogs existing analytical outputs, exposes
normalized APIs, runs registered pipeline plugins, and presents SOH forecasts
and validation evidence in a role-aware React/ECharts workspace.

The existing model code remains authoritative. BattryML invokes its public CLI
modules and records the resulting runs; it does not copy the degradation logic.

## Architecture

```text
PostgreSQL/TimescaleDB -- ingestion plugin --> Veera_SOH raw Parquet
                                               |
Veera_SOH analytical Parquet + JSON -----------+
                                               v
                                      DuckDB run catalog
                                               |
                                               v
                                           FastAPI API
                                               |
                                               v
                                      React + Apache ECharts
```

Main components:

- `backend/battryml/catalog`: batteries, runs, datasets, artifacts, and jobs.
- `backend/battryml/plugins`: versioned adapters around analytical pipelines.
- `backend/battryml/data`: normalized forecast and validation series.
- `backend/battryml/api`: local API and artifact delivery.
- `frontend`: sign-in, extraction library, role-aware workflows, and reports.

See [architecture.md](docs/architecture.md) for ownership boundaries and the
incremental roadmap.

## Setup

The checked-in `.env.example` already points at the current local source project.
The local `.env` has the same non-secret defaults and is ignored by Git.
`BATTRYML_WORKER_THREADS` controls concurrent extraction jobs, while
`BATTRYML_ML_WORKER_THREADS` controls the independent preprocessing and ML pool.

Node.js 20 or newer is required for the dashboard. Launch scripts automatically
use `.tools/node` when a portable local runtime is present; otherwise they use
the system `PATH`.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Set-Location frontend
npm install
Set-Location ..
```

Git Bash:

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install -e '.[dev]'
cd frontend
npm install
cd ..
```

## Import Existing Runs

```bash
source .venv/Scripts/activate
battryml scan
battryml status
```

The scan is idempotent. It discovers full-cycle model runs, registers Parquet
schemas and row counts, and links matching PNG/HTML reports.

## Run Locally

Start the API in one terminal:

```bash
source .venv/Scripts/activate
battryml serve
```

Use `battryml serve --reload` only while editing backend code. A normal server
restart no longer loses active jobs: detached workers keep running and the API
reattaches using their persisted PID and atomic `result.json` record.

Start the dashboard in another terminal:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. API documentation is available at
`http://127.0.0.1:8000/docs`.

After login, select the **NCA** or **LFP** workspace. Extraction libraries,
device workflows, jobs, cell-signal readiness, and anomaly overviews are scoped
to that chemistry. Devices are ordered by ascending verified module count and
then serial number. A new serial can be imported directly from either chemistry
workspace; it remains role-unassigned until its durable hardware variant is added
to `config/battery_profiles.json`.

The local default login is `admin` / `battryml`. Set
`BATTRYML_AUTH_USERNAME`, `BATTRYML_AUTH_PASSWORD`, and a long random
`BATTRYML_AUTH_SECRET` in `.env` before sharing the service with another user.

## Import Raw Telemetry

Open **Extraction** and enter or choose a serial. Start and end are optional: a
blank start resolves to the first database timestamp and a blank end resolves to
the last database timestamp. For SOC, BMS SOH, stack voltage, BMS temperature
summaries, and stack current, choose **Do not import**, **1 second**, **5 minutes**,
or **Both**. UTC timestamp and serial are included automatically.

The backend invokes `src.extract_configurable_telemetry`. One-second rows are
written as month-partitioned Parquet under `RAW_PARQUET_DIR` for compatibility
with existing preprocessing. Database-aggregated five-minute rows are stored
separately under `Veera_SOH/data/telemetry_5min`. Every request writes a JSON
manifest under `Veera_SOH/data/import_manifests` containing chemistry, resolved
range, signal selections, physical source columns, aggregation rules, row counts,
and output files. The chemistry-specific source mappings and UI defaults are in
`config/telemetry_import_profiles.json`.

This general importer uses BMS stack voltage and BMS temperature summaries.
Per-cell voltage and per-pack temperature channels remain in the hierarchical
electrothermal extraction workflow. See
`docs/CONFIGURABLE_TELEMETRY_IMPORT.md` for the complete contract.

Database credentials are never sent to the browser. The extractor reads them
from the source project's environment. Imports run as persisted jobs, and their
status appears beside the extraction tool.

Database-backed stages (`raw_telemetry_import`, electrothermal source audit, and
hierarchical feature extraction) run only when `DB_HOST:DB_PORT` from
`Veera_SOH/.env` is reachable. The header reports **Database offline** and the
API rejects the request before queueing a job when the endpoint is unavailable.
Local Parquet preprocessing and analysis remain available offline. Prefer a
stable database hostname over a laptop's changing Tailscale address.

Import and verify module-count metadata for every serial configured in
`config/battery_profiles.json`:

```powershell
.\scripts\import_module_counts.ps1
```

The script reads the latest 1,000 non-null `batt_modulecount` samples per serial,
requires a positive integer dominant value with at least 95% agreement, and writes the compact
snapshot to `Veera_SOH/data/processed/device_metadata/module_counts.parquet`.
Refresh the Extraction page to see each serial as verified, missing, invalid, or unstable.
Verified counts above two derive nominal capacity as `module_count * 32 Ah`; two-module
devices retain the synthetic 64 Ah nominal capacity. Transferred slope 0.12 was
learned from a two-module reference device.

## Battery Profiles

Device-specific assumptions live in `config/battery_profiles.json`. Every
configured device has a role enforced by both the API and the frontend:

Device names use two layers. `display_name` in `config/battery_profiles.json`
is the source-controlled default. The pencil button beside the device name in
the Machine Learning workspace creates a local override in the DuckDB catalog;
the reset button removes that override and restores the configured default.
Serial numbers remain the immutable identity used by telemetry, jobs, folders,
and analysis joins. Renaming a device therefore never renames or moves data.

- `field_tester`: normal home operation. Nominal capacity is `64 Ah` for the
  configured NCA variant. These devices can build row classifications and
  operating events. NCA field devices can then verify user-supplied loss slopes
  against BMS SOH using the actual cumulative Ah from all discharge events. No
  slope is fitted, and that NCA reference slope is not exposed to LFP devices.
- `full_cycle_test`: controlled 100-to-0 cycling. These devices can additionally
  build capacity trends and train the full-cycle capacity degradation model.

The 19 new home-storage devices and serial `100000011` are field testers.
Serial `100000017` is the full-cycle reference device; its nominal capacity can
still be derived from its first qualified cycle.

Windows launchers are also provided:

```text
run_backend.bat
run_backend_dev.bat
run_frontend.bat
```

`run_backend.bat` is the stable launcher for imports and model runs.
`run_backend_dev.bat` enables hot reload for backend development.

After `npm run build`, FastAPI also serves the compiled dashboard from port 8000.

## Pipeline Plugins

The registered workflow plugins are:

- `battryml.soh_signal_audit.v1` -> native daily BMS SOH quality and
  recalibration-event audit
- `veera.raw_telemetry_import.v1` -> `python -m src.extract_db`
- `veera.bms_electrothermal_source_audit.v1` ->
  `python -m src.bms_electrothermal_source_audit`
- `veera.bms_hierarchical_features.v1` ->
  `python -m src.bms_hierarchical_features`
- `veera.hierarchical_voltage_anomaly.v2` ->
  `python -m src.hierarchical_voltage_anomaly`
- `veera.seasonal_charge_analysis.v1` ->
  `python -m src.seasonal_charge_analysis`
- `veera.classify_rows.v1` -> `python -m src.classify_rows`
- `veera.build_events.v1` -> `python -m src.build_events_from_classified`
- `veera.capacity_trend.v1` -> `python -m src.discharge_cycle_builder`
- `veera.full_cycle_soh.v1` -> `python -m src.capacity_ml`
- `veera.fixed_slope_verification.v1` ->
  `python -m src.fixed_slope_verification`

The command modules remain in `Veera_SOH`; BattryML invokes them through their
CLI contracts instead of duplicating preprocessing or model code. Capacity
trend and full-cycle SOH are restricted to full-cycle devices. Fixed-slope
verification is restricted to field devices.

The BMS SOH audit is the first ML workspace stage. It detects stable steps,
recoveries, gaps, transient values, coverage loss, and cumulative charge/discharge.
It records BMS SOH as a noisy auxiliary label; even a flat signal is not promoted to
physical SOH ground truth without an independent capacity measurement.

After login, choose the NCA or LFP workspace. Device libraries, extraction jobs,
signal qualification, hierarchical features, anomaly results, and seasonal
analysis are filtered by the configured chemistry. Add LFP variants and serials
to `config/battery_profiles.json` before importing them; do not relabel unknown
devices merely to make them appear in the LFP workspace. Every device list is
ordered by verified module count, with unknown counts last.

The NCA workspace also exposes a chemistry-separated **NCA fleet capacity** view.
It combines the latest field-device fixed-slope verification runs, actual cumulative
discharge Ah, verified module metadata, EFC, and the controlled-device reference slope.
The displayed BMS-minus-reference difference measures agreement with two estimates;
it is not presented as independently measured physical capacity loss. The reference
slope can be changed interactively without fitting it to the field-device BMS values.

The **Cell signals** view implements the hierarchical electrothermal workflow.
The source audit checks one representative day directly in PostgreSQL and expects
28 voltage channels and 8 pack-temperature sensors per installed module. It gates
processing on schema presence, valid coverage, physical ranges, frozen channels,
duplicate timestamps, and agreement with BMS-reported voltage/temperature extrema.
Exact database names are retained alongside normalized module, pack, and channel
identities.

After the quality gate passes, the UI builds month-partitioned five-minute
operating-context, voltage-channel, temperature-sensor, and system feature files.
Anomaly v2 combines fixed engineering spread bands with physical pack-peer
residuals, low-rank reconstruction, EWMA context, and persistent episode grouping.
One-second BMS data is retained only around selected sustained episodes. These are
BMS-signal inconsistency candidates, not independently confirmed physical defects.
The retired `cell_signal_audit`, `cell_health`, and `cell_anomaly` pipelines are
not registered in BatteryML and cannot be launched from the API or dashboard.
Existing files in those folders are historical artifacts only.

The equivalent source commands are:

```powershell
python -m src.bms_electrothermal_source_audit `
  --serial 100000001 `
  --module-count 2

python -m src.bms_hierarchical_features `
  --serial 100000001 `
  --module-count 2 `
  --start "2023-08-01T00:00:00+00:00" `
  --end "2025-02-01T00:00:00+00:00" `
  --aggregation-minutes 5 `
  --compression zstd

python -m src.hierarchical_voltage_anomaly `
  --serial 100000001 `
  --maximum-evidence-episodes 50 `
  --compression zstd
```

Run the fleet audit from PowerShell with a comma-separated serial list:

```powershell
python -m battryml.analytics.fleet_soh_audit `
  --serials 100000001,100000002,100000004 `
  --raw-root "../brain\data\raw_parquet" `
  --output-root "data\processed\fleet_soh_audit" `
  --default-nominal-capacity-ah 64 `
  --workers 1 `
  --reuse-existing
```

Inspect a completed fleet run with:

```powershell
python scripts\summarize_fleet_soh_audit.py `
  data\processed\fleet_soh_audit\fleet_run=<timestamp>
```

See [ml_platform_roadmap.md](docs/ml_platform_roadmap.md) for the signal tiers,
label contract, NCA/LFP model ladder, and leakage-safe validation plan.
See
[README_SEASONAL_CHARGING_ANALYSIS.md](docs/README_SEASONAL_CHARGING_ANALYSIS.md)
for the charging-episode, daily stress, seasonal-profile, and BMS SOH-transition
data contracts and interpretation limits.

Jobs execute with the Python interpreter and working directory configured for
`Veera_SOH`. Standard output and errors are stored under `var/jobs/<job_id>/`.
Each durable worker also writes `result.json`; its PID and result path are stored
in the catalog so an API restart can recover the job. Successful jobs trigger a
catalog scan and link to the new analysis run.

## API Surface

```text
GET  /api/health
POST /api/auth/login
GET  /api/auth/me
POST /api/auth/logout
GET  /api/devices
GET  /api/devices/{serial}
PATCH /api/devices/{serial}/display-name
GET  /api/analytics/nca-capacity-overview
GET  /api/analytics/cell-signal-overview
GET  /api/batteries
GET  /api/runs
GET  /api/runs/{run_id}/summary
GET  /api/runs/{run_id}/series
GET  /api/runs/{run_id}/validation
GET  /api/runs/{run_id}/artifacts
GET  /api/runs/{run_id}/datasets
GET  /api/pipelines
POST /api/catalog/scan
POST /api/jobs
GET  /api/jobs
GET  /api/jobs/{job_id}/logs
```

## Tests

```bash
pytest
ruff check backend
cd frontend && npm run build
```

With the API and dashboard running, install the isolated QA browser once and
capture desktop/mobile screenshots with overflow and console-error checks:

```bash
cd frontend
npm run qa:install-browser
npm run qa:visual
```

Screenshots are written to `var/screenshots/` and are ignored by Git.

## Current Scope

This version is intentionally local and single-machine. DuckDB is the catalog,
a bounded thread pool is the worker, and signed local sessions protect the API.
Those are replaceable platform boundaries. The plugin and React API contracts do
not need to change when PostgreSQL, a durable queue, anomaly plugins, or remote
object storage are introduced later.
