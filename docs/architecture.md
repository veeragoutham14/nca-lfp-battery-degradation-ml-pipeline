# Platform Architecture

## Design Goals

1. Keep analytical algorithms independent from UI and orchestration.
2. Give every model execution a stable run ID and traceable artifacts.
3. Normalize output for dashboards without flattening source evidence.
4. Add new battery analyses through versioned plugins.
5. Preserve current `Veera_SOH` results and commands during migration.

## Ownership Boundaries

| Layer | Owns | Does not own |
|---|---|---|
| Source project | signal processing, event building, SOH mathematics, Parquet production | dashboard state, central run catalog |
| Battery profile | chemistry, product variant, and per-serial modelling assumptions | measured capacity or model output |
| Plugin adapter | parameter contract, command construction, output family | model implementation |
| Catalog | run identity, lineage, datasets, artifacts, job status | source Parquet contents |
| Data service | stable chart-oriented response shapes | model fitting |
| API | access, validation, orchestration endpoints | visual rendering |
| Dashboard | selection, comparison, charts, job controls | filesystem discovery, pipeline logic |

## Device Roles

`config/battery_profiles.json` is the authority for workflow eligibility. The
API validates roles before queuing a command, so a caller cannot bypass the UI.

```text
field_tester:    raw -> classified -> events -> fixed-slope verification
full_cycle_test: raw -> classified -> events -> capacity trend -> SOH model
```

Both paths compute cumulative discharge from every discharging operating event.
Only the controlled full-cycle path creates capacity labels suitable for model
fitting. The field path treats every supplied slope as fixed and reports its
agreement with BMS SOH; verification ranking is not model training.

## Run Identity

Imported run IDs are deterministic:

```text
<pipeline-key>:<serial>:<source-run-timestamp>
```

Example:

```text
veera.full_cycle_soh.v1:100000017:20260703T145154Z
```

The source run directory and summary fingerprint are retained. Re-scanning the
same source replaces its catalog bundle without duplicating it.

## Data Contracts

The catalog stores metadata only. Parquet remains the analytical data plane.
The API reads registered artifacts and returns normalized structures:

- history: timestamp, capacity or BMS SOH, and cumulative Ah;
- forecast: scenario, timestamp, SOH, model interval, usage rate, cumulative Ah;
- validation: actual and predicted SOH/capacity plus errors.

This keeps the dashboard contract independent from the source file layout while
the analytical artifacts remain independently auditable.

## Worker Model

The local orchestrator uses a bounded thread pool. Each command runs inside a
detached worker process that survives an API restart. Each job records:

- plugin key and validated request;
- exact command;
- timestamps and state;
- worker PID and atomic result path;
- return code and logs;
- resulting catalog run ID.

On API restart, queued jobs are launched and running workers are reattached by
PID. A worker writes `result.json` atomically before it exits, allowing the new
API process to finalize the job without guessing from logs. A job becomes
`abandoned` only when its worker is gone and no valid result was written.

## Adding Anomaly Detection

An anomaly pipeline should be added as a new plugin, for example:

```text
veera.telemetry_anomaly.v1
```

Its run can register episode, score, and explanation Parquet artifacts. New API
series should be added under an anomaly-specific service; the catalog tables and
job runner remain unchanged.

## Roadmap

1. Current: local catalog, authenticated API, role-aware preprocessing,
   full-cycle SOH, fixed-slope verification, and React workspace.
2. Add explicit dataset-version records for raw and standardized telemetry.
3. Add anomaly plugin and cross-run battery timeline.
4. Add model registry records and approved deployment aliases.
5. Replace detached local workers with an external queue when multi-host or
   concurrent-user operation requires it.
6. Replace DuckDB metadata with PostgreSQL only when multi-user writes require it.
