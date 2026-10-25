# Workflow Freshness And Dependency Locking

BatteryML distinguishes an existing historical output from an output that is
current for the latest imported telemetry. Importing new telemetry changes the
raw filesystem watermark for only that serial. Direct consumers become
`stale`, and downstream consumers become `blocked` until their prerequisites
are updated.

## Stage States

- `missing`: no source exists yet;
- `ready`: all prerequisites are current and the stage has no output;
- `current`: output is at least as new as every prerequisite;
- `stale`: output exists but a prerequisite has a newer watermark;
- `blocked`: a prerequisite is missing, stale, failed, or blocked;
- `failed`: the latest quality gate did not pass;
- `running`: supplied by the active job state in the UI.

Historical run folders are never deleted by freshness evaluation. They remain
available for inspection but do not satisfy a current prerequisite.

## Current Watermark Contract

Version 1 compares persisted filesystem modification watermarks from Parquet,
`model_summary.json`, and `dataset_manifest.json` artifacts. It reads file
metadata only and never scans telemetry rows during a page refresh. This keeps
the check bounded and makes an appended raw part invalidate old processing.

The backend is authoritative. It returns the state, reason, output watermark,
and required upstream watermark. Job submission repeats the same prerequisite
check, so a browser cannot bypass a lock.

Filesystem watermarks detect normal BatteryML imports and reruns, but they are
not content hashes. Manually copying files while preserving timestamps can
defeat this version. A future manifest revision may add content fingerprints
and explicit processed-through timestamps without changing the API state model.

## Expected Update Sequence

After importing new telemetry:

1. direct raw consumers show **Update required**;
2. their descendants show **Waiting** with the stale prerequisite named;
3. updating a direct consumer makes its immediate child runnable;
4. continue until the desired reports are current.

This is a dependency cascade, not deletion or automatic recomputation.

## Guided Workflows

The Extraction page offers **Prepare or update analysis data**. One persisted
job runs the required source stages in order:

1. optionally import newly available raw telemetry;
2. qualify the BMS SOH signal;
3. audit hierarchical voltage and temperature source channels;
4. build or incrementally update the five-minute hierarchical history.

The Machine Learning page exposes one ordered operator path after events and
the transferred-slope check: **Build electrothermal features**, **Validate
analysis inputs**, **Run capacity and SOH investigation**, then **Analyze
charging and seasonal stress**. Validation internally refreshes the BMS SOH
signal audit; there is no separate operator-facing SOH-audit prerequisite.
Seasonal analysis is a separate job and reads the existing joined
`bms_hierarchical_features` data without changing its report contract.

The preparation workflow discovers raw timestamp bounds from Parquet row-group
metadata. If a five-minute history already exists, it starts from the latest
recorded feature end rather than rebuilding the full history. This is fast for
normal appended imports, but it does not silently repair an older internal gap;
such a gap must be rebuilt with the individual feature-stage control.
