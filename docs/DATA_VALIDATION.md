# Prepare Analysis Data

## Question Answered

The Stage 1 validation gate answers one question:

> Is the locally stored evidence for this battery complete and internally
> consistent enough for the selected analysis?

It does not diagnose battery health and it does not prove that a BMS sensor is
physically calibrated. It prevents downstream analysis from quietly using
missing, stale, contradictory, or structurally incomplete inputs.

## Operator Workflow

1. Open **Extraction** and select the device.
2. Import new telemetry only when the local data range must be extended.
3. Use **Prepare or update analysis data** to run the source audit, update the
   hierarchy-preserving five-minute history, and validate the resulting local
   evidence in dependency order.
4. Use **Validate existing data** when no database access or import is needed.
   If the input fingerprint is unchanged, the previous validation is reused.
5. Read purpose-specific readiness before starting capacity, exposure, or
   voltage-group analysis.

## Purpose-Specific Readiness

**Capacity & BMS SOH** requires one-second UTC SOC, SOH, signed current, and pack
voltage. Current direction must not conflict with observed SOC movement.

**Monthly exposure** requires those one-second operating signals plus the
hierarchy-preserving five-minute electrothermal history.

**Voltage-group health** requires every expected voltage group to be mapped
and the latest electrothermal source audit to pass.

NCA and LFP artifacts are stored separately. Module count controls topology:

- each module has two packs;
- each pack has 14 voltage-group channels;
- each pack has four pack-temperature sensors.

A two-module device therefore expects 56 voltage groups and 16 temperature
sensors. Temperature channels are BMS pack-sensor measurements, not cell-core
temperatures.

## Quality Metrics

- **Calendar coverage:** distinct imported timestamps divided by elapsed wall
  clock seconds between the first and last timestamp. Planned downtime and
  telemetry gaps both reduce this value, so it is a data-availability metric,
  not battery performance.
- **Monthly observed-span coverage:** the same calculation within each UTC
  month, bounded by that month's first and last observation.
- **Maximum gap:** largest interval between distinct one-second timestamps.
- **Duplicates:** rows sharing an identical timestamp. They are reported and
  collapsed for coverage calculations.
- **Invalid SOC/SOH:** BMS values outside 0-100%.
- **Suspicious temperature:** five-minute windows outside the source contract's
  broad plausibility range of -40 to 100 C. These are data-quality findings,
  not manufacturer safety thresholds.
- **Stale:** the final local timestamp is more than 30 days behind validation
  time.

Current sign is checked in five-minute evidence windows. A window is usable
when mean absolute current is at least 1 A and absolute SOC movement is at
least 0.1 percentage points. The configured convention is verified when at
least 100 windows exist and at least 80% agree that positive current raises SOC
and negative current lowers SOC. Fewer windows or mixed agreement is reported
as inconclusive; it is never silently treated as verified.

The **BMS lifecycle counter** corresponds to the requested `sys.cycles`
concept. It remains missing unless a recognized source column was imported.
Even when present, its semantics are explicitly unverified until documented by
the BMS owner.

Resistance-reference metadata is displayed but disabled in Stage 1. It is not
used for health or degradation claims.

## Stored Artifacts

```text
Veera_SOH/data/processed/data_validation/
  chemistry=<NCA|LFP>/serial=<serial>/validation_run=<UTC stamp>/
```

- `validation_summary.json`: readiness, headline quality, provenance limits;
- `signal_inventory.parquet`: required signal availability and semantics;
- `coverage_by_month.parquet`: UTC monthly coverage and evidence layers;
- `topology_inventory.parquet`: module, pack, voltage-group, and sensor map;
- `quality_findings.parquet`: compact operator findings;
- `input_manifest.json`: paths, sizes, modification times, validator version,
  and deterministic SHA-256 fingerprint;
- `model_summary.json`: catalog-compatible copy of the validation summary.

Raw and feature Parquet files are never rewritten by validation. DuckDB reads
only required columns, pushes aggregation into Parquet scans, and the manifest
prevents repeated scans when inputs are unchanged.
