# Configurable Telemetry Import

## Operator Workflow

1. Select the NCA or LFP workspace.
2. Open **Extraction**.
3. Select a configured battery or enter a new serial.
4. Choose **Capacity degradation** or **Custom telemetry**.
5. For custom telemetry, select the required signal groups and resolutions.
6. Review the per-dataset continuation plan and start the persisted import job.
7. Run **Validate imported data** before classification and event building.

Leave the date overrides blank for normal operation. Each selected dataset then
continues independently after its own local watermark; a new dataset begins at
the configured device start or the first database row. One-second data resumes
one second after its watermark. Five-minute data resumes at the next complete
five-minute bucket. An explicit start is clamped when it overlaps existing data.

## Signal Contract

**Capacity degradation** is a locked one-second contract:

- SOC
- BMS SOH
- signed stack current
- stack voltage

**Custom telemetry** imports only the explicitly selected groups:

| UI field | Initial physical source | Five-minute aggregation |
| --- | --- | --- |
| State of charge | `batt_soc_pct` | arithmetic mean |
| BMS state of health | `batt_soh_pct` | last non-null value |
| Stack voltage | `batt_voltage_v` | arithmetic mean |
| Temperature channels | All topology-resolved pack sensors | arithmetic mean plus sample count |
| Stack current | `batt_current_a` | arithmetic mean |
| Voltage-group channels | All topology-resolved module/pack/group channels | arithmetic mean plus sample count |
| Cycle counter | `sys.cycles` mapping from the profile catalog | last non-null value plus sample count |

`time` and `serial` are always included. Five-minute timestamps are UTC bucket
starts. BMS SOH is retained as a weak label and is not physical capacity ground
truth.

NCA and LFP mappings are separate entries in
`config/telemetry_import_profiles.json`. They currently use the same standardized
BMS column names, but changes to one chemistry do not modify the other.

Every successful capacity-degradation import also refreshes the selected
device's module-count metadata. The extractor reads the latest 1,000 non-null
`batt_modulecount` samples, requires a positive integer with at least 95%
agreement, and writes one metadata row under
`data/processed/device_metadata/module_counts/serial=<serial>/module_count.parquet`.
The value is not duplicated into every one-second telemetry row. This makes a
clean installation ready for electrothermal import without first running the
legacy fleet-wide module-count script.

## Storage

- New one-second telemetry: `Veera_SOH/data/raw_parquet/chemistry=NCA|LFP/serial=.../year=.../month=...`
- Custom telemetry: `Veera_SOH/data/custom_telemetry/<group>/<resolution>/chemistry=NCA|LFP/serial=.../year=.../month=...`
- Existing NCA `serial=...` folders remain readable in place and are not rewritten.
- Each serial must have one authoritative location. Chemistry conflicts and simultaneous
  legacy/canonical locations are rejected before another import is started.

For a verified one-time legacy migration, preview first and apply only while no job for the
serial is active:

```powershell
.venv\Scripts\python.exe -m battryml.data.storage_migration --source-root C:\path\to\Veera_SOH --serial 100000020 --chemistry LFP
.venv\Scripts\python.exe -m battryml.data.storage_migration --source-root C:\path\to\Veera_SOH --serial 100000020 --chemistry LFP --apply
```
- Import evidence: `Veera_SOH/data/import_manifests/serial=.../import_run=...`

Files use deterministic `part-NNNNN.parquet` continuation within each monthly
partition. Zstandard is the default compression. Raw and aggregated rows are
never written into the same dataset.

The schema-2 import manifest is the authoritative record of what was requested
and written. It records purpose, module count, database bounds, per-signal
columns and aggregation, watermarks, actual start/end-exclusive ranges, coverage
sample-count columns, row counts, and file paths.

## Scope Boundary

Capacity-core imports remain deliberately small. Temperature and voltage-group
channels are separate custom datasets so they can be updated independently and
joined to physical discharge IDs later without duplicating the core telemetry.
Their columns are resolved from chemistry and verified module topology.

## LFP Device Configuration

LFP field devices are configured in `config/battery_profiles.json` under the
`lfp_field_storage` variant with device role `field_tester`. No verified LFP
nameplate capacity exists yet, so the variant intentionally omits
`nominal_capacity_ah`, `module_capacity_ah`, and `module_energy_kwh`:

- Capacity-dependent pipelines ask the operator for the value per job; no NCA
  default is inherited and none may be copied into this variant without a
  source-backed nameplate.
- Module counts are not assumed. They are verified against the database with:

  ```powershell
  Set-Location C:\path\to\Veera_SOH
  .\venv\Scripts\python.exe -m src.import_module_counts `
    --profiles-file ..\BatteryML\config\battery_profiles.json
  ```

  The snapshot at `data/processed/device_metadata/module_counts.parquet` then
  carries per-serial counts with agreement and verification status for both
  chemistries. As of 2026-08-24 all configured LFP serials are verified
  (eleven 2-module, five 3-module, three 4-module systems).
- NCA reference assumptions (slope transfer, full-cycle tooling) remain NCA-only
  and never apply to these devices.
## Electrothermal workflow

The fixed **Electrothermal history** preset stores independently resumable
five-minute datasets under the source project:

```text
data/electrothermal_telemetry/temperature/5m/chemistry=<CHEMISTRY>/serial=<SERIAL>/
data/electrothermal_telemetry/cell_voltage/5m/chemistry=<CHEMISTRY>/serial=<SERIAL>/
```

TimescaleDB calculates `mean`, `min`, `max`, `std`, and `count` for each
temperature sensor and voltage-group channel. The operator then uses:

1. **Extraction > Validate imported data > Validate electrothermal data**.
2. **Machine Learning > Device workflow > Build electrothermal features**.

The Extraction page displays electrothermal validation separately from capacity
validation. It shows only build-gating evidence: temperature and voltage-group
readiness, expected channel counts, minimum channel coverage, validated UTC
range, duplicate timestamps, misaligned five-minute rows, and stale status.

The build is locked until the latest temperature and voltage validation passes.
It joins both datasets by serial and five-minute UTC bucket and adds SOC, BMS
SOH, current, and stack-voltage context aggregated locally from capacity-core
telemetry. It writes the established feature contract under
`data/processed/bms_hierarchical_features/serial=<SERIAL>/`.

Operating-event Parquet is not an input to this build. Phase 3 and later stages
use their existing event IDs and timestamps to select bounded periods from the
continuous feature history.
