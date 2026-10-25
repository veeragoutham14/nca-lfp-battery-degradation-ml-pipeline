export interface Health {
  status: 'ok' | 'degraded'
  version: string
  source_root: string
  catalog_path: string
  auth_enabled: boolean
  source_problems: string[]
  database: {
    configured: boolean
    reachable: boolean
    host: string | null
    port: number | null
    message: string
  }
  counts: Record<string, number>
}

export interface AuthUser {
  username: string
  auth_enabled: boolean
  authenticated: boolean
}

export type BatteryChemistry = 'NCA' | 'LFP'

export type ImportPurpose = 'capacity_core' | 'electrothermal_history' | 'custom'
export type ImportResolution = 'off' | '1s' | '5m' | 'both'

export interface ImportPlanSignal {
  group: string
  label: string
  resolution: '1s' | '5m'
  chemistry: BatteryChemistry
  column_count: number | null
  local_range: { start: string | null; end: string | null }
  planned_start: string | null
  planned_rule: 'continue_after_watermark' | 'device_configuration_start' | 'database_first_timestamp'
  planned_end_open: boolean
  database_end: string | null
  estimated_rows_per_month: number | null
  requires_module_topology: boolean
  topology_resolved: boolean
}

export interface ImportPlan {
  purpose: ImportPurpose
  signals: ImportPlanSignal[]
}

export interface DataValidationReadiness {
  status: 'ready' | 'blocked'
  reason: string
}

export interface DataValidationResult {
  run_id: string
  summary: {
    model_family: 'battery_data_validation'
    validator_version: string
    serial: string
    chemistry: BatteryChemistry
    module_count: number
    validated_at: string
    timestamp_semantics: 'UTC'
    observed_start: string | null
    observed_end: string | null
    raw_file_count: number
    raw_row_count: number
    distinct_timestamp_count: number
    calendar_coverage_pct: number
    maximum_gap_seconds: number
    duplicate_rows: number
    invalid_core_rows: number
    stale_days: number | null
    suspicious_temperature_windows: number | null
    current_sign_verification: {
      status: 'verified' | 'inconclusive' | 'conflict'
      agreement_pct: number | null
      evidence_windows: number
      tested_semantics?: string
    }
    feature_evidence: {
      feature_available: boolean
      feature_start: string | null
      feature_end: string | null
      aggregation_minutes: number
      expected_voltage_groups: number
      available_voltage_groups: number
      expected_temperature_sensors: number
      available_temperature_sensors: number
      source_audit_available: boolean
      source_audit_ready: boolean
    }
    readiness: Record<'capacity_bms' | 'monthly_exposure' | 'voltage_group_health', DataValidationReadiness>
    resistance_reference: {
      available: boolean
      interpretation_verified: boolean
      analysis_enabled: boolean
      note: string
    }
    claim_limit: string
  }
  signal_inventory: Array<{
    signal_key: string
    source_column: string
    label: string
    unit: string
    resolution: string
    available: boolean
    purposes: string
    semantics: string
  }>
  coverage_by_month: Array<{
    month: string
    distinct_timestamps: number
    row_count: number
    duplicate_rows: number
    first_timestamp: string
    last_timestamp: string
    maximum_gap_seconds: number | null
    observed_span_coverage_pct: number
    invalid_soc_rows: number
    invalid_soh_rows: number
    one_second_available: boolean
    five_minute_available: boolean
    voltage_groups_available: number
    temperature_sensors_available: number
  }>
  topology_inventory: Array<{
    measurement_type: 'cell_voltage_channel' | 'pack_temperature_sensor'
    module_index: number
    pack_index: number
    source_channel_index: number
    channel_key: string
    available: boolean
  }>
  quality_findings: Array<{
    finding_key: string
    severity: 'pass' | 'info' | 'warning' | 'critical'
    value: number | string | null
    message: string
  }>
}

export interface ElectrothermalValidationGroup {
  ready: boolean
  expected_channels: number
  minimum_channel_coverage_pct: number
  start_utc: string | null
  end_utc: string | null
  duplicate_timestamps: number
  misaligned_five_minute_rows: number
}

export interface ElectrothermalValidationResult {
  run_id: string
  serial: string
  chemistry: BatteryChemistry
  module_count: number
  validated_at: string
  stale: boolean
  build_ready: boolean
  status: 'ready' | 'stale' | 'failed'
  reason: string
  groups: {
    temperature: ElectrothermalValidationGroup
    cell_voltage: ElectrothermalValidationGroup
  }
}

export interface DeviceReadiness {
  raw: boolean
  cell_signal_audit: boolean
  cell_health: boolean
  electrothermal_source_audit: boolean
  electrothermal_validation: boolean
  hierarchical_features: boolean
  soh_signal_audit: boolean
  data_validation: boolean
  discharge_capacity_evidence: boolean
  soh_recalibration: boolean
  capacity_soh_investigation: boolean
  seasonal_charge_profile: boolean
  classified: boolean
  events: boolean
  capacity_trend: boolean
  analysis: boolean
}

export interface WorkflowStageState {
  status: 'missing' | 'ready' | 'running' | 'current' | 'stale' | 'blocked' | 'failed'
  reason: string
  output_modified_at: string | null
  required_modified_at: string | null
}

export interface DeviceLibraryItem {
  serial: string
  display_name: string
  display_name_source: 'custom' | 'profile' | 'fallback'
  variant_key: string | null
  chemistry: string | null
  device_role: 'field_tester' | 'full_cycle_test' | 'unassigned'
  nominal_capacity_ah: number | null
  configured_nominal_capacity_ah: number | null
  module_capacity_ah: number | null
  module_energy_kwh: number | null
  loss_slope_reference_module_count: number | null
  module_count: number | null
  module_count_source: 'database_snapshot' | 'configured_profile' | 'missing'
  module_count_agreement: number
  system_energy_kwh: number | null
  observed_module_counts: string
  module_count_status: 'verified' | 'unstable' | 'invalid' | 'missing' | 'unconfigured'
  configured_start: string | null
  default_loss_slopes_per_1000ah: number[]
  raw_status: 'not_imported' | 'importing' | 'ready'
  raw_file_count: number
  raw_row_count: number
  five_minute_file_count: number
  five_minute_row_count: number
  five_minute_start_at: string | null
  five_minute_end_at: string | null
  observed_start_at: string | null
  observed_end_at: string | null
  recommended_import_start_at: string | null
  latest_import_job_status: string | null
  run_count: number
  latest_run_id: string | null
  readiness: DeviceReadiness
  workflow: Record<string, WorkflowStageState>
}

export interface Battery {
  serial: string
  source_root: string
  first_observed_at: string | null
  last_observed_at: string | null
  latest_run_id: string | null
  run_count: number
  updated_at: string
  latest_model_family: string | null
  latest_validation_mae_soh_pct: number | null
  latest_loss_slope_per_1000ah: number | null
}

export interface AnalysisRun {
  run_id: string
  source_run_id: string
  serial: string
  pipeline_key: string
  model_family: string
  status: string
  nominal_capacity_ah: number | null
  training_rows: number | null
  validation_rows: number | null
  validation_mae_soh_pct: number | null
  validation_max_error_soh_pct: number | null
  loss_slope_per_1000ah: number | null
  forecast_rows: number | null
  observed_start_at: string | null
  observed_end_at: string | null
  run_created_at: string | null
}

export interface HistoryPoint {
  timestamp: string
  capacity_ah: number | null
  soh_pct: number
  bms_soh_pct: number | null
  cumulative_discharge_ah: number | null
  valid_training_row: boolean | null
  sample_count: number | null
}

export interface ForecastPoint {
  timestamp: string
  days_ahead: number
  soh_pct: number
  lower_95_soh_pct: number | null
  upper_95_soh_pct: number | null
  capacity_ah: number
  cumulative_discharge_ah: number
  forecast_ah_per_day: number
  usage_error_soh_pct_final: number | null
  usage_error_soh_pct_slope: number | null
}

export interface ForecastScenario {
  scenario: string
  points: ForecastPoint[]
}

export interface SeasonalProfilePoint {
  month_start_utc: string
  season: string | null
  expected_days_in_observed_scope: number | null
  observed_days: number | null
  telemetry_coverage_fraction: number | null
  seasonal_profile_valid: boolean
  charge_ah: number | null
  discharge_ah: number | null
  charge_ah_per_observed_day: number | null
  discharge_ah_per_observed_day: number | null
  charging_hours: number | null
  discharging_hours: number | null
  rest_hours: number | null
  high_soc_95_hours: number | null
  low_soc_5_hours: number | null
  soc_0_5_hours: number | null
  soc_5_20_hours: number | null
  soc_20_40_hours: number | null
  soc_40_60_hours: number | null
  soc_60_80_hours: number | null
  soc_80_95_hours: number | null
  soc_95_100_hours: number | null
  soc_known_hours: number | null
  soc_unknown_hours: number | null
  soc_known_coverage_fraction: number | null
  temperature_mean_c: number | null
  temperature_p95_daily_c: number | null
  temperature_max_c: number | null
  hot_35_hours: number | null
  voltage_spread_p95_daily_mv: number | null
  voltage_spread_max_mv: number | null
  charge_episode_count: number | null
  near_full_charge_episode_count: number | null
  median_charge_start_hour_utc: number | null
  bms_soh_first_pct: number | null
  bms_soh_last_pct: number | null
  audited_soh_change_pp: number | null
  raw_rms_current_a: number | null
  raw_peak_charge_current_a: number | null
  raw_peak_discharge_current_a: number | null
  raw_peak_10s_mean_absolute_current_a: number | null
  raw_peak_60s_mean_absolute_current_a: number | null
  raw_p95_current_step_a: number | null
  raw_large_current_step_count: number | null
  raw_current_reversal_count: number | null
  raw_high_current_episode_count: number | null
  raw_c0_hours: number | null
  raw_c1_hours: number | null
  raw_c2_hours: number | null
  raw_c3_hours: number | null
  raw_c4_hours: number | null
  high_discharge_low_soc_hours: number | null
  high_charge_high_soc_hours: number | null
  charging_above_95_soc_hours: number | null
  high_current_hot_hours: number | null
  high_charge_cold_hours: number | null
  high_current_imbalanced_hours: number | null
  low_soc_high_discharge_imbalanced_hours: number | null
  temperature_below_10_hours: number | null
  temperature_10_20_hours: number | null
  temperature_20_30_hours: number | null
  temperature_30_40_hours: number | null
  temperature_above_40_hours: number | null
}

export interface SohTransitionContextPoint {
  transition_id: string
  event_time_utc: string
  event_type: string
  previous_soh_pct: number | null
  new_soh_pct: number | null
  soh_delta_pp: number | null
  context_quality_ready: boolean
  event_during_charge_episode: boolean | null
  near_full_charge_within_72h_before: boolean | null
  hours_since_near_full_charge: number | null
  authoritative_recalibration_match: boolean | null
  recalibration_transition_detection_status: string | null
  recalibration_complete_discharge_within_lookback: boolean | null
  recalibration_hours_since_complete_discharge: number | null
  recalibration_raw_start_soc_pct: number | null
  recalibration_raw_end_soc_pct: number | null
  recalibration_raw_soc_drop_pct: number | null
  recalibration_discharged_ah_1s: number | null
  recalibration_charge_intrusion_ah_1s: number | null
  recalibration_raw_coverage_fraction: number | null
  recalibration_rest_minutes_inside_cycle: number | null
  recalibration_longest_rest_minutes: number | null
  recalibration_rest_interruption_count: number | null
  recalibration_cycle_continuity_class: string | null
  recalibration_capacity_validation_status: string | null
  recalibration_direct_capacity_soh_pct: number | null
  recalibration_soc_normalized_capacity_soh_pct: number | null
  recalibration_bms_after_event_minus_capacity_soh_pp: number | null
  recalibration_primary_context_pattern: string | null
  recalibration_context_match_strength: string | null
  pre_mean_coverage_fraction: number | null
  pre_charge_ah: number | null
  pre_discharge_ah: number | null
  pre_charge_episode_count: number | null
  pre_near_full_charge_episode_count: number | null
  pre_high_soc_95_hours: number | null
  pre_low_soc_5_hours: number | null
  pre_temperature_mean_c: number | null
  pre_temperature_max_c: number | null
  pre_voltage_spread_p95_mv: number | null
  pre_voltage_spread_max_mv: number | null
  post_mean_coverage_fraction: number | null
  post_charge_ah: number | null
  post_discharge_ah: number | null
  post_charge_episode_count: number | null
  post_near_full_charge_episode_count: number | null
  post_high_soc_95_hours: number | null
  post_low_soc_5_hours: number | null
  post_temperature_mean_c: number | null
  post_temperature_max_c: number | null
  post_voltage_spread_p95_mv: number | null
  post_voltage_spread_max_mv: number | null
}

export interface RunSeries {
  run_id: string
  serial: string
  model_family: string
  nominal_capacity_ah: number | null
  history: HistoryPoint[]
  forecasts: ForecastScenario[]
  seasonal_profile: SeasonalProfilePoint[]
  soh_transitions: SohTransitionContextPoint[]
}

export interface ValidationPoint {
  timestamp: string
  event_id: number | null
  cumulative_discharge_ah: number | null
  actual_capacity_ah: number | null
  predicted_capacity_ah: number | null
  capacity_error_ah: number | null
  actual_soh_pct: number | null
  predicted_soh_pct: number | null
  soh_error_pct: number | null
  abs_soh_error_pct: number | null
}

export interface ValidationSeries {
  run_id: string
  points: ValidationPoint[]
}

export interface Artifact {
  artifact_id: string
  run_id: string
  kind: string
  name: string
  path: string
  media_type: string
  size_bytes: number
  modified_at: string
  download_url: string
}

export interface Dataset {
  dataset_id: string
  stage: string
  name: string
  format: string
  row_count: number | null
  schema_json: string | null
  observed_start_at: string | null
  observed_end_at: string | null
}

export interface PluginParameter {
  key: string
  label: string
  type: 'text' | 'number' | 'integer' | 'boolean' | 'choice' | 'datetime' | 'path'
  default: unknown
  required: boolean
  choices: string[]
  description: string
  group: string
  serial_defaults?: Record<string, unknown>
}

export interface PipelinePlugin {
  key: string
  name: string
  description: string
  output_family: string
  workflow_stage: string
  device_roles: string[]
  chemistries: BatteryChemistry[]
  requires_database: boolean
  scan_output_families: string[]
  requires_stage: string | null
  requires_stages: string[]
  parameters: PluginParameter[]
}

export interface PipelineJob {
  job_id: string
  pipeline_key: string
  serial: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'abandoned'
  created_at: string
  started_at: string | null
  completed_at: string | null
  return_code: number | null
  output_run_id: string | null
  error_message: string | null
  archived_at: string | null
}

export interface NcaCapacityOverviewMetrics {
  inventory_device_count: number
  deferred_device_count: number
  configured_device_count: number
  available_device_count: number
  missing_device_count: number
  total_cumulative_discharge_ah: number
  median_latest_bms_soh_pct: number | null
  median_net_bms_soh_change_pct: number | null
  median_bms_minus_reference_soh_pct: number | null
  median_bms_equivalent_loss_slope_ah_per_1000ah: number | null
  median_telemetry_coverage_pct: number | null
  devices_below_reference_by_more_than_tolerance: number
  devices_above_twice_reference_loss_rate: number
  devices_with_bms_recoveries: number
  devices_with_verified_module_count: number
}

export interface NcaCapacityOverviewDevice {
  serial: string
  display_name: string
  variant_key: string | null
  chemistry: string
  module_count: number | null
  module_count_status: string
  comparison_scope: 'matched_reference_configuration' | 'exploratory_efc_normalized_cross_module'
  nominal_capacity_ah: number
  capacity_per_module_ah: number | null
  system_energy_kwh: number | null
  observed_start: string
  observed_end: string
  observation_rows: number
  first_bms_soh_pct: number
  latest_bms_soh_pct: number
  net_bms_soh_change_pct: number
  bms_implied_capacity_ah: number
  cumulative_discharge_ah: number
  throughput_span_ah: number
  equivalent_full_cycles: number
  discharge_ah_per_module: number | null
  reference_loss_slope_ah_per_1000ah: number
  reference_capacity_ah: number
  reference_soh_pct: number
  bms_minus_reference_soh_pct: number
  bms_minus_reference_capacity_ah: number
  reference_mae_soh_pct: number
  within_bms_tolerance_pct: number
  bms_equivalent_loss_slope_ah_per_1000ah: number | null
  step_drop_count: number
  step_recovery_count: number
  telemetry_coverage_pct: number | null
  quality_flags: string[]
  source_run_id: string
  source_modified_at: string
}

export interface NcaCapacityOverview {
  model_family: 'nca_capacity_overview'
  chemistry: 'NCA'
  generated_at: string
  latest_source_modified_at: string | null
  cohort: {
    device_role: 'field_tester'
    module_count: number | null
    module_count_status: 'verified'
    comparison_basis: string
    policy: string
  }
  field_inventory: {
    total_device_count: number
    active_device_count: number
    deferred_device_count: number
  }
  reference: {
    serial: string
    display_name: string
    module_count: number
    loss_slope_ah_per_1000ah: number
    source: string
    transfer_basis: string
  }
  metrics: NcaCapacityOverviewMetrics
  devices: NcaCapacityOverviewDevice[]
  missing_devices: Array<{ serial: string; display_name: string; reason: string }>
  deferred_devices: Array<{
    serial: string
    display_name: string
    variant_key: string | null
    module_count: number | null
    module_count_status: string
    nominal_capacity_ah: number | null
    system_energy_kwh: number | null
    cohort_status: 'deferred'
    cohort_reason: string
  }>
  evidence_policy: {
    throughput_source: string
    capacity_signal: string
    bms_soh_role: string
    reference_curve_role: string
    ground_truth_status: string
    claim_limit: string
  }
}

export interface CellSignalOverviewMetrics {
  inventory_device_count: number
  configured_device_count: number
  field_device_count: number
  all_field_device_count: number
  deferred_field_device_count: number
  reference_device_count: number
  audited_device_count: number
  ready_device_count: number
  fresh_ready_device_count: number
  stale_device_count: number
  pending_device_count: number
  cell_health_built_device_count: number
  hierarchical_history_built_device_count: number
  verified_module_count_devices: number
  total_expected_active_cell_channels: number
  total_ready_active_cell_channels: number
  total_expected_active_temperature_sensors: number
  total_ready_active_temperature_sensors: number
}

export interface CellSignalOverviewDevice {
  serial: string
  display_name: string
  variant_key: string | null
  device_role: 'field_tester' | 'full_cycle_test'
  cohort_role: 'field_device' | 'reference_control'
  cohort_status: 'included' | 'deferred' | 'reference'
  cohort_reason: string
  in_active_cohort: boolean
  module_count: number | null
  module_count_status: string
  configured_start: string | null
  observed_start: string | null
  observed_end: string | null
  audited: boolean
  cell_voltage_ready: boolean
  cell_health_built: boolean
  feature_contract: 'hierarchical_electrothermal_v1' | 'legacy_cell_health_v1' | 'missing'
  source_contract: string
  quality_gate_status: string | null
  blocking_reasons: string[]
  quality_warnings: string[]
  telemetry_fresh: boolean
  freshness_lag_days: number | null
  sample_start: string | null
  sample_end: string | null
  total_rows: number | null
  nominal_1hz_coverage_pct: number | null
  expected_cell_channels: number | null
  present_cell_channels: number | null
  ready_cell_channels: number | null
  minimum_cell_channel_valid_coverage_pct: number | null
  expected_temperature_sensors: number | null
  active_temperature_sensors: number | null
  inactive_temperature_sensors: number | null
  ready_temperature_sensors: number | null
  minimum_temperature_sensor_valid_coverage_pct: number | null
  minimum_active_temperature_sensor_valid_coverage_pct: number | null
  global_cell_voltage_signals_ready: boolean
  cell_temperature_signals_ready: boolean
  source_run_id: string | null
  source_modified_at: string | null
  feature_start: string | null
  feature_end: string | null
  recommended_feature_start: string | null
  recommended_feature_end: string | null
  feature_build_mode: 'full' | 'incremental' | 'up_to_date'
  feature_lag_seconds: number | null
  feature_run_id: string | null
}

export interface CellSignalOverview {
  model_family: 'cell_signal_fleet_overview'
  chemistry: 'NCA'
  generated_at: string
  latest_source_modified_at: string | null
  cohort: {
    active_module_count: number
    reference_serial: string
    reference_module_count: number
    policy: string
  }
  metrics: CellSignalOverviewMetrics
  devices: CellSignalOverviewDevice[]
}

export interface CellAnomalyOverviewMetrics {
  inventory_device_count: number
  deferred_device_count: number
  configured_device_count: number
  field_device_count: number
  reference_device_count: number
  history_device_count: number
  analyzed_device_count: number
  archived_legacy_device_count: number
  candidate_device_count: number
  total_sustained_episode_count: number
  total_anomaly_days: number
  elevated_epoch_shift_device_count: number
}

export interface BmsSohRecalibrationOverviewMetrics {
  configured_device_count: number
  prerequisite_ready_device_count: number
  analyzed_device_count: number
  total_event_count: number
  total_step_drop_count: number
  total_step_recovery_count: number
  total_data_gap_risk_count: number
  high_confidence_event_count: number
  strong_context_match_count: number
  exact_transition_count: number
  daily_fallback_transition_count: number
  limited_context_event_count: number
}

export interface BmsSohRecalibrationOverviewDevice {
  serial: string
  display_name: string
  variant_key: string | null
  device_role: 'field_tester' | 'full_cycle_test' | 'unassigned'
  module_count: number | null
  module_count_status: string
  nominal_capacity_ah: number | null
  prerequisites_ready: boolean
  analyzed: boolean
  source_run_id: string | null
  source_modified_at: string | null
  observed_start: string | null
  observed_end: string | null
  detector_version: string | null
  event_rows: number
  step_drop_rows: number
  step_recovery_rows: number
  change_after_gap_rows: number
  high_confidence_rows: number
  strong_context_match_rows: number
  exact_transition_rows: number
  daily_fallback_transition_rows: number
  limited_context_rows: number
  classification_counts: Record<string, number>
}

export interface BmsSohRecalibrationOverview {
  model_family: 'bms_soh_recalibration_overview'
  chemistry: BatteryChemistry
  generated_at: string
  latest_source_modified_at: string | null
  metrics: BmsSohRecalibrationOverviewMetrics
  devices: BmsSohRecalibrationOverviewDevice[]
  evidence_policy: {
    candidate_source: string
    time_resolution: string
    operating_evidence: string
    claim_limit: string
  }
}

export interface BmsSohRecalibrationPoint {
  time: string
  measured_soh_pct: number | null
  bms_soh_end: number | null
  coverage_fraction: number | null
  daily_soh_range_pct: number | null
  discharge_ah: number | null
  charge_ah: number | null
  max_gap_seconds: number | null
  recalibration_event_count: number
  step_drop_count: number
  step_recovery_count: number
}

export interface BmsSohRecalibrationEvent {
  event_id: string
  event_time: string
  event_time_resolution: string
  daily_event_anchor_utc?: string | null
  event_time_lower_utc?: string | null
  event_time_upper_utc?: string | null
  event_time_uncertainty_seconds?: number | null
  transition_detection_status?: 'exact' | 'interval_censored' | 'raw_transition_not_resolved'
  pre_transition_support_fraction?: number | null
  post_transition_support_fraction?: number | null
  event_type: string
  previous_soh_pct: number | null
  new_soh_pct: number | null
  soh_delta_pp: number | null
  audit_coverage_fraction: number | null
  audit_cumulative_discharge_ah: number | null
  context_lookback_hours: number
  context_observed_minutes: number
  context_coverage_fraction: number
  minimum_soc_pct: number | null
  maximum_soc_pct: number | null
  reached_low_soc: boolean
  reached_near_full_soc: boolean
  near_full_charge_within_lookback: boolean
  hours_since_near_full_charge: number | null
  near_full_charge_start_soc_pct?: number | null
  near_full_charge_soc_gain_pct?: number | null
  near_full_charge_ah?: number | null
  near_full_charge_taper_minutes?: number | null
  near_full_charge_maximum_c_rate?: number | null
  meaningful_charge_before_update?: boolean
  low_before_near_full?: boolean
  complete_discharge_within_lookback?: boolean
  hours_since_complete_discharge?: number | null
  complete_discharge_start_time_utc?: string | null
  complete_discharge_end_time_utc?: string | null
  complete_discharge_start_soc_pct?: number | null
  complete_discharge_end_soc_pct?: number | null
  complete_discharge_soc_drop_pct?: number | null
  complete_discharge_ah?: number | null
  complete_discharge_equivalent_capacity_fraction?: number | null
  complete_discharge_duration_minutes?: number | null
  complete_discharge_coverage_fraction?: number | null
  complete_discharge_dominance_fraction?: number | null
  raw_discharge_detection_status?: 'resolved' | 'not_observed' | 'invalid_anchor' | 'soc_traversal_without_negative_discharge'
  raw_discharge_candidate_count?: number
  raw_discharge_ambiguous?: boolean
  raw_discharge_selection?: string | null
  raw_discharge_qualified?: boolean
  raw_discharge_dominance_fraction?: number | null
  raw_start_time_utc?: string | null
  raw_end_time_utc?: string | null
  raw_start_soc_pct?: number | null
  raw_end_soc_pct?: number | null
  raw_soc_drop_pct?: number | null
  first_low_band_time_utc?: string | null
  minimum_soc_time_utc?: string | null
  terminal_low_soc_tail_minutes?: number | null
  post_minimum_low_soc_tail_minutes?: number | null
  terminal_low_soc_rest_tail_minutes?: number | null
  terminal_low_soc_potential_rest_tail_minutes?: number | null
  potential_gap_spanning_rest_minutes?: number | null
  raw_coverage_fraction?: number | null
  maximum_raw_gap_seconds?: number | null
  unobserved_discharge_ah_estimate?: number | null
  missing_ah_assumption_current_a?: number | null
  missing_ah_estimation_method?: string | null
  discharged_ah_1s?: number | null
  charge_intrusion_ah_1s?: number | null
  rest_before_event_minutes: number | null
  maximum_temperature_c: number | null
  mean_temperature_c?: number | null
  p95_temperature_c?: number | null
  maximum_voltage_spread_mv: number | null
  p95_voltage_spread_mv?: number | null
  largest_data_gap_minutes: number | null
  charging_window_count_in_context: number
  discharge_ah_in_context: number
  charge_ah_in_context?: number
  high_soc_hours?: number
  low_soc_hours?: number
  maximum_absolute_c_rate?: number | null
  prior_same_direction_events: number
  evidence_classification: string
  classification_confidence: 'high' | 'medium' | 'limited'
  primary_context_pattern?: string
  context_match_strength?: 'strong' | 'moderate' | 'limited'
  near_full_within_1h?: boolean
  near_full_within_3h?: boolean
  near_full_within_6h?: boolean
  near_full_within_12h?: boolean
  near_full_within_24h?: boolean
  near_full_within_48h?: boolean
  near_full_within_72h?: boolean
  complete_discharge_within_1h?: boolean
  complete_discharge_within_3h?: boolean
  complete_discharge_within_6h?: boolean
  complete_discharge_within_12h?: boolean
  complete_discharge_within_24h?: boolean
  complete_discharge_within_48h?: boolean
  complete_discharge_within_72h?: boolean
  evidence_tags: string
  interpretation: string
}

export interface BmsSohEvidenceComparison {
  feature_key: string
  feature_label: string
  event_count: number
  event_total: number
  event_rate: number | null
  event_rate_ci95_low: number | null
  event_rate_ci95_high: number | null
  control_count: number
  control_total: number
  control_rate: number | null
  control_rate_ci95_low: number | null
  control_rate_ci95_high: number | null
  risk_difference: number | null
  prevalence_lift: number | null
}

export interface BmsSohReferenceComparison {
  available: boolean
  reason: string | null
  reference_serial: string
  is_reference_device?: boolean
  reference_run_id?: string
  selected?: Record<string, number | null>
  reference?: Record<string, number | null>
  claim_limit?: string
  comparison_scope?: 'matched_reference_configuration' | 'exploratory_normalized_cross_module'
}

export interface BmsSohInterpretableModel {
  available: boolean
  reason?: string
  model_type?: string
  purpose?: string
  event_groups?: number
  rows?: number
  regularization_strength?: number
  validation_method?: string
  cross_validated_roc_auc?: number | null
  cross_validated_brier_score?: number | null
  constant_prevalence_brier_score?: number | null
  status?: string
  claim_limit?: string
  coefficients?: Array<{
    feature_key: string
    coefficient: number
    odds_ratio: number
    unit: string
  }>
}

export interface BmsSohCapacityErrorMetrics {
  rows: number
  mae_pp: number | null
  bias_pp: number | null
  maximum_absolute_error_pp: number | null
}

export interface BmsSohCycleContinuityMetrics {
  cycle_rows: number
  linked_soh_event_rows: number
  linked_soh_event_rate: number | null
  median_rest_minutes: number | null
  median_longest_rest_minutes: number | null
}

export interface BmsSohCapacityValidationSummary {
  available: boolean
  reason: string | null
  cycle_rows: number
  raw_resolved_cycle_rows?: number
  strict_full_boundary_rows?: number
  validation_grade_rows?: number
  linked_validation_grade_rows?: number
  median_discharged_ah?: number | null
  median_capacity_soh_pct?: number | null
  operational_window_rows?: number
  linked_operational_window_rows?: number
  median_operational_soc_span_pp?: number | null
  median_soc_normalized_capacity_ah?: number | null
  median_soc_normalized_capacity_soh_pct?: number | null
  bms_at_completion_error?: BmsSohCapacityErrorMetrics
  bms_after_linked_event_error?: BmsSohCapacityErrorMetrics
  bms_at_completion_normalized_error?: BmsSohCapacityErrorMetrics
  bms_after_linked_event_normalized_error?: BmsSohCapacityErrorMetrics
  continuity_comparison?: {
    continuous: BmsSohCycleContinuityMetrics
    interrupted: BmsSohCycleContinuityMetrics
  }
  capacity_label_semantics?: string
  operational_window_semantics?: string
  claim_limit?: string
}

export interface BmsSohCapacityValidation {
  discharge_episode_id: string
  raw_start_time_utc: string | null
  raw_end_time_utc: string | null
  raw_start_soc_pct: number | null
  raw_end_soc_pct: number | null
  raw_soc_drop_pct: number | null
  raw_sample_rows: number | null
  raw_coverage_fraction: number | null
  maximum_raw_gap_seconds: number | null
  unobserved_discharge_ah_estimate: number | null
  missing_ah_assumption_current_a: number | null
  missing_ah_estimation_method: string | null
  discharged_ah_1s: number | null
  charge_intrusion_ah_1s: number | null
  direct_capacity_soh_pct: number | null
  soc_normalized_capacity_ah: number | null
  soc_normalized_capacity_soh_pct: number | null
  strict_full_soc_boundaries: boolean | null
  capacity_validation_status: string
  cycle_continuity_class: string | null
  active_discharge_minutes: number | null
  rest_minutes_inside_cycle: number | null
  charge_minutes_inside_cycle: number | null
  rest_fraction: number | null
  discharge_dominance_fraction_1s: number | null
  rest_interruption_count: number | null
  longest_rest_minutes: number | null
  bms_soh_at_cycle_start_pct: number | null
  bms_soh_at_cycle_completion_pct: number | null
  bms_completion_minus_capacity_soh_pp: number | null
  bms_completion_minus_normalized_soh_pp: number | null
  linked_soh_event_id: string | null
  linked_soh_event_time_utc: string | null
  hours_to_linked_soh_event: number | null
  linked_soh_event_delta_pp: number | null
  bms_soh_after_linked_event_pct: number | null
  bms_after_event_minus_capacity_soh_pp: number | null
  bms_after_event_minus_normalized_soh_pp: number | null
  raw_discharge_candidate_count?: number | null
  raw_discharge_ambiguous?: boolean | null
  raw_discharge_selection?: string | null
  raw_discharge_qualified?: boolean | null
  raw_discharge_dominance_fraction?: number | null
  first_low_band_time_utc?: string | null
  minimum_soc_time_utc?: string | null
  terminal_low_soc_tail_minutes?: number | null
  post_minimum_low_soc_tail_minutes?: number | null
  terminal_low_soc_rest_tail_minutes?: number | null
  terminal_low_soc_potential_rest_tail_minutes?: number | null
  potential_gap_spanning_rest_minutes?: number | null
  rms_discharge_current_a?: number | null
  peak_discharge_current_a?: number | null
  pack_temperature_mean_c?: number | null
  pack_temperature_p95_c?: number | null
  pack_temperature_max_c?: number | null
  cell_voltage_spread_p95_mv?: number | null
  cell_voltage_spread_max_mv?: number | null
  channel_identity?: string | null
  weakest_channel_episode_count?: number | null
}

export type CapacityEvidenceClass =
  | 'strict_full_discharge'
  | 'near_full_discharge'
  | 'operational_full_discharge'
  | 'partial_discharge'
  | 'none'

export interface CapacityBmsDischargeEvent {
  device: string
  discharge_event_id: string
  evidence_class: CapacityEvidenceClass
  measurement_resolution: 'one_second_event_anchors'
  capacity_claim_eligible: boolean
  capacity_estimate_eligible: boolean
  meaningful_deep_discharge: boolean
  start_utc: string | null
  end_utc: string | null
  duration_s: number | null
  soc_start_pct: number | null
  soc_end_pct: number | null
  soc_span_pp: number | null
  discharged_ah: number | null
  discharged_wh: number | null
  gross_discharged_ah: number | null
  gross_discharged_wh: number | null
  charge_intrusion_ah: number | null
  charge_intrusion_wh: number | null
  net_discharged_ah: number | null
  net_discharged_wh: number | null
  mean_discharge_current_a: number | null
  rms_current_a: number | null
  p95_current_a: number | null
  p99_current_a: number | null
  max_discharge_current_a: number | null
  rms_c_rate: number | null
  data_coverage_fraction: number | null
  maximum_gap_s: number | null
  missing_duration_s: number | null
  rest_duration_s: number | null
  longest_rest_s: number | null
  rest_interruption_count: number
  charge_intrusion_count: number
  charge_intrusion_duration_s: number | null
  temperature_mean_c: number | null
  temperature_median_c: number | null
  temperature_min_c: number | null
  temperature_max_c: number | null
  temperature_coverage_fraction: number | null
  reference_capacity_ah: number
  reference_capacity_source: string
  estimated_full_capacity_ah: number | null
  calculated_soh_pct: number | null
  discharge_derived_soh_proxy_pct: number | null
  soc_span_normalized_soh_estimate_pct: number | null
  capacity_above_reference: boolean | null
  measurement_quality: 'high' | 'medium' | 'low' | 'indeterminate'
  measurement_quality_reasons: string[]
  reference_comparability: 'directly_comparable' | 'comparable_with_caution' | 'not_comparable' | 'needs_electrothermal_review' | 'indeterminate'
  reference_comparability_reasons: string[]
  comparison_reference_device: string | null
  comparison_reference_event_id: string | null
  comparison_scope: string
  continuity: string
  trajectory_end_reason: string
}

export interface ElectrothermalCapacityContext {
  discharge_event_id: string
  evidence_class: CapacityEvidenceClass
  start_utc: string
  end_utc: string
  phase3_status: 'measured_context_ready' | 'indeterminate'
  phase3_reasons: string[]
  native_coverage_fraction: number | null
  rms_discharge_current_a: number | null
  peak_discharge_current_a: number | null
  rms_c_rate: number | null
  temperature_mean_c: number | null
  temperature_min_c: number | null
  temperature_max_c: number | null
  temperature_sensor_spread_p95_c: number | null
  minutes_below_15c: number | null
  minutes_below_10c: number | null
  minutes_above_30c: number | null
  minutes_above_35c: number | null
  minutes_above_40c: number | null
  stack_voltage_min_v: number | null
  pack_voltage_sum_delta_p95_v: number | null
  voltage_group_spread_p95_mv: number | null
  voltage_group_spread_max_mv: number | null
  low_soc_voltage_group_spread_p95_mv: number | null
  recurring_lowest_voltage_group: string | null
  recurring_lowest_voltage_group_fraction: number | null
  soc_20_sample_rows?: number
  soc_20_current_median_a?: number | null
  soc_20_temperature_median_c?: number | null
  soc_20_voltage_spread_p95_mv?: number | null
  soc_10_sample_rows?: number
  soc_10_current_median_a?: number | null
  soc_10_temperature_median_c?: number | null
  soc_10_voltage_spread_p95_mv?: number | null
  soc_5_sample_rows?: number
  soc_5_current_median_a?: number | null
  soc_5_temperature_median_c?: number | null
  soc_5_voltage_spread_p95_mv?: number | null
  m01_p01_temperature_min_c?: number | null
  m01_p01_temperature_max_c?: number | null
  m01_p02_temperature_min_c?: number | null
  m01_p02_temperature_max_c?: number | null
  m02_p01_temperature_min_c?: number | null
  m02_p01_temperature_max_c?: number | null
  m02_p02_temperature_min_c?: number | null
  m02_p02_temperature_max_c?: number | null
  reference_comparability: 'high' | 'medium' | 'low' | 'indeterminate' | 'reference_measurement'
  reference_comparability_reasons: string[]
  comparison_reference_serial: string | null
  comparison_reference_event_id: string | null
}

export interface CapacityBmsSohTransition {
  device: string
  transition_id: string
  previous_soh_pct: number | null
  new_soh_pct: number | null
  delta_soh_pp: number | null
  direction: 'up' | 'down'
  transition_time_utc: string
  timestamp_quality: 'exact' | 'bounded' | 'approximate'
  data_coverage_fraction: number | null
}

export interface CapacityBmsDischargeMatch {
  transition_id: string
  transition_time_utc: string
  primary_preceding_discharge_event_id: string | null
  nearest_qualifying_discharge_event_id: string | null
  discharge_evidence_class: CapacityEvidenceClass | 'no_qualifying_discharge'
  discharge_end_to_soh_update_hours: number | null
  discharge_derived_capacity_ah: number | null
  preceding_discharge_start_soc_pct: number | null
  preceding_discharge_end_soc_pct: number | null
  preceding_discharge_soc_drop_pp: number | null
  preceding_discharge_gross_ah: number | null
  preceding_discharge_charge_intrusion_ah: number | null
  preceding_discharge_net_ah: number | null
  discharge_derived_soh_proxy_pct: number | null
  soh_proxy_semantics: string
  bms_soh_before_pct: number | null
  bms_soh_after_pct: number | null
  bms_minus_capacity_proxy_pp: number | null
  measurement_quality: string
  reference_comparability: string
  interpretation: string
}

export interface CapacityBmsMatchCandidate {
  transition_id: string
  discharge_event_id: string
  evidence_class: CapacityEvidenceClass
  measurement_quality: string
  soc_start_pct: number
  soc_end_pct: number
  soc_span_pp: number
  gross_discharged_ah: number
  charge_intrusion_ah: number
  net_discharged_ah: number
  time_to_transition_h: number
  within_12h: boolean
  within_24h: boolean
  within_48h: boolean
  within_72h: boolean
  evidence_rank: number
  proximity_rank: number
  is_primary_capacity_evidence: boolean
  is_nearest_qualifying_candidate: boolean
}

export interface RawDischargeTrajectory {
  transition_id: string
  trajectory_id: string
  trajectory_start_utc: string
  trajectory_end_utc: string
  trajectory_end_reason: string
  continuity: string
  soc_start_pct: number
  soc_end_pct: number
  soc_span_pp: number
  evidence_class: string
  discharged_ah: number
  charge_intrusion_ah: number
  rest_minutes: number
  longest_rest_minutes: number
  rest_interruption_count: number
  coverage_fraction: number
  maximum_gap_s: number
  delay_to_transition_hours: number
  within_12h: boolean
  within_24h: boolean
  within_48h: boolean
  within_72h: boolean
  direct_capacity_proxy_eligible: boolean
  supporting_capacity_estimate_eligible: boolean
  capacity_soh_pct: number
}

export interface DownwardTrajectoryMatch {
  transition_id: string
  trajectory_context_status: string
  deepest_trajectory_id?: string | null
  deepest_evidence_class?: string | null
  deepest_soc_start_pct?: number | null
  deepest_soc_end_pct?: number | null
  deepest_soc_span_pp?: number | null
  deepest_delay_hours?: number | null
  deepest_trajectory_end_reason?: string | null
  deepest_continuity?: string | null
  nearest_trajectory_id?: string | null
  nearest_delay_hours?: number | null
  evidence_within_12h?: boolean
  evidence_within_24h?: boolean
  evidence_within_48h?: boolean
  evidence_within_72h?: boolean
  trajectory_candidate_count?: number
  qualifying_trajectory_count?: number
}

export interface CompactEndpointReconciliation {
  compact_endpoint_id: string
  compact_end_utc: string
  compact_start_soc_pct: number | null
  compact_end_soc_pct: number | null
  reconciliation_class: string
  raw_min_soc_pct_near_endpoint: number | null
  raw_min_soc_time_utc: string | null
  raw_min_soc_offset_seconds: number | null
  invalid_all_zero_rows_near_endpoint: number
  matched_trajectory_id: string | null
  one_second_rows_near_endpoint: number
}

export interface BmsSohRecalibrationSeries {
  serial: string
  run_id: string
  detector_version: string | null
  summary: {
    chemistry: BatteryChemistry
    module_count: number
    nominal_capacity_ah: number
    raw_discharge_detector_version?: string | null
    observed_start: string | null
    observed_end: string | null
    feature_observed_start?: string | null
    feature_observed_end?: string | null
    soh_observed_start?: string | null
    soh_observed_end?: string | null
    analysis_overlap_start?: string | null
    analysis_overlap_end?: string | null
    aggregation_minutes: number
    current_sign_convention?: {
      source_signal: string
      charging: string
      discharging: string
      rest: string
      charge_ah: string
      discharge_ah: string
    }
    event_detection_method: string
    event_time_semantics: string
    event_rows: number
    step_drop_rows: number
    step_recovery_rows: number
    change_after_gap_rows: number
    high_confidence_rows: number
    strong_context_match_rows?: number
    limited_context_rows: number
    exact_transition_rows?: number
    interval_censored_transition_rows?: number
    daily_fallback_transition_rows?: number
    exact_transition_rate?: number | null
    gross_step_drop_pp?: number
    gross_step_recovery_pp?: number
    net_bms_soh_change_pp?: number
    event_rate_per_observed_year?: number | null
    event_rate_per_1000_discharge_ah?: number | null
    median_context_coverage_fraction?: number | null
    matched_control_rows?: number
    controls_per_event_requested?: number
    classification_counts: Record<string, number>
    evidence_tag_counts?: Record<string, number>
    strongest_matched_control_enrichment?: BmsSohEvidenceComparison | null
    primary_discharge_hypothesis?: BmsSohEvidenceComparison | null
    primary_discharge_hypothesis_method?: string
    complete_discharge_event_rows?: number
    qualified_calibration_discharge_rows?: number
    one_second_capacity_validation?: BmsSohCapacityValidationSummary
    raw_capacity_validation_scan?: Record<string, number>
    interpretable_model?: BmsSohInterpretableModel
    raw_transition_scan?: Record<string, number>
    findings?: string[]
    control_method?: string
    claim_limit: string
    configuration: Record<string, number>
    source_soh_audit_run_id: string | null
    source_soh_audit_label_status: string | null
    source_capacity_evidence_run_id?: string | null
    phase2?: {
      available: boolean
      model_family: 'capacity_bms_validation'
      phase2_version: string
      source_validation_run_id: string
      source_validation_fingerprint: string
      source_physical_catalog_run_id: string
      discharge_event_rows: number
      capacity_claim_eligible_rows: number
      meaningful_80pp_discharge_rows: number
      capacity_estimate_90pp_rows: number
      strict_full_discharge_rows: number
      near_full_discharge_rows: number
      operational_full_discharge_rows: number
      context_only_discharge_rows: number
      high_quality_capacity_rows: number
      soh_transition_rows: number
      downward_soh_transition_rows?: number
      downward_transitions_with_raw_trajectory_rows?: number
      raw_trajectory_rows?: number
      raw_trajectory_evidence_within_12h_rows?: number
      raw_trajectory_evidence_within_24h_rows?: number
      raw_trajectory_evidence_within_48h_rows?: number
      raw_trajectory_evidence_within_72h_rows?: number
      compact_zero_endpoint_rows?: number
      compact_zero_endpoint_classification_counts?: Record<string, number>
      transitions_with_preceding_discharge_rows: number
      qualifying_discharges_without_soh_update_rows: number
      claim_limit: string
    }
  }
  points: BmsSohRecalibrationPoint[]
  events: BmsSohRecalibrationEvent[]
  comparison: BmsSohEvidenceComparison[]
  capacity_validation: BmsSohCapacityValidation[]
  capacity_events: CapacityBmsDischargeEvent[]
  soh_transitions: CapacityBmsSohTransition[]
  discharge_matches: CapacityBmsDischargeMatch[]
  match_candidates: CapacityBmsMatchCandidate[]
  raw_discharge_trajectories: RawDischargeTrajectory[]
  downward_trajectory_matches: DownwardTrajectoryMatch[]
  transition_trajectory_candidates: RawDischargeTrajectory[]
  compact_endpoint_reconciliation: CompactEndpointReconciliation[]
  electrothermal_context: ElectrothermalCapacityContext[]
  electrothermal_summary: {
    phase3_version: string
    selected_event_rows: number
    measured_context_ready_rows: number
    reference_comparability_counts: Record<string, number>
    claim_limit: string
  } | null
  reference_comparison: BmsSohReferenceComparison
}

export interface CellAnomalyOverviewDevice {
  serial: string
  display_name: string
  variant_key: string | null
  device_role: 'field_tester' | 'full_cycle_test'
  cohort_role: 'field_device' | 'reference_control'
  module_count: number | null
  module_count_status: string
  cohort_status: 'included' | 'deferred'
  cohort_reason: string
  in_active_cohort: boolean
  history_built: boolean
  legacy_history_archived: boolean
  analyzed: boolean
  legacy_analysis_archived: boolean
  detector_version: 'hierarchical_v2'
  analysis_start: string | null
  analysis_end: string | null
  feature_rows: number | null
  quality_valid_pct: number | null
  scored_pct_of_valid: number | null
  sustained_anomaly_episode_count: number
  anomaly_days: number
  maximum_anomaly_score: number | null
  p95_analysis_spread_mv: number | null
  maximum_analysis_spread_mv: number | null
  module_configuration_epoch_count: number
  elevated_module_epoch_shift_count: number
  maximum_module_epoch_baseline_shift_mv: number | null
  first_episode_start: string | null
  latest_episode_end: string | null
  source_feature_run_id: string | null
  source_run_id: string | null
  source_modified_at: string | null
}

export interface CellAnomalyOverview {
  model_family: 'nca_cell_voltage_anomaly_overview'
  chemistry: 'NCA'
  generated_at: string
  latest_source_modified_at: string | null
  cohort: {
    active_module_count: number
    policy: string
  }
  metrics: CellAnomalyOverviewMetrics
  devices: CellAnomalyOverviewDevice[]
  evidence_policy: {
    method: string
    conditioning: string
    ground_truth_status: string
    claim_limit: string
  }
}

export interface CellAnomalySeriesPoint {
  time: string
  module_count: number | null
  module_configuration_epoch: number | null
  spread_p50_mv: number | null
  spread_p95_mv: number | null
  spread_max_mv: number | null
  raw_spread_p95_mv: number | null
  compensated_spread_p95_mv: number | null
  baseline_threshold_median_mv: number | null
  baseline_threshold_p95_mv: number | null
  point_candidate_bins: number
  sustained_candidate_bins: number
  sustained_episode_minutes: number | null
  sustained_charging_minutes: number
  sustained_discharging_minutes: number
  sustained_rest_minutes: number
  sustained_mixed_minutes: number
  maximum_anomaly_score: number | null
  soc_mean_pct: number | null
  temperature_mean_c: number | null
}

export interface CellAnomalyEpisode {
  episode_id: string
  start_time: string
  end_time: string
  duration_minutes: number
  candidate_bins?: number
  module_configuration_epoch?: number | null
  module_count?: number | null
  operating_states?: string
  detection_reasons?: string
  contextual_candidate_bins?: number
  absolute_candidate_bins?: number
  maximum_spread_mv?: number | null
  median_spread_mv?: number | null
  median_baseline_mv?: number | null
  maximum_anomaly_score?: number | null
  mean_soc_pct?: number | null
  mean_current_a?: number | null
  mean_temperature_c?: number | null
  severity?: string
  affected_channel_key?: string | null
  affected_module_index?: number | null
  affected_pack_index?: number | null
  affected_channel_index?: number | null
  maximum_peer_z?: number | null
  maximum_low_rank_z?: number | null
  maximum_temporal_ewma_score?: number | null
  minimum_temperature_c?: number | null
  maximum_temperature_c?: number | null
  explanation?: string | null
  has_raw_evidence?: boolean
}

export interface CellAnomalySeries {
  serial: string
  run_id: string
  detector_version: 'hierarchical_v2'
  summary: {
    analysis_start: string | null
    analysis_end: string | null
    feature_rows?: number
    quality_valid_rows?: number
    quality_valid_pct: number | null
    sustained_anomaly_episode_count: number
    anomaly_days: number
    p95_analysis_spread_mv?: number | null
    maximum_analysis_spread_mv?: number | null
    maximum_bms_reported_voltage_spread_mv?: number | null
    bms_global_spread_implausible_windows?: number
    spread_basis?: string | null
    episode_severity_counts?: Record<string, number>
    evidence_episode_count?: number
    evidence_requested_episode_count?: number
    evidence_status?: 'complete' | 'partial' | 'unavailable' | 'disabled' | 'not_required' | null
    evidence_error?: string | null
    first_episode_start?: string | null
    latest_episode_end?: string | null
    module_epoch_baselines: Array<Record<string, unknown>>
    engineering_bands_mv?: Record<string, [number, number | null]>
    ground_truth_status?: string | null
    claim_limit: string | null
    context_value_semantics?: string | null
  }
  points: CellAnomalySeriesPoint[]
  episodes: CellAnomalyEpisode[]
}

export interface SohGapDiagnosis {
  run_id: string
  summary: Record<string, unknown> & {
    serial: string
    reference_serial: string
    phase4_version: string
    field_nameplate_capacity_ah: number
    legacy_phase3_candidate_rows: number
    canonical_capacity_observation_rows: number
    soh_transition_rows: number
    candidate_reconciliation_counts: Record<string, number>
    reference_comparability_counts: Record<string, number>
    diagnostic_classification_counts: Record<string, number>
    uncertainty_scope: string
    uncertainty_excludes: string
    claim_limit: string
  }
  manifest: Record<string, unknown>
  observations: Array<Record<string, unknown>>
  transitions: Array<Record<string, unknown>>
  assessments: Array<Record<string, unknown>>
  stress_intervals: Array<Record<string, unknown>>
  device_month_stress: Array<Record<string, unknown>>
  verification_points: Array<Record<string, unknown>>
  candidate_reconciliation: Array<Record<string, unknown>>
}
