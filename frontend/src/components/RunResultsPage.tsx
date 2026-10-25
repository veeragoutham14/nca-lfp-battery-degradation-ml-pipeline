import {
  Activity,
  ArrowLeft,
  ChartNoAxesCombined,
  CircleGauge,
  Database,
  FileOutput,
  Gauge,
  Rows3,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type {
  AnalysisRun,
  Artifact,
  Dataset,
  RunSeries,
  ValidationSeries,
} from '../types'
import { compactNumber, familyLabel, integer, number, shortDate, shortDateTime } from '../utils'
import { ArtifactTable } from './ArtifactTable'
import { ForecastChart } from './ForecastChart'
import { MetricCard } from './MetricCard'
import { SeasonalChargeProfileChart } from './SeasonalChargeProfileChart'
import { SeasonalSohTransitionTable } from './SeasonalSohTransitionTable'
import { ValidationChart } from './ValidationChart'

type ResultsTab = 'overview' | 'validation' | 'files'

interface RunResultsPageProps {
  runId: string
  onBack: () => void
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function numeric(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function statusTone(mae: number | null): 'neutral' | 'good' | 'warning' | 'critical' {
  if (mae == null) return 'neutral'
  if (mae <= 1) return 'good'
  if (mae <= 3) return 'warning'
  return 'critical'
}

export function RunResultsPage({ runId, onBack }: RunResultsPageProps) {
  const [run, setRun] = useState<AnalysisRun | null>(null)
  const [summary, setSummary] = useState<Record<string, unknown>>({})
  const [series, setSeries] = useState<RunSeries | null>(null)
  const [validation, setValidation] = useState<ValidationSeries | null>(null)
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [activeTab, setActiveTab] = useState<ResultsTab>('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    setActiveTab('overview')
    setLoading(true)
    setError('')
    Promise.all([
      api.run(runId),
      api.summary(runId),
      api.series(runId),
      api.validation(runId),
      api.artifacts(runId),
      api.datasets(runId),
    ])
      .then(([runData, summaryData, seriesData, validationData, artifactData, datasetData]) => {
        if (cancelled) return
        setRun(runData)
        setSummary(summaryData)
        setSeries(seriesData)
        setValidation(validationData)
        setArtifacts(artifactData)
        setDatasets(datasetData)
      })
      .catch((reason) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [runId])

  const isFixedSlope = run?.model_family === 'fixed_slope_transfer_verification'
  const isSohAudit = run?.model_family === 'bms_soh_signal_audit'
  const isCellSignalAvailability = run?.model_family === 'cell_signal_availability'
  const isCellHealthFeatures = run?.model_family === 'cell_health_features'
  const isCellVoltageAnomaly = run?.model_family === 'nca_cell_voltage_anomaly'
  const isElectrothermalSource = run?.model_family === 'bms_electrothermal_source_quality'
  const isHierarchicalFeatures = run?.model_family === 'bms_hierarchical_electrothermal_features'
  const isHierarchicalAnomaly = run?.model_family === 'hierarchical_cell_voltage_anomaly'
  const isSeasonalCharge = run?.model_family === 'seasonal_charging_stress_profile'
  const isCellAnalysis = isCellSignalAvailability || isCellHealthFeatures || isCellVoltageAnomaly
    || isElectrothermalSource || isHierarchicalFeatures || isHierarchicalAnomaly
  const degradation = record(summary.degradation_forecast_model)
  const cutoff = record(summary.cutoff_validation)
  const comparison = record(summary.slope_comparison)
  const bySlope = record(comparison.by_slope)
  const quality = record(summary.data_quality)
  const audit = record(summary.audit_metrics)
  const anomalyParameters = record(summary.parameters)
  const detector = record(summary.detector)
  const seasonalConfiguration = record(summary.configuration)
  const provenance = record(summary.provenance)
  const reconciliation = record(summary.reconciliation)
  const epochBaselines = Array.isArray(summary.module_epoch_baselines)
    ? summary.module_epoch_baselines.map(record)
    : []
  const primaryEpoch = epochBaselines[0] ?? {}
  const bestSlope = String(comparison.best_loss_slope_per_1000ah ?? '')
  const bestMetrics = record(bySlope[bestSlope])
  const latestHistory = series?.history.at(-1)
  const finalForecast = series?.forecasts[0]?.points.at(-1)
  const seasonalTransitions = series?.soh_transitions ?? []
  const nearFullTransitionCount = seasonalTransitions.filter(
    (item) => item.near_full_charge_within_72h_before,
  ).length
  const comparisonRows = useMemo(
    () => Object.values(bySlope).map(record).sort((left, right) =>
      Number(left.mae_soh_pct ?? Number.POSITIVE_INFINITY) - Number(right.mae_soh_pct ?? Number.POSITIVE_INFINITY),
    ),
    [bySlope],
  )
  const reportStart = isSeasonalCharge
    ? String(summary.observed_start ?? '')
    : isElectrothermalSource || isCellSignalAvailability
    ? String(summary.sample_start ?? '')
    : isHierarchicalFeatures || isCellHealthFeatures
      ? String(summary.start ?? '')
      : isHierarchicalAnomaly || isCellVoltageAnomaly
        ? String(summary.analysis_start ?? '')
        : run?.observed_start_at
  const reportEnd = isSeasonalCharge
    ? String(summary.observed_end ?? '')
    : isElectrothermalSource || isCellSignalAvailability
    ? String(summary.sample_end ?? '')
    : isHierarchicalFeatures || isCellHealthFeatures
      ? String(summary.end ?? '')
      : isHierarchicalAnomaly || isCellVoltageAnomaly
        ? String(summary.analysis_end ?? '')
        : run?.observed_end_at

  if (loading) return <div className="section-loader full-page-loader">Loading analysis report...</div>

  if (error || !run) {
    return (
      <div className="empty-workspace">
        <FileOutput size={26} aria-hidden="true" />
        <h1>Report unavailable</h1>
        <p>{error || 'The requested run is not in the catalog.'}</p>
        <button className="button secondary" type="button" onClick={onBack}><ArrowLeft size={16} aria-hidden="true" />Back to machine learning</button>
      </div>
    )
  }

  return (
    <div className="page-stack results-page">
      <header className="results-header">
        <button className="back-button" type="button" onClick={onBack}><ArrowLeft size={17} aria-hidden="true" />Machine learning</button>
        <div className="results-title-row">
          <div>
            <span className="eyebrow">Battery {run.serial}</span>
            <h1>{familyLabel(run.model_family)}</h1>
            <span className="run-window">{shortDate(reportStart)} to {shortDate(reportEnd)}</span>
          </div>
          <div className="run-identity"><span>Run</span><strong>{shortDateTime(run.run_created_at)}</strong></div>
        </div>
      </header>

      <nav className="tabs results-tabs" aria-label="Report views">
        <button className={activeTab === 'overview' ? 'active' : ''} type="button" onClick={() => setActiveTab('overview')}><ChartNoAxesCombined size={16} aria-hidden="true" />Overview</button>
        {!isSohAudit && !isCellAnalysis && !isSeasonalCharge && <button className={activeTab === 'validation' ? 'active' : ''} type="button" onClick={() => setActiveTab('validation')}><CircleGauge size={16} aria-hidden="true" />Validation</button>}
        <button className={activeTab === 'files' ? 'active' : ''} type="button" onClick={() => setActiveTab('files')}><FileOutput size={16} aria-hidden="true" />Files</button>
      </nav>

      {activeTab === 'overview' && (
        <div className="tab-content">
          {isSeasonalCharge ? (
            <section className="metric-strip">
              <MetricCard label="Charge episodes" value={integer(numeric(summary.charge_episode_rows))} detail={`${integer(numeric(summary.daily_stress_rows))} daily profiles`} icon={Activity} />
              <MetricCard label="Qualified months" value={`${integer(numeric(summary.valid_monthly_profile_rows))} / ${integer(numeric(summary.monthly_profile_rows))}`} detail="Coverage-gated seasonal evidence" icon={Rows3} tone={(numeric(summary.valid_monthly_profile_rows) ?? 0) > 0 ? 'good' : 'warning'} />
              <MetricCard label="SOH transitions" value={integer(numeric(summary.soh_transition_rows))} detail={`${integer(numeric(summary.context_ready_transition_rows))} with complete context`} icon={Gauge} tone={(numeric(summary.soh_transition_rows) ?? 0) > 0 ? 'warning' : 'neutral'} />
              <MetricCard label="Charge throughput" value={`${compactNumber(numeric(summary.total_charge_ah))} Ah`} detail={`${compactNumber(numeric(summary.total_discharge_ah))} Ah discharged · ${number(numeric(summary.total_discharge_equivalent_full_cycles), 1)} EFC`} icon={Database} />
            </section>
          ) : isElectrothermalSource ? (
            <section className="metric-strip">
              <MetricCard label="Voltage channels" value={`${integer(numeric(summary.ready_voltage_channels))} / ${integer(numeric(summary.expected_voltage_channels))}`} detail={`${number(numeric(summary.readiness_threshold_pct), 0)}% valid coverage required`} icon={Rows3} tone={summary.source_quality_ready === true ? 'good' : 'critical'} />
              <MetricCard label="Pack-temperature sensors" value={`${integer(numeric(summary.ready_temperature_sensors))} / ${integer(numeric(summary.expected_temperature_sensors))}`} detail="Installed hierarchy only" icon={Activity} tone={summary.source_quality_ready === true ? 'good' : 'critical'} />
              <MetricCard label="Lowest voltage validity" value={`${number(numeric(summary.minimum_voltage_channel_valid_coverage_pct), 1)}%`} detail="Worst installed voltage channel" icon={CircleGauge} />
              <MetricCard label="Source quality" value={String(summary.quality_gate_status ?? 'unknown')} detail={`${integer(numeric(summary.module_count))} installed modules`} icon={Database} tone={summary.source_quality_ready === true ? 'good' : 'critical'} />
            </section>
          ) : isHierarchicalFeatures ? (
            <section className="metric-strip">
              <MetricCard label="Five-minute windows" value={compactNumber(numeric(summary.feature_rows))} detail={`${integer(numeric(summary.aggregation_minutes))}-minute database aggregation`} icon={Rows3} />
              <MetricCard label="Source samples" value={compactNumber(numeric(summary.source_sample_rows))} detail="Rows reduced in PostgreSQL" icon={Database} />
              <MetricCard label="Voltage channels" value={integer(numeric(summary.voltage_channel_count))} detail="Identity preserved by module and pack" icon={CircleGauge} />
              <MetricCard label="Temperature sensors" value={integer(numeric(summary.temperature_sensor_count))} detail={`${integer(numeric(summary.partition_count))} monthly partitions`} icon={Gauge} />
            </section>
          ) : isHierarchicalAnomaly ? (
            <section className="metric-strip">
              <MetricCard label="Sustained episodes" value={integer(numeric(summary.sustained_anomaly_episode_count))} detail={`${integer(numeric(summary.anomaly_days))} affected days`} icon={Activity} tone={numeric(summary.sustained_anomaly_episode_count) ? 'warning' : 'good'} />
              <MetricCard label="Spread p95" value={`${number(numeric(summary.p95_analysis_spread_mv), 1)} mV`} detail={`${number(numeric(summary.maximum_analysis_spread_mv), 1)} mV maximum`} icon={Gauge} />
              <MetricCard label="Quality-valid windows" value={`${number(numeric(summary.quality_valid_pct), 1)}%`} detail={`${compactNumber(numeric(summary.quality_valid_rows))} usable windows`} icon={CircleGauge} />
              <MetricCard label="Raw evidence" value={integer(numeric(summary.evidence_episode_count))} detail="Bounded one-second episode windows" icon={Database} />
            </section>
          ) : isCellSignalAvailability ? (
            <section className="metric-strip">
              <MetricCard label="Ready cell channels" value={`${integer(numeric(summary.ready_cell_channels))} / ${integer(numeric(summary.expected_cell_channels))}`} detail={`${number(numeric(summary.readiness_threshold_pct), 0)}% coverage required`} icon={Rows3} tone={summary.cell_voltage_ready === true ? 'good' : 'critical'} />
              <MetricCard label="Nominal 1 Hz coverage" value={`${number(numeric(summary.nominal_1hz_coverage_pct), 1)}%`} detail={`${compactNumber(numeric(summary.total_rows))} received samples`} icon={Activity} tone={(numeric(summary.nominal_1hz_coverage_pct) ?? 0) >= 95 ? 'good' : 'warning'} />
              <MetricCard label="Minimum channel validity" value={`${number(numeric(summary.minimum_cell_channel_valid_coverage_pct), 1)}%`} detail="Worst individual cell channel" icon={CircleGauge} tone={(numeric(summary.minimum_cell_channel_valid_coverage_pct) ?? 0) >= 95 ? 'good' : 'critical'} />
              <MetricCard label="Installed modules" value={integer(numeric(summary.module_count))} detail={`${integer(numeric(summary.present_cell_channels))} channels present`} icon={Database} />
            </section>
          ) : isCellHealthFeatures ? (
            <section className="metric-strip">
              <MetricCard label="Five-minute windows" value={compactNumber(numeric(summary.feature_rows))} detail={`${integer(numeric(summary.aggregation_minutes))}-minute aggregation`} icon={Rows3} />
              <MetricCard label="Source samples" value={compactNumber(numeric(summary.source_sample_rows))} detail="One-second telemetry processed" icon={Database} />
              <MetricCard label="Valid voltage samples" value={`${number(numeric(summary.valid_cell_voltage_sample_pct), 3)}%`} detail={`${compactNumber(numeric(summary.valid_cell_voltage_sample_rows))} usable rows`} icon={CircleGauge} tone={(numeric(summary.valid_cell_voltage_sample_pct) ?? 0) >= 95 ? 'good' : 'critical'} />
              <MetricCard label="Observed modules" value={integer(numeric(summary.observed_module_count))} detail="Stable configuration used by features" icon={Gauge} />
            </section>
          ) : isCellVoltageAnomaly ? (
            <section className="metric-strip">
              <MetricCard label="Sustained episodes" value={integer(numeric(summary.sustained_anomaly_episode_count))} detail={`${integer(numeric(summary.anomaly_days))} affected days`} icon={Activity} tone={numeric(summary.sustained_anomaly_episode_count) ? 'warning' : 'good'} />
              <MetricCard label="Spread p95" value={`${number(numeric(summary.p95_analysis_spread_mv), 1)} mV`} detail={`${number(numeric(summary.maximum_analysis_spread_mv), 1)} mV maximum`} icon={Gauge} />
              <MetricCard label="Quality-valid windows" value={`${number(numeric(summary.quality_valid_pct), 3)}%`} detail={`${compactNumber(numeric(summary.quality_valid_rows))} five-minute windows`} icon={CircleGauge} tone={(numeric(summary.quality_valid_pct) ?? 0) >= 95 ? 'good' : 'warning'} />
              <MetricCard label="Module epochs" value={integer(numeric(summary.module_configuration_epoch_count))} detail={`${integer(numeric(summary.elevated_module_epoch_shift_count))} elevated shifts`} icon={Database} tone={numeric(summary.elevated_module_epoch_shift_count) ? 'warning' : 'good'} />
            </section>
          ) : isSohAudit ? (
            <section className="metric-strip">
              <MetricCard label="Latest BMS SOH" value={`${number(numeric(audit.last_bms_soh_pct), 1)}%`} detail={shortDate(latestHistory?.timestamp)} icon={Gauge} />
              <MetricCard label="Stable drops" value={integer(numeric(audit.step_drop_count))} detail={`${number(Math.abs(numeric(audit.largest_step_drop_pct) ?? 0), 1)} pp largest`} icon={Activity} tone={numeric(audit.step_drop_count) ? 'warning' : 'good'} />
              <MetricCard label="SOH recoveries" value={integer(numeric(audit.step_recovery_count))} detail="Reversible BMS movement" icon={Rows3} tone={numeric(audit.step_recovery_count) ? 'warning' : 'good'} />
              <MetricCard label="Median coverage" value={`${number((numeric(audit.median_coverage_fraction) ?? 0) * 100, 1)}%`} detail={`${integer(run.training_rows)} qualified days`} icon={Database} />
            </section>
          ) : isFixedSlope ? (
            <section className="metric-strip">
              <MetricCard label="Latest BMS SOH" value={`${number(latestHistory?.soh_pct, 2)}%`} detail={shortDate(latestHistory?.timestamp)} icon={Gauge} />
              <MetricCard label="Recorded discharge" value={`${compactNumber(numeric(summary.actual_cumulative_discharge_ah))} Ah`} detail="All discharging events" icon={Database} />
              <MetricCard label="Best verification MAE" value={`${number(numeric(bestMetrics.mae_soh_pct), 2)}%`} detail={`${number(numeric(comparison.best_loss_slope_per_1000ah), 4)} Ah / 1,000 Ah`} icon={CircleGauge} tone={statusTone(numeric(bestMetrics.mae_soh_pct))} />
              <MetricCard label="Within BMS tolerance" value={`${number(numeric(bestMetrics.within_bms_tolerance_pct), 1)}%`} detail={`Tolerance +/-${number(numeric(summary.bms_tolerance_pct), 1)}%`} icon={Rows3} />
            </section>
          ) : (
            <section className="metric-strip">
              <MetricCard label="Latest measured SOH" value={`${number(latestHistory?.soh_pct, 2)}%`} detail={shortDate(latestHistory?.timestamp)} icon={Gauge} />
              <MetricCard label="Loss slope" value={`${number(run.loss_slope_per_1000ah, 4)} Ah`} detail="Per 1,000 discharged Ah" icon={Activity} />
              <MetricCard label="Validation MAE" value={`${number(run.validation_mae_soh_pct, 3)}%`} detail={`${integer(run.validation_rows)} cutoff rows`} icon={CircleGauge} tone={statusTone(run.validation_mae_soh_pct)} />
              <MetricCard label="Forecast endpoint" value={`${number(finalForecast?.soh_pct, 2)}%`} detail={shortDate(finalForecast?.timestamp)} icon={ChartNoAxesCombined} />
            </section>
          )}

          {!isCellAnalysis && !isSeasonalCharge && <section className="panel chart-panel">
            <div className="section-title"><div><h2>{isSohAudit ? 'Daily BMS SOH evidence' : isFixedSlope ? 'Transferred slope comparison' : 'Capacity-based SOH forecast'}</h2><span>{isSohAudit ? 'Screened daily state, with step changes retained as diagnostic evidence' : isFixedSlope ? 'BMS SOH against fixed slopes using actual cumulative discharge Ah' : 'Measured history and usage-rate forecasts'}</span></div></div>
            <ForecastChart data={series} />
          </section>}

          {isSeasonalCharge ? (
            <>
              <section className="panel chart-panel seasonal-profile-panel">
                <div className="section-title"><div><h2>Monthly charging and operating profile</h2><span>Coverage-qualified exposure with BMS SOH overlaid in UTC</span></div></div>
                <SeasonalChargeProfileChart
                  points={series?.seasonal_profile ?? []}
                  socExposureMethod={String(summary.soc_exposure_method ?? '')}
                />
              </section>
              <section className="panel seasonal-transition-panel">
                <div className="section-title"><div><h2>BMS SOH transition evidence</h2><span>{seasonalTransitions.length > 0 ? `${nearFullTransitionCount} of ${seasonalTransitions.length} transitions were observed within 72 h after a near-full charge` : 'Measured conditions in the seven days before and after each audited change'}</span></div></div>
                <SeasonalSohTransitionTable rows={seasonalTransitions} />
              </section>
              <section className="panel setup-panel">
              <div className="section-title"><div><h2>Profile evidence contract</h2><span>Measured charging behavior and exposure, separated from causal claims</span></div></div>
              <dl className="definition-grid">
                <div><dt>Chemistry</dt><dd>{String(summary.chemistry ?? 'unknown')}</dd></div>
                <div><dt>Installed modules</dt><dd>{integer(numeric(summary.module_count))}</dd></div>
                <div><dt>Nominal capacity</dt><dd>{number(numeric(summary.nominal_capacity_ah), 1)} Ah</dd></div>
                <div><dt>Time basis</dt><dd>{String(summary.timestamp_semantics ?? 'UTC')}</dd></div>
                <div><dt>Compact interval</dt><dd>{integer(numeric(summary.aggregation_minutes))} min</dd></div>
                <div><dt>High-current threshold</dt><dd>{number(numeric(seasonalConfiguration.high_current_c_rate_threshold), 2)} C</dd></div>
                <div><dt>Monthly coverage gate</dt><dd>{number((numeric(seasonalConfiguration.minimum_month_coverage) ?? 0) * 100, 0)}%</dd></div>
                <div><dt>SOH context window</dt><dd>{integer(numeric(seasonalConfiguration.transition_window_days))} days before/after</dd></div>
                <div><dt>High-SOC exposure</dt><dd>{number(numeric(summary.total_high_soc_95_hours), 1)} h at or above 95%</dd></div>
                <div><dt>Maximum pack temperature</dt><dd>{number(numeric(summary.maximum_stable_pack_temperature_c), 1)} C</dd></div>
                <div><dt>SOH audit source</dt><dd>{String(summary.source_soh_audit_run_id ?? 'unknown')}</dd></div>
                <div><dt>Stable temperature sensors</dt><dd>{integer(numeric(summary.stable_temperature_sensor_count))}</dd></div>
              </dl>
              {(numeric(summary.maximum_stable_pack_temperature_c) ?? 0) >= 60 && (
                <div className="inline-notice evidence-notice"><strong>Temperature outlier</strong><span>The recorded maximum is unusually high relative to this device profile. Verify the exact raw sensor interval before interpreting it as physical battery temperature.</span></div>
              )}
              <div className="inline-notice evidence-notice"><strong>Interpretation limit</strong><span>{String(summary.claim_limit ?? '')}</span></div>
              </section>
            </>
          ) : isElectrothermalSource ? (
            <section className="panel setup-panel">
              <div className="section-title"><div><h2>Source-quality decision</h2><span>Database schema, signal behavior, and BMS-extrema reconciliation</span></div></div>
              <dl className="definition-grid">
                <div><dt>Voltage extrema</dt><dd>{summary.voltage_extrema_reconciled === true ? 'Reconciled' : 'Mismatch or unavailable'}</dd></div>
                <div><dt>Temperature extrema</dt><dd>{summary.temperature_extrema_reconciled === true ? 'Reconciled' : 'Mismatch or unavailable'}</dd></div>
                <div><dt>Duplicate timestamps</dt><dd>{number(numeric(summary.duplicate_timestamp_pct), 3)}%</dd></div>
                <div><dt>Nominal 1 Hz coverage</dt><dd>{number(numeric(summary.nominal_1hz_coverage_pct), 1)}%</dd></div>
                <div><dt>Voltage p95 extrema error</dt><dd>{number(Math.max(numeric(reconciliation.voltage_min_p95_error_mv) ?? 0, numeric(reconciliation.voltage_max_p95_error_mv) ?? 0), 1)} mV</dd></div>
                <div><dt>Temperature p95 extrema error</dt><dd>{number(Math.max(numeric(reconciliation.temperature_min_p95_error_c) ?? 0, numeric(reconciliation.temperature_max_p95_error_c) ?? 0), 2)} C</dd></div>
                <div><dt>Database schema</dt><dd>{String(provenance.database_schema_sha256 ?? 'unknown').slice(0, 12)}</dd></div>
                <div><dt>Timestamp semantics</dt><dd>{String(provenance.timestamp_semantics ?? 'UTC')}</dd></div>
              </dl>
              <div className="inline-notice evidence-notice"><strong>Decision</strong><span>{summary.source_quality_ready === true ? 'Passed. Hierarchical feature generation is allowed.' : `Blocked: ${(Array.isArray(summary.blocking_reasons) ? summary.blocking_reasons : []).join(', ') || 'source quality requirements not met'}`}</span></div>
            </section>
          ) : isHierarchicalFeatures ? (
            <section className="panel setup-panel">
              <div className="section-title"><div><h2>Hierarchical feature contract</h2><span>Compact BMS evidence prepared without losing module, pack, or channel identity</span></div></div>
              <dl className="definition-grid">
                <div><dt>Source audit</dt><dd>{String(summary.source_audit_run ?? 'unknown')}</dd></div>
                <div><dt>Quality gate</dt><dd>{String(summary.source_quality_gate ?? 'unknown')}</dd></div>
                <div><dt>Feature start</dt><dd>{shortDateTime(String(summary.start ?? ''))}</dd></div>
                <div><dt>Feature end</dt><dd>{shortDateTime(String(summary.end ?? ''))}</dd></div>
                <div><dt>Monthly partitions</dt><dd>{integer(numeric(summary.partition_count))}</dd></div>
                <div><dt>Installed modules</dt><dd>{integer(numeric(summary.module_count))}</dd></div>
              </dl>
              <div className="inline-notice evidence-notice"><strong>Interpretation limit</strong><span>{String(summary.claim_limit ?? '')}</span></div>
            </section>
          ) : isHierarchicalAnomaly ? (
            <section className="panel setup-panel">
              <div className="section-title"><div><h2>Detection evidence</h2><span>Fixed engineering severity plus contextual and persistent evidence</span></div></div>
              <dl className="definition-grid">
                <div><dt>Point candidates</dt><dd>{integer(numeric(summary.point_anomaly_windows))}</dd></div>
                <div><dt>First episode</dt><dd>{shortDateTime(String(summary.first_episode_start ?? ''))}</dd></div>
                <div><dt>Latest episode</dt><dd>{shortDateTime(String(summary.latest_episode_end ?? ''))}</dd></div>
                <div><dt>Pack-peer maximum z</dt><dd>{number(numeric(summary.maximum_peer_z), 1)}</dd></div>
                <div><dt>Low-rank maximum z</dt><dd>{number(numeric(summary.maximum_low_rank_z), 1)}</dd></div>
                <div><dt>EWMA maximum</dt><dd>{number(numeric(summary.maximum_temporal_ewma_score), 1)}</dd></div>
                <div><dt>Baseline rows</dt><dd>{compactNumber(numeric(detector.baseline_rows))}</dd></div>
                <div><dt>Context threshold</dt><dd>{number(numeric(detector.robust_z_threshold), 1)} z</dd></div>
              </dl>
              <div className="inline-notice evidence-notice"><strong>Interpretation limit</strong><span>{String(summary.claim_limit ?? '')}</span></div>
            </section>
          ) : isCellSignalAvailability ? (
            <section className="panel setup-panel">
              <div className="section-title"><div><h2>Signal readiness</h2><span>Representative telemetry window used as an ingestion gate</span></div></div>
              <dl className="definition-grid">
                <div><dt>Cell-voltage channels</dt><dd>{summary.cell_voltage_ready === true ? 'Ready' : 'Not ready'}</dd></div>
                <div><dt>Global voltage signals</dt><dd>{summary.global_cell_voltage_signals_ready === true ? 'Ready' : 'Not ready'}</dd></div>
                <div><dt>Temperature signals</dt><dd>{summary.cell_temperature_signals_ready === true ? 'Ready' : 'Not ready'}</dd></div>
                <div><dt>Channels in schema</dt><dd>{integer(numeric(summary.present_cell_channels))} / {integer(numeric(summary.expected_cell_channels))}</dd></div>
                <div><dt>First sample</dt><dd>{shortDateTime(String(summary.first_timestamp ?? ''))}</dd></div>
                <div><dt>Last sample</dt><dd>{shortDateTime(String(summary.last_timestamp ?? ''))}</dd></div>
                <div><dt>Mean samples/second</dt><dd>{number(numeric(summary.mean_samples_per_second), 3)}</dd></div>
                <div><dt>Window selection</dt><dd>{summary.sample_selected_automatically === true ? 'Automatic recent day' : 'Operator supplied'}</dd></div>
              </dl>
            </section>
          ) : isCellHealthFeatures ? (
            <section className="panel setup-panel">
              <div className="section-title"><div><h2>Feature envelope</h2><span>Compact operating-condition and cell-spread features prepared for anomaly analysis</span></div></div>
              <dl className="definition-grid">
                <div><dt>Maximum raw spread</dt><dd>{number(numeric(summary.maximum_cell_spread_mv), 1)} mV</dd></div>
                <div><dt>Maximum compensated spread</dt><dd>{number(numeric(summary.maximum_compensated_cell_spread_mv), 1)} mV</dd></div>
                <div><dt>Maximum temperature spread</dt><dd>{number(numeric(summary.maximum_cell_temperature_spread_c), 1)} C</dd></div>
                <div><dt>Invalid voltage samples</dt><dd>{compactNumber((numeric(summary.source_sample_rows) ?? 0) - (numeric(summary.valid_cell_voltage_sample_rows) ?? 0))}</dd></div>
                <div><dt>Feature start</dt><dd>{shortDateTime(String(summary.start ?? ''))}</dd></div>
                <div><dt>Feature end</dt><dd>{shortDateTime(String(summary.end ?? ''))}</dd></div>
              </dl>
              <div className="inline-notice evidence-notice"><strong>Purpose</strong><span>This run prepares quality-controlled features; it does not classify any window as anomalous.</span></div>
            </section>
          ) : isCellVoltageAnomaly ? (
            <>
              <section className="panel setup-panel">
                <div className="section-title"><div><h2>Detection evidence</h2><span>Conditional robust spread candidates and sustained-event grouping</span></div></div>
                <dl className="definition-grid">
                  <div><dt>Point candidates</dt><dd>{integer(numeric(summary.point_anomaly_candidate_rows))}</dd></div>
                  <div><dt>Sustained candidate bins</dt><dd>{integer(numeric(summary.sustained_anomaly_candidate_rows))}</dd></div>
                  <div><dt>First sustained episode</dt><dd>{shortDateTime(String(summary.first_episode_start ?? ''))}</dd></div>
                  <div><dt>Latest sustained episode</dt><dd>{shortDateTime(String(summary.latest_episode_end ?? ''))}</dd></div>
                  <div><dt>Baseline groups</dt><dd>{integer(numeric(summary.baseline_group_count))}</dd></div>
                  <div><dt>Epoch baseline</dt><dd>{number(numeric(primaryEpoch.baseline_median_mv), 1)} mV median</dd></div>
                  <div><dt>Absolute spread floor</dt><dd>{number(numeric(anomalyParameters.absolute_threshold_floor_mv), 1)} mV</dd></div>
                  <div><dt>Minimum sustained duration</dt><dd>{integer((numeric(anomalyParameters.minimum_episode_bins) ?? 0) * (numeric(anomalyParameters.aggregation_minutes) ?? 0))} min</dd></div>
                </dl>
              </section>
              <div className="inline-notice evidence-notice"><strong>Interpretation limit</strong><span>{String(summary.claim_limit ?? 'These are unsupervised system-level anomaly candidates, not confirmed weak-cell labels.')}</span></div>
            </>
          ) : isSohAudit ? (
            <section className="panel setup-panel">
              <div className="section-title"><div><h2>Label decision</h2><span>{String(audit.label_status ?? 'unknown').replaceAll('_', ' ')}</span></div></div>
              <dl className="definition-grid">
                <div><dt>Observed discharge</dt><dd>{compactNumber(numeric(audit.total_discharge_ah))} Ah</dd></div>
                <div><dt>Net BMS SOH change</dt><dd>{number(numeric(audit.net_bms_soh_change_pct), 2)} pp</dd></div>
                <div><dt>Unstable SOH days</dt><dd>{integer(numeric(audit.unstable_soh_days))}</dd></div>
                <div><dt>Changes after gaps</dt><dd>{integer(numeric(audit.change_after_gap_count))}</dd></div>
                <div><dt>BMS movement / 1,000 Ah</dt><dd>{number(numeric(audit.observed_bms_loss_pp_per_1000ah), 3)} pp</dd></div>
                <div><dt>Signal behavior</dt><dd>{String(audit.signal_quality_status ?? 'unknown').replaceAll('_', ' ')}</dd></div>
                <div><dt>Ground-truth status</dt><dd>{String(audit.ground_truth_status ?? 'missing independent capacity label').replaceAll('_', ' ')}</dd></div>
                <div><dt>Ground truth required</dt><dd>{record(summary.label_policy).ground_truth_required === true ? 'Yes' : 'No'}</dd></div>
              </dl>
            </section>
          ) : isFixedSlope ? (
            <section className="panel comparison-panel">
              <div className="section-title"><div><h2>Verified slopes</h2><span>{String(comparison.selection_note ?? '')}</span></div></div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Loss slope / 1,000 Ah</th><th>MAE</th><th>RMSE</th><th>Bias</th><th>Maximum error</th><th>Within tolerance</th></tr></thead>
                  <tbody>
                    {comparisonRows.map((item) => (
                      <tr key={String(item.loss_slope_per_1000ah)} className={Number(item.loss_slope_per_1000ah) === Number(comparison.best_loss_slope_per_1000ah) ? 'best-row' : ''}>
                        <td>{number(numeric(item.loss_slope_per_1000ah), 5)}</td>
                        <td>{number(numeric(item.mae_soh_pct), 3)}%</td>
                        <td>{number(numeric(item.rmse_soh_pct), 3)}%</td>
                        <td>{number(numeric(item.prediction_bias_soh_pct), 3)}%</td>
                        <td>{number(numeric(item.max_abs_soh_error_pct), 3)}%</td>
                        <td>{number(numeric(item.within_bms_tolerance_pct), 1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : (
            <section className="panel setup-panel">
              <div className="section-title"><div><h2>Model setup</h2><span>{String(degradation.model_name ?? run.pipeline_key)}</span></div></div>
              <dl className="definition-grid">
                <div><dt>Nominal capacity</dt><dd>{number(run.nominal_capacity_ah, 3)} Ah</dd></div>
                <div><dt>Baseline capacity</dt><dd>{number(numeric(degradation.baseline_capacity_ah), 3)} Ah</dd></div>
                <div><dt>Reference usage</dt><dd>{compactNumber(numeric(degradation.reference_cumulative_ah))} Ah</dd></div>
                <div><dt>Loss intercept</dt><dd>{number(numeric(degradation.intercept_loss_ah), 3)} Ah</dd></div>
                <div><dt>Model fit rows</dt><dd>{integer(run.training_rows)}</dd></div>
                <div><dt>Residual spread</dt><dd>{number(numeric(degradation.residual_std), 3)} Ah</dd></div>
              </dl>
            </section>
          )}
        </div>
      )}

      {activeTab === 'validation' && (
        <div className="tab-content">
          <section className="metric-strip validation-metrics">
            <MetricCard label="SOH MAE" value={`${number(numeric(cutoff.mae_soh_pct), 3)}%`} detail="Mean absolute error" icon={CircleGauge} tone={statusTone(numeric(cutoff.mae_soh_pct))} />
            <MetricCard label="Maximum error" value={`${number(numeric(cutoff.max_abs_soh_error_pct), 3)}%`} detail="Largest absolute SOH error" icon={Activity} tone={statusTone(numeric(cutoff.max_abs_soh_error_pct))} />
            <MetricCard label="Capacity RMSE" value={`${number(numeric(cutoff.rmse_ah), 3)} Ah`} detail="Capacity-equivalent residual" icon={Gauge} />
            <MetricCard label="Validation rows" value={integer(numeric(cutoff.rows))} detail={`${shortDate(String(cutoff.start_time ?? ''))} to ${shortDate(String(cutoff.end_time ?? ''))}`} icon={Rows3} />
          </section>
          <section className="panel chart-panel">
            <div className="section-title"><div><h2>{isFixedSlope ? 'BMS agreement' : 'Cutoff validation'}</h2><span>{isFixedSlope ? 'BMS observations and lowest-MAE tested slope' : 'Actual and predicted SOH on unseen rows'}</span></div></div>
            <ValidationChart data={validation} />
          </section>
          {isFixedSlope && (
            <section className="panel quality-panel">
              <div className="section-title"><div><h2>Telemetry coverage</h2><span>Inputs used by this verification</span></div></div>
              <dl className="definition-grid">
                <div><dt>Observed span</dt><dd>{number((numeric(quality.observed_span_seconds) ?? 0) / 86_400, 1)} days</dd></div>
                <div><dt>Missing duration</dt><dd>{number((numeric(quality.missing_duration_seconds) ?? 0) / 3_600, 1)} hours</dd></div>
                <div><dt>Telemetry coverage</dt><dd>{number(numeric(quality.telemetry_coverage_pct), 1)}%</dd></div>
                <div><dt>Aggregation</dt><dd>{String(summary.aggregation ?? 'daily')}</dd></div>
              </dl>
            </section>
          )}
        </div>
      )}

      {activeTab === 'files' && (
        <section className="panel artifacts-panel tab-content">
          <div className="section-title"><div><h2>Run files</h2><span>{artifacts.length} registered artifacts</span></div></div>
          <ArtifactTable artifacts={artifacts} datasets={datasets} />
        </section>
      )}
    </div>
  )
}
