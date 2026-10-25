import {
  BatteryCharging,
  Check,
  CircleAlert,
  DatabaseZap,
  LoaderCircle,
  RefreshCw,
  ScanSearch,
  Thermometer,
  Waves,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type {
  CellSignalOverview,
  CellSignalOverviewDevice,
  BatteryChemistry,
  PipelineJob,
} from '../types'
import { integer, number, shortDateTime } from '../utils'
import { MetricCard } from './MetricCard'

interface CellSignalReadinessPageProps {
  chemistry: BatteryChemistry
  jobs: PipelineJob[]
  onOpenAnomaly: (serial: string) => void
}

const AUDIT_KEY = 'veera.bms_electrothermal_source_audit.v1'
const FEATURE_KEY = 'veera.bms_hierarchical_features.v1'
const HIERARCHICAL_FEATURE_CONTRACT = 'hierarchical_electrothermal_v1'

function isActive(job: PipelineJob) {
  return job.status === 'queued' || job.status === 'running'
}

function rowStatus(device: CellSignalOverviewDevice) {
  if (!device.audited) return { label: 'Not audited', tone: 'waiting' }
  if (!device.cell_voltage_ready) return { label: 'Quality gate blocked', tone: 'warning' }
  const limitedTemperature = device.quality_warnings.includes(
    'inactive_temperature_channels_excluded_from_temperature_context',
  )
  if (!device.telemetry_fresh) {
    return {
      label: limitedTemperature ? 'Ready, stale + limited temperature' : 'Ready, stale data',
      tone: 'warning',
    }
  }
  if (limitedTemperature) {
    return { label: 'Ready, limited temperature coverage', tone: 'warning' }
  }
  return { label: 'Ready', tone: 'complete' }
}

export function CellSignalReadinessPage({
  chemistry,
  jobs,
  onOpenAnomaly,
}: CellSignalReadinessPageProps) {
  const [overview, setOverview] = useState<CellSignalOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const activeSerials = useMemo(
    () => new Set(
      jobs
        .filter((job) => isActive(job) && (job.pipeline_key === AUDIT_KEY || job.pipeline_key === FEATURE_KEY))
        .map((job) => job.serial),
    ),
    [jobs],
  )
  const completedVersion = useMemo(
    () => jobs
      .filter((job) => (
        (job.pipeline_key === AUDIT_KEY || job.pipeline_key === FEATURE_KEY)
        && (job.status === 'completed' || job.status === 'failed')
      ))
      .map((job) => `${job.job_id}:${job.completed_at ?? ''}`)
      .join('|'),
    [jobs],
  )
  const latestFeatureJobs = useMemo(() => {
    const result = new Map<string, PipelineJob>()
    jobs
      .filter((job) => job.pipeline_key === FEATURE_KEY)
      .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
      .forEach((job) => {
        if (!result.has(job.serial)) result.set(job.serial, job)
      })
    return result
  }, [jobs])

  function load() {
    setLoading(true)
    setError('')
    api.cellSignalOverview(chemistry)
      .then(setOverview)
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [chemistry, completedVersion])

  if (loading && !overview) {
    return <div className="section-loader full-page-loader">Loading cell-signal readiness...</div>
  }

  const metrics = overview?.metrics

  return (
    <div className="page-stack cell-readiness-page">
      {error && <div className="error-banner">{error}</div>}
      <header className="page-header cell-readiness-header">
        <div>
          <span className="eyebrow">NCA signal qualification</span>
          <h1>Electrothermal source quality</h1>
        </div>
        <div className="cell-readiness-actions">
          <button
            className="icon-button"
            type="button"
            onClick={load}
            disabled={loading}
            title="Refresh cell-signal readiness"
          >
            <RefreshCw size={17} className={loading ? 'spin' : ''} aria-hidden="true" />
          </button>
        </div>
      </header>

      {metrics && (
        <>
          <section className="metric-strip cell-readiness-metrics">
            <MetricCard
              label="Cohort audited"
              value={`${metrics.audited_device_count} / ${metrics.configured_device_count}`}
              detail={`${metrics.field_device_count} verified field + ${metrics.reference_device_count} control; ${metrics.deferred_field_device_count} require module metadata`}
              icon={ScanSearch}
              tone={metrics.pending_device_count ? 'warning' : 'good'}
            />
            <MetricCard
              label="Fresh and cell-ready"
              value={`${metrics.fresh_ready_device_count} / ${metrics.configured_device_count}`}
              detail={`${metrics.stale_device_count} device${metrics.stale_device_count === 1 ? '' : 's'} with stale source data`}
              icon={Waves}
              tone={metrics.fresh_ready_device_count === metrics.configured_device_count ? 'good' : 'warning'}
            />
            <MetricCard
              label="Ready voltage channels"
              value={`${metrics.total_ready_active_cell_channels} / ${metrics.total_expected_active_cell_channels}`}
              detail={`${integer(metrics.total_ready_active_temperature_sensors)} / ${integer(metrics.total_expected_active_temperature_sensors)} pack-temperature sensors ready`}
              icon={BatteryCharging}
            />
            <MetricCard
              label="Analysis datasets"
              value={integer(metrics.hierarchical_history_built_device_count)}
              detail="Five-minute inputs ready for anomaly detection"
              icon={DatabaseZap}
            />
          </section>

          <div className="inline-notice evidence-notice">
            <CircleAlert size={15} aria-hidden="true" />
            <span>Devices are grouped by verified module count. Two-, three-, and four-module cohorts can all be prepared and analyzed, while conclusions remain separated by chemistry and module configuration.</span>
          </div>

          <section className="panel cell-readiness-table">
            <div className="section-title">
              <div>
                <h2>{chemistry} source inventory</h2>
                <span>Every module, pack, voltage channel, and pack-temperature sensor is qualified</span>
              </div>
              <small className="source-freshness">Source {shortDateTime(overview?.latest_source_modified_at)}</small>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                     <th>Device</th>
                    <th>Role</th>
                    <th>Modules</th>
                    <th>Status</th>
                    <th>Voltage channels</th>
                    <th>Temperature sensors</th>
                    <th>Sample</th>
                    <th>1 Hz coverage</th>
                    <th>Lowest valid coverage</th>
                    <th>Extrema check</th>
                    <th>Analysis data</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview?.devices ?? []).map((device) => {
                    const status = rowStatus(device)
                    const active = activeSerials.has(device.serial)
                    const latestFeatureJob = latestFeatureJobs.get(device.serial)
                    const featureUpToDate = device.feature_build_mode === 'up_to_date'
                    const boundaryRefresh = device.feature_build_mode === 'incremental'
                      && device.feature_lag_seconds != null
                      && device.feature_lag_seconds <= 300
                    const cohortLabel = device.cohort_role === 'reference_control'
                      ? 'reference control'
                      : device.in_active_cohort
                        ? `${device.module_count}-module cohort`
                        : 'module count required'
                    return (
                      <tr key={device.serial} className={device.in_active_cohort ? undefined : 'cohort-deferred-row'}>
                        <td>
                          <button
                            className="table-link serial-cell"
                            type="button"
                            onClick={() => onOpenAnomaly(device.serial)}
                            title={`Open anomaly results for ${device.serial}`}
                          >
                            <BatteryCharging size={14} aria-hidden="true" />
                            <span className="device-identity"><strong>{device.display_name}</strong><small>{device.serial}</small></span>
                          </button>
                        </td>
                        <td>
                          <span className={`role-label role-${device.device_role}`}>
                            {cohortLabel}
                          </span>
                          <small className="table-subtext" title={device.cohort_reason}>{device.cohort_status}</small>
                        </td>
                        <td>
                          {device.module_count ?? 'n/a'}
                          <small className="table-subtext">{device.module_count_status}</small>
                        </td>
                        <td>
                          <span className={`step-state step-${status.tone}`}>
                            {active && <LoaderCircle size={12} className="spin" aria-hidden="true" />}
                            {active ? 'Running' : status.label}
                          </span>
                        </td>
                        <td className="numeric">
                          {device.audited
                            ? `${integer(device.ready_cell_channels)} / ${integer(device.expected_cell_channels)}`
                            : `${integer(device.expected_cell_channels)} expected`}
                        </td>
                        <td className="numeric">
                          {device.audited
                            ? `${integer(device.ready_temperature_sensors)} active / ${integer(device.expected_temperature_sensors)} contract`
                            : `${integer(device.expected_temperature_sensors)} expected`}
                          {device.inactive_temperature_sensors ? (
                            <small className="table-subtext">
                              {integer(device.inactive_temperature_sensors)} inactive source channels
                            </small>
                          ) : null}
                        </td>
                        <td>{shortDateTime(device.sample_start)}</td>
                        <td className="numeric">
                          {device.nominal_1hz_coverage_pct == null
                            ? 'n/a'
                            : `${number(device.nominal_1hz_coverage_pct, 1)}%`}
                        </td>
                        <td className="numeric">
                          V {device.minimum_cell_channel_valid_coverage_pct == null ? 'n/a' : `${number(device.minimum_cell_channel_valid_coverage_pct, 1)}%`}
                          <small className="table-subtext">T active {device.minimum_active_temperature_sensor_valid_coverage_pct == null ? 'n/a' : `${number(device.minimum_active_temperature_sensor_valid_coverage_pct, 1)}%`}</small>
                        </td>
                        <td>
                          <span className="auxiliary-status" title="Global min/max cell voltage">
                            {device.global_cell_voltage_signals_ready
                              ? <Check size={14} aria-hidden="true" />
                              : <CircleAlert size={14} aria-hidden="true" />}
                            V reconciled
                          </span>
                          <span className="auxiliary-status" title="Cell temperature min/max">
                            {device.cell_temperature_signals_ready
                              ? <Thermometer size={14} aria-hidden="true" />
                              : <CircleAlert size={14} aria-hidden="true" />}
                            T reconciled
                          </span>
                        </td>
                        <td>
                          {latestFeatureJob?.status === 'failed'
                            ? (
                              <span
                                className="step-state step-warning"
                                title={latestFeatureJob.error_message ?? 'Feature build failed'}
                              >
                                <CircleAlert size={12} aria-hidden="true" />
                                Build failed
                              </span>
                            )
                            : device.feature_contract === HIERARCHICAL_FEATURE_CONTRACT
                            ? (
                              <>
                                {featureUpToDate
                                  ? 'Ready'
                                  : boundaryRefresh
                                    ? 'Final window incomplete'
                                    : 'New telemetry available'}
                                <small className="table-subtext">
                                  {featureUpToDate
                                    ? `Up to date through ${shortDateTime(device.feature_end)}`
                                    : boundaryRefresh
                                      ? `Rebuild final five-minute window through ${shortDateTime(device.recommended_feature_end)}`
                                      : `Append ${shortDateTime(device.recommended_feature_start)} to ${shortDateTime(device.recommended_feature_end)}`}
                                </small>
                              </>
                            )
                            : device.cell_health_built
                              ? 'Old format; rebuild required'
                              : 'Not prepared'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

    </div>
  )
}
