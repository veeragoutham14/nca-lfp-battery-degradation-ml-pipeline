import { Activity, CalendarRange, Database, Gauge, LoaderCircle, Play, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type {
  AnalysisRun,
  BatteryChemistry,
  DeviceLibraryItem,
  PipelineJob,
  PipelinePlugin,
  RunSeries,
} from '../types'
import { compactNumber, integer, number, shortDate, shortDateTime } from '../utils'
import { MetricCard } from './MetricCard'
import { PipelineDialog } from './PipelineDialog'
import { SeasonalChargeProfileChart } from './SeasonalChargeProfileChart'

const PIPELINE_KEY = 'veera.seasonal_charge_analysis.v1'
const MODEL_FAMILY = 'seasonal_charging_stress_profile'

interface SeasonalProfilesPageProps {
  chemistry: BatteryChemistry
  devices: DeviceLibraryItem[]
  plugins: PipelinePlugin[]
  jobs: PipelineJob[]
  onCreateJob: (pipelineKey: string, parameters: Record<string, unknown>) => Promise<void>
}

function numeric(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function SeasonalProfilesPage({
  chemistry,
  devices,
  plugins,
  jobs,
  onCreateJob,
}: SeasonalProfilesPageProps) {
  const [runs, setRuns] = useState<AnalysisRun[]>([])
  const [selectedSerial, setSelectedSerial] = useState('')
  const [summary, setSummary] = useState<Record<string, unknown>>({})
  const [series, setSeries] = useState<RunSeries | null>(null)
  const [dialogSerial, setDialogSerial] = useState('')
  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [error, setError] = useState('')

  const plugin = plugins.find((item) => item.key === PIPELINE_KEY)
  const latestBySerial = useMemo(() => {
    const result = new Map<string, AnalysisRun>()
    runs.forEach((run) => {
      if (!result.has(run.serial)) result.set(run.serial, run)
    })
    return result
  }, [runs])
  const rows = useMemo(() => devices.map((device) => ({
    device,
    run: latestBySerial.get(device.serial) ?? null,
    latestJob: jobs
      .filter((job) => job.serial === device.serial && job.pipeline_key === PIPELINE_KEY)
      .sort((left, right) => right.created_at.localeCompare(left.created_at))[0] ?? null,
    ready: device.readiness.hierarchical_features
      && device.readiness.soh_signal_audit
      && device.readiness.soh_recalibration,
    running: jobs.some((job) => job.serial === device.serial
      && job.pipeline_key === PIPELINE_KEY
      && (job.status === 'queued' || job.status === 'running')),
  })), [devices, jobs, latestBySerial])
  const selectedDevice = devices.find((device) => device.serial === selectedSerial) ?? null
  const selectedRun = latestBySerial.get(selectedSerial) ?? null
  const selectedJob = rows.find((row) => row.device.serial === selectedSerial)?.latestJob ?? null
  const selectedJobActive = selectedJob?.status === 'queued' || selectedJob?.status === 'running'
  const completionKey = jobs
    .filter((job) => job.pipeline_key === PIPELINE_KEY)
    .map((job) => `${job.job_id}:${job.status}`)
    .join('|')

  async function loadRuns() {
    setLoading(true)
    setError('')
    try {
      const result = await api.latestRuns(MODEL_FAMILY)
      const allowed = new Set(devices.map((device) => device.serial))
      setRuns(result.filter((run) => allowed.has(run.serial)))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void loadRuns() }, [chemistry, completionKey])

  useEffect(() => {
    if (selectedSerial && devices.some((device) => device.serial === selectedSerial)) return
    const firstAnalyzed = rows.find((row) => row.run)?.device.serial
    setSelectedSerial(firstAnalyzed ?? devices[0]?.serial ?? '')
  }, [devices, rows, selectedSerial])

  useEffect(() => {
    if (!selectedRun) {
      setSummary({})
      setSeries(null)
      return
    }
    let cancelled = false
    setDetailLoading(true)
    Promise.all([api.summary(selectedRun.run_id), api.series(selectedRun.run_id)])
      .then(([nextSummary, nextSeries]) => {
        if (cancelled) return
        setSummary(nextSummary)
        setSeries(nextSeries)
      })
      .catch((reason) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason))
      })
      .finally(() => { if (!cancelled) setDetailLoading(false) })
    return () => { cancelled = true }
  }, [selectedRun?.run_id])

  const analyzedCount = rows.filter((row) => row.run).length
  const readyCount = rows.filter((row) => row.ready).length
  return (
    <div className="page-stack seasonal-profiles-page">
      <header className="page-header">
        <div>
          <span className="eyebrow">{chemistry} fleet analysis</span>
          <h1>Seasonal degradation profiles</h1>
          <p>Monthly current, SOC, thermal, cell-voltage, throughput, and BMS SOH history for each configured battery.</p>
        </div>
        <button className="button secondary" type="button" onClick={() => void loadRuns()} disabled={loading}>
          <RefreshCw size={16} className={loading ? 'spin' : ''} aria-hidden="true" /> Refresh results
        </button>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="metric-strip">
        <MetricCard label="Analyzed devices" value={`${analyzedCount} / ${rows.length}`} detail="Latest cataloged seasonal run" icon={CalendarRange} tone={analyzedCount > 0 ? 'good' : 'warning'} />
        <MetricCard label="Ready to analyze" value={integer(readyCount)} detail="SOH audit, five-minute history, and recalibration evidence available" icon={Play} />
        <MetricCard label="Qualified months" value={integer(numeric(summary.valid_monthly_profile_rows))} detail={`${integer(numeric(summary.monthly_profile_rows))} total monthly windows`} icon={Activity} />
        <MetricCard label="Charge throughput" value={`${compactNumber(numeric(summary.total_charge_ah))} Ah`} detail={`${compactNumber(numeric(summary.total_discharge_ah))} Ah discharged`} icon={Database} />
      </section>

      <section className="panel seasonal-device-picker">
        <div className="section-title">
          <div><h2>Latest device profile</h2><span>Selecting a device opens its newest successful analysis automatically.</span></div>
          <label className="field compact-field">
            <span>Battery</span>
            <select value={selectedSerial} onChange={(event) => setSelectedSerial(event.target.value)}>
              {devices.map((device) => <option key={device.serial} value={device.serial}>{device.display_name} ({device.serial})</option>)}
            </select>
          </label>
        </div>
        {selectedRun ? (
          <div className="selected-run-meta">
            <span><strong>{selectedDevice?.display_name}</strong> · {selectedDevice?.module_count ?? 'n/a'} modules</span>
            <span>Observed {shortDate(selectedRun.observed_start_at)} to {shortDate(selectedRun.observed_end_at)} · analyzed {shortDateTime(selectedRun.run_created_at)}</span>
            <button className="button secondary compact-button" type="button" disabled={selectedJobActive || !plugin || !rows.find((row) => row.device.serial === selectedSerial)?.ready} onClick={() => setDialogSerial(selectedSerial)}>{selectedJobActive ? <LoaderCircle size={15} className="spin" aria-hidden="true" /> : <Play size={15} aria-hidden="true" />} {selectedJobActive ? selectedJob?.status : 'Run updated analysis'}</button>
          </div>
        ) : (
          <div className="empty-state seasonal-empty-state">
            <strong>No seasonal profile exists for this battery.</strong>
            <span>{selectedDevice?.readiness.hierarchical_features && selectedDevice.readiness.soh_signal_audit && selectedDevice.readiness.soh_recalibration ? 'The required source evidence is ready.' : 'Complete the SOH audit, five-minute history, and BMS recalibration analysis first.'}</span>
            <button className="button primary" type="button" disabled={!plugin || !selectedDevice || !(selectedDevice.readiness.hierarchical_features && selectedDevice.readiness.soh_signal_audit && selectedDevice.readiness.soh_recalibration)} onClick={() => setDialogSerial(selectedSerial)}><Play size={16} aria-hidden="true" /> Run analysis</button>
          </div>
        )}
        {selectedJob && (
          <div className={`inline-notice seasonal-job-status status-${selectedJob.status}`} role="status" aria-live="polite">
            <strong>Analysis job: {selectedJob.status}</strong>
            <span>
              {selectedJob.status === 'queued' && 'Waiting for an ML worker.'}
              {selectedJob.status === 'running' && `Started ${shortDateTime(selectedJob.started_at)}. The result page will refresh when processing finishes.`}
              {selectedJob.status === 'completed' && `Completed ${shortDateTime(selectedJob.completed_at)}${selectedJob.output_run_id ? `; result ${selectedJob.output_run_id}` : ''}.`}
              {selectedJob.status === 'failed' && (selectedJob.error_message || `Failed with exit code ${selectedJob.return_code ?? 'unknown'}.`)}
              {selectedJob.status === 'abandoned' && 'The worker stopped reporting before completion. Review the backend job log before rerunning.'}
            </span>
          </div>
        )}
      </section>

      {selectedRun && (detailLoading ? (
        <div className="section-loader">Loading latest seasonal evidence...</div>
      ) : (
        <>
          <section className="panel chart-panel seasonal-profile-panel">
            <div className="section-title"><div><h2>Monthly degradation history</h2><span>Coverage-qualified operating exposure with one-second current dynamics and BMS SOH overlaid in UTC.</span></div></div>
            <SeasonalChargeProfileChart points={series?.seasonal_profile ?? []} socExposureMethod={String(summary.soc_exposure_method ?? '')} />
          </section>
        </>
      ))}

      <section className="panel seasonal-inventory">
        <div className="section-title"><div><h2>{chemistry} source inventory</h2><span>One row per configured device; results always point to the latest run.</span></div></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Battery</th><th>Modules</th><th>Prerequisites</th><th>Latest profile</th><th>Observed through</th><th>Action</th></tr></thead>
            <tbody>{rows.map(({ device, run, ready, running, latestJob }) => (
              <tr key={device.serial} className={selectedSerial === device.serial ? 'selected-row' : ''}>
                <td><strong>{device.display_name}</strong><small>{device.serial}</small></td>
                <td>{device.module_count ?? 'n/a'}</td>
                <td><span className={`step-state ${ready ? 'step-complete' : 'step-locked'}`}>{ready ? 'Ready' : 'Waiting'}</span></td>
                <td>{running ? <span className={`job-status status-${latestJob?.status}`}><span />{latestJob?.status}</span> : run ? shortDateTime(run.run_created_at) : latestJob?.status === 'failed' ? <span className="job-status status-failed"><span />failed</span> : 'Not analyzed'}</td>
                <td>{run ? shortDate(run.observed_end_at) : 'n/a'}</td>
                <td><button className="button secondary compact-button seasonal-row-action" type="button" disabled={running || (!run && (!ready || !plugin))} onClick={() => run ? setSelectedSerial(device.serial) : setDialogSerial(device.serial)}>{run ? <><Gauge size={15} aria-hidden="true" /> View latest</> : <><Play size={15} aria-hidden="true" /> {running ? 'Running' : 'Analyze'}</>}</button></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>

      <PipelineDialog
        open={Boolean(dialogSerial)}
        serial={dialogSerial}
        plugins={plugin ? [plugin] : []}
        eyebrow={`${chemistry} degradation analysis`}
        title="Analyze monthly degradation history"
        submitLabel="Start analysis"
        parameterOverrides={{
          module_count: devices.find((device) => device.serial === dialogSerial)?.module_count,
          nominal_capacity_ah: devices.find((device) => device.serial === dialogSerial)?.nominal_capacity_ah,
        }}
        onClose={() => setDialogSerial('')}
        onSubmit={onCreateJob}
      />
    </div>
  )
}
