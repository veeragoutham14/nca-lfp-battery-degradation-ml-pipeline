import { Archive, ArrowRight, Database, Download, HardDrive, LoaderCircle, Search, Trash2 } from 'lucide-react'
import { FormEvent, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { BatteryChemistry, DataValidationResult, DeviceLibraryItem, ElectrothermalValidationResult, ImportPlan, ImportPurpose, ImportResolution, PipelineJob, PipelinePlugin } from '../types'
import { compactNumber, compareDevicesByModuleCount, shortDate } from '../utils'
import { DataValidationPanel } from './DataValidationPanel'

interface ExtractionPageProps {
  chemistry: BatteryChemistry
  devices: DeviceLibraryItem[]
  jobs: PipelineJob[]
  plugins: PipelinePlugin[]
  importPlugin: PipelinePlugin | undefined
  onCreateJob: (pipelineKey: string, parameters: Record<string, unknown>) => Promise<void>
  onArchiveJob: (jobId: string) => Promise<void>
  onDeleteJob: (jobId: string) => Promise<void>
  onOpenMl: (serial: string) => void
}

type SignalKey = 'soc' | 'soh' | 'current' | 'voltage' | 'temperature' | 'cell_voltage' | 'cycle_count'

const IMPORT_KEYS = new Set(['veera.raw_telemetry_import.v1', 'veera.raw_telemetry_import.v2'])
const DATA_VALIDATION_KEY = 'veera.data_validation.v1'
const ELECTROTHERMAL_VALIDATION_KEY = 'veera.electrothermal_import_validation.v1'
const CUSTOM_SIGNALS: Array<{ key: SignalKey; label: string; detail: string }> = [
  { key: 'soc', label: 'SOC', detail: 'BMS state of charge' },
  { key: 'soh', label: 'BMS SOH', detail: 'BMS state-of-health estimate' },
  { key: 'current', label: 'Stack current', detail: 'Positive charge, negative discharge' },
  { key: 'voltage', label: 'Stack voltage', detail: 'System-level BMS voltage' },
  { key: 'temperature', label: 'Temperature channels', detail: 'All pack sensors resolved from module topology' },
  { key: 'cell_voltage', label: 'Voltage-group channels', detail: 'All module, pack and voltage-group channels' },
  { key: 'cycle_count', label: 'Cycle counter', detail: 'BMS sys.cycles value' },
]

function emptySelections(): Record<SignalKey, ImportResolution> {
  return { soc: 'off', soh: 'off', current: 'off', voltage: 'off', temperature: 'off', cell_voltage: 'off', cycle_count: 'off' }
}

function isActive(job: PipelineJob): boolean {
  return job.status === 'queued' || job.status === 'running'
}

function utcInputToIso(value: string): string {
  return new Date(`${value}Z`).toISOString()
}

function importedResolutionLabel(device: DeviceLibraryItem, importing: boolean): string {
  if (importing || device.raw_status === 'importing') return 'importing'
  if (device.raw_file_count > 0 && device.five_minute_file_count > 0) return '1 second + 5 minutes'
  if (device.raw_file_count > 0) return '1 second'
  if (device.five_minute_file_count > 0) return '5 minutes'
  return 'not imported'
}

function planRuleLabel(rule: string): string {
  if (rule === 'continue_after_watermark') return 'Continue'
  if (rule === 'device_configuration_start') return 'Configured start'
  return 'First database row'
}

export function ExtractionPage({ chemistry, devices, jobs, plugins, importPlugin, onCreateJob, onArchiveJob, onDeleteJob, onOpenMl }: ExtractionPageProps) {
  const firstPending = devices.find((device) => !device.readiness.raw) ?? devices[0]
  const [serialChoice, setSerialChoice] = useState(firstPending?.serial ?? '__custom__')
  const [customSerial, setCustomSerial] = useState('')
  const [purpose, setPurpose] = useState<ImportPurpose>('capacity_core')
  const [selections, setSelections] = useState<Record<SignalKey, ImportResolution>>(emptySelections)
  const [signalSearch, setSignalSearch] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [chunkSize, setChunkSize] = useState(100_000)
  const [compression, setCompression] = useState('zstd')
  const [plan, setPlan] = useState<ImportPlan | null>(null)
  const [planLoading, setPlanLoading] = useState(false)
  const [planError, setPlanError] = useState('')
  const [planRetryVersion, setPlanRetryVersion] = useState(0)
  const [busy, setBusy] = useState(false)
  const [activityView, setActivityView] = useState<'recent' | 'archive'>('recent')
  const [activityBusy, setActivityBusy] = useState('')
  const [error, setError] = useState('')
  const [validation, setValidation] = useState<DataValidationResult | null>(null)
  const [validationLoading, setValidationLoading] = useState(false)
  const [electrothermalValidation, setElectrothermalValidation] = useState<ElectrothermalValidationResult | null>(null)
  const [electrothermalValidationLoading, setElectrothermalValidationLoading] = useState(false)

  const validationPlugin = plugins.find((plugin) => plugin.key === DATA_VALIDATION_KEY)
  const electrothermalValidationPlugin = plugins.find((plugin) => plugin.key === ELECTROTHERMAL_VALIDATION_KEY)
  const selectedSerial = serialChoice === '__custom__' ? customSerial.trim() : serialChoice
  const selectedDevice = devices.find((device) => device.serial === selectedSerial)
  const selectedModuleCount = selectedDevice?.module_count ?? null
  const hasDataValidation = selectedDevice?.readiness.data_validation ?? false
  const hasElectrothermalValidation = selectedDevice?.readiness.electrothermal_validation ?? false
  const orderedDevices = useMemo(() => [...devices].sort(compareDevicesByModuleCount), [devices])
  const deviceNames = useMemo(() => new Map(devices.map((device) => [device.serial, device.display_name])), [devices])
  const importJobs = jobs.filter((job) => IMPORT_KEYS.has(job.pipeline_key))
  const recentImportJobs = importJobs.filter((job) => !job.archived_at)
  const archivedImportJobs = importJobs.filter((job) => job.archived_at)
  const activeJobs = recentImportJobs.filter(isActive)
  const visibleJobs = activityView === 'recent'
    ? [...activeJobs, ...recentImportJobs.filter((job) => !isActive(job))].slice(0, Math.max(5, activeJobs.length))
    : archivedImportJobs
  const importingSerials = new Set(activeJobs.map((job) => job.serial))
  const selectedImportRunning = selectedSerial !== '' && importingSerials.has(selectedSerial)
  const selectedSignals = Object.values(selections).filter((value) => value !== 'off').length
  const planVersion = importJobs.map((job) => `${job.job_id}:${job.status}:${job.completed_at ?? ''}`).join('|')
  const validationVersion = jobs.filter((job) => job.pipeline_key === DATA_VALIDATION_KEY).map((job) => `${job.job_id}:${job.status}:${job.completed_at ?? ''}`).join('|')
  const electrothermalValidationVersion = jobs.filter((job) => job.pipeline_key === ELECTROTHERMAL_VALIDATION_KEY).map((job) => `${job.job_id}:${job.status}:${job.completed_at ?? ''}`).join('|')
  const validationRunning = jobs.some((job) => job.serial === selectedSerial && job.pipeline_key === DATA_VALIDATION_KEY && isActive(job))
  const electrothermalValidationRunning = jobs.some((job) => job.serial === selectedSerial && job.pipeline_key === ELECTROTHERMAL_VALIDATION_KEY && isActive(job))
  const updating = plan?.signals.some((signal) => signal.local_range.end != null) ?? false
  const topologyMissing = plan?.signals.some((signal) => signal.requires_module_topology && !signal.topology_resolved) ?? false
  const visibleSignals = CUSTOM_SIGNALS.filter((signal) => {
    const query = signalSearch.trim().toLowerCase()
    return !query || `${signal.label} ${signal.detail}`.toLowerCase().includes(query)
  })

  useEffect(() => {
    setPurpose('capacity_core')
    setSelections(emptySelections())
    setStart('')
    setEnd('')
  }, [chemistry])

  useEffect(() => {
    if (devices.length === 0) setSerialChoice('__custom__')
    else if (serialChoice !== '__custom__' && !devices.some((device) => device.serial === serialChoice)) {
      setSerialChoice(devices.find((device) => !device.readiness.raw)?.serial ?? devices[0].serial)
    }
  }, [devices, serialChoice])

  useEffect(() => {
    if (!selectedSerial || (purpose === 'custom' && selectedSignals === 0)) {
      setPlan(null)
      setPlanError('')
      return
    }
    setPlan(null)
    setPlanError('')
    setPlanLoading(false)
    let active = true
    const timer = window.setTimeout(() => {
      setPlanLoading(true)
      api.importPlan(selectedSerial, purpose, chemistry, selectedModuleCount, selections)
        .then((result) => { if (active) setPlan(result) })
        .catch((reason) => {
          if (!active) return
          setPlan(null)
          setPlanError(reason instanceof Error ? reason.message : String(reason))
        })
        .finally(() => { if (active) setPlanLoading(false) })
    }, 200)
    return () => { active = false; window.clearTimeout(timer) }
  }, [chemistry, planRetryVersion, planVersion, purpose, selectedModuleCount, selectedSerial, selectedSignals, selections])

  useEffect(() => {
    if (!selectedSerial || !hasDataValidation) {
      setValidation(null)
      setValidationLoading(false)
      return
    }
    let active = true
    setValidationLoading(true)
    api.dataValidation(selectedSerial, chemistry)
      .then((result) => { if (active) setValidation(result) })
      .catch(() => { if (active) setValidation(null) })
      .finally(() => { if (active) setValidationLoading(false) })
    return () => { active = false }
  }, [chemistry, hasDataValidation, selectedSerial, validationVersion])

  useEffect(() => {
    if (!selectedSerial || !hasElectrothermalValidation) {
      setElectrothermalValidation(null)
      setElectrothermalValidationLoading(false)
      return
    }
    let active = true
    setElectrothermalValidationLoading(true)
    api.electrothermalValidation(selectedSerial, chemistry)
      .then((result) => { if (active) setElectrothermalValidation(result) })
      .catch(() => { if (active) setElectrothermalValidation(null) })
      .finally(() => { if (active) setElectrothermalValidationLoading(false) })
    return () => { active = false }
  }, [chemistry, electrothermalValidationVersion, hasElectrothermalValidation, selectedSerial])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!importPlugin || !selectedSerial || selectedImportRunning) return
    setBusy(true)
    setError('')
    try {
      const parameters: Record<string, unknown> = { serial: selectedSerial, chemistry, purpose, chunk_size: chunkSize, compression }
      if (selectedModuleCount != null) parameters.module_count = selectedModuleCount
      if (start) parameters.start = utcInputToIso(start)
      if (end) parameters.end = utcInputToIso(end)
      if (purpose === 'custom') {
        for (const signal of CUSTOM_SIGNALS) parameters[`${signal.key}_resolution`] = selections[signal.key]
      }
      await onCreateJob(importPlugin.key, parameters)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally { setBusy(false) }
  }

  async function validateExistingData() {
    if (!validationPlugin || !selectedDevice?.module_count) return
    setError('')
    try { await onCreateJob(validationPlugin.key, { serial: selectedDevice.serial, module_count: selectedDevice.module_count }) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  async function validateElectrothermalData() {
    if (!electrothermalValidationPlugin || !selectedDevice?.module_count) return
    setError('')
    try { await onCreateJob(electrothermalValidationPlugin.key, { serial: selectedDevice.serial, chemistry, module_count: selectedDevice.module_count }) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  async function archiveJob(jobId: string) {
    setActivityBusy(jobId)
    try { await onArchiveJob(jobId) } finally { setActivityBusy('') }
  }

  async function archiveCompleted() {
    setActivityBusy('__all__')
    try { for (const job of recentImportJobs.filter((item) => !isActive(item))) await onArchiveJob(job.job_id) }
    finally { setActivityBusy('') }
  }

  async function deleteJob(jobId: string) {
    if (!window.confirm('Remove this import record? Imported Parquet data will remain unchanged.')) return
    setActivityBusy(jobId)
    try { await onDeleteJob(jobId) } finally { setActivityBusy('') }
  }

  return (
    <div className="page-stack extraction-page">
      <header className="page-header">
        <div><span className="eyebrow">{chemistry} data extraction</span><h1>Telemetry imports</h1></div>
        <span className="page-count">{devices.filter((device) => device.readiness.raw).length} imported</span>
      </header>

      <div className="extraction-grid">
        <section className="panel extraction-tool">
          <div className="section-title"><div><h2>Database import</h2><span>Purpose-based, independently resumable Parquet datasets</span></div></div>
          <form className="tool-form" onSubmit={submit}>
            <label className="field"><span>Battery</span><select value={serialChoice} onChange={(event) => setSerialChoice(event.target.value)}>
              {orderedDevices.map((device) => <option key={device.serial} value={device.serial}>{device.display_name} | {device.serial}</option>)}
              <option value="__custom__">Custom serial</option>
            </select></label>
            {serialChoice === '__custom__' && <label className="field"><span>Custom serial number</span><input value={customSerial} onChange={(event) => setCustomSerial(event.target.value)} placeholder="Enter serial number" required /></label>}

            <div className="field-grid import-range">
              <label className="field"><span>Start override (UTC)</span><input type="datetime-local" step="1" value={start} onChange={(event) => setStart(event.target.value)} /><small>Blank continues from the stored watermark.</small></label>
              <label className="field"><span>End, exclusive (UTC)</span><input type="datetime-local" step="1" value={end} onChange={(event) => setEnd(event.target.value)} /><small>Blank imports through the latest database row.</small></label>
            </div>

            <div className="purpose-switch" role="tablist" aria-label="Import purpose">
              <button type="button" className={purpose === 'capacity_core' ? 'active' : ''} onClick={() => setPurpose('capacity_core')}>Capacity degradation</button>
              <button type="button" className={purpose === 'electrothermal_history' ? 'active' : ''} onClick={() => setPurpose('electrothermal_history')}>Electrothermal history</button>
              <button type="button" className={purpose === 'custom' ? 'active' : ''} onClick={() => setPurpose('custom')}>Custom telemetry</button>
            </div>

            {purpose === 'capacity_core' ? <div className="import-contract"><div><strong>Fixed one-second contract</strong><span>6 columns</span></div><ul><li>UTC timestamp</li><li>Device serial</li><li>SOC</li><li>BMS SOH</li><li>Stack current</li><li>Stack voltage</li></ul><small>Timestamp and serial are mandatory dataset keys; the other four columns are measured BMS signals.</small></div> : purpose === 'electrothermal_history' ? <div className="import-contract"><div><strong>Fixed five-minute contract</strong><span>2 signal groups</span></div><ul><li>All pack temperature sensors</li><li>All module/pack voltage-group channels</li></ul><small>Channel counts are resolved from the installed module count and stored in chemistry-partitioned datasets.</small></div> : (
              <fieldset className="signal-resolution-fieldset">
                <legend>Signal groups</legend>
                <label className="signal-search"><Search size={14} /><input value={signalSearch} onChange={(event) => setSignalSearch(event.target.value)} placeholder="Find a signal" /></label>
                <div className="signal-resolution-header"><span>Signal</span><span>Resolution</span></div>
                {visibleSignals.map((signal) => <label className="signal-resolution-row" key={signal.key}>
                  <span><strong>{signal.label}</strong><small>{signal.detail}</small></span>
                  <select value={selections[signal.key]} onChange={(event) => setSelections((current) => ({ ...current, [signal.key]: event.target.value as ImportResolution }))}>
                    <option value="off">Not selected</option><option value="1s">1 second</option><option value="5m">5 minutes</option><option value="both">Both</option>
                  </select>
                </label>)}
              </fieldset>
            )}

            <div className="import-plan">
              <div className="import-plan-heading"><strong>Import plan</strong>{planLoading && <LoaderCircle size={14} className="spin" />}</div>
              {purpose === 'custom' && selectedSignals === 0 ? <p className="plan-empty">Select at least one signal group.</p> : planError ? <div className="form-error"><span>{planError}</span><button type="button" className="button secondary" onClick={() => setPlanRetryVersion((value) => value + 1)}>Retry plan</button></div> : plan && <div className="import-plan-table">{plan.signals.map((signal) => <div className="import-plan-row" key={`${signal.group}-${signal.resolution}`}>
                <span><strong>{signal.label}</strong><small>{signal.resolution}{signal.column_count != null ? ` | ${signal.column_count} channels` : ''}</small></span>
                <span><strong>{planRuleLabel(signal.planned_rule)}</strong><small>{signal.planned_start ? new Date(signal.planned_start).toISOString().replace('.000Z', 'Z') : 'database range'}</small></span>
                <span><strong>{signal.estimated_rows_per_month == null ? 'New' : compactNumber(signal.estimated_rows_per_month)}</strong><small>{signal.estimated_rows_per_month == null ? 'dataset' : 'rows / month'}</small></span>
              </div>)}</div>}
              <small className="plan-note">Blank dates apply each dataset's own stored watermark. Five-minute datasets continue at the next complete bucket.</small>
            </div>

            <details className="advanced-fields"><summary>Storage options</summary><div className="field-grid">
              <label className="field"><span>Rows per file</span><input type="number" min="1" step="1" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} /></label>
              <label className="field"><span>Compression</span><select value={compression} onChange={(event) => setCompression(event.target.value)}><option value="zstd">Zstandard</option><option value="snappy">Snappy</option><option value="gzip">Gzip</option></select></label>
            </div></details>
            {topologyMissing && <div className="form-error">A verified module count is required for temperature or voltage-group channels.</div>}
            {error && <div className="form-error">{error}</div>}
            <button className="button primary tool-submit" type="submit" disabled={busy || selectedImportRunning || !importPlugin || !selectedSerial || !plan || topologyMissing}>
              {busy || selectedImportRunning ? <LoaderCircle size={16} className="spin" /> : <Download size={16} />}{busy ? 'Starting...' : selectedImportRunning ? 'Import running' : updating ? 'Update import' : 'Start import'}
            </button>
          </form>
        </section>

        <section className="panel import-activity">
          <div className="section-title activity-title"><div><h2>Import activity</h2><span>{recentImportJobs.length} recent, {archivedImportJobs.length} archived</span></div><div className="activity-view-switch"><button className={activityView === 'recent' ? 'active' : ''} type="button" onClick={() => setActivityView('recent')} title="Recent imports"><Download size={14} /></button><button className={activityView === 'archive' ? 'active' : ''} type="button" onClick={() => setActivityView('archive')} title="Archived imports"><Archive size={14} /></button></div></div>
          {visibleJobs.length === 0 ? <div className="empty-state compact-empty">No {activityView} imports.</div> : <div className="job-list">{visibleJobs.map((job) => <div className="job-row" key={job.job_id} title={job.error_message ?? undefined}>
            <span className={`job-status status-${job.status}`}><span />{job.status}</span><strong>{deviceNames.get(job.serial) ?? `Battery ${job.serial}`}</strong><small>{job.pipeline_key.endsWith('.v2') ? 'Purpose-based import' : 'Legacy import'} | {job.serial} | {job.return_code == null ? 'pending' : `exit ${job.return_code}`}</small>
            {activityView === 'recent' ? <button className="job-action" type="button" disabled={activityBusy !== '' || isActive(job)} onClick={() => void archiveJob(job.job_id)} title="Archive"><Archive size={14} /></button> : <button className="job-action danger" type="button" disabled={activityBusy !== ''} onClick={() => void deleteJob(job.job_id)} title="Remove record"><Trash2 size={14} /></button>}
          </div>)}</div>}
          {activityView === 'recent' && recentImportJobs.some((job) => !isActive(job)) && <div className="activity-footer"><button className="activity-archive-all" type="button" disabled={activityBusy !== ''} onClick={() => void archiveCompleted()}><Archive size={14} />Archive completed</button></div>}
        </section>
      </div>

      <section className="panel validation-panel-shell">
        <div className="section-title"><div><h2>Validate imported data</h2><span>Electrothermal import check and final analysis-input gate</span></div></div>
        {!selectedDevice ? <div className="empty-state compact-empty">Select a configured battery to validate its local telemetry.</div> : <DataValidationPanel data={validation} loading={validationLoading} running={validationRunning} canValidate={Boolean(validationPlugin && selectedDevice.module_count && selectedDevice.readiness.raw && selectedDevice.workflow.events?.status === 'current' && selectedDevice.workflow.hierarchical_features?.status === 'current')} validationBlockedReason="Complete Classify telemetry rows & Build operating events, validate electrothermal history, and Build electrothermal features first." onValidate={() => void validateExistingData()} electrothermalRunning={electrothermalValidationRunning} canValidateElectrothermal={Boolean(electrothermalValidationPlugin && selectedDevice.module_count)} onValidateElectrothermal={() => void validateElectrothermalData()} electrothermalData={electrothermalValidation} electrothermalLoading={electrothermalValidationLoading} />}
      </section>

      <section className="panel library-panel">
        <div className="section-title"><div><h2>Telemetry library</h2><span>Configured devices and local Parquet coverage</span></div></div>
        <div className="table-wrap"><table className="device-table"><thead><tr><th>Device</th><th>Role</th><th>Status</th><th>Modules</th><th>Coverage</th><th>Files</th><th>Rows</th><th /></tr></thead><tbody>{orderedDevices.map((device) => {
          const importing = importingSerials.has(device.serial)
          return <tr key={device.serial}><td><span className="serial-cell"><Database size={15} /><span className="device-identity"><strong>{device.display_name}</strong><small>{device.serial}</small></span></span></td><td><span className={`role-label role-${device.device_role}`}>{device.device_role.replaceAll('_', ' ')}</span></td><td><span className={`library-status library-${importing ? 'importing' : device.raw_status}`}>{importedResolutionLabel(device, importing)}</span></td><td><span className={`module-status module-${device.module_count_status}`}>{device.module_count_status === 'verified' ? `${device.module_count} modules | ${device.nominal_capacity_ah} Ah` : device.module_count_status.replaceAll('_', ' ')}</span></td><td>{device.readiness.raw ? `${shortDate(device.observed_start_at)} to ${shortDate(device.observed_end_at)}` : device.five_minute_file_count ? `${shortDate(device.five_minute_start_at)} to ${shortDate(device.five_minute_end_at)}` : shortDate(device.configured_start)}</td><td className="numeric"><span className="resolution-count">1s {device.raw_file_count}<small>5m {device.five_minute_file_count}</small></span></td><td className="numeric"><span className="resolution-count">1s {compactNumber(device.raw_row_count)}<small>5m {compactNumber(device.five_minute_row_count)}</small></span></td><td className="table-action"><button className="icon-button" type="button" onClick={() => onOpenMl(device.serial)} disabled={!device.readiness.raw} title="Open in machine learning"><ArrowRight size={16} /></button></td></tr>
        })}</tbody></table></div>
        {devices.length === 0 && <div className="empty-state"><HardDrive size={22} />No configured {chemistry} devices.</div>}
      </section>
    </div>
  )
}
