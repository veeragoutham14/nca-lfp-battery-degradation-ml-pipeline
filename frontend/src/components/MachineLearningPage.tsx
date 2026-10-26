import {
  ArrowRight,
  BatteryCharging,
  CalendarRange,
  ChartNoAxesCombined,
  Check,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  Clock3,
  Database,
  FileChartColumn,
  Gauge,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  Pencil,
  Play,
  RotateCcw,
  Rows3,
  Waves,
  Workflow,
  X,
} from 'lucide-react'
import { FormEvent, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type {
  AnalysisRun,
  BatteryChemistry,
  DeviceLibraryItem,
  PipelineJob,
  PipelinePlugin,
} from '../types'
import { compareDevicesByModuleCount, familyLabel, number, shortDateTime } from '../utils'
import { PipelineDialog } from './PipelineDialog'
import { CellAnalysisPage } from './CellAnalysisPage'
import { NcaCapacityOverviewPage } from './NcaCapacityOverviewPage'

interface MachineLearningPageProps {
  chemistry: BatteryChemistry
  devices: DeviceLibraryItem[]
  plugins: PipelinePlugin[]
  jobs: PipelineJob[]
  initialSerial: string
  onSerialChange: (serial: string) => void
  onCreateJob: (pipelineKey: string, parameters: Record<string, unknown>) => Promise<void>
  onViewRun: (runId: string) => void
  onOpenCapacitySoh: (serial: string) => void
  onUpdateDeviceDisplayName: (serial: string, displayName: string | null) => Promise<void>
}

interface WorkflowStep {
  key: string
  title: string
  detail: string
  readinessKey?: keyof DeviceLibraryItem['readiness']
  additionalReadinessKeys?: Array<keyof DeviceLibraryItem['readiness']>
  icon: typeof Rows3
}

const CLASSIFY_KEY = 'veera.classify_rows.v1'
const CAPACITY_SOH_INVESTIGATION_KEY = 'battryml.capacity_soh_investigation.v1'
const SEASONAL_CHARGE_KEY = 'veera.seasonal_charge_analysis.v1'
const EVENTS_KEY = 'veera.build_events.v1'
const CAPACITY_KEY = 'veera.capacity_trend.v1'
const FULL_CYCLE_KEY = 'veera.full_cycle_soh.v1'
const FIXED_SLOPE_KEY = 'veera.fixed_slope_verification.v1'
const ELECTROTHERMAL_FEATURES_KEY = 'veera.bms_hierarchical_features.v1'
const DATA_VALIDATION_KEY = 'veera.data_validation.v1'
const SOH_GAP_DIAGNOSIS_KEY = 'battryml.soh_gap_diagnosis.v1'

const preprocessingSteps: WorkflowStep[] = [
  {
    key: CLASSIFY_KEY,
    title: 'Classify telemetry rows',
    detail: 'Charging, discharging, idle, and invalid operating states',
    readinessKey: 'classified',
    icon: Rows3,
  },
  {
    key: EVENTS_KEY,
    title: 'Build operating events',
    detail: 'Continuous events with measured all-discharge Ah throughput',
    readinessKey: 'events',
    icon: Layers3,
  },
]

const analysisSteps: WorkflowStep[] = [
  {
    key: CAPACITY_SOH_INVESTIGATION_KEY,
    title: 'Run capacity and SOH investigation',
    detail: 'Validate rare discharges first, then relate BMS SOH transitions to that evidence',
    readinessKey: 'capacity_soh_investigation',
    icon: CircleAlert,
  },
  {
    key: SEASONAL_CHARGE_KEY,
    title: 'Analyze charging and seasonal stress',
    detail: 'Charge episodes, monthly use, thermal/SOC exposure, and SOH-step context',
    readinessKey: 'seasonal_charge_profile',
    icon: CalendarRange,
  },
]

const sohGapDiagnosisStep: WorkflowStep = {
  key: SOH_GAP_DIAGNOSIS_KEY,
  title: 'Diagnose the SOH gap',
  detail: 'Reference-normalised capacity evidence and a classification for each BMS SOH step',
  readinessKey: 'soh_gap_diagnosis',
  icon: CircleAlert,
}

const fullCycleSteps: WorkflowStep[] = [
  {
    key: CAPACITY_KEY,
    title: 'Build capacity trend',
    detail: 'Qualified full discharge cycles and capacity measurements',
    readinessKey: 'capacity_trend',
    icon: Gauge,
  },
  {
    key: FULL_CYCLE_KEY,
    title: 'Train SOH forecast',
    detail: 'Capacity degradation model and long-term usage scenarios',
    readinessKey: 'analysis',
    additionalReadinessKeys: ['soh_signal_audit'],
    icon: FileChartColumn,
  },
]

const fieldSteps: WorkflowStep[] = [
  {
    key: FIXED_SLOPE_KEY,
    title: 'Verify transferred loss slopes',
    detail: 'Compare fixed NCA slopes with BMS SOH using recorded cumulative Ah',
    readinessKey: 'analysis',
    icon: FileChartColumn,
  },
  {
    key: ELECTROTHERMAL_FEATURES_KEY,
    title: 'Build electrothermal features',
    detail: 'Join validated five-minute temperature and voltage history with operating context',
    readinessKey: 'hierarchical_features',
    icon: Waves,
  },
]

const electrothermalFeatureStep: WorkflowStep = {
  key: ELECTROTHERMAL_FEATURES_KEY,
  title: 'Build electrothermal features',
  detail: 'Join validated five-minute temperature and voltage history with operating context',
  readinessKey: 'hierarchical_features',
  icon: Waves,
}

const analysisValidationStep: WorkflowStep = {
  key: DATA_VALIDATION_KEY,
  title: 'Validate analysis inputs',
  detail: 'Qualify BMS SOH and verify telemetry, events, and electrothermal features',
  readinessKey: 'data_validation',
  icon: Check,
}

function latestJob(jobs: PipelineJob[], serial: string, pipelineKey: string) {
  return jobs.find((job) => job.serial === serial && job.pipeline_key === pipelineKey)
}

function stepState(
  step: WorkflowStep,
  device: DeviceLibraryItem,
  jobs: PipelineJob[],
  plugin: PipelinePlugin | undefined,
) {
  const job = latestJob(jobs, device.serial, step.key)
  if (job?.status === 'queued' || job?.status === 'running') return 'running'
  const authoritative = plugin ? device.workflow?.[plugin.workflow_stage] : undefined
  if (authoritative?.status === 'current') return 'complete'
  if (authoritative?.status === 'stale') return 'stale'
  if (authoritative?.status === 'failed') return 'failed'
  if (job?.status === 'failed' || job?.status === 'abandoned') return 'failed'
  if (!plugin) return 'unavailable'
  if (authoritative?.status === 'ready') return 'ready'
  if (authoritative?.status === 'blocked' || authoritative?.status === 'missing') return 'locked'
  if (step.readinessKey && device.readiness[step.readinessKey]) return 'complete'
  const requiredReadiness = [
    ...((plugin.requires_stages?.length ? plugin.requires_stages : [plugin.requires_stage])
      .filter(Boolean) as Array<keyof DeviceLibraryItem['readiness']>),
    ...(step.additionalReadinessKeys ?? []),
  ]
  if (requiredReadiness.every((key) => device.readiness[key])) {
    return 'ready'
  }
  return 'locked'
}

function stateLabel(state: string) {
  if (state === 'complete') return 'Complete'
  if (state === 'running') return 'Running'
  if (state === 'failed') return 'Needs attention'
  if (state === 'stale') return 'Update required'
  if (state === 'ready') return 'Ready'
  return 'Waiting'
}

export function MachineLearningPage({
  chemistry,
  devices,
  plugins,
  jobs,
  initialSerial,
  onSerialChange,
  onCreateJob,
  onViewRun,
  onOpenCapacitySoh,
  onUpdateDeviceDisplayName,
}: MachineLearningPageProps) {
  const importedDevices = useMemo(
    () => devices
      .filter((device) => device.readiness.raw)
      .sort(compareDevicesByModuleCount),
    [devices],
  )
  const [selectedSerial, setSelectedSerial] = useState(initialSerial)
  const [editingName, setEditingName] = useState(false)
  const [nameDraft, setNameDraft] = useState('')
  const [nameBusy, setNameBusy] = useState(false)
  const [nameError, setNameError] = useState('')
  const [workspaceView, setWorkspaceView] = useState<'fleet' | 'cells' | 'device'>(
    initialSerial ? 'device' : chemistry === 'NCA' ? 'fleet' : 'cells',
  )
  const selectedDevice = importedDevices.find(
    (device) => device.serial === selectedSerial,
  ) ?? null
  const transferredSlope = selectedDevice?.default_loss_slopes_per_1000ah[0] ?? null
  const normalizedSohLossPer1000Ah = (
    transferredSlope != null && selectedDevice?.nominal_capacity_ah != null
      ? 100 * transferredSlope / selectedDevice.nominal_capacity_ah
      : null
  )
  const [runs, setRuns] = useState<AnalysisRun[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [reportsCollapsed, setReportsCollapsed] = useState(false)
  const [dialogPlugin, setDialogPlugin] = useState<PipelinePlugin | null>(null)
  const [error, setError] = useState('')
  const dialogParameterOverrides = useMemo(
    () => {
      if (!selectedDevice) return {}
      return Object.fromEntries(
        Object.entries({
          nominal_capacity_ah: selectedDevice.nominal_capacity_ah,
          module_count: selectedDevice.module_count,
          chemistry,
          slope_reference_module_count: selectedDevice.loss_slope_reference_module_count,
        }).filter(([, value]) => value != null),
      )
    },
    [chemistry, selectedDevice],
  )
  const completedJobVersion = useMemo(
    () => jobs
      .filter((job) => job.serial === selectedSerial && job.status === 'completed')
      .map((job) => `${job.job_id}:${job.completed_at ?? ''}`)
      .join('|'),
    [jobs, selectedSerial],
  )

  useEffect(() => {
    const nextSerial = importedDevices.some((device) => device.serial === initialSerial)
      ? initialSerial
      : (importedDevices[0]?.serial ?? '')
    setSelectedSerial(nextSerial)
    if (nextSerial) onSerialChange(nextSerial)
  }, [importedDevices, initialSerial, onSerialChange])

  useEffect(() => {
    setWorkspaceView(initialSerial ? 'device' : chemistry === 'NCA' ? 'fleet' : 'cells')
  }, [chemistry, initialSerial])

  useEffect(() => {
    if (!selectedSerial) {
      setRuns([])
      return
    }
    let cancelled = false
    setRunsLoading(true)
    api.runs(selectedSerial)
      .then((items) => {
        if (!cancelled) setRuns(items)
      })
      .catch((reason) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason))
      })
      .finally(() => {
        if (!cancelled) setRunsLoading(false)
      })
    return () => { cancelled = true }
  }, [selectedSerial, completedJobVersion])

  useEffect(() => {
    setNameDraft(selectedDevice?.display_name ?? '')
    setEditingName(false)
    setNameError('')
  }, [selectedDevice?.serial, selectedDevice?.display_name])

  const pluginMap = useMemo(
    () => new Map(plugins.map((plugin) => [plugin.key, plugin])),
    [plugins],
  )
  const steps = selectedDevice?.device_role === 'full_cycle_test'
    ? [...preprocessingSteps, ...fullCycleSteps, electrothermalFeatureStep, analysisValidationStep, ...analysisSteps]
    : selectedDevice?.device_role === 'field_tester'
      ? [
          ...preprocessingSteps,
          ...(chemistry === 'NCA' ? fieldSteps : [electrothermalFeatureStep]),
          analysisValidationStep,
          ...analysisSteps,
          ...(chemistry === 'NCA' ? [sohGapDiagnosisStep] : []),
        ]
      : [...preprocessingSteps, ...analysisSteps]
  const workflowStates = selectedDevice
    ? steps.map((step) => stepState(step, selectedDevice, jobs, pluginMap.get(step.key)))
    : []

  function selectDevice(serial: string) {
    setSelectedSerial(serial)
    onSerialChange(serial)
    setError('')
  }

  function openFleetDevice(serial: string) {
    selectDevice(serial)
    setWorkspaceView('device')
  }

  async function createJob(pipelineKey: string, parameters: Record<string, unknown>) {
    await onCreateJob(pipelineKey, parameters)
  }

  async function saveDisplayName(event: FormEvent) {
    event.preventDefault()
    if (!selectedDevice || !nameDraft.trim()) return
    setNameBusy(true)
    setNameError('')
    try {
      await onUpdateDeviceDisplayName(selectedDevice.serial, nameDraft.trim())
      setEditingName(false)
    } catch (reason) {
      setNameError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setNameBusy(false)
    }
  }

  async function restoreDisplayName() {
    if (!selectedDevice) return
    setNameBusy(true)
    setNameError('')
    try {
      await onUpdateDeviceDisplayName(selectedDevice.serial, null)
      setEditingName(false)
    } catch (reason) {
      setNameError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setNameBusy(false)
    }
  }

  return (
    <div className="page-stack">
      <nav className="tabs ml-view-tabs" aria-label="Machine learning views">
        {chemistry === 'NCA' && (
          <button
            className={workspaceView === 'fleet' ? 'active' : ''}
            type="button"
            onClick={() => setWorkspaceView('fleet')}
          >
            <ChartNoAxesCombined size={16} aria-hidden="true" />NCA fleet capacity
          </button>
        )}
        <button
          className={workspaceView === 'cells' ? 'active' : ''}
          type="button"
          onClick={() => setWorkspaceView('cells')}
        >
          <Waves size={16} aria-hidden="true" />Cell signals
        </button>
        <button
          className={workspaceView === 'device' ? 'active' : ''}
          type="button"
          onClick={() => setWorkspaceView('device')}
        >
          <Workflow size={16} aria-hidden="true" />Device workflow
        </button>
      </nav>

      {workspaceView === 'fleet' && chemistry === 'NCA' ? (
        <NcaCapacityOverviewPage
          onOpenDevice={openFleetDevice}
          onOpenCellSignals={() => setWorkspaceView('cells')}
        />
      ) : workspaceView === 'cells' ? (
        <CellAnalysisPage
          chemistry={chemistry}
          devices={devices}
          plugins={plugins}
          jobs={jobs}
          onCreateJob={onCreateJob}
        />
      ) : (
        <div className="ml-layout">
      <aside className="device-browser" aria-label="Imported batteries">
        <div className="browser-heading">
          <span>Imported batteries</span>
          <small>{importedDevices.length}</small>
        </div>
        <div className="device-browser-list">
          {importedDevices.map((device) => (
            <button
              className={device.serial === selectedSerial ? 'active' : ''}
              key={device.serial}
              type="button"
              onClick={() => selectDevice(device.serial)}
            >
              <BatteryCharging size={17} aria-hidden="true" />
              <span><strong>{device.display_name}</strong><small>{device.serial} | {device.device_role.replaceAll('_', ' ')}</small></span>
              <ArrowRight size={15} aria-hidden="true" />
            </button>
          ))}
        </div>
      </aside>

      <div className="ml-workspace">
        {error && <div className="error-banner">{error}</div>}
        {!selectedDevice ? (
          <div className="empty-workspace">
            <Database size={26} aria-hidden="true" />
            <h1>No imported telemetry</h1>
            <p>Import a configured battery from the Extraction workspace.</p>
          </div>
        ) : (
          <>
            <header className="page-header ml-header">
              <div>
                <span className="eyebrow">Machine learning | {selectedDevice.serial}</span>
                {editingName ? (
                  <form className="device-name-editor" onSubmit={(event) => void saveDisplayName(event)}>
                    <input
                      autoFocus
                      maxLength={80}
                      value={nameDraft}
                      onChange={(event) => setNameDraft(event.target.value)}
                      aria-label="Device display name"
                    />
                    <button className="icon-button" type="submit" disabled={nameBusy || !nameDraft.trim()} title="Save device name"><Check size={16} aria-hidden="true" /></button>
                    <button className="icon-button" type="button" disabled={nameBusy} onClick={() => setEditingName(false)} title="Cancel"><X size={16} aria-hidden="true" /></button>
                  </form>
                ) : (
                  <div className="device-name-title">
                    <h1>{selectedDevice.display_name}</h1>
                    <button className="icon-button" type="button" onClick={() => setEditingName(true)} title="Rename device"><Pencil size={15} aria-hidden="true" /></button>
                    {selectedDevice.display_name_source === 'custom' && (
                      <button className="icon-button" type="button" disabled={nameBusy} onClick={() => void restoreDisplayName()} title="Restore configured name"><RotateCcw size={15} aria-hidden="true" /></button>
                    )}
                  </div>
                )}
                {nameError && <small className="field-error">{nameError}</small>}
              </div>
              <div className="device-role-block">
                <span className={`role-label role-${selectedDevice.device_role}`}>
                  {selectedDevice.device_role.replaceAll('_', ' ')}
                </span>
                <small>
                  {selectedDevice.nominal_capacity_ah == null
                    ? 'Nominal derived from cycle'
                    : `${number(selectedDevice.nominal_capacity_ah, 1)} Ah nominal`}
                  {selectedDevice.module_count_status === 'verified' && selectedDevice.module_count != null
                    ? ` / ${selectedDevice.module_count} modules / ${number(selectedDevice.system_energy_kwh, 0)} kWh`
                    : ''}
                </small>
              </div>
            </header>

            <dl className="device-spec-strip">
              <div>
                <dt>Installed modules</dt>
                <dd>{selectedDevice.module_count ?? 'n/a'}</dd>
                <span>{selectedDevice.module_count_status}</span>
              </div>
              <div>
                <dt>Nominal capacity</dt>
                <dd>{number(selectedDevice.nominal_capacity_ah, 1)} Ah</dd>
                <span>{selectedDevice.module_count != null && selectedDevice.module_count > 2 ? 'derived from modules' : 'calibrated two-module value'}</span>
              </div>
              <div>
                <dt>System energy</dt>
                <dd>{number(selectedDevice.system_energy_kwh, 0)} kWh</dd>
                <span>{number(selectedDevice.module_energy_kwh, 0)} kWh per module</span>
              </div>
              <div>
                <dt>Transferred slope</dt>
                <dd>{number(transferredSlope, 4)} Ah / 1k Ah</dd>
                <span>{selectedDevice.loss_slope_reference_module_count ?? 'n/a'}-module reference</span>
              </div>
              <div>
                <dt>SOH loss / 1k Ah</dt>
                <dd>{number(normalizedSohLossPer1000Ah, 4)} pp</dd>
                <span>using this system capacity</span>
              </div>
            </dl>

            <section className="workflow-section">
              <div className="section-title">
                <div><h2>Processing workflow</h2><span>Stages allowed by the assigned device role</span></div>
              </div>
              <div className="workflow-list">
                <div className="workflow-row workflow-source-row">
                  <span className="workflow-index"><Check size={16} aria-hidden="true" /></span>
                  <span className="workflow-icon"><Database size={19} aria-hidden="true" /></span>
                  <div className="workflow-copy"><strong>Raw telemetry</strong><span>{selectedDevice.raw_file_count} Parquet files, {selectedDevice.raw_row_count.toLocaleString()} rows</span></div>
                  <span className="step-state step-complete">Available</span>
                  <span className="workflow-action-spacer" />
                </div>
                {steps.map((step, index) => {
                  const plugin = pluginMap.get(step.key)
                  const state = workflowStates[index]
                  const StepIcon = step.icon
                  const activeJob = latestJob(jobs, selectedDevice.serial, step.key)
                  return (
                    <div className={`workflow-row workflow-${state}`} key={step.key}>
                      <span className="workflow-index">
                        {state === 'complete' ? <Check size={16} aria-hidden="true" /> : state === 'running' ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : state === 'failed' ? <CircleAlert size={16} aria-hidden="true" /> : state === 'locked' || state === 'unavailable' ? <LockKeyhole size={15} aria-hidden="true" /> : index + 2}
                      </span>
                      <span className="workflow-icon"><StepIcon size={19} aria-hidden="true" /></span>
                      <div className="workflow-copy"><strong>{step.title}</strong><span>{selectedDevice.workflow?.[plugin?.workflow_stage ?? '']?.reason ?? step.detail}</span></div>
                      <span className={`step-state step-${state}`}>{stateLabel(state)}</span>
                      <button
                        className={state === 'complete' ? 'button secondary compact-button' : 'button primary compact-button'}
                        type="button"
                        disabled={state === 'locked' || state === 'running' || state === 'unavailable'}
                        onClick={() => plugin && setDialogPlugin(plugin)}
                        title={activeJob?.error_message ?? undefined}
                      >
                        <Play size={14} aria-hidden="true" />
                        {state === 'complete' ? 'Run again' : state === 'stale' ? 'Update' : state === 'failed' ? 'Retry' : 'Run'}
                      </button>
                    </div>
                  )
                })}
              </div>
            </section>

            <section className="run-library-section">
              <div className="section-title">
                <div><h2>Analysis reports</h2><span>Completed model and verification runs</span></div>
                <div className="section-actions">
                  {runsLoading && <LoaderCircle size={17} className="spin" aria-label="Loading reports" />}
                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => setReportsCollapsed((collapsed) => !collapsed)}
                    title={reportsCollapsed ? 'Expand analysis reports' : 'Collapse analysis reports'}
                    aria-label={reportsCollapsed ? 'Expand analysis reports' : 'Collapse analysis reports'}
                    aria-expanded={!reportsCollapsed}
                  >
                    {reportsCollapsed
                      ? <ChevronDown size={16} aria-hidden="true" />
                      : <ChevronUp size={16} aria-hidden="true" />}
                  </button>
                </div>
              </div>
              {!reportsCollapsed && (runs.length === 0 && !runsLoading ? (
                <div className="empty-state compact-empty">No analysis report available for this battery.</div>
              ) : (
                <div className="run-list">
                  {runs.map((run) => (
                    <button
                      className="run-row"
                      type="button"
                      key={run.run_id}
                      onClick={() => {
                        if (run.model_family === 'bms_soh_recalibration_evidence') {
                          onOpenCapacitySoh(selectedSerial)
                          return
                        }
                        onViewRun(run.run_id)
                      }}
                    >
                      <span className="run-family-icon"><FileChartColumn size={18} aria-hidden="true" /></span>
                      <span className="run-copy"><strong>{familyLabel(run.model_family)}</strong><small>{shortDateTime(run.run_created_at)}</small></span>
                      <span className="run-metric"><small>{run.model_family === 'bms_soh_signal_audit' ? 'Qualified days' : run.model_family === 'seasonal_charging_stress_profile' ? 'Qualified months' : 'Validation MAE'}</small><strong>{run.model_family === 'bms_soh_signal_audit' || run.model_family === 'seasonal_charging_stress_profile' ? run.training_rows ?? 'n/a' : `${number(run.validation_mae_soh_pct, 2)}%`}</strong></span>
                      <span className="run-metric"><small>{run.model_family === 'bms_soh_signal_audit' ? 'Observed equivalent' : run.model_family === 'seasonal_charging_stress_profile' ? 'Time basis' : 'Loss slope'}</small><strong>{run.model_family === 'seasonal_charging_stress_profile' ? 'UTC' : run.loss_slope_per_1000ah == null ? 'n/a' : number(run.loss_slope_per_1000ah, 4)}</strong></span>
                      <ArrowRight size={17} aria-hidden="true" />
                    </button>
                  ))}
                </div>
              ))}
            </section>
          </>
        )}
      </div>

      <PipelineDialog
        open={dialogPlugin != null}
        serial={selectedDevice?.serial ?? ''}
        plugins={dialogPlugin ? [dialogPlugin] : []}
        eyebrow="Pipeline stage"
        title={dialogPlugin?.name ?? 'Run stage'}
        submitLabel="Start stage"
        parameterOverrides={dialogParameterOverrides}
        onClose={() => setDialogPlugin(null)}
        onSubmit={createJob}
      />
        </div>
      )}
    </div>
  )
}
