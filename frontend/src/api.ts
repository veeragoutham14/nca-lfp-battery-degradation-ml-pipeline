import type {
  AnalysisRun,
  Artifact,
  AuthUser,
  BatteryChemistry,
  Battery,
  CellAnomalyOverview,
  CellAnomalySeries,
  BmsSohRecalibrationOverview,
  BmsSohRecalibrationSeries,
  CellSignalOverview,
  DataValidationResult,
  ElectrothermalValidationResult,
  Dataset,
  DeviceLibraryItem,
  Health,
  ImportPlan,
  ImportPurpose,
  ImportResolution,
  NcaCapacityOverview,
  PipelineJob,
  PipelinePlugin,
  RunSeries,
  SohGapDiagnosis,
  ValidationSeries,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    cache: 'no-store',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (response.status === 204) return undefined as T
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.toLowerCase().includes('application/json')) {
    const body = await response.text()
    const looksLikeHtml = body.trimStart().toLowerCase().startsWith('<!doctype')
      || body.trimStart().toLowerCase().startsWith('<html')
    throw new Error(looksLikeHtml
      ? `The frontend received HTML for ${path} (${response.status}). Reload the page; if it persists, verify the frontend API proxy.`
      : `The BatteryML API returned ${contentType || 'an unknown response type'} instead of JSON.`)
  }
  const payload = await response.json()
  if (!response.ok) {
    if (response.status === 401 && path !== '/api/auth/login') {
      window.dispatchEvent(new Event('battryml:unauthorized'))
    }
    const detail = (payload as { detail?: unknown }).detail
    throw new Error(typeof detail === 'string' ? detail : detail == null ? response.statusText : JSON.stringify(detail))
  }
  return payload as T
}

export const api = {
  login: (username: string, password: string) =>
    request<AuthUser>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  me: () => request<AuthUser>('/api/auth/me'),
  logout: () => request<void>('/api/auth/logout', { method: 'POST' }),
  health: () => request<Health>('/api/health'),
  devices: (refresh = false) =>
    request<DeviceLibraryItem[]>(`/api/devices${refresh ? '?refresh=true' : ''}`),
  device: (serial: string, refresh = false) =>
    request<DeviceLibraryItem>(
      `/api/devices/${encodeURIComponent(serial)}${refresh ? '?refresh=true' : ''}`,
    ),
  updateDeviceDisplayName: (serial: string, displayName: string | null) =>
    request<DeviceLibraryItem>(
      `/api/devices/${encodeURIComponent(serial)}/display-name`,
      {
        method: 'PATCH',
        body: JSON.stringify({ display_name: displayName }),
      },
    ),
  ncaCapacityOverview: (referenceSlope = 0.12, bmsTolerance = 1) =>
    request<NcaCapacityOverview>(
      `/api/analytics/nca-capacity-overview?reference_loss_slope_ah_per_1000ah=${encodeURIComponent(referenceSlope)}&bms_tolerance_pct=${encodeURIComponent(bmsTolerance)}`,
    ),
  cellSignalOverview: (chemistry: BatteryChemistry) =>
    request<CellSignalOverview>(`/api/analytics/cell-signal-overview?chemistry=${chemistry}`),
  dataValidation: (serial: string, chemistry: BatteryChemistry) =>
    request<DataValidationResult>(
      `/api/analytics/data-validation/${encodeURIComponent(serial)}?chemistry=${chemistry}`,
    ),
  electrothermalValidation: (serial: string, chemistry: BatteryChemistry) =>
    request<ElectrothermalValidationResult>(
      `/api/analytics/electrothermal-validation/${encodeURIComponent(serial)}?chemistry=${chemistry}`,
    ),
  importPlan: (
    serial: string,
    purpose: ImportPurpose,
    chemistry: BatteryChemistry,
    moduleCount: number | null,
    selections: Record<string, ImportResolution>,
  ) => {
    const parameters = new URLSearchParams({ serial, purpose, chemistry })
    if (moduleCount != null) parameters.set('module_count', String(moduleCount))
    if (purpose === 'custom') parameters.set('selections', JSON.stringify(selections))
    return request<ImportPlan>(`/api/import/plan?${parameters.toString()}`)
  },
  cellAnomalyOverview: (chemistry: BatteryChemistry) =>
    request<CellAnomalyOverview>(`/api/analytics/cell-anomaly-overview?chemistry=${chemistry}`),
  cellAnomalySeries: (serial: string) =>
    request<CellAnomalySeries>(
      `/api/analytics/cell-anomaly-series/${encodeURIComponent(serial)}`,
    ),
  bmsSohRecalibrationOverview: (chemistry: BatteryChemistry) =>
    request<BmsSohRecalibrationOverview>(
      `/api/analytics/bms-soh-recalibration-overview?chemistry=${chemistry}`,
    ),
  bmsSohRecalibrationSeries: (serial: string) =>
    request<BmsSohRecalibrationSeries>(
      `/api/analytics/bms-soh-recalibration-series/${encodeURIComponent(serial)}`,
    ),
  sohGapDiagnosis: (serial: string, chemistry: BatteryChemistry) =>
    request<SohGapDiagnosis>(
      `/api/analytics/soh-gap-diagnosis/${encodeURIComponent(serial)}?chemistry=${chemistry}`,
    ),
  batteries: () => request<Battery[]>('/api/batteries'),
  runs: (serial = '', modelFamily = '', limit = 100) => {
    const parameters = new URLSearchParams({ limit: String(limit) })
    if (serial) parameters.set('serial', serial)
    if (modelFamily) parameters.set('model_family', modelFamily)
    return request<AnalysisRun[]>(`/api/runs?${parameters.toString()}`)
  },
  latestRuns: (modelFamily: string) => request<AnalysisRun[]>(
    `/api/latest-runs?model_family=${encodeURIComponent(modelFamily)}`,
  ),
  run: (runId: string) => request<AnalysisRun>(`/api/runs/${encodeURIComponent(runId)}`),
  summary: (runId: string) => request<Record<string, unknown>>(`/api/runs/${encodeURIComponent(runId)}/summary`),
  series: (runId: string) => request<RunSeries>(`/api/runs/${encodeURIComponent(runId)}/series`),
  validation: (runId: string) => request<ValidationSeries>(`/api/runs/${encodeURIComponent(runId)}/validation`),
  artifacts: (runId: string) => request<Artifact[]>(`/api/runs/${encodeURIComponent(runId)}/artifacts`),
  datasets: (runId: string) => request<Dataset[]>(`/api/runs/${encodeURIComponent(runId)}/datasets`),
  pipelines: () => request<PipelinePlugin[]>('/api/pipelines'),
  jobs: (includeArchived = false) =>
    request<PipelineJob[]>(`/api/jobs${includeArchived ? '?include_archived=true' : ''}`),
  scan: () => request<{ runs_discovered: number }>('/api/catalog/scan', { method: 'POST' }),
  createJob: (pipelineKey: string, parameters: Record<string, unknown>) =>
    request<PipelineJob>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ pipeline_key: pipelineKey, parameters }),
    }),
  archiveJob: (jobId: string) =>
    request<PipelineJob>(`/api/jobs/${encodeURIComponent(jobId)}/archive`, {
      method: 'PATCH',
    }),
  deleteJob: (jobId: string) =>
    request<void>(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' }),
}
