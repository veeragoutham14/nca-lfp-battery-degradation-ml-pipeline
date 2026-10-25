import { BatteryCharging } from 'lucide-react'
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import { ExtractionPage } from './components/ExtractionPage'
import { ChemistrySelectPage } from './components/ChemistrySelectPage'
import { HomePage } from './components/HomePage'
import { LoginPage } from './components/LoginPage'
import { MachineLearningPage } from './components/MachineLearningPage'
import { PortalShell, type PortalPage } from './components/PortalShell'
import type {
  AuthUser,
  BatteryChemistry,
  DeviceLibraryItem,
  Health,
  PipelineJob,
  PipelinePlugin,
} from './types'

interface RouteState {
  page: PortalPage
  runId: string
  serial: string
  chemistry: BatteryChemistry | null
}

const RAW_IMPORT_KEY = 'veera.raw_telemetry_import.v2'
const PREPARE_ANALYSIS_KEY = 'battryml.prepare_analysis_data.v1'
const ACTIVE_JOB_STATUSES = new Set(['queued', 'running'])
const RunResultsPage = lazy(() => import('./components/RunResultsPage').then(
  (module) => ({ default: module.RunResultsPage }),
))
const SeasonalProfilesPage = lazy(() => import('./components/SeasonalProfilesPage').then(
  (module) => ({ default: module.SeasonalProfilesPage }),
))
const SohGapDiagnosisPage = lazy(() => import('./components/SohGapDiagnosisPage').then(
  (module) => ({ default: module.SohGapDiagnosisPage }),
))

function readRoute(): RouteState {
  const value = window.location.hash.replace(/^#\/?/, '')
  const [path = 'home', query = ''] = value.split('?')
  const parameters = new URLSearchParams(query)
  const segments = path.split('/').filter(Boolean)
  const chemistry = segments[0]?.toUpperCase() === 'LFP'
    ? 'LFP'
    : segments[0]?.toUpperCase() === 'NCA'
      ? 'NCA'
      : null
  const scopedPath = chemistry ? segments.slice(1).join('/') : path
  if (scopedPath.startsWith('run/')) {
    return {
      page: 'run',
      runId: decodeURIComponent(scopedPath.slice(4)),
      serial: '',
      chemistry: chemistry ?? 'NCA',
    }
  }
  if (scopedPath === 'extraction') return { page: 'extraction', runId: '', serial: '', chemistry: chemistry ?? 'NCA' }
  if (scopedPath === 'ml') return { page: 'ml', runId: '', serial: parameters.get('serial') ?? '', chemistry: chemistry ?? 'NCA' }
  if (scopedPath === 'seasonal') return { page: 'seasonal', runId: '', serial: parameters.get('serial') ?? '', chemistry: chemistry ?? 'NCA' }
  if (scopedPath === 'gap') return { page: 'gap', runId: '', serial: parameters.get('serial') ?? '', chemistry: chemistry ?? 'NCA' }
  return { page: 'home', runId: '', serial: '', chemistry }
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [user, setUser] = useState<AuthUser | null>(null)
  const [devices, setDevices] = useState<DeviceLibraryItem[]>([])
  const [plugins, setPlugins] = useState<PipelinePlugin[]>([])
  const [jobs, setJobs] = useState<PipelineJob[]>([])
  const [route, setRoute] = useState<RouteState>(readRoute)
  const [selectedSerial, setSelectedSerial] = useState('')
  const [initializing, setInitializing] = useState(true)
  const [workspaceLoading, setWorkspaceLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState('')
  const jobsRef = useRef<PipelineJob[]>([])
  const pollInFlightRef = useRef(false)

  const replaceJobs = useCallback((nextJobs: PipelineJob[]) => {
    jobsRef.current = nextJobs
    setJobs(nextJobs)
  }, [])

  const mergeDevices = useCallback((updatedDevices: DeviceLibraryItem[]) => {
    if (updatedDevices.length === 0) return
    const updates = new Map(updatedDevices.map((device) => [device.serial, device]))
    setDevices((current) => current.map((device) => updates.get(device.serial) ?? device))
  }, [])

  const loadWorkspace = useCallback(async (refreshDevices = false) => {
    const [deviceData, pluginData, jobData] = await Promise.all([
      api.devices(refreshDevices),
      api.pipelines(),
      api.jobs(true),
    ])
    setDevices(deviceData)
    setPlugins(pluginData)
    replaceJobs(jobData)
  }, [replaceJobs])

  useEffect(() => {
    const handleHash = () => setRoute(readRoute())
    const handleUnauthorized = () => {
      setUser(null)
      setDevices([])
      setPlugins([])
      replaceJobs([])
    }
    window.addEventListener('hashchange', handleHash)
    window.addEventListener('battryml:unauthorized', handleUnauthorized)
    return () => {
      window.removeEventListener('hashchange', handleHash)
      window.removeEventListener('battryml:unauthorized', handleUnauthorized)
    }
  }, [replaceJobs])

  useEffect(() => {
    let cancelled = false
    api.health()
      .then((healthData) => {
        if (!cancelled) setHealth(healthData)
        return api.me()
      })
      .then((userData) => {
        if (!cancelled && userData.authenticated) {
          setWorkspaceLoading(true)
          setUser(userData)
        }
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setInitializing(false)
      })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!user) return
    setError('')
    setWorkspaceLoading(true)
    void loadWorkspace()
      .catch((reason) => {
        setError(reason instanceof Error ? reason.message : String(reason))
      })
      .finally(() => setWorkspaceLoading(false))
  }, [loadWorkspace, user])

  const hasActiveJobs = jobs.some((job) => ACTIVE_JOB_STATUSES.has(job.status))

  useEffect(() => {
    if (!user || !hasActiveJobs) return
    let cancelled = false

    const pollJobs = async () => {
      if (pollInFlightRef.current) return
      pollInFlightRef.current = true
      try {
        const jobData = await api.jobs(true)
        if (cancelled) return

        const previousJobs = new Map(jobsRef.current.map((job) => [job.job_id, job]))
        const terminalSerials = jobData
          .filter((job) => {
            const previous = previousJobs.get(job.job_id)
            return Boolean(previous && ACTIVE_JOB_STATUSES.has(previous.status) && !ACTIVE_JOB_STATUSES.has(job.status))
          })
          .map((job) => job.serial)

        // Parquet inventory scans are intentionally deferred until a job finishes.
        // Live progress comes from the lightweight jobs endpoint.
        const serialsToRefresh = [...new Set(terminalSerials)]
        if (serialsToRefresh.length > 0) {
          const refreshedDevices = (await Promise.all(
            serialsToRefresh.map((serial) => api.device(serial, true).catch(() => null)),
          )).filter((device): device is DeviceLibraryItem => device !== null)
          if (cancelled) return
          mergeDevices(refreshedDevices)
        }

        // Publish the terminal job state only after its persisted workflow outputs
        // have been rescanned, otherwise effect cleanup can cancel the device refresh.
        replaceJobs(jobData)
      } catch {
        // Background synchronization is best-effort and retries on the next poll.
        // Foreground actions retain their own visible, actionable error handling.
      } finally {
        pollInFlightRef.current = false
      }
    }

    void pollJobs()
    const timer = window.setInterval(() => { void pollJobs() }, 3_000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [hasActiveJobs, mergeDevices, replaceJobs, user])

  useEffect(() => {
    if (route.serial) setSelectedSerial(route.serial)
  }, [route.serial])

  const importPlugin = useMemo(
    () => plugins.find((plugin) => plugin.key === RAW_IMPORT_KEY),
    [plugins],
  )

  const chemistryDevices = useMemo(
    () => route.chemistry
      ? devices.filter((device) => device.chemistry === route.chemistry)
      : [],
    [devices, route.chemistry],
  )
  const chemistrySerials = useMemo(
    () => new Set(chemistryDevices.map((device) => device.serial)),
    [chemistryDevices],
  )
  const chemistryJobs = useMemo(
    () => jobs.filter((job) => chemistrySerials.has(job.serial)),
    [jobs, chemistrySerials],
  )

  function navigate(page: 'home' | 'extraction' | 'ml' | 'seasonal' | 'gap') {
    if (page === 'home' && !route.chemistry) {
      window.location.hash = '/home'
      return
    }
    const prefix = route.chemistry?.toLowerCase()
    if (!prefix) {
      window.location.hash = '/home'
      return
    }
    if (page === 'ml' && selectedSerial) {
      window.location.hash = `/${prefix}/ml?serial=${encodeURIComponent(selectedSerial)}`
      return
    }
    window.location.hash = page === 'home' ? `/${prefix}` : `/${prefix}/${page}`
  }

  function selectChemistry(chemistry: BatteryChemistry) {
    setSelectedSerial('')
    window.location.hash = `/${chemistry.toLowerCase()}`
  }

  function openMl(serial: string) {
    setSelectedSerial(serial)
    const prefix = route.chemistry?.toLowerCase() ?? 'nca'
    window.location.hash = `/${prefix}/ml?serial=${encodeURIComponent(serial)}`
  }

  function viewRun(runId: string) {
    const prefix = route.chemistry?.toLowerCase() ?? 'nca'
    window.location.hash = `/${prefix}/run/${encodeURIComponent(runId)}`
  }

  const handleSerialChange = useCallback((serial: string) => {
    setSelectedSerial(serial)
    const prefix = route.chemistry?.toLowerCase() ?? 'nca'
    const nextHash = `/${prefix}/ml?serial=${encodeURIComponent(serial)}`
    if (window.location.hash.slice(1) !== nextHash) {
      window.location.hash = nextHash
    }
  }, [route.chemistry])

  async function login(username: string, password: string) {
    const loggedInUser = await api.login(username, password)
    setWorkspaceLoading(true)
    setUser(loggedInUser)
    if (!window.location.hash) window.location.hash = '/home'
  }

  async function logout() {
    try {
      await api.logout()
    } finally {
      setUser(null)
      setDevices([])
      setPlugins([])
      replaceJobs([])
      setWorkspaceLoading(false)
    }
  }

  async function refresh() {
    setRefreshing(true)
    setError('')
    try {
      await api.scan()
      const [healthData] = await Promise.all([api.health(), loadWorkspace(true)])
      setHealth(healthData)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setRefreshing(false)
    }
  }

  async function createJob(pipelineKey: string, parameters: Record<string, unknown>) {
    const job = await api.createJob(pipelineKey, parameters)
    const nextJobs = [job, ...jobsRef.current.filter((item) => item.job_id !== job.job_id)]
    replaceJobs(nextJobs)
    if (pipelineKey === RAW_IMPORT_KEY || pipelineKey === PREPARE_ANALYSIS_KEY) {
      const updatedDevice = await api.device(job.serial, true)
      if (updatedDevice) mergeDevices([updatedDevice])
    }
  }

  async function archiveJob(jobId: string) {
    const updated = await api.archiveJob(jobId)
    replaceJobs(jobsRef.current.map((item) => item.job_id === jobId ? updated : item))
  }

  async function deleteJob(jobId: string) {
    await api.deleteJob(jobId)
    replaceJobs(jobsRef.current.filter((item) => item.job_id !== jobId))
  }

  async function updateDeviceDisplayName(serial: string, displayName: string | null) {
    const updated = await api.updateDeviceDisplayName(serial, displayName)
    setDevices((current) => current.map((device) => (
      device.serial === serial ? updated : device
    )))
  }

  if (initializing) {
    return (
      <div className="loading-screen">
        <BatteryCharging size={28} aria-hidden="true" />
        <span>Loading BatteryML...</span>
      </div>
    )
  }

  if (!user) return <LoginPage health={health} onLogin={login} />

  return (
    <PortalShell
      activePage={route.page}
      chemistry={route.chemistry}
      user={user}
      health={health}
      refreshing={refreshing}
      onNavigate={navigate}
      onChooseChemistry={() => { window.location.hash = '/home' }}
      onRefresh={() => void refresh()}
      onLogout={() => void logout()}
    >
      {error && <div className="error-banner global-error">{error}</div>}
      {workspaceLoading ? (
        <div className="section-loader full-page-loader">Loading device library...</div>
      ) : route.page === 'home' && route.chemistry == null ? (
        <ChemistrySelectPage devices={devices} onSelect={selectChemistry} />
      ) : route.page === 'home' && route.chemistry ? (
        <HomePage chemistry={route.chemistry} devices={chemistryDevices} jobs={chemistryJobs.filter((job) => !job.archived_at)} onNavigate={navigate} />
      ) : route.page === 'extraction' ? (
        <ExtractionPage
          chemistry={route.chemistry ?? 'NCA'}
          devices={chemistryDevices}
          jobs={chemistryJobs}
          plugins={plugins}
          importPlugin={importPlugin}
          onCreateJob={createJob}
          onArchiveJob={archiveJob}
          onDeleteJob={deleteJob}
          onOpenMl={openMl}
        />
      ) : route.page === 'ml' ? (
        <MachineLearningPage
          chemistry={route.chemistry ?? 'NCA'}
          devices={chemistryDevices}
          plugins={plugins}
          jobs={chemistryJobs.filter((job) => !job.archived_at)}
          initialSerial={route.serial || selectedSerial}
          onSerialChange={handleSerialChange}
          onCreateJob={createJob}
          onViewRun={viewRun}
          onOpenCapacitySoh={(serial) => {
            const prefix = (route.chemistry ?? 'NCA').toLowerCase()
            window.location.hash = `/${prefix}/gap?serial=${encodeURIComponent(serial)}`
          }}
          onUpdateDeviceDisplayName={updateDeviceDisplayName}
        />
      ) : route.page === 'seasonal' ? (
        <Suspense fallback={<div className="section-loader full-page-loader">Loading seasonal profiles...</div>}>
          <SeasonalProfilesPage
            chemistry={route.chemistry ?? 'NCA'}
            devices={chemistryDevices}
            plugins={plugins}
            jobs={chemistryJobs.filter((job) => !job.archived_at)}
            onCreateJob={createJob}
          />
        </Suspense>
      ) : route.page === 'gap' ? (
        <Suspense fallback={<div className="section-loader full-page-loader">Loading capacity and BMS evidence...</div>}>
          <SohGapDiagnosisPage
            devices={chemistryDevices}
            initialSerial={route.serial || selectedSerial}
          />
        </Suspense>
      ) : route.page === 'run' ? (
        <Suspense fallback={<div className="section-loader full-page-loader">Loading analysis charts...</div>}>
          <RunResultsPage runId={route.runId} onBack={() => navigate('ml')} />
        </Suspense>
      ) : null}
    </PortalShell>
  )
}

export default App
