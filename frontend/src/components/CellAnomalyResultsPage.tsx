import {
  Activity,
  BatteryCharging,
  CircleAlert,
  Clock3,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type {
  CellAnomalyOverview,
  CellAnomalyOverviewDevice,
  CellAnomalySeries,
  BatteryChemistry,
  DeviceLibraryItem,
  PipelineJob,
  PipelinePlugin,
} from '../types'
import { integer, number, shortDate, shortDateTime } from '../utils'
import { CellAnomalyTrendChart } from './CellAnomalyTrendChart'
import { MetricCard } from './MetricCard'
import { PipelineDialog } from './PipelineDialog'

interface CellAnomalyResultsPageProps {
  chemistry: BatteryChemistry
  devices: DeviceLibraryItem[]
  plugins: PipelinePlugin[]
  jobs: PipelineJob[]
  onCreateJob: (
    pipelineKey: string,
    parameters: Record<string, unknown>,
  ) => Promise<void>
  initialSerial?: string
}

const ANOMALY_KEY = 'veera.hierarchical_voltage_anomaly.v2'

const severityRank: Record<string, number> = {
  severe: 6,
  serious: 5,
  engineering_review: 4,
  anomaly_candidate: 3,
  monitor: 2,
  acceptable: 1,
  excellent: 0,
}

function isActive(job: PipelineJob) {
  return job.status === 'queued' || job.status === 'running'
}

function readableSeverity(value: string | null | undefined) {
  return (value ?? 'none').replaceAll('_', ' ')
}

function mostFrequent(values: Array<string | null | undefined>) {
  const counts = new Map<string, number>()
  values.forEach((value) => {
    if (value) counts.set(value, (counts.get(value) ?? 0) + 1)
  })
  return [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? null
}

export function CellAnomalyResultsPage({
  chemistry,
  plugins,
  jobs,
  onCreateJob,
  initialSerial = '',
}: CellAnomalyResultsPageProps) {
  const [overview, setOverview] = useState<CellAnomalyOverview | null>(null)
  const [series, setSeries] = useState<CellAnomalySeries | null>(null)
  const [selectedSerial, setSelectedSerial] = useState('')
  const [runDevice, setRunDevice] = useState<CellAnomalyOverviewDevice | null>(null)
  const [loading, setLoading] = useState(true)
  const [seriesLoading, setSeriesLoading] = useState(false)
  const [error, setError] = useState('')
  const [preferredSerial, setPreferredSerial] = useState(initialSerial)
  const [episodesExpanded, setEpisodesExpanded] = useState(false)
  const anomalyPlugin = plugins.find((plugin) => plugin.key === ANOMALY_KEY)
  const activeSerials = useMemo(
    () => new Set(
      jobs
        .filter((job) => isActive(job) && job.pipeline_key === ANOMALY_KEY)
        .map((job) => job.serial),
    ),
    [jobs],
  )
  const completedVersion = useMemo(
    () => jobs
      .filter((job) => job.pipeline_key === ANOMALY_KEY && job.status === 'completed')
      .map((job) => `${job.job_id}:${job.completed_at ?? ''}`)
      .join('|'),
    [jobs],
  )
  const selectedCompletedVersion = useMemo(
    () => jobs
      .filter((job) => (
        job.pipeline_key === ANOMALY_KEY
        && job.serial === selectedSerial
        && job.status === 'completed'
      ))
      .map((job) => `${job.job_id}:${job.completed_at ?? ''}`)
      .join('|'),
    [jobs, selectedSerial],
  )

  useEffect(() => {
    if (initialSerial) setPreferredSerial(initialSerial)
  }, [initialSerial])

  function load() {
    setLoading(true)
    setError('')
    api.cellAnomalyOverview(chemistry)
      .then((payload) => {
        setOverview(payload)
        const analyzed = payload.devices.filter(
          (device) => device.in_active_cohort && device.analyzed,
        )
        const preferredDevice = payload.devices.find(
          (device) => device.serial === preferredSerial,
        )
        setSelectedSerial((current) => (
          analyzed.some((device) => device.serial === preferredSerial)
            ? preferredSerial
            : preferredDevice && !preferredDevice.analyzed
              ? ''
            : analyzed.some((device) => device.serial === current)
            ? current
            : analyzed[0]?.serial ?? ''
        ))
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [chemistry, completedVersion])

  useEffect(() => {
    let cancelled = false
    if (!selectedSerial) {
      setSeries(null)
      return
    }
    setSeriesLoading(true)
    setError('')
    api.cellAnomalySeries(selectedSerial)
      .then((payload) => {
        if (!cancelled) setSeries(payload)
      })
      .catch((reason) => {
        if (cancelled) return
        setSeries(null)
        setError(reason instanceof Error ? reason.message : String(reason))
      })
      .finally(() => {
        if (!cancelled) setSeriesLoading(false)
      })
    return () => { cancelled = true }
  }, [selectedSerial, selectedCompletedVersion])

  if (loading && !overview) {
    return <div className="section-loader full-page-loader">Loading anomaly results...</div>
  }

  const metrics = overview?.metrics
  const rankedDevices = [...(overview?.devices ?? [])].sort((left, right) => (
    (left.module_count ?? Number.POSITIVE_INFINITY) - (right.module_count ?? Number.POSITIVE_INFINITY)
    || Number(right.in_active_cohort) - Number(left.in_active_cohort)
    || Number(right.cohort_role === 'reference_control') - Number(left.cohort_role === 'reference_control')
    || right.elevated_module_epoch_shift_count - left.elevated_module_epoch_shift_count
    || right.sustained_anomaly_episode_count - left.sustained_anomaly_episode_count
    || left.serial.localeCompare(right.serial)
  ))
  const allEpisodes = series?.episodes ?? []
  const prioritizedEpisodes = [...allEpisodes]
    .sort((left, right) => (
      (severityRank[right.severity ?? ''] ?? -1) - (severityRank[left.severity ?? ''] ?? -1)
      || (right.maximum_spread_mv ?? -1) - (left.maximum_spread_mv ?? -1)
      || Date.parse(right.start_time) - Date.parse(left.start_time)
    ))
    .slice(0, 50)
  const selectedDevice = overview?.devices.find(
    (device) => device.serial === selectedSerial,
  )
  const preferredDevice = overview?.devices.find(
    (device) => device.serial === preferredSerial,
  )
  const analysisSummary = series?.summary
  const highestEpisode = prioritizedEpisodes[0]
  const dominantChannel = mostFrequent(allEpisodes.map((episode) => episode.affected_channel_key))
  const episodeTemperatures = allEpisodes.flatMap((episode) => (
    [episode.minimum_temperature_c, episode.mean_temperature_c, episode.maximum_temperature_c]
      .filter((value): value is number => value != null && Number.isFinite(value))
  ))
  const episodeTemperatureRange = episodeTemperatures.length > 0
    ? `${number(Math.min(...episodeTemperatures), 1)} to ${number(Math.max(...episodeTemperatures), 1)} C`
    : 'No temperature context'
  const stateMinutes = (series?.points ?? []).reduce(
    (totals, point) => ({
      charging: totals.charging + point.sustained_charging_minutes,
      discharging: totals.discharging + point.sustained_discharging_minutes,
      rest: totals.rest + point.sustained_rest_minutes,
      mixed: totals.mixed + point.sustained_mixed_minutes,
    }),
    { charging: 0, discharging: 0, rest: 0, mixed: 0 },
  )
  const dominantState = Object.entries(stateMinutes)
    .sort((left, right) => right[1] - left[1])[0]
  const totalEpisodeCount = analysisSummary?.sustained_anomaly_episode_count ?? allEpisodes.length
  const evidenceStatus = analysisSummary?.evidence_status
  const validMaximumSpread = analysisSummary?.maximum_analysis_spread_mv
    ?? selectedDevice?.maximum_analysis_spread_mv
  const validP95Spread = analysisSummary?.p95_analysis_spread_mv
    ?? selectedDevice?.p95_analysis_spread_mv

  async function createAnomalyJob(
    pipelineKey: string,
    parameters: Record<string, unknown>,
  ) {
    const serial = String(parameters.serial ?? runDevice?.serial ?? '')
    if (serial) setPreferredSerial(serial)
    await onCreateJob(pipelineKey, parameters)
  }

  return (
    <div className="page-stack cell-anomaly-page">
      {error && <div className="error-banner">{error}</div>}
      <header className="page-header cell-anomaly-header">
        <div>
          <span className="eyebrow">{chemistry} diagnostics by module cohort</span>
          <h1>Cell-voltage anomaly results</h1>
        </div>
        <button
          className="icon-button"
          type="button"
          onClick={load}
          disabled={loading}
          title="Refresh anomaly results"
        >
          <RefreshCw size={17} className={loading ? 'spin' : ''} aria-hidden="true" />
        </button>
      </header>

      {metrics && (
        <>
          <section className="metric-strip cell-anomaly-metrics">
            <MetricCard
              label="Devices analyzed"
              value={`${metrics.analyzed_device_count} / ${metrics.configured_device_count}`}
              detail={`${metrics.archived_legacy_device_count} legacy result${metrics.archived_legacy_device_count === 1 ? '' : 's'} archived and excluded`}
              icon={Activity}
              tone={metrics.analyzed_device_count ? 'good' : 'warning'}
            />
            <MetricCard
              label="Candidate systems"
              value={integer(metrics.candidate_device_count)}
              detail="With at least one sustained event"
              icon={ShieldAlert}
              tone={metrics.candidate_device_count ? 'warning' : 'good'}
            />
            <MetricCard
              label="Sustained episodes"
              value={integer(metrics.total_sustained_episode_count)}
              detail={`${integer(metrics.total_anomaly_days)} device-days affected`}
              icon={Clock3}
              tone={metrics.total_sustained_episode_count ? 'warning' : 'good'}
            />
          </section>

          <div className="inline-notice evidence-notice">
            <CircleAlert size={15} aria-hidden="true" />
            <span>{overview?.evidence_policy.claim_limit}</span>
          </div>

          {preferredDevice && !preferredDevice.analyzed && (
            <div className="inline-notice evidence-notice">
              <CircleAlert size={15} aria-hidden="true" />
              <span>
                {preferredDevice.serial} has no active hierarchical anomaly result.
                {preferredDevice.history_built
                  ? ' Start its analysis from the highlighted row below.'
                  : ' Build its hierarchical five-minute history before analysis.'}
              </span>
            </div>
          )}

          {series && analysisSummary && !seriesLoading && (
            <section className="panel anomaly-findings">
              <div className="section-title">
                <div>
                  <h2>What the analysis found for {selectedDevice?.display_name ?? series.serial}</h2>
                  <span>Validated hierarchical cell-channel evidence from the completed run</span>
                </div>
                <small className="source-freshness">Run {series.run_id.replace('anomaly_run=', '')}</small>
              </div>
              <p className="anomaly-finding-summary">
                {totalEpisodeCount > 0
                  ? `${integer(totalEpisodeCount)} sustained episodes occurred on ${integer(analysisSummary.anomaly_days)} UTC days. The strongest quality-valid spread was ${number(validMaximumSpread, 1)} mV and the overall p95 was ${number(validP95Spread, 1)} mV.`
                  : `No sustained episode met the persistence rule. The strongest quality-valid spread was ${number(validMaximumSpread, 1)} mV and the overall p95 was ${number(validP95Spread, 1)} mV.`}
              </p>
              <dl className="anomaly-findings-grid">
                <div>
                  <dt>Usable signal</dt>
                  <dd>{number(analysisSummary.quality_valid_pct, 1)}%</dd>
                  <small>{integer(analysisSummary.quality_valid_rows ?? 0)} of {integer(analysisSummary.feature_rows ?? 0)} five-minute windows</small>
                </div>
                <div>
                  <dt>Valid spread</dt>
                  <dd>{number(validP95Spread, 1)} / {number(validMaximumSpread, 1)} mV</dd>
                  <small>overall p95 / maximum</small>
                </div>
                <div>
                  <dt>Highest severity</dt>
                  <dd>{readableSeverity(highestEpisode?.severity)}</dd>
                  <small>{highestEpisode ? `${number(highestEpisode.maximum_spread_mv, 1)} mV at ${shortDateTime(highestEpisode.start_time)}` : 'No retained episode'}</small>
                </div>
                <div>
                  <dt>Recurring channel</dt>
                  <dd>{dominantChannel ?? 'No recurring channel'}</dd>
                  <small>Most frequently identified channel across episodes</small>
                </div>
                <div>
                  <dt>Temperature during episodes</dt>
                  <dd>{episodeTemperatureRange}</dd>
                  <small>Observed pack-sensor context; not temperature-adjusted scoring</small>
                </div>
                <div>
                  <dt>Dominant operating state</dt>
                  <dd>{dominantState && dominantState[1] > 0 ? dominantState[0] : 'No sustained minutes'}</dd>
                  <small>{dominantState ? `${integer(dominantState[1])} retained minutes` : 'n/a'}</small>
                </div>
              </dl>
              {(analysisSummary.bms_global_spread_implausible_windows ?? 0) > 0 && (
                <div className="inline-notice evidence-notice anomaly-reconciliation-notice">
                  <CircleAlert size={15} aria-hidden="true" />
                  <span>
                    Excluded {integer(analysisSummary.bms_global_spread_implausible_windows)} implausible BMS global-spread windows from scoring and plotting. The largest reported global value was {number(analysisSummary.maximum_bms_reported_voltage_spread_mv, 1)} mV; detector results above use the validated 56 cell channels instead.
                  </span>
                </div>
              )}
              {evidenceStatus === 'unavailable' && (
                <div className="inline-notice evidence-notice anomaly-reconciliation-notice">
                  <CircleAlert size={15} aria-hidden="true" />
                  <span>
                    The five-minute anomaly analysis completed, but one-second episode evidence could not be retrieved from the source database. {analysisSummary.evidence_error ?? 'The database was unavailable.'}
                  </span>
                </div>
              )}
              <small className="anomaly-claim-limit">
                {analysisSummary.ground_truth_status ?? 'No independent cell-voltage ground truth.'} {analysisSummary.claim_limit}
              </small>
            </section>
          )}

          <section className="panel cell-anomaly-chart-panel">
            <div className="section-title anomaly-chart-title">
              <div>
                <h2>Daily cell-voltage spread</h2>
                <span>Voltage severity with operating state and observed pack temperature in the UTC tooltip</span>
              </div>
              <label className="field anomaly-device-select">
                <span>Analyzed battery</span>
                <select
                  value={selectedSerial}
                  onChange={(event) => setSelectedSerial(event.target.value)}
                >
                  {(overview?.devices ?? [])
                    .filter((device) => device.in_active_cohort && device.analyzed)
                    .map((device) => (
                      <option key={device.serial} value={device.serial}>
                        {device.display_name} | {device.serial}
                      </option>
                    ))}
                </select>
              </label>
            </div>
            {seriesLoading && series?.serial !== selectedSerial
              ? <div className="section-loader anomaly-chart-loader"><LoaderCircle size={18} className="spin" />Loading daily series...</div>
              : <CellAnomalyTrendChart points={series?.points ?? []} />}
            {selectedDevice && (
              <dl className="anomaly-selection-strip">
                <div><dt>Analysis window</dt><dd>{shortDate(selectedDevice.analysis_start)} - {shortDate(selectedDevice.analysis_end)}</dd></div>
                <div><dt>Quality-valid bins</dt><dd>{number(selectedDevice.quality_valid_pct, 1)}%</dd></div>
                <div><dt>Sustained episodes</dt><dd>{integer(selectedDevice.sustained_anomaly_episode_count)}</dd></div>
                <div><dt>Spread p95</dt><dd>{number(selectedDevice.p95_analysis_spread_mv, 1)} mV</dd></div>
              </dl>
            )}
          </section>

          <section className="panel cell-anomaly-episodes">
            <div className="section-title">
              <div>
                <h2>Episode evidence</h2>
                <span>Highest engineering severity first, then newest episodes</span>
              </div>
              <small className="source-freshness">
                {prioritizedEpisodes.length} of {integer(totalEpisodeCount)} highest-priority episodes shown
              </small>
              <button
                className="icon-button"
                type="button"
                onClick={() => setEpisodesExpanded((current) => !current)}
                aria-expanded={episodesExpanded}
                title={episodesExpanded ? 'Collapse episode evidence' : 'Expand episode evidence'}
              >
                {episodesExpanded
                  ? <ChevronUp size={17} aria-hidden="true" />
                  : <ChevronDown size={17} aria-hidden="true" />}
              </button>
            </div>
            {!episodesExpanded ? (
              <div className="empty-state compact-empty">
                {integer(totalEpisodeCount)} sustained episodes. Expand to inspect the highest-priority evidence.
              </div>
            ) : prioritizedEpisodes.length === 0 ? (
              <div className="empty-state compact-empty">No sustained anomaly episode was retained for this battery.</div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Severity</th><th>UTC interval</th><th>Duration</th><th>Affected location</th><th>Voltage evidence</th><th>Operating context</th></tr></thead>
                  <tbody>
                    {prioritizedEpisodes.map((episode) => {
                      const critical = episode.severity === 'serious' || episode.severity === 'severe'
                      const spread = number(episode.maximum_spread_mv, 1)
                      const severity = (episode.severity ?? 'candidate').replaceAll('_', ' ')
                      return (
                        <tr key={episode.episode_id}>
                          <td><span className={`episode-severity ${critical ? 'critical' : 'warning'}`}>{severity}</span></td>
                          <td><strong>{shortDateTime(episode.start_time)}</strong><small className="episode-end">to {shortDateTime(episode.end_time)}</small></td>
                          <td className="numeric">{number(episode.duration_minutes, 0)} min</td>
                          <td>
                            <strong>{episode.affected_channel_key ?? 'Unknown'}</strong>
                            <small className="episode-end">module {episode.affected_module_index ?? 'n/a'}, pack {episode.affected_pack_index ?? 'n/a'}, channel {episode.affected_channel_index ?? 'n/a'}</small>
                          </td>
                          <td>
                            <strong>{spread} mV max</strong>
                            <small className="episode-end">{(episode.detection_reasons ?? 'contextual evidence').replaceAll('_', ' ')}</small>
                            <small className="episode-end">{episode.has_raw_evidence ? 'One-second evidence retained' : 'Five-minute evidence'}</small>
                          </td>
                          <td>
                            <strong>{episode.operating_states?.replaceAll(',', ', ') || 'Unknown state'}</strong>
                            <small className="episode-end">{number(episode.mean_soc_pct, 1)}% mean SOC</small>
                            <small className="episode-end">{number(episode.mean_current_a, 1)} A mean current</small>
                            <small className="episode-end">{number(episode.mean_temperature_c, 1)} C mean pack temperature</small>
                            <small className="episode-end">{number(episode.minimum_temperature_c, 1)} to {number(episode.maximum_temperature_c, 1)} C observed range</small>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <div className="inline-notice evidence-notice episode-rule">
              <CircleAlert size={15} aria-hidden="true" />
              <span>An episode requires at least three qualifying five-minute bins separated by no more than 15 minutes. Fixed bands are 20 mV monitor, 30 mV candidate, 50 mV engineering review, 100 mV serious, and 200 mV severe. Temperature is currently supporting context, not proof that temperature caused the voltage behavior.</span>
            </div>
          </section>

          <section className="panel cell-anomaly-table">
            <div className="section-title">
              <div>
                <h2>{chemistry} device comparison</h2>
                <span>Devices are ordered by module count; results remain separated between module configurations</span>
              </div>
              <small className="source-freshness">Source {shortDateTime(overview?.latest_source_modified_at)}</small>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Device</th>
                    <th>Modules</th>
                    <th>Analysis data</th>
                    <th>Analysis</th>
                    <th>Episodes</th>
                    <th>Days</th>
                    <th>Spread p95</th>
                    <th>Maximum spread</th>
                    <th>Latest event</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {rankedDevices.map((device) => {
                    const active = activeSerials.has(device.serial)
                    const analysisLabel = !device.in_active_cohort
                      ? 'Module count required'
                      : active
                        ? 'Running'
                        : device.analyzed
                          ? 'Complete'
                          : device.legacy_analysis_archived
                            ? 'New analysis required'
                          : 'Not run'
                    return (
                      <tr
                        key={device.serial}
                        className={device.serial === preferredSerial
                          ? 'analysis-target-row'
                          : device.in_active_cohort
                            ? undefined
                            : 'cohort-deferred-row'}
                      >
                        <td>
                          <span className="serial-cell">
                            <BatteryCharging size={14} aria-hidden="true" />
                            <span className="device-identity"><strong>{device.display_name}</strong><small>{device.serial}</small></span>
                          </span>
                        </td>
                        <td>{device.module_count ?? 'n/a'}</td>
                        <td>{device.history_built
                          ? 'Ready'
                          : device.legacy_history_archived
                            ? 'Old format; rebuild required'
                            : 'Not prepared'}</td>
                        <td>
                          <span className={`step-state step-${active ? 'running' : device.in_active_cohort && device.analyzed ? 'complete' : 'waiting'}`}>
                            {active && <LoaderCircle size={12} className="spin" aria-hidden="true" />}
                            {analysisLabel}
                          </span>
                        </td>
                        <td className="numeric">{device.analyzed ? integer(device.sustained_anomaly_episode_count) : 'n/a'}</td>
                        <td className="numeric">{device.analyzed ? integer(device.anomaly_days) : 'n/a'}</td>
                        <td className="numeric">{device.analyzed ? `${number(device.p95_analysis_spread_mv, 1)} mV` : 'n/a'}</td>
                        <td className="numeric">{device.analyzed ? `${number(device.maximum_analysis_spread_mv, 1)} mV` : 'n/a'}</td>
                        <td>{shortDate(device.latest_episode_end)}</td>
                        <td className="table-action cell-anomaly-actions">
                          {device.in_active_cohort && device.analyzed && (
                            <button
                              className="icon-button table-open-button"
                              type="button"
                              onClick={() => {
                                setPreferredSerial(device.serial)
                                setSelectedSerial(device.serial)
                              }}
                              title={`View ${device.serial}`}
                            >
                              <Activity size={15} aria-hidden="true" />
                            </button>
                          )}
                          <button
                            className="icon-button table-open-button"
                            type="button"
                            disabled={active || !anomalyPlugin || !device.history_built || !device.in_active_cohort}
                            onClick={() => setRunDevice(device)}
                            title={device.in_active_cohort
                              ? `Analyze ${device.serial}`
                              : `${device.serial} requires a verified module count`}
                          >
                            <Play size={15} aria-hidden="true" />
                          </button>
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

      <PipelineDialog
        open={runDevice != null && anomalyPlugin != null}
        serial={runDevice?.serial ?? ''}
        plugins={anomalyPlugin ? [anomalyPlugin] : []}
        eyebrow="Cell-voltage diagnostics"
        title="Analyze anomaly candidates"
        submitLabel="Start analysis"
        parameterOverrides={{ serial: runDevice?.serial ?? '' }}
        onClose={() => setRunDevice(null)}
        onSubmit={createAnomalyJob}
      />
    </div>
  )
}
