import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  BatteryMedium,
  CheckCircle2,
  Clock3,
  DatabaseZap,
  RefreshCw,
  TriangleAlert,
} from 'lucide-react'
import { api } from '../api'
import type {
  BmsSohRecalibrationSeries,
  CapacityBmsDischargeEvent,
  DeviceLibraryItem,
} from '../types'
import { BmsSohRecalibrationTrendChart } from './BmsSohRecalibrationTrendChart'

interface Props {
  devices: DeviceLibraryItem[]
  initialSerial?: string
}

const PAGE_SIZE = 80

const number = (value: unknown, digits = 1) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a'
const integer = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) ? Math.round(value).toLocaleString() : '0'
const percent = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) ? `${(100 * value).toFixed(0)}%` : 'n/a'
const readable = (value: unknown) =>
  value == null || value === '' ? 'n/a' : String(value).replaceAll('_', ' ')

function utc(value: unknown) {
  return value
    ? `${new Date(String(value)).toLocaleString('en-GB', {
        timeZone: 'UTC', dateStyle: 'medium', timeStyle: 'short',
      })} UTC`
    : 'n/a'
}

function horizon(hours: number | null) {
  if (hours == null) return 'No match'
  if (hours <= 12) return '<=12 h'
  if (hours <= 24) return '<=24 h'
  if (hours <= 48) return '<=48 h'
  return '<=72 h'
}

function qualityTone(value: string) {
  if (value === 'high') return 'step-complete'
  if (value === 'medium') return 'step-ready'
  return 'step-locked'
}

export function SohGapDiagnosisPage({ devices, initialSerial = '' }: Props) {
  const choices = useMemo(() => devices.filter((device) => device.chemistry === 'NCA'), [devices])
  const [serial, setSerial] = useState(initialSerial || choices[0]?.serial || '100000001')
  const [result, setResult] = useState<BmsSohRecalibrationSeries | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [evidenceFilter, setEvidenceFilter] = useState<'all' | 'capacity' | 'attribution'>('all')
  const [visibleRows, setVisibleRows] = useState(PAGE_SIZE)

  useEffect(() => {
    if (initialSerial) setSerial(initialSerial)
  }, [initialSerial])

  useEffect(() => {
    setVisibleRows(PAGE_SIZE)
    setEvidenceFilter('all')
    void load()
  }, [serial])

  async function load() {
    setLoading(true)
    setError('')
    try {
      setResult(await api.bmsSohRecalibrationSeries(serial))
    } catch (reason) {
      setResult(null)
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setLoading(false)
    }
  }

  const phase2 = result?.summary.phase2?.phase2_version === 'v4'
    ? result.summary.phase2
    : undefined
  const downwardTransitions = useMemo(
    () => (result?.soh_transitions ?? [])
      .filter((row) => row.direction === 'down')
      .sort((a, b) => Date.parse(b.transition_time_utc) - Date.parse(a.transition_time_utc)),
    [result],
  )
  const matchesByTransition = useMemo(
    () => new Map((result?.discharge_matches ?? []).map((row) => [row.transition_id, row])),
    [result],
  )
  const candidateCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const candidate of result?.match_candidates ?? []) {
      counts.set(candidate.transition_id, (counts.get(candidate.transition_id) ?? 0) + 1)
    }
    return counts
  }, [result])
  const phase3ByEvent = useMemo(
    () => new Map((result?.electrothermal_context ?? []).map((row) => [row.discharge_event_id, row])),
    [result],
  )
  const filteredDischarges = useMemo(() => {
    const rows = (result?.capacity_events ?? [])
      .filter((row) => row.meaningful_deep_discharge)
      .filter((row) => evidenceFilter === 'all'
        || (evidenceFilter === 'capacity' ? row.capacity_estimate_eligible : !row.capacity_estimate_eligible))
    return rows.sort((a, b) => Date.parse(b.end_utc ?? '') - Date.parse(a.end_utc ?? ''))
  }, [evidenceFilter, result])
  const displayedDischarges = filteredDischarges.slice(0, visibleRows)
  const latestDischarge = filteredDischarges[0]
  const matchedTransitions = downwardTransitions.filter(
    (row) => matchesByTransition.get(row.transition_id)?.primary_preceding_discharge_event_id,
  ).length

  return <div className="page-stack phase2-dashboard">
    <header className="page-heading compact-heading phase2-heading">
      <div>
        <span className="eyebrow">NCA Phase 2</span>
        <h1>Capacity &amp; BMS SOH</h1>
        <p className="page-subtitle">Which deep discharge preceded each downward BMS SOH update, and what usable-capacity evidence did it provide?</p>
      </div>
      <div className="phase2-heading-actions">
        <label className="compact-field"><span>Battery</span><select value={serial} onChange={(event) => setSerial(event.target.value)}>{choices.map((device) => <option key={device.serial} value={device.serial}>{device.display_name || device.serial} ({device.serial})</option>)}</select></label>
        <button className="icon-button" type="button" onClick={() => void load()} disabled={loading} title="Refresh Phase 2 evidence" aria-label="Refresh Phase 2 evidence"><RefreshCw size={17} className={loading ? 'spin' : ''} aria-hidden="true" /></button>
      </div>
    </header>

    {loading && !result && <div className="section-loader">Loading latest Phase 2 evidence...</div>}
    {error && <div className="error-banner">{error}. Run Capacity and SOH investigation for this device first.</div>}

    {result && phase2 && <>
      <section className="phase2-status-band">
        <div><CheckCircle2 size={18} aria-hidden="true" /><span><strong>Phase 2 evidence available</strong><small>Run {result.run_id.replace('recalibration_run=', '')} | UTC | {readable(phase2.phase2_version)}</small></span></div>
        <p>Association means a discharge occurred before an SOH update. It does not prove causation.</p>
      </section>

      <section className="metric-grid phase2-metrics">
        <div className="metric-card"><Activity size={18} /><span>Downward SOH updates</span><strong>{integer(downwardTransitions.length)}</strong><small>Persistent audited transitions</small></div>
        <div className="metric-card"><Clock3 size={18} /><span>Matched within 72 h</span><strong>{matchedTransitions} / {downwardTransitions.length}</strong><small>Deepest preceding discharge selected</small></div>
        <div className="metric-card"><BatteryMedium size={18} /><span>Meaningful discharges</span><strong>{integer(phase2.meaningful_80pp_discharge_rows)}</strong><small>SOC drop at least 80 pp</small></div>
        <div className="metric-card"><DatabaseZap size={18} /><span>Capacity estimates</span><strong>{integer(phase2.capacity_estimate_90pp_rows)}</strong><small>At least 90 pp; {integer(phase2.strict_full_discharge_rows)} strict 100 to 0</small></div>
      </section>

      <section className="panel chart-panel phase2-chart-panel">
        <div className="section-title"><div><h2>BMS SOH and discharge evidence</h2><span>Daily audited BMS SOH with persistent updates and capacity observations. Zoom without reloading the dataset.</span></div>{latestDischarge && <div className="section-kicker"><strong>{number(latestDischarge.soc_span_pp)} pp</strong><span>latest deep discharge</span></div>}</div>
        <BmsSohRecalibrationTrendChart points={result.points} events={result.events} capacityEvents={result.capacity_events} />
      </section>

      <section className="panel phase2-transition-panel">
        <div className="section-title"><div><h2>Downward SOH transition evidence</h2><span>Primary means deepest within 72 hours; nearest remains available in the stored artifact.</span></div></div>
        {downwardTransitions.length ? <div className="table-wrap"><table><thead><tr><th>SOH update UTC</th><th>BMS change</th><th>Deepest preceding discharge</th><th>Delay</th><th>Ah evidence</th><th>Capacity comparison</th><th>Candidates</th></tr></thead><tbody>{downwardTransitions.map((transition) => {
          const match = matchesByTransition.get(transition.transition_id)
          const matched = Boolean(match?.primary_preceding_discharge_event_id)
          return <tr key={transition.transition_id}>
            <td><strong>{utc(transition.transition_time_utc)}</strong><small className="table-subtext">{readable(transition.timestamp_quality)} timing | coverage {percent(transition.data_coverage_fraction)}</small></td>
            <td><strong>{number(transition.previous_soh_pct)}% to {number(transition.new_soh_pct)}%</strong><small className="table-subtext">{number(transition.delta_soh_pp, 2)} pp</small></td>
            <td>{matched ? <><strong>{number(match?.preceding_discharge_start_soc_pct)}% to {number(match?.preceding_discharge_end_soc_pct)}%</strong><small className="table-subtext">drop {number(match?.preceding_discharge_soc_drop_pp)} pp | {readable(match?.discharge_evidence_class)}</small></> : <><strong>No qualifying discharge</strong><small className="table-subtext">No &gt;=80 pp discharge ended in the lookback</small></>}</td>
            <td><span className={`step-state ${matched ? 'step-complete' : 'step-locked'}`}>{horizon(match?.discharge_end_to_soh_update_hours ?? null)}</span><small className="table-subtext">{number(match?.discharge_end_to_soh_update_hours, 1)} h exact delay</small></td>
            <td>{matched ? <><strong>{number(match?.preceding_discharge_net_ah, 2)} Ah net</strong><small className="table-subtext">gross {number(match?.preceding_discharge_gross_ah, 2)} | intrusion {number(match?.preceding_discharge_charge_intrusion_ah, 3)} Ah</small></> : 'n/a'}</td>
            <td>{match?.discharge_derived_soh_proxy_pct != null ? <><strong>{number(match.discharge_derived_soh_proxy_pct, 2)}% calculated SOH</strong><small className="table-subtext">BMS difference {number(match.bms_minus_capacity_proxy_pp, 2)} pp</small></> : <><strong>Not calculated</strong><small className="table-subtext">Requires at least 90 pp SOC span</small></>}</td>
            <td><strong>{candidateCounts.get(transition.transition_id) ?? 0}</strong><small className="table-subtext">all candidates retained</small></td>
          </tr>
        })}</tbody></table></div> : <div className="empty-state compact-empty">No persistent downward BMS SOH transition is available in this run.</div>}
      </section>

      <section className="panel phase2-discharge-panel">
        <div className="section-title"><div><h2>Physical discharge evidence</h2><span>One row per rest-aware physical discharge. Short charge intrusions remain inside until SOC recovers by 1 pp.</span></div><div className="segmented-control" role="group" aria-label="Discharge evidence filter"><button type="button" className={evidenceFilter === 'all' ? 'active' : ''} onClick={() => setEvidenceFilter('all')}>All</button><button type="button" className={evidenceFilter === 'capacity' ? 'active' : ''} onClick={() => setEvidenceFilter('capacity')}>Capacity &gt;=90 pp</button><button type="button" className={evidenceFilter === 'attribution' ? 'active' : ''} onClick={() => setEvidenceFilter('attribution')}>Attribution 80-90 pp</button></div></div>
        <div className="table-wrap"><table><thead><tr><th>Discharge UTC</th><th>SOC span</th><th>Net throughput</th><th>Calculated SOH</th><th>Continuity</th><th>Quality</th><th>Phase 3</th></tr></thead><tbody>{displayedDischarges.map((event: CapacityBmsDischargeEvent) => {
          const context = phase3ByEvent.get(event.discharge_event_id)
          return <tr key={event.discharge_event_id}>
            <td><strong>{utc(event.end_utc)}</strong><small className="table-subtext">started {utc(event.start_utc)}</small></td>
            <td><strong>{number(event.soc_start_pct)}% to {number(event.soc_end_pct)}%</strong><small className="table-subtext">{number(event.soc_span_pp)} pp | {readable(event.evidence_class)}</small></td>
            <td><strong>{number(event.net_discharged_ah, 2)} Ah</strong><small className="table-subtext">gross {number(event.gross_discharged_ah, 2)} | intrusion {number(event.charge_intrusion_ah, 3)} Ah</small></td>
            <td>{event.capacity_estimate_eligible ? <><strong>{number(event.calculated_soh_pct, 2)}%</strong><small className="table-subtext">{number(event.estimated_full_capacity_ah, 2)} Ah estimated full capacity</small></> : <><strong>Attribution only</strong><small className="table-subtext">No capacity extrapolation below 90 pp</small></>}</td>
            <td><strong>{readable(event.continuity)}</strong><small className="table-subtext">{integer(event.rest_interruption_count)} rest | {integer(event.charge_intrusion_count)} charge intrusion | {readable(event.trajectory_end_reason)}</small></td>
            <td><span className={`step-state ${qualityTone(event.measurement_quality)}`}>{readable(event.measurement_quality)}</span><small className="table-subtext">coverage {percent(event.data_coverage_fraction)} | max gap {number(event.maximum_gap_s, 0)} s</small></td>
            <td>{context ? <><strong>{readable(context.phase3_status)}</strong><small className="table-subtext">{number(context.temperature_mean_c)} C | RMS {number(context.rms_c_rate, 2)} C</small></> : <><strong>Not attached</strong><small className="table-subtext">Phase 2 remains valid</small></>}</td>
          </tr>
        })}</tbody></table></div>
        {visibleRows < filteredDischarges.length && <div className="table-load-more"><button className="button secondary" type="button" onClick={() => setVisibleRows((value) => value + PAGE_SIZE)}>Show {Math.min(PAGE_SIZE, filteredDischarges.length - visibleRows)} more</button><span>{visibleRows} of {filteredDischarges.length}</span></div>}
      </section>

      <details className="panel evidence-details phase2-method"><summary>Detection and calculation rules</summary><div className="phase2-rule-grid"><div><strong>Physical discharge</strong><span>Discharge continues through rest. Charging closes it only after at least 1 pp SOC recovery.</span></div><div><strong>SOH attribution</strong><span>All &gt;=80 pp discharges ending within 72 hours before a downward transition are retained.</span></div><div><strong>Capacity estimate</strong><span>Only spans &gt;=90 pp: net Ah / SOC-span fraction, normalized by configured nominal capacity.</span></div><div><strong>Scientific boundary</strong><span>Temporal association is evidence for investigation, not proof that the discharge caused recalibration.</span></div></div></details>
      <section className="inline-notice"><TriangleAlert size={17} /><span><strong>Interpretation limit</strong><br />{phase2.claim_limit}</span></section>
    </>}

    {result && !phase2 && !loading && <div className="empty-state"><TriangleAlert size={22} /><h2>Updated Phase 2 run required</h2><p>The latest run uses the previous evidence contract. Run Capacity and SOH investigation again.</p></div>}
  </div>
}
