import { AlertTriangle, Check, Gauge, LoaderCircle, Minus, ShieldCheck, Thermometer, X } from 'lucide-react'
import type { CSSProperties } from 'react'
import type { DataValidationResult, ElectrothermalValidationResult } from '../types'
import { compactNumber, shortDateTime } from '../utils'

interface DataValidationPanelProps {
  data: DataValidationResult | null
  loading: boolean
  running: boolean
  canValidate: boolean
  validationBlockedReason?: string
  onValidate: () => void
  electrothermalRunning: boolean
  canValidateElectrothermal: boolean
  onValidateElectrothermal: () => void
  electrothermalData: ElectrothermalValidationResult | null
  electrothermalLoading: boolean
}

function stateIcon(status: string) {
  if (status === 'ready' || status === 'verified' || status === 'pass') return <Check size={14} aria-hidden="true" />
  if (status === 'blocked' || status === 'conflict' || status === 'critical') return <X size={14} aria-hidden="true" />
  return <AlertTriangle size={14} aria-hidden="true" />
}

function duration(seconds: number | null | undefined): string {
  if (seconds == null) return 'n/a'
  if (seconds < 60) return `${seconds.toFixed(0)} s`
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)} h`
  return `${(seconds / 86400).toFixed(1)} d`
}

function coverageClass(value: number): string {
  if (value >= 80) return 'coverage-good'
  if (value >= 50) return 'coverage-warning'
  return 'coverage-poor'
}

function ElectrothermalValidationSummary({
  data,
  loading,
}: {
  data: ElectrothermalValidationResult | null
  loading: boolean
}) {
  if (loading) return <div className="validation-empty"><LoaderCircle className="spin" size={16} />Loading electrothermal validation...</div>
  if (!data) return <div className="validation-empty"><Thermometer size={18} /><span><strong>Electrothermal validation not run</strong><small>Validate imported temperature and voltage data before building features.</small></span></div>
  const groups = [
    ['Temperature', data.groups.temperature],
    ['Voltage groups', data.groups.cell_voltage],
  ] as const
  return (
    <details className="validation-section" open>
      <summary>Electrothermal feature readiness</summary>
      <div className="validation-readiness">
        {groups.map(([label, group]) => (
          <div className={`readiness-band readiness-${group.ready && !data.stale ? 'ready' : 'blocked'}`} key={label}>
            <span>{stateIcon(group.ready && !data.stale ? 'ready' : 'blocked')}{label}</span>
            <strong>{group.ready ? `${group.expected_channels} channels` : 'failed'}</strong>
            <small>{group.minimum_channel_coverage_pct.toFixed(1)}% minimum coverage | {shortDateTime(group.start_utc)} to {shortDateTime(group.end_utc)} UTC</small>
            {(group.duplicate_timestamps > 0 || group.misaligned_five_minute_rows > 0) && <small>{group.duplicate_timestamps} duplicate timestamps | {group.misaligned_five_minute_rows} misaligned rows</small>}
          </div>
        ))}
      </div>
      <p className="validation-claim-limit"><strong>{data.status}</strong> | {data.reason} Validated {shortDateTime(data.validated_at)}.</p>
    </details>
  )
}

export function DataValidationPanel({
  data,
  loading,
  running,
  canValidate,
  validationBlockedReason,
  onValidate,
  electrothermalRunning,
  canValidateElectrothermal,
  onValidateElectrothermal,
  electrothermalData,
  electrothermalLoading,
}: DataValidationPanelProps) {
  if (loading) {
    return <div className="validation-empty"><LoaderCircle className="spin" size={18} />Loading validation evidence...</div>
  }
  if (!data) {
    return (
      <div className="validation-empty">
        <ShieldCheck size={20} aria-hidden="true" />
        <span><strong>No analysis-input validation yet</strong><small>This final gate requires current operating events and built electrothermal features.</small></span>
        <button className="button secondary" type="button" disabled={!canValidate || running} onClick={onValidate}>
          {running ? <LoaderCircle className="spin" size={15} /> : <ShieldCheck size={15} />}
          Validate analysis inputs
        </button>
        {!canValidate && validationBlockedReason && <small>{validationBlockedReason}</small>}
        <button className="button secondary" type="button" disabled={!canValidateElectrothermal || electrothermalRunning} onClick={onValidateElectrothermal}>
          {electrothermalRunning ? <LoaderCircle className="spin" size={15} /> : <Thermometer size={15} />}
          Validate electrothermal data
        </button>
        <ElectrothermalValidationSummary data={electrothermalData} loading={electrothermalLoading} />
      </div>
    )
  }

  const { summary } = data
  const readiness = [
    ['Capacity & BMS SOH', summary.readiness.capacity_bms],
    ['Monthly exposure', summary.readiness.monthly_exposure],
    ['Voltage-group health', summary.readiness.voltage_group_health],
  ] as const
  const packs = Array.from({ length: summary.module_count * 2 }, (_, index) => ({
    module: Math.floor(index / 2) + 1,
    pack: (index % 2) + 1,
  }))

  return (
    <div className="data-validation">
      <div className="validation-header">
        <div>
          <span className="eyebrow">Scientific data gate</span>
          <h3>Can this device be analyzed?</h3>
          <small>{shortDateTime(summary.observed_start)} to {shortDateTime(summary.observed_end)} UTC | validated {shortDateTime(summary.validated_at)}</small>
        </div>
        <button className="button secondary" type="button" disabled={!canValidate || running} onClick={onValidate}>
          {running ? <LoaderCircle className="spin" size={15} /> : <ShieldCheck size={15} />}
          Validate analysis inputs
        </button>
        <button className="button secondary" type="button" disabled={!canValidateElectrothermal || electrothermalRunning} onClick={onValidateElectrothermal}>
          {electrothermalRunning ? <LoaderCircle className="spin" size={15} /> : <Thermometer size={15} />}
          Validate electrothermal data
        </button>
      </div>

      <div className="validation-readiness">
        {readiness.map(([label, item]) => (
          <div className={`readiness-band readiness-${item.status}`} key={label}>
            <span>{stateIcon(item.status)}{label}</span>
            <strong>{item.status}</strong>
            <small>{item.reason}</small>
          </div>
        ))}
      </div>

      <ElectrothermalValidationSummary data={electrothermalData} loading={electrothermalLoading} />

      <details className="validation-section" open>
        <summary>Coverage timeline</summary>
        <div className="coverage-timeline" style={{ '--month-count': Math.max(1, data.coverage_by_month.length) } as CSSProperties}>
          <span className="timeline-row-label" />
          <div className="timeline-cells timeline-labels">
            {data.coverage_by_month.map((item, index) => <span key={item.month}>{index % 3 === 0 ? item.month.slice(0, 7) : ''}</span>)}
          </div>
          <span className="timeline-row-label">1-second</span>
          <div className="timeline-cells">
            {data.coverage_by_month.map((item) => <i key={item.month} className={coverageClass(item.observed_span_coverage_pct)} title={`${item.month.slice(0, 7)}: ${item.observed_span_coverage_pct.toFixed(1)}% observed-span coverage`} />)}
          </div>
          <span className="timeline-row-label">5-minute</span>
          <div className="timeline-cells">
            {data.coverage_by_month.map((item) => <i key={item.month} className={item.five_minute_available ? 'coverage-good' : 'coverage-missing'} title={`${item.month.slice(0, 7)}: ${item.five_minute_available ? 'available' : 'missing'}`} />)}
          </div>
          <span className="timeline-row-label">Voltage</span>
          <div className="timeline-cells">
            {data.coverage_by_month.map((item) => <i key={item.month} className={item.voltage_groups_available === summary.feature_evidence.expected_voltage_groups ? 'coverage-good' : 'coverage-missing'} title={`${item.month.slice(0, 7)}: ${item.voltage_groups_available}/${summary.feature_evidence.expected_voltage_groups} groups`} />)}
          </div>
          <span className="timeline-row-label">Temperature</span>
          <div className="timeline-cells">
            {data.coverage_by_month.map((item) => <i key={item.month} className={item.temperature_sensors_available === summary.feature_evidence.expected_temperature_sensors ? 'coverage-good' : 'coverage-missing'} title={`${item.month.slice(0, 7)}: ${item.temperature_sensors_available}/${summary.feature_evidence.expected_temperature_sensors} sensors`} />)}
          </div>
        </div>
        <div className="coverage-legend"><span className="coverage-good" />80%+ / available<span className="coverage-warning" />50-80%<span className="coverage-poor" />below 50%<span className="coverage-missing" />missing</div>
      </details>

      <details className="validation-section" open>
        <summary>Signal inventory</summary>
        <div className="validation-table-wrap">
          <table className="validation-table">
            <thead><tr><th>Signal</th><th>Status</th><th>Resolution</th><th>Source</th><th>Meaning</th></tr></thead>
            <tbody>{data.signal_inventory.map((signal) => (
              <tr key={signal.signal_key}>
                <td><strong>{signal.label}</strong><small>{signal.unit}</small></td>
                <td><span className={`validation-state ${signal.available ? 'state-pass' : 'state-missing'}`}>{signal.available ? <Check size={13} /> : <Minus size={13} />}{signal.available ? 'Available' : 'Missing'}</span></td>
                <td>{signal.resolution}</td>
                <td><code>{signal.source_column}</code></td>
                <td>{signal.semantics}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </details>

      <details className="validation-section" open>
        <summary>Hardware topology and channel identity</summary>
        <div className="topology-summary">
          <span><Gauge size={15} />{summary.feature_evidence.available_voltage_groups}/{summary.feature_evidence.expected_voltage_groups} voltage groups</span>
          <span><Thermometer size={15} />{summary.feature_evidence.available_temperature_sensors}/{summary.feature_evidence.expected_temperature_sensors} temperature sensors</span>
        </div>
        <div className="topology-packs">{packs.map(({ module, pack }) => {
          const voltage = data.topology_inventory.filter((item) => item.module_index === module && item.pack_index === pack && item.measurement_type === 'cell_voltage_channel' && item.available).length
          const temperature = data.topology_inventory.filter((item) => item.module_index === module && item.pack_index === pack && item.measurement_type === 'pack_temperature_sensor' && item.available).length
          return <div key={`${module}-${pack}`}><strong>Module {module} / Pack {pack}</strong><span>{voltage}/14 voltage groups</span><span>{temperature}/4 temperature sensors</span></div>
        })}</div>
      </details>

      <details className="validation-section" open>
        <summary>Quality and interpretation</summary>
        <div className="quality-strip">
          <span><small>Calendar coverage</small><strong>{summary.calendar_coverage_pct.toFixed(1)}%</strong></span>
          <span><small>Maximum gap</small><strong>{duration(summary.maximum_gap_seconds)}</strong></span>
          <span><small>Duplicates</small><strong>{compactNumber(summary.duplicate_rows)}</strong></span>
          <span><small>Invalid SOC/SOH</small><strong>{compactNumber(summary.invalid_core_rows)}</strong></span>
          <span><small>Suspicious temperature</small><strong>{summary.suspicious_temperature_windows == null ? 'not evaluated' : compactNumber(summary.suspicious_temperature_windows)}</strong></span>
        </div>
        <div className="validation-interpretation">
          <span className={`validation-state state-${summary.current_sign_verification.status}`}>{stateIcon(summary.current_sign_verification.status)}Current sign: {summary.current_sign_verification.status}</span>
          <p>Positive current = charge and negative current = discharge. Agreement with SOC movement: {summary.current_sign_verification.agreement_pct == null ? 'not enough evidence' : `${summary.current_sign_verification.agreement_pct.toFixed(1)}% across ${compactNumber(summary.current_sign_verification.evidence_windows)} five-minute checks`}.</p>
          <span className="validation-state state-inconclusive"><AlertTriangle size={13} />Resistance reference: disabled / interpretation unverified</span>
          <p>{summary.resistance_reference.note}</p>
        </div>
        <p className="validation-claim-limit">{summary.claim_limit}</p>
      </details>
    </div>
  )
}
