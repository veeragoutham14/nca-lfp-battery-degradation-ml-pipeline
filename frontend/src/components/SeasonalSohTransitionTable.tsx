import type { SohTransitionContextPoint } from '../types'
import { number, shortDateTime } from '../utils'

function changeLabel(row: SohTransitionContextPoint) {
  const previous = number(row.previous_soh_pct, 1)
  const next = number(row.new_soh_pct, 1)
  const delta = row.soh_delta_pp == null ? 'n/a' : `${row.soh_delta_pp > 0 ? '+' : ''}${number(row.soh_delta_pp, 1)} pp`
  return `${previous}% -> ${next}% (${delta})`
}

function pair(before: number | null, after: number | null, unit: string) {
  return `${number(before, 1)} -> ${number(after, 1)} ${unit}`
}

function chargeContextLabel(row: SohTransitionContextPoint) {
  if (row.event_during_charge_episode) return 'During charging'
  if (row.near_full_charge_within_72h_before) {
    return `${number(row.hours_since_near_full_charge, 1)} h after near-full charge`
  }
  return 'No near-full charge within 72 h'
}

function dischargeContextLabel(row: SohTransitionContextPoint) {
  if (!row.authoritative_recalibration_match) return 'No v4 event match'
  if (!row.recalibration_complete_discharge_within_lookback) return 'No qualified high-to-low discharge'
  const start = number(row.recalibration_raw_start_soc_pct, 1)
  const end = number(row.recalibration_raw_end_soc_pct, 1)
  const delay = number(row.recalibration_hours_since_complete_discharge, 1)
  return `${start}% -> ${end}%, ${delay} h before update`
}

export function SeasonalSohTransitionTable({
  rows,
}: {
  rows: SohTransitionContextPoint[]
}) {
  if (rows.length === 0) {
    return <div className="empty-state">No audited BMS SOH transition was observed.</div>
  }

  return (
    <div className="table-wrap seasonal-transition-table">
      <table>
        <thead>
          <tr>
            <th>Transition time UTC</th>
            <th>BMS SOH</th>
            <th>Discharge trigger</th>
            <th>Raw cycle evidence</th>
            <th>Capacity check</th>
            <th>Charging context</th>
            <th>Discharge Ah, 7 d pre/post</th>
            <th>Max temp, 7 d pre/post</th>
            <th>Spread p95, 7 d pre/post</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.transition_id}>
              <td data-label="Transition time UTC">{shortDateTime(row.event_time_utc)}</td>
              <td data-label="BMS SOH"><span className={`transition-change ${row.soh_delta_pp != null && row.soh_delta_pp < 0 ? 'drop' : 'recovery'}`}>{changeLabel(row)}</span></td>
              <td data-label="Discharge trigger">{dischargeContextLabel(row)}</td>
              <td data-label="Raw cycle evidence">
                {row.authoritative_recalibration_match ? <>
                  <strong>{number(row.recalibration_discharged_ah_1s, 2)} Ah</strong>
                  <small>{number((row.recalibration_raw_coverage_fraction ?? 0) * 100, 1)}% coverage; {number(row.recalibration_rest_minutes_inside_cycle, 0)} min rest; {number(row.recalibration_charge_intrusion_ah_1s, 2)} Ah charge intrusion</small>
                </> : <strong>Rerun updated analysis</strong>}
              </td>
              <td data-label="Capacity check">
                {row.authoritative_recalibration_match ? <>
                  <strong>{number(row.recalibration_soc_normalized_capacity_soh_pct, 1)}% normalized SOH</strong>
                  <small>{row.recalibration_capacity_validation_status ?? 'not available'}; BMS error {number(row.recalibration_bms_after_event_minus_capacity_soh_pp, 1)} pp</small>
                </> : <span>Not present in this legacy result</span>}
              </td>
              <td data-label="Charging context">{chargeContextLabel(row)}</td>
              <td data-label="Discharge Ah, 7 d pre/post">{pair(row.pre_discharge_ah, row.post_discharge_ah, 'Ah')}</td>
              <td data-label="Max temp, 7 d pre/post">{pair(row.pre_temperature_max_c, row.post_temperature_max_c, 'C')}</td>
              <td data-label="Spread p95, 7 d pre/post">{pair(row.pre_voltage_spread_p95_mv, row.post_voltage_spread_p95_mv, 'mV')}</td>
              <td data-label="Evidence"><span className={`evidence-status ${row.authoritative_recalibration_match && row.context_quality_ready ? 'ready' : 'incomplete'}`}>{!row.authoritative_recalibration_match ? 'Legacy result' : row.context_quality_ready ? 'Complete' : 'Insufficient coverage'}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
