import {
  Activity,
  ArrowRight,
  BatteryCharging,
  CircleGauge,
  RefreshCw,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { NcaCapacityOverview } from '../types'
import { compactNumber, integer, number, shortDateTime } from '../utils'
import { MetricCard } from './MetricCard'
import { NcaCapacityGapChart } from './NcaCapacityGapChart'

interface NcaCapacityOverviewPageProps {
  onOpenDevice: (serial: string) => void
  onOpenCellSignals: () => void
}

type FleetSort = 'modules' | 'gap' | 'soh' | 'throughput' | 'serial'

export function NcaCapacityOverviewPage({
  onOpenDevice,
  onOpenCellSignals,
}: NcaCapacityOverviewPageProps) {
  const [overview, setOverview] = useState<NcaCapacityOverview | null>(null)
  const [referenceSlope, setReferenceSlope] = useState(0.12)
  const [draftSlope, setDraftSlope] = useState('0.12')
  const [sort, setSort] = useState<FleetSort>('modules')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  function load(slope = referenceSlope) {
    setLoading(true)
    setError('')
    api.ncaCapacityOverview(slope, 1)
      .then(setOverview)
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load(0.12)
  }, [])

  const orderedDevices = useMemo(() => {
    const devices = [...(overview?.devices ?? [])]
    if (sort === 'modules') return devices.sort((a, b) => (a.module_count ?? Number.POSITIVE_INFINITY) - (b.module_count ?? Number.POSITIVE_INFINITY) || a.serial.localeCompare(b.serial))
    if (sort === 'soh') return devices.sort((a, b) => a.latest_bms_soh_pct - b.latest_bms_soh_pct)
    if (sort === 'throughput') return devices.sort((a, b) => b.cumulative_discharge_ah - a.cumulative_discharge_ah)
    if (sort === 'serial') return devices.sort((a, b) => a.serial.localeCompare(b.serial))
    return devices.sort((a, b) => a.bms_minus_reference_soh_pct - b.bms_minus_reference_soh_pct)
  }, [overview, sort])

  function applySlope() {
    const value = Number(draftSlope)
    if (!Number.isFinite(value) || value <= 0) {
      setError('Reference slope must be a positive number.')
      return
    }
    setReferenceSlope(value)
    load(value)
  }

  if (loading && !overview) {
    return <div className="section-loader full-page-loader">Building NCA fleet overview...</div>
  }

  const metrics = overview?.metrics
  const reference = overview?.reference

  return (
    <div className="page-stack fleet-capacity-page">
      {error && <div className="error-banner">{error}</div>}
      <header className="page-header fleet-capacity-header">
        <div>
          <span className="eyebrow">NCA field cohorts by module count</span>
          <h1>Capacity comparison</h1>
        </div>
        <div className="fleet-controls">
          <label className="field compact-field">
            Reference slope
            <span className="inline-number-control">
              <input
                type="number"
                min="0.0001"
                step="0.0001"
                value={draftSlope}
                onChange={(event) => setDraftSlope(event.target.value)}
              />
              <span>Ah / 1k Ah</span>
            </span>
          </label>
          <button className="button secondary" type="button" onClick={applySlope} disabled={loading}>
            Apply
          </button>
          <button className="button secondary" type="button" onClick={onOpenCellSignals}>
            Reference cell features
          </button>
          <button className="icon-button" type="button" onClick={() => load()} disabled={loading} title="Refresh NCA overview">
            <RefreshCw size={17} className={loading ? 'spin' : ''} aria-hidden="true" />
          </button>
        </div>
      </header>

      {metrics && reference && (
        <>
          <section className="metric-strip fleet-metric-strip">
            <MetricCard
              label="Active field devices"
              value={`${metrics.available_device_count} / ${metrics.configured_device_count}`}
              detail={`${metrics.inventory_device_count} NCA field devices retained; cross-module rows are exploratory`}
              icon={BatteryCharging}
              tone={metrics.missing_device_count ? 'warning' : 'good'}
            />
            <MetricCard
              label="Below NCA reference"
              value={integer(metrics.devices_below_reference_by_more_than_tolerance)}
              detail="More than 1 percentage point"
              icon={CircleGauge}
              tone={metrics.devices_below_reference_by_more_than_tolerance ? 'warning' : 'good'}
            />
          </section>

          <dl className="fleet-reference-strip">
            <div><dt>Reference device</dt><dd>{reference.display_name}</dd><span>{reference.serial} / {reference.module_count} modules / full-cycle control</span></div>
            <div><dt>Transferred slope</dt><dd>{number(reference.loss_slope_ah_per_1000ah, 4)}</dd><span>Ah / 1,000 pack Ah</span></div>
            <div><dt>Median BMS-equivalent slope</dt><dd>{number(metrics.median_bms_equivalent_loss_slope_ah_per_1000ah, 4)}</dd><span>Ah / 1,000 pack Ah</span></div>
            <div><dt>Above 2x reference</dt><dd>{integer(metrics.devices_above_twice_reference_loss_rate)}</dd><span>Net BMS-implied rate</span></div>
            <div><dt>Module count verified</dt><dd>{metrics.devices_with_verified_module_count} / {metrics.available_device_count}</dd><span>Latest metadata import</span></div>
          </dl>

          <div className="inline-notice evidence-notice">
            <Activity size={15} aria-hidden="true" />
            <span>{overview.evidence_policy.claim_limit}</span>
          </div>

          <section className="panel chart-panel fleet-gap-panel">
            <div className="section-title">
              <div><h2>BMS difference from transferred NCA reference</h2><span>Negative values mean the BMS reports lower SOH than the reference trajectory</span></div>
              <small className="source-freshness">Source {shortDateTime(overview.latest_source_modified_at)}</small>
            </div>
            <NcaCapacityGapChart devices={overview.devices} tolerancePct={1} />
          </section>

          <section className="panel fleet-device-table">
            <div className="section-title">
              <div><h2>NCA field comparison</h2><span>{overview.devices.length} completed runs; two-module rows are matched and other module counts are EFC-normalized experiments</span></div>
              <label className="field sort-field">
                Sort
                <select value={sort} onChange={(event) => setSort(event.target.value as FleetSort)}>
                  <option value="modules">Module count</option>
                  <option value="gap">Reference difference</option>
                  <option value="soh">Latest BMS SOH</option>
                  <option value="throughput">Discharge throughput</option>
                  <option value="serial">Serial number</option>
                </select>
              </label>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Device</th><th>Modules</th><th>Scope</th><th>Nominal</th><th>Observed</th><th>Discharge</th><th>EFC</th><th>BMS SOH</th><th>Reference</th><th>BMS - ref.</th><th>BMS-equivalent loss</th><th>Coverage</th><th /></tr>
                </thead>
                <tbody>
                  {orderedDevices.map((device) => (
                    <tr key={device.serial}>
                      <td><span className="serial-cell"><BatteryCharging size={14} aria-hidden="true" /><span className="device-identity"><strong>{device.display_name}</strong><small>{device.serial}</small></span></span></td>
                      <td>{device.module_count ?? 'n/a'}</td>
                      <td>{device.comparison_scope === 'matched_reference_configuration' ? 'Matched' : 'Exploratory'}</td>
                      <td>{number(device.nominal_capacity_ah, 1)} Ah</td>
                      <td>{shortDateTime(device.observed_end)}</td>
                      <td className="numeric">{compactNumber(device.cumulative_discharge_ah)} Ah</td>
                      <td className="numeric">{number(device.equivalent_full_cycles, 0)}</td>
                      <td className="numeric">{number(device.latest_bms_soh_pct, 1)}%</td>
                      <td className="numeric">{number(device.reference_soh_pct, 2)}%</td>
                      <td className={`numeric ${device.bms_minus_reference_soh_pct < -1 ? 'gap-critical' : 'gap-normal'}`}>
                        {device.bms_minus_reference_soh_pct >= 0 ? '+' : ''}{number(device.bms_minus_reference_soh_pct, 2)} pp
                      </td>
                      <td className="numeric">{number(device.bms_equivalent_loss_slope_ah_per_1000ah, 4)}</td>
                      <td className="numeric">{number(device.telemetry_coverage_pct, 1)}%</td>
                      <td className="table-action">
                        <button className="icon-button table-open-button" type="button" onClick={() => onOpenDevice(device.serial)} title={`Open battery ${device.serial}`}>
                          <ArrowRight size={15} aria-hidden="true" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {overview.deferred_devices.length > 0 && (
            <section className="panel fleet-device-table deferred-cohort-panel">
              <div className="section-title">
                <div>
                  <h2>Unverified NCA devices</h2>
                  <span>Excluded until module count and nominal capacity are reliable</span>
                </div>
                <small className="source-freshness">{overview.deferred_devices.length} devices retained</small>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Device</th><th>Modules</th><th>Module status</th><th>Nominal</th><th>Energy</th><th>Cohort decision</th><th /></tr>
                  </thead>
                  <tbody>
                    {overview.deferred_devices.map((device) => (
                      <tr key={device.serial} className="cohort-deferred-row">
                        <td><span className="serial-cell"><BatteryCharging size={14} aria-hidden="true" /><span className="device-identity"><strong>{device.display_name}</strong><small>{device.serial}</small></span></span></td>
                        <td>{device.module_count ?? 'n/a'}</td>
                        <td>{device.module_count_status}</td>
                        <td>{device.nominal_capacity_ah == null ? 'n/a' : `${number(device.nominal_capacity_ah, 1)} Ah`}</td>
                        <td>{device.system_energy_kwh == null ? 'n/a' : `${number(device.system_energy_kwh, 0)} kWh`}</td>
                        <td>{device.cohort_reason}</td>
                        <td className="table-action">
                          <button className="icon-button table-open-button" type="button" onClick={() => onOpenDevice(device.serial)} title={`Open battery ${device.serial}`}>
                            <ArrowRight size={15} aria-hidden="true" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {overview.missing_devices.length > 0 && (
            <section className="panel missing-device-panel">
              <div className="section-title"><div><h2>Pending devices</h2><span>Verification output not available yet</span></div></div>
              <div className="missing-device-list">
                {overview.missing_devices.map((device) => <span key={device.serial}><strong>{device.display_name}</strong><small>{device.serial}</small>{device.reason}</span>)}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  )
}
