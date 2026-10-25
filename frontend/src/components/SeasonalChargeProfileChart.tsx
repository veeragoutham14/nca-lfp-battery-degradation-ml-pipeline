import ReactEChartsCore from 'echarts-for-react/lib/core'
import { useMemo, useState } from 'react'
import { echarts } from '../echarts'
import type { SeasonalProfilePoint } from '../types'

type ChartMode = 'usage' | 'current' | 'joint' | 'soc' | 'fullSoc' | 'thermal' | 'voltage'

const MODES: Array<{ key: ChartMode; label: string }> = [
  { key: 'usage', label: 'Charge and discharge' },
  { key: 'current', label: 'Current dynamics' },
  { key: 'joint', label: 'Combined exposures' },
  { key: 'soc', label: 'SOC exposure' },
  { key: 'fullSoc', label: 'Full SOC' },
  { key: 'thermal', label: 'Temperature' },
  { key: 'voltage', label: 'Voltage spread' },
]

const FULL_SOC_BANDS = [
  { key: 'soc_0_5_hours', label: '0-5%', color: '#6c7e91' },
  { key: 'soc_5_20_hours', label: '5-20%', color: '#4f73a5' },
  { key: 'soc_20_40_hours', label: '20-40%', color: '#26868a' },
  { key: 'soc_40_60_hours', label: '40-60%', color: '#4f8755' },
  { key: 'soc_60_80_hours', label: '60-80%', color: '#99a442' },
  { key: 'soc_80_95_hours', label: '80-95%', color: '#d18a24' },
  { key: 'soc_95_100_hours', label: '95-100%', color: '#b64f58' },
  { key: 'soc_unknown_hours', label: 'Unknown', color: '#c9d0d3' },
] as const

function finite(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function fixed(value: number | null | undefined, digits = 1) {
  return finite(value)?.toFixed(digits) ?? 'n/a'
}

function perObservedDay(value: number | null, point: SeasonalProfilePoint) {
  const days = finite(point.observed_days)
  return days && days > 0 && finite(value) != null ? Number(value) / days : null
}

function perExpectedDay(value: number | null | undefined, point: SeasonalProfilePoint) {
  const days = finite(point.expected_days_in_observed_scope)
  return days && days > 0 && finite(value) != null ? Number(value) / days : null
}

function monthLabel(timestamp: string) {
  return timestamp.slice(0, 7)
}

interface SeasonalChargeProfileChartProps {
  points: SeasonalProfilePoint[]
  socExposureMethod?: string
}

export function SeasonalChargeProfileChart({
  points,
  socExposureMethod,
}: SeasonalChargeProfileChartProps) {
  const [mode, setMode] = useState<ChartMode>('usage')
  const fullSocAvailable = points.some((point) =>
    FULL_SOC_BANDS.some((band) => finite(point[band.key]) != null),
  )
  const staleBackendSchema = Boolean(socExposureMethod) && !fullSocAvailable
  const currentDynamicsAvailable = points.some((point) => finite(point.raw_rms_current_a) != null)
  const combinedExposureObserved = points.some((point) => [
    point.charging_above_95_soc_hours,
    point.high_discharge_low_soc_hours,
    point.high_charge_high_soc_hours,
    point.high_current_hot_hours,
    point.high_current_imbalanced_hours,
  ].some((value) => (finite(value) ?? 0) > 0))
  const option = useMemo(() => {
    const validValue = (
      point: SeasonalProfilePoint,
      value: number | null | undefined,
    ) => point.seasonal_profile_valid ? finite(value) : null
    const modeSeries = mode === 'usage'
      ? [
          {
            name: 'Charge Ah/day',
            type: 'bar',
            yAxisIndex: 0,
            data: points.map((point) => validValue(point, point.charge_ah_per_observed_day)),
            itemStyle: { color: '#167c80' },
          },
          {
            name: 'Discharge Ah/day',
            type: 'bar',
            yAxisIndex: 0,
            data: points.map((point) => validValue(point, point.discharge_ah_per_observed_day)),
            itemStyle: { color: '#4f73a5' },
          },
        ]
      : mode === 'current'
        ? [
            { name: 'RMS current', type: 'line', yAxisIndex: 0, data: points.map((point) => validValue(point, point.raw_rms_current_a)), itemStyle: { color: '#167c80' }, lineStyle: { color: '#167c80', width: 2 } },
            { name: 'Peak 60 s mean |I|', type: 'line', yAxisIndex: 0, data: points.map((point) => validValue(point, point.raw_peak_60s_mean_absolute_current_a)), itemStyle: { color: '#d18a24' }, lineStyle: { color: '#d18a24', width: 2 } },
            { name: 'Peak discharge', type: 'line', yAxisIndex: 0, data: points.map((point) => validValue(point, point.raw_peak_discharge_current_a)), itemStyle: { color: '#b64f58' }, lineStyle: { color: '#b64f58', width: 2 } },
          ]
      : mode === 'joint'
        ? [
            { name: 'Charging at SOC >=95%', type: 'bar', yAxisIndex: 0, data: points.map((point) => validValue(point, perObservedDay(point.charging_above_95_soc_hours, point))), itemStyle: { color: '#4f73a5' } },
            { name: 'High discharge + SOC <10%', type: 'bar', yAxisIndex: 0, data: points.map((point) => validValue(point, perObservedDay(point.high_discharge_low_soc_hours, point))), itemStyle: { color: '#b64f58' } },
            { name: 'High charge + SOC >90%', type: 'bar', yAxisIndex: 0, data: points.map((point) => validValue(point, perObservedDay(point.high_charge_high_soc_hours, point))), itemStyle: { color: '#d18a24' } },
            { name: 'High current + T >=30 C', type: 'bar', yAxisIndex: 0, data: points.map((point) => validValue(point, perObservedDay(point.high_current_hot_hours, point))), itemStyle: { color: '#2f7d56' } },
            { name: 'High current + spread >=30 mV', type: 'bar', yAxisIndex: 0, data: points.map((point) => validValue(point, perObservedDay(point.high_current_imbalanced_hours, point))), itemStyle: { color: '#765a9f' } },
          ]
      : mode === 'soc'
        ? [
            {
              name: 'SOC >=95 h/day',
              type: 'bar',
              yAxisIndex: 0,
              data: points.map((point) => validValue(
                point,
                perObservedDay(point.high_soc_95_hours, point),
              )),
              itemStyle: { color: '#d18a24' },
            },
            {
              name: 'SOC <=5 h/day',
              type: 'bar',
              yAxisIndex: 0,
              data: points.map((point) => validValue(
                point,
                perObservedDay(point.low_soc_5_hours, point),
              )),
              itemStyle: { color: '#6c7e91' },
            },
          ]
        : mode === 'fullSoc'
          ? FULL_SOC_BANDS.map((band) => ({
              name: band.label,
              type: 'bar',
              stack: 'full-soc-day',
              yAxisIndex: 0,
              barMaxWidth: 44,
              data: points.map((point) => validValue(
                point,
                perExpectedDay(point[band.key], point),
              )),
              itemStyle: { color: band.color },
            }))
        : mode === 'thermal'
          ? [
              {
                name: 'Mean temperature',
                type: 'line',
                yAxisIndex: 0,
                data: points.map((point) => validValue(point, point.temperature_mean_c)),
                symbolSize: 6,
                lineStyle: { width: 2, color: '#2f7d56' },
                itemStyle: { color: '#2f7d56' },
              },
              {
                name: 'Maximum temperature',
                type: 'line',
                yAxisIndex: 0,
                data: points.map((point) => validValue(point, point.temperature_max_c)),
                symbolSize: 6,
                lineStyle: { width: 2, color: '#c05a3d' },
                itemStyle: { color: '#c05a3d' },
              },
            ]
          : [
              {
                name: 'Daily spread p95',
                type: 'line',
                yAxisIndex: 0,
                data: points.map((point) => validValue(
                  point,
                  point.voltage_spread_p95_daily_mv,
                )),
                symbolSize: 6,
                lineStyle: { width: 2, color: '#765a9f' },
                itemStyle: { color: '#765a9f' },
              },
              {
                name: 'Maximum spread',
                type: 'line',
                yAxisIndex: 0,
                data: points.map((point) => validValue(point, point.voltage_spread_max_mv)),
                symbolSize: 6,
                lineStyle: { width: 2, color: '#b64f58' },
                itemStyle: { color: '#b64f58' },
              },
            ]
    const leftUnit = mode === 'usage'
      ? 'Ah/day'
      : mode === 'current'
        ? 'A'
      : mode === 'joint'
        ? 'hours/day'
      : mode === 'soc'
        ? 'hours/day'
        : mode === 'fullSoc'
          ? 'hours/day'
        : mode === 'thermal'
          ? 'C'
          : 'mV'

    return {
      animationDuration: 250,
      grid: { top: 54, right: 70, bottom: 82, left: 68 },
      legend: {
        type: 'scroll',
        top: 8,
        left: 8,
        right: 8,
        textStyle: { color: '#536168', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (items: Array<{ dataIndex: number }>) => {
          const point = points[items[0]?.dataIndex]
          if (!point) return ''
          const quality = point.seasonal_profile_valid
            ? 'Qualified month'
            : 'Excluded by coverage gate'
          const modeLines = mode === 'usage'
            ? [
                `Charged: ${fixed(point.charge_ah, 0)} Ah total / ${fixed(point.charge_ah_per_observed_day)} Ah/day`,
                `Discharged: ${fixed(point.discharge_ah, 0)} Ah total / ${fixed(point.discharge_ah_per_observed_day)} Ah/day`,
                `Charge episodes: ${fixed(point.charge_episode_count, 0)}`,
              ]
            : mode === 'current'
              ? [
                  `RMS current: ${fixed(point.raw_rms_current_a)} A`,
                  `Peak 10 s / 60 s mean |I|: ${fixed(point.raw_peak_10s_mean_absolute_current_a)} / ${fixed(point.raw_peak_60s_mean_absolute_current_a)} A`,
                  `Peak charge / discharge: ${fixed(point.raw_peak_charge_current_a)} / ${fixed(point.raw_peak_discharge_current_a)} A`,
                  `Reversals / large steps: ${fixed(point.raw_current_reversal_count, 0)} / ${fixed(point.raw_large_current_step_count, 0)}`,
                  `High-current episodes: ${fixed(point.raw_high_current_episode_count, 0)}`,
                ]
            : mode === 'joint'
              ? [
                  `Charging at SOC >=95%: ${fixed(point.charging_above_95_soc_hours)} h`,
                  `High discharge + SOC <10%: ${fixed(point.high_discharge_low_soc_hours)} h`,
                  `High charge + SOC >90%: ${fixed(point.high_charge_high_soc_hours)} h`,
                  `High current + T >=30 C: ${fixed(point.high_current_hot_hours)} h`,
                  `High current + spread >=30 mV: ${fixed(point.high_current_imbalanced_hours)} h`,
                  `Low SOC + high discharge + imbalance: ${fixed(point.low_soc_high_discharge_imbalanced_hours)} h`,
                ]
            : mode === 'soc'
              ? [
                  `SOC >=95%: ${fixed(point.high_soc_95_hours)} h`,
                  `SOC <=5%: ${fixed(point.low_soc_5_hours)} h`,
                  `Near-full charges: ${fixed(point.near_full_charge_episode_count, 0)}`,
                ]
              : mode === 'fullSoc'
                ? [
                    ...FULL_SOC_BANDS.map((band) =>
                      `${band.label}: ${fixed(perExpectedDay(point[band.key], point))} h/day`,
                    ),
                    `Valid SOC coverage: ${fixed((point.soc_known_coverage_fraction ?? 0) * 100)}%`,
                  ]
              : mode === 'thermal'
                ? [
                    `Mean: ${fixed(point.temperature_mean_c)} C`,
                    `Maximum: ${fixed(point.temperature_max_c)} C`,
                    `Time >=35 C: ${fixed(point.hot_35_hours)} h`,
                  ]
                : [
                    `Daily p95: ${fixed(point.voltage_spread_p95_daily_mv)} mV`,
                    `Maximum: ${fixed(point.voltage_spread_max_mv)} mV`,
                  ]
          return [
            `<strong>${monthLabel(point.month_start_utc)} · ${point.season ?? 'unknown season'}</strong>`,
            quality,
            `Coverage: ${fixed((point.telemetry_coverage_fraction ?? 0) * 100)}%`,
            ...modeLines,
            `BMS SOH: ${fixed(point.bms_soh_last_pct)}%`,
            `Audited SOH change: ${fixed(point.audited_soh_change_pp)} pp`,
          ].join('<br>')
        },
      },
      dataZoom: [
        { type: 'inside', filterMode: 'none', zoomOnMouseWheel: false, moveOnMouseMove: true },
        { type: 'slider', height: 20, bottom: 22, filterMode: 'none' },
      ],
      xAxis: {
        type: 'category',
        data: points.map((point) => monthLabel(point.month_start_utc)),
        axisLine: { lineStyle: { color: '#9ba8ae' } },
        axisLabel: { color: '#5e6a70', hideOverlap: true },
      },
      yAxis: [
        {
          type: 'value',
          axisLabel: { color: '#5e6a70', formatter: `{value} ${leftUnit}` },
          splitLine: { lineStyle: { color: '#e5e9eb' } },
          min: 0,
          max: mode === 'fullSoc' ? 24 : undefined,
        },
        {
          type: 'value',
          axisLabel: { color: '#5e6a70', formatter: '{value}%' },
          splitLine: { show: false },
          min: (value: { min: number }) => Math.max(0, Math.floor(value.min - 2)),
          max: (value: { max: number }) => Math.min(105, Math.ceil(value.max + 2)),
        },
      ],
      series: [
        ...modeSeries,
        {
          name: 'BMS SOH',
          type: 'line',
          yAxisIndex: 1,
          step: 'end',
          data: points.map((point) => finite(point.bms_soh_last_pct)),
          symbolSize: 6,
          lineStyle: { width: 2, color: '#343f45' },
          itemStyle: { color: '#343f45' },
          connectNulls: false,
        },
      ],
    }
  }, [mode, points])

  if (points.length === 0) {
    return <div className="empty-chart">No monthly operating profile is available.</div>
  }

  return (
    <div className="seasonal-chart-stack">
      <div className="seasonal-mode-control" role="tablist" aria-label="Monthly profile metric">
        {MODES.map((item) => (
          <button
            className={mode === item.key ? 'active' : ''}
            key={item.key}
            type="button"
            role="tab"
            aria-selected={mode === item.key}
            onClick={() => setMode(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <ReactEChartsCore
        echarts={echarts}
        option={option}
        style={{ height: 440, width: '100%' }}
        notMerge
      />
      {mode === 'fullSoc' && (
        <div className="inline-notice full-soc-method">
          <strong>
            {fullSocAvailable
              ? 'Estimated SOC exposure'
              : staleBackendSchema
                ? 'Backend restart required'
                : 'Full SOC unavailable'}
          </strong>
          <span>
            {fullSocAvailable
              ? 'Duration-weighted five-minute mean BMS SOC. Each qualified bar totals 24 h per scoped day; Unknown includes missing telemetry, missing or invalid SOC, and entirely missing days. Treat the bands as approximate BMS-reported exposure, not exact cell-level time.'
              : staleBackendSchema
                ? 'This report contains Full SOC evidence, but the running API is using an older response schema. Restart the BatteryML backend; rerunning the analysis is not required.'
                : 'This legacy analysis does not contain the mutually exclusive SOC bands. Run Analyze charging and seasonal stress again to generate the Full SOC view.'}
          </span>
        </div>
      )}
      {mode === 'current' && !currentDynamicsAvailable && (
        <div className="inline-notice full-soc-method">
          <strong>Current dynamics unavailable</strong>
          <span>This result predates the UTC-aligned one-second current profile or was built before the corrected monthly join. Run the updated analysis once.</span>
        </div>
      )}
      {mode === 'joint' && !combinedExposureObserved && (
        <div className="inline-notice full-soc-method">
          <strong>No configured exposure crossings</strong>
          <span>The available windows did not cross the displayed combinations. The high-current conditions use 0.5C, which equals 32 A for a 64 Ah system; zero is a valid result, not missing data.</span>
        </div>
      )}
    </div>
  )
}
