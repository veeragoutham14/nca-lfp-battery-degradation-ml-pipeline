import ReactEChartsCore from 'echarts-for-react/lib/core'
import { echarts } from '../echarts'
import type { NcaCapacityOverviewDevice } from '../types'

interface NcaCapacityGapChartProps {
  devices: NcaCapacityOverviewDevice[]
  tolerancePct: number
}

export function NcaCapacityGapChart({ devices, tolerancePct }: NcaCapacityGapChartProps) {
  function gapGroup(device: NcaCapacityOverviewDevice): 0 | 1 | 2 {
    if (device.bms_minus_reference_soh_pct < -tolerancePct) return 2
    if (device.bms_minus_reference_soh_pct > tolerancePct) return 1
    return 0
  }

  const ordered = [...devices].sort((left, right) => (
    gapGroup(left) - gapGroup(right)
    || (left.module_count ?? Number.POSITIVE_INFINITY) - (right.module_count ?? Number.POSITIVE_INFINITY)
    || left.bms_minus_reference_soh_pct - right.bms_minus_reference_soh_pct
  ))
  if (ordered.length === 0) {
    return <div className="empty-chart">No completed NCA verification runs are available.</div>
  }

  const chartRows: Array<
    | { kind: 'group'; label: string; tone: 'critical' | 'normal' | 'above' }
    | { kind: 'device'; device: NcaCapacityOverviewDevice }
  > = []
  let currentGroup: number | undefined
  const groupDetails = [
    { label: 'Within tolerance', tone: 'normal' as const },
    { label: 'Above reference', tone: 'above' as const },
    { label: 'Below reference', tone: 'critical' as const },
  ] as const
  for (const device of ordered) {
    const nextGroup = gapGroup(device)
    if (currentGroup !== nextGroup) {
      currentGroup = nextGroup
      chartRows.push({ kind: 'group', ...groupDetails[nextGroup] })
    }
    chartRows.push({ kind: 'device', device })
  }

  const option = {
    animationDuration: 300,
    grid: { top: 20, right: 42, bottom: 52, left: 190 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (items: Array<{ dataIndex: number; value: number }>) => {
        const item = items[0]
        const row = chartRows[item.dataIndex]
        if (!row || row.kind !== 'device') return ''
        const device = row.device
        return [
          `<strong>${device.display_name}</strong>`,
          `Serial: ${device.serial}`,
          `Modules: ${device.module_count ?? 'unknown'}`,
          `BMS minus reference: ${item.value >= 0 ? '+' : ''}${item.value.toFixed(2)} pp`,
          `BMS SOH: ${device.latest_bms_soh_pct.toFixed(1)}%`,
          `Reference: ${device.reference_soh_pct.toFixed(2)}%`,
          `Discharge: ${Math.round(device.cumulative_discharge_ah).toLocaleString()} Ah`,
        ].join('<br>')
      },
    },
    xAxis: {
      type: 'value',
      name: 'BMS minus reference SOH (percentage points)',
      nameLocation: 'middle',
      nameGap: 32,
      axisLabel: { color: '#5e6a70', formatter: '{value} pp' },
      splitLine: { lineStyle: { color: '#e5e9eb' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: chartRows.map((row, index) => row.kind === 'group'
        ? `gap-group-${row.tone}-${index}`
        : `${row.device.serial}-${index}`),
      axisLine: { lineStyle: { color: '#9ba8ae' } },
      axisTick: { show: false },
      axisLabel: {
        color: '#44545b',
        fontFamily: 'Cascadia Code, Consolas, monospace',
        formatter: (_value: string, index: number) => {
          const row = chartRows[index]
          if (row.kind === 'group') {
            return `{${row.tone}|${row.label}}`
          }
          return `${row.device.display_name} | ${row.device.module_count ?? '?'} modules`
        },
        rich: {
          critical: {
            color: '#9d3f39',
            fontFamily: 'Inter, Arial, sans-serif',
            fontSize: 11,
            fontWeight: 700,
            padding: [9, 0, 3, 0],
          },
          normal: {
            color: '#476733',
            fontFamily: 'Inter, Arial, sans-serif',
            fontSize: 11,
            fontWeight: 700,
            padding: [9, 0, 3, 0],
          },
          above: {
            color: '#365f8e',
            fontFamily: 'Inter, Arial, sans-serif',
            fontSize: 11,
            fontWeight: 700,
            padding: [9, 0, 3, 0],
          },
        },
      },
    },
    series: [
      {
        name: 'BMS minus reference',
        type: 'bar',
        barMaxWidth: 17,
        data: chartRows.map((row) => {
          if (row.kind === 'group') return { value: null }
          const device = row.device
          return {
            value: device.bms_minus_reference_soh_pct,
            itemStyle: {
              color: device.bms_minus_reference_soh_pct < -tolerancePct
                ? '#bb5148'
                : device.bms_minus_reference_soh_pct > tolerancePct
                  ? '#416fa5'
                  : '#58763e',
            },
          }
        }),
        markLine: {
          silent: true,
          symbol: ['none', 'none'],
          label: { color: '#68767c', fontSize: 10 },
          lineStyle: { color: '#8b989e', type: 'dashed' },
          data: [
            { xAxis: 0, label: { formatter: 'Reference' } },
            { xAxis: -tolerancePct, label: { formatter: `-${tolerancePct} pp` } },
            { xAxis: tolerancePct, label: { formatter: `+${tolerancePct} pp` } },
          ],
        },
      },
    ],
  }

  return (
    <ReactEChartsCore
      echarts={echarts}
      option={option}
      style={{ height: Math.max(430, chartRows.length * 31 + 90), width: '100%' }}
      notMerge
    />
  )
}
