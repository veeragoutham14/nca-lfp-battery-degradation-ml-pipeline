import { BarChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import { memo } from 'react'
import type { BmsSohEvidenceComparison } from '../types'

echarts.use([BarChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

interface BmsSohEvidenceComparisonChartProps {
  rows: BmsSohEvidenceComparison[]
}

function percent(value: number | null) {
  return value == null ? 'n/a' : `${(100 * value).toFixed(1)}%`
}

export const BmsSohEvidenceComparisonChart = memo(function BmsSohEvidenceComparisonChart({
  rows,
}: BmsSohEvidenceComparisonChartProps) {
  const ranked = [...rows]
    .filter((row) => row.event_total > 0 && row.control_total > 0)
    .sort((left, right) => (right.risk_difference ?? -Infinity) - (left.risk_difference ?? -Infinity))
    .slice(0, 9)
  const byLabel = new Map(ranked.map((row) => [row.feature_label, row]))
  const option = {
    animation: false,
    grid: { top: 38, right: 38, bottom: 46, left: 245 },
    legend: { top: 0, textStyle: { color: '#59676d', fontSize: 10 } },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (items: Array<{ axisValue: string }>) => {
        const row = byLabel.get(items[0]?.axisValue)
        if (!row) return ''
        return [
          `<strong>${row.feature_label}</strong>`,
          `Events: ${row.event_count}/${row.event_total} (${percent(row.event_rate)})`,
          `Matched controls: ${row.control_count}/${row.control_total} (${percent(row.control_rate)})`,
          `Difference: ${row.risk_difference == null ? 'n/a' : `${(100 * row.risk_difference).toFixed(1)} pp`}`,
          `Prevalence lift: ${row.prevalence_lift == null ? 'n/a' : `${row.prevalence_lift.toFixed(2)}x`}`,
          'Intervals are 95% Wilson intervals for each prevalence.',
        ].join('<br>')
      },
    },
    xAxis: {
      type: 'value',
      min: 0,
      max: 100,
      name: 'Prevalence (%)',
      axisLabel: { color: '#5e6a70', formatter: '{value}%' },
      splitLine: { lineStyle: { color: '#e5e9eb' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: ranked.map((row) => row.feature_label),
      axisLabel: { color: '#46545a', width: 218, overflow: 'truncate', fontSize: 10 },
      axisLine: { lineStyle: { color: '#9ba8ae' } },
    },
    series: [
      {
        name: 'SOH event windows',
        type: 'bar',
        barMaxWidth: 15,
        itemStyle: { color: '#b34f49' },
        data: ranked.map((row) => row.event_rate == null ? null : 100 * row.event_rate),
      },
      {
        name: 'Matched non-event windows',
        type: 'bar',
        barMaxWidth: 15,
        itemStyle: { color: '#39768b' },
        data: ranked.map((row) => row.control_rate == null ? null : 100 * row.control_rate),
      },
    ],
  }

  return (
    <ReactEChartsCore
      echarts={echarts}
      option={option}
      style={{ height: Math.max(330, 65 + ranked.length * 43), width: '100%' }}
      opts={{ renderer: 'canvas' }}
      notMerge
    />
  )
})
