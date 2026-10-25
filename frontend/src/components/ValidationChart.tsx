import ReactEChartsCore from 'echarts-for-react/lib/core'
import { echarts } from '../echarts'
import type { ValidationSeries } from '../types'

export function ValidationChart({ data }: { data: ValidationSeries | null }) {
  if (!data || data.points.length === 0) {
    return <div className="empty-chart">This run has no cutoff validation series.</div>
  }

  const option = {
    animationDuration: 300,
    grid: { top: 58, right: 24, bottom: 64, left: 62 },
    legend: {
      top: 8,
      left: 8,
      data: ['Actual SOH', 'Predicted SOH'],
      textStyle: { color: '#3f4c54' },
    },
    tooltip: {
      trigger: 'axis',
      valueFormatter: (value: unknown) =>
        typeof value === 'number' ? `${value.toFixed(3)}%` : String(value ?? ''),
    },
    xAxis: {
      type: 'time',
      axisLine: { lineStyle: { color: '#9ba8ae' } },
      axisTick: { show: false },
      axisLabel: { color: '#5e6a70', hideOverlap: true },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: 'SOH %',
      scale: true,
      axisLine: { show: true, lineStyle: { color: '#9ba8ae' } },
      axisTick: { show: false },
      axisLabel: { color: '#5e6a70', formatter: '{value}%' },
      splitLine: { lineStyle: { color: '#e5e9eb' } },
    },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 16 }],
    series: [
      {
        name: 'Actual SOH',
        type: 'scatter',
        symbol: 'circle',
        symbolSize: 7,
        itemStyle: { color: '#26343d' },
        data: data.points.map((point) => [point.timestamp, point.actual_soh_pct]),
      },
      {
        name: 'Predicted SOH',
        type: 'line',
        symbol: 'emptyDiamond',
        symbolSize: 7,
        showSymbol: true,
        lineStyle: { color: '#d99018', width: 2 },
        itemStyle: { color: '#d99018' },
        data: data.points.map((point) => [point.timestamp, point.predicted_soh_pct]),
      },
    ],
  }

  return (
    <ReactEChartsCore
      echarts={echarts}
      option={option}
      style={{ height: 420, width: '100%' }}
      notMerge
    />
  )
}
