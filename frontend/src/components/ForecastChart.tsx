import ReactEChartsCore from 'echarts-for-react/lib/core'
import { echarts } from '../echarts'
import type { RunSeries } from '../types'

const colors = ['#176b87', '#d99018', '#c45147', '#4776b8', '#6b7d3e']
const lineTypes = ['solid', 'dashed', 'dotted'] as const

interface ForecastChartProps {
  data: RunSeries | null
}

export function ForecastChart({ data }: ForecastChartProps) {
  if (!data || (data.history.length === 0 && data.forecasts.length === 0)) {
    return <div className="empty-chart">No SOH series is available for this run.</div>
  }

  const chartSeries: Record<string, unknown>[] = []
  const fixedSlopeVerification = data.model_family === 'fixed_slope_transfer_verification'
  const sohSignalAudit = data.model_family === 'bms_soh_signal_audit'
  const historyLabel = fixedSlopeVerification || sohSignalAudit
    ? 'BMS SOH observations'
    : 'Measured capacity SOH'
  const primary = data.forecasts[0]
  const hasConfidenceInterval = Boolean(primary?.points.some((point) => point.lower_95_soh_pct != null && point.upper_95_soh_pct != null))
  if (hasConfidenceInterval) {
    chartSeries.push(
      {
        name: 'Confidence baseline',
        type: 'line',
        stack: 'confidence',
        symbol: 'none',
        silent: true,
        lineStyle: { opacity: 0 },
        areaStyle: { opacity: 0 },
        data: primary.points.map((point) => [point.timestamp, point.lower_95_soh_pct]),
      },
      {
        name: 'Model 95% interval',
        type: 'line',
        stack: 'confidence',
        symbol: 'none',
        silent: true,
        lineStyle: { opacity: 0 },
        areaStyle: { color: '#176b87', opacity: 0.12 },
        data: primary.points.map((point) => [
          point.timestamp,
          point.upper_95_soh_pct != null && point.lower_95_soh_pct != null
            ? point.upper_95_soh_pct - point.lower_95_soh_pct
            : null,
        ]),
      },
    )
  }

  chartSeries.push({
    name: historyLabel,
    type: sohSignalAudit ? 'line' : 'scatter',
    symbolSize: sohSignalAudit ? 3 : 6,
    showSymbol: !sohSignalAudit,
    lineStyle: sohSignalAudit ? { color: '#176b87', width: 2 } : undefined,
    itemStyle: { color: '#26343d', opacity: 0.62 },
    emphasis: { itemStyle: { opacity: 1 } },
    data: data.history.map((point) => [point.timestamp, point.soh_pct]),
    markLine: {
      silent: true,
      symbol: ['none', 'none'],
      label: { formatter: '80% threshold', color: '#52616b', position: 'insideEndTop' },
      lineStyle: { color: '#52616b', type: 'dashed', width: 1.5 },
      data: [{ yAxis: 80 }],
    },
  })

  const bmsPoints = data.history.filter((point) => point.bms_soh_pct != null)
  if (!fixedSlopeVerification && !sohSignalAudit && bmsPoints.length >= 2) {
    chartSeries.push({
      name: 'BMS SOH',
      type: 'line',
      symbol: 'none',
      lineStyle: { color: '#7d8a91', width: 1.5, type: 'dashed' },
      data: bmsPoints.map((point) => [point.timestamp, point.bms_soh_pct]),
    })
  }

  data.forecasts.forEach((scenario, index) => {
    const label = fixedSlopeVerification && scenario.scenario.startsWith('slope_')
      ? `Slope ${scenario.scenario.slice('slope_'.length)} Ah / 1,000 Ah`
      : scenario.scenario.replaceAll('_', ' ')
    chartSeries.push({
      name: label,
      type: 'line',
      showSymbol: false,
      symbol: ['circle', 'diamond', 'triangle'][index % 3],
      lineStyle: {
        color: colors[index % colors.length],
        width: index === 0 ? 2.5 : 2,
        type: lineTypes[index % lineTypes.length],
      },
      itemStyle: { color: colors[index % colors.length] },
      data: scenario.points.map((point) => [point.timestamp, point.soh_pct]),
    })
  })

  const option = {
    animationDuration: 350,
    color: colors,
    grid: { top: 62, right: 28, bottom: 74, left: 64, containLabel: false },
    legend: {
      type: 'scroll',
      top: 8,
      left: 8,
      right: 8,
      itemWidth: 26,
      pageIconSize: 10,
      pageTextStyle: { color: '#5e6a70', fontSize: 11 },
      textStyle: { color: '#3f4c54', fontSize: 12 },
      data: [
        historyLabel,
        ...(!fixedSlopeVerification && !sohSignalAudit && bmsPoints.length >= 2 ? ['BMS SOH'] : []),
        ...data.forecasts.map((scenario) => fixedSlopeVerification && scenario.scenario.startsWith('slope_')
          ? `Slope ${scenario.scenario.slice('slope_'.length)} Ah / 1,000 Ah`
          : scenario.scenario.replaceAll('_', ' ')),
        ...(hasConfidenceInterval ? ['Model 95% interval'] : []),
      ],
    },
    tooltip: {
      trigger: 'axis',
      valueFormatter: (value: unknown) =>
        typeof value === 'number' ? `${value.toFixed(2)}%` : String(value ?? ''),
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
      nameTextStyle: { color: '#5e6a70', padding: [0, 0, 8, 0] },
      min: 75,
      max: (value: { max: number }) => Math.max(102, Math.ceil(value.max + 1)),
      axisLine: { show: true, lineStyle: { color: '#9ba8ae' } },
      axisTick: { show: false },
      axisLabel: { color: '#5e6a70', formatter: '{value}%' },
      splitLine: { lineStyle: { color: '#e5e9eb' } },
    },
    dataZoom: [
      { type: 'inside', filterMode: 'none' },
      {
        type: 'slider',
        height: 22,
        bottom: 18,
        borderColor: '#d5dcdf',
        fillerColor: 'rgba(23, 107, 135, 0.12)',
        backgroundColor: '#f6f8f9',
        dataBackground: { lineStyle: { color: '#80939c' }, areaStyle: { color: '#dce4e7' } },
        selectedDataBackground: { lineStyle: { color: '#176b87' }, areaStyle: { color: '#b8d1da' } },
      },
    ],
    series: chartSeries,
  }

  return (
    <ReactEChartsCore
      echarts={echarts}
      option={option}
      style={{ height: 440, width: '100%' }}
      notMerge
    />
  )
}
