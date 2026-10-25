import { memo, useRef } from 'react'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import { RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { echarts } from '../echarts'
import type { CellAnomalySeriesPoint } from '../types'

interface CellAnomalyTrendChartProps {
  points: CellAnomalySeriesPoint[]
}

export const CellAnomalyTrendChart = memo(function CellAnomalyTrendChart({ points }: CellAnomalyTrendChartProps) {
  const chartRef = useRef<ReactEChartsCore | null>(null)
  const zoomWindow = useRef({ start: 0, end: 100 })

  if (points.length === 0) {
    return <div className="empty-chart">No daily anomaly series is available.</div>
  }

  function setZoom(start: number, end: number) {
    const boundedStart = Math.max(0, Math.min(start, 99))
    const boundedEnd = Math.max(boundedStart + 1, Math.min(end, 100))
    zoomWindow.current = { start: boundedStart, end: boundedEnd }
    chartRef.current?.getEchartsInstance().dispatchAction({
      type: 'dataZoom',
      dataZoomIndex: 0,
      start: boundedStart,
      end: boundedEnd,
    })
  }

  function zoomBy(multiplier: number) {
    const { start, end } = zoomWindow.current
    const center = (start + end) / 2
    const span = Math.min(100, Math.max(1, (end - start) * multiplier))
    let nextStart = center - span / 2
    let nextEnd = center + span / 2
    if (nextStart < 0) {
      nextEnd -= nextStart
      nextStart = 0
    }
    if (nextEnd > 100) {
      nextStart -= nextEnd - 100
      nextEnd = 100
    }
    setZoom(nextStart, nextEnd)
  }

  function showLastDays(days: number) {
    const lastTimestamp = Date.parse(points[points.length - 1].time)
    const cutoff = lastTimestamp - days * 24 * 60 * 60 * 1000
    const firstVisible = points.findIndex((point) => Date.parse(point.time) >= cutoff)
    const startIndex = firstVisible < 0 ? 0 : firstVisible
    const denominator = Math.max(1, points.length - 1)
    setZoom((100 * startIndex) / denominator, 100)
  }

  function rememberZoom() {
    const dataZoom = chartRef.current?.getEchartsInstance().getOption().dataZoom
    const current = Array.isArray(dataZoom) ? dataZoom[0] : undefined
    if (!current || typeof current !== 'object') return
    const start = Number(current.start)
    const end = Number(current.end)
    if (Number.isFinite(start) && Number.isFinite(end)) {
      zoomWindow.current = { start, end }
    }
  }

  const byTime = new Map(points.map((point) => [point.time, point]))
  const observedSpreadMaximum = Math.max(
    0,
    ...points.flatMap((point) => [point.spread_p95_mv, point.spread_max_mv])
      .filter((value): value is number => value != null && Number.isFinite(value)),
  )
  const spreadAxisMaximum = observedSpreadMaximum <= 50
    ? 60
    : observedSpreadMaximum <= 100
      ? 110
      : observedSpreadMaximum <= 200
        ? 220
        : Math.ceil((observedSpreadMaximum * 1.1) / 50) * 50
  const engineeringLines = [
    { name: 'Monitor 20 mV', yAxis: 20, lineStyle: { color: '#889398', type: 'dashed' } },
    { name: 'Candidate 30 mV', yAxis: 30, lineStyle: { color: '#b18737', type: 'dashed' } },
    { name: 'Review 50 mV', yAxis: 50, lineStyle: { color: '#d07532', type: 'dashed' } },
    { name: 'Serious 100 mV', yAxis: 100, lineStyle: { color: '#b34f49', type: 'dashed' } },
    { name: 'Severe 200 mV', yAxis: 200, lineStyle: { color: '#7d2d35', type: 'dashed' } },
  ].filter((line) => line.yAxis <= spreadAxisMaximum)
  const chartInitOptions = { renderer: 'canvas', useDirtyRect: false } as const
  const currentZoom = zoomWindow.current
  const option = {
    animation: false,
    grid: { top: 64, right: 64, bottom: 72, left: 64 },
    legend: {
      top: 6,
      textStyle: { color: '#59676d', fontSize: 10 },
    },
    tooltip: {
      trigger: 'axis',
      formatter: (items: Array<{ axisValue: string }>) => {
        const point = byTime.get(items[0]?.axisValue)
        if (!point) return ''
        return [
          `<strong>${new Intl.DateTimeFormat('en-GB', {
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            timeZone: 'UTC',
          }).format(new Date(point.time))} UTC</strong>`,
          `Spread p50: ${point.spread_p50_mv?.toFixed(1) ?? 'n/a'} mV`,
          `Spread p95: ${point.spread_p95_mv?.toFixed(1) ?? 'n/a'} mV`,
          `Maximum spread: ${point.spread_max_mv?.toFixed(1) ?? 'n/a'} mV`,
          `Maximum contextual score: ${point.maximum_anomaly_score?.toFixed(1) ?? 'n/a'}`,
          `Total sustained: ${point.sustained_episode_minutes?.toFixed(0) ?? '0'} min`,
          `Charging: ${point.sustained_charging_minutes.toFixed(0)} min`,
          `Discharging: ${point.sustained_discharging_minutes.toFixed(0)} min`,
          `Rest: ${point.sustained_rest_minutes.toFixed(0)} min`,
          `Mixed: ${point.sustained_mixed_minutes.toFixed(0)} min`,
          `Module epoch: ${point.module_configuration_epoch ?? 'n/a'} (${point.module_count ?? 'n/a'} modules)`,
          `Mean SOC: ${point.soc_mean_pct?.toFixed(1) ?? 'n/a'}%`,
          `Mean BMS pack temperature: ${point.temperature_mean_c?.toFixed(1) ?? 'n/a'} C`,
        ].join('<br>')
      },
    },
    xAxis: {
      type: 'category',
      data: points.map((point) => point.time),
      axisLabel: {
        color: '#5e6a70',
        formatter: (value: string) => new Date(value).toLocaleDateString(undefined, {
          day: '2-digit',
          month: 'short',
          year: '2-digit',
          timeZone: 'UTC',
        }),
      },
      axisLine: { lineStyle: { color: '#9ba8ae' } },
    },
    yAxis: [
      {
        type: 'value',
        name: 'Cell spread (mV)',
        min: 0,
        max: spreadAxisMaximum,
        nameTextStyle: { color: '#5e6a70' },
        axisLabel: { color: '#5e6a70', formatter: '{value} mV' },
        splitLine: { lineStyle: { color: '#e5e9eb' } },
      },
      {
        type: 'value',
        name: 'Sustained minutes',
        nameTextStyle: { color: '#8a5a20' },
        axisLabel: { color: '#8a5a20' },
        splitLine: { show: false },
      },
    ],
    dataZoom: [
      {
        type: 'inside',
        start: currentZoom.start,
        end: currentZoom.end,
        filterMode: 'filter',
        throttle: 16,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        moveOnMouseWheel: false,
        preventDefaultMouseMove: true,
      },
      {
        type: 'slider',
        start: currentZoom.start,
        end: currentZoom.end,
        filterMode: 'filter',
        throttle: 16,
        height: 24,
        bottom: 18,
        borderColor: '#d5dcdf',
        brushSelect: true,
        realtime: true,
        showDetail: true,
        handleSize: '110%',
        moveHandleSize: 8,
      },
    ],
    series: [
      {
        name: 'Cell spread p95',
        type: 'line',
        symbol: 'none',
        lineStyle: { width: 2, color: '#9f413c' },
        itemStyle: { color: '#9f413c' },
        data: points.map((point) => point.spread_p95_mv),
        markLine: {
          silent: true,
          symbol: 'none',
          label: { position: 'insideEndTop', fontSize: 9 },
          data: engineeringLines,
        },
      },
      {
        name: 'Daily maximum spread',
        type: 'line',
        symbol: 'none',
        lineStyle: { width: 1, color: '#55646b', opacity: 0.72 },
        itemStyle: { color: '#55646b' },
        data: points.map((point) => point.spread_max_mv),
      },
      {
        name: 'Charging',
        type: 'bar',
        yAxisIndex: 1,
        stack: 'sustained-minutes',
        barMaxWidth: 8,
        itemStyle: { color: 'rgba(47, 132, 115, 0.58)' },
        data: points.map((point) => point.sustained_charging_minutes),
      },
      {
        name: 'Discharging',
        type: 'bar',
        yAxisIndex: 1,
        stack: 'sustained-minutes',
        barMaxWidth: 8,
        itemStyle: { color: 'rgba(187, 81, 72, 0.58)' },
        data: points.map((point) => point.sustained_discharging_minutes),
      },
      {
        name: 'Rest',
        type: 'bar',
        yAxisIndex: 1,
        stack: 'sustained-minutes',
        barMaxWidth: 8,
        itemStyle: { color: 'rgba(101, 113, 119, 0.46)' },
        data: points.map((point) => point.sustained_rest_minutes),
      },
      {
        name: 'Mixed',
        type: 'bar',
        yAxisIndex: 1,
        stack: 'sustained-minutes',
        barMaxWidth: 8,
        itemStyle: { color: 'rgba(178, 126, 45, 0.54)' },
        data: points.map((point) => point.sustained_mixed_minutes),
      },
    ],
    media: [
      {
        query: { maxWidth: 600 },
        option: {
          grid: { top: 132, right: 48, bottom: 72, left: 48 },
          legend: {
            top: 5,
            left: 8,
            orient: 'vertical',
            itemGap: 4,
            itemWidth: 18,
            textStyle: { color: '#59676d', fontSize: 9 },
          },
          yAxis: [
            {
              name: '',
              axisLabel: { color: '#5e6a70', fontSize: 9, formatter: '{value}' },
            },
            {
              name: '',
              axisLabel: { color: '#8a5a20', fontSize: 9 },
            },
          ],
        },
      },
    ],
  }

  return (
    <div className="anomaly-chart-shell">
      <div className="anomaly-chart-controls" aria-label="Chart zoom controls">
        <div className="chart-range-control" role="group" aria-label="Visible date range">
          <button type="button" onClick={() => showLastDays(30)}>30d</button>
          <button type="button" onClick={() => showLastDays(90)}>90d</button>
          <button type="button" onClick={() => showLastDays(365)}>1y</button>
          <button type="button" onClick={() => setZoom(0, 100)}>All</button>
        </div>
        <div className="chart-zoom-actions">
          <button className="icon-button" type="button" onClick={() => zoomBy(0.5)} title="Zoom in" aria-label="Zoom in">
            <ZoomIn size={16} aria-hidden="true" />
          </button>
          <button className="icon-button" type="button" onClick={() => zoomBy(2)} title="Zoom out" aria-label="Zoom out">
            <ZoomOut size={16} aria-hidden="true" />
          </button>
          <button className="icon-button" type="button" onClick={() => setZoom(0, 100)} title="Reset zoom" aria-label="Reset zoom">
            <RotateCcw size={16} aria-hidden="true" />
          </button>
        </div>
      </div>
      <ReactEChartsCore
        ref={chartRef}
        echarts={echarts}
        option={option}
        onEvents={{ datazoom: rememberZoom }}
        style={{ height: 430, width: '100%' }}
        opts={chartInitOptions}
        notMerge
      />
    </div>
  )
})
