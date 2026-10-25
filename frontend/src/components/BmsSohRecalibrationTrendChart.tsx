import { memo, useRef } from 'react'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import { RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { echarts } from '../echarts'
import type { BmsSohRecalibrationEvent, BmsSohRecalibrationPoint, CapacityBmsDischargeEvent } from '../types'

interface BmsSohRecalibrationTrendChartProps {
  points: BmsSohRecalibrationPoint[]
  events: BmsSohRecalibrationEvent[]
  capacityEvents?: CapacityBmsDischargeEvent[]
}

function utcDate(value: string) {
  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(value))
}

function readable(value: string) {
  return value.replaceAll('_', ' ')
}

export const BmsSohRecalibrationTrendChart = memo(function BmsSohRecalibrationTrendChart({
  points,
  events,
  capacityEvents = [],
}: BmsSohRecalibrationTrendChartProps) {
  const chartRef = useRef<ReactEChartsCore | null>(null)
  const zoomWindow = useRef({ start: 0, end: 100 })

  if (points.length === 0) {
    return <div className="empty-chart">No audited daily BMS SOH timeline is available.</div>
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
    const cutoff = lastTimestamp - days * 86_400_000
    const firstVisible = points.findIndex((point) => Date.parse(point.time) >= cutoff)
    const denominator = Math.max(1, points.length - 1)
    setZoom(100 * Math.max(0, firstVisible) / denominator, 100)
  }

  function rememberZoom() {
    const dataZoom = chartRef.current?.getEchartsInstance().getOption().dataZoom
    const current = Array.isArray(dataZoom) ? dataZoom[0] : undefined
    if (!current || typeof current !== 'object') return
    const start = Number(current.start)
    const end = Number(current.end)
    if (Number.isFinite(start) && Number.isFinite(end)) zoomWindow.current = { start, end }
  }

  const pointsByDay = new Map(points.map((point) => [point.time.slice(0, 10), point]))
  const eventsByDay = new Map<string, BmsSohRecalibrationEvent[]>()
  events.forEach((event) => {
    const day = event.event_time.slice(0, 10)
    const existing = eventsByDay.get(day) ?? []
    existing.push(event)
    eventsByDay.set(day, existing)
  })
  const capacityByDay = new Map<string, CapacityBmsDischargeEvent[]>()
  capacityEvents.forEach((event) => {
    if (!event.end_utc) return
    const day = event.end_utc.slice(0, 10)
    const existing = capacityByDay.get(day) ?? []
    existing.push(event)
    capacityByDay.set(day, existing)
  })
  const sohValues = points
    .flatMap((point) => [point.measured_soh_pct, point.bms_soh_end])
    .filter((value): value is number => value != null && Number.isFinite(value))
  const yMinimum = sohValues.length ? Math.max(0, Math.floor((Math.min(...sohValues) - 2) / 5) * 5) : 0
  const yMaximum = sohValues.length ? Math.min(105, Math.ceil((Math.max(...sohValues) + 2) / 5) * 5) : 100
  const option = {
    animation: false,
    grid: { top: 58, right: 62, bottom: 72, left: 62 },
    legend: { top: 5, textStyle: { color: '#59676d', fontSize: 10 } },
    tooltip: {
      trigger: 'axis',
      formatter: (items: Array<{ axisValue: string | number }>) => {
        const axisValue = items[0]?.axisValue
        const time = new Date(axisValue).toISOString()
        const day = time.slice(0, 10)
        const point = pointsByDay.get(day)
        const dailyEvents = eventsByDay.get(day) ?? []
        const evidence = dailyEvents.map((event) => (
          `${event.event_type.replace('step_', '')}: ${event.soh_delta_pp?.toFixed(2) ?? 'n/a'} pp, ${readable(event.primary_context_pattern ?? event.evidence_classification)} (${event.context_match_strength ?? event.classification_confidence})`
        ))
        const capacityEvidence = (capacityByDay.get(day) ?? []).map((event) => (
          `${readable(event.evidence_class)}: ${event.discharged_ah?.toFixed(2) ?? 'n/a'} Ah, ${readable(event.measurement_quality)} quality`
        ))
        return [
          `<strong>${utcDate(time)} UTC</strong>`,
          `Daily BMS SOH: ${point?.measured_soh_pct?.toFixed(2) ?? 'n/a'}%`,
          `Daily range: ${point?.daily_soh_range_pct?.toFixed(2) ?? 'n/a'} pp`,
          `Audit coverage: ${point?.coverage_fraction == null ? 'n/a' : `${(100 * point.coverage_fraction).toFixed(0)}%`}`,
          `Charge / discharge: ${point?.charge_ah?.toFixed(1) ?? 'n/a'} / ${point?.discharge_ah?.toFixed(1) ?? 'n/a'} Ah`,
          `Largest raw gap: ${point?.max_gap_seconds?.toFixed(0) ?? 'n/a'} s`,
          ...evidence,
          ...capacityEvidence,
        ].join('<br>')
      },
    },
    xAxis: {
      type: 'time',
      axisLabel: {
        color: '#5e6a70',
        formatter: (value: number) => new Date(value).toLocaleDateString(undefined, {
          day: '2-digit', month: 'short', year: '2-digit', timeZone: 'UTC',
        }),
      },
      axisLine: { lineStyle: { color: '#9ba8ae' } },
    },
    yAxis: [
      {
        type: 'value',
        name: 'BMS SOH (%)',
        min: yMinimum,
        max: yMaximum,
        nameTextStyle: { color: '#5e6a70' },
        axisLabel: { color: '#5e6a70', formatter: '{value}%' },
        splitLine: { lineStyle: { color: '#e5e9eb' } },
      },
      {
        type: 'value',
        name: 'Events/day',
        minInterval: 1,
        nameTextStyle: { color: '#946a1f' },
        axisLabel: { color: '#946a1f' },
        splitLine: { show: false },
      },
    ],
    dataZoom: [
      {
        type: 'inside', start: zoomWindow.current.start, end: zoomWindow.current.end,
        filterMode: 'filter', throttle: 16, zoomOnMouseWheel: true,
        moveOnMouseMove: true, moveOnMouseWheel: false, preventDefaultMouseMove: true,
      },
      {
        type: 'slider', start: zoomWindow.current.start, end: zoomWindow.current.end,
        filterMode: 'filter', throttle: 16, height: 24, bottom: 18,
        borderColor: '#d5dcdf', brushSelect: true, realtime: true, showDetail: true,
        handleSize: '110%', moveHandleSize: 8,
      },
    ],
    series: [
      {
        name: 'Daily BMS SOH', type: 'line', symbol: 'none',
        lineStyle: { width: 2, color: '#176b87' }, itemStyle: { color: '#176b87' },
        data: points.map((point) => [point.time, point.measured_soh_pct]),
      },
      {
        name: 'Step drop', type: 'scatter', symbolSize: 10,
        itemStyle: { color: '#b34f49' },
        data: events.filter((event) => event.event_type === 'step_drop' && event.transition_detection_status === 'exact').map((event) => [event.event_time, event.new_soh_pct]),
      },
      {
        name: 'Recovery', type: 'scatter', symbolSize: 10,
        itemStyle: { color: '#2f7b59' },
        data: events.filter((event) => event.event_type === 'step_recovery' && event.transition_detection_status === 'exact').map((event) => [event.event_time, event.new_soh_pct]),
      },
      {
        name: 'Interval or fallback', type: 'scatter', symbolSize: 11, symbol: 'diamond',
        itemStyle: { color: '#c68121' },
        data: events.filter((event) => event.transition_detection_status !== 'exact').map((event) => [event.event_time, event.new_soh_pct]),
      },
      {
        name: 'Capacity observation', type: 'scatter', symbolSize: 11, symbol: 'triangle',
        itemStyle: { color: '#6b5b95' },
        data: capacityEvents
          .filter((event) => event.end_utc && (event.discharge_derived_soh_proxy_pct != null || event.soc_span_normalized_soh_estimate_pct != null))
          .map((event) => [event.end_utc, event.discharge_derived_soh_proxy_pct ?? event.soc_span_normalized_soh_estimate_pct]),
      },
      {
        name: 'Audited events/day', type: 'bar', yAxisIndex: 1, barMaxWidth: 9,
        itemStyle: { color: 'rgba(181, 134, 50, 0.45)' },
        data: points.map((point) => [point.time, point.recalibration_event_count]),
      },
    ],
  }

  return (
    <div className="anomaly-chart-shell recalibration-chart-shell">
      <div className="anomaly-chart-controls" aria-label="Chart zoom controls">
        <div className="chart-range-control" role="group" aria-label="Visible date range">
          <button type="button" onClick={() => showLastDays(30)}>30d</button>
          <button type="button" onClick={() => showLastDays(90)}>90d</button>
          <button type="button" onClick={() => showLastDays(365)}>1y</button>
          <button type="button" onClick={() => setZoom(0, 100)}>All</button>
        </div>
        <div className="chart-zoom-actions">
          <button className="icon-button" type="button" onClick={() => zoomBy(0.5)} title="Zoom in" aria-label="Zoom in"><ZoomIn size={16} aria-hidden="true" /></button>
          <button className="icon-button" type="button" onClick={() => zoomBy(2)} title="Zoom out" aria-label="Zoom out"><ZoomOut size={16} aria-hidden="true" /></button>
          <button className="icon-button" type="button" onClick={() => setZoom(0, 100)} title="Reset zoom" aria-label="Reset zoom"><RotateCcw size={16} aria-hidden="true" /></button>
        </div>
      </div>
      <ReactEChartsCore
        ref={chartRef}
        echarts={echarts}
        option={option}
        onEvents={{ datazoom: rememberZoom }}
        style={{ height: 430, width: '100%' }}
        opts={{ renderer: 'canvas' }}
        notMerge
      />
    </div>
  )
})
