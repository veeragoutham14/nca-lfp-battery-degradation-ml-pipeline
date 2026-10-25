import type { LucideIcon } from 'lucide-react'

interface MetricCardProps {
  label: string
  value: string
  detail: string
  icon: LucideIcon
  tone?: 'neutral' | 'good' | 'warning' | 'critical'
}

export function MetricCard({ label, value, detail, icon: Icon, tone = 'neutral' }: MetricCardProps) {
  return (
    <div className={`metric-card metric-${tone}`}>
      <div className="metric-label">
        <Icon size={16} aria-hidden="true" />
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <span className="metric-detail">{detail}</span>
    </div>
  )
}

