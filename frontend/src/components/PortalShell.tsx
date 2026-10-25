import {
  BatteryCharging,
  BrainCircuit,
  DatabaseZap,
  CalendarRange,
  GitCompareArrows,
  Home,
  LogOut,
  RefreshCw,
} from 'lucide-react'
import type { ReactNode } from 'react'
import type { AuthUser, BatteryChemistry, Health } from '../types'

export type PortalPage = 'home' | 'extraction' | 'ml' | 'seasonal' | 'gap' | 'run'

interface PortalShellProps {
  activePage: PortalPage
  chemistry: BatteryChemistry | null
  user: AuthUser
  health: Health | null
  refreshing: boolean
  children: ReactNode
  onNavigate: (page: 'home' | 'extraction' | 'ml' | 'seasonal' | 'gap') => void
  onChooseChemistry: () => void
  onRefresh: () => void
  onLogout: () => void
}

const navigation = [
  { page: 'home' as const, label: 'Home', icon: Home },
  { page: 'extraction' as const, label: 'Extraction', icon: DatabaseZap },
  { page: 'ml' as const, label: 'Machine learning', icon: BrainCircuit },
  { page: 'seasonal' as const, label: 'Seasonal profiles', icon: CalendarRange },
  { page: 'gap' as const, label: 'Capacity & BMS SOH', icon: GitCompareArrows },
]

export function PortalShell({
  activePage,
  chemistry,
  user,
  health,
  refreshing,
  children,
  onNavigate,
  onChooseChemistry,
  onRefresh,
  onLogout,
}: PortalShellProps) {
  return (
    <div className="portal-shell">
      <header className="portal-topbar">
        <button className="portal-brand" type="button" onClick={onChooseChemistry}>
          <span className="brand-mark"><BatteryCharging size={21} aria-hidden="true" /></span>
          <span><strong>BatteryML</strong><small>Battery analytics</small></span>
        </button>
        <nav className="primary-navigation" aria-label="Primary navigation">
          {chemistry && <button className="chemistry-context" type="button" onClick={onChooseChemistry}>{chemistry}</button>}
          {chemistry && navigation.filter(({ page }) => page !== 'gap' || chemistry === 'NCA').map(({ page, label, icon: Icon }) => (
            <button
              key={page}
              type="button"
              className={activePage === page || (activePage === 'run' && page === 'ml') ? 'active' : ''}
              onClick={() => onNavigate(page)}
            >
              <Icon size={16} aria-hidden="true" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="portal-actions">
          <span className={`connection-badge connection-${health?.status ?? 'degraded'}`}>
            <span className="status-dot" />
            {health?.database?.reachable
              ? 'Database connected'
              : health
                ? 'Database offline'
                : 'Checking source'}
          </span>
          <button
            className="icon-button"
            type="button"
            onClick={onRefresh}
            disabled={refreshing}
            title="Refresh platform data"
          >
            <RefreshCw size={17} className={refreshing ? 'spin' : ''} aria-hidden="true" />
          </button>
          <span className="operator-name">{user.username}</span>
          <button className="icon-button" type="button" onClick={onLogout} title="Sign out">
            <LogOut size={17} aria-hidden="true" />
          </button>
        </div>
      </header>
      <main className="portal-content">{children}</main>
    </div>
  )
}
