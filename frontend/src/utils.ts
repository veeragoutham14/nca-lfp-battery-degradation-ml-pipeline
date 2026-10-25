import type { DeviceLibraryItem } from './types'

export function compareDevicesByModuleCount(
  left: DeviceLibraryItem,
  right: DeviceLibraryItem,
): number {
  const leftModules = left.module_count_status === 'verified'
    ? (left.module_count ?? Number.POSITIVE_INFINITY)
    : Number.POSITIVE_INFINITY
  const rightModules = right.module_count_status === 'verified'
    ? (right.module_count ?? Number.POSITIVE_INFINITY)
    : Number.POSITIVE_INFINITY
  return leftModules - rightModules || left.serial.localeCompare(right.serial)
}

export function number(value: number | null | undefined, digits = 2): string {
  return value == null || Number.isNaN(value) ? 'n/a' : value.toFixed(digits)
}

export function integer(value: number | null | undefined): string {
  return value == null ? 'n/a' : new Intl.NumberFormat('en-US').format(value)
}

export function compactNumber(value: number | null | undefined): string {
  return value == null
    ? 'n/a'
    : new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

export function shortDate(value: string | null | undefined): string {
  if (!value) return 'n/a'
  const formatted = new Intl.DateTimeFormat('en-GB', {
    day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC',
  }).format(new Date(value))
  return `${formatted} UTC`
}

export function shortDateTime(value: string | null | undefined): string {
  if (!value) return 'n/a'
  const formatted = new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    timeZone: 'UTC',
  }).format(new Date(value))
  return `${formatted} UTC`
}

export function fileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

export function familyLabel(value: string | null | undefined): string {
  if (value === 'full_discharge_capacity') return 'Full discharge'
  if (value === 'fixed_slope_transfer_verification') return 'Fixed-slope verification'
  if (value === 'bms_soh_signal_audit') return 'BMS SOH signal audit'
  if (value === 'bms_soh_recalibration_evidence') return 'BMS SOH recalibration evidence'
  if (value === 'bms_electrothermal_source_quality') return 'Electrothermal source quality'
  if (value === 'bms_hierarchical_electrothermal_features') return 'Hierarchical electrothermal features'
  if (value === 'hierarchical_cell_voltage_anomaly') return 'Hierarchical cell-voltage anomaly'
  if (value === 'seasonal_charging_stress_profile') return 'Seasonal charging and stress profile'
  return value?.replaceAll('_', ' ') ?? 'Unknown model'
}
