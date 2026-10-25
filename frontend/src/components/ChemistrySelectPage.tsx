import { ArrowRight, BatteryCharging, FlaskConical } from 'lucide-react'
import type { BatteryChemistry, DeviceLibraryItem } from '../types'

interface ChemistrySelectPageProps {
  devices: DeviceLibraryItem[]
  onSelect: (chemistry: BatteryChemistry) => void
}

export function ChemistrySelectPage({ devices, onSelect }: ChemistrySelectPageProps) {
  const counts = (chemistry: BatteryChemistry) => ({
    configured: devices.filter((device) => device.chemistry === chemistry).length,
    imported: devices.filter(
      (device) => device.chemistry === chemistry && device.readiness.raw,
    ).length,
  })

  return (
    <div className="page-stack chemistry-select-page">
      <header className="page-header chemistry-select-header">
        <div>
          <span className="eyebrow">Battery chemistry</span>
          <h1>Select an analysis workspace</h1>
          <p>NCA and LFP data, models, thresholds, and conclusions remain isolated.</p>
        </div>
      </header>
      <section className="chemistry-destinations" aria-label="Battery chemistry workspaces">
        {(['NCA', 'LFP'] as BatteryChemistry[]).map((chemistry) => {
          const chemistryCounts = counts(chemistry)
          const Icon = chemistry === 'NCA' ? BatteryCharging : FlaskConical
          return (
            <button type="button" key={chemistry} onClick={() => onSelect(chemistry)}>
              <span className={`chemistry-mark chemistry-${chemistry.toLowerCase()}`}>
                <Icon size={25} aria-hidden="true" />
              </span>
              <span className="chemistry-copy">
                <strong>{chemistry}</strong>
                <span>{chemistry === 'NCA' ? 'Current field and full-cycle analysis' : 'Independent LFP extraction and analysis'}</span>
              </span>
              <span className="chemistry-counts">
                <b>{chemistryCounts.imported} imported</b>
                <small>{chemistryCounts.configured} configured</small>
              </span>
              <ArrowRight size={19} aria-hidden="true" />
            </button>
          )
        })}
      </section>
    </div>
  )
}
