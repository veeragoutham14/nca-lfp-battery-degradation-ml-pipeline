import { ScanSearch, ShieldAlert } from 'lucide-react'
import { useState } from 'react'
import type { BatteryChemistry, DeviceLibraryItem, PipelineJob, PipelinePlugin } from '../types'
import { CellAnomalyResultsPage } from './CellAnomalyResultsPage'
import { CellSignalReadinessPage } from './CellSignalReadinessPage'

interface CellAnalysisPageProps {
  chemistry: BatteryChemistry
  devices: DeviceLibraryItem[]
  plugins: PipelinePlugin[]
  jobs: PipelineJob[]
  onCreateJob: (
    pipelineKey: string,
    parameters: Record<string, unknown>,
  ) => Promise<void>
}

export function CellAnalysisPage(props: CellAnalysisPageProps) {
  const [view, setView] = useState<'readiness' | 'anomalies'>('readiness')
  const [anomalySerial, setAnomalySerial] = useState('')

  function openAnomalyResults(serial: string) {
    setAnomalySerial(serial)
    setView('anomalies')
  }

  return (
    <div className="page-stack cell-analysis-page">
      <nav className="tabs cell-analysis-tabs" aria-label="Cell-voltage analysis views">
        <button
          className={view === 'readiness' ? 'active' : ''}
          type="button"
          onClick={() => setView('readiness')}
        >
          <ScanSearch size={15} aria-hidden="true" />Readiness
        </button>
        <button
          className={view === 'anomalies' ? 'active' : ''}
          type="button"
          onClick={() => setView('anomalies')}
        >
          <ShieldAlert size={15} aria-hidden="true" />Anomaly results
        </button>
      </nav>
      {view === 'readiness'
        ? <CellSignalReadinessPage {...props} onOpenAnomaly={openAnomalyResults} />
        : <CellAnomalyResultsPage {...props} initialSerial={anomalySerial} />}
    </div>
  )
}
