import { ArrowRight, BrainCircuit, DatabaseZap, HardDrive, Workflow } from 'lucide-react'
import type { BatteryChemistry, DeviceLibraryItem, PipelineJob } from '../types'
import { shortDateTime } from '../utils'

interface HomePageProps {
  chemistry: BatteryChemistry
  devices: DeviceLibraryItem[]
  jobs: PipelineJob[]
  onNavigate: (page: 'extraction' | 'ml') => void
}

export function HomePage({ chemistry, devices, jobs, onNavigate }: HomePageProps) {
  const imported = devices.filter((device) => device.readiness.raw).length
  const readyForMl = devices.filter(
    (device) => device.readiness.hierarchical_features || device.readiness.events,
  ).length
  const activeJobs = jobs.filter((job) => job.status === 'queued' || job.status === 'running')

  return (
    <div className="page-stack home-page">
      <header className="page-header">
        <div>
          <span className="eyebrow">{chemistry} workspace</span>
          <h1>{chemistry} battery data operations</h1>
        </div>
        {activeJobs.length > 0 && (
          <span className="active-job-badge"><span />{activeJobs.length} active</span>
        )}
      </header>

      <section className="destination-list" aria-label="Workspace destinations">
        <button className="destination-row" type="button" onClick={() => onNavigate('extraction')}>
          <span className="destination-icon extraction-icon"><DatabaseZap size={24} aria-hidden="true" /></span>
          <span className="destination-copy">
            <strong>Data extraction</strong>
            <span>Database import and local telemetry library</span>
          </span>
          <span className="destination-count"><b>{imported}</b><small>imported</small></span>
          <ArrowRight size={20} aria-hidden="true" />
        </button>
        <button className="destination-row" type="button" onClick={() => onNavigate('ml')}>
          <span className="destination-icon ml-icon"><BrainCircuit size={24} aria-hidden="true" /></span>
          <span className="destination-copy">
            <strong>Machine learning</strong>
            <span>Preprocessing, model execution, verification, and reports</span>
          </span>
          <span className="destination-count"><b>{readyForMl}</b><small>analysis-ready</small></span>
          <ArrowRight size={20} aria-hidden="true" />
        </button>
      </section>

      <div className="home-grid">
        <section className="panel system-summary">
          <div className="section-title"><div><h2>Data estate</h2><span>Configured and local devices</span></div></div>
          <dl className="summary-list">
            <div><dt><HardDrive size={16} aria-hidden="true" />Configured devices</dt><dd>{devices.length}</dd></div>
            <div><dt><DatabaseZap size={16} aria-hidden="true" />Raw telemetry available</dt><dd>{imported}</dd></div>
            <div><dt><Workflow size={16} aria-hidden="true" />Operating events available</dt><dd>{readyForMl}</dd></div>
          </dl>
        </section>
        <section className="panel recent-jobs">
          <div className="section-title"><div><h2>Recent activity</h2><span>Latest pipeline jobs</span></div></div>
          {jobs.length === 0 ? <div className="empty-state compact-empty">No jobs recorded.</div> : (
            <div className="activity-list">
              {jobs.slice(0, 5).map((job) => (
                <div className="activity-row" key={job.job_id}>
                  <span className={`activity-state state-${job.status}`}><span /></span>
                  <div><strong>{job.serial}</strong><span>{job.pipeline_key.split('.')[1]?.replaceAll('_', ' ')} - {job.status}</span></div>
                  <time>{shortDateTime(job.started_at ?? job.created_at)}</time>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
