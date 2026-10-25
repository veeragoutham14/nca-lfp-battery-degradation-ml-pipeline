import { Download, Play, ServerCog } from 'lucide-react'
import type { PipelineJob, PipelinePlugin } from '../types'
import { familyLabel, shortDateTime } from '../utils'

interface PipelineWorkspaceProps {
  plugins: PipelinePlugin[]
  jobs: PipelineJob[]
  canRunAnalysis: boolean
  canImport: boolean
  onRunAnalysis: () => void
  onImport: () => void
}

export function PipelineWorkspace({
  plugins,
  jobs,
  canRunAnalysis,
  canImport,
  onRunAnalysis,
  onImport,
}: PipelineWorkspaceProps) {
  return (
    <div className="tab-content pipeline-layout">
      <section className="panel">
        <div className="section-title">
          <div><h2>Available pipelines</h2><span>{plugins.length} registered plugins</span></div>
          <div className="topbar-actions">
            <button className="button secondary" type="button" onClick={onImport} disabled={!canImport}>
              <Download size={16} aria-hidden="true" />Import data
            </button>
            <button className="button primary" type="button" onClick={onRunAnalysis} disabled={!canRunAnalysis}>
              <Play size={16} aria-hidden="true" />Run analysis
            </button>
          </div>
        </div>
        <div className="plugin-list">
          {plugins.map((plugin) => (
            <div className="plugin-row" key={plugin.key}>
              <span className="plugin-icon"><ServerCog size={18} aria-hidden="true" /></span>
              <div><strong>{plugin.name}</strong><span>{familyLabel(plugin.output_family)}</span></div>
              <code>{plugin.key}</code>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <div className="section-title"><div><h2>Job history</h2><span>{jobs.length} recorded jobs</span></div></div>
        {jobs.length === 0 ? <div className="empty-state">No platform jobs have been started.</div> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Battery</th><th>Pipeline</th><th>Status</th><th>Started</th><th>Code</th></tr></thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.job_id} title={job.error_message ?? undefined}>
                    <td>{job.serial}</td>
                    <td>{job.pipeline_key.split('.')[1]?.replaceAll('_', ' ')}</td>
                    <td><span className={`job-status status-${job.status}`}><span />{job.status}</span></td>
                    <td>{shortDateTime(job.started_at ?? job.created_at)}</td>
                    <td className="numeric">{job.return_code ?? 'n/a'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
