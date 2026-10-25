import { ExternalLink, FileJson, FileText, Image, Table2 } from 'lucide-react'
import type { Artifact, Dataset } from '../types'
import { fileSize, integer, shortDateTime } from '../utils'

function ArtifactIcon({ mediaType }: { mediaType: string }) {
  if (mediaType === 'image/png') return <Image size={17} aria-hidden="true" />
  if (mediaType === 'application/json') return <FileJson size={17} aria-hidden="true" />
  if (mediaType.includes('parquet')) return <Table2 size={17} aria-hidden="true" />
  return <FileText size={17} aria-hidden="true" />
}

export function ArtifactTable({ artifacts, datasets }: { artifacts: Artifact[]; datasets: Dataset[] }) {
  const rowsByName = new Map(datasets.map((dataset) => [dataset.name, dataset.row_count]))
  if (artifacts.length === 0) {
    return <div className="empty-state">No registered artifacts for this run.</div>
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Artifact</th>
            <th>Role</th>
            <th>Rows</th>
            <th>Size</th>
            <th>Modified</th>
            <th aria-label="Open" />
          </tr>
        </thead>
        <tbody>
          {artifacts.map((artifact) => (
            <tr key={artifact.artifact_id}>
              <td>
                <span className="file-name">
                  <ArtifactIcon mediaType={artifact.media_type} />
                  {artifact.name}
                </span>
              </td>
              <td><span className="role-label">{artifact.kind.replaceAll('_', ' ')}</span></td>
              <td className="numeric">{integer(rowsByName.get(artifact.name))}</td>
              <td className="numeric">{fileSize(artifact.size_bytes)}</td>
              <td>{shortDateTime(artifact.modified_at)}</td>
              <td>
                <a className="icon-button" href={artifact.download_url} target="_blank" rel="noreferrer" title={`Open ${artifact.name}`}>
                  <ExternalLink size={16} aria-hidden="true" />
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

