import { LogViewer } from './LogViewer'
import { formatDuration, formatTime, liveElapsed, stateClass } from '../lib/format'
import type { Job } from '../lib/types'

interface Props {
  job: Job
  nowEpoch: number
  onClose: () => void
  onToggleHidden: (job: Job) => void
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="detail-row">
      <span className="detail-label">{label}</span>
      <span className="detail-value">{value ?? '—'}</span>
    </div>
  )
}

export function JobDetail({ job, nowEpoch, onClose, onToggleHidden }: Props) {
  const elapsed = liveElapsed(job.elapsed_s, job.last_seen, job.state, nowEpoch)
  const remaining =
    job.time_limit_s !== null && elapsed !== null && job.start_ts && !job.is_terminal
      ? Math.max(0, job.time_limit_s - elapsed)
      : null

  return (
    <div className="detail">
      <header className="detail-header">
        <div>
          <span className={`badge ${stateClass(job.state)}`}>{job.state}</span>
          <h2>{job.name ?? job.job_id}</h2>
          <p className="detail-subtitle">Job {job.job_id}</p>
        </div>
        <div className="detail-actions">
          <button type="button" onClick={() => onToggleHidden(job)}>
            {job.hidden ? 'Restore' : 'Dismiss'}
          </button>
          <button type="button" className="close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
      </header>

      <div className="detail-grid">
        <section>
          <h3>Timing</h3>
          <Row label="Submitted" value={formatTime(job.submit_ts)} />
          {job.est_start_ts && !job.start_ts ? (
            <Row label="Est. start" value={formatTime(job.est_start_ts)} />
          ) : (
            <Row label="Started" value={formatTime(job.start_ts)} />
          )}
          <Row label="Ended" value={formatTime(job.end_ts)} />
          <Row label="Elapsed" value={formatDuration(elapsed)} />
          <Row label="Wall time limit" value={formatDuration(job.time_limit_s)} />
          <Row
            label="Remaining"
            value={remaining === null ? '—' : formatDuration(remaining)}
          />
        </section>

        <section>
          <h3>Requested</h3>
          <Row label="Partition" value={job.partition} />
          <Row label="GPU" value={job.gres} />
          <Row label="Memory" value={job.mem_req} />
          <Row label="CPUs" value={job.cpus} />
          <Row label="Nodes" value={job.nodes} />
          <Row label="Req TRES" value={<code>{job.req_tres ?? '—'}</code>} />
        </section>

        <section>
          <h3>Outcome</h3>
          <Row label="Exit code" value={job.exit_code} />
          <Row label="Reason" value={job.reason} />
          <Row label="Dependency" value={job.dependency} />
          <Row label="Node list" value={job.node_list} />
          <Row label="Peak RSS" value={job.max_rss} />
          <Row label="Alloc TRES" value={<code>{job.alloc_tres ?? '—'}</code>} />
        </section>

        <section>
          <h3>Location</h3>
          <Row label="Work dir" value={<code>{job.work_dir ?? '—'}</code>} />
          <Row label="Log file" value={<code>{job.log_path ?? '—'}</code>} />
          <Row label="Last seen" value={formatTime(job.last_seen)} />
        </section>
      </div>

      <LogViewer job={job} />
    </div>
  )
}
