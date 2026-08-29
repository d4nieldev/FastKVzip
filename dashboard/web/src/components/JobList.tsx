import { formatDuration, formatTime, liveElapsed, stateClass } from '../lib/format'
import type { Job } from '../lib/types'

interface Props {
  jobs: Job[]
  selectedId: string | null
  nowEpoch: number
  onSelect: (job: Job) => void
  onToggleHidden: (job: Job) => void
}

function WallClock({ job, nowEpoch }: { job: Job; nowEpoch: number }) {
  const elapsed = liveElapsed(job.elapsed_s, job.last_seen, job.state, nowEpoch)
  const limit = job.time_limit_s

  // A queued job has no wall clock to show -- it has not started consuming one.
  if (!job.start_ts && !job.is_terminal) {
    return (
      <span className="wall-text">
        queued{limit ? ` · ${formatDuration(limit)} requested` : ''}
        {job.est_start_ts && <em> · starts ~{formatTime(job.est_start_ts)}</em>}
      </span>
    )
  }

  if (elapsed === null || !limit) {
    return <span className="wall-text">{formatDuration(elapsed)}</span>
  }

  const fraction = Math.min(1, elapsed / limit)
  const remaining = Math.max(0, limit - elapsed)
  // Warn while there is still time to react to a job about to hit its limit.
  const level = job.is_terminal ? 'done' : fraction > 0.9 ? 'critical' : fraction > 0.75 ? 'warn' : 'ok'

  return (
    <div className="wall">
      <div className="wall-bar">
        <div className={`wall-fill ${level}`} style={{ width: `${fraction * 100}%` }} />
      </div>
      <span className="wall-text">
        {formatDuration(elapsed)} / {formatDuration(limit)}
        {!job.is_terminal && <em> · {formatDuration(remaining)} left</em>}
      </span>
    </div>
  )
}

export function JobList({ jobs, selectedId, nowEpoch, onSelect, onToggleHidden }: Props) {
  if (jobs.length === 0) {
    return <p className="empty">No jobs match this window and filter.</p>
  }

  return (
    <ul className="job-list">
      {jobs.map((job) => (
        <li key={job.job_id}>
          <button
            type="button"
            className={`job-card ${job.job_id === selectedId ? 'selected' : ''} ${
              job.hidden ? 'is-hidden' : ''
            }`}
            onClick={() => onSelect(job)}
          >
            <div className="job-top">
              <span className={`badge ${stateClass(job.state)}`}>{job.state}</span>
              <span className="job-name">{job.name ?? job.job_id}</span>
              {job.is_agent && <span className="tag">agent</span>}
              {job.hidden && <span className="tag">dismissed</span>}
              <span className="spacer" />
              <span className="job-id">#{job.job_id}</span>
            </div>

            <WallClock job={job} nowEpoch={nowEpoch} />

            <div className="job-meta">
              {job.gres && <span>{job.gres}</span>}
              {job.mem_req && <span>{job.mem_req}</span>}
              {job.node_list && <span>{job.node_list}</span>}
              {job.exit_code && <span className="alert">exit {job.exit_code}</span>}
              {/* Surfaces a stalled afterok chain: the grid's downstream jobs
                  sit in PENDING with Reason=Dependency after an early failure. */}
              {!job.is_terminal && job.reason && job.reason !== 'None' && (
                <span className="reason">{job.reason}</span>
              )}
              <span className="spacer" />
              <span className="job-time">
                {job.is_terminal ? formatTime(job.end_ts) : formatTime(job.start_ts ?? job.submit_ts)}
              </span>
            </div>
          </button>

          {(job.is_failure || job.is_terminal) && (
            <button
              type="button"
              className="dismiss"
              title={job.hidden ? 'Restore to the list' : 'Hide from the dashboard (does not touch the cluster)'}
              aria-label={job.hidden ? 'Restore job' : 'Dismiss job'}
              onClick={() => onToggleHidden(job)}
            >
              {job.hidden ? '↩' : '✕'}
            </button>
          )}
        </li>
      ))}
    </ul>
  )
}
