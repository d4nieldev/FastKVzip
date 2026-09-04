import { useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { formatClock, formatDuration, formatTime, liveElapsed, stateClass } from '../lib/format'
import type { Job } from '../lib/types'

/** Pixels per second a name that does not fit drifts across its space. */
const NAME_SCROLL_SPEED = 35

/** The share of one leg spent moving; the remainder is a pause at each end. */
const NAME_SCROLL_MOVING = 0.6

/**
 * A run name, drifting sideways when it is too long to fit.
 *
 * These names encode the whole experiment -- architecture, gate, seed, ratio --
 * and the part that distinguishes one from another is at the end, which is
 * exactly what an ellipsis eats. Only a name that actually overflows moves; the
 * rest sit still, so the list is not in perpetual motion for the sake of it.
 */
function JobName({ name }: { name: string }) {
  const track = useRef<HTMLSpanElement>(null)
  const text = useRef<HTMLSpanElement>(null)
  const [overflow, setOverflow] = useState(0)

  useLayoutEffect(() => {
    const box = track.current
    const inner = text.current
    if (!box || !inner) return
    // offsetWidth, not a bounding rect: it ignores the transform this very
    // element may already be animating under, so re-measuring cannot drift.
    const measure = () => setOverflow(Math.max(0, inner.offsetWidth - box.clientWidth))
    measure()
    // The list column narrows when the detail pane opens, so a name that fitted
    // a moment ago may not any more.
    const observer = new ResizeObserver(measure)
    observer.observe(box)
    return () => observer.disconnect()
  }, [name])

  const style = overflow
    ? ({
        '--name-shift': `${-overflow}px`,
        // Proportional to the distance, so every name travels at the same
        // speed rather than long ones racing to fit a fixed duration.
        '--name-duration': `${overflow / (NAME_SCROLL_SPEED * NAME_SCROLL_MOVING)}s`,
      } as CSSProperties)
    : undefined

  return (
    <span className={overflow ? 'job-name is-overflowing' : 'job-name'} ref={track} title={name}>
      <span className={overflow ? 'job-name-text scrolls' : 'job-name-text'} ref={text} style={style}>
        {name}
      </span>
    </span>
  )
}

/**
 * Every timestamp the job has, labelled.
 *
 * The card used to show one stamp whose meaning changed with the state --
 * submitted while queued, started while running, ended once finished -- so the
 * same position on two cards meant two different things. A stamp drops its
 * date once the one above it already carries the same day, since three full
 * timestamps are usually the same date written three times.
 */
function stamps(job: Job): Array<{ label: string; text: string }> {
  const out: Array<{ label: string; text: string }> = []
  let lastDay: string | null = null
  for (const [label, epoch] of [
    ['submitted', job.submit_ts],
    ['started', job.start_ts],
    ['ended', job.end_ts],
  ] as Array<[string, number | null]>) {
    if (!epoch) continue
    const day = new Date(epoch * 1000).toDateString()
    out.push({ label, text: day === lastDay ? formatClock(epoch) : formatTime(epoch) })
    lastDay = day
  }
  return out
}

interface Props {
  jobs: Job[]
  selectedId: string | null
  nowEpoch: number
  onSelect: (job: Job) => void
  /** Label each card with its owner, when more than one user is on screen. */
  showUser?: boolean
  /** Job ids picked for filing into a project. */
  picked: string[]
  onTogglePick: (jobId: string) => void
  /** Pick or unpick everything currently listed. */
  onPickAll: (jobIds: string[]) => void
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

export function JobList({
  jobs,
  selectedId,
  nowEpoch,
  onSelect,
  showUser,
  picked,
  onTogglePick,
  onPickAll,
}: Props) {
  if (jobs.length === 0) {
    return <p className="empty">No jobs match this window and filter.</p>
  }

  const ids = jobs.map((job) => job.job_id)

  return (
    <div className="job-list">
      {/* The head of the selection rail: one box for everything listed,
          directly above the per-job ones it governs. */}
      <div className="list-header">
        <label className="pick-all">
          <input
            type="checkbox"
            checked={ids.length > 0 && ids.every((id) => picked.includes(id))}
            onChange={() => onPickAll(ids)}
            aria-label="Select every job listed"
          />
        </label>
        <span className="list-count">
          {jobs.length} job{jobs.length === 1 ? '' : 's'}
        </span>
      </div>

      <ul>
      {jobs.map((job) => (
        <li key={job.job_id}>
          <button
            type="button"
            // A run that finished since it was last opened announces itself
            // until it is read, which is the whole point of glancing at this
            // page: something ended, and you have not seen how.
            className={`job-card ${job.job_id === selectedId ? 'selected' : ''} ${
              job.unseen ? `unseen ${stateClass(job.state)}` : ''
            }`}
            onClick={() => onSelect(job)}
          >
            <div className="job-top">
              <span className={`badge ${stateClass(job.state)}`}>{job.state}</span>
              {showUser && job.user && <span className="tag user">{job.user}</span>}
              <JobName name={job.name ?? job.job_id} />
              {job.is_agent && <span className="tag">agent</span>}
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
            </div>

            <div className="job-times">
              {stamps(job).map((stamp) => (
                <span key={stamp.label}>
                  <span className="stamp-label">{stamp.label}</span> {stamp.text}
                </span>
              ))}
            </div>
          </button>

          {/* Beside the card, not inside it: a checkbox nested in a button is
              invalid markup and would swallow the click that opens the job. */}
          <label className="job-pick" title="Select for filing into a project">
            <input
              type="checkbox"
              checked={picked.includes(job.job_id)}
              onChange={() => onTogglePick(job.job_id)}
              aria-label={`Select job ${job.job_id}`}
            />
          </label>
        </li>
      ))}
      </ul>
    </div>
  )
}
