import { formatAgo, liveSince } from '../lib/format'
import type { UserSummary } from '../lib/types'

export const WINDOW_PRESETS = [
  { label: '1h', seconds: 3600 },
  { label: '6h', seconds: 6 * 3600 },
  { label: '24h', seconds: 24 * 3600 },
  { label: '7d', seconds: 7 * 86400 },
  { label: '30d', seconds: 30 * 86400 },
] as const

export const STATE_FILTERS = [
  'RUNNING',
  'PENDING',
  'FAILED',
  'COMPLETED',
  'CANCELLED',
  'TIMEOUT',
] as const

interface HeaderProps {
  /** The agents whose jobs are on screen. */
  agents: UserSummary[]
  error: string | null
  nowEpoch: number
  /** Browser-clock second at which `status` arrived. */
  fetchedAt: number
  onSelectJob: (jobId: string) => void
}

/**
 * Agent liveness is the single most important element on the page: if an agent
 * dies, everything below is frozen history for that user, and that must be
 * impossible to mistake for live data.
 *
 * With several users on screen the banner speaks for all of them, because one
 * stale agent among four means part of the list is stale and the rest is not --
 * which is worse than none of it being live, since nothing says which part.
 */
export function AgentBanner({ agents, error, nowEpoch, fetchedAt, onSelectJob }: HeaderProps) {
  if (error) {
    return <div className="banner down">Cannot reach the dashboard server — {error}</div>
  }
  if (!agents.length) {
    return <div className="banner">Connecting…</div>
  }

  const reports = agents.map((agent) => {
    const since = liveSince(agent.seconds_since_heartbeat, fetchedAt, nowEpoch)
    const interval = agent.poll_interval ?? 30
    // Two missed polls is noise; three means something is actually wrong.
    return { agent, since, stale: since === null || since > Math.max(120, interval * 3) }
  })
  const stale = reports.filter((report) => report.stale)

  if (stale.length === reports.length) {
    return (
      <div className="banner down">
        <span className="dot" />
        <span>
          <strong>{reports.length > 1 ? 'All agents stale' : 'Agent stale'}</strong> —{' '}
          {reports
            .map((r) => `${r.agent.user} ${r.since === null ? 'never reported' : formatAgo(r.since)}`)
            .join(', ')}
          . Job data below is not current.
        </span>
      </div>
    )
  }

  return (
    <div className={stale.length ? 'banner mixed' : 'banner live'}>
      <span className="dot" />
      <span>
        {stale.length > 0 && (
          <>
            <strong>{stale.map((r) => r.agent.user).join(', ')} stale</strong> — their jobs below
            are not current.{' '}
          </>
        )}
        {reports
          .filter((report) => !report.stale)
          .map((report, index) => (
            <span key={report.agent.user}>
              {index > 0 && ' · '}
              {reports.length > 1 && <strong>{report.agent.user}</strong>}
              {reports.length > 1 ? ' ' : <strong>Live — agent </strong>}
              reported {formatAgo(report.since)}
              {report.agent.job_id && (
                <>
                  {' ('}
                  <button
                    type="button"
                    className="linklike"
                    onClick={() => onSelectJob(report.agent.job_id as string)}
                  >
                    job #{report.agent.job_id}
                  </button>
                  {report.agent.host ? ` on ${report.agent.host})` : ')'}
                </>
              )}
            </span>
          ))}
      </span>
    </div>
  )
}

interface ControlsProps {
  windowSeconds: number
  onWindowChange: (seconds: number) => void
  states: string[]
  onStatesChange: (states: string[]) => void
  search: string
  onSearchChange: (value: string) => void
  counts: Record<string, number>
}

export function Controls({
  windowSeconds,
  onWindowChange,
  states,
  onStatesChange,
  search,
  onSearchChange,
  counts,
}: ControlsProps) {
  const toggleState = (state: string) => {
    onStatesChange(
      states.includes(state) ? states.filter((item) => item !== state) : [...states, state],
    )
  }

  return (
    <div className="controls">
      <div className="control-group">
        <span className="control-label">Window</span>
        {WINDOW_PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            className={windowSeconds === preset.seconds ? 'chip on' : 'chip'}
            onClick={() => onWindowChange(preset.seconds)}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="control-group">
        <span className="control-label">State</span>
        {STATE_FILTERS.map((state) => (
          <button
            key={state}
            type="button"
            className={states.includes(state) ? 'chip on' : 'chip'}
            onClick={() => toggleState(state)}
          >
            {state.toLowerCase()}
            {counts[state] ? <em>{counts[state]}</em> : null}
          </button>
        ))}
      </div>

      <input
        type="search"
        className="search"
        placeholder="Filter by run name or job id"
        value={search}
        onChange={(event) => onSearchChange(event.target.value)}
      />
    </div>
  )
}
