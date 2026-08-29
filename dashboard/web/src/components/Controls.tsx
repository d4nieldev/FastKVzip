import { formatAgo } from '../lib/format'
import type { Status } from '../lib/types'

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
  status: Status | null
  error: string | null
}

/**
 * Agent liveness is the single most important element on the page: if the
 * agent job dies, everything below is frozen history, and that must be
 * impossible to mistake for live data.
 */
export function AgentBanner({ status, error }: HeaderProps) {
  if (error) {
    return <div className="banner down">Cannot reach the dashboard server — {error}</div>
  }
  if (!status) {
    return <div className="banner">Connecting…</div>
  }

  const since = status.agent.seconds_since_heartbeat
  const interval = status.agent.poll_interval ?? 30
  // Two missed polls is noise; three means something is actually wrong.
  const stale = since === null || since > Math.max(120, interval * 3)

  return (
    <div className={stale ? 'banner down' : 'banner live'}>
      <span className="dot" />
      {stale ? (
        <span>
          <strong>Agent stale</strong> — last report {formatAgo(since)}. Job data below is not
          current.
        </span>
      ) : (
        <span>
          <strong>Live</strong> — agent reported {formatAgo(since)}
          {status.agent.job_id && ` (job #${status.agent.job_id}`}
          {status.agent.host && ` on ${status.agent.host})`}
        </span>
      )}
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
  includeHidden: boolean
  onIncludeHiddenChange: (value: boolean) => void
  counts: Record<string, number>
  hiddenCount: number
}

export function Controls({
  windowSeconds,
  onWindowChange,
  states,
  onStatesChange,
  search,
  onSearchChange,
  includeHidden,
  onIncludeHiddenChange,
  counts,
  hiddenCount,
}: ControlsProps) {
  const toggleState = (state: string) => {
    onStatesChange(
      states.includes(state) ? states.filter((item) => item !== state) : [...states, state],
    )
  }

  return (
    <div className="controls">
      <div className="control-row">
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

      <div className="control-row">
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

      <div className="control-row">
        <input
          type="search"
          className="search"
          placeholder="Filter by run name or job id"
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
        />
        <button
          type="button"
          className={includeHidden ? 'chip on' : 'chip'}
          onClick={() => onIncludeHiddenChange(!includeHidden)}
        >
          dismissed{hiddenCount ? <em>{hiddenCount}</em> : null}
        </button>
      </div>
    </div>
  )
}

export function SresPanel({ status }: { status: Status | null }) {
  if (!status?.sres?.body) return null
  return (
    <details className="sres">
      <summary>GPU availability (sres) · {formatAgo(status.server_time - status.sres.updated_at)}</summary>
      <pre>{status.sres.body}</pre>
    </details>
  )
}
