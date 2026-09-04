import { formatAgo, liveSince, stateClass } from '../lib/format'
import type { UserSummary } from '../lib/types'

interface Props {
  users: UserSummary[]
  nowEpoch: number
  fetchedAt: number
  /** Open one user's dashboard. */
  onOpen: (user: string) => void
  /** Open several at once, with each job labelled by whose it is. */
  onOpenMany: (users: string[]) => void
  selected: string[]
  onToggle: (user: string) => void
}

/** Live, stale or never heard from -- the same judgement the banner makes. */
function agentState(user: UserSummary, since: number | null) {
  if (since === null) return { label: 'no agent', tone: 'down' as const }
  const interval = user.poll_interval ?? 30
  if (since > Math.max(120, interval * 3)) return { label: `stale · ${formatAgo(since)}`, tone: 'down' as const }
  return { label: `reported ${formatAgo(since)}`, tone: 'live' as const }
}

/**
 * The way in: who is reporting, and how their work is going.
 *
 * Several people can run an agent against the same server, so the dashboard
 * opens on the roster rather than on one person's jobs. Picking one is the
 * common case; ticking several answers "how is the group doing" in one list.
 */
export function UserPicker({
  users,
  nowEpoch,
  fetchedAt,
  onOpen,
  onOpenMany,
  selected,
  onToggle,
}: Props) {
  if (!users.length) {
    return (
      <p className="empty">
        No agent has reported yet. Start one with{' '}
        <code>sbatch dashboard/agent/dashboard_agent.sbatch</code> on the cluster.
      </p>
    )
  }

  return (
    <div className="users">
      <div className="users-head">
        <h2>Users</h2>
        {selected.length > 0 && (
          <button type="button" className="chip on" onClick={() => onOpenMany(selected)}>
            Open {selected.length} together
          </button>
        )}
      </div>

      <ul className="user-list">
        {users.map((user) => {
          const since = liveSince(user.seconds_since_heartbeat, fetchedAt, nowEpoch)
          const agent = agentState(user, since)
          const states = Object.entries(user.state_counts).sort((a, b) => b[1] - a[1])
          const checked = selected.includes(user.user)
          return (
            <li key={user.user}>
              <button
                type="button"
                className={checked ? 'user-card checked' : 'user-card'}
                onClick={() => onOpen(user.user)}
              >
                <div className="user-top">
                  <span className="user-name">{user.user}</span>
                  {user.unseen_count > 0 && (
                    <span className="user-unread">{user.unseen_count} unread</span>
                  )}
                  <span className={`user-agent ${agent.tone}`}>
                    <span className="dot" />
                    {agent.label}
                  </span>
                </div>

                <div className="user-states">
                  {states.length ? (
                    states.map(([state, count]) => (
                      <span key={state} className={`state-pill ${stateClass(state)}`}>
                        {count} {state.toLowerCase()}
                      </span>
                    ))
                  ) : (
                    <span className="user-empty">no jobs on file</span>
                  )}
                </div>

                {user.host && <div className="user-host">agent on {user.host}</div>}
              </button>

              {/* Separate from the card, so picking several never fights with
                  opening one. */}
              <label className="user-pick" onClick={(event) => event.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onToggle(user.user)}
                  aria-label={`Include ${user.user}`}
                />
                compare
              </label>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
