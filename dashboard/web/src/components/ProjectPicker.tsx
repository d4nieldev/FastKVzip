import { useState } from 'react'
import type { CSSProperties } from 'react'
import { ColorPicker } from './ColorPicker'
import { formatAgo, stateClass } from '../lib/format'
import type { Project } from '../lib/types'

function TrashIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true" focusable="false">
      <path
        fill="none"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M2.5 4h11M6.5 4V2.5h3V4M4 4l.7 9a1 1 0 0 0 1 .9h4.6a1 1 0 0 0 1-.9L12 4M6.6 6.8v4.4M9.4 6.8v4.4"
      />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true" focusable="false">
      <path
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3.2 8.6l3.1 3.1 6.5-6.9"
      />
    </svg>
  )
}

function CrossIcon() {
  return (
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true" focusable="false">
      <path
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        d="M4 4l8 8M12 4l-8 8"
      />
    </svg>
  )
}

interface Props {
  projects: Project[]
  serverTime: number
  onOpen: (projectId: string) => void
  onCreate: (name: string) => void
  onRecolor: (projectId: string, color: string) => void
  onDelete: (projectId: string) => void
}

/**
 * The other cut through the same jobs.
 *
 * A user is who ran something; a project is what was being run. One experiment
 * is often several people's jobs submitted hours apart, which the user cut can
 * only ever show separately.
 */
export function ProjectPicker({
  projects,
  serverTime,
  onOpen,
  onCreate,
  onRecolor,
  onDelete,
}: Props) {
  const [name, setName] = useState('')
  // Which project is one more click from being deleted. Asking in place rather
  // than in a browser dialog, and only ever for one at a time.
  const [confirming, setConfirming] = useState<string | null>(null)

  const submit = (event: React.FormEvent) => {
    event.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    onCreate(trimmed)
    setName('')
  }

  return (
    <div className="projects">
      <div className="users-head">
        <h2>Projects</h2>
        <form className="project-new" onSubmit={submit}>
          <input
            type="text"
            placeholder="New project name"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <button type="submit" className="chip" disabled={!name.trim()}>
            Create
          </button>
        </form>
      </div>

      {!projects.length ? (
        <p className="user-empty project-none">
          No projects yet. Create one above, or have whatever submits your jobs{' '}
          <code>POST /api/projects/&lt;id&gt;/jobs</code> right after sbatch.
        </p>
      ) : (
        <ul className="user-list">
          {projects.map((project) => {
            const states = Object.entries(project.state_counts).sort((a, b) => b[1] - a[1])
            return (
              <li key={project.id}>
                <button
                  type="button"
                  className="user-card"
                  onClick={() => onOpen(project.id)}
                >
                  <div className="user-top">
                    <span
                      className="user-name owner-name"
                      style={{ '--tag': project.color ?? 'var(--muted)' } as CSSProperties}
                    >
                      {project.name}
                    </span>
                    {project.unseen_count > 0 && (
                      <span className="user-unread">{project.unseen_count} unread</span>
                    )}
                  </div>

                  <div className="user-states">
                    {states.length ? (
                      states.map(([state, count]) => (
                        <span key={state} className={`state-pill ${stateClass(state)}`}>
                          {count} {state.toLowerCase()}
                        </span>
                      ))
                    ) : null}
                  </div>

                  {/* Named because a project spanning people is the reason it
                      exists rather than an edge case. */}
                  <div className="user-host">
                    {project.users.length > 0
                      ? `${project.job_count} jobs · ${project.users.join(', ')}`
                      : 'nothing filed here yet'}
                    {project.last_activity
                      ? ` · ${formatAgo(serverTime - project.last_activity)}`
                      : ''}
                  </div>
                </button>

                <div className="card-foot" onClick={(event) => event.stopPropagation()}>
                  <ColorPicker
                    color={project.color}
                    label={project.name}
                    onPick={(color) => onRecolor(project.id, color)}
                  />
                </div>

                {/* Top right, out of the card's own click target. Asks once,
                    because the grouping cannot be got back. */}
                <div className="card-delete" onClick={(event) => event.stopPropagation()}>
                  {confirming === project.id ? (
                    <span className="confirm-delete">
                      {/* Only the two buttons: the corner is too narrow for a
                          sentence, and putting one there landed it on the
                          project's own name. The reassurance is on the
                          tooltip, where it does not have to fit. */}
                      <button
                        type="button"
                        className="icon-button danger"
                        title={
                          project.job_count
                            ? `Delete — its ${project.job_count} job${
                                project.job_count === 1 ? '' : 's'
                              } stay`
                            : 'Delete — nothing is filed here'
                        }
                        aria-label={`Confirm deleting ${project.name}`}
                        onClick={() => {
                          onDelete(project.id)
                          setConfirming(null)
                        }}
                      >
                        <CheckIcon />
                      </button>
                      <button
                        type="button"
                        className="icon-button"
                        title="Keep it"
                        aria-label="Cancel"
                        onClick={() => setConfirming(null)}
                      >
                        <CrossIcon />
                      </button>
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="icon-button trash"
                      title={`Delete ${project.name} — its jobs stay`}
                      aria-label={`Delete project ${project.name}`}
                      onClick={() => setConfirming(project.id)}
                    >
                      <TrashIcon />
                    </button>
                  )}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
