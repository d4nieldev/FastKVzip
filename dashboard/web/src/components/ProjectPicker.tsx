import { useState } from 'react'
import type { CSSProperties } from 'react'
import { ColorPicker } from './ColorPicker'
import { formatAgo, stateClass } from '../lib/format'
import type { Project } from '../lib/types'

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
                    <span className="user-agent">
                      {project.last_activity
                        ? formatAgo(serverTime - project.last_activity)
                        : 'no jobs yet'}
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
                      <span className="user-empty">nothing filed here yet</span>
                    )}
                  </div>

                  {/* Named because a project spanning people is the reason it
                      exists rather than an edge case. */}
                  {project.users.length > 0 && (
                    <div className="user-host">
                      {project.job_count} jobs · {project.users.join(', ')}
                    </div>
                  )}
                </button>

                <div className="card-foot" onClick={(event) => event.stopPropagation()}>
                  <ColorPicker
                    color={project.color}
                    label={project.name}
                    onPick={(color) => onRecolor(project.id, color)}
                  />
                  {confirming === project.id ? (
                    <span className="confirm-delete">
                      {/* Says what survives, because the word "delete" beside a
                          count of jobs reads like it takes them with it. */}
                      <span className="confirm-note">
                        {project.job_count
                          ? `${project.job_count} job${project.job_count === 1 ? '' : 's'} stay`
                          : 'nothing filed here'}
                      </span>
                      <button
                        type="button"
                        className="linklike danger"
                        onClick={() => {
                          onDelete(project.id)
                          setConfirming(null)
                        }}
                      >
                        delete
                      </button>
                      <button
                        type="button"
                        className="linklike"
                        onClick={() => setConfirming(null)}
                      >
                        cancel
                      </button>
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="linklike delete-project"
                      onClick={() => setConfirming(project.id)}
                    >
                      delete
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
