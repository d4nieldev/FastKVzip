import { useState } from 'react'
import { formatAgo, stateClass } from '../lib/format'
import type { Project } from '../lib/types'

interface Props {
  projects: Project[]
  serverTime: number
  onOpen: (projectId: string) => void
  onCreate: (name: string) => void
}

/**
 * The other cut through the same jobs.
 *
 * A user is who ran something; a project is what was being run. One experiment
 * is often several people's jobs submitted hours apart, which the user cut can
 * only ever show separately.
 */
export function ProjectPicker({ projects, serverTime, onOpen, onCreate }: Props) {
  const [name, setName] = useState('')

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
                    <span className="user-name">{project.name}</span>
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
                        <span key={state} className={`group-pill ${stateClass(state)}`}>
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
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
