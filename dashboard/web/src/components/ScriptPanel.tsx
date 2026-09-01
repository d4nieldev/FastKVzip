import { useEffect, useState } from 'react'
import { fetchScript } from '../lib/api'
import type { Job, JobScript } from '../lib/types'

/** What each source means, said plainly rather than left as a bare word. */
const SCRIPT_SOURCE: Record<string, string> = {
  scontrol: 'exactly as submitted, from the controller',
  sacct: 'exactly as submitted, from accounting',
  disk: 'the file at this path as it stands now — the repository may have moved on since',
}

const ENV_SOURCE: Record<string, string> = {
  sacct: 'the environment the job was submitted from',
}

/**
 * The sbatch script a job was submitted with, and the environment it came from.
 *
 * Loaded when the panel is opened rather than with the job: a script is read
 * once by one person, and the job list is polled every ten seconds.
 */
export function ScriptPanel({ job }: { job: Job }) {
  const [open, setOpen] = useState(false)
  const [record, setRecord] = useState<JobScript | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setRecord(null)
    setError(null)
  }, [job.job_id])

  // Deps are only what should restart the fetch. `loading` must not be one:
  // setLoading(true) would re-run the effect, whose cleanup aborts the request
  // it had just started, and the retry loops forever on a panel that never
  // stops saying "Loading".
  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    fetchScript(job.job_id, controller.signal)
      .then(setRecord)
      .catch((err: Error) => {
        if (err.name !== 'AbortError') setError(err.message)
      })
      .finally(() => setLoading(false))
    return () => controller.abort()
  }, [open, job.job_id])

  const script = record?.batch_script
  const env = record?.job_env

  return (
    <details className="script-panel" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>Submitted script &amp; environment</summary>

      {loading && <p className="script-note">Loading…</p>}
      {error && <p className="script-note error">Could not load: {error}</p>}

      {record && !loading && (
        <>
          <h4>
            sbatch script
            {script && record.script_source && (
              <span className="script-source">
                {SCRIPT_SOURCE[record.script_source] ?? record.script_source}
              </span>
            )}
          </h4>
          {script ? <pre>{script}</pre> : <p className="script-note">Not stored.</p>}

          <h4>
            Submission environment
            {env && record.env_source && (
              <span className="script-source">
                {ENV_SOURCE[record.env_source] ?? record.env_source}
              </span>
            )}
          </h4>
          {env ? <pre>{env}</pre> : <p className="script-note">Not stored.</p>}

          {/* Why it is missing is more useful than the fact that it is: both
              sources depend on cluster configuration the agent cannot change. */}
          {record.note && !(script && env) && <p className="script-note">{record.note}</p>}

          {!record.note && !script && !env && (
            <p className="script-note">
              The agent has not collected this job yet. It fetches a few per poll, oldest
              request first.
            </p>
          )}
        </>
      )}
    </details>
  )
}
