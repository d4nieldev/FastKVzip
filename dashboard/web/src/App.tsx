import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AgentBanner, Controls } from './components/Controls'
import { SresPanel } from './components/SresPanel'
import { UserPicker } from './components/UserPicker'
import { JobDetail } from './components/JobDetail'
import { JobList } from './components/JobList'
import { fetchJob, fetchJobs, fetchStatus, markSeen, markSeenMany } from './lib/api'
import type { Job, Status } from './lib/types'

const REFRESH_INTERVAL_MS = 10_000
const DEFAULT_WINDOW_SECONDS = 24 * 3600

/** Filter state lives in the URL so a view can be bookmarked or shared. */
interface View {
  windowSeconds: number
  states: string[]
  search: string
  selected: string | null
  /** Whose jobs to show. Empty means nobody is chosen yet: the roster. */
  users: string[]
}

function readView(): View {
  const params = new URLSearchParams(window.location.search)
  return {
    windowSeconds: Number(params.get('window')) || DEFAULT_WINDOW_SECONDS,
    states: (params.get('states') ?? '').split(',').filter(Boolean),
    search: params.get('q') ?? '',
    selected: params.get('job'),
    users: (params.get('users') ?? '').split(',').filter(Boolean),
  }
}

function writeView(view: View) {
  const params = new URLSearchParams()
  if (view.windowSeconds !== DEFAULT_WINDOW_SECONDS) params.set('window', String(view.windowSeconds))
  if (view.states.length) params.set('states', view.states.join(','))
  if (view.search) params.set('q', view.search)
  if (view.selected) params.set('job', view.selected)
  if (view.users.length) params.set('users', view.users.join(','))
  const query = params.toString()
  window.history.replaceState(null, '', query ? `?${query}` : window.location.pathname)
}

export function App() {
  const [view, setView] = useState<View>(readView)
  const [jobs, setJobs] = useState<Job[]>([])
  const [status, setStatus] = useState<Status | null>(null)
  const [statusFetchedAt, setStatusFetchedAt] = useState(() => Math.floor(Date.now() / 1000))
  const [error, setError] = useState<string | null>(null)
  const [nowEpoch, setNowEpoch] = useState(() => Math.floor(Date.now() / 1000))
  const [loaded, setLoaded] = useState(false)

  // The poller reads these without re-subscribing, so changing a filter does
  // not tear down and rebuild the interval.
  const viewRef = useRef(view)
  viewRef.current = view

  const update = useCallback((patch: Partial<View>) => {
    setView((current) => {
      const next = { ...current, ...patch }
      writeView(next)
      return next
    })
  }, [])

  const refresh = useCallback(async (signal?: AbortSignal) => {
    const current = viewRef.current
    try {
      const [statusResult, jobsResult] = await Promise.all([
        fetchStatus(signal),
        !current.users.length
          ? Promise.resolve({ jobs: [] })
          : fetchJobs(
          {
            from: Math.floor(Date.now() / 1000) - current.windowSeconds,
            to: Math.floor(Date.now() / 1000),
            states: current.states,
            q: current.search,
            users: current.users,
          },
          signal,
        ),
      ])
      setStatus(statusResult)
      setStatusFetchedAt(Math.floor(Date.now() / 1000))
      setJobs(jobsResult.jobs)
      setError(null)
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setError((err as Error).message)
    } finally {
      setLoaded(true)
    }
  }, [])

  // Refetch on any filter change, and on an interval while the tab is visible.
  useEffect(() => {
    const controller = new AbortController()
    void refresh(controller.signal)

    const timer = window.setInterval(() => {
      if (!document.hidden) void refresh(controller.signal)
    }, REFRESH_INTERVAL_MS)
    const onVisible = () => {
      if (!document.hidden) void refresh(controller.signal)
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      window.clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
      controller.abort()
    }
  }, [refresh, view.windowSeconds, view.states, view.search, view.users])

  // Drives the live elapsed-time counters between polls.
  useEffect(() => {
    const timer = window.setInterval(() => setNowEpoch(Math.floor(Date.now() / 1000)), 1000)
    return () => window.clearInterval(timer)
  }, [])

  // The agent's own job is kept out of the list, so a selection can point at
  // something the list does not carry; it is fetched on its own rather than
  // being unopenable.
  const [detachedJob, setDetachedJob] = useState<Job | null>(null)
  const inList = jobs.some((job) => job.job_id === view.selected)

  useEffect(() => {
    if (!view.selected || inList) {
      setDetachedJob(null)
      return
    }
    const controller = new AbortController()
    fetchJob(view.selected, controller.signal)
      .then(setDetachedJob)
      .catch(() => undefined)
    return () => controller.abort()
  }, [view.selected, inList, nowEpoch])

  const selectedJob = useMemo(
    () =>
      jobs.find((job) => job.job_id === view.selected) ??
      (detachedJob?.job_id === view.selected ? detachedJob : null),
    [jobs, view.selected, detachedJob],
  )

  // Only what is on screen. A run outside the current window or filter has not
  // been shown, so a click here has no business marking it read.
  const unseen = useMemo(() => jobs.filter((job) => job.unseen), [jobs])

  // Ticked on the roster, not yet opened. Kept out of the URL because it is a
  // half-made choice, unlike `users`, which is the view itself.
  const [compare, setCompare] = useState<string[]>([])

  // The state chips count only the users being shown, so the numbers agree
  // with the list beneath them.
  const counts = useMemo(() => {
    const chosen = view.users
    if (!status) return {}
    if (!chosen.length) return status.state_counts
    const totals: Record<string, number> = {}
    for (const entry of status.users) {
      if (!chosen.includes(entry.user)) continue
      for (const [state, n] of Object.entries(entry.state_counts)) {
        totals[state] = (totals[state] ?? 0) + n
      }
    }
    return totals
  }, [status, view.users])

  const markAllRead = useCallback(async () => {
    const ids = unseen.map((job) => job.job_id)
    if (!ids.length) return
    setJobs((current) => current.map((job) => ({ ...job, unseen: false })))
    try {
      await markSeenMany(ids)
    } catch (err) {
      setError((err as Error).message)
    }
    void refresh()
  }, [unseen, refresh])

  // Opening a finished run is what marks it read, so the glow clears on the
  // click that shows the outcome rather than needing a second gesture.
  const select = useCallback((job: Job) => {
    update({ selected: job.job_id })
    if (!job.unseen) return
    setJobs((current) =>
      current.map((item) => (item.job_id === job.job_id ? { ...item, unseen: false } : item)),
    )
    void markSeen(job.job_id).catch(() => undefined)
  }, [update])

  const roster = status?.users ?? []
  const chosen = view.users
  const showingSeveral = chosen.length > 1

  // Whoever is chosen, as the server described them; the banner reads from
  // these rather than from a single agent that no longer exists.
  const chosenAgents = useMemo(
    () => roster.filter((entry) => chosen.includes(entry.user)),
    [roster, chosen],
  )

  if (!chosen.length) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>SLURM jobs</h1>
          {error && <div className="banner down">Cannot reach the dashboard server — {error}</div>}
        </header>

        {!loaded ? (
          <p className="empty">Loading…</p>
        ) : (
          <UserPicker
            users={roster}
            nowEpoch={nowEpoch}
            fetchedAt={statusFetchedAt}
            onOpen={(user) => update({ users: [user], selected: null })}
            onOpenMany={(users) => update({ users, selected: null })}
            selected={compare}
            onToggle={(user) =>
              setCompare((current) =>
                current.includes(user)
                  ? current.filter((name) => name !== user)
                  : [...current, user],
              )
            }
          />
        )}

        <SresPanel status={status} />
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          <button
            type="button"
            className="linklike back"
            onClick={() => update({ users: [], selected: null })}
          >
            ← all users
          </button>
          <h1>{chosen.join(', ')}</h1>
        </div>
        <AgentBanner
          agents={chosenAgents}
          error={error}
          nowEpoch={nowEpoch}
          fetchedAt={statusFetchedAt}
          onSelectJob={(jobId) => update({ selected: jobId })}
        />
      </header>

      <Controls
        windowSeconds={view.windowSeconds}
        onWindowChange={(seconds) => update({ windowSeconds: seconds })}
        states={view.states}
        onStatesChange={(states) => update({ states })}
        search={view.search}
        onSearchChange={(search) => update({ search })}
        counts={counts}
      />

      <SresPanel status={status} />

      <main className={selectedJob ? 'main with-detail' : 'main'}>
        <div className="list-pane">
          {unseen.length > 0 && (
            <div className="list-actions">
              <span className="unread-note">
                {unseen.length} finished {unseen.length === 1 ? 'run' : 'runs'} you have not read
              </span>
              <button type="button" className="chip mark-read" onClick={markAllRead}>
                Mark all read
              </button>
            </div>
          )}
          {!loaded ? (
            <p className="empty">Loading…</p>
          ) : (
            <JobList
              jobs={jobs}
              selectedId={view.selected}
              nowEpoch={nowEpoch}
              onSelect={select}
              showUser={showingSeveral}
            />
          )}
        </div>

        {selectedJob && (
          <div className="detail-pane">
            <JobDetail
              job={selectedJob}
              nowEpoch={nowEpoch}
              onClose={() => update({ selected: null })}
            />
          </div>
        )}
      </main>
    </div>
  )
}
