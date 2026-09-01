import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AgentBanner, Controls } from './components/Controls'
import { SresPanel } from './components/SresPanel'
import { JobDetail } from './components/JobDetail'
import { JobList } from './components/JobList'
import { fetchJobs, fetchStatus, setHidden } from './lib/api'
import type { Job, Status } from './lib/types'

const REFRESH_INTERVAL_MS = 10_000
const DEFAULT_WINDOW_SECONDS = 24 * 3600

/** Filter state lives in the URL so a view can be bookmarked or shared. */
interface View {
  windowSeconds: number
  states: string[]
  search: string
  includeHidden: boolean
  selected: string | null
}

function readView(): View {
  const params = new URLSearchParams(window.location.search)
  return {
    windowSeconds: Number(params.get('window')) || DEFAULT_WINDOW_SECONDS,
    states: (params.get('states') ?? '').split(',').filter(Boolean),
    search: params.get('q') ?? '',
    includeHidden: params.get('hidden') === '1',
    selected: params.get('job'),
  }
}

function writeView(view: View) {
  const params = new URLSearchParams()
  if (view.windowSeconds !== DEFAULT_WINDOW_SECONDS) params.set('window', String(view.windowSeconds))
  if (view.states.length) params.set('states', view.states.join(','))
  if (view.search) params.set('q', view.search)
  if (view.includeHidden) params.set('hidden', '1')
  if (view.selected) params.set('job', view.selected)
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
        fetchJobs(
          {
            from: Math.floor(Date.now() / 1000) - current.windowSeconds,
            to: Math.floor(Date.now() / 1000),
            states: current.states,
            q: current.search,
            includeHidden: current.includeHidden,
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
  }, [refresh, view.windowSeconds, view.states, view.search, view.includeHidden])

  // Drives the live elapsed-time counters between polls.
  useEffect(() => {
    const timer = window.setInterval(() => setNowEpoch(Math.floor(Date.now() / 1000)), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const selectedJob = useMemo(
    () => jobs.find((job) => job.job_id === view.selected) ?? null,
    [jobs, view.selected],
  )

  const toggleHidden = useCallback(
    async (job: Job) => {
      // Optimistic: the button should feel instant, and the next poll is the
      // source of truth if the request fails.
      setJobs((current) =>
        current.map((item) =>
          item.job_id === job.job_id ? { ...item, hidden: !item.hidden } : item,
        ),
      )
      try {
        await setHidden(job.job_id, !job.hidden)
      } catch (err) {
        setError((err as Error).message)
      }
      void refresh()
    },
    [refresh],
  )

  return (
    <div className="app">
      <header className="app-header">
        <h1>SLURM jobs</h1>
        <AgentBanner
          status={status}
          error={error}
          nowEpoch={nowEpoch}
          fetchedAt={statusFetchedAt}
        />
      </header>

      <Controls
        windowSeconds={view.windowSeconds}
        onWindowChange={(seconds) => update({ windowSeconds: seconds })}
        states={view.states}
        onStatesChange={(states) => update({ states })}
        search={view.search}
        onSearchChange={(search) => update({ search })}
        includeHidden={view.includeHidden}
        onIncludeHiddenChange={(includeHidden) => update({ includeHidden })}
        counts={status?.state_counts ?? {}}
        hiddenCount={status?.hidden_count ?? 0}
      />

      <SresPanel status={status} />

      <main className={selectedJob ? 'main with-detail' : 'main'}>
        <div className="list-pane">
          {!loaded ? (
            <p className="empty">Loading…</p>
          ) : (
            <JobList
              jobs={jobs}
              selectedId={view.selected}
              nowEpoch={nowEpoch}
              onSelect={(job) => update({ selected: job.job_id })}
              onToggleHidden={toggleHidden}
            />
          )}
        </div>

        {selectedJob && (
          <div className="detail-pane">
            <JobDetail
              job={selectedJob}
              nowEpoch={nowEpoch}
              onClose={() => update({ selected: null })}
              onToggleHidden={toggleHidden}
            />
          </div>
        )}
      </main>
    </div>
  )
}
