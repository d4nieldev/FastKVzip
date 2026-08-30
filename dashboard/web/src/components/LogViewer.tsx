import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchLog, logDownloadUrl } from '../lib/api'
import { formatBytes } from '../lib/format'
import { ERROR_PATTERN, toLogLines } from '../lib/logLines'
import type { Job } from '../lib/types'

// Opening tail. Training logs run to hundreds of MB, so the whole file is never
// pulled into the browser -- the newest bytes are what matter.
const INITIAL_TAIL_BYTES = 256 * 1024
const FOLLOW_INTERVAL_MS = 5000
const MAX_RENDERED_LINES = 5000

interface Props {
  job: Job
}

export function LogViewer({ job }: Props) {
  const [text, setText] = useState('')
  const [nextOffset, setNextOffset] = useState(0)
  const [totalSize, setTotalSize] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [follow, setFollow] = useState(!job.is_terminal)
  const [errorsOnly, setErrorsOnly] = useState(false)
  const [showProgress, setShowProgress] = useState(false)

  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Load the tail whenever the selected job changes.
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setText('')
    setFollow(!job.is_terminal)

    fetchLog(job.job_id, { tail: INITIAL_TAIL_BYTES }, controller.signal)
      .then((slice) => {
        setText(slice.text)
        setNextOffset(slice.next_offset)
        setTotalSize(slice.total_size)
      })
      .catch((err: Error) => {
        if (err.name !== 'AbortError') setError(err.message)
      })
      .finally(() => setLoading(false))

    return () => controller.abort()
  }, [job.job_id, job.is_terminal])

  // Follow mode: pull only the bytes appended since the last read.
  useEffect(() => {
    if (!follow) return
    const controller = new AbortController()
    const timer = window.setInterval(() => {
      if (document.hidden) return
      fetchLog(job.job_id, { offset: nextOffset }, controller.signal)
        .then((slice) => {
          setTotalSize(slice.total_size)
          if (!slice.text) return
          setText((current) => current + slice.text)
          setNextOffset(slice.next_offset)
        })
        .catch(() => undefined)
    }, FOLLOW_INTERVAL_MS)

    return () => {
      window.clearInterval(timer)
      controller.abort()
    }
  }, [follow, job.job_id, nextOffset])

  const lines = useMemo(() => {
    let all = toLogLines(text, showProgress)
    if (errorsOnly) all = all.filter((line) => ERROR_PATTERN.test(line.text))
    // Cap what the DOM has to hold; the tail is the interesting end.
    return all.length > MAX_RENDERED_LINES ? all.slice(-MAX_RENDERED_LINES) : all
  }, [text, errorsOnly, showProgress])

  useEffect(() => {
    if (follow) bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [lines, follow])

  const jumpToEnd = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [])

  // How much of the file is actually loaded: everything from the tail's start
  // offset onward.
  const shownBytes = totalSize - Math.max(0, nextOffset - text.length)

  return (
    <div className="log-viewer">
      <div className="log-toolbar">
        <label className={follow ? 'toggle on' : 'toggle'}>
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          Follow
        </label>
        <label className={errorsOnly ? 'toggle on' : 'toggle'}>
          <input
            type="checkbox"
            checked={errorsOnly}
            onChange={(e) => setErrorsOnly(e.target.checked)}
          />
          Errors only
        </label>
        <label
          className={showProgress ? 'toggle on' : 'toggle'}
          title="tqdm rewrites one line thousands of times; collapsed shows each run's final state"
        >
          <input
            type="checkbox"
            checked={showProgress}
            onChange={(e) => setShowProgress(e.target.checked)}
          />
          All progress steps
        </label>
        <span className="spacer" />
        <span className="log-size">{formatBytes(totalSize)}</span>
        <button type="button" onClick={jumpToEnd}>
          End
        </button>
        <a href={logDownloadUrl(job.job_id)} download>
          Download
        </a>
      </div>

      <div className="log-body" ref={scrollRef}>
        {loading && <div className="log-note">Loading log…</div>}
        {error && <div className="log-note error">Could not load log: {error}</div>}
        {!loading && !error && totalSize === 0 && (
          <div className="log-note">
            No log stored yet. The agent ships logs as the job writes them.
          </div>
        )}
        {!loading && totalSize > INITIAL_TAIL_BYTES && (
          <div className="log-note">
            Showing the last {formatBytes(shownBytes)} of {formatBytes(totalSize)} — download for
            the whole file.
          </div>
        )}
        {lines.map((line, index) => (
          <div
            key={index}
            className={ERROR_PATTERN.test(line.text) ? 'log-line alert' : 'log-line'}
          >
            {line.text || ' '}
            {line.collapsed && line.collapsed > 1 ? (
              <span className="collapsed-count"> ← {line.collapsed} updates</span>
            ) : null}
          </div>
        ))}
        {errorsOnly && lines.length === 0 && !loading && (
          <div className="log-note">No error-like lines in the loaded portion.</div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
