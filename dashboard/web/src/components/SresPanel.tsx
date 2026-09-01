import { useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import { formatAgo } from '../lib/format'
import { availabilityColor, parseSres } from '../lib/sres'
import type { Status } from '../lib/types'

/** Matches SRES_EVERY_N_POLLS in the agent. */
const SRES_EVERY_N_POLLS = 10

function tint(score: number): CSSProperties {
  return { '--avail': availabilityColor(score) } as CSSProperties
}

/**
 * The `sres` snapshot as a table, each node tinted by how free it is.
 *
 * Colour is the fast scan, never the answer: every row keeps its own
 * `free/total` counts as text and states its score as a number and a bar
 * length, so the ranking survives red-green colour blindness, greyscale, and a
 * format this parser turns out not to understand.
 */
export function SresPanel({ status }: { status: Status | null }) {
  const body = status?.sres?.body
  const [query, setQuery] = useState('')
  const [raw, setRaw] = useState(false)

  const view = useMemo(() => (body ? parseSres(body) : null), [body])

  const needle = query.trim().toLowerCase()

  // One substring over the whole row, so "6000" finds the 6000 and 6000pro
  // totals and every cs-6000 node without anyone having to know which column
  // holds which.
  const rows = useMemo(() => {
    const all = view?.table?.rows ?? []
    return needle ? all.filter((row) => row.haystack.includes(needle)) : all
  }, [view, needle])

  const totals = useMemo(() => {
    const all = view?.totals ?? []
    return needle ? all.filter((entry) => entry.label.toLowerCase().includes(needle)) : all
  }, [view, needle])

  if (!status) return null

  if (!body) {
    return (
      <details className="sres">
        <summary>GPU availability (sres) · never reported</summary>
        <p className="sres-empty">
          The agent has not sent an <code>sres</code> snapshot. It collects one every{' '}
          {SRES_EVERY_N_POLLS} polls, so allow a few minutes after it starts; if this
          persists, <code>sres</code> could not be run where the agent is.
        </p>
      </details>
    )
  }

  const age = formatAgo(status.server_time - (status.sres?.updated_at ?? status.server_time))
  const table = view?.table ?? null

  return (
    <details className="sres">
      <summary>GPU availability (sres) · {age}</summary>

      {!view || raw ? (
        <>
          {view && (
            <button type="button" className="sres-raw-toggle" onClick={() => setRaw(false)}>
              Back to table
            </button>
          )}
          <pre>{body}</pre>
        </>
      ) : (
        <div className="sres-body">
          <div className="sres-controls">
            <input
              type="search"
              className="sres-search"
              placeholder="Filter — try 6000"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            {table && (
              <span className="sres-count">
                {rows.length === table.rows.length
                  ? `${table.rows.length} nodes`
                  : `${rows.length} of ${table.rows.length} nodes`}
              </span>
            )}
            <button type="button" className="sres-raw-toggle" onClick={() => setRaw(true)}>
              Raw
            </button>
          </div>

          {/* The node table counts GPUs per node without saying whether they are
              6000pro or 6000, so this block is the only thing that answers the
              question actually being asked before a submission. */}
          {!!totals.length && (
            <div className="sres-totals">
              {totals.map((entry) => (
                <span key={entry.label} className="sres-total" style={tint(entry.value)}>
                  <span className="sres-total-label">{entry.label}</span>
                  <span className="sres-total-count">
                    {entry.free}/{entry.total}
                  </span>
                  <span className="sres-total-pct">{Math.round(entry.value * 100)}%</span>
                </span>
              ))}
            </div>
          )}

          {table && (
            <div className="sres-scroll">
              <table className="sres-table">
                <thead>
                  <tr>
                    <th className="sres-score-head">Avail</th>
                    {table.headers.map((header, index) => (
                      <th key={index}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, index) => {
                    const score = row.score
                    return (
                      <tr
                        key={index}
                        className={score === null ? '' : 'scored'}
                        style={score === null ? undefined : tint(score)}
                      >
                        <td className="sres-score">
                          <span className="sres-score-inner">
                            {score === null ? (
                              <span className="sres-percent">—</span>
                            ) : (
                              <>
                                <span className="sres-bar">
                                  <span style={{ width: `${score * 100}%` }} />
                                </span>
                                <span className="sres-percent">{Math.round(score * 100)}%</span>
                              </>
                            )}
                          </span>
                        </td>
                        {table.headers.map((_, column) => (
                          <td
                            key={column}
                            className={row.resources.has(column) ? 'num' : undefined}
                          >
                            {row.cells[column] ?? ''}
                          </td>
                        ))}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          {!rows.length && !totals.length && (
            <p className="sres-empty">Nothing matches “{query}”.</p>
          )}
        </div>
      )}
    </details>
  )
}
