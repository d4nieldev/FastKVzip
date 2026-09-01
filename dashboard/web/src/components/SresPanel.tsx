import { useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import { formatAgo } from '../lib/format'
import { availabilityColor, parseSres } from '../lib/sres'
import type { Status } from '../lib/types'

/**
 * The `sres` snapshot as a table, each row tinted by how free that node is.
 *
 * Colour is the fast scan, never the answer: every row keeps its own
 * `free/total` counts as text and states its score as a number, so the ranking
 * survives red-green colour blindness, greyscale, and a format this parser
 * turns out not to understand.
 */
export function SresPanel({ status }: { status: Status | null }) {
  const body = status?.sres?.body
  const [query, setQuery] = useState('')
  const [raw, setRaw] = useState(false)

  const table = useMemo(() => (body ? parseSres(body) : null), [body])

  const rows = useMemo(() => {
    if (!table) return []
    const needle = query.trim().toLowerCase()
    if (!needle) return table.rows
    // One substring over the whole row, so "6000" finds rtx_6000 and
    // rtx_pro_6000 alike without anyone having to know which column holds it.
    return table.rows.filter((row) => row.haystack.includes(needle))
  }, [table, query])

  if (!body || !status) return null

  const age = formatAgo(status.server_time - (status.sres?.updated_at ?? status.server_time))

  return (
    <details className="sres">
      <summary>GPU availability (sres) · {age}</summary>

      {!table || raw ? (
        <>
          {table && (
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
              placeholder="Filter nodes — try 6000"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <span className="sres-count">
              {rows.length === table.rows.length
                ? `${table.rows.length} nodes`
                : `${rows.length} of ${table.rows.length} nodes`}
            </span>
            <button type="button" className="sres-raw-toggle" onClick={() => setRaw(true)}>
              Raw
            </button>
          </div>

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
                  const tint =
                    score === null
                      ? undefined
                      : ({ '--avail': availabilityColor(score) } as CSSProperties)
                  return (
                    <tr key={index} className={score === null ? '' : 'scored'} style={tint}>
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
                        <td key={column} className={row.resources.has(column) ? 'num' : undefined}>
                          {row.cells[column] ?? ''}
                        </td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {!rows.length && <p className="sres-empty">No node matches “{query}”.</p>}
        </div>
      )}
    </details>
  )
}
