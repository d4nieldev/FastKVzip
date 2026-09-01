/** Formatting helpers shared by the list and detail views. */

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—'
  const total = Math.max(0, Math.floor(seconds))
  const days = Math.floor(total / 86400)
  const hours = Math.floor((total % 86400) / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const pad = (value: number) => String(value).padStart(2, '0')
  if (days > 0) return `${days}d ${pad(hours)}:${pad(minutes)}:${pad(secs)}`
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}`
}

/** Epochs arrive in UTC; render them in the viewer's own timezone. */
export function formatTime(epoch: number | null | undefined): string {
  if (!epoch) return '—'
  return new Date(epoch * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function formatAgo(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return 'never'
  if (seconds < 60) return `${Math.floor(seconds)}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}

const STATE_CLASS: Record<string, string> = {
  RUNNING: 'running',
  PENDING: 'pending',
  COMPLETED: 'completed',
  FAILED: 'failed',
  TIMEOUT: 'failed',
  OUT_OF_MEMORY: 'failed',
  NODE_FAIL: 'failed',
  BOOT_FAIL: 'failed',
  DEADLINE: 'failed',
  CANCELLED: 'cancelled',
  SUSPENDED: 'pending',
  CONFIGURING: 'pending',
  COMPLETING: 'running',
}

export function stateClass(state: string): string {
  return STATE_CLASS[state] ?? 'unknown'
}

/**
 * Seconds since the agent's last heartbeat, ticking between polls.
 *
 * Only the delta since the response landed is added, both ends measured on the
 * browser's clock. The gap itself stays the server's number, so a browser whose
 * clock is minutes off the cluster's cannot skew the count -- or drive it
 * negative and flip the banner to a false "stale".
 */
export function liveSince(
  seconds: number | null | undefined,
  fetchedAt: number,
  nowEpoch: number,
): number | null {
  if (seconds === null || seconds === undefined) return null
  return seconds + Math.max(0, nowEpoch - fetchedAt)
}

/**
 * Live elapsed time for a job.
 *
 * The agent's elapsed_s is only as fresh as its last poll, so for a job that is
 * actually on a node the seconds since that poll are added back -- otherwise
 * the clock visibly stutters between polls. Only RUNNING extrapolates: a queued
 * job has not started, so its elapsed time must stay pinned at zero.
 */
export function liveElapsed(
  elapsed: number | null,
  lastSeen: number,
  state: string,
  nowEpoch: number,
): number | null {
  if (elapsed === null) return null
  if (state !== 'RUNNING' && state !== 'COMPLETING') return elapsed
  return elapsed + Math.max(0, nowEpoch - lastSeen)
}
