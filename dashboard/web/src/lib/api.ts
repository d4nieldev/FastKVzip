import type { Job, LogSlice, Status } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

export function fetchStatus(signal?: AbortSignal): Promise<Status> {
  return request<Status>('/api/status', { signal })
}

export interface JobQuery {
  from?: number
  to?: number
  states?: string[]
  q?: string
}

export function fetchJobs(query: JobQuery, signal?: AbortSignal): Promise<{ jobs: Job[] }> {
  const params = new URLSearchParams()
  if (query.from !== undefined) params.set('from', String(Math.floor(query.from)))
  if (query.to !== undefined) params.set('to', String(Math.floor(query.to)))
  if (query.states?.length) params.set('states', query.states.join(','))
  if (query.q) params.set('q', query.q)
  return request<{ jobs: Job[] }>(`/api/jobs?${params}`, { signal })
}

export function fetchJob(jobId: string, signal?: AbortSignal): Promise<Job> {
  return request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal })
}

/** Read a slice of a job's log: the tail on open, then forward from an offset. */
export function fetchLog(
  jobId: string,
  options: { tail?: number; offset?: number },
  signal?: AbortSignal,
): Promise<LogSlice> {
  const params = new URLSearchParams()
  if (options.tail !== undefined) params.set('tail', String(options.tail))
  if (options.offset !== undefined) params.set('offset', String(options.offset))
  return request<LogSlice>(`/api/jobs/${encodeURIComponent(jobId)}/log?${params}`, { signal })
}

export function logDownloadUrl(jobId: string): string {
  return `/api/jobs/${encodeURIComponent(jobId)}/log/download`
}

/** Note that the user has read this job, so a finished run stops glowing. */
export function markSeen(jobId: string): Promise<unknown> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/seen`, { method: 'POST' })
}
