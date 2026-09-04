import type { Job, LogSlice, Project, Status } from './types'

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
  users?: string[]
  project?: string | null
}

export function fetchJobs(query: JobQuery, signal?: AbortSignal): Promise<{ jobs: Job[] }> {
  const params = new URLSearchParams()
  if (query.from !== undefined) params.set('from', String(Math.floor(query.from)))
  if (query.to !== undefined) params.set('to', String(Math.floor(query.to)))
  if (query.states?.length) params.set('states', query.states.join(','))
  if (query.q) params.set('q', query.q)
  if (query.users?.length) params.set('users', query.users.join(','))
  if (query.project) params.set('project', query.project)
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

/** Create a project, or get back the one that already has this id. */
export function createProject(name: string, id?: string): Promise<Project> {
  return request<Project>('/api/projects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, id }),
  })
}

/** File jobs under a project; `null` takes them out of whichever they are in. */
export function assignJobs(projectId: string | null, jobIds: string[]): Promise<unknown> {
  if (projectId === null) {
    return request('/api/projects/none/jobs', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_ids: jobIds }),
    })
  }
  return request(`/api/projects/${encodeURIComponent(projectId)}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_ids: jobIds }),
  })
}

/** Note that the user has read this job, so a finished run stops glowing. */
export function markSeen(jobId: string): Promise<unknown> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}/seen`, { method: 'POST' })
}

/** The same for a whole list of jobs, in one request. */
export function markSeenMany(jobIds: string[]): Promise<{ seen: number }> {
  return request<{ seen: number }>('/api/jobs/seen', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_ids: jobIds }),
  })
}
