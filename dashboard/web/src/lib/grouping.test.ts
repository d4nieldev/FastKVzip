import { describe, expect, it } from 'vitest'
import { groupBySubmission, stateSummary } from './grouping'
import type { Job } from './types'

function job(jobId: string, submitAt: string | null, state = 'COMPLETED'): Job {
  return {
    job_id: jobId,
    submit_ts: submitAt === null ? null : Math.floor(new Date(submitAt).getTime() / 1000),
    state,
  } as Job
}

// Real submissions read off the cluster with `sacct -o JobID,Submit`, newest
// first, which is the order the list renders in.
const REAL: Job[] = [
  job('20824344', '2026-09-01T16:16:52'),
  job('20824258', '2026-09-01T16:14:51', 'RUNNING'),
  job('20824255', '2026-09-01T16:14:50'),
  job('20824254', '2026-09-01T16:14:50'),
  job('20824160', '2026-09-01T16:12:49'),
  job('20824158', '2026-09-01T16:12:48'),
  job('20824157', '2026-09-01T16:12:47'),
  job('20824156', '2026-09-01T16:12:47'),
  job('20824155', '2026-09-01T16:12:45'),
  job('20823811', '2026-09-01T16:01:10'),
  job('20823805', '2026-09-01T16:00:38', 'FAILED'),
  job('20823804', '2026-09-01T16:00:38', 'FAILED'),
  job('20823803', '2026-09-01T16:00:35', 'FAILED'),
  job('20823802', '2026-09-01T16:00:34', 'FAILED'),
  job('20823801', '2026-09-01T16:00:34', 'FAILED'),
  job('20823800', '2026-09-01T16:00:33', 'FAILED'),
  job('20823799', '2026-09-01T16:00:33', 'FAILED'),
  job('20823798', '2026-09-01T16:00:32', 'FAILED'),
  job('20823796', '2026-09-01T16:00:29', 'FAILED'),
]

describe('groupBySubmission', () => {
  it('recovers the batches these jobs were actually submitted in', () => {
    expect(groupBySubmission(REAL).map((group) => group.jobs.length)).toEqual([1, 3, 5, 1, 9])
  })

  it('keeps a grid together across its internal three-second gaps', () => {
    // 16:00:29 -> 16:00:32 is the widest step inside a real grid.
    const grid = groupBySubmission(REAL).at(-1)!
    expect(grid.jobs).toHaveLength(9)
    expect(grid.jobs[0].job_id).toBe('20823805')
    expect(grid.jobs[8].job_id).toBe('20823796')
  })

  it('does not swallow a separate run submitted 32 seconds later', () => {
    // The case that rules out a one-minute window: a lone evaluation at
    // 16:01:10, half a minute after a grid finished going in at 16:00:38.
    const groups = groupBySubmission(REAL)
    const alone = groups.find((group) => group.jobs.length === 1 && group.key === '20823811')
    expect(alone).toBeDefined()
  })

  it('labels a batch with the moment it started, not the last job in it', () => {
    const grid = groupBySubmission(REAL).at(-1)!
    // Compared the way the fixture was built; toISOString would answer in UTC
    // and disagree with these local-time strings by the runner's offset.
    expect(grid.submittedAt).toBe(Math.floor(new Date('2026-09-01T16:00:29').getTime() / 1000))
  })

  it('chains a slow submission as long as no single step stalls', () => {
    // The gap is between consecutive jobs, not across the batch, so a grid
    // spanning a minute holds together while a real pause splits it.
    const slow = [
      job('5', '2026-09-01T10:01:00'),
      job('4', '2026-09-01T10:00:45'),
      job('3', '2026-09-01T10:00:30'),
      job('2', '2026-09-01T10:00:15'),
      job('1', '2026-09-01T10:00:00'),
    ]
    expect(groupBySubmission(slow)).toHaveLength(1)
  })

  it('starts a new batch when a job reports no submission time', () => {
    const jobs = [job('3', '2026-09-01T10:00:02'), job('2', null), job('1', '2026-09-01T10:00:00')]
    expect(groupBySubmission(jobs).map((group) => group.jobs.length)).toEqual([1, 1, 1])
  })

  it('has nothing to group in an empty list', () => {
    expect(groupBySubmission([])).toEqual([])
  })
})

describe('stateSummary', () => {
  it('counts states, most common first', () => {
    const groups = groupBySubmission(REAL)
    expect(stateSummary(groups[1])).toEqual([
      ['COMPLETED', 2],
      ['RUNNING', 1],
    ])
    expect(stateSummary(groups.at(-1)!)).toEqual([['FAILED', 9]])
  })
})
