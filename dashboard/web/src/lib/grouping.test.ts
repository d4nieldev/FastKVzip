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
  it('gathers each afternoon of submissions into its batches', () => {
    // Two clusters of activity, a quarter of an hour apart.
    expect(groupBySubmission(REAL).map((group) => group.jobs.length)).toEqual([9, 10])
  })

  it('keeps a grid together with what was submitted around it', () => {
    const [recent] = groupBySubmission(REAL)
    expect(recent.jobs[0].job_id).toBe('20824344')
    expect(recent.jobs.at(-1)!.job_id).toBe('20824155')
  })

  it('measures the window from the batch start, so it cannot chain onward', () => {
    // Every step here is inside the window, but the batch is not: anchoring on
    // the previous job instead would run all three together, and on real data
    // that collapses a whole afternoon into one group.
    const drip = [
      job('3', '2026-09-01T10:28:00'),
      job('2', '2026-09-01T10:14:00'),
      job('1', '2026-09-01T10:00:00'),
    ]
    expect(groupBySubmission(drip).map((group) => group.jobs.length)).toEqual([2, 1])
  })

  it('labels a batch with the moment it started, not the last job in it', () => {
    // Compared the way the fixture was built; toISOString would answer in UTC
    // and disagree with these local-time strings by the runner's offset.
    const [recent] = groupBySubmission(REAL)
    expect(recent.submittedAt).toBe(Math.floor(new Date('2026-09-01T16:12:45').getTime() / 1000))
  })

  it('starts a new batch when a job reports no submission time', () => {
    const jobs = [job('3', '2026-09-01T10:00:02'), job('2', null), job('1', '2026-09-01T10:00:00')]
    expect(groupBySubmission(jobs).map((group) => group.jobs.length)).toEqual([1, 1, 1])
  })

  it('honours a narrower window when one is given', () => {
    expect(groupBySubmission(REAL, 15).map((group) => group.jobs.length)).toEqual([1, 3, 5, 1, 9])
  })

  it('has nothing to group in an empty list', () => {
    expect(groupBySubmission([])).toEqual([])
  })
})

describe('stateSummary', () => {
  it('counts states, most common first', () => {
    const [recent, earlier] = groupBySubmission(REAL)
    expect(stateSummary(recent)).toEqual([
      ['COMPLETED', 8],
      ['RUNNING', 1],
    ])
    expect(stateSummary(earlier)).toEqual([
      ['FAILED', 9],
      ['COMPLETED', 1],
    ])
  })
})
