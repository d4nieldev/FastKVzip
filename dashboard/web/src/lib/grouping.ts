import type { Job } from './types'

/**
 * How wide a batch may be: jobs landing within this of the newest one in it.
 *
 * Measured from the batch's own start, not between consecutive jobs. The
 * distinction only matters at this width, and it decides everything: the real
 * submissions on this cluster are close enough in sequence that a
 * consecutive-gap rule at fifteen minutes chains straight through an
 * afternoon, collapsing nineteen jobs across five separate submissions into
 * one group. Anchoring on the batch start bounds each group instead, so a
 * quiet stretch is what ends it.
 *
 * Fifteen minutes is deliberately generous -- it gathers a grid together with
 * whatever was submitted around it rather than splitting on the couple of
 * minutes between two arms of the same experiment.
 */
export const SUBMISSION_WINDOW_SECONDS = 15 * 60

export interface JobGroup {
  /** Stable across polls: the id of the newest job in the batch. */
  key: string
  /** When the batch started going in, or null if nothing reported a time. */
  submittedAt: number | null
  jobs: Job[]
}

/**
 * Split jobs into the batches they were submitted in.
 *
 * Expects the list in job-id order, which is also submission order -- SLURM
 * hands ids out monotonically, so consecutive entries are adjacent in time.
 */
export function groupBySubmission(
  jobs: Job[],
  windowSeconds: number = SUBMISSION_WINDOW_SECONDS,
): JobGroup[] {
  const groups: JobGroup[] = []

  for (const job of jobs) {
    const current = groups[groups.length - 1]
    // The list runs newest first, so the batch's first entry is its newest and
    // the window reaches back from there. Comparing against the *previous*
    // job instead would let each one extend the reach of the next, and a busy
    // afternoon would arrive as a single group.
    const anchor = current?.jobs[0]
    const continues =
      current !== undefined &&
      job.submit_ts !== null &&
      anchor?.submit_ts != null &&
      Math.abs(anchor.submit_ts - job.submit_ts) <= windowSeconds

    if (continues) {
      current.jobs.push(job)
      // Each job added is the earlier one, so the batch ends up labelled with
      // the moment it started rather than the moment it finished going in.
      if (job.submit_ts !== null) current.submittedAt = job.submit_ts
    } else {
      groups.push({ key: job.job_id, submittedAt: job.submit_ts, jobs: [job] })
    }
  }

  return groups
}

/** States present in a batch with their counts, most common first. */
export function stateSummary(group: JobGroup): Array<[string, number]> {
  const counts = new Map<string, number>()
  for (const job of group.jobs) counts.set(job.state, (counts.get(job.state) ?? 0) + 1)
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
}
