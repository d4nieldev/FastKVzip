import type { Job } from './types'

/**
 * How far apart two submissions can be and still count as one batch.
 *
 * Calibrated against real submissions from this cluster rather than guessed.
 * `slurm/submit_graph_grid.sh` loops over sbatch with nothing in between, so a
 * grid lands in a burst: the widest gap seen *inside* one is 3 seconds, across
 * grids of 3, 5 and 9 jobs. The narrowest gap between two genuinely separate
 * submissions is 32 seconds -- a grid at 16:00:38 followed by a lone
 * evaluation at 16:01:10 -- so a minute-wide window, the obvious first guess,
 * would have swallowed that one into the grid above it.
 *
 * Fifteen seconds sits with room on both sides. It is compared between
 * *consecutive* jobs rather than across the whole group, so a grid that takes
 * a minute to submit still holds together as long as no single step stalls.
 */
export const SUBMISSION_GAP_SECONDS = 15

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
  gapSeconds: number = SUBMISSION_GAP_SECONDS,
): JobGroup[] {
  const groups: JobGroup[] = []

  for (const job of jobs) {
    const current = groups[groups.length - 1]
    const previous = current?.jobs[current.jobs.length - 1]
    const continues =
      current !== undefined &&
      job.submit_ts !== null &&
      previous?.submit_ts != null &&
      Math.abs(job.submit_ts - previous.submit_ts) <= gapSeconds

    if (continues) {
      current.jobs.push(job)
      // The list runs newest first, so each job added is the earlier one, and
      // the batch is labelled with the moment it started rather than ended.
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
