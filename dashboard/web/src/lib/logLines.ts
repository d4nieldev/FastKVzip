/** Turning raw job-log bytes into renderable lines. */

/**
 * tqdm separates its updates with a carriage return so each overwrites the last
 * in a terminal. In a redirected log they therefore all sit on one physical
 * line: splitting on \n alone yields a single unreadable blob where every
 * intermediate step is invisible, even though the file contains them all.
 */
export const LINE_BREAK = /\r\n|\r|\n/

/**
 * A tqdm bar anywhere in the line, not just at its start.
 *
 * These runs carry a description prefix, so anchoring would miss them:
 *   [11/11] scbench_repoqa:  78%|#######   | 69/88 [5:33:35<1:19:44, ...]
 */
export const PROGRESS_PATTERN = /\d+%\|[^|]*\|\s*\d+\/\d+/

export const ERROR_PATTERN =
  /traceback|error|exception|cuda|assert|oom|out of memory|killed|srun:|slurmstepd/i

export interface LogLine {
  text: string
  /** How many source lines this one stands for, when a run was collapsed. */
  collapsed?: number
  /** Why it was collapsed: a progress run, or plain repetition. */
  reason?: 'progress' | 'repeat'
}

/**
 * Reduce each run of consecutive tqdm updates to its final state.
 *
 * That is what the terminal would have shown -- every update overwrote the one
 * before it -- and it keeps the current percentage visible instead of dropping
 * progress entirely or burying real output under thousands of bars.
 */
export function collapseProgress(lines: string[]): LogLine[] {
  const out: LogLine[] = []
  let index = 0

  while (index < lines.length) {
    if (!PROGRESS_PATTERN.test(lines[index])) {
      out.push({ text: lines[index] })
      index += 1
      continue
    }
    const start = index
    while (index < lines.length && PROGRESS_PATTERN.test(lines[index])) index += 1
    out.push({ text: lines[index - 1], collapsed: index - start, reason: 'progress' })
  }
  return out
}

/**
 * Fold runs of byte-identical adjacent lines into one.
 *
 * Applied even when every progress step is being shown: an exactly repeated
 * line carries no information the first one did not, so nothing is lost, and
 * these logs repeat both stalled progress bars and plain status lines.
 */
export function collapseRepeats(lines: LogLine[]): LogLine[] {
  const out: LogLine[] = []

  for (const line of lines) {
    const previous = out[out.length - 1]
    if (previous && previous.text === line.text) {
      previous.collapsed = (previous.collapsed ?? 1) + (line.collapsed ?? 1)
      previous.reason = previous.reason ?? 'repeat'
      continue
    }
    out.push({ ...line })
  }
  return out
}

/** Split raw log text into lines, collapsing progress runs and repetition. */
export function toLogLines(text: string, showProgress: boolean): LogLine[] {
  const raw = text.split(LINE_BREAK)
  const staged = showProgress ? raw.map((line) => ({ text: line })) : collapseProgress(raw)
  return collapseRepeats(staged)
}

/** Label for a collapsed run, or null when there is nothing to say. */
export function collapsedLabel(line: LogLine): string | null {
  if (!line.collapsed || line.collapsed < 2) return null
  // Blank runs are just spacing; counting them would be noise.
  if (!line.text.trim()) return null
  return line.reason === 'progress' ? `← ${line.collapsed} updates` : `× ${line.collapsed}`
}
