/** Turning raw job-log bytes into renderable lines. */

/**
 * tqdm separates its updates with a carriage return so each overwrites the last
 * in a terminal. In a redirected log they therefore all sit on one physical
 * line: splitting on \n alone yields a single unreadable blob where every
 * intermediate step is invisible, even though the file contains them all.
 */
export const LINE_BREAK = /\r\n|\r|\n/

export const PROGRESS_PATTERN = /^\s*\d+%\|.*\|\s*\d+\/\d+/

export const ERROR_PATTERN =
  /traceback|error|exception|cuda|assert|oom|out of memory|killed|srun:|slurmstepd/i

export interface LogLine {
  text: string
  /** Progress updates this line stands in for, when a run was collapsed. */
  collapsed?: number
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
    out.push({ text: lines[index - 1], collapsed: index - start })
  }
  return out
}

/** Split raw log text into lines, optionally collapsing progress runs. */
export function toLogLines(text: string, showProgress: boolean): LogLine[] {
  const raw = text.split(LINE_BREAK)
  return showProgress ? raw.map((line) => ({ text: line })) : collapseProgress(raw)
}
