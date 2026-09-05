/**
 * Turning `sres` output into a scored table.
 *
 * `sres` is a site-local BGU command with no documented, stable format, so the
 * table is discovered from the output rather than hard-coded. BGU's output puts
 * a banner and a per-GPU-type summary above the per-node table, and writes its
 * cells as `3 / 3` -- spaces inside a cell, which is why columns are separated
 * on runs of two or more spaces and not on whitespace. When discovery fails the
 * caller still has the raw text, so an unrecognised format degrades to what the
 * panel showed before rather than to an empty table.
 */

export type ResourceKind = 'gpu' | 'mem' | 'cpu'

export const RESOURCE_KINDS: ResourceKind[] = ['gpu', 'mem', 'cpu']

/**
 * The most GPUs one user can hold at once.
 *
 * Read off the cluster rather than guessed: the `gpu` partition runs QOS
 * `gpu-part`, whose `MaxTRESPU` is `gres/gpu=5`. It does not appear on a user's
 * own QOS, which is why a job blocked by it only ever says
 * `Reason=QOSMaxGRESPerUser`.
 *
 * It is the denominator for the GPU score, because a node's total is not the
 * question. Twenty free GPUs and five free GPUs are the same offer to someone
 * who may hold five, and scoring the first as "36% free" made a node that
 * could take the whole allocation look worse than one that could not.
 */
export const GPU_PER_USER_CAP = 5

/**
 * How much each resource moves the score.
 *
 * The GPU is what these jobs actually queue for; memory and CPUs are, on this
 * cluster, never the thing that blocks a submission. Equal thirds would let a
 * node that has lost half its GPUs still read comfortably green.
 */
export const RESOURCE_WEIGHTS: Record<ResourceKind, number> = {
  gpu: 0.6,
  mem: 0.2,
  cpu: 0.2,
}

const KIND_PATTERNS: Array<[ResourceKind, RegExp]> = [
  ['gpu', /gpu/i],
  ['mem', /mem|ram/i],
  ['cpu', /cpu|core/i],
]

/** `3/4`, `3 / 3`, `120G/256G`, `982 / 1031`. */
const FRACTION = /^(\d+(?:\.\d+)?)\s*([kmgtp]i?b?)?\s*\/\s*(\d+(?:\.\d+)?)\s*([kmgtp]i?b?)?$/i

const UNIT_SCALE: Record<string, number> = { k: 1e3, m: 1e6, g: 1e9, t: 1e12, p: 1e15 }

function scaleOf(unit: string | undefined): number {
  if (!unit) return 1
  return UNIT_SCALE[unit[0].toLowerCase()] ?? 1
}

export interface Fraction {
  free: number
  total: number
  /** free / total, clamped to [0, 1]. */
  value: number
}

/** A `free/total` cell, or null when the cell is not one. */
export function parseFraction(cell: string): Fraction | null {
  const match = FRACTION.exec(cell.trim())
  if (!match) return null
  const [, freeText, freeUnit, totalText, totalUnit] = match
  // `120/256G` means both sides are gigabytes; only one side needs to say so.
  const unit = freeUnit ?? totalUnit
  const free = Number(freeText) * scaleOf(freeUnit ?? unit)
  const total = Number(totalText) * scaleOf(totalUnit ?? unit)
  if (!Number.isFinite(free) || !Number.isFinite(total) || total <= 0) return null
  return { free, total, value: Math.min(1, Math.max(0, free / total)) }
}

/**
 * How much of a resource is usable, in [0, 1].
 *
 * GPUs are measured against the per-user cap and not against the node: what is
 * free beyond the cap cannot be used, so it does not make a node any greener.
 * Memory and CPUs stay fractions of the node, since nothing here says how much
 * of either a given job will ask for.
 */
export function usableFraction(kind: ResourceKind, fraction: Fraction): number {
  if (kind !== 'gpu') return fraction.value
  return Math.min(fraction.free, GPU_PER_USER_CAP) / GPU_PER_USER_CAP
}

/**
 * Availability in [0, 1] over whatever resources the row actually reports.
 *
 * A weighted geometric mean, not an average: any resource at zero takes the
 * whole score to zero. A node with every GPU busy is useless for these jobs no
 * matter how much of its memory sits idle, and an arithmetic mean would paint
 * exactly that node green. Resources the output does not mention are dropped
 * and the remaining weights renormalised -- never read as zero, which would
 * turn the whole table red.
 */
export function availabilityScore(
  fractions: Partial<Record<ResourceKind, number>>,
): number | null {
  const present = RESOURCE_KINDS.filter((kind) => fractions[kind] !== undefined)
  if (!present.length) return null
  const weight = present.reduce((sum, kind) => sum + RESOURCE_WEIGHTS[kind], 0)
  return present.reduce(
    (score, kind) => score * Math.pow(fractions[kind] as number, RESOURCE_WEIGHTS[kind] / weight),
    1,
  )
}

export interface SresRow {
  cells: string[]
  /** Parsed resources, by column index. */
  resources: Map<number, Fraction>
  score: number | null
  /** Everything in the row, lowercased, for the filter box to match against. */
  haystack: string
}

export interface SresTable {
  headers: string[]
  /** Which resource each column carries, by column index. */
  kinds: Map<number, ResourceKind>
  rows: SresRow[]
}

/** One GPU type in the "GPU UTILIZATION" block above the node table. */
export interface SresTotal {
  label: string
  free: number
  total: number
  value: number
}

export interface SresView {
  totals: SresTotal[] | null
  table: SresTable | null
}

type Splitter = (line: string) => string[]

// Two-or-more spaces first: BGU writes cells as "3 / 3" and headers as
// "MEM [GB]", so a single space is *inside* a cell, not between two. Plain
// whitespace is the fallback for an output that separates columns by one space.
const SPLITTERS: Splitter[] = [
  (line) => line.trim().split(/\s{2,}/),
  (line) => line.trim().split(/\s+/),
]

/**
 * Which resource each fraction-shaped column carries.
 *
 * A column's own header usually says ("GPUs", "MEM [GB]"), but a name column
 * followed by its counts -- `GPU  FREE` with `rtx_6000  3/4` -- is just as
 * common, so an unlabelled column inherits from the nearest labelled column to
 * its left.
 */
function classifyColumns(headers: string[], rows: string[][]): Map<number, ResourceKind> {
  const kinds = new Map<number, ResourceKind>()

  const kindOfHeader = (header: string | undefined): ResourceKind | null => {
    if (!header) return null
    for (const [kind, pattern] of KIND_PATTERNS) if (pattern.test(header)) return kind
    return null
  }

  for (let column = 0; column < headers.length; column += 1) {
    const cells = rows.map((row) => row[column]).filter((cell) => cell && cell !== '-')
    if (!cells.length) continue
    const parsed = cells.filter((cell) => parseFraction(cell) !== null)
    // Half is enough: a node may report "-" or "N/A" for a resource it lacks.
    if (parsed.length * 2 < cells.length) continue

    let kind = kindOfHeader(headers[column])
    for (let left = column - 1; kind === null && left >= 0; left -= 1) {
      // Only a non-resource column can lend its name, or two adjacent count
      // columns would collapse onto the same kind.
      if (rows.some((row) => row[left] && parseFraction(row[left]) !== null)) break
      kind = kindOfHeader(headers[left])
    }
    if (kind && !Array.from(kinds.values()).includes(kind)) kinds.set(column, kind)
  }
  return kinds
}

function buildTable(headers: string[], rows: string[][], kinds: Map<number, ResourceKind>): SresTable {
  return {
    headers,
    kinds,
    rows: rows.map((cells) => {
      const resources = new Map<number, Fraction>()
      const fractions: Partial<Record<ResourceKind, number>> = {}
      for (const [column, kind] of kinds) {
        const fraction = cells[column] ? parseFraction(cells[column]) : null
        if (!fraction) continue
        resources.set(column, fraction)
        fractions[kind] = usableFraction(kind, fraction)
      }
      return {
        cells,
        resources,
        score: availabilityScore(fractions),
        haystack: cells.join(' ').toLowerCase(),
      }
    }),
  }
}

/**
 * The widest run of aligned rows anywhere in the output.
 *
 * The node table does not start at line one -- a banner and a per-GPU-type
 * summary come first -- so every line is tried as a header and kept only if
 * the lines directly under it split to the same width and carry counts. The
 * candidate with the most rows wins, which is the node table by a wide margin.
 */
function findTable(lines: string[], split: Splitter): SresTable | null {
  let best: SresTable | null = null

  for (let start = 0; start < lines.length - 1; start += 1) {
    const headers = split(lines[start])
    if (headers.length < 2) continue

    const rows: string[][] = []
    for (let index = start + 1; index < lines.length; index += 1) {
      const cells = split(lines[index])
      if (cells.length !== headers.length) break
      if (!cells.some((cell) => parseFraction(cell) !== null)) break
      rows.push(cells)
    }
    if (!rows.length) continue

    const kinds = classifyColumns(headers, rows)
    if (!kinds.size) continue

    const table = buildTable(headers, rows, kinds)
    if (!best || table.rows.length > best.rows.length) best = table
  }
  return best
}

/**
 * The "GPU UTILIZATION" block: Free / In use / Total per GPU type.
 *
 * Worth keeping separately because it is the only place the *type* of a free
 * GPU is named -- the node table counts GPUs per node without saying whether
 * they are 6000pro or 6000, which is exactly the choice to make before
 * submitting. Its columns are single-space separated, unlike the node table.
 */
function findTotals(lines: string[]): SresTotal[] | null {
  let labels: string[] | null = null
  let free: number[] | null = null
  let total: number[] | null = null

  for (let index = 0; index < lines.length; index += 1) {
    const match = /^\s*(free|in use|total)\s*:\s*(.+)$/i.exec(lines[index])
    if (!match) continue
    const numbers = match[2].trim().split(/\s+/).map(Number)
    if (!numbers.length || numbers.some((value) => !Number.isFinite(value))) continue

    if (!labels) {
      // The line above the first metric row names the columns.
      const above = (lines[index - 1] ?? '').trim().split(/\s+/).filter(Boolean)
      if (above.length === numbers.length) labels = above
    }
    if (/free/i.test(match[1])) free = numbers
    if (/^total$/i.test(match[1].trim())) total = numbers
  }

  if (!labels || !free || !total) return null
  if (labels.length !== free.length || labels.length !== total.length) return null

  // Scored against the cap for the same reason the node table is: what matters
  // is whether this GPU type can fill an allocation, not what share of the
  // cluster's stock of it happens to be idle.
  return labels.map((label, index) => ({
    label,
    free: free![index],
    total: total![index],
    value: Math.min(free![index], GPU_PER_USER_CAP) / GPU_PER_USER_CAP,
  }))
}

export function parseSres(text: string): SresView | null {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (!lines.length) return null

  let table: SresTable | null = null
  for (const split of SPLITTERS) {
    const candidate = findTable(lines, split)
    if (!table || (candidate && candidate.rows.length > table.rows.length)) {
      table = candidate ?? table
    }
  }

  const totals = findTotals(lines)
  return table || totals ? { table, totals } : null
}

/**
 * The availability ramp, in the dashboard's own state colours.
 *
 * Red to green the short way passes through a desaturated olive at the middle,
 * where the reader can least afford ambiguity. Going around through orange and
 * amber -- the colours this UI already uses for failed and pending -- keeps
 * every step saturated and ordered. Interpolation is in OKLab so the steps are
 * perceptually even rather than even in sRGB.
 */
const RAMP = ['#f85149', '#db6d28', '#d29922', '#3fb950'] as const

type Oklab = [number, number, number]

function toLinear(channel: number): number {
  return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4)
}

function hexToOklab(hex: string): Oklab {
  const r = toLinear(parseInt(hex.slice(1, 3), 16) / 255)
  const g = toLinear(parseInt(hex.slice(3, 5), 16) / 255)
  const b = toLinear(parseInt(hex.slice(5, 7), 16) / 255)
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

const RAMP_OKLAB: Oklab[] = RAMP.map(hexToOklab)

/** A CSS colour for a score in [0, 1]; 0 is red, 1 is green. */
export function availabilityColor(score: number): string {
  const clamped = Math.min(1, Math.max(0, score))
  const position = clamped * (RAMP_OKLAB.length - 1)
  const lower = Math.min(Math.floor(position), RAMP_OKLAB.length - 2)
  const t = position - lower
  const from = RAMP_OKLAB[lower]
  const to = RAMP_OKLAB[lower + 1]
  const [l, a, b] = from.map((value, index) => value + (to[index] - value) * t)
  return `oklab(${(l * 100).toFixed(2)}% ${a.toFixed(4)} ${b.toFixed(4)})`
}
