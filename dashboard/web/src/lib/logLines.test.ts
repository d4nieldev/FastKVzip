import { describe, expect, it } from 'vitest'
import { collapseProgress, toLogLines } from './logLines'

/** A log shaped the way tqdm actually writes one: updates joined by \r. */
function tqdmLog(steps: number): string {
  let out = 'Loading Qwen/Qwen3-8B\nteacher cache hit\n'
  for (let i = 0; i <= steps; i += 1) {
    out += `\r${i}%|${'#'.repeat(i / 5)}| ${i * 30}/3000 [00:0${i % 10}<02:41, 3.4it/s]`
  }
  return `${out}\nstep 3000 | train/bce 0.0871\n`
}

describe('splitting', () => {
  it('treats a carriage return as a line break', () => {
    // The bug: tqdm updates share one physical line, so splitting on \n alone
    // rendered every intermediate step as a single blob.
    const lines = toLogLines(tqdmLog(100), true)
    expect(lines.length).toBeGreaterThan(100)
    expect(lines.some((l) => l.text.startsWith('50%|'))).toBe(true)
  })

  it('counts \\r\\n as one break, not two', () => {
    expect(toLogLines('a\r\nb', true).map((l) => l.text)).toEqual(['a', 'b'])
  })

  it('handles plain newline logs unchanged', () => {
    expect(toLogLines('one\ntwo\nthree', true).map((l) => l.text)).toEqual([
      'one',
      'two',
      'three',
    ])
  })
})

describe('collapsing progress runs', () => {
  it('keeps the final state of a run and counts what it replaced', () => {
    const lines = toLogLines(tqdmLog(100), false)
    const progress = lines.filter((l) => l.collapsed)
    expect(progress).toHaveLength(1)
    expect(progress[0].collapsed).toBe(101)
    // The last update is what a terminal would have been showing.
    expect(progress[0].text).toContain('100%|')
  })

  it('leaves real output alone', () => {
    const lines = toLogLines(tqdmLog(100), false)
    const texts = lines.map((l) => l.text)
    expect(texts).toContain('Loading Qwen/Qwen3-8B')
    expect(texts).toContain('teacher cache hit')
    expect(texts).toContain('step 3000 | train/bce 0.0871')
  })

  it('collapses each run separately, so interleaved output survives', () => {
    const lines = collapseProgress([
      '10%|#| 1/10 [00:01]',
      '20%|##| 2/10 [00:02]',
      'checkpoint saved',
      '30%|###| 3/10 [00:03]',
      '40%|####| 4/10 [00:04]',
    ])
    expect(lines.map((l) => l.text)).toEqual([
      '20%|##| 2/10 [00:02]',
      'checkpoint saved',
      '40%|####| 4/10 [00:04]',
    ])
    expect(lines.map((l) => l.collapsed)).toEqual([2, undefined, 2])
  })

  it('does not mistake ordinary output for a progress bar', () => {
    const lines = collapseProgress(['100% of examples matched', 'done'])
    expect(lines.every((l) => l.collapsed === undefined)).toBe(true)
  })

  it('is a no-op on a log with no progress bars', () => {
    const raw = ['a', 'b', 'c']
    expect(collapseProgress(raw).map((l) => l.text)).toEqual(raw)
  })
})
