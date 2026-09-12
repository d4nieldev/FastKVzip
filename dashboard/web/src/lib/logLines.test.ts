import { describe, expect, it } from 'vitest'
import {
  collapseProgress,
  collapseRepeats,
  collapsedLabel,
  PROGRESS_PATTERN,
  toLogLines,
} from './logLines'

/** A log shaped the way tqdm actually writes one: updates joined by \r. */
function tqdmLog(steps: number): string {
  let out = 'Loading Qwen/Qwen3-8B\nteacher cache hit\n'
  for (let i = 0; i <= steps; i += 1) {
    out += `\r${i}%|${'#'.repeat(i / 5)}| ${i * 30}/3000 [00:0${i % 10}<02:41, 3.4it/s]`
  }
  return `${out}\nstep 3000 | train/bce 0.0871\n`
}

// A real line from an eval run: the bar carries a description prefix.
const DESCRIBED_BAR =
  '[11/11] scbench_repoqa:  78%|███████▊  | 69/88 [5:33:35<1:19:44, 251.81s/it, ' +
  'max_tokens=72499, prefill=8.8±2.2%, mixer=20.7±5.1%, gen=70.5±7.3%, max_gpu=26.9/47.4GiB]'

describe('splitting', () => {
  it('treats a carriage return as a line break', () => {
    // tqdm updates share one physical line, so splitting on \n alone rendered
    // every intermediate step as a single blob.
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

describe('recognising a progress bar', () => {
  it('matches a bar carrying a description prefix', () => {
    // Anchoring to the start of the line missed these entirely.
    expect(PROGRESS_PATTERN.test(DESCRIBED_BAR)).toBe(true)
  })

  it('matches a bare bar', () => {
    expect(PROGRESS_PATTERN.test('  0%|          | 0/3000 [00:00<02:41]')).toBe(true)
  })

  it('does not mistake ordinary output for a bar', () => {
    expect(PROGRESS_PATTERN.test('100% of examples matched')).toBe(false)
    expect(PROGRESS_PATTERN.test('accuracy 78% | run 1/3')).toBe(false)
  })
})

describe('collapsing progress runs', () => {
  it('keeps the final state of a run and counts what it replaced', () => {
    const lines = toLogLines(tqdmLog(100), false)
    const progress = lines.filter((l) => l.reason === 'progress')
    expect(progress).toHaveLength(1)
    expect(progress[0].collapsed).toBe(101)
    expect(progress[0].text).toContain('100%|')
  })

  it('leaves real output alone', () => {
    const texts = toLogLines(tqdmLog(100), false).map((l) => l.text)
    expect(texts).toContain('Loading Qwen/Qwen3-8B')
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
})

describe('collapsing repeated lines', () => {
  it('folds an identical progress bar emitted twice', () => {
    const lines = toLogLines([DESCRIBED_BAR, DESCRIBED_BAR].join('\n'), true)
    expect(lines).toHaveLength(1)
    expect(lines[0].collapsed).toBe(2)
  })

  it('folds a repeated status line', () => {
    const lines = toLogLines('include_score..\ninclude_score..\ninclude_score..', false)
    expect(lines.map((l) => l.text)).toEqual(['include_score..'])
    expect(collapsedLabel(lines[0])).toBe('× 3')
  })

  it('only folds lines that are adjacent', () => {
    const lines = toLogLines('a\na\nb\na', false)
    expect(lines.map((l) => l.text)).toEqual(['a', 'b', 'a'])
    expect(lines.map((l) => l.collapsed)).toEqual([2, undefined, undefined])
  })

  it('applies even while showing every progress step', () => {
    // "All progress steps" means every distinct step, not duplicated ones:
    // an identical line carries nothing the first did not.
    const lines = toLogLines('5%|#| 1/20\n5%|#| 1/20\n6%|#| 2/20', true)
    expect(lines.map((l) => l.text)).toEqual(['5%|#| 1/20', '6%|#| 2/20'])
  })

  it('does not label a run of blank lines', () => {
    const lines = toLogLines('a\n\n\n\nb', false)
    expect(lines.map((l) => l.text)).toEqual(['a', '', 'b'])
    expect(collapsedLabel(lines[1])).toBeNull()
  })

  it('is a no-op when nothing repeats', () => {
    const raw = [{ text: 'a' }, { text: 'b' }, { text: 'c' }]
    expect(collapseRepeats(raw)).toEqual(raw)
  })
})

describe('the collapsed label', () => {
  it('says nothing for a line standing only for itself', () => {
    expect(collapsedLabel({ text: 'a' })).toBeNull()
    expect(collapsedLabel({ text: 'a', collapsed: 1 })).toBeNull()
  })

  it('distinguishes a progress run from plain repetition', () => {
    expect(collapsedLabel({ text: 'a', collapsed: 9, reason: 'progress' })).toBe('← 9 updates')
    expect(collapsedLabel({ text: 'a', collapsed: 9, reason: 'repeat' })).toBe('× 9')
  })
})

it('splits an empty log to one empty line, not to none', () => {
  // Worth pinning: a caller counting lines to decide whether any content has
  // arrived would be wrong here, since '' splits to [''].
  expect(toLogLines('', false)).toHaveLength(1)
  expect(toLogLines('', false)[0].text).toBe('')
})
