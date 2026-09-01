import { describe, expect, it } from 'vitest'
import {
  availabilityColor,
  availabilityScore,
  parseFraction,
  parseSres,
} from './sres'

describe('parseFraction', () => {
  it('reads plain counts', () => {
    expect(parseFraction('3/4')).toEqual({ free: 3, total: 4, value: 0.75 })
  })

  it('reads sized memory on both sides', () => {
    expect(parseFraction('120G/256G')?.value).toBeCloseTo(120 / 256, 6)
  })

  it('carries a unit named on one side only', () => {
    // "120/256G" is 120 gigabytes of 256, not 120 bytes of 256 gigabytes.
    expect(parseFraction('120/256G')?.value).toBeCloseTo(120 / 256, 6)
  })

  it('rejects cells that are not fractions', () => {
    expect(parseFraction('rtx_pro_6000')).toBeNull()
    expect(parseFraction('-')).toBeNull()
  })

  it('rejects a zero total rather than dividing by it', () => {
    expect(parseFraction('0/0')).toBeNull()
  })

  it('never reports more than fully free', () => {
    expect(parseFraction('5/4')?.value).toBe(1)
  })
})

describe('availabilityScore', () => {
  it('is 1 only when everything is free', () => {
    expect(availabilityScore({ gpu: 1, mem: 1, cpu: 1 })).toBe(1)
  })

  it('is 0 when any single resource is exhausted', () => {
    // The whole point of the geometric mean: an arithmetic one would score
    // this node 0.67 and paint it green, though no job can start on it.
    expect(availabilityScore({ gpu: 0, mem: 1, cpu: 1 })).toBe(0)
    expect(availabilityScore({ gpu: 1, mem: 0, cpu: 1 })).toBe(0)
    expect(availabilityScore({ gpu: 1, mem: 1, cpu: 0 })).toBe(0)
  })

  it('weights the GPU above memory and CPUs', () => {
    const gpuHalf = availabilityScore({ gpu: 0.5, mem: 1, cpu: 1 })
    const cpuHalf = availabilityScore({ gpu: 1, mem: 1, cpu: 0.5 })
    expect(gpuHalf).toBeCloseTo(Math.pow(0.5, 0.6), 6)
    expect(cpuHalf).toBeCloseTo(Math.pow(0.5, 0.2), 6)
    expect(gpuHalf!).toBeLessThan(cpuHalf!)
  })

  it('renormalises over the resources actually reported', () => {
    // GPU alone carries the whole weight rather than being diluted by two
    // resources the output never mentioned.
    expect(availabilityScore({ gpu: 0.4 })).toBeCloseTo(0.4, 6)
    expect(availabilityScore({ gpu: 0.5, mem: 1 })).toBeCloseTo(Math.pow(0.5, 0.75), 6)
  })

  it('has no score when nothing was reported', () => {
    expect(availabilityScore({})).toBeNull()
  })
})

describe('parseSres', () => {
  it('takes the resource from the column naming it, not the counts column', () => {
    const table = parseSres(
      [
        'PARTITION  NODE     GPU              FREE',
        'main       gpu-01   rtx_pro_6000     0/2',
        'main       gpu-04   rtx_6000         3/4',
      ].join('\n'),
    )
    expect(table).not.toBeNull()
    expect(table!.kinds.get(3)).toBe('gpu')
    expect(table!.rows.map((row) => row.score)).toEqual([0, 0.75])
  })

  it('reads columns that name their own resource', () => {
    const table = parseSres(
      [
        'NODE     GPUS   MEM         CPUS',
        'gpu-01   0/2    100G/256G   8/32',
        'gpu-02   4/4    256G/256G   32/32',
      ].join('\n'),
    )!
    expect([...table.kinds.values()].sort()).toEqual(['cpu', 'gpu', 'mem'])
    expect(table.rows[0].score).toBe(0)
    expect(table.rows[1].score).toBe(1)
  })

  it('lowercases the whole row for the filter box', () => {
    const table = parseSres('NODE  GPU           FREE\nn1    RTX_PRO_6000  1/2')!
    expect(table.rows[0].haystack).toContain('rtx_pro_6000')
  })

  it('gives up rather than guessing when no column can be identified', () => {
    // Without a header there is nothing to say what "3/4" counts, so the panel
    // falls back to the raw output instead of inventing a meaning.
    expect(parseSres('main gpu-01 rtx_6000 3/4')).toBeNull()
    expect(parseSres('some unrelated output')).toBeNull()
    expect(parseSres('')).toBeNull()
  })
})

describe('availabilityColor', () => {
  it('runs red at 0 and green at 1', () => {
    // OKLab b is the yellow(+)/blue(-) axis and a is red(+)/green(-); red and
    // green must land on opposite sides of the a axis.
    const red = availabilityColor(0)
    const green = availabilityColor(1)
    expect(Number(/oklab\([\d.]+% (-?[\d.]+)/.exec(red)![1])).toBeGreaterThan(0)
    expect(Number(/oklab\([\d.]+% (-?[\d.]+)/.exec(green)![1])).toBeLessThan(0)
  })

  it('moves steadily from red towards green', () => {
    const a = [0, 0.25, 0.5, 0.75, 1].map(
      (score) => Number(/oklab\([\d.]+% (-?[\d.]+)/.exec(availabilityColor(score))![1]),
    )
    expect(a).toEqual([...a].sort((x, y) => y - x))
  })

  it('clamps scores outside the range', () => {
    expect(availabilityColor(-1)).toBe(availabilityColor(0))
    expect(availabilityColor(9)).toBe(availabilityColor(1))
  })
})
