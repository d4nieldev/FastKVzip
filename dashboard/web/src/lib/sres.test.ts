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
    )!.table!
    expect(table.kinds.get(3)).toBe('gpu')
    // 3 free of 4 is 3 of the 5 a user may hold, not three quarters of a node.
    expect(table.rows.map((row) => row.score)).toEqual([0, 0.6])
  })

  it('reads columns that name their own resource', () => {
    const table = parseSres(
      [
        'NODE     GPUS   MEM         CPUS',
        'gpu-01   0/2    100G/256G   8/32',
        'gpu-02   4/4    256G/256G   32/32',
      ].join('\n'),
    )!.table!
    expect([...table.kinds.values()].sort()).toEqual(['cpu', 'gpu', 'mem'])
    expect(table.rows[0].score).toBe(0)
    // Idle in every other respect, but four free GPUs cannot fill an
    // allocation of five, so it is not fully green.
    expect(table.rows[1].score).toBeCloseTo(Math.pow(4 / 5, 0.6), 6)
  })

  it('lowercases the whole row for the filter box', () => {
    const table = parseSres(
      'NODE  GPU           FREE\nn1    RTX_PRO_6000  1/2\nn2    RTX_6000      2/2',
    )!.table!
    expect(table.rows[0].haystack).toContain('rtx_pro_6000')
  })

  it('gives up rather than guessing when no column can be identified', () => {
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

// The output BGU's sres actually prints. Cells are "3 / 3" with spaces inside
// them, a banner and a per-type summary sit above the node table, and one
// header cell ("MEM [GB]") contains a space of its own.
const REAL_SRES = `**************************************************
GPU UTILIZATION:
          6000pro 6000 4090 3090 2080 1080
 Free:        11    41   25   16   34   33
 In use:      45    99   68   65   17    8
 Total:       56   140   93   81   51   41
**************************************************
Available Resources Per Node [avail / total]
GPU Nodes
NODE            GPUs      MEM [GB]      CPUs
cs-1080-01      3 / 3     257 / 257     32 / 32
cs-3090-01      2 / 8     233 / 515     22 / 64
cs-3090-06      8 / 8     515 / 515     64 / 64
cs-4090-01      0 / 3     174 / 257     10 / 32
cs-6000-03      7 / 8     491 / 515     58 / 64
cs-cpu256-01    0 / 1     982 / 1031    250 / 256`

describe('parseSres on real BGU output', () => {
  it('finds the node table below the banner and the summary', () => {
    const table = parseSres(REAL_SRES)!.table!
    expect(table.headers).toEqual(['NODE', 'GPUs', 'MEM [GB]', 'CPUs'])
    expect(table.rows).toHaveLength(6)
    expect(table.rows[0].cells[0]).toBe('cs-1080-01')
  })

  it('identifies all three resources from their own headers', () => {
    const table = parseSres(REAL_SRES)!.table!
    expect(table.kinds.get(1)).toBe('gpu')
    expect(table.kinds.get(2)).toBe('mem')
    expect(table.kinds.get(3)).toBe('cpu')
  })

  it('reads counts written with spaces around the slash', () => {
    const table = parseSres(REAL_SRES)!.table!
    expect(table.rows[1].resources.get(1)).toEqual({ free: 2, total: 8, value: 0.25 })
  })

  it('scores against what a user may hold, not against the node', () => {
    const rows = parseSres(REAL_SRES)!.table!.rows
    const score = (node: string) => rows.find((row) => row.cells[0] === node)!.score
    // Wholly idle, but a three-GPU node can only ever offer three of five.
    expect(score('cs-1080-01')).toBeCloseTo(Math.pow(3 / 5, 0.6), 6)
    // Eight free is past the cap, so it is as good as an allocation can get.
    expect(score('cs-3090-06')).toBe(1)
    // Every GPU busy, though two thirds of its memory is free.
    expect(score('cs-4090-01')).toBe(0)
    expect(score('cs-cpu256-01')).toBe(0)
    // Seven free is also past the cap, so only memory and CPUs hold it back.
    expect(score('cs-6000-03')!).toBeCloseTo(
      Math.pow(491 / 515, 0.2) * Math.pow(58 / 64, 0.2),
      6,
    )
  })

  it('keeps the per-GPU-type summary, which the node table cannot supply', () => {
    // The node table counts GPUs per node without saying whether they are
    // 6000pro or 6000 -- the choice to make before submitting.
    const totals = parseSres(REAL_SRES)!.totals!
    expect(totals.map((entry) => entry.label)).toEqual([
      '6000pro', '6000', '4090', '3090', '2080', '1080',
    ])
    // 11 free is past the cap of 5, so the type can fill an allocation: 1.
    expect(totals[0]).toEqual({ label: '6000pro', free: 11, total: 56, value: 1 })
    expect(totals[5].value).toBe(1)
  })

  it('never mistakes the summary block for the node table', () => {
    expect(parseSres(REAL_SRES)!.table!.rows).toHaveLength(6)
  })
})

describe('the per-user GPU cap', () => {
  // Memory and CPUs are left wholly free so the GPU term is what is measured.
  const NODES = `NODE            GPUs      MEM [GB]      CPUs
big-idle        20 / 24   512 / 512     64 / 64
exactly-cap      5 / 8    512 / 512     64 / 64
half-cap         2 / 8    512 / 512     64 / 64
none             0 / 8    512 / 512     64 / 64`

  const score = (node: string) =>
    parseSres(NODES)!.table!.rows.find((row) => row.cells[0] === node)!.score

  it('is full green once the cap can be filled, however many more are idle', () => {
    // 20 free and 5 free are the same offer to someone who may hold five.
    expect(score('big-idle')).toBeCloseTo(score('exactly-cap')!, 6)
    expect(score('exactly-cap')).toBeCloseTo(1, 6)
  })

  it('scores what is usable, not what share of the node is idle', () => {
    // 2 of 8 would have been 0.25 of the node; it is 2 of the 5 allowed.
    expect(score('half-cap')).toBeCloseTo(Math.pow(2 / 5, 0.6), 6)
  })

  it('still goes to red with nothing free', () => {
    expect(score('none')).toBe(0)
  })

  it('applies the same rule to the per-type totals', () => {
    const totals = parseSres(
      `GPU UTILIZATION:
          6000pro 6000 4090
 Free:        16    3    0
 In use:      40  137   93
 Total:       56  140   93`,
    )!.totals!
    expect(totals[0].value).toBe(1)
    expect(totals[1].value).toBeCloseTo(3 / 5, 6)
    expect(totals[2].value).toBe(0)
  })
})
