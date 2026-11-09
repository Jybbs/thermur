import { describe, expect, it } from 'vitest'

import { computed } from '../../common/sheets.ts'

/**
 * The milliseconds each step of the fill takes at the reader's own pace.
 */
const TIMES = {
  '--th-fill-cool'  : 480,
  '--th-fill-cross' : 620,
  '--th-fill-fade'  : 120,
  '--th-fill-flare' : 60,
  '--th-fill-ink'   : 70,
  '--th-fill-turn'  : 10,
  '--th-fill-wait'  : 90
} as const
/**
 * The milliseconds each step of the fill takes under a press, which completes
 * the fill 80 ms into it, short of the turn, which takes none at any pace.
 */
const PRESS = {
  '--th-fill-press-cross' : 60,
  '--th-fill-press-flare' : 20,
  '--th-fill-press-ink'   : 20,
  '--th-fill-press-wait'  : 20
} as const

describe('the fill’s timings', () => {
  it.for(Object.entries({ ...TIMES, ...PRESS }))('scales %s by the fill’s pace', ([property, time]) => {
    expect(computed('', property, ':root', 'motion.css')).toBe(`calc(${time}ms * 1)`)
  })

  it.for(Object.keys({ ...TIMES, ...PRESS }))('sets %s to nothing for a reader who prefers reduced motion', property => {
    expect(computed('', property, ':root', 'motion.css', true)).toMatch(/\* 0\)$/u)
  })

  it.for([false, true])('turns the ink at once under a press, where reduced motion is %s', reduced => {
    expect(computed('', '--th-fill-press-turn', ':root', 'motion.css', reduced)).toBe('0ms')
  })
})
