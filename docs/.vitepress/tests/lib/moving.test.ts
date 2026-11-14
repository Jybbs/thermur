import { describe, expect, it } from 'vitest'

import { animated } from '../../lib/moving.ts'

describe('animated', () => {
  it.for([
    [{ relativePath: 'index.md' }, true],
    [{ relativePath: 'data/index.md' }, true],
    [{ isNotFound: true, relativePath: '404.md' }, true],
    [{ relativePath: 'methods/alarm.md' }, false]
  ] as const)('reports whether %j moves on its own', ([page, moving]) => {
    expect(animated(page)).toBe(moving)
  })
})
