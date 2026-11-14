import { describe, expect, it } from 'vitest'

import { shared } from '../../lib/vite.ts'

describe('the settings the build and the suite share', () => {
  it('compiles the kit’s TSX onto Vue’s own JSX runtime and serves the identity’s stylesheet', () => {
    const settings = shared()
    expect(settings.oxc).toEqual({ jsx: { importSource: 'vue', runtime: 'automatic' } })
    expect(settings.plugins).toEqual([expect.objectContaining({ name: 'thermur:identity' })])
  })
})
