import { describe, expect, it } from 'vitest'

import { computed } from '../../common/sheets.ts'

/**
 * A paragraph holding a chip beside a code block, inside a page's content.
 */
const PAGE = '<div class="vp-doc"><p><code id="chip">x</code></p><pre><code id="block">x</code></pre></div>'

describe('the mono’s size', () => {
  it('sets a chip where the mono’s x-height matches the body’s', () => {
    expect(computed(PAGE, '--vp-code-font-size', '#chip', 'type.css')).toBe('0.764em')
  })

  it('leaves a code block at the default theme’s size', () => {
    expect(computed(PAGE, '--vp-code-font-size', '#block', 'type.css')).toBe('')
  })
})
