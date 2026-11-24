import { wcagContrast }         from 'culori'
import { describe, expect, it } from 'vitest'

import * as identity        from '../../../lib/identity.ts'
import sources              from '../../../theme/styles/sources.css?raw'
import { FLOORS }           from '../../common/contrast.ts'
import { declared, styled } from '../../common/sheets.ts'

/**
 * Each number the sheet sets, beside the custom property it sets the number
 * in, the one its ground shows, and the background the sheet declares for it,
 * none where the number stands on the page's own ground.
 */
const NUMBERS = [
  ['.vp-doc a.cite', '--th-lead', '--vp-c-bg', ''],
  ['.vp-doc .source-number', '--th-lead', '--vp-c-bg', ''],
  ['.vp-doc .source[data-current] .source-numeral', '--vp-c-bg', '--th-lead', 'var(--th-lead)']
] as const

/**
 * Reads the color custom property `property` paints under `scheme`, the page's
 * ground or the lead hue.
 */
const shade = (property: string, scheme: identity.Scheme): string =>
  identity.shadeOf(property === '--th-lead' ? 'lead' : 'ground', scheme)

describe('the numbers', () => {
  it.for(NUMBERS)('sets %s in %s over %s', ([selector, color, , background]) => {
    expect(declared('color', selector, sources)).toBe(`var(${color})`)
    expect(declared('background', selector, sources)).toBe(background)
  })

  it.for(NUMBERS.flatMap(([selector, color, ground]) => (['dark', 'light'] as const).map(scheme => [selector, scheme, color, ground] as const)))(
    'holds %s to the text contrast in the %s scheme', ([, scheme, color, ground]) => {
      expect(wcagContrast(shade(color, scheme), shade(ground, scheme))).toBeGreaterThanOrEqual(FLOORS.text)
    }
  )
})

describe('the targets', () => {
  it('gives each marker a target 24 px square around its number', () => {
    expect(declared('height', '.vp-doc a.cite::after', sources)).toBe('24px')
    expect(declared('min-width', '.vp-doc a.cite::after', sources)).toBe('24px')
  })

  it('gives each return a target at least 24 px square', () => {
    expect(declared('min-height', '.vp-doc a.back', sources)).toBe('24px')
    expect(declared('min-width', '.vp-doc a.back', sources)).toBe('24px')
  })

  it('sets each line of a DOI address 24 px tall', () => {
    expect(declared('line-height', '.vp-doc a.source-address', sources)).toBe('1.5rem')
  })
})

describe('the motion', () => {
  it('times every transition and the band by the site’s shared motion, which reduced motion sets to nothing', () => {
    const timed = styled(sources).flatMap(rule => [rule.style.getPropertyValue('transition'), rule.style.getPropertyValue('animation')])
      .filter(Boolean)
    expect(timed.length).toBeGreaterThan(0)
    for (const each of timed) expect(each).toMatch(/^(?:[\w-]+ var\(--th-(?:settle|fill-cross)\)[^,]*,? ?)+$/u)
  })
})
