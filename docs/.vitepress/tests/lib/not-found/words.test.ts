import { describe, expect, it } from 'vitest'

import { WORDS, trackOf }             from '../../../lib/not-found/words.ts'
import { REGISTRY, missingOverviews } from '../../common/sections.ts'

describe('trackOf', () => {
  it('starts at the front page and numbers each section after it, ringed in its series color', () => {
    expect(trackOf(REGISTRY)).toEqual([
      WORDS.start,
      { href: '/data/', hue: 'var(--th-series-2)', label: 'Data', note: REGISTRY[0].question, numeral: '01' },
      {
        href    : '/methods/',
        hue     : 'var(--th-series-3)',
        label   : 'Methods',
        note    : REGISTRY[1].question,
        numeral : '02'
      }
    ])
  })

  it('leads every stop to a page the site builds', () => {
    expect(missingOverviews(trackOf().map(stop => stop.href))).toEqual([])
  })
})

it.for([WORDS.title, WORDS.lead, WORDS.start.note])('words %j as published prose', text => {
  expect(text).toReadAsPublished()
})
