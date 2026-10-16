// @vitest-environment happy-dom
import { expect, it } from 'vitest'

import { Numerals } from '../../../lib/not-found/numerals.ts'
import source       from '../../../lib/rendered/identity.json' with { type: 'json' }
import { browser }  from '../../common/browser.ts'

it('sets the 404 a third of the drawing across, at most 230 pixels, with its ink centered', () => {
  const stand    = browser()
  const numerals = new Numerals(source.faces.display)
  numerals.fit(1200, 800)
  numerals.paint(stand.pen, '#123456')
  expect(numerals).toMatchObject({ cx: 600, cy: 400, height: 124, width: 200 })
  expect(stand.context.fillText).toHaveBeenCalledWith('404', 600, 458)
  expect(stand.wrote('font')).toContain("400 230px 'Bellefair', Didot, Georgia, serif")
  numerals.fit(300, 800)
  expect(stand.wrote('font').at(-1)).toBe("400 102px 'Bellefair', Didot, Georgia, serif")
})
