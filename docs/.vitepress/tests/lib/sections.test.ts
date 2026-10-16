import { expect, it } from 'vitest'

import * as sections                  from '../../lib/sections.ts'
import { REGISTRY, missingOverviews } from '../common/sections.ts'

it.for([
  ['data/index.md', { label: 'Data', place: 1 }],
  ['methods/alarm.md', { label: 'Methods', place: 2 }],
  ['index.md', undefined],
  ['reference/settings.md', undefined]
] as const)('places the page at %s', ([relativePath, placed]) => {
  expect(sections.sectionOf(relativePath, REGISTRY)).toEqual(placed)
})

it('labels each section by its folder’s name, capitalized', () => {
  expect(REGISTRY.map(section => sections.labelOf(section))).toEqual(['Data', 'Methods'])
})

it('keeps the registry in alphabetical order, each section opening on an overview', () => {
  const registry: readonly sections.Section[] = sections.SECTIONS
  const names                                 = registry.map(section => section.name)
  expect(names).toEqual(names.toSorted())
  expect(missingOverviews(names)).toEqual([])
})
