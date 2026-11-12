import path from 'node:path'

import { describe, expect } from 'vitest'

import { navOf, sidebarOf } from '../../lib/navigation.ts'
import { test }             from '../common/repository.ts'
import { REGISTRY }         from '../common/sections.ts'

describe('navOf', () => {
  test('leads each section to its overview under its label, active on any page beneath it', () => {
    expect(navOf(REGISTRY)).toEqual([
      { activeMatch: '^/data/', link: '/data/', text: 'Data' },
      { activeMatch: '^/methods/', link: '/methods/', text: 'Methods' }
    ])
  })
})

describe('sidebarOf', () => {
  test('lists every page but the overview by its first heading, in alphabetical order', ({ repository }) => {
    repository.write('pages/data/index.md', '# Data\n')
    repository.write('pages/methods/index.md', '# Methods\n')
    repository.write('pages/methods/rules.md', 'Lead.\n\n# The Rules\n')
    repository.write('pages/methods/alarm.md', '# An Alarm\n')
    expect(sidebarOf(path.join(repository.root, 'pages'), REGISTRY)).toEqual({
      '/data/'    : [{ items: [], link: '/data/', text: 'Data' }],
      '/methods/' : [{
        items: [{ link: '/methods/alarm', text: 'An Alarm' }, { link: '/methods/rules', text: 'The Rules' }],
        link : '/methods/',
        text : 'Methods'
      }]
    })
  })

  test('refuses a page carrying no first-level heading', ({ repository }) => {
    repository.write('pages/data/notes.md', '## Notes\n')
    expect(() => sidebarOf(path.join(repository.root, 'pages'), REGISTRY.slice(0, 1)))
      .toThrow(/notes\.md carries no first-level heading/u)
  })
})
