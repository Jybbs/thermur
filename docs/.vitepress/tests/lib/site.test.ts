import fs   from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { navOf, sidebarOf }    from '../../lib/navigation.ts'
import { pagesDir, publicDir } from '../../lib/paths.ts'
import source                  from '../../lib/rendered/identity.json' with { type: 'json' }
import { themeConfig }         from '../../lib/site.ts'
import { FACTS }               from '../common/facts.ts'

const config = themeConfig(FACTS, pagesDir(import.meta.url))

describe('themeConfig', () => {
  it('sets the wordmark in place of the site’s title, a file the public folder holds per scheme', () => {
    expect(config.siteTitle).toBe(false)
    const { dark, light } = source.marks.wordmark
    expect(config.logo).toEqual({ alt: 'Thermur', dark: `/${dark}`, light: `/${light}` })
    for (const file of Object.values(source.marks.wordmark)) {
      expect(fs.existsSync(path.join(publicDir(import.meta.url), file))).toBe(true)
    }
  })

  it('links the repository the manifest names', () => {
    expect(config.socialLinks).toEqual([{ icon: 'github', link: 'https://github.com/example/flock' }])
  })

  it('lists each section in the navigation bar and gives each a sidebar of its pages', () => {
    expect(config.nav).toEqual(navOf())
    expect(config.sidebar).toEqual(sidebarOf(pagesDir(import.meta.url)))
    expect(config.sidebar).toHaveProperty(['/reference/', 0, 'items'], [
      { link: '/reference/kit', text: 'The Kit' },
      { link: '/reference/identity', text: 'The Visual Identity' }
    ])
  })
})
