import fs   from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { publicDir }   from '../../lib/paths.ts'
import source          from '../../lib/rendered/identity.json' with { type: 'json' }
import { themeConfig } from '../../lib/site.ts'
import { FACTS }       from '../common/facts.ts'

const config = themeConfig(FACTS)

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
})
