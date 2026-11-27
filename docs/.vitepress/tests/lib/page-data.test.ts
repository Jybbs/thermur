import { describe, expect, it } from 'vitest'

import front        from '../../../pages/index.md?raw'
import { pageData } from '../../lib/page-data.ts'
import { docsDir }  from '../../lib/paths.ts'
import source       from '../../lib/rendered/identity.json' with { type: 'json' }

const REPOSITORY = 'https://github.com/example/flock'
const DOCS       = docsDir(import.meta.url)

/**
 * Declares to `vitest --changed` each file `pageData` reads off the site
 * through `kitData`, meaning each piece's declaration and each module
 * `READING` names outside `LEFT`, so a change to one selects this test.
 */
import.meta.glob([
  '../../../kit/src/*/*.json',
  '../../lib/**/*.ts',
  '../../theme/**/*.ts',
  '!../../lib/kit/**',
  '!../../theme/kit/**'
], { query: '?raw' })

describe('pageData', () => {
  it('lays the front page out as the home page with its class, its button, and its title alone', async () => {
    expect(await pageData({ frontmatter: { kept: true }, relativePath: 'index.md' }, REPOSITORY, DOCS)).toEqual({
      frontmatter: {
        hero      : { actions: [{ link: REPOSITORY, text: 'The Code on GitHub', theme: 'brand' }] },
        kept      : true,
        layout    : 'home',
        pageClass : 'front-page'
      },
      titleTemplate: false
    })
  })

  it('lays the roost page out in the default page layout with a class of its own, keeping its frontmatter', async () => {
    expect(await pageData({ frontmatter: { kept: true }, relativePath: 'roost.md' }, REPOSITORY, DOCS)).toEqual({
      frontmatter: { kept: true, layout: 'page', pageClass: 'roost-page' }
    })
  })

  it.for(['methods/index.md', 'methods/alarm.md'])('leaves %s as it was written', async (relativePath) => {
    expect(await pageData({ frontmatter: {}, relativePath }, REPOSITORY, DOCS)).toBeUndefined()
  })

  it('sets the modules reading each piece on the kit’s page, keeping what its frontmatter held', async () => {
    const page = await pageData({ frontmatter: { kept: true }, relativePath: 'reference/kit.md' }, REPOSITORY, DOCS)
    expect(page!.frontmatter!.kept).toBe(true)
    expect(page!.frontmatter!.kit.slider).toContain('.vitepress/theme/threat.ts')
  })

  it('finds the front page opening on the tagline the identity declares', () => {
    expect(front.split('\n\n')[1]).toBe(`*${source.tagline.text}*`)
  })
})
