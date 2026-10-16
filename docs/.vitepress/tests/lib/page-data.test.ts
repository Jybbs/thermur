import fs   from 'node:fs'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { animated, pageData } from '../../lib/page-data.ts'
import { pagesDir }           from '../../lib/paths.ts'
import source                 from '../../lib/rendered/identity.json' with { type: 'json' }

const REPOSITORY = 'https://github.com/example/flock'

describe('pageData', () => {
  it('lays the front page out as the home page with its class, its button, and its title alone', () => {
    expect(pageData({ frontmatter: { kept: true }, relativePath: 'index.md' }, REPOSITORY)).toEqual({
      frontmatter: {
        hero      : { actions: [{ link: REPOSITORY, text: 'The Code on GitHub', theme: 'brand' }] },
        kept      : true,
        layout    : 'home',
        pageClass : 'front-page'
      },
      titleTemplate: false
    })
  })

  it.for(['methods/index.md', 'methods/alarm.md'])('leaves %s as it was written', relativePath => {
    expect(pageData({ frontmatter: {}, relativePath }, REPOSITORY)).toBeUndefined()
  })

  it('finds the front page opening on the tagline the identity declares', () => {
    const front = fs.readFileSync(path.join(pagesDir(import.meta.url), 'index.md'), 'utf8')
    expect(front.split('\n\n')[1]).toBe(`*${source.tagline.text}*`)
  })
})

describe('animated', () => {
  it.for([
    [{ relativePath: 'index.md' }, true],
    [{ isNotFound: true, relativePath: '404.md' }, true],
    [{ relativePath: 'methods/alarm.md' }, false]
  ] as const)('reports whether %j moves on its own', ([page, moving]) => {
    expect(animated(page)).toBe(moving)
  })
})
