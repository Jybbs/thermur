import fs   from 'node:fs'
import os   from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { KIT_PAGE, kitData, readersOf } from '../../../lib/kit/readers.ts'
import { docsDir }                      from '../../../lib/paths.ts'

/**
 * Writes a scratch site holding one declaration of a piece exporting `Slider`
 * and `stepped`, beside each module `modules` names under `.vitepress/`.
 */
function scratch(modules: Readonly<Record<string, string>>): string {
  const docs = fs.mkdtempSync(path.join(os.tmpdir(), 'thermur-86-readers-'))
  fs.mkdirSync(path.join(docs, 'kit', 'src', 'controls'), { recursive: true })
  fs.writeFileSync(path.join(docs, 'kit', 'src', 'controls', 'slider.json'), JSON.stringify({ exports: ['Slider', 'stepped'] }))
  for (const [file, source] of Object.entries(modules)) {
    fs.mkdirSync(path.dirname(path.join(docs, '.vitepress', file)), { recursive: true })
    fs.writeFileSync(path.join(docs, '.vitepress', file), source)
  }
  return docs
}

describe('readersOf', () => {
  it('lists each module importing a value of a piece, sorted, past types, other packages, and the catalog', async () => {
    const docs = scratch({
      'lib/kit/walk.ts'    : "import { Slider } from '@thermur/kit'\n",
      'lib/one.ts'         : "import { stepped } from '@thermur/kit'\n",
      'lib/types.ts'       : "import type { Slider } from '@thermur/kit'\nimport { type Slider as Two } from '@thermur/kit'\n",
      'theme/kit/scene.ts' : "import { Slider } from '@thermur/kit'\n",
      'theme/other.ts'     : "import { Slider } from 'elsewhere'\nimport * as kit from '@thermur/kit'\n",
      'theme/two.ts'       : "import { 'Slider' as Named } from '@thermur/kit'\n"
    })
    expect(await readersOf(docs)).toEqual({ slider: ['.vitepress/lib/one.ts', '.vitepress/theme/two.ts'] })
  })

  it('reads the site’s own modules, counting a piece no module reads under an empty list', async () => {
    const readers = await readersOf(docsDir(import.meta.url))
    expect(readers.slider).toEqual(expect.arrayContaining(['.vitepress/theme/threat.ts', '.vitepress/theme/rules/neighbors.ts']))
    expect(Object.values(readers).flat().filter(file => file.includes('/kit/'))).toEqual([])
  })
})

describe('kitData', () => {
  it('sets the readers on the kit’s page alone, keeping what its frontmatter held', async () => {
    const docs = scratch({ 'theme/one.ts': "import { Slider } from '@thermur/kit'\n" })
    expect(await kitData({ frontmatter: { kept: true }, relativePath: KIT_PAGE }, docs))
      .toEqual({ frontmatter: { kept: true, kit: { slider: ['.vitepress/theme/one.ts'] } } })
    expect(await kitData({ frontmatter: {}, relativePath: 'data/index.md' }, docs)).toBeUndefined()
  })
})
