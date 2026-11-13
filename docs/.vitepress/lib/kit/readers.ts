import { glob, readFile } from 'node:fs/promises'
import path               from 'node:path'

import type { PageData } from 'vitepress'
import { parseSync }     from 'vite'

import manifest         from '../../../kit/package.json' with { type: 'json' }
import type { Readers } from './pieces.ts'

/**
 * The page the kit's catalog stands on, as VitePress names it.
 */
export const KIT_PAGE = 'reference/kit.md'

/**
 * The modules under `docs/` that may read the kit, meaning the theme and the
 * pure modules beside it, leaving out the catalog's own widget and the
 * modules it draws through, which set every piece out.
 */
const READING = ['.vitepress/theme/**/*.ts', '.vitepress/lib/**/*.ts']
const LEFT    = ['.vitepress/lib/kit/**', '.vitepress/theme/kit/**']

/**
 * The key of each piece the kit declares under `docs/kit/src/`, beside the
 * names it exports, read off the declarations themselves, since the
 * configuration this runs in resolves no glob of Vite's.
 */
async function declaredIn(docs: string): Promise<Map<string, readonly string[]>> {
  const declared = new Map<string, readonly string[]>()
  for await (const file of glob('kit/src/*/*.json', { cwd: docs })) {
    const { exports } = JSON.parse(await readFile(path.join(docs, file), 'utf8')) as { exports: readonly string[] }
    declared.set(path.basename(file, '.json'), exports)
  }
  return declared
}

/**
 * The key of every piece of `declared` that `source` imports a value of, read off
 * its import declarations as Vite's parser reads them, leaving out a type it
 * imports, which renders nothing of the piece.
 */
function readIn(file: string, source: string, declared: Map<string, readonly string[]>): string[] {
  const { program } = parseSync(file, source)
  return program.body.flatMap((node) => {
    if (node.type !== 'ImportDeclaration' || node.importKind === 'type' || node.source.value !== manifest.name) return []
    return node.specifiers.flatMap((specifier) => {
      if (specifier.type !== 'ImportSpecifier' || specifier.importKind === 'type') return []
      const name = specifier.imported.type === 'Identifier' ? specifier.imported.name : specifier.imported.value
      return [...declared].filter(([, exports]) => exports.includes(name)).map(([key]) => key)
    })
  })
}

/**
 * Reads every module of the site importing a piece of the kit, under the
 * piece's key, each module as its path under `docs`, sorted, and every piece
 * no module reads under an empty list.
 *
 * Args:
 *   docs : The folder holding the site.
 */
export async function readersOf(docs: string): Promise<Readers> {
  const declared = await declaredIn(docs)
  const readers  = new Map([...declared.keys()].map(key => [key, new Set<string>()]))
  for await (const file of glob(READING, { cwd: docs, exclude: LEFT })) {
    for (const key of readIn(file, await readFile(path.join(docs, file), 'utf8'), declared)) readers.get(key)!.add(file)
  }
  return Object.fromEntries([...readers].map(([key, files]) => [key, [...files].toSorted()]))
}

/**
 * Sets the modules reading each piece on the frontmatter of the kit's page,
 * under `kit`, for its catalog to count, leaving every other page alone.
 *
 * Args:
 *   page : The page VitePress is about to render.
 *   docs : The folder holding the site.
 */
export async function kitData(
  page : Pick<PageData, 'frontmatter' | 'relativePath'>,
  docs : string
): Promise<Partial<PageData> | undefined> {
  if (page.relativePath !== KIT_PAGE) return undefined
  return { frontmatter: { ...page.frontmatter, kit: await readersOf(docs) } }
}
