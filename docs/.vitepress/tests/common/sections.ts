import fs   from 'node:fs'
import path from 'node:path'

import { pagesDir } from '../../lib/paths.ts'

/**
 * A registry of two sections, which a page under either folder sits in, each
 * beside the question its pages answer.
 */
export const REGISTRY = [
  { label: 'Data', name: 'data', question: 'Where the plume simulations come from and how to fetch them.' },
  { label: 'Methods', name: 'methods', question: 'How each drone reads its neighbors and the air around it.' }
] as const

/**
 * Lists the overview page each folder of `folders` lacks under the site's
 * pages.
 */
export function missingOverviews(folders: readonly string[]): string[] {
  const pages = pagesDir(import.meta.url)
  return folders.map(folder => path.join(pages, folder, 'index.md')).filter(file => !fs.existsSync(file))
}
