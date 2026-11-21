import path from 'node:path'

/**
 * A registry of two sections, which a page under either folder sits in, each
 * beside the question its pages answer.
 */
export const REGISTRY = [
  { name: 'data', question: 'Where the plume simulations come from and how to fetch them.' },
  { name: 'methods', question: 'How each drone reads its neighbors and the air around it.' }
] as const
/**
 * The path of each overview page under the site's pages, relative to that
 * folder, read off a glob, which `vitest --changed` counts as a file every test
 * importing this module reads.
 */
const overviews = new Set(
  Object.keys(import.meta.glob('./**/index.md', { base: '../../../pages/', query: '?raw' }))
    .map(key => path.posix.normalize(key))
)

/**
 * Lists the overview page each folder of `folders` lacks under the site's
 * pages, each as its path from there.
 */
export function missingOverviews(folders: readonly string[]): string[] {
  return folders.map(folder => path.posix.join('.', folder, 'index.md')).filter(file => !overviews.has(file))
}
