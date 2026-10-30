import fs                from 'node:fs'
import path              from 'node:path'
import { fileURLToPath } from 'node:url'

function docsDir(metaUrl: string): string {
  return path.join(repoRoot(metaUrl), 'docs')
}

export function cacheDir(metaUrl: string, name: string): string {
  return path.join(repoRoot(metaUrl), '.cache', name)
}

/**
 * Joins the folder the package ships its faces in, one folder per family, which
 * the preview cards set their words from.
 */
export function fontsDir(metaUrl: string): string {
  return path.join(repoRoot(metaUrl), 'src', 'thermur', 'identity', 'fonts')
}

export function pagesDir(metaUrl: string): string {
  return path.join(docsDir(metaUrl), 'pages')
}

export function publicDir(metaUrl: string): string {
  return path.join(docsDir(metaUrl), 'public')
}

/**
 * Finds the repository's root, the nearest folder above the module at `metaUrl`
 * holding `pyproject.toml`, which a clone and an exported tree both carry.
 *
 * Raises:
 *   Error: Where no folder above the module holds `pyproject.toml`.
 */
export function repoRoot(metaUrl: string): string {
  let folder = path.dirname(fileURLToPath(metaUrl))
  while (!fs.existsSync(path.join(folder, 'pyproject.toml'))) {
    const parent = path.dirname(folder)
    if (parent === folder) throw new Error(`no pyproject.toml above ${metaUrl}`)
    folder = parent
  }
  return folder
}
