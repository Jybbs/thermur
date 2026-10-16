import fs   from 'node:fs'
import path from 'node:path'

import { parse } from 'smol-toml'

import { repoRoot } from './paths.ts'

/**
 * What the site states about the project, each read out of `pyproject.toml`,
 * where the base the site is served under is the homepage's own path and the
 * host it is served from is the homepage's own host.
 */
export interface ProjectFacts {
  base        : string
  description : string
  homepage    : string
  host        : string
  repository  : string
  version     : string
}

/**
 * Reads the string the manifest holds at the dotted `key`, naming the key and
 * the file where the manifest holds none.
 */
function required(manifest: Record<string, unknown>, key: string, file: string): string {
  const value = key.split('.').reduce<unknown>(
    (table, part) => typeof table === 'object' && table !== null
      ? (table as Record<string, unknown>)[part]
      : undefined,
    manifest
  )
  if (typeof value !== 'string') throw new Error(`no ${key} in ${file}`)
  return value
}

/**
 * Reads what the site states about the project out of the `pyproject.toml` at
 * the root of the repository holding the module at `metaUrl`.
 *
 * Raises:
 *   Error: Where the manifest holds no string at a key the site reads, or a
 *     homepage that does not end in a slash, which the base and every address
 *     joined onto the homepage depend on.
 */
export function projectFacts(metaUrl: string): ProjectFacts {
  const file     = path.join(repoRoot(metaUrl), 'pyproject.toml')
  const manifest = parse(fs.readFileSync(file, 'utf8'))
  const homepage = required(manifest, 'project.urls.Homepage', file)
  if (!homepage.endsWith('/')) throw new Error(`project.urls.Homepage in ${file} must end in a slash`)
  const { host, pathname } = new URL(homepage)

  return {
    base        : pathname,
    description : required(manifest, 'project.description', file),
    homepage    : homepage,
    host        : host,
    repository  : required(manifest, 'project.urls.Repository', file),
    version     : required(manifest, 'project.version', file)
  }
}
