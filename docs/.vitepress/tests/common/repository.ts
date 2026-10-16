import fs                from 'node:fs'
import os                from 'node:os'
import path              from 'node:path'
import { pathToFileURL } from 'node:url'

import { test as base } from 'vitest'

/**
 * A scratch repository a test writes, holding a `pyproject.toml` at its root,
 * with the address of a module inside it, which every function under `lib/`
 * that reads the disk resolves the repository from.
 */
export interface Repository {
  metaUrl : string
  root    : string
  write   : (file: string, text: string) => void
}

/**
 * The manifest a scratch repository opens with, declaring each field the site
 * reads.
 */
export const MANIFEST = [
  '[project]',
  'description = "Flies a flock."',
  'version     = "2.7.1"',
  '',
  '[project.urls]',
  'Homepage   = "https://example.org/flock/"',
  'Repository = "https://github.com/example/flock"'
].join('\n')

/**
 * The site's suite with a `repository` fixture writing a scratch repository
 * under the machine's temporary folder, named for the test that uses it and
 * removed once that test ends.
 */
export const test = base.extend('repository', ({ task }, { onCleanup }): Repository => {
  const root  = fs.mkdtempSync(path.join(os.tmpdir(), `thermur-site-${task.id}-`))
  const write = (file: string, text: string): void => {
    fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true })
    fs.writeFileSync(path.join(root, file), text)
  }
  onCleanup(() => fs.rmSync(root, { force: true, recursive: true }))
  write('pyproject.toml', MANIFEST)
  return { metaUrl: pathToFileURL(path.join(root, 'docs', '.vitepress', 'config.ts')).href, root, write }
})
