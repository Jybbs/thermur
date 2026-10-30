import path              from 'node:path'
import { pathToFileURL } from 'node:url'

import { expect } from 'vitest'

import * as paths from '../../lib/paths.ts'
import { test }   from '../common/repository.ts'

test('every folder joins onto the root of the repository holding the module', ({ repository }) => {
  const { metaUrl, root } = repository
  expect(paths.repoRoot(metaUrl)).toBe(root)
  expect(paths.cacheDir(metaUrl, 'vitest')).toBe(path.join(root, '.cache', 'vitest'))
  expect(paths.fontsDir(metaUrl)).toBe(path.join(root, 'src', 'thermur', 'identity', 'fonts'))
  expect(paths.pagesDir(metaUrl)).toBe(path.join(root, 'docs', 'pages'))
  expect(paths.publicDir(metaUrl)).toBe(path.join(root, 'docs', 'public'))
})

test('a module under no manifest names the module it searched from', () => {
  const metaUrl = pathToFileURL(path.join(path.parse(process.cwd()).root, 'config.ts')).href
  expect(() => paths.repoRoot(metaUrl)).toThrow(`no pyproject.toml above ${metaUrl}`)
})
