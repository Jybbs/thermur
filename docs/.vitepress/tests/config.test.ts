import { spawnSync } from 'node:child_process'
import path          from 'node:path'

import { resolveConfig } from 'vitepress'
import { expect, it }    from 'vitest'

import ignored      from '../../../.gitignore?raw'
import vitest       from '../config/vitest.config.ts'
import { repoRoot } from '../lib/paths.ts'
import { test }     from './common/repository.ts'

const root     = repoRoot(import.meta.url)
const site     = await resolveConfig(path.join(root, 'docs'), 'build')
const triggers = vitest.test!.forceRerunTriggers!

/**
 * Declares to `vitest --changed` the configuration `resolveConfig` loads,
 * with every module it imports, and each file the configuration reads as it
 * resolves (each page, the public folder, the manifest, and the faces), so a
 * change to one selects this test.
 */
import.meta.glob('../config.ts')
import.meta.glob([
  '../../../pyproject.toml',
  '../../../src/thermur/identity/fonts/**',
  '../../pages/**/*.md',
  '../../pages/public/*'
], { query: '?raw' })

test.for(['cacheDir', 'outDir', 'tempDir'] as const)(
  'git ignores the whole %s VitePress resolves',
  (folder, { repository }) => {
    const held = path.relative(root, site[folder])
    const git  = (...args: string[]): number | null =>
      // Blanks `core.excludesFile`, whose patterns otherwise count toward `check-ignore`.
      spawnSync('git', ['-c', 'core.excludesFile=', ...args], { cwd: repository.root }).status

    repository.write('.gitignore', ignored)
    // A pattern ending in a slash matches a last segment only where a folder sits, so this write creates it.
    repository.write(path.join(held, 'written'), '')

    // Copies no template, whose `info/exclude` otherwise counts toward `check-ignore`.
    expect(git('init', '--quiet', '--template=')).toBe(0)
    expect(git('check-ignore', '--quiet', held)).toBe(0)
  }
)

it.for([
  ['docs/.vitepress/lib/moving.ts', false],
  ['docs/pages/reference/kit.md', false],
  ['.github/README.md', false],
  ['docs/bun.lock', true],
  ['uv.lock', true],
  ['.mise/config.toml', true]
] as const)('reruns the whole suite on a change to %s: %s', ([file, whole]) => {
  expect(triggers.some(trigger => path.matchesGlob(path.join(root, file), trigger))).toBe(whole)
})
