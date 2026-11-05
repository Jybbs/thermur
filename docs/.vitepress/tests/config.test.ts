import { spawnSync } from 'node:child_process'
import fs            from 'node:fs'
import path          from 'node:path'

import { resolveConfig } from 'vitepress'
import { expect }        from 'vitest'

import { repoRoot } from '../lib/paths.ts'
import { test }     from './common/repository.ts'

const root = repoRoot(import.meta.url)
const site = await resolveConfig(path.join(root, 'docs'), 'build')

test.for(['cacheDir', 'outDir', 'tempDir'] as const)(
  'git ignores what a build writes under the %s VitePress resolves',
  (folder, { repository }) => {
    const file = path.join(path.relative(root, site[folder]), 'written')
    const git  = (...args: string[]): number | null =>
      // Blanks the machine's own excludes file, so only what the repository lists counts.
      spawnSync('git', ['-c', 'core.excludesFile=', ...args], { cwd: repository.root }).status

    repository.write('.gitignore', fs.readFileSync(path.join(root, '.gitignore'), 'utf8'))

    expect(git('init', '--quiet')).toBe(0)
    expect(git('check-ignore', '--quiet', file)).toBe(0)
  }
)
