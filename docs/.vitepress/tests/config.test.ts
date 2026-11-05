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
  'git ignores the whole %s VitePress resolves',
  (folder, { repository }) => {
    const held = path.relative(root, site[folder])
    const git  = (...args: string[]): number | null =>
      // Blanks `core.excludesFile`, whose patterns otherwise count toward `check-ignore`.
      spawnSync('git', ['-c', 'core.excludesFile=', ...args], { cwd: repository.root }).status

    repository.write('.gitignore', fs.readFileSync(path.join(root, '.gitignore'), 'utf8'))
    // A pattern ending in a slash matches no path missing from the disk, so this write creates the folder.
    repository.write(path.join(held, 'written'), '')

    expect(git('init', '--quiet')).toBe(0)
    expect(git('check-ignore', '--quiet', held)).toBe(0)
  }
)
