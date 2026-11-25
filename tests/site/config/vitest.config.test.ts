import { execFileSync } from 'node:child_process'
import path             from 'node:path'

import { expect, it } from 'vitest'

import vitest       from '../../../docs/.vitepress/config/vitest.config.ts'
import { repoRoot } from '../../../docs/.vitepress/lib/paths.ts'

const root = repoRoot(import.meta.url)

it('gives a case a minute before it fails', ({ task }) => {
  expect(task.timeout).toBe(60_000)
})

it('runs every test file git tracks under tests/', () => {
  const tracked = execFileSync('git', ['ls-files', '--', 'tests/*.test.ts', 'tests/*.test.tsx'], { cwd: root, encoding: 'utf8' })
  const missed  = tracked.split('\n').filter(Boolean).filter(file =>
    !vitest.test!.include!.some(glob => path.matchesGlob(path.join(root, file), path.join(root, 'docs', glob))))

  expect(missed).toEqual([])
})
