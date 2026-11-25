import { execFileSync } from 'node:child_process'
import path             from 'node:path'

import { expect, it } from 'vitest'

import tsconfig     from '../../docs/.vitepress/tsconfig.json?raw'
import { repoRoot } from '../../docs/.vitepress/lib/paths.ts'

const root = repoRoot(import.meta.url)

it('type-checks every TypeScript file git tracks under tests/', () => {
  const tracked = execFileSync('git', ['ls-files', '--', 'tests/*.ts', 'tests/*.tsx'], { cwd: root, encoding: 'utf8' })
  const entries = (JSON.parse(tsconfig) as { include: string[] }).include
  // An entry naming a folder reaches every file below it, as `tsc` reads a folder in `include`.
  const globs   = entries.flatMap(entry => [entry, `${entry}/**`]).map(glob => path.join(root, 'docs', '.vitepress', glob))
  const missed  = tracked.split('\n').filter(Boolean).filter(file => !globs.some(glob => path.matchesGlob(path.join(root, file), glob)))

  expect(missed).toEqual([])
})
