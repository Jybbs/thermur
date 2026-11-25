import { configDefaults, defineConfig } from 'vitest/config'

import { cacheDir, repoRoot } from '../lib/paths.ts'
import { shared }             from '../lib/vite.ts'

export default defineConfig({
  ...shared(),
  cacheDir : cacheDir(import.meta.url, 'vitest'),
  server   : { fs: { allow: [repoRoot(import.meta.url)] } },

  test: {
    css                 : { include: [/\.css\?raw$/u] },
    expect              : { requireAssertions: true },
    forceRerunTriggers  : [
      ...configDefaults.forceRerunTriggers,
      '**/*.lock',
      '**/.mise/**'
    ],
    include             : ['../tests/kit/**/*.test.{ts,tsx}', '../tests/site/**/*.test.ts'],
    pool                : 'threads',
    resolveSnapshotPath : (testPath, extension) => testPath + extension,
    restoreMocks        : true,
    sequence            : { shuffle: true },
    setupFiles          : ['../tests/site/common/matchers.ts'],
    testTimeout         : 60_000,
    unstubGlobals       : true,

    coverage: {
      include          : ['.vitepress/lib/**', '.vitepress/theme/**', 'kit/src/**'],
      reportOnFailure  : true,
      reporter         : ['text', 'html'],
      reportsDirectory : cacheDir(import.meta.url, 'coverage/site'),
      thresholds       : { branches: 95, functions: 95, lines: 95, statements: 95 }
    },

    reporters: process.env.GITHUB_ACTIONS === undefined
             ? ['default']
             : ['default', ['github-actions', { jobSummary: { enabled: false } }]]
  }
})
