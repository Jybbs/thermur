import { defineConfig } from 'vitest/config'

import { identityModule } from './lib/identity.ts'
import { cacheDir }       from './lib/paths.ts'

/**
 * The milliseconds a case runs before it fails. The slowest cases spend their
 * time on the CPU, so on four workers the slowest takes half a second at a
 * 1-minute load average of 5, under 20 seconds as that average climbs to 160,
 * and 37 seconds as it climbs to 380.
 */
const TEST_TIMEOUT = 60_000

export default defineConfig({
  cacheDir : cacheDir(import.meta.url, 'vitest'),
  plugins  : [identityModule()],

  test: {
    expect              : { requireAssertions: true },
    include             : ['.vitepress/tests/**/*.test.ts'],
    pool                : 'threads',
    resolveSnapshotPath : (testPath, extension) => testPath + extension,
    restoreMocks        : true,
    sequence            : { shuffle: true },
    setupFiles          : ['.vitepress/tests/common/matchers.ts'],
    testTimeout         : TEST_TIMEOUT,
    unstubGlobals       : true,

    coverage: {
      include          : ['.vitepress/lib/**', '.vitepress/theme/**'],
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
