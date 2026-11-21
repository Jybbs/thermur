import { execFileSync, spawnSync } from 'node:child_process'
import fs                          from 'node:fs'
import os                          from 'node:os'
import path                        from 'node:path'

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

/**
 * Copies every file git tracks or would track into `folder`, keeping each
 * file's mode and a link as a link, commits it there to `main`, links the
 * site's packages in, and switches to a branch named `change`.
 *
 * Returns:
 *   A function that commits a newline added to the file at the path it is
 *   given and lists the site's test files `vitest --changed main` selects
 *   there, under the site's own configuration.
 */
function copied(folder: string): (file: string) => Set<string> {
  // Reads no git configuration of the machine's own, whose hooks or signing a commit would otherwise run.
  const env = { ...process.env, GIT_CONFIG_GLOBAL: os.devNull, GIT_CONFIG_NOSYSTEM: '1' }
  const run = (command: string, args: string[], cwd = folder): string =>
    execFileSync(command, args, { cwd, encoding: 'utf8', env })
  const git = (...args: string[]): string =>
    run('git', ['-c', 'user.email=thermur@example.com', '-c', 'user.name=Thermur', ...args])

  const listed = run('git', ['ls-files', '--cached', '--exclude-standard', '--others', '-z'], root)
  for (const file of listed.split('\0').filter(name => name && fs.existsSync(path.join(root, name)))) {
    fs.cpSync(path.join(root, file), path.join(folder, file), { preserveTimestamps: true, verbatimSymlinks: true })
  }
  fs.symlinkSync(path.join(root, 'docs', 'node_modules'), path.join(folder, 'docs', 'node_modules'))
  git('init', '--initial-branch=main', '--quiet')
  fs.writeFileSync(path.join(folder, '.git', 'info', 'exclude'), '/docs/node_modules\n', { flag: 'a' })
  git('add', '--all')
  git('commit', '--message=Copy', '--quiet')
  git('switch', '--create', 'change', '--quiet')

  return file => {
    fs.appendFileSync(path.join(folder, file), '\n')
    git('commit', '--all', '--message=Change', '--quiet')
    return new Set(run(
      path.join(folder, 'docs', 'node_modules', '.bin', 'vitest'),
      ['list', '--changed', 'main', '--filesOnly', '--config', '.vitepress/config/vitest.config.ts'],
      path.join(folder, 'docs')
    ).split('\n').filter(Boolean))
  }
}

test.for([
  [
    'docs/.vitepress/lib/moving.ts',
    [
      '.vitepress/tests/lib/kit/readers.test.ts',
      '.vitepress/tests/lib/moving.test.ts',
      '.vitepress/tests/lib/page-data.test.ts',
      '.vitepress/tests/theme/components.test.ts',
      '.vitepress/tests/theme/pause.test.ts'
    ]
  ],
  [
    'docs/.vitepress/lib/math.ts',
    [
      '.vitepress/tests/config.test.ts',
      '.vitepress/tests/lib/kit/readers.test.ts',
      '.vitepress/tests/lib/math.test.ts',
      '.vitepress/tests/lib/page-data.test.ts'
    ]
  ],
  [
    'docs/pages/reference/kit.md',
    [
      '.vitepress/tests/config.test.ts',
      '.vitepress/tests/lib/kit/pieces.test.ts',
      '.vitepress/tests/lib/site.test.ts'
    ]
  ]
] as const)(
  'vitest --changed adds to its list for the contributor guide exactly the test files reaching %s',
  // Each case copies the repository and starts Vitest twice, measured at 3.2 to 3.8 seconds alone and 4.1 to 6.1 beside the whole suite at a load average near 20, past Vitest's default 5.
  { timeout: 60_000 },
  ([file, selected], { repository }) => {
    // Vitest lists the tests mounting a kit control in the copy for any change, so the list for the guide, which no test reads, sets them aside.
    const changed = copied(repository.root)

    expect(changed('.github/CONTRIBUTING.md').symmetricDifference(changed(file))).toEqual(new Set(selected))
  }
)
