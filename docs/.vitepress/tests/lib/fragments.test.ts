import { expect } from 'vitest'

import { githubSlug, strayFragments } from '../../lib/fragments.ts'
import { test }                        from '../common/repository.ts'

const README = [
  '<a id="start"></a>',
  '',
  '## ⌨️ Quick Start',
  '',
  '## 🐦‍⬛ The Flock',
  '',
  '## Usage',
  '',
  '## Usage',
  '',
  '## Tagged {#tagged}',
  ''
].join('\n')

test.for([
  ['⌨️ Quick Start', '\u{FE0F}-quick-start'],
  ['🐦‍⬛ The Flock', '\u{200D}-the-flock'],
  ['🔥 Thermur', '-thermur'],
  ['5. Monitoring & Metrics', '5-monitoring--metrics'],
  ['snake_case and kebab-case', 'snake_case-and-kebab-case']
] as const)('the heading %s takes the fragment GitHub writes', ([heading, fragment]) => {
  expect(githubSlug(heading)).toBe(fragment)
})

test('a fragment GitHub writes passes, percent-encoded or raw, across files and within one', async ({ repository }) => {
  repository.write('.github/README.md', README)
  repository.write('.github/CONTRIBUTING.md', [
    '[a](README.md#%EF%B8%8F-quick-start)',
    '[b](README.md#\u{FE0F}-quick-start)',
    '[c](README.md#%E2%80%8D-the-flock)',
    '[d](README.md#usage-1)',
    '[e](README.md#start)',
    '[f](#notes)',
    '[g](https://example.org/#nowhere)',
    '[h](../docs/page.html#nowhere)',
    '[i](README.md)',
    '',
    '## Notes'
  ].join('\n\n'))

  expect(await strayFragments(repository.metaUrl, ['.github/README.md', '.github/CONTRIBUTING.md'])).toEqual([])
})

test('a fragment GitHub writes for no heading is named beside its line', async ({ repository }) => {
  repository.write('.github/README.md', README)
  repository.write('.github/CONTRIBUTING.md', [
    '[a](README.md#-quick-start)',
    '[b](README.md#tagged)',
    '[c](#Notes)',
    '[d](GONE.md#notes)',
    '',
    '## Notes'
  ].join('\n\n'))

  expect(await strayFragments(repository.metaUrl, ['.github/CONTRIBUTING.md'])).toEqual([
    '.github/CONTRIBUTING.md:1: `README.md#-quick-start` names no heading GitHub writes in `.github/README.md`',
    '.github/CONTRIBUTING.md:3: `README.md#tagged` names no heading GitHub writes in `.github/README.md`',
    '.github/CONTRIBUTING.md:5: `#Notes` names no heading GitHub writes in `.github/CONTRIBUTING.md`',
    '.github/CONTRIBUTING.md:7: `GONE.md#notes` names no heading GitHub writes in `.github/GONE.md`'
  ])
})
