import { expect } from 'vitest'

import { projectFacts }   from '../../lib/manifest.ts'
import { MANIFEST, test } from '../common/repository.ts'

test('the base and the host come from the homepage, and the version from the project', ({ repository }) => {
  expect(projectFacts(repository.metaUrl)).toEqual({
    base        : '/flock/',
    description : 'Flies a flock.',
    homepage    : 'https://example.org/flock/',
    host        : 'example.org',
    repository  : 'https://github.com/example/flock',
    version     : '2.7.1'
  })
})

test.for([
  ['project.description', MANIFEST.replace('description = "Flies a flock."', '')],
  ['project.urls.Homepage', MANIFEST.replace('Homepage   = "https://example.org/flock/"', '')],
  ['project.urls.Repository', MANIFEST.replace('Repository = "https://github.com/example/flock"', '')],
  ['project.version', MANIFEST.replace('version     = "2.7.1"', '')]
] as const)('a manifest without %s names the key it lacks', ([key, manifest], { repository }) => {
  repository.write('pyproject.toml', manifest)
  expect(() => projectFacts(repository.metaUrl)).toThrow(`no ${key} in `)
})

test('a manifest writing its urls as a value names the key read through them', ({ repository }) => {
  const flat = MANIFEST.replace(/\[project\.urls\][^]*$/u, 'urls = "https://example.org/flock/"')
  repository.write('pyproject.toml', flat)
  expect(() => projectFacts(repository.metaUrl)).toThrow('no project.urls.')
})

test('a homepage without its closing slash fails by name', ({ repository }) => {
  repository.write('pyproject.toml', MANIFEST.replace('/flock/"', '/flock"'))
  expect(() => projectFacts(repository.metaUrl)).toThrow('must end in a slash')
})
