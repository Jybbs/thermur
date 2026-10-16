import type { ProjectFacts } from '../../lib/manifest.ts'

/**
 * What the site states about a project served from the root of its host, as the
 * suites that hand the facts in rather than read a manifest take them.
 */
export const FACTS: ProjectFacts = {
  base        : '/',
  description : 'Flies a flock.',
  homepage    : 'https://example.org/',
  host        : 'example.org',
  repository  : 'https://github.com/example/flock',
  version     : '2.7.1'
}
