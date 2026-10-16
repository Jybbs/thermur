import type { DefaultTheme } from 'vitepress'

import { TITLE }             from './head.ts'
import type { Identity }     from './identity.ts'
import type { ProjectFacts } from './manifest.ts'
import source                from './rendered/identity.json' with { type: 'json' }

/**
 * Builds the default theme's configuration, meaning the wordmark in the
 * navigation bar in place of the site's title and a link to the repository
 * beside it.
 *
 * Args:
 *   facts    : What the manifest states, the repository's address among it.
 *   identity : The identity the wordmark's files are named in.
 */
export function themeConfig(facts: ProjectFacts, identity: Identity = source): DefaultTheme.Config {
  return {
    logo        : wordmarkImage(identity),
    siteTitle   : false,
    socialLinks : [{ icon: 'github', link: facts.repository }]
  }
}

/**
 * Names the wordmark's file in each scheme beneath the site's root, as the
 * image the default theme's logo and hero take, under the site's title.
 */
export function wordmarkImage(identity: Identity = source): DefaultTheme.ThemeableImage {
  const { wordmark } = identity.marks
  return { alt: TITLE, dark: `/${wordmark.dark}`, light: `/${wordmark.light}` }
}
