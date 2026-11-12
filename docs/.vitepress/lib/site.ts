import type { DefaultTheme } from 'vitepress'

import { TITLE }             from './head.ts'
import type { Identity }     from './identity.ts'
import type { ProjectFacts } from './manifest.ts'
import { navOf, sidebarOf }  from './navigation.ts'
import source                from './rendered/identity.json' with { type: 'json' }

/**
 * Builds the default theme's configuration, meaning the wordmark in the
 * navigation bar in place of the site's title, each section in the navigation
 * bar and its pages in the sidebar, and a link to the repository.
 *
 * Args:
 *   facts    : What the manifest states, the repository's address among it.
 *   pages    : The folder the pages sit under, which the sidebar is read from.
 *   identity : The identity the wordmark's files are named in.
 */
export function themeConfig(facts: ProjectFacts, pages: string, identity: Identity = source): DefaultTheme.Config {
  return {
    logo        : wordmarkImage(identity),
    nav         : navOf(),
    sidebar     : sidebarOf(pages),
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
