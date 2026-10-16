import type { PageData } from 'vitepress'

import { FRONT_PAGE } from './head.ts'

/**
 * Reports whether `page` carries a drawing that moves on its own, the front
 * page and the page met at a missing address being the two that do.
 */
export const animated = (page: Pick<PageData, 'isNotFound' | 'relativePath'>): boolean =>
  page.isNotFound === true || page.relativePath === FRONT_PAGE

/**
 * Gives the front page the framework options it takes, so the page itself stays
 * bare Markdown: the default theme's home layout, its hero's one button leading
 * to the repository at `repository`, a class the theme sets the page through,
 * and its own title standing alone in the browser's tab. Every other page
 * renders as it was written.
 *
 * Args:
 *   page       : The page VitePress is about to render.
 *   repository : The address of the repository the manifest names.
 */
export function pageData(
  page       : Pick<PageData, 'frontmatter' | 'relativePath'>,
  repository : string
): Partial<PageData> | undefined {
  if (page.relativePath !== FRONT_PAGE) return undefined
  return {
    titleTemplate : false,
    frontmatter   : {
      ...page.frontmatter,
      hero      : { actions: [{ link: repository, text: 'The Code on GitHub', theme: 'brand' }] },
      layout    : 'home',
      pageClass : 'front-page'
    }
  }
}
