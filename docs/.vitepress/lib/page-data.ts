import type { PageData } from 'vitepress'

import { FRONT_PAGE } from './head.ts'
import { kitData }    from './kit/readers.ts'

/**
 * Gives the front page the framework options it takes, so the page itself stays
 * bare Markdown: the default theme's home layout, its hero's one button leading
 * to the repository at `repository`, a class the theme sets the page through,
 * and its own title standing alone in the browser's tab. The kit's page takes
 * the modules reading each piece, per `kitData`, and every other page renders
 * as it was written.
 *
 * Args:
 *   page       : The page VitePress is about to render.
 *   repository : The address of the repository the manifest names.
 *   docs       : The folder holding the site.
 */
export async function pageData(
  page       : Pick<PageData, 'frontmatter' | 'relativePath'>,
  repository : string,
  docs       : string
): Promise<Partial<PageData> | undefined> {
  if (page.relativePath !== FRONT_PAGE) return await kitData(page, docs)
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
