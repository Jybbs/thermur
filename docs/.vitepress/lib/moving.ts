import type { PageData } from 'vitepress'

import { FRONT_PAGE } from './head.ts'

/**
 * The pages carrying a drawing that moves on its own, the front page and the
 * Data page's drawing of the threat's front.
 */
const MOVING: ReadonlySet<string> = new Set([FRONT_PAGE, 'data/index.md'])

/**
 * Reports whether `page` carries a drawing that moves on its own, one of the
 * pages `MOVING` names or the page met at a missing address.
 */
export const animated = (page: Pick<PageData, 'isNotFound' | 'relativePath'>): boolean =>
  page.isNotFound === true || MOVING.has(page.relativePath)
