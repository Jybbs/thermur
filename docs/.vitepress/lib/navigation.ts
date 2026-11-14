import fs   from 'node:fs'
import path from 'node:path'

import type { DefaultTheme } from 'vitepress'

import { SECTIONS, type Section, labelOf } from './sections.ts'

/**
 * Reads the first heading of the page at `file`, which VitePress also reads as
 * the page's title.
 *
 * Raises:
 *   Error: Where the page carries no first-level heading.
 */
function titleOf(file: string): string {
  const [, title] = /^# (.+)$/mu.exec(fs.readFileSync(file, 'utf8')) ?? []
  if (title === undefined) throw new Error(`${file} carries no first-level heading`)
  return title
}

/**
 * Lists each section in the navigation bar under its label, leading to its
 * overview and marked active on any page beneath it.
 */
export function navOf(sections: readonly Section[] = SECTIONS): DefaultTheme.NavItemWithLink[] {
  return sections.map(section => ({
    activeMatch : `^/${section.name}/`,
    link        : `/${section.name}/`,
    text        : labelOf(section)
  }))
}

/**
 * Builds the sidebar each section's pages show, one group per section under its
 * label leading to its overview, listing every other page of the folder by its
 * first heading in alphabetical order.
 *
 * Args:
 *   pages    : The folder the pages sit under.
 *   sections : The registry each folder is read from.
 */
export function sidebarOf(pages: string, sections: readonly Section[] = SECTIONS): DefaultTheme.SidebarMulti {
  return Object.fromEntries(sections.map(section => {
    const folder = path.join(pages, section.name)
    const items  = fs.globSync('*.md', { cwd: folder, exclude: ['index.md'] })
      .map(file => ({
        link : `/${section.name}/${path.basename(file, '.md')}`,
        text : titleOf(path.join(folder, file))
      }))
      .toSorted((left, right) => left.text.localeCompare(right.text))
    return [`/${section.name}/`, [{ items, link: `/${section.name}/`, text: labelOf(section) }]]
  }))
}
