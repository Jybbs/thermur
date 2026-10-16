/**
 * A section of the site placed in the registry, under its label and its place
 * in the registry, counting from one.
 */
export interface Placed {
  label : string
  place : number
}

/**
 * A section of the site, meaning the folder under `docs/pages/` it opens on,
 * named `name`, and the question its pages answer, which the page met at a
 * missing address sets beside its label.
 */
export interface Section {
  name     : string
  question : string
}

/**
 * The sections of the site in the alphabetical order a registry keeps, each
 * one's place in the list numbering it from one.
 */
export const SECTIONS = [] as const satisfies readonly Section[]

/**
 * Labels a section by its folder's name with the first letter capitalized, the
 * label its cards and the page met at a missing address set.
 */
export const labelOf = ({ name }: Section): string => name.charAt(0).toUpperCase() + name.slice(1)

/**
 * Writes a section's place in reading order as two digits.
 */
export const numeralOf = (place: number): string => String(place).padStart(2, '0')

/**
 * Finds the section the page at `relativePath` sits in, the one named by the
 * folder the page sits under, or `undefined` for a page under no folder the
 * registry names.
 *
 * Args:
 *   relativePath : The page's path beneath the pages' folder.
 *   sections     : The registry the folder is looked up in.
 */
export function sectionOf(relativePath: string, sections: readonly Section[] = SECTIONS): Placed | undefined {
  const folder = relativePath.split('/', 2)
  const index  = folder.length === 2 ? sections.findIndex(section => section.name === folder[0]) : -1
  return index === -1 ? undefined : { label: labelOf(sections[index]!), place: index + 1 }
}
