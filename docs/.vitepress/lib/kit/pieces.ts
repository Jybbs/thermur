import manifest from '../../../kit/package.json' with { type: 'json' }

/**
 * One example of a piece at work, which the catalog runs live: its name and
 * what a reader sees the piece do in it.
 */
interface Example {
  name : string
  says : string
}

/**
 * One piece of the kit as the declaration beside its code records it: its
 * name, the examples the catalog runs, the names a widget imports it by, the
 * Zag.js machine it runs on, none where no machine covers it, what it
 * replaces, and when a widget reads it.
 */
interface Declared {
  examples : readonly Example[]
  exports  : readonly string[]
  machine  : string | null
  name     : string
  replaces : string
  when     : string
}

/**
 * One piece as the catalog walks it: its declaration beside its key, the name
 * its declaration's file opens on, the part of the kit it sits in, the folder
 * holding it, and each file of code beside its declaration, as its path under
 * `docs/kit/src/`.
 */
export interface Piece extends Declared {
  key     : string
  part    : string
  sources : readonly string[]
}

/**
 * The folder under `docs/` holding the kit's code, which the paths the
 * globs below return open on.
 */
const SRC = '../../../kit/src/'

/**
 * Loads the text of each file of the kit's code, by its path under
 * `docs/kit/src/`, only once a reader asks to see it.
 */
export const SOURCES: Readonly<Record<string, () => Promise<string>>> = Object.fromEntries(
  Object.entries(import.meta.glob<string>('../../../kit/src/*/*.{ts,tsx}', { import: 'default', query: '?raw' }))
    .map(([path, load]) => [path.slice(SRC.length), load])
)

/**
 * Each piece's declaration, in the path order The Kit's table lists them in,
 * every file of code sharing the declaration's name read as its source, set
 * on the declaration the glob loaded, which no other module reads.
 */
export const PIECES: readonly Piece[] = Object.entries(
  import.meta.glob<Declared>('../../../kit/src/*/*.json', { eager: true, import: 'default' })
).toSorted(([a], [b]) => a.localeCompare(b)).map(([path, declared]) => {
  const [part, file] = path.slice(SRC.length).split('/') as [string, string]
  const key          = file.replace(/\.json$/u, '')
  return Object.assign(declared, {
    key,
    part,
    sources : Object.keys(SOURCES).filter(source => source.replace(/\.tsx?$/u, '') === `${part}/${key}`)
  })
})

/**
 * Each part of the kit, as the folder holding its pieces, in the order the
 * catalog walks them.
 */
export const PARTS: readonly string[] = [...new Set(PIECES.map(piece => piece.part))]

/**
 * The name the site reaches the kit under, as its manifest declares it.
 */
const PACKAGE = manifest.name

/**
 * Reads the piece whose key is `key`.
 *
 * Raises:
 *   Error: Where no piece's key is `key`.
 */
export function pieceOf(key: string): Piece {
  const piece = PIECES.find(one => one.key === key)
  if (piece === undefined) throw new Error(`the kit holds no piece ${key}`)
  return piece
}

/**
 * The line a widget writes to take up every name `piece` exports, sorted as
 * an import block sorts them.
 */
export const lineOf = (piece: Piece): string =>
  `import { ${piece.exports.toSorted().join(', ')} } from '${PACKAGE}'`

/**
 * The modules of the site reading each piece, by the piece's key, each as its
 * path under `docs/`.
 */
export type Readers = Readonly<Record<string, readonly string[]>>
