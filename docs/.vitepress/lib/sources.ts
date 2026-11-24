import { type Box, mulberry, smoothstep } from './common.ts'
import { seriesProperty }                from './identity.ts'

/**
 * One source's number: its numeral's box, and whether its source is current.
 */
interface Anchor {
  box     : Box
  current : boolean
}

/**
 * One bird of the flock: traced bird `glyph` of the mark with its head at `x`
 * and `y`, turned to `angle` and `length` pixels long, colored at `t` along
 * the mark's gradient, or in the lead hue where it labels the current source.
 */
export interface Bird {
  angle    : number
  glyph    : number
  length   : number
  t        : number
  x        : number
  y        : number
  current? : boolean
}

/**
 * A source and the citation of it a reader followed, either of which may be
 * missing.
 */
interface Citation {
  cite   : HTMLElement | null
  source : HTMLElement | null
}

/**
 * What the flock is laid out around, in the Sources section's own pixels: each
 * source's number, the column the numbers stand in from the first source to the
 * last, the heading's row and its text, and the section's width.
 */
export interface Scene {
  anchors : readonly Anchor[]
  rail    : Box
  row     : Box
  title   : Box
  width   : number
}

/**
 * A point of the line the flock follows, with its share of the way along it.
 */
interface Spine {
  along : number
  x     : number
  y     : number
}

/**
 * The pixels the flock stops short of the section's right edge.
 */
const GAP = 72
/**
 * The traced bird of the mark each number labels.
 */
const LABEL = 26
/**
 * The pixels between two points of the flock's line up the column, and so the
 * flock's density along it.
 */
const PITCH = 15
/**
 * The seed every layout draws from, so a page at one width lays out alike on
 * every load.
 */
const SEED = 23

/**
 * Finds the element of `page` whose id is `id`, and none for an empty `id`.
 */
const byId = (page: Document, id: string): HTMLElement | null => id ? page.querySelector(`#${CSS.escape(id)}`) : null

/**
 * The center and reach of a bird's body, the circle a layout keeps clear of
 * the numbers, the heading's text, and the other birds.
 */
export const body = ({ angle, length, x, y }: Bird): { r: number, x: number, y: number } =>
  ({ r: length * 0.42, x: x - Math.cos(angle) * length * 0.55, y: y - Math.sin(angle) * length * 0.55 })

/**
 * Lists the source and the citation `target` names: a marker names itself and
 * the source it leads to, a return the marker it leads back to and the source
 * holding it, and a source itself alone.
 */
function citationOf(target: HTMLElement): Citation {
  const linked = byId(target.ownerDocument, target.getAttribute('href')?.slice(1) ?? '')
  if (target.matches('a.cite')) return { cite: target, source: linked }
  if (target.matches('a.back')) return { cite: linked, source: target.closest('.source') }
  return { cite: null, source: target }
}

/**
 * Whether two birds' bodies come within `pad` pixels of each other.
 */
export function crowds(a: Bird, b: Bird, pad: number): boolean {
  const [p, q] = [body(a), body(b)]
  return Math.hypot(p.x - q.x, p.y - q.y) < p.r + q.r + pad
}

/**
 * The fill of a bird: the lead hue where it labels the current source, and
 * otherwise its place along the mark's gradient, from the ink at the flock's
 * foot to Fire Lord at its head, eased as the mark eases it.
 */
export const fillOf = ({ current, t }: Bird): string => current
  ? 'var(--th-lead)'
  : `color-mix(in srgb, var(${seriesProperty(0)}) ${(smoothstep(0, 0.8, t) * 100).toFixed(1)}%, var(--th-ink))`

/**
 * Lays the flock out over `scene`, drawing from `glyphs` traced birds: it
 * rises up the column of numbers from the last source to the first and turns
 * right along the heading's row past its text, as the mark's flock rises and
 * turns. Each number labels a bird whose head sits just short of it, and every
 * other bird keeps clear of the numbers, the heading's text, the entries, and
 * the other birds, or is left out.
 */
export function flock(scene: Scene, glyphs: number): Bird[] {
  const { anchors, rail, row, title, width } = scene
  const draw     = mulberry(SEED)
  const end      = width - GAP
  const line     = spine(rail, end, (row.y0 + title.y1) / 2 + 4)
  const at       = (y: number): number => line.reduce((best, each) => Math.abs(each.y - y) < Math.abs(best.y - y) ? each : best).along
  const numbered = anchors.map(anchor => labeled(anchor, 0.1 + 0.9 * at(anchor.box.y0)))
  const heading  = { ...title, x1: title.x1 + 8 }
  const loose: Bird[] = []
  for (let tries = 0; tries < line.length * 12 && loose.length < line.length * 1.2; tries++) {
    const place  = Math.floor(draw() * line.length)
    const point  = line[place]!
    const ahead  = line[Math.min(line.length - 1, place + 1)]!
    const angle  = Math.atan2(ahead.y - point.y, ahead.x - point.x) + (draw() - 0.5) * 0.5
    const spread = (draw() + draw() + draw() - 1.5) * (14 - 6 * point.along)
    const bird   = {
      angle  : angle,
      glyph  : Math.floor(draw() * glyphs),
      length : 8 + 7 * point.along + draw() * 2,
      t      : 0.1 + 0.9 * point.along + (draw() - 0.5) * 0.1,
      x      : point.x - Math.sin(angle) * spread,
      y      : point.y + Math.cos(angle) * spread
    }
    const { r, x, y } = body(bird)
    const inRail = x - r >= rail.x0 + 2 && x + r <= rail.x1 - 2 && y + r <= rail.y1
    const inRow  = y - r >= row.y0 + 2 && y + r <= rail.y0 - 4 && x + r <= end + 16
    if ((inRail || inRow) && !touches(body(bird), heading, 4) && loose.every(other => !crowds(bird, other, 1))) loose.push(bird)
  }
  const clear = loose.filter(each =>
    anchors.every(anchor => !touches(body(each), anchor.box, 3)) && numbered.every(other => !crowds(each, other, 3)))
  return [...numbered, ...clear]
}

/**
 * The bird a source's number labels, its head just short of the numeral and
 * turned up to the right as the mark's birds are, colored at `t`.
 */
const labeled = ({ box, current }: Anchor, t: number): Bird =>
  ({ angle: -0.5, current, glyph: LABEL, length: 15, t, x: box.x0 - 3, y: box.y0 + (box.y1 - box.y0) * 0.4 })

/**
 * Marks as current the source of `citation` and every marker leading to it,
 * and the citation a reader followed with the return leading back to it,
 * clearing whatever `page` marked before.
 */
function mark(page: Document, { cite, source }: Citation): void {
  for (const each of page.querySelectorAll<HTMLElement>('.source[data-current], a.cite[data-current]')) delete each.dataset.current
  for (const each of page.querySelectorAll('a.cite[aria-current], a.back[aria-current]')) each.removeAttribute('aria-current')
  if (!source?.matches('.source')) return
  source.dataset.current = ''
  for (const each of page.querySelectorAll<HTMLElement>(`a.cite[href='#${CSS.escape(source.id)}']`)) each.dataset.current = ''
  if (!cite) return
  cite.setAttribute('aria-current', 'location')
  source.querySelector(`a.back[href='#${CSS.escape(cite.id)}']`)?.setAttribute('aria-current', 'location')
}

/**
 * Reads the scene the Sources section `section` lays its flock out over, in
 * the section's own pixels whatever transform the page sets on it.
 */
function sceneOf(section: HTMLElement): Scene {
  const frame   = section.getBoundingClientRect()
  const scale   = frame.width / section.offsetWidth || 1
  const boxOf   = (r: DOMRect): Box => ({
    x0 : (r.left - frame.left) / scale,
    x1 : (r.right - frame.left) / scale,
    y0 : (r.top - frame.top) / scale,
    y1 : (r.bottom - frame.top) / scale
  })
  const sources = [...section.querySelectorAll<HTMLElement>('.source')]
  const cells   = sources.map(source => boxOf(source.querySelector('.source-number')!.getBoundingClientRect()))
  const heading = section.querySelector('.sources-title')!
  const words   = section.ownerDocument.createRange()
  words.selectNodeContents(heading)
  return {
    anchors : sources.map(source => ({
      box     : boxOf(source.querySelector('.source-numeral')!.getBoundingClientRect()),
      current : 'current' in source.dataset
    })),
    rail    : { ...cells[0]!, y0: Math.min(...cells.map(cell => cell.y0)), y1: Math.max(...cells.map(cell => cell.y1)) },
    row     : boxOf(heading.getBoundingClientRect()),
    title   : boxOf(words.getBoundingClientRect()),
    width   : section.offsetWidth
  }
}

/**
 * Samples the line the flock follows, one point every `PITCH` pixels up the
 * column `rail` from the last source to the first, then bending right into the
 * heading's row at `row` and running along it to `end`, each point carrying
 * its share of the way along the line.
 */
export function spine(rail: Box, end: number, row: number): Spine[] {
  const x      = rail.x0 + (rail.x1 - rail.x0) * 0.45
  const bend   = Math.min(80, (rail.y1 - row) * 0.3)
  const rise   = Math.max(40, Math.ceil((rail.y1 - 8 - row - bend) / PITCH))
  const points = [
    ...Array.from({ length: rise }, (_, k) => ({ x: x - 6 + 6 * k / (rise - 1), y: rail.y1 - 8 - (rail.y1 - 8 - row - bend) * k / (rise - 1) })),
    ...Array.from({ length: 24 }, (_, k) => {
      const s = (k + 1) / 24
      return { x: x + (end - x) * s * s * 0.6, y: row + bend * (1 - s) * (1 - s) }
    }),
    ...Array.from({ length: 16 }, (_, k) => ({ x: x + (end - x) * (0.6 + 0.4 * (k + 1) / 16), y: row }))
  ]
  const steps  = points.map((each, k) => k ? Math.hypot(each.x - points[k - 1]!.x, each.y - points[k - 1]!.y) : 0)
  const total  = steps.reduce((sum, each) => sum + each, 0)
  let walked   = 0
  return points.map(({ x: across, y: down }, k) => ({ along: (walked += steps[k]!) / total, x: across, y: down }))
}

/**
 * Whether the circle at `x` and `y` of radius `r` comes within `pad` pixels of
 * `box`.
 */
export function touches({ r, x, y }: { r: number, x: number, y: number }, box: Box, pad: number): boolean {
  const dx = Math.max(box.x0 - pad - x, 0, x - box.x1 - pad)
  const dy = Math.max(box.y0 - pad - y, 0, y - box.y1 - pad)
  return dx * dx + dy * dy < r * r
}

/**
 * Lays the flock of the Sources section `section` out, drawing from `glyphs`
 * traced birds, and hands each layout to `drawn`, again in the next frame after
 * the section changes size or another source becomes current. The source a
 * reader reaches through a marker, a return, or the page's address becomes
 * current. Returns the function that releases every listener, observer, and
 * frame it holds.
 */
export function watchSources(section: HTMLElement, glyphs: number, drawn: (birds: Bird[]) => void): () => void {
  const page   = section.ownerDocument
  const abort  = new AbortController()
  let queued   = 0
  const soon   = (): void => {
    cancelAnimationFrame(queued)
    queued = requestAnimationFrame(() => drawn(flock(sceneOf(section), glyphs)))
  }
  const follow = (citation: Citation): void => {
    mark(page, citation)
    soon()
  }
  const arrive = (): void => {
    const target = byId(page, decodeURIComponent(page.location.hash.slice(1)))
    if (target?.matches('a.cite') || target?.matches('.source:not([data-current])')) follow(citationOf(target))
  }
  const resize = new ResizeObserver(soon)
  page.addEventListener('click', event => {
    const link = (event.target as Element | null)?.closest?.<HTMLElement>('a.cite, a.back')
    if (link) follow(citationOf(link))
  }, { signal: abort.signal })
  page.defaultView?.addEventListener('hashchange', arrive, { signal: abort.signal })
  resize.observe(section)
  arrive()
  soon()
  return () => {
    abort.abort()
    resize.disconnect()
    cancelAnimationFrame(queued)
  }
}
