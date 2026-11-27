import type { Box }                     from '../common.ts'
import { type Bird, type Scene, flock } from './drawing.ts'

/**
 * A source and the citation of it a reader followed, either of which may be
 * missing.
 */
interface Citation {
  cite   : HTMLElement | null
  source : HTMLElement | null
}

/**
 * Finds the element of `page` whose id is `id`, and none for an empty `id`.
 */
const byId = (page: Document, id: string): HTMLElement | null =>
  id ? page.querySelector(`#${CSS.escape(id)}`) : null

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
 * Marks as current the source of `citation` and every marker leading to it,
 * and the citation a reader followed with the return leading back to it,
 * clearing whatever `page` marked before.
 */
function mark(page: Document, { cite, source }: Citation): void {
  for (const each of page.querySelectorAll<HTMLElement>('.source[data-current], a.cite[data-current]')) {
    delete each.dataset.current
  }
  for (const each of page.querySelectorAll('a.cite[aria-current], a.back[aria-current]')) {
    each.removeAttribute('aria-current')
  }
  if (!source?.matches('.source')) return
  source.dataset.current = ''
  for (const each of page.querySelectorAll<HTMLElement>(`a.cite[href='#${CSS.escape(source.id)}']`)) {
    each.dataset.current = ''
  }
  if (!cite) return
  cite.setAttribute('aria-current', 'location')
  source.querySelector(`a.back[href='#${CSS.escape(cite.id)}']`)?.setAttribute('aria-current', 'location')
}

/**
 * Reads the id the address of `page` names, and none where its fragment holds
 * a malformed escape.
 */
function named(page: Document): string {
  try {
    return decodeURIComponent(page.location.hash.slice(1))
  } catch {
    return ''
  }
}

/**
 * Reads the scene the Sources section `section` lays its flock out over, in
 * the section's own pixels whatever transform the page sets on it.
 */
function sceneOf(section: HTMLElement): Scene {
  const frame = section.getBoundingClientRect()
  const scale = frame.width / section.offsetWidth || 1
  const boxOf = (rect: DOMRect): Box => ({
    x0 : (rect.left - frame.left) / scale,
    x1 : (rect.right - frame.left) / scale,
    y0 : (rect.top - frame.top) / scale,
    y1 : (rect.bottom - frame.top) / scale
  })
  const sources = [...section.querySelectorAll<HTMLElement>('.source')]
  const cells   = sources.map(each => boxOf(each.querySelector('.source-number')!.getBoundingClientRect()))
  const heading = section.querySelector('.sources-title')!
  const words   = section.ownerDocument.createRange()
  words.selectNodeContents(heading)
  return {
    row   : boxOf(heading.getBoundingClientRect()),
    title : boxOf(words.getBoundingClientRect()),
    width : section.offsetWidth,

    anchors: sources.map(source => ({
      box     : boxOf(source.querySelector('.source-numeral')!.getBoundingClientRect()),
      current : 'current' in source.dataset
    })),

    rail: {
      ...cells[0]!,
      y0 : Math.min(...cells.map(cell => cell.y0)),
      y1 : Math.max(...cells.map(cell => cell.y1))
    }
  }
}

/**
 * Lays the flock of the Sources section `section` out, drawing from `glyphs`
 * traced birds, and hands each layout to `drawn`, again in the next frame after
 * the section changes size or another source becomes current. The source a
 * reader reaches through a marker, a return, or the page's address becomes
 * current. Returns the function that releases every listener, observer, and
 * frame it holds.
 */
export function watchSources(
  section : HTMLElement,
  glyphs  : number,
  drawn   : (birds: Bird[]) => void
): () => void {
  const page  = section.ownerDocument
  const abort = new AbortController()
  let queued  = 0
  const soon  = (): void => {
    cancelAnimationFrame(queued)
    queued = requestAnimationFrame(() => drawn(flock(sceneOf(section), glyphs)))
  }
  const follow = (citation: Citation): void => {
    mark(page, citation)
    soon()
  }
  const arrive = (): void => {
    const target = byId(page, named(page))
    if (target?.matches('a.cite, .source:not([data-current])')) follow(citationOf(target))
  }
  const resize = new ResizeObserver(soon)
  page.addEventListener('click', event => {
    const link = event.target instanceof Element ? event.target.closest<HTMLElement>('a.cite, a.back') : null
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
