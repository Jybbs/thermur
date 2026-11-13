import { type VNode }     from 'vue'
import { renderToString } from 'vue/server-renderer'

/**
 * The browser globals a page rendered on the server goes without.
 */
const ABSENT = ['ResizeObserver', 'document', 'getComputedStyle', 'matchMedia', 'requestAnimationFrame'] as const

/**
 * Renders `view` to HTML the way the site's build renders a page, with each
 * global `ABSENT` names taken off the global object, so a component that reads
 * any of them before it mounts fails the render. Each global returns once the
 * render settles.
 */
export async function served(view: VNode): Promise<string> {
  const kept = ABSENT.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)] as const)
  for (const name of ABSENT) Reflect.deleteProperty(globalThis, name)
  try {
    return await renderToString(view)
  }
  finally {
    for (const [name, held] of kept) if (held) Object.defineProperty(globalThis, name, held)
  }
}
