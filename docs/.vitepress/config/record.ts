import { type ChildProcess, spawn }                          from 'node:child_process'
import { createHash }                                        from 'node:crypto'
import { globSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir }                                            from 'node:os'
import { join, relative, resolve }                           from 'node:path'

import { animate, sizeOf } from '../lib/record/webp.ts'

/**
 * The frames a second the recording keeps, the width its screenshots keep of
 * the view's own, and the quality they are encoded at.
 */
const FPS     = 10
const QUALITY = 60
const SCALE   = 0.75

/**
 * The most frames one recording holds, which stops a loop that never closes.
 */
const MOST = 600

/**
 * The files the recording is made from, as globs under the docs folder, whose
 * hashes the stamp beside the recordings holds.
 */
const SOURCES = [
  '.vitepress/lib/front/composite.ts',
  '.vitepress/lib/roost/*.ts',
  '.vitepress/theme/roost.ts',
  '.vitepress/theme/styles/roost.css'
]

/**
 * Replaces the page's clock and frame requests with ones the recording
 * advances by hand, so each frame it keeps is one exact step of the page.
 */
const CLOCK = `(() => {
  let now = 0, next = 1, queue = new Map()
  performance.now = () => now
  window.requestAnimationFrame = callback => { queue.set(next, callback); return next++ }
  window.cancelAnimationFrame = id => { queue.delete(id) }
  window.__advance = milliseconds => {
    now += milliseconds
    const run = [...queue.values()]
    queue.clear()
    for (const callback of run) callback(now)
  }
})()`

/**
 * A box on the page, in pixels.
 */
interface Rect {
  height : number
  width  : number
  x      : number
  y      : number
}

/**
 * Sends one command over the DevTools socket and waits for its answer.
 */
class Page {
  #id = 0
  readonly #pending = new Map<number, (result: any) => void>()
  readonly #socket  : WebSocket

  constructor(socket: WebSocket) {
    this.#socket = socket
    socket.addEventListener('message', ({ data }) => {
      const message = JSON.parse(String(data))
      this.#pending.get(message.id)?.(message.result)
      this.#pending.delete(message.id)
    })
  }

  send(method: string, params: object = {}): Promise<any> {
    return new Promise(done => {
      this.#pending.set(++this.#id, done)
      this.#socket.send(JSON.stringify({ id: this.#id, method, params }))
    })
  }

  async value<T>(expression: string): Promise<T> {
    const { result } = await this.send('Runtime.evaluate', { expression, returnByValue: true })
    return result.value
  }
}

/**
 * Waits `milliseconds`.
 */
const pause = (milliseconds: number): Promise<void> => new Promise<void>(done => {
  setTimeout(done, milliseconds)
})

/**
 * Calls `task` until it returns a value, trying again every quarter second and
 * giving up after `tries` attempts.
 */
async function until<T>(task: () => Promise<T | null>, tries: number): Promise<T> {
  const value = await task().catch(() => null)
  if (value !== null) return value
  if (tries <= 1) throw new Error('Timed out waiting on the recording\'s page')
  await pause(250)
  return until(task, tries - 1)
}

/**
 * Starts a process and resolves with the first line of its output that
 * matches `ready`, along with the process.
 */
function started(command: string, args: string[], ready: RegExp): Promise<{ child: ChildProcess, line: RegExpMatchArray }> {
  return new Promise((done, fail) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] })
    const read  = (chunk: Buffer): void => {
      const line = String(chunk).match(ready)
      if (line) done({ child, line })
    }
    child.stdout.on('data', read)
    child.stderr.on('data', read)
    child.on('exit', code => {
      fail(new Error(`${command} exited with ${code}`))
    })
  })
}

/**
 * Keeps a screenshot of `rect` and steps the page one frame, until the view's
 * veil has fallen and risen again, which closes its loop, or `MOST` frames are
 * kept.
 */
async function capture(page: Page, rect: Rect, frames: Uint8Array[] = [], fallen = false): Promise<Uint8Array[]> {
  const veil = Number(await page.value<string>('getComputedStyle(document.querySelector(\'.roost\')).getPropertyValue(\'--veil\')'))
  const low  = fallen || veil < 0.5
  if ((low && veil > 0.999) || frames.length >= MOST) return frames
  const { data } = await page.send('Page.captureScreenshot', { clip: { ...rect, scale: SCALE }, format: 'webp', quality: QUALITY })
  frames.push(Buffer.from(data, 'base64'))
  await page.value(`window.__advance(${1000 / FPS})`)
  return capture(page, rect, frames, low)
}

/**
 * Records one scheme's flight as an animated WebP.
 */
async function record(browser: string, origin: string, scheme: 'dark' | 'light'): Promise<Uint8Array> {
  const { child, line } = await started(browser, [
    '--headless=new', '--mute-audio', '--no-sandbox', '--hide-scrollbars', '--remote-debugging-port=0',
    `--user-data-dir=${mkdtempSync(join(tmpdir(), 'thermur-68-chrome-'))}`, 'about:blank'
  ], /ws:\/\/127\.0\.0\.1:(\d+)\//u)
  try {
    const target = await (await fetch(`http://127.0.0.1:${line[1]}/json/new?about:blank`, { method: 'PUT' })).json()
    const socket = new WebSocket(target.webSocketDebuggerUrl)
    await new Promise<void>(open => {
      socket.addEventListener('open', () => {
        open()
      })
    })
    const page   = new Page(socket)
    await page.send('Emulation.setDeviceMetricsOverride', { deviceScaleFactor: 1, height: 900, mobile: false, width: 1280 })
    await page.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `localStorage.setItem('vitepress-theme-appearance', '${scheme}');${CLOCK}`
    })
    await page.send('Page.navigate', { url: `${origin}/roost` })
    const rect   = await until(() => page.value<Rect | null>(
      '(() => { const box = document.querySelector(\'.roost-stage canvas\')?.getBoundingClientRect(); return box && box.width ? { height: box.height, width: box.width, x: box.x, y: box.y } : null })()'
    ), 120)
    const frames = await capture(page, rect)
    return animate(frames, sizeOf(frames[0]!), 1000 / FPS)
  } finally {
    child.kill()
  }
}

/**
 * Writes the stamp: one line for each source, its SHA-256 beside its path
 * from the repository's root.
 */
function stamp(docs: string): string {
  return SOURCES.flatMap(pattern => globSync(pattern, { cwd: docs }).toSorted()).map(file =>
    `${createHash('sha256').update(readFileSync(join(docs, file))).digest('hex')}  docs/${file}\n`).join('')
}

const docs   = resolve(import.meta.dirname, '../..')
const out    = join(docs, 'pages/public/readme')
const port   = 4173
const browser = process.env.CHROMIUM ?? globSync(`${process.env.PLAYWRIGHT_BROWSERS_PATH ?? '/ms-playwright'}/chromium-*/chrome-linux*/chrome`)[0]
if (!browser) throw new Error('No Chromium found: set CHROMIUM to its path')
const { child: server } = await started(join(docs, 'node_modules/.bin/vitepress'), ['preview', '--port', String(port)], /localhost:\d+/u)
try {
  const [light, dark] = await Promise.all((['light', 'dark'] as const).map(scheme => record(browser, `http://localhost:${port}`, scheme)))
  writeFileSync(join(out, 'flight-light.webp'), light!)
  writeFileSync(join(out, 'flight-dark.webp'), dark!)
  writeFileSync(join(out, 'flight.stamp'), stamp(docs))
  console.log(`Recorded ${relative(docs, out)}/flight-{light,dark}.webp`)
} finally {
  server.kill()
}
