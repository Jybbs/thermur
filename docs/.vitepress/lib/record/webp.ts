/**
 * The size of a canvas or a frame, in pixels.
 */
export interface Size {
  height : number
  width  : number
}

/**
 * The chunks of one still WebP file by their four-letter names, in order, each
 * with its payload.
 */
type Chunk = readonly [name: string, payload: Uint8Array]

/**
 * Writes `value` into `bytes` at `at` as `size` little-endian bytes.
 */
const put = (bytes: Uint8Array, at: number, value: number, size: number): void => {
  for (let k = 0; k < size; k++) bytes[at + k] = (value >>> (8 * k)) & 0xff
}

/**
 * Writes one chunk: its name, its payload's length, the payload, and a pad
 * byte where the length is odd.
 */
const chunk = (name: string, payload: Uint8Array): Uint8Array => {
  const out = new Uint8Array(8 + payload.length + (payload.length & 1))
  for (let k = 0; k < 4; k++) out[k] = name.codePointAt(k)!
  put(out, 4, payload.length, 4)
  out.set(payload, 8)
  return out
}

/**
 * Joins `parts` end to end.
 */
const join = (parts: readonly Uint8Array[]): Uint8Array => {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0))
  let at    = 0
  for (const part of parts) {
    out.set(part, at)
    at += part.length
  }
  return out
}

/**
 * Splits a still WebP file into its chunks, failing on a file that is not one.
 */
export function chunks(file: Uint8Array): Chunk[] {
  const text = (at: number): string => String.fromCodePoint(...file.subarray(at, at + 4))
  if (text(0) !== 'RIFF' || text(8) !== 'WEBP') throw new Error('Not a WebP file')
  const view = new DataView(file.buffer, file.byteOffset, file.byteLength)
  const out  : Chunk[] = []
  for (let at = 12; at + 8 <= file.length;) {
    const size = view.getUint32(at + 4, true)
    out.push([text(at), file.subarray(at + 8, at + 8 + size)])
    at += 8 + size + (size & 1)
  }
  return out
}

/**
 * Assembles still WebP files, each `size` pixels, into one animated WebP that
 * shows each for `milliseconds` and loops without end. A frame's own header
 * chunk is dropped, since the animation's header names the canvas.
 */
export function animate(frames: readonly Uint8Array[], { height, width }: Size, milliseconds: number): Uint8Array {
  const head = new Uint8Array(10)
  head[0]    = 0x02
  put(head, 4, width - 1, 3)
  put(head, 7, height - 1, 3)
  const anim = new Uint8Array(6)
  const body = frames.map(frame => {
    const data = join(chunks(frame).filter(([name]) => name !== 'VP8X').map(([name, payload]) => chunk(name, payload)))
    const meta = new Uint8Array(16)
    put(meta, 6, width - 1, 3)
    put(meta, 9, height - 1, 3)
    put(meta, 12, milliseconds, 3)
    meta[15]   = 0b10
    return chunk('ANMF', join([meta, data]))
  })
  const inner = join([new TextEncoder().encode('WEBP'), chunk('VP8X', head), chunk('ANIM', anim), ...body])
  return join([chunk('RIFF', inner).subarray(0, 8), inner])
}

/**
 * Reads the size of a still lossy WebP file from its bitstream header.
 */
export function sizeOf(file: Uint8Array): Size {
  const [, payload] = chunks(file).find(([name]) => name === 'VP8 ') ?? ['', new Uint8Array(0)]
  if (payload.length < 10) throw new Error('No lossy bitstream in the WebP file')
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength)
  return { height: view.getUint16(8, true) & 0x3fff, width: view.getUint16(6, true) & 0x3fff }
}
