import { describe, expect, it } from 'vitest'

import { animate, chunks, sizeOf } from '../../../lib/record/webp.ts'

/**
 * Writes a still WebP file holding `held`, each as its name and payload.
 */
function still(...held: [string, number[]][]): Uint8Array {
  const body = held.flatMap(([name, payload]) => Array.from(name, letter => letter.codePointAt(0)!)
    .concat([payload.length & 0xff, 0, 0, 0], payload, payload.length & 1 ? [0] : []))
  const size = body.length + 4
  return Uint8Array.from([0x52, 0x49, 0x46, 0x46, size & 0xff, size >> 8, 0, 0, 0x57, 0x45, 0x42, 0x50, ...body])
}

describe('chunks', () => {
  it('lists each chunk with its payload, past a pad byte', () => {
    const listed = chunks(still(['VP8 ', [1, 2, 3]], ['EXIF', [9, 9]]))
    expect(listed.map(([name, payload]) => [name, [...payload]])).toEqual([['VP8 ', [1, 2, 3]], ['EXIF', [9, 9]]])
  })

  it('fails on a file that is no WebP', () => {
    expect(() => chunks(new Uint8Array(16))).toThrow('Not a WebP file')
  })
})

describe('sizeOf', () => {
  it('reads the size from the lossy bitstream header', () => {
    expect(sizeOf(still(['VP8 ', [0, 0, 0, 0x9d, 0x01, 0x2a, 5, 0, 7, 0]]))).toEqual({ height: 7, width: 5 })
  })

  it('fails on a file with no lossy bitstream', () => {
    expect(() => sizeOf(still(['EXIF', [1, 2]]))).toThrow('No lossy bitstream')
  })
})

describe('animate', () => {
  const frames = [still(['VP8X', [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]], ['VP8 ', [1, 2, 3]]), still(['VP8 ', [4, 5]])]
  const made   = animate(frames, { height: 2, width: 3 }, 100)
  const listed = chunks(made)

  it('opens on a header naming the animation and the canvas', () => {
    const [name, payload] = listed[0]!
    expect(name).toBe('VP8X')
    expect(payload[0]).toBe(0x02)
    expect([...payload.subarray(4, 10)]).toEqual([2, 0, 0, 1, 0, 0])
  })

  it('loops without end, then holds one frame chunk for each source', () => {
    expect(listed.map(([name]) => name)).toEqual(['VP8X', 'ANIM', 'ANMF', 'ANMF'])
    expect([...listed[1]![1]]).toEqual([0, 0, 0, 0, 0, 0])
  })

  it('shows each frame for its duration, without its own header chunk', () => {
    const [, payload] = listed[2]!
    expect([...payload.subarray(6, 15)]).toEqual([2, 0, 0, 1, 0, 0, 100, 0, 0])
    const inner = payload.subarray(16)
    expect(String.fromCodePoint(...inner.subarray(0, 4))).toBe('VP8 ')
    expect(inner.length).toBe(8 + 4)
  })

  it('sizes its RIFF header to the file', () => {
    expect(new DataView(made.buffer).getUint32(4, true)).toBe(made.length - 8)
  })
})
