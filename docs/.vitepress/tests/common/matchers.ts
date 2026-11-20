import { expect } from 'vitest'

import { CONTROL_LABEL, labeled } from './sheets.ts'

declare module 'vitest' {
  interface Matchers<R> {
    toReadAsPublished   : () => R
    toSetAsControlLabel : () => R
  }
}

/**
 * Each mark and word `writing-style.md` bans from a published sentence that a
 * pattern can find, beside what the failure names it.
 */
const BANNED = [
  [/—/u, 'an em-dash'],
  [/[a-z]; [a-z]/u, 'a semicolon joining two clauses'],
  [/\bfootgun\b/iu, 'the word footgun'],
  [/\bnamely\b/iu, 'the word namely']
] as const

expect.extend({
  toReadAsPublished(received: string) {
    const broken = BANNED.filter(([pattern]) => pattern.test(received)).map(([, name]) => name)
    return {
      message : () => `${JSON.stringify(received)} carries ${broken.join(' and ') || 'nothing banned'}`,
      pass    : broken.length === 0
    }
  },

  toSetAsControlLabel(received: Element) {
    const read = labeled(received)
    return {
      message : () => `${received.textContent} reads ${JSON.stringify(read)} from the kit’s label rule`,
      pass    : this.equals(read, CONTROL_LABEL)
    }
  }
})
