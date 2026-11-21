import { expect, it } from 'vitest'

it('gives a case a minute before it fails', ({ task }) => {
  expect(task.timeout).toBe(60_000)
})
