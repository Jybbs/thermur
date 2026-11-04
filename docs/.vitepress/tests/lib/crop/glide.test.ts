import { describe, expect, it } from 'vitest'

import { Glide } from '../../../lib/crop/glide.ts'

describe('Glide', () => {
  it('closes on its target and comes to rest there without passing it', () => {
    const glide  = new Glide(0)
    glide.target = 1
    for (let k = 0; k < 600; k++) {
      glide.step(1 / 60)
      expect(glide.value).toBeLessThanOrEqual(1)
    }
    expect(glide.value).toBeCloseTo(1)
    expect(glide.velocity).toBeCloseTo(0)
  })

  it('bends at a new target taken mid-glide, with no jump in where it stands or how fast it moves', () => {
    const glide  = new Glide(0)
    glide.target = 1
    for (let k = 0; k < 20; k++) glide.step(1 / 60)
    const { value, velocity } = glide
    glide.target              = -1
    glide.step(1e-6)
    expect(glide.value).toBeCloseTo(value, 4)
    expect(glide.velocity).toBeCloseTo(velocity, 3)
  })

  it('stands at its target at rest once settled', () => {
    const glide  = new Glide(0)
    glide.target = 1
    glide.step(1 / 60)
    glide.settle()
    expect(glide).toMatchObject({ value: 1, velocity: 0 })
  })
})
