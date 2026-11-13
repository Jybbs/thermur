import { type Ref, onBeforeUnmount, onMounted, ref } from 'vue'

import { onMotion, still } from './gate.ts'
import { idle }            from './timing.ts'

/**
 * A loop of frames that passes the kit's gate, calling `frame` with the
 * seconds since the last frame on each frame while it runs, starting only
 * while motion may run and stopping as the page's pause turns on or the
 * reader comes to prefer reduced motion, or on the first frame to find
 * motion standing still. `moved` is called with whether motion stands still
 * each time the gate changes, and on that frame, before the loop stops.
 * `destroy` releases its frame and its listeners.
 */
export class Frames {
  #handle            = 0
  #last              = 0
  readonly #frame    : (seconds: number) => void
  readonly #gated    : () => void
  readonly #released : () => void

  constructor(frame: (seconds: number) => void, moved: (standing: boolean) => void = idle) {
    this.#frame    = frame
    this.#gated    = () => {
      const standing = still()
      moved(standing)
      if (standing) this.run(false)
    }
    this.#released = onMotion(this.#gated)
  }

  /**
   * Reads whether the loop runs.
   */
  get running(): boolean {
    return this.#handle !== 0
  }

  /**
   * Starts the loop where `on` is set and motion may run, or stops it,
   * returning whether it runs.
   */
  run(on: boolean): boolean {
    cancelAnimationFrame(this.#handle)
    this.#handle = 0
    if (!on || still()) return false
    this.#last   = performance.now()
    this.#handle = requestAnimationFrame(this.#tick)
    return true
  }

  /**
   * Stops the loop where motion has come to stand still, and otherwise asks
   * for the next frame before calling `frame`, so a frame that stops the loop
   * cancels the one it asked for.
   */
  readonly #tick = (now: number): void => {
    if (still()) {
      this.#gated()
      return
    }
    const seconds = (now - this.#last) / 1000
    this.#last    = now
    this.#handle  = requestAnimationFrame(this.#tick)
    this.#frame(seconds)
  }

  destroy(): void {
    this.run(false)
    this.#released()
  }
}

/**
 * A loop of frames a reader starts and stops inside a component, `running`
 * holding whether it runs and `standing` whether the gate stands motion
 * still.
 */
export interface Loop {
  running  : Ref<boolean>
  standing : Ref<boolean>
  run      : (on: boolean) => void
}

/**
 * Builds a loop calling `frame` with the seconds since the last frame, each
 * frame while it runs, which `run` starts and stops, on frames that start
 * once the component mounts and are released once it unmounts.
 */
export function useLoop(frame: (seconds: number) => void): Loop {
  const running  = ref(false)
  const standing = ref(false)
  let frames: Frames | undefined
  const run      = (on: boolean): void => {
    running.value = frames?.run(on) ?? false
  }
  onMounted(() => {
    standing.value = still()
    frames         = new Frames(frame, now => {
      standing.value = now
      if (now) running.value = false
    })
  })
  onBeforeUnmount(() => {
    frames?.destroy()
    running.value = false
  })
  return { run, running, standing }
}
