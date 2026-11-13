import { idle, onMotion, still }                     from '@thermur/kit'
import { type Ref, onBeforeUnmount, onMounted, ref } from 'vue'

/**
 * A loop of frames a reader starts and stops, which passes the kit's gate by
 * running only while motion may run and stopping as the page's pause turns on
 * or the reader comes to prefer reduced motion.
 */
export interface Loop {
  running  : Ref<boolean>
  standing : Ref<boolean>
  run      : (on: boolean) => void
}

/**
 * Builds a loop calling `frame` with the seconds since the last frame, each
 * frame while it runs, which `run` starts and stops, `standing` holding
 * whether the gate stands motion still. Every frame and listener is released
 * once the component unmounts.
 */
export function useLoop(frame: (seconds: number) => void): Loop {
  const running  = ref(false)
  const standing = ref(false)
  let handle     = 0
  let last       = 0
  let release    = idle
  const tick     = (now: number): void => {
    frame((now - last) / 1000)
    last   = now
    handle = requestAnimationFrame(tick)
  }
  const run      = (on: boolean): void => {
    cancelAnimationFrame(handle)
    handle        = 0
    running.value = on && !standing.value
    if (!running.value) return
    last   = performance.now()
    handle = requestAnimationFrame(tick)
  }
  onMounted(() => {
    standing.value = still()
    release        = onMotion(() => {
      standing.value = still()
      if (standing.value) run(false)
    })
  })
  onBeforeUnmount(() => {
    run(false)
    release()
  })
  return { run, running, standing }
}
