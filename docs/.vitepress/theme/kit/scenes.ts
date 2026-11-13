import { Chip, Radios, Select, Slider, Toggle, idle, markup, onMotion, pause, paused, still, tween }  from '@thermur/kit'
import { type Component, type Ref, defineComponent, h, onBeforeUnmount, onMounted, ref, watchEffect } from 'vue'

import { lerp }                                                 from '../../lib/common.ts'
import { FUELS, Front, GROUND, SNAPSHOT, SPAN, WINDS, reachOf } from '../../lib/kit/front.ts'
import { Flight }                                               from '../../lib/kit/flight.ts'
import { useLoop }                                              from '../../lib/kit/loop.ts'
import { STEPS, part, playing, turn, useSteps }                 from './parts.ts'

/**
 * The stops the front's time takes, one at each snapshot of its run.
 */
const STOPS = Array.from({ length: SPAN / SNAPSHOT + 1 }, (_, place) => place * SNAPSHOT)

/**
 * The seconds of the front's run a second of play covers.
 */
const PACE = 12

/**
 * Calls `changed` with each width the element `host` holds takes once it
 * mounts, releasing the watch once the component unmounts.
 */
function useWidth(host: Ref<HTMLElement | undefined>, changed: (width: number) => void): void {
  let release = idle
  onMounted(() => {
    release = markup.widthOf(host.value!, changed)
  })
  onBeforeUnmount(() => release())
}

/**
 * Sets a front into the element `host` holds once it mounts, drawn to the
 * meters `reach` returns whenever they change and laid out again at each width
 * the element takes, so a page rendered on the server builds no drawing.
 */
function useFront(host: Ref<HTMLElement | undefined>, reach: () => number): void {
  let front!: Front
  onMounted(() => {
    front = new Front()
    host.value!.append(front.node)
    watchEffect(() => front.draw(reach()))
  })
  useWidth(host, width => front.fit(width))
}

/**
 * The controls a widget sets around a drawing of a fire front: a menu inside
 * a sentence picking the fuel, radios picking the wind, a toggle playing the
 * front's run, and a slider setting the seconds since ignition, each moving
 * the same front.
 */
const Controls = defineComponent({
  name: 'KitControlsScene',
  setup() {
    const fuel    = ref(0)
    const wind    = ref(1)
    const seconds = ref(45)
    const moved   = (value: number): void => {
      seconds.value = value
    }
    const host    = ref<HTMLElement>()
    useFront(host, () => reachOf(fuel.value, wind.value, seconds.value))
    const loop    = useLoop((step) => {
      seconds.value = Math.min(SPAN, seconds.value + step * PACE)
      if (seconds.value === SPAN) loop.run(false)
    })
    const play    = (on: boolean): void => {
      if (on && seconds.value === SPAN) seconds.value = 0
      if (on && loop.standing.value) seconds.value = SPAN
      loop.run(on)
    }
    return () => h('div', { class: 'kit-catalog-scene' }, [
      h('div', { class: 'kit-catalog-plate', ref: host }),
      part('select', h('p', { class: 'kit-catalog-sentence' }, [
        'Burn ',
        h(Select, {
          choices  : FUELS.map(one => one.words),
          label    : 'Fuel',
          picked   : fuel.value,
          onChange : (place: number) => fuel.value = place
        }),
        '.'
      ])),
      part('radios', h(Radios, {
        label    : 'Wind',
        options  : WINDS,
        value    : WINDS[wind.value]!.value,
        onChange : (value: string) => wind.value = WINDS.findIndex(one => one.value === value)
      })),
      part('toggle', playing('Play the front', loop.running.value, play)),
      part('slider', h(Slider, {
        label    : 'Seconds since ignition',
        max      : SPAN,
        min      : 0,
        spoken   : (at: number) => `${Math.round(at)} seconds`,
        stops    : STOPS,
        value    : seconds.value,
        onChange : moved,
        onCommit : moved
      }), true)
    ])
  }
})

/**
 * The chrome a widget sets around the same front: two chips reading out the
 * seconds since ignition and how far the front has spread, which a press
 * steps on a snapshot, and the heat ramp the burned ground is painted in,
 * whose steps a press copies.
 */
const Chrome = defineComponent({
  name: 'KitChromeScene',
  setup() {
    const seconds = ref(30)
    const host    = ref<HTMLElement>()
    const time    = ref<HTMLElement>()
    const reach   = ref<HTMLElement>()
    const meters  = (): number => reachOf(0, 1, seconds.value)
    const steps   = useSteps()
    useFront(host, meters)
    onMounted(() => watchEffect(() => {
      time.value!.textContent  = `${seconds.value} s`
      reach.value!.textContent = `${Math.round(meters())} m`
    }))
    return () => h('div', { class: 'kit-catalog-scene' }, [
      part('chip', [
        h(Chip, { name: 'Time', shape: `${SPAN} s`, value: time }),
        h(Chip, { name: 'Reach', shape: `${GROUND} m`, value: reach }),
        turn('Next snapshot', () => seconds.value = (seconds.value + SNAPSHOT) % (SPAN + SNAPSHOT))
      ], true),
      h('div', { class: 'kit-catalog-plate', ref: host }),
      part('copy', steps(), true)
    ])
  }
})

/**
 * The page's pause at work: a toggle turning it on and off beside a line
 * saying whether motion runs, read through the gate again whenever the pause
 * or the reduced-motion preference changes.
 */
export const Gate = defineComponent({
  name: 'KitGate',
  setup() {
    const on       = ref(false)
    const standing = ref(false)
    let release    = idle
    const read     = (): void => {
      on.value       = paused()
      standing.value = still()
    }
    onMounted(() => {
      read()
      release = onMotion(read)
    })
    onBeforeUnmount(() => release())
    return () => h('div', { class: 'kit-catalog-row' }, [
      h(Toggle, {
        class    : 'kit-catalog-turn',
        label    : 'Pause the motion on this page',
        pressed  : on.value,
        onChange : pause
      }, { default: ({ pressed }: { pressed: boolean }) => (pressed ? 'Resume' : 'Pause the page') }),
      h('span', { class: 'kit-catalog-said' }, standing.value ? 'Motion stands still' : 'Motion runs')
    ])
  }
})

/**
 * The motion a widget passes through the kit: the flock wheeling over the
 * ground while a toggle flies it and the gate lets it, the page's pause
 * beside it, and a press gathering the flock in or letting it loose again
 * over the theme's settling time.
 */
const Motion = defineComponent({
  name: 'KitMotionScene',
  setup() {
    const host     = ref<HTMLElement>()
    let flight!: Flight
    const gathered = ref(false)
    let seconds    = 0
    let stop       = idle
    const loop     = useLoop((step) => {
      seconds += step
      flight.draw(seconds)
    })
    const gather   = (): void => {
      stop()
      gathered.value = !gathered.value
      const from     = flight.gathered
      const to       = Number(gathered.value)
      stop           = tween((share) => {
        flight.gathered = lerp(from, to, share)
        flight.draw(seconds)
      }, 3)
    }
    onMounted(() => {
      flight = new Flight()
      host.value!.append(flight.node)
    })
    onBeforeUnmount(() => stop())
    return () => h('div', { class: 'kit-catalog-scene' }, [
      h('div', { class: 'kit-catalog-plate', ref: host }),
      part('gate', [playing('Fly the flock', loop.running.value, loop.run), h(Gate)], true),
      part('timing', h('div', { class: 'kit-catalog-row' }, [
        turn(gathered.value ? 'Let it loose' : 'Gather the flock', gather)
      ]), true)
    ])
  }
})

/**
 * The heat ramp drawn by the kit's markup as a row of bars across the width
 * the drawing stands at, written again each time that width changes, with the
 * width it was last drawn at written beneath.
 */
export const Width = defineComponent({
  name: 'KitWidth',
  setup() {
    const host = ref<HTMLElement>()
    useWidth(host, (width) => {
      const bar = (width - 8 * (STEPS.length - 1)) / STEPS.length
      host.value!.replaceChildren(markup.svg('svg', { 'aria-hidden': 'true', 'height': 56, 'viewBox': `0 0 ${width} 56`, 'width': width },
        ...STEPS.map((_, place) => markup.svg('rect', {
          fill   : `var(--th-heat-${place + 1})`,
          height : 22,
          rx     : 3,
          width  : bar,
          x      : place * (bar + 8),
          y      : 4
        })),
        markup.text(markup.FACE.mono, 11, `DRAWN AT ${width} PX`, 0, 48, { 'fill': markup.TONE.muted, 'letter-spacing': '0.1em' })))
    })
    return () => h('div', { class: 'kit-catalog-width', ref: host })
  }
})

/**
 * The drawing helpers at work, the heat ramp written at the stage's width.
 */
const Drawing = defineComponent(
  () => () => h('div', { class: 'kit-catalog-scene' }, [part('markup', h(Width), true)]),
  { name: 'KitDrawingScene' }
)

/**
 * The scene the stage sets out for each part of the kit, by the part's folder.
 */
export const SCENES: Readonly<Record<string, Component>> = { chrome: Chrome, controls: Controls, drawing: Drawing, motion: Motion }
