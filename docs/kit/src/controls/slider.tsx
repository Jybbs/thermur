import * as slider                                         from '@zag-js/slider'
import { mergeProps, normalizeProps, useMachine }          from '@zag-js/vue'
import { type PropType, computed, defineComponent, useId } from 'vue'

/**
 * Which way each arrow key steps a slider with stops.
 */
const ARROWS: Readonly<Record<string, -1 | 1>> = { ArrowDown: -1, ArrowLeft: -1, ArrowRight: 1, ArrowUp: 1 }
/**
 * The machine's event each arrow key across the track sends, since the
 * `slider` machine handles neither Up nor Down on a level slider.
 */
const ACROSS: Readonly<Record<string, 'DECREMENT' | 'INCREMENT'>> = { ArrowDown: 'DECREMENT', ArrowUp: 'INCREMENT' }
/**
 * How far past a stop a value may sit and still read as standing on it.
 */
const ON_STOP = 1e-6

/**
 * Finds the stop past `value` in `direction`, or the last stop that way where
 * none lies past it, `stops` sorted from the least.
 */
export function stepped(stops: readonly number[], value: number, direction: -1 | 1): number {
  const ahead = direction > 0
    ? stops.find(stop => stop > value + ON_STOP)
    : stops.findLast(stop => stop < value - ON_STOP)
  return ahead ?? (direction > 0 ? stops.at(-1)! : stops[0]!)
}

/**
 * A slider on the `slider` machine, setting one number between `min` and `max`
 * by a drag along its track or by the keys the ARIA slider pattern sets, Up
 * and Down among them, each value it takes reported through `change` and the
 * release that ends a drag through `commit`. Given `stops`, an arrow key steps
 * to the next stop rather than by `step` and reports through `commit`, so a
 * caller can glide there. It holds no value of its own, showing whatever
 * `value` the caller passes back, and a drawing laid beneath it can draw its
 * own track and thumb, the kit's stylesheet leaving both clear under
 * `kit-slider-over`.
 */
export const Slider = defineComponent({
  name  : 'Slider',
  props : {
    label  : { required: true, type: String },
    max    : { required: true, type: Number },
    min    : { required: true, type: Number },
    spoken : { default: undefined, type: Function as PropType<(value: number) => string> },
    step   : { default: 1, type: Number },
    stops  : { default: () => [], type: Array as PropType<readonly number[]> },
    value  : { required: true, type: Number }
  },
  emits : {
    change : (value: number) => typeof value === 'number',
    commit : (value: number) => typeof value === 'number'
  },
  setup(props, { emit }) {
    const id        = useId()
    let moved: number | undefined
    const service   = useMachine(slider.machine, computed(() => ({
      'aria-label'       : [props.label],
      'getAriaValueText' : props.spoken && (({ value }: slider.ValueTextDetails) => props.spoken!(value)),
      'id'               : id,
      'max'              : props.max,
      'min'              : props.min,
      'onValueChange'    : ({ value }: slider.ValueChangeDetails) => emit('change', moved = value[0]!),
      'onValueChangeEnd' : ({ value }: slider.ValueChangeDetails) => {
        emit('commit', moved ?? value[0]!)
        moved = undefined
      },
      'step'             : props.step,
      'thumbAlignment'   : 'center' as const,
      'value'            : [props.value]
    })))
    const api       = computed(() => slider.connect(service, normalizeProps))
    const onKeydown = (event: KeyboardEvent): void => {
      const direction = ARROWS[event.key]
      const across    = ACROSS[event.key]
      if (props.stops.length > 0 && direction !== undefined) {
        event.preventDefault()
        emit('commit', stepped(props.stops, props.value, direction))
      } else if (across !== undefined) {
        event.preventDefault()
        service.send({ index: 0, step: service.prop(event.shiftKey ? 'largeStep' : 'step'), type: across })
      }
    }
    return () => (
      <div {...api.value.getRootProps()} class="kit-slider">
        <div {...api.value.getControlProps()} class="kit-slider-control">
          <div {...api.value.getTrackProps()} class="kit-slider-track">
            <div {...api.value.getRangeProps()} class="kit-slider-range" />
          </div>
          <div {...mergeProps(api.value.getThumbProps({ index: 0 }), { onKeydown })} class="kit-slider-thumb">
            <input {...api.value.getHiddenInputProps({ index: 0 })} />
          </div>
        </div>
      </div>
    )
  }
})
