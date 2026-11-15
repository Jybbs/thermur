import * as toggle                                   from '@zag-js/toggle'
import { normalizeProps, useMachine }                from '@zag-js/vue'
import { type SlotsType, computed, defineComponent } from 'vue'

/**
 * A button that stays pressed until pressed again, on the `toggle` machine, so
 * a screen reader hears its state through `aria-pressed`. It holds no state of
 * its own, reporting each press through `change` and showing whatever
 * `pressed` the caller passes back, and its slot draws the face the press
 * shows.
 */
export const Toggle = defineComponent({
  name  : 'Toggle',
  props : {
    label   : { required: true, type: String },
    pressed : { required: true, type: Boolean }
  },
  emits : { change: (pressed: boolean) => typeof pressed === 'boolean' },
  slots : Object as SlotsType<{ default: { pressed: boolean } }>,
  setup(props, { emit, slots }) {
    const service = useMachine(toggle.machine, computed(() => ({
      onPressedChange : (pressed: boolean) => emit('change', pressed),
      pressed         : props.pressed
    })))
    const api     = computed(() => toggle.connect(service, normalizeProps))
    return () => (
      <button {...api.value.getRootProps()} aria-label={props.label} class="kit-toggle">
        {slots.default({ pressed: api.value.pressed })}
      </button>
    )
  }
})
