import * as radio                     from '@zag-js/radio-group'
import { normalizeProps, useMachine } from '@zag-js/vue'
import { type PropType, type SlotsType, type StyleValue, computed, defineComponent, useId } from 'vue'

/**
 * One radio of a group: the value it picks, the words it reads, and, where the
 * caller sets them, its title and where it stands over a drawing.
 */
export interface Option {
  value  : string
  words  : string
  style? : StyleValue
  title? : string
}

/**
 * A group of radios on the `radio-group` machine, each a label over a radio
 * input the browser keeps out of sight, so a press picks one, the arrow keys
 * move the choice around the group, and a screen reader hears the group's
 * name from `label`. A thumb the machine measures slides under the radio that
 * is on, and the `item` slot draws what stands before each radio's words. It
 * holds no choice of its own, showing whatever `value` the caller passes back.
 */
export const Radios = defineComponent({
  name  : 'Radios',
  props : {
    label   : { required: true, type: String },
    options : { required: true, type: Array as PropType<readonly Option[]> },
    value   : { required: true, type: String }
  },
  emits : { change: (value: string) => typeof value === 'string' },
  slots : Object as SlotsType<{ item?: Option }>,
  setup(props, { emit, slots }) {
    const id      = useId()
    const service = useMachine(radio.machine, computed(() => ({
      id,
      onValueChange : ({ value }: radio.ValueChangeDetails) => emit('change', value!),
      value         : props.value
    })))
    const api     = computed(() => radio.connect(service, normalizeProps))
    return () => (
      <div {...api.value.getRootProps()} class="kit-radios">
        <span {...api.value.getLabelProps()} class="visually-hidden">{props.label}</span>
        <span {...api.value.getIndicatorProps()} class="kit-radios-thumb" />
        {props.options.map(option => (
          <label {...api.value.getItemProps(option)} class="kit-radios-item" style={option.style} title={option.title}>
            {slots.item?.(option)}
            <span {...api.value.getItemTextProps(option)}>{option.words}</span>
            <input {...api.value.getItemHiddenInputProps(option)} />
          </label>
        ))}
      </div>
    )
  }
})
