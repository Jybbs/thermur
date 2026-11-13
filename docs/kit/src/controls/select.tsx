import * as select                                         from '@zag-js/select'
import { normalizeProps, useMachine }                      from '@zag-js/vue'
import { type PropType, computed, defineComponent, useId } from 'vue'

/**
 * A menu on the `select` machine, a button that opens a list of `choices`
 * below it and picks one, with the keys and the focus the ARIA select-only
 * combobox pattern sets and the name `label` gives it for a screen reader. The
 * button stacks every choice's words in one cell and shows only the one
 * picked, so it stands at its widest choice's width and a pick moves nothing
 * around it. It holds no pick of its own, showing whichever choice `picked`
 * places and reporting each new pick's place through `change`.
 */
export const Select = defineComponent({
  name  : 'Select',
  props : {
    choices : { required: true, type: Array as PropType<readonly string[]> },
    label   : { required: true, type: String },
    picked  : { required: true, type: Number }
  },
  emits : { change: (place: number) => Number.isInteger(place) },
  setup(props, { emit }) {
    const id         = useId()
    const collection = computed(() => select.collection({ items: props.choices }))
    const service    = useMachine(select.machine, computed(() => ({
      'collection'    : collection.value,
      'id'            : id,
      'onValueChange' : ({ value }: select.ValueChangeDetails) => emit('change', props.choices.indexOf(value[0]!)),
      'positioning'   : { placement: 'bottom-start' as const, sameWidth: false },
      'value'         : [props.choices[props.picked]!]
    })))
    const api        = computed(() => select.connect(service, normalizeProps))
    return () => (
      <span {...api.value.getRootProps()} class="kit-select">
        <span {...api.value.getLabelProps()} class="visually-hidden">{props.label}</span>
        <button {...api.value.getTriggerProps()} class="kit-select-trigger">
          <span class="kit-select-words">
            {props.choices.map((choice, place) => <span data-picked={place === props.picked || undefined}>{choice}</span>)}
          </span>
        </button>
        <span {...api.value.getPositionerProps()} class="kit-select-positioner">
          <span {...api.value.getContentProps()} class="kit-select-content">
            {props.choices.map(choice => (
              <span {...api.value.getItemProps({ item: choice })} class="kit-select-item">
                <span {...api.value.getItemTextProps({ item: choice })}>{choice}</span>
              </span>
            ))}
          </span>
        </span>
      </span>
    )
  }
})
