import { type PropType, type Ref, defineComponent } from 'vue'

/**
 * One reading beneath or above a drawing: its name in capitals beside a value
 * the drawing writes into, as wide as `shape` so the row never shifts as the
 * value changes. The drawing reaches the value's box through `value` and the
 * name's through `named`, writing each only as it changes rather than on every
 * frame through Vue.
 */
export const Chip = defineComponent({
  name  : 'Chip',
  props : {
    name  : { required: true, type: String },
    named : { default: undefined, type: Object as PropType<Ref<HTMLElement | undefined>> },
    shape : { default: '', type: String },
    value : { default: undefined, type: Object as PropType<Ref<HTMLElement | undefined>> }
  },
  setup(props) {
    return () => (
      <span class="kit-chip">
        <span class="kit-chip-name" ref={props.named}>{props.name}</span>
        <span class="kit-chip-value" ref={props.value} style={{ minWidth: `${props.shape.length}ch` }} />
      </span>
    )
  }
})
