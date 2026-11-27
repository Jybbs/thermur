import { defineComponent, h } from 'vue'

import CreditsSeal from './seal.ts'

/**
 * The plume dataset's credit closing the Data page, its lines set beside the
 * seal.
 */
export default defineComponent({
  name: 'CreditsSection',
  setup: (_, { slots }) => () => h('div', { class: 'credits-section' }, [
    h(CreditsSeal),
    h('div', { class: 'credits-lines' }, slots.default?.())
  ])
})
