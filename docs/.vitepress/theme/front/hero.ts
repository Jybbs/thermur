import { VPImage }            from 'vitepress/theme-without-fonts'
import { defineComponent, h } from 'vue'

import source            from '../../lib/rendered/identity.json' with { type: 'json' }
import { wordmarkImage } from '../../lib/site.ts'

/**
 * The front page's hero above its button: the wordmark in each scheme's ink as
 * the page's heading, and the tagline beneath it.
 */
export default defineComponent({
  name: 'FrontHero',
  setup() {
    const image = wordmarkImage()
    return () => [
      h('h1', { class: 'front-hero-mark' }, h(VPImage, { image })),
      h('p', { class: 'front-hero-tagline' }, source.tagline.text)
    ]
  }
})
