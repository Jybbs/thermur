import type { Theme } from 'vitepress'
import DefaultTheme   from 'vitepress/theme-without-fonts'
import { h }          from 'vue'

import { COMPONENTS } from './components.ts'
import FrontHero      from './front/hero.ts'
import FrontSky       from './front/sky.ts'
import NotFoundPage   from './not-found/page.ts'
import MotionPause    from './pause.ts'

import '@thermur/kit/styles.css'
import 'katex/dist/katex.min.css'
import 'virtual:identity.css'
import './styles/motion.css'
import './styles/type.css'
import './styles/tables.css'
import './styles/fill.css'
import './styles/front/sky.css'
import './styles/front/hero.css'
import './styles/not-found/page.css'
import './styles/not-found/track.css'
import './styles/pause.css'
import './styles/crop.css'
import './styles/threat.css'
import './styles/rules/figure.css'
import './styles/rules/neighbors.css'
import './styles/values.css'
import './styles/identity/specimen.css'
import './styles/identity/palette.css'
import './styles/identity/heat.css'
import './styles/identity/faces.css'
import './styles/identity/marks.css'
import './styles/kit/parts.css'
import './styles/kit/catalog.css'
import './styles/kit/card.css'

/**
 * The default layout's slots the theme fills: the front page's background
 * behind its hero, the wordmark and tagline as the hero's words, the switch
 * pausing a page's motion in the navigation bar, and the page met at a missing
 * address.
 */
const SLOTS = {
  'home-hero-before'      : () => h(FrontSky),
  'home-hero-info'        : () => h(FrontHero),
  'nav-bar-content-after' : () => h(MotionPause),
  'not-found'             : () => h(NotFoundPage)
}

export default {
  Layout  : () => h(DefaultTheme.Layout, null, SLOTS),
  extends : DefaultTheme,

  enhanceApp: ({ app }) => {
    for (const [name, component] of Object.entries(COMPONENTS)) app.component(name, component)
  }
} satisfies Theme
