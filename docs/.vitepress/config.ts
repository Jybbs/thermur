import { defineConfig } from 'vitepress'

import { Cards }          from './lib/cards.ts'
import * as head          from './lib/head.ts'
import { identityModule } from './lib/identity.ts'
import { projectFacts }   from './lib/manifest.ts'
import { mathRules }      from './lib/math.ts'
import { pageData }       from './lib/page-data.ts'
import * as paths         from './lib/paths.ts'
import { themeConfig }    from './lib/site.ts'

const facts    = projectFacts(import.meta.url)
const previews = new head.Previews(Cards.under(import.meta.url, facts), facts)

export default defineConfig({
  base              : facts.base,
  buildEnd          : ({ outDir }) => previews.write(outDir),
  cacheDir          : paths.cacheDir(import.meta.url, 'vitepress'),
  cleanUrls         : true,
  description       : facts.description,
  head              : head.siteHead(facts.base),
  markdown          : { config: mathRules },
  sitemap           : { hostname: facts.homepage },
  srcDir            : paths.pagesDir(import.meta.url),
  themeConfig       : themeConfig(facts),
  title             : head.TITLE,
  transformHead     : context => previews.head(context),
  transformPageData : page => pageData(page, facts.repository),

  vite: {
    css       : { transformer: 'lightningcss' },
    plugins   : [identityModule()],
    publicDir : paths.publicDir(import.meta.url)
  }
})
