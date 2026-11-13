import type { UserConfig } from 'vite'

import { identityModule } from './identity.ts'

/**
 * Builds the Vite settings the site's build and its suite share: the kit's TSX
 * compiled onto Vue's own JSX runtime, and the module serving the identity's
 * stylesheet.
 */
export const shared = (): Pick<UserConfig, 'oxc' | 'plugins'> => ({
  oxc     : { jsx: { importSource: 'vue', runtime: 'automatic' } },
  plugins : [identityModule()]
})
