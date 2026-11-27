import IdentityFaces   from './faces.ts'
import IdentityHeat    from './heat.ts'
import IdentityMarks   from './marks.ts'
import IdentityPalette from './palette.ts'

/**
 * The widgets the identity page's markers write, by the name each is
 * registered under.
 */
export const IDENTITY = {
  IdentityFaces,
  IdentityHeat,
  IdentityMarks,
  IdentityPalette
} as const
