import CropKept        from './crop.ts'
import IdentityFaces   from './identity/faces.ts'
import IdentityHeat    from './identity/heat.ts'
import IdentityMarks   from './identity/marks.ts'
import IdentityPalette from './identity/palette.ts'
import KitCatalog      from './kit/catalog.ts'
import RulesNeighbors  from './rules/neighbors.ts'
import RulesReach      from './rules/reach.ts'
import RulesSpacing    from './rules/spacing.ts'
import ThreatFront     from './threat.ts'

/**
 * The widgets a page's markers write, by the name each is registered under.
 */
export const COMPONENTS = {
  CropKept,
  IdentityFaces,
  IdentityHeat,
  IdentityMarks,
  IdentityPalette,
  KitCatalog,
  RulesNeighbors,
  RulesReach,
  RulesSpacing,
  ThreatFront
} as const
