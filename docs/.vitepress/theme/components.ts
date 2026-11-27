import CropKept       from './crop.ts'
import { IDENTITY }   from './identity/components.ts'
import KitCatalog     from './kit/catalog.ts'
import LookoutFlight  from './lookout.ts'
import RulesNeighbors from './rules/neighbors.ts'
import RulesReach     from './rules/reach.ts'
import RulesSpacing   from './rules/spacing.ts'
import ThreatFront    from './threat.ts'

/**
 * The widgets a page's markers write, by the name each is registered under.
 */
export const COMPONENTS = {
  CropKept,
  ...IDENTITY,
  KitCatalog,
  LookoutFlight,
  RulesNeighbors,
  RulesReach,
  RulesSpacing,
  ThreatFront
} as const
