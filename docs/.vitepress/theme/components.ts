import CreditsSection                                                  from './credits/section.ts'
import CreditsSnapshots                                                from './credits/snapshots.ts'
import CropKept                                                        from './crop.ts'
import { IdentityFaces, IdentityHeat, IdentityMarks, IdentityPalette } from './identity/widgets.ts'
import KitCatalog                                                      from './kit/catalog.ts'
import RulesNeighbors                                                  from './rules/neighbors.ts'
import RulesReach                                                      from './rules/reach.ts'
import RulesSpacing                                                    from './rules/spacing.ts'
import ThreatFront                                                     from './threat.ts'

/**
 * The widgets a page's markers write, by the name each is registered under.
 */
export const COMPONENTS = {
  CreditsSection,
  CreditsSnapshots,
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
