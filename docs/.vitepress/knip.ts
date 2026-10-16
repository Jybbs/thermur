import type { KnipConfig } from 'knip'

export default {
  ignoreDependencies : ['oxlint'],
  rules              : { cycles: 'error' },
  typescript         : { config: ['.vitepress/tsconfig.json'] },
  vitest             : { config: ['.vitepress/vitest.config.ts'] }
} satisfies KnipConfig
