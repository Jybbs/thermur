"""
Holds `DECLARED`, the patterns of every tracked file outside the package's
code, its tests, and the site's code, meaning the manifests, the lockfiles,
the workflows, the tasks, and the pages, which a `reads` mark passes so a
case's checkout copies only the declarations a check or a task reads.
"""

DECLARED = (
    ".*", "*.*", ".github/**", ".mise/**",
    "LICENSE", "docs/*", "docs/kit/package.json", "docs/pages/**"
)
