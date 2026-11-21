"""
Holds the patterns a `reads` mark passes so a case's checkout copies
only the files a check or a task reads. `DECLARED` matches every tracked
file outside the package's code, its tests, and the site's code, such as
the manifests, the lockfiles, the workflows, the tasks, and the pages.
`PACKAGED` adds the package's code, and `RENDERED` adds the records the
render writes and the kit's source beside them.
"""

DECLARED = (
    ".*", "*.*", ".github/**", ".mise/**",
    "LICENSE", "docs/*", "docs/kit/package.json", "docs/pages/**"
)

PACKAGED = (*DECLARED, "src/**")
RENDERED = (*PACKAGED, "docs/.vitepress/lib/rendered/**", "docs/kit/src/**")
