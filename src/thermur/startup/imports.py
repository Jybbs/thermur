"""
Holds `start`, the entry point the package installs in `site-packages` as
`thermur.start`, which `site` calls once every path a `.pth` file names is
on `sys.path`, and `lazily`, the filter `start` sets. Every interpreter the
environment starts then loads the module each top-level import names where
a line first reads the name it binds, short of an import inside a `try`, a
`from … import *`, and each import a package `EAGER` names makes, which load
as they run.
"""

from sys import set_lazy_imports, set_lazy_imports_filter

EAGER = frozenset({"_pytest", "pandas"})
"""
The packages whose own imports load as they run, since each fails once they
load lazily, each listed below beside its defect, its upstream issue where one
exists, and the command reproducing it. Whenever a lift moves one of them, its
command is rerun with its entry removed, and the entry goes once the command
passes, with its issue read for whether a fix has shipped:

- `_pytest`, whose assertion rewriter matches each module's path through
  `fnmatch` and so enters itself again through the lazy import of `fnmatch`,
  raising `ImportCycleError` (pytest-dev/pytest#14632, with a fix proposed
  in pytest-dev/pytest#14844), reproduced by `mise test -- --collect-only`
- `pandas`, which imports the modules setting its C API capsules and its
  options only for that effect, so its compiled modules read them unset
  (pandas-dev/pandas#71173, with a fix proposed in pandas-dev/pandas#71178),
  reproduced by `mise x -- uv run --locked python -c "import pandas; pandas.Index"`
"""


def lazily(importer: str | None, name: str, fromlist: tuple[str, ...] | None) -> bool:  # prose: ignore[unsorted-positionals]
    """
    Lets every import load lazily except one that a module of a package in
    `EAGER` makes, which loads as it runs.

    Args:
        importer : The name of the module the import runs in, or `None`
                   where it runs in no module.
        name     : The name of the module the import loads.
        fromlist : The names a `from` import binds, or `None` for a plain
                   `import`.
    """
    return importer is None or importer.partition(".")[0] not in EAGER


def start():
    """
    Sets `lazily` as the filter, then makes each import at the top of
    a module lazy, short of one inside a `try` or a `from … import *`,
    wherever the filter lets it be.
    """
    set_lazy_imports_filter(lazily)
    set_lazy_imports("all")
