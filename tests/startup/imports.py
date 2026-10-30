"""
Pins the startup entry point the package installs, covering the file `site`
reads it from, the global mode and the filter it sets in every interpreter
the environment starts, and which modules' imports the filter lets load
lazily.
"""

from importlib          import reload
from importlib.metadata import distribution
from pkgutil            import resolve_name
from pytest             import mark, param
from subprocess         import run
from sys                import executable, get_lazy_imports, get_lazy_imports_filter

from thermur.startup         import imports
from thermur.startup.imports import EAGER, lazily


@mark.parametrize(
    ("flags", "printed"),
    [
        param((), "all lazily\n", id="every-interpreter-the-environment-starts"),
        param(("-S",), "normal None\n", id="one-started-without-site")
    ]
)
def test_an_interpreter_starts_under_the_mode_the_entry_point_sets(
    flags   : tuple[str, ...],
    printed : str
):
    """
    Asserts that an interpreter of the environment, started with no variable
    set, begins with every import potentially lazy behind `lazily`, whereas
    one started under `-S`, which skips `site`, begins in the mode Python
    starts in.
    """
    started = run(
        [
            executable,
            *flags,
            "-c",
            "import sys; filtered = sys.get_lazy_imports_filter(); "
            "print(sys.get_lazy_imports(), filtered and filtered.__qualname__)"
        ],
        capture_output = True,
        check          = True,
        env            = {},
        text           = True
    )

    assert started.stdout == printed


def test_start_sets_the_filter_and_the_mode_where_it_runs():
    """
    Asserts that `start`, run once its module loads again in this
    interpreter, makes every import potentially lazy behind that module's
    own `lazily`. `site` runs the module before coverage starts, so loading
    it again here is what measures it.
    """
    reloaded = reload(imports)
    reloaded.start()

    assert (get_lazy_imports(), get_lazy_imports_filter()) == ("all", reloaded.lazily)


@mark.parametrize(
    ("importer", "lazy"),
    [
        *(param(f"{package}.module", False, id=package) for package in sorted(EAGER)),
        param("thermur.pipeline.graph", True, id="the-package"),
        param("json.decoder", True, id="the-standard-library"),
        param("xarray.core.dataset", True, id="another-library"),
        param("pandasx", True, id="a-name-an-entry-prefixes"),
        param("__main__", True, id="the-main-module"),
        param(None, True, id="outside-a-module")
    ]
)
def test_the_filter_lets_every_import_but_the_named_packages_load_lazily(
    importer : str | None,
    lazy     : bool
):
    """
    Asserts that an import a module of a package in `EAGER` makes loads as
    it runs, whatever it imports, while one that a module of the package,
    the standard library, or any other library makes, or one made outside a
    module, may load lazily.
    """
    assert lazily(importer, "pandas", ("Index",)) is lazy


def test_the_installed_start_file_names_the_entry_point():
    """
    Asserts that the distribution installs `thermur.start` in its
    `site-packages`, naming one entry point, which resolves to `start`.
    """
    text = distribution("thermur").locate_file("thermur.start").read_text()

    assert resolve_name(text.strip(), strict=True) is imports.start
