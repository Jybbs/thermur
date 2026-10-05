"""
Pins where the settings read each value from, meaning the order of a value
passed in, a `THERMUR_` variable, and the `[tool.thermur]` table of the
installed clone, and what the model refuses, over a stand-in field and a
stand-in component.
"""

from common.edits    import append, edit
from common.settings import Heading, Tuned
from pydantic        import ValidationError
from pytest          import MonkeyPatch, TempPathFactory, fixture, mark, param, raises

from thermur.repo.checkout import Checkout

TABLE = """
[tool.thermur]
radius_m = 7.0

[tool.thermur.heading]
coupling_per_s = 2.5
"""


@fixture
def installed(checkout: Checkout, monkeypatch: MonkeyPatch) -> Checkout:
    """
    Stands the copy `checkout` makes in for the clone the package was
    installed from, holding a `[tool.thermur]` table that sets the stand-in
    field and the stand-in component.
    """
    append(checkout, "pyproject.toml", TABLE)
    monkeypatch.setattr(Checkout, "installed", classmethod(lambda _: checkout))

    return checkout


@mark.parametrize(
    ("variables", "passed", "radius_m"),
    [
        param({"RADIUS_M": "1"}, {}, 7.0, id="bare"),
        param({"THERMUR_RADIUS_M": "3"}, {}, 3.0, id="prefixed"),
        param({"RADIUS_M": "1", "THERMUR_RADIUS_M": "3"}, {}, 3.0, id="both"),
        param({"THERMUR_RADIUS_M": "3"}, {"radius_m": 2.0}, 2.0, id="passed-in")
    ]
)
def test_the_sources_set_a_field_in_order_and_a_bare_variable_sets_nothing(
    installed   : Checkout,
    monkeypatch : MonkeyPatch,
    passed      : dict[str, float],
    radius_m    : float,
    variables   : dict[str, str]
):
    """
    Asserts that a value passed in beats a variable naming the field behind
    `THERMUR_`, which beats the table, and that the same name without the
    prefix sets nothing, whether or not the prefixed one is set beside it.
    """
    for name, setting in variables.items():
        monkeypatch.setenv(name, setting)

    assert Tuned(**passed).radius_m == radius_m


def test_a_component_outside_its_bound_fails(installed: Checkout):
    """
    Asserts that a component's parameter beyond the bound its field declares
    fails validation at that parameter.
    """
    with raises(ValidationError) as error:
        Tuned(heading={"coupling_per_s": 50})

    assert [detail["loc"] for detail in error.value.errors()] == [
        ("heading", "coupling_per_s")
    ]


def test_a_key_no_field_declares_fails(installed: Checkout):
    """
    Asserts that a `[tool.thermur]` key no field declares fails validation
    at that key, so a misspelled setting stops the run.
    """
    edit(installed, "pyproject.toml", new="radius = 3.0\nradius_m", old="radius_m")

    with raises(ValidationError) as error:
        Tuned()

    assert [(detail["loc"], detail["type"]) for detail in error.value.errors()] == [
        (("radius",), "extra_forbidden")
    ]


def test_a_variable_sets_a_component_as_json(
    installed   : Checkout,
    monkeypatch : MonkeyPatch
):
    """
    Asserts that a prefixed variable holding a JSON object sets a component
    over the table.
    """
    monkeypatch.setenv("THERMUR_HEADING", '{"coupling_per_s": 4}')

    assert Tuned().heading == Heading(coupling_per_s=4.0)


def test_the_settings_are_frozen(installed: Checkout):
    """
    Asserts that a field cannot be set once the settings are built, so one
    node cannot move a value another node has read.
    """
    with raises(ValidationError):
        Tuned().radius_m = 2.0


def test_the_table_is_read_from_the_installed_clone(
    installed        : Checkout,
    monkeypatch      : MonkeyPatch,
    tmp_path_factory : TempPathFactory
):
    """
    Asserts that the table comes from the clone the package was installed
    from, whatever `pyproject.toml` the working directory holds.
    """
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    (elsewhere / "pyproject.toml").write_text("[tool.thermur]\nradius_m = 99.0\n")
    monkeypatch.chdir(elsewhere)

    assert Tuned().radius_m == 7.0


def test_the_table_sets_a_field_and_a_component(installed: Checkout):
    """
    Asserts that the table sets the stand-in field and the stand-in
    component, which reads back as the component's own `Module` type.
    """
    settings = Tuned()

    assert (settings.radius_m, settings.heading) == (7.0, Heading(coupling_per_s=2.5))
    assert type(settings.heading) is Heading


def test_without_a_table_each_field_keeps_its_default(installed: Checkout):
    """
    Asserts that a clone whose `pyproject.toml` holds no `[tool.thermur]`
    table leaves every field on the default it declares.
    """
    edit(installed, "pyproject.toml", new="", old=TABLE)

    assert Tuned() == Tuned(heading=Heading(), radius_m=5.0)
