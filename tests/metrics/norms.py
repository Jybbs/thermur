"""
Pins `Norm` and `Norms`, meaning the norms `norms.toml` declares, the bounds
a norm holds a statistic to, and the spreads it refuses.
"""

from pydantic import ValidationError
from pytest   import mark, param, raises

from thermur.metrics.norms import Norm, Norms, Spread


def test_the_shipped_norms_hold_each_measurement_the_still_air_page_cites():
    """
    Asserts that the shipped table declares a norm for each statistic the
    still-air page sets beside one, each naming the study's DOI.
    """
    norms = Norms.shipped()

    assert set(norms.root) == {
        "aspect_ratio", "border_density", "middle_aspect_ratio", "polarization",
        "spacing_m", "speed_by_size", "speed_correlation_ratio", "wave_speed_m_s"
    }
    assert norms["polarization"] == Norm(
        doi    = "10.1073/pnas.1005766107",
        figure = 0.96,
        flocks = 24,
        kind   = Spread.DEVIATION,
        spread = 0.03,
        system = "starlings",
        unit   = ""
    )
    assert all(norm.doi.startswith("10.") for norm in norms.root.values())


@mark.parametrize(
    ("statistic", "value", "held"),
    [
        param("polarization", 0.93, True, id="at-the-spread"),
        param("polarization", 0.92, False, id="past-the-spread"),
        param("border_density", 1.01, True, id="past-the-bound"),
        param("border_density", 1.0, False, id="at-the-bound"),
        param("speed_correlation_ratio", 0.36, True, id="at-the-figure")
    ]
)
def test_a_norm_holds_a_value_within_its_spread_or_past_its_bound(
    statistic : str,
    value     : float,
    held      : bool
):
    """
    Asserts that a norm holds a value within its spread about its figure,
    a norm stating only a lower bound holds a value past it, and a norm
    stating no spread holds its figure alone.
    """
    assert Norms.shipped()[statistic].holds(value) is held


@mark.parametrize(
    "changes",
    [
        param({"spread": 0.1}, id="no-kind"),
        param({"kind": Spread.ERROR}, id="no-spread")
    ]
)
def test_a_norm_setting_its_spread_or_its_kind_alone_is_refused(
    changes: dict[str, object]
):
    """
    Asserts that a norm setting `spread` without `kind`, or `kind` without
    `spread`, fails validation.
    """
    with raises(ValidationError, match="sets `spread` and `kind` together"):
        Norm(doi="10.1/x", figure=1.0, system="drones", unit="", **changes)
