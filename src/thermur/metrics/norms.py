"""
Holds `Norm`, one measurement of a starling flock or an outdoor drone flock
a statistic of `Flight` is set beside, and `Norms`, every norm `norms.toml`
beside this module declares, read through `Norms.shipped()`.
"""

from enum                import StrEnum
from importlib.resources import read_text
from pydantic            import BaseModel, RootModel, model_validator
from tomllib             import loads
from typing              import Literal, Self

from thermur.environment.schemas import Count, Positive


class Spread(StrEnum):
    """
    What a norm's spread measures about its figure.
    """

    CONFIDENCE = "95% confidence interval"
    DEVIATION  = "standard deviation"
    ERROR      = "standard error"
    RANGE      = "half range"


class Norm(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One measurement a statistic is set beside, its figure in the unit the
    statistic reads.
    """

    doi: str
    """
    Whether the source states only that the statistic exceeds `figure`.
    """

    figure: float
    """
    The DOI of the study the measurement comes from.
    """

    system: Literal["drones", "starlings"]
    """
    The measured value, or the middle of a measured range.
    """

    unit: str
    """
    The number of flocks the measurement spans, or `None` where the source
    states none.
    """

    above: bool = False
    """
    What `spread` measures, or `None` for a figure stated with no spread.
    """

    flocks: Count | None = None
    """
    The spread about `figure` that `kind` names, or `None` for none.
    """

    kind: Spread | None = None
    """
    The kind of flock measured.
    """

    spread: Positive | None = None
    """
    The unit of `figure` and `spread`, empty for a dimensionless one.
    """

    @property
    def bounds(self) -> tuple[float, float]:
        """
        Spans `figure` less and plus `spread`, from `figure` upward for a
        norm holding only a lower bound, and `figure` alone for one stating
        no spread.
        """
        if self.above:
            return self.figure, float("inf")

        reach = self.spread or 0.0
        return self.figure - reach, self.figure + reach

    def holds(self, value: float) -> bool:
        """
        Reads whether `value` sits within `bounds`, past `figure` for a norm
        holding only a lower bound.
        """
        low, high = self.bounds
        return value > low if self.above else low <= value <= high

    @model_validator(mode="after")
    def paired(self) -> Self:
        """
        Holds `spread` and `kind` together, both set or both left out.

        Raises:
            ValueError: Where one of the two is set without the other.
        """
        if (self.spread is None) != (self.kind is None):
            raise ValueError("A norm sets `spread` and `kind` together or neither")

        return self


class Norms(RootModel[dict[str, Norm]], frozen=True):
    """
    Every norm, by the name of the statistic it is set beside.
    """

    def __getitem__(self, statistic: str) -> Norm:
        """
        Reads the norm set beside `statistic`.
        """
        return self.root[statistic]

    @classmethod
    def shipped(cls) -> Self:
        """
        Validates `norms.toml`, the declaration beside this module.
        """
        return cls.model_validate(loads(read_text(__package__, "norms.toml")))
