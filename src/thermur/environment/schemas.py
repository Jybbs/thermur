"""
Holds the records the fire data is read through, meaning `Crop`, `Grid`,
`ParametricThreat`, `Ranges`, and `Sample`, beside the constants WRF and the
wrf-python diagnostics share.
"""

from hashlib            import file_digest
from importlib.metadata import version
from numpy              import arange, tanh, zeros
from pathlib            import Path
from pydantic           import BaseModel, Field, model_validator
from shutil             import copyfileobj
from tarfile            import TarFile
from tempfile           import TemporaryFile
from typing             import Annotated, Self
from urllib.request     import Request, urlopen
from xarray             import DataArray, Dataset

# `BASE_K`, `DRY_AIR_J_KG_K`, `GRAVITY_M_S2`, `HEAT_CAPACITY_J_KG_K`, and
# `REFERENCE_PA` hold the constants wrf-python's `wrf_constants` names `T_BASE`,
# `RD`, `G`, `CP`, and `P1000MB`, meaning in that order the potential
# temperature the perturbation `T` departs from, the gas constant of dry air,
# the gravitational acceleration, the specific heat of dry air at constant
# pressure, and the pressure potential temperature is referred to.
type Count           = Annotated[int, Field(ge=1)]
type NonNegative     = Annotated[float, Field(ge=0)]
type Positive        = Annotated[float, Field(gt=0)]
type Range           = tuple[float, float]
type Span            = Range | None
BASE_K               = 300.0
DRY_AIR_J_KG_K       = 287.0
GRAVITY_M_S2         = 9.81
HEAT_CAPACITY_J_KG_K = 1004.5
REFERENCE_PA         = 100000.0


class Crop(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The window, the neighborhood, and the band a WRF-SFIRE output is
    cropped to before any other field loads, reading only its times and its
    base-state geopotential, and the fixed heights its fields are regridded
    onto.
    """

    floor_m: Positive = 20.0
    """
    The lowest fixed height above the ground, in meters.
    """

    levels: Count = 25
    """
    The number of fixed heights, which reach 500 m above the ground at the
    default floor and spacing.
    """

    spacing_m: Positive = 20.0
    """
    The distance between neighboring fixed heights, in meters.
    """

    window_s: Span = None
    """
    The first and the last time kept, in seconds since the simulation
    started, or `None` to keep every snapshot.
    """

    x_m: Span = None
    """
    The westernmost and the easternmost mass point kept, in meters from the
    domain's western edge, or `None` to keep every column.
    """

    y_m: Span = None
    """
    The southernmost and the northernmost mass point kept, in meters from
    the domain's southern edge, or `None` to keep every row.
    """

    @property
    def heights(self) -> DataArray:
        """
        Lays out the fixed heights above the ground, evenly spaced
        from `floor_m`, along a `height` dimension carrying them as its
        coordinate.
        """
        heights = self.floor_m + self.spacing_m * arange(self.levels)
        return DataArray(heights, coords={"height": heights}, dims="height")


class Grid(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The grid a `ParametricThreat` lays its dataset out on, spaced as the
    WRF-SFIRE plume dataset spaces its simulations horizontally and in time.
    """

    columns: Count = 100
    """
    The number of mass points from west to east.
    """

    interval_s: Positive = 15.0
    """
    The time between neighboring snapshots, in seconds, the interval each
    output of the plume dataset is written at.
    """

    level_m: Positive = 20.0
    """
    The depth of each model level, in meters.
    """

    levels: Count = 30
    """
    The number of model levels.
    """

    rows: Count = 50
    """
    The number of mass points from south to north.
    """

    snapshots: Count = 9
    """
    The number of snapshots, which span two minutes at the default interval.
    """

    spacing_m: Positive = 40.0
    """
    The distance between neighboring mass points, in meters, the horizontal
    spacing of the plume dataset.
    """

    @property
    def centers(self) -> DataArray:
        """
        Places each mass point east of the domain's western edge, in meters,
        along `west_east`.
        """
        return DataArray(
            (arange(self.columns) + 0.5) * self.spacing_m,
            dims = "west_east"
        )

    @property
    def depths(self) -> DataArray:
        """
        Places each staggered level above the ground, in meters, along
        `bottom_top_stag`.
        """
        return DataArray(arange(self.levels + 1) * self.level_m, dims="bottom_top_stag")

    @property
    def heights(self) -> DataArray:
        """
        Places each model level's center above the ground, in meters, along
        `bottom_top`.
        """
        return DataArray((arange(self.levels) + 0.5) * self.level_m, dims="bottom_top")

    @property
    def length_m(self) -> float:
        """
        Measures the domain from its western edge to its eastern one, in
        meters.
        """
        return self.columns * self.spacing_m

    @property
    def seconds(self) -> DataArray:
        """
        Places each snapshot in seconds since the first, along `Time`.
        """
        return DataArray(arange(self.snapshots) * self.interval_s, dims="Time")

    @property
    def sizes(self) -> dict[str, int]:
        """
        Sizes each dimension WRF names, a staggered dimension holding one
        point more than the mass points it bounds.
        """
        mass = {
            "bottom_top"  : self.levels,
            "south_north" : self.rows,
            "west_east"   : self.columns
        }
        return {"Time": self.snapshots} | mass | {
            f"{name}_stag": size + 1 for name, size in mass.items()
        }

    @property
    def span_s(self) -> float:
        """
        Measures the time from the first snapshot to the last, in seconds.
        """
        return (self.snapshots - 1) * self.interval_s

    def zeros(self, *dimensions: str) -> DataArray:
        """
        Lays out zeros over the WRF dimensions `dimensions` names, in that
        order, each sized as `sizes` gives it.
        """
        return DataArray(
            zeros([self.sizes[name] for name in dimensions]),
            dims = dimensions
        )


class Ranges(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The values each field `WrfOutput` converts may take, which a misread
    variable or a fill value falls outside.
    """

    height_m: Range = (0.0, 5000.0)
    """
    The height above the ground a mass point may take, in meters, from the
    ground to the top of the plume dataset's deepest simulations.
    """

    temperature_k: Range = (183.95, 1603.15)
    """
    The air temperature a mass point may take, in kelvin, from the -89.2 °C
    recorded at Vostok station, the lowest at the Earth's surface, to the
    1,330 °C peak air temperatures exceeded inside a boreal crown fire.
    """

    wind_m_s: Range = (-113.3, 113.3)
    """
    The wind along each axis a mass point may take, in meters per second,
    either way up to the 113.3 m/s gust recorded on Barrow Island, the
    world record.
    """


class Sample(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    The WRF-SFIRE sample the `thermur fetch sample` command downloads, an
    archive pinned to one revision of the Hugging Face dataset publishing it
    and to its SHA-256 digest, holding the output the graph reads.
    """

    archive: str = "samples.tar.gz"
    """
    The archive the dataset publishes the sample as.
    """

    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] = (
        "6f97d0bd35b56e224720692f81747fb5e1deddfbc3d1397e6b82ddb42518acfa"
    )
    """
    The SHA-256 digest of the archive.
    """

    folder: Path = Path("data")
    """
    The folder the archive extracts into, relative to the clone.
    """

    hub: str = "https://huggingface.co"
    """
    The address of the Hugging Face Hub publishing the dataset.
    """

    length_m: Positive = 20000.0
    """
    The length of the domain the output covers from west to east, in meters.
    """

    member: Path = Path("samples/wrf_sample.nc")
    """
    The WRF-SFIRE output inside the archive.
    """

    repository: str = "Jybbs/sfire-samples"
    """
    The Hugging Face dataset publishing the archive.
    """

    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] = (
        "e7d4fc2ca1f99e82dfeec3ca2f7ba1c714965913"
    )
    """
    The commit of the dataset the archive is read at.
    """

    timeout_s: Positive = 60.0
    """
    The longest the download waits on the connection or on one read, in
    seconds.
    """

    times_s: tuple[float, ...] = (3615.0, 3630.0, 3645.0, 3660.0)
    """
    The time of each snapshot the output holds, in seconds since the simulation
    started.
    """

    @property
    def address(self) -> str:
        """
        Composes the address the archive downloads from at `revision`.
        """
        return "/".join(
            [
                self.hub, "datasets", self.repository,
                "resolve", self.revision, self.archive
            ]
        )

    @property
    def request(self) -> Request:
        """
        Prepares the request for the archive at `address`, naming the client
        as the package at its installed version, such as `thermur/0.1.0`,
        and nothing about the reader running it.
        """
        return Request(
            self.address,
            headers = {"User-Agent": f"thermur/{version('thermur')}"}
        )

    def fetch(self, root: Path) -> Path:
        """
        Downloads the archive into a temporary file under `folder`, waiting
        on the connection and on each read for at most `timeout_s`, checks
        it against `digest` through `hashlib.file_digest`, and extracts
        it into `folder` under `tarfile`'s `data` filter, which refuses a
        member reaching outside the folder, a link leading out of it, and
        a device.

        Args:
            root: The clone `folder` sits under.

        Returns:
            The path of `member` once extracted.

        Raises:
            ValueError: Where the archive's digest differs from `digest`,
                        before anything is extracted.
        """
        folder = root / self.folder
        folder.mkdir(exist_ok=True, parents=True)
        with TemporaryFile(dir=folder) as file:
            with urlopen(self.request, timeout=self.timeout_s) as response:
                copyfileobj(response, file)

            file.seek(0)
            if (digest := file_digest(file, "sha256").hexdigest()) != self.digest:
                raise ValueError(
                    f"`{self.archive}` hashes to `{digest}` rather than the "
                    f"pinned `{self.digest}`"
                )

            file.seek(0)
            with TarFile.open(fileobj=file) as bundle:
                bundle.extractall(folder, filter="data")

        return folder / self.member


class ParametricThreat(
    BaseModel,
    extra                    = "forbid",
    frozen                   = True,
    use_attribute_docstrings = True
):
    """
    A front of hot air entering the domain at its western edge as the first
    snapshot is written and advancing east at a steady speed, the cells it
    has passed burning, laid out as the dataset a WRF-SFIRE output holds so
    every conversion of `WrfOutput` reads it as it reads a simulation.

    The air holds the potential temperature WRF's base state takes at every
    height, warmed behind the front by `anomaly_k`, which a hyperbolic
    tangent of scale `edge_m` carries to zero ahead of it, and the smoke
    tracer follows the same profile from one behind the front to zero ahead.
    """

    anomaly_k: Positive = 20.0
    """
    The potential temperature behind the front in excess of the air ahead
    of it, in kelvin.
    """

    edge_m: Positive = 40.0
    """
    The scale of the hyperbolic tangent carrying the anomaly to zero ahead of
    the front, in meters, one grid spacing by default, so 76% of the anomaly
    falls within this distance on either side of the front.
    """

    grid: Grid = Grid()
    """
    The grid the dataset is laid out on.
    """

    speed_m_s: NonNegative = 1.0
    """
    The speed at which the front advances east, in meters per second.
    """

    @property
    def dataset(self) -> Dataset:
        """
        Lays the front out under the variable and dimension names WRF
        writes, in single precision as WRF writes them.

        The base state is the hydrostatic atmosphere whose potential
        temperature holds `BASE_K` at every height and whose pressure holds
        `REFERENCE_PA` at the ground, so the air ahead of the front cools
        with height at the dry adiabatic lapse rate. The winds are still,
        and the ground heat flux `GRNHFX` holds one watt per square meter on
        each cell whose center the front has reached and zero elsewhere.
        """
        grid     = self.grid
        mass     = grid.zeros("Time", "bottom_top", "south_north", "west_east")
        faces    = grid.zeros("Time", "bottom_top_stag", "south_north", "west_east")
        front    = self.speed_m_s * grid.seconds
        profile  = (1 - tanh((grid.centers - front) / self.edge_m)) / 2
        exner    = 1 - GRAVITY_M_S2 * grid.heights / (HEAT_CAPACITY_J_KG_K * BASE_K)
        pressure = REFERENCE_PA * exner ** (HEAT_CAPACITY_J_KG_K / DRY_AIR_J_KG_K)
        return Dataset(
            {
                "GRNHFX" : mass.isel(bottom_top=0, drop=True) + (grid.centers <= front),
                "P"      : mass,
                "PB"     : mass + pressure,
                "PH"     : faces,
                "PHB"    : faces + GRAVITY_M_S2 * grid.depths,
                "T"      : mass + self.anomaly_k * profile,
                "U" : grid.zeros("Time", "bottom_top", "south_north", "west_east_stag"),
                "V" : grid.zeros("Time", "bottom_top", "south_north_stag", "west_east"),
                "W" : faces,
                "XTIME"  : grid.seconds / 60,
                "tr17_1" : mass + profile
            },
            attrs = {"DX": grid.spacing_m, "DY": grid.spacing_m}
        ).astype("float32")

    @model_validator(mode="after")
    def held(self) -> Self:
        """
        Holds the front and five times `edge_m` ahead of it at or west of
        the center of the first column a median over the columns reads,
        through the last snapshot, so the median `WrfOutput.ambient` takes
        still reads the air ahead of the front, the anomaly having fallen
        under 0.005% of itself that far ahead.

        Raises:
            ValueError: Where the front's edge passes that column's center
                        by the last snapshot.
        """
        middle = float(self.grid.centers[(self.grid.columns - 1) // 2])
        reach  = self.speed_m_s * self.grid.span_s + 5 * self.edge_m
        if reach > middle:
            raise ValueError(
                f"The front's edge reaches {reach:g} m by the last snapshot, past "
                f"the column at {middle:g} m a median reads on a domain "
                f"{self.grid.length_m:g} m long"
            )

        return self
