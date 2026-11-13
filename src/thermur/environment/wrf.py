"""
Holds `WrfOutput`, the WRF-SFIRE output a run reads, beside `checked`,
`destaggered`, `distances`, `extremes`, `spans`, and `unfilled`, the
functions it applies to a field, along one dimension, or to one snapshot.
The `wrf_output` step builds the record and the `air` step reads it.
"""

from functools     import cached_property
from math          import prod
from netCDF4       import default_fillvals
from numpy         import arange, full, inf, int32, interp, ndarray
from pathlib       import Path
from pydantic      import BaseModel, InstanceOf
from scipy.ndimage import distance_transform_edt
from typing        import ClassVar, Self
from xarray        import DataArray, Dataset, apply_ufunc, open_dataset

from thermur.environment.air     import Air
from thermur.environment.schemas import (
    BASE_K,
    Crop,
    DRY_AIR_J_KG_K,
    GRAVITY_M_S2,
    HEAT_CAPACITY_J_KG_K,
    ParametricThreat,
    REFERENCE_PA,
    Range,
    Ranges,
    Sample,
    Span
)
from thermur.pipeline.settings import Settings
from thermur.repo.checkout     import Checkout


class WrfOutput(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One WRF-SFIRE output under WRF's own names, holding only the variables
    the record reads, cropped to `crop` before any other field loads,
    reading only the times and the base-state geopotential, with its mass
    and staggered points placed in meters from the domain's southwest corner
    and its snapshots in seconds since the simulation started.
    """

    ranges: ClassVar[Ranges] = Ranges()
    """
    The values each converted field may take.
    """

    variables: ClassVar[tuple[str, ...]] = (
        "GRNHFX", "P", "PB", "PH", "PHB", "T",
        "U", "V", "W", "XTIME", "tr17_1"
    )
    """
    The variables the record keeps of an output, each under the name WRF
    writes it.
    """

    crop: Crop
    """
    The window, the neighborhood, and the band the output is cropped to, and
    the fixed heights its fields are regridded onto.
    """

    dataset: InstanceOf[Dataset]
    """
    The cropped output.
    """

    @property
    def air(self) -> Air:
        """
        Lays out the air temperature, its excess over the ambient profile,
        the smoke tracer, the winds, and the distance to the burning front
        on the fixed heights as the `Air` a flight samples, the distance
        taking the same value at every height.
        """
        return Air.from_dataset(self.aloft.assign(excess=self.excess, front=self.front))

    @cached_property
    def aloft(self) -> Dataset:
        """
        Regrids the air temperature, the winds, and the smoke tracer
        `tr17_1` from the model's levels onto the fixed heights above the
        ground `crop` lays out, interpolating each column linearly in height
        through `np.interp`.

        Each column's extent is compared to the millimeter, since a height
        in single precision lands tens of micrometers off the level it
        names.

        Raises:
            ValueError: Where a column's levels begin above the lowest fixed
                        height or end below the highest, which `np.interp`
                        would read as the value at the column's edge, or
                        where `tr17_1` holds a missing or fill value.
        """
        heights, band = self.heights, self.crop.heights
        lowest        = round(float(heights.isel(bottom_top=0).max()), 3)
        highest       = round(float(heights.isel(bottom_top=-1).min()), 3)
        if lowest > band[0] or highest < band[-1]:
            raise ValueError(
                f"The model's levels cover {lowest:g} to {highest:g} m in every "
                f"column, short of the band from {float(band[0]):g} to "
                f"{float(band[-1]):g} m"
            )

        fields = self.wind.assign(
            smoke       = unfilled(self.dataset.tr17_1),
            temperature = self.temperature
        )
        return apply_ufunc(
            interp,
            band,
            heights,
            fields,
            input_core_dims  = [["height"], ["bottom_top"], ["bottom_top"]],
            output_core_dims = [["height"]],
            vectorize        = True
        ).transpose(*Air.axes)

    @property
    def ambient(self) -> DataArray:
        """
        Profiles the air temperature the fire has not reached, as the
        median over every mass point the crop keeps at each fixed height and
        snapshot, which leaves out the heated air while it covers less than
        half of the height the crop keeps.
        """
        return self.aloft.temperature.median(["south_north", "west_east"])

    @cached_property
    def burning(self) -> DataArray:
        """
        Marks each mass point the crop keeps whose ground heat flux `GRNHFX`
        shows burning, one snapshot at a time.

        Raises:
            ValueError: Where `GRNHFX` holds a missing or fill value.
        """
        return unfilled(self.dataset.GRNHFX) > 0

    @property
    def excess(self) -> DataArray:
        """
        Measures the air temperature at each point on the fixed heights in
        excess of the ambient profile at its height and snapshot, in kelvin.
        """
        return self.aloft.temperature - self.ambient

    @cached_property
    def front(self) -> DataArray:
        """
        Measures each mass point's horizontal distance to the nearest one
        `burning` marks, in meters, one snapshot at a time.
        """
        return apply_ufunc(
            distances,
            self.burning,
            input_core_dims  = [["south_north", "west_east"]],
            kwargs           = {"sampling": self.sampling},
            output_core_dims = [["south_north", "west_east"]],
            vectorize        = True
        )

    @cached_property
    def heights(self) -> DataArray:
        """
        Measures the height above the ground of each mass point on the
        model's levels, in meters, as wrf-python's `height_agl` does from
        the geopotential `PH` plus `PHB`.

        The ground is the lowest staggered level, where WRF holds the
        geopotential of the terrain, so the height needs no `HGT`.

        Raises:
            ValueError: Where a height falls outside `ranges.height_m`.
        """
        geopotential = self.dataset.PH + self.dataset.PHB
        return checked(
            field = (
                (
                    destaggered(geopotential, dimension="bottom_top_stag")
                    - geopotential.isel(bottom_top_stag=0)
                ) / GRAVITY_M_S2
            ).rename("height"),
            bounds = self.ranges.height_m,
            unit   = "m"
        )

    @property
    def plume(self) -> Dataset:
        """
        Measures the fire and the air above it in each snapshot, meaning:

        - `burning_m2`, the area `burning` marks, in square meters
        - `convected_k`, the largest excess over the ambient profile in air
          the smoke tracer has not reached, which the ambient convection
          produces
        - `heated_k`, the largest excess over it above a burning cell
        - `hottest_k`, the hottest air on the fixed heights

        An excess with no air to measure it over is NaN.
        """
        excess = self.excess
        return Dataset(
            {
                "burning_m2"  : self.burning.sum(["south_north", "west_east"])
                * prod(self.sampling),
                "convected_k" : excess.where(self.smoke <= 0).max(Air.axes[1:]),
                "heated_k"    : excess.where(self.burning).max(Air.axes[1:]),
                "hottest_k"   : self.aloft.temperature.max(Air.axes[1:])
            }
        )

    @property
    def sampling(self) -> tuple[float, float]:
        """
        Reads the spacing of the rows and of the columns of mass points, in
        meters, from the output's `DY` and `DX` attributes.
        """
        return self.dataset.attrs["DY"], self.dataset.attrs["DX"]

    @property
    def smoke(self) -> DataArray:
        """
        Reads the smoke tracer `tr17_1` on the fixed heights, which the fire
        emits in proportion to the fuel it burns and the winds carry.
        """
        return self.aloft.smoke

    @cached_property
    def temperature(self) -> DataArray:
        """
        Converts the perturbation potential temperature `T` into the air
        temperature at each mass point on the model's levels, in kelvin, as
        wrf-python's `tk` does from the pressure `P` plus `PB`.

        Raises:
            ValueError: Where a temperature falls outside
                        `ranges.temperature_k`.
        """
        data = self.dataset
        return checked(
            field = (
                (data.T + BASE_K) * ((data.P + data.PB) / REFERENCE_PA) ** (
                    DRY_AIR_J_KG_K / HEAT_CAPACITY_J_KG_K
                )
            ).rename("temperature"),
            bounds = self.ranges.temperature_k,
            unit   = "K"
        )

    @cached_property
    def wind(self) -> Dataset:
        """
        Destaggers each wind component onto the mass points on the model's
        levels, as wrf-python's `ua`, `va`, and `wa` do, under the names
        `u`, `v`, and `w`.

        Raises:
            ValueError: Where a component falls outside `ranges.wind_m_s`.
        """
        return Dataset(
            {
                "u" : destaggered(self.dataset.U, dimension="west_east_stag"),
                "v" : destaggered(self.dataset.V, dimension="south_north_stag"),
                "w" : destaggered(self.dataset.W, dimension="bottom_top_stag")
            }
        ).map(checked, bounds=self.ranges.wind_m_s, unit="m/s")

    @classmethod
    def from_dataset(cls, dataset: Dataset, *, crop: Crop) -> Self:
        """
        Crops `dataset`, a WRF-SFIRE output under WRF's own names, to the
        snapshots inside `crop.window_s`, the mass points inside `crop.x_m`
        and `crop.y_m` with the staggered points bounding them, and the
        model levels reaching just past the top of the band, before any
        value but the times `XTIME` holds and the base-state geopotential of
        the first snapshot loads.

        The levels kept are read from the lowest height above the ground
        that geopotential gives each level, keeping one level more than
        the first to reach the top of the band, since the perturbation
        geopotential can lower a level.

        Raises:
            ValueError: Where the window holds no snapshot, or the
                        neighborhood no mass point along a dimension.
        """
        placed   = dataset[list(cls.variables)].reset_coords(drop=True)
        spacings = {"south_north": placed.attrs["DY"], "west_east": placed.attrs["DX"]}
        placed   = placed.assign_coords(
            Time = placed.XTIME.drop_attrs() * 60,
            **{
                **{
                    name           : (arange(placed.sizes[name]) + 0.5) * spacing,
                    f"{name}_stag" : arange(placed.sizes[f"{name}_stag"]) * spacing
                }
                for name, spacing in spacings.items()
            }
        ).sel(Time=slice(*crop.window_s or (None, None)))
        framed = placed.isel(
            {
                **spans(placed[name], bounds=bounds)
                for name, bounds in (("south_north", crop.y_m), ("west_east", crop.x_m))
            }
        )
        if empty := [name for name, size in framed.sizes.items() if not size]:
            raise ValueError(f"The crop keeps no point along {empty}")

        base   = framed.PHB.isel(Time=0)
        lowest = destaggered(
            field = ((base - base.isel(bottom_top_stag=0)) / GRAVITY_M_S2).min(
                ["south_north", "west_east"]
            ),
            dimension = "bottom_top_stag"
        )
        levels = int((lowest < crop.heights[-1]).sum()) + 2
        return cls(
            crop    = crop,
            dataset = framed.isel(
                bottom_top      = slice(levels),
                bottom_top_stag = slice(levels + 1)
            ).drop_vars("XTIME")
        )

    @classmethod
    def from_path(cls, path: Path, *, crop: Crop) -> Self:
        """
        Opens the WRF-SFIRE output at `path` through netCDF4 and crops it
        as `from_dataset` does, which reads the times and the base-state
        geopotential, leaving every other value unread until a caller reads
        it.
        """
        return cls.from_dataset(open_dataset(path, engine="netcdf4"), crop=crop)

    @classmethod
    def from_sample(cls, path: Path, *, sample: Sample, settings: Settings) -> Self:
        """
        Opens the full WRF-SFIRE output at `path` as `from_path` does,
        cropped to `settings.crop` over the window `sample` holds, and
        checks the figures it measures there through `Sample.check` against
        a flight lasting `settings.plan.duration_s`.

        Raises:
            ValueError: Where `path` names another output than
                        `sample.simulation`, or the window misses a figure
                        `Sample.check` holds it to.
        """
        if path.name != sample.simulation:
            raise ValueError(
                f"`{path.name}` is not `{sample.simulation}`, the output the sample "
                "is cut from"
            )
        output = cls.from_path(
            path,
            crop = settings.crop.model_copy(update={"window_s": sample.window_s})
        )
        sample.check(output.plume, duration_s=settings.plan.duration_s)
        return output

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """
        Opens the output `settings.simulation` names under the clone, or
        wraps the dataset `settings.threat` builds where it names none, each
        cropped to `settings.crop`.
        """
        match settings.simulation:
            case None : return cls.from_threat(settings.threat, crop=settings.crop)
            case path : return cls.from_path(
                Checkout.installed().root / path,
                crop = settings.crop
            )

    @classmethod
    def from_threat(cls, threat: ParametricThreat, *, crop: Crop) -> Self:
        """
        Wraps the WRF-shaped dataset `threat` builds, cropped as
        `from_dataset` crops a simulation.
        """
        return cls.from_dataset(threat.dataset, crop=crop)

    def write(self, path: Path, *, sample: Sample) -> Path:
        """
        Writes the cropped output to `path` as one NetCDF-4 file through
        netCDF4, each data variable compressed with zlib, `XTIME` restored
        from the snapshots' times, WRF's grid attributes giving the
        sizes the crop keeps, so `from_path` reads the file as it reads
        a simulation, and the license and the source `Sample.attributes`
        composes for `sample`.

        Returns:
            `path`, once written.
        """
        path.parent.mkdir(exist_ok=True, parents=True)
        written = self.dataset.drop_encoding().assign(
            XTIME = (self.dataset.Time / 60).astype("float32").assign_attrs(
                description = "minutes since simulation start"
            )
        ).assign_attrs(
            {
                f"{name.upper().replace('_', '-')}_{attribute}": int32(
                    self.dataset.sizes[dimension]
                )
                for name in ("bottom_top", "south_north", "west_east")
                for attribute, dimension in (
                    ("GRID_DIMENSION", f"{name}_stag"),
                    ("PATCH_END_STAG", f"{name}_stag"),
                    ("PATCH_END_UNSTAG", name)
                )
            } | sample.attributes
        )
        written.to_netcdf(
            path,
            encoding = {name: {"zlib": True} for name in written.data_vars},
            engine   = "netcdf4"
        )
        return path


def checked(field: DataArray, *, bounds: Range, unit: str) -> DataArray:
    """
    Checks every value of `field` against `bounds`, reading a missing value
    as one outside them.

    Args:
        field  : The converted field, named as a failure reports it.
        bounds : The lowest and the highest value it may take.
        unit   : The unit of its values, as a failure reports it.

    Returns:
        `field`, unchanged.

    Raises:
        ValueError: Where a value falls outside `bounds` or is missing.
    """
    floor, ceiling  = bounds
    lowest, highest = extremes(field)
    if not floor <= lowest <= highest <= ceiling:
        raise ValueError(
            f"`{field.name}` runs from {lowest:g} to {highest:g} {unit}, outside "
            f"the range from {floor:g} to {ceiling:g} {unit}"
        )

    return field


def destaggered(field: DataArray, *, dimension: str) -> DataArray:
    """
    Averages each pair of neighboring values of `field` along the staggered
    `dimension` through `rolling(...).mean()`, placing each average on the
    mass point between the pair, as wrf-python's `destagger` does.

    Returns:
        The averages along the mass dimension `dimension` names without its
        `_stag` suffix, carrying no coordinate along it.
    """
    return (
        field.rolling({dimension: 2})
             .mean()
             .isel({dimension: slice(1, None)})
             .drop_vars(dimension, errors="ignore")
             .rename({dimension: dimension.removesuffix("_stag")})
    )


def distances(burning: ndarray, sampling: tuple[float, float]) -> ndarray:
    """
    Measures each cell's distance to the nearest burning cell of one
    snapshot through `distance_transform_edt`, in the unit of `sampling`.

    Every distance is infinite in a snapshot where no cell burns, since the
    transform returns finite distances to a point outside the grid there.

    Args:
        burning  : Whether each cell burns, `(n_rows, n_columns)`.
        sampling : The spacing of the rows and of the columns.
    """
    if burning.any():
        return distance_transform_edt(~burning, sampling=sampling)

    return full(burning.shape, inf)


def extremes(field: DataArray) -> Range:
    """
    Reads the lowest and the highest value of `field`, each one NaN where a
    value is missing.
    """
    return float(field.min(skipna=False)), float(field.max(skipna=False))


def spans(coordinate: DataArray, *, bounds: Span) -> dict[str, slice]:
    """
    Finds the mass points along the dimension `coordinate` places whose
    coordinates fall inside `bounds`, and the staggered points bounding
    them.

    Returns:
        The slice of each dimension by its name, every point where `bounds`
        is `None`.
    """
    start, stop = coordinate.to_index().slice_locs(*bounds or (None, None))
    return {
        coordinate.name           : slice(start, stop),
        f"{coordinate.name}_stag" : slice(start, stop + 1)
    }


def unfilled(field: DataArray) -> DataArray:
    """
    Checks that `field`, which holds no range of its own, holds no missing
    value and nothing at netCDF's default fill value for single precision.

    A value past the fill value reads as one, as netCDF's attribute
    conventions read a positive fill value as the valid maximum, and the
    same magnitude is refused below zero.

    Returns:
        `field`, unchanged.

    Raises:
        ValueError: Where a value is missing or reaches the fill value.
    """
    fill            = default_fillvals["f4"]
    lowest, highest = extremes(field)
    if not -fill < lowest <= highest < fill:
        raise ValueError(
            f"`{field.name}` runs from {lowest:g} to {highest:g}, holding a missing "
            f"value or reaching the netCDF fill value of {fill:g}"
        )

    return field
