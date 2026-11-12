# Data

The air *Thermur*'s flock flies through comes from the [WRF-SFIRE LES Synthetic Wildfire Plume Dataset](https://doi.org/10.20383/102.0314), the large-eddy simulations of wildfire plumes from which Moisseeva and Stull derive the height a smoke plume rises to[^moisseeva2021]. They ran each simulation in WRF-SFIRE, which couples a fire spreading over the ground to the Weather Research and Forecasting model (WRF). A clone fetches a slice of one simulation's output as a sample, and wherever no simulation is named, the flock flies through a parametric threat instead, a front of hot air laid out the way a simulation lays out its fields.

## The Simulations

The dataset holds 147 simulations, each over a domain 20 km long and 10 km wide at 40 m horizontal spacing, either 3,000 m deep on 51 levels or 5,000 m deep on 71, and each running 20 or 30 minutes after an hour of spin-up, with the fire spreading on a 4 m mesh beneath the atmosphere. The simulations vary the wind, the sounding the atmosphere starts from, the fuel, and the length of the fireline. The Federated Research Data Repository publishes the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and serves each file over HTTPS with no account, as well as through [Globus](https://www.globus.org/), which takes a Globus account.

## The Sample

The sample is `wrfout_W5F7R4`, the output of one simulation 3,000 m deep whose fire burns for 20 minutes after an hour of spin-up, cut to its last two minutes, and it stays under the dataset's CC BY 4.0. It covers the whole domain on its 500 by 250 grid and the model levels the default crop keeps, holding the variables the graph reads:

| **Variable** | **Holds** |
|---|---|
| `GRNHFX` | *The ground heat flux the fire releases, in W/m²* |
| `T` | *The perturbation potential temperature, in kelvin, over a base of 300 K* |
| `P`, `PB` | *The perturbation pressure and the base-state pressure, in pascals* |
| `PH`, `PHB` | *The perturbation geopotential and the base-state geopotential on the staggered levels, in m²/s²* |
| `U`, `V`, `W` | *The winds along each axis on the faces of each cell, in m/s* |
| `tr17_1` | *The smoke tracer the fire emits* |
| `XTIME` | *The time of each snapshot, in minutes since the simulation started* |
| `Time` | *The time of each snapshot, in seconds since the simulation started* |
| `south_north`, `west_east`, `south_north_stag`, `west_east_stag` | *The position of each cell's center and of each face along each axis, in meters from the domain's southwest corner* |

The table below pins the snapshots the window holds, the area of the cells whose ground heat flux shows burning in its first snapshot and its last, and the hottest air on the fixed heights. In every snapshot, the hottest air above the burning cells exceeds the median at its height by more than any air the smoke has not reached exceeds it in any snapshot, since only the ambient convection warms that air.

`mise fetch` runs `thermur fetch sample`, which downloads the file the table names from `host`, under a key carrying its digest, checks it against that digest, moves it into `data/samples/`, and prints its path. A later cut takes a key and an address of its own rather than replacing this one. `thermur cut sample` writes the same file from the full `wrfout_W5F7R4` the dataset publishes, cropping it to the snapshots `times_s` pins under the crop below and writing it as one NetCDF-4 file, each data variable compressed with zlib, and it refuses the cut where the file is not the one `simulation` names or the window misses any figure the table pins. The table is written from the record the package pins the sample in, and `mise render` rewrites it.

<!-- sample -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `burning_m2` | `[1272000.0, 1443200.0]` |  | The area of the cells whose ground heat flux shows burning, in the first snapshot and in the last, in square meters |
| `digest` | `e046eeddac6576bea4922804fe36a15c585762e19b0681c76898a95e63521946` |  | The SHA-256 digest of the file |
| `file` | `wrf_sample.nc` |  | The file the bucket serves the sample as, under the name the fetch writes it by |
| `folder` | `data/samples` |  | The folder the fetch writes the file into, relative to the clone |
| `host` | `https://data.thermur.fyi` |  | The address of the public Cloudflare R2 bucket serving the file, on its custom domain |
| `hottest_k` | `322.0` | above 0 | The hottest air on the fixed heights in any snapshot, in kelvin |
| `length_m` | `20000.0` | above 0 | The length of the domain the output covers from west to east, in meters |
| `simulation` | `wrfout_W5F7R4` |  | The output of the plume dataset the sample is cut from |
| `timeout_s` | `60.0` | above 0 | The longest the download waits on the connection or on one read, in seconds |
| `times_s` | `[4680.0, 4695.0, 4710.0, 4725.0, 4740.0, 4755.0, 4770.0, 4785.0, 4800.0]` |  | The time of each snapshot the output holds, in seconds since the simulation started |

<!-- /sample -->

The graph reads the sample once the `simulation` setting names the path the command prints, set in the `[tool.thermur]` table of `pyproject.toml` or through the `THERMUR_SIMULATION` variable:

```toml
[tool.thermur]
simulation = "data/samples/wrf_sample.nc"
```

## Reading an Output

The graph opens an output without reading it and crops it to the window, the columns, and the heights the table below sets before any field but the times and the base-state geopotential loads, then converts each field the way the [wrf-python diagnostics](https://wrf-python.readthedocs.io/en/latest/diagnostics.html) define it:

- The air temperature, from the perturbation potential temperature `T` and the pressure `P` plus `PB`
- The height above the ground, from the geopotential `PH` plus `PHB`, along which each column is interpolated onto evenly spaced heights
- The winds `U`, `V`, and `W`, each averaged from the two faces of a cell onto its center
- The air temperature in excess of the median at the same height and time, which leaves out heated air covering less than half of that height
- The smoke tracer `tr17_1`, and the distance from each column to the nearest cell whose ground heat flux `GRNHFX` shows burning

The menus below set how much of an output the graph reads, and the lines beneath them are what the picks write under `[tool.thermur.crop]`. The strip above the menus is a sketch with made-up sizes, its heights on a square root and its flock the front page's, which shows as many birds as the share of the most the menus can keep and draws nothing from the output.

<!-- kept -->

The median and the burning cells come from the columns the crop keeps, so a crop lying wholly behind the front shows no excess, and one holding no burning cell shows an infinite distance.

The table is written from the record the crop is set in, each field set under `[tool.thermur.crop]`, and `mise render` rewrites it.

<!-- crop -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `floor_m` | `20.0` | above 0 | The lowest fixed height above the ground, in meters |
| `levels` | `25` | at least 1 | The number of fixed heights, which reach 500 m above the ground at the default floor and spacing |
| `spacing_m` | `20.0` | above 0 | The distance between neighboring fixed heights, in meters |
| `window_s` | `None` |  | The first and the last time kept, in seconds since the simulation started, or `None` to keep every snapshot |
| `x_m` | `None` |  | The westernmost and the easternmost mass point kept, in meters from the domain's western edge, or `None` to keep every column |
| `y_m` | `None` |  | The southernmost and the northernmost mass point kept, in meters from the domain's southern edge, or `None` to keep every row |

<!-- /crop -->

Each conversion refuses a field outside its range below, which a misread variable or a fill value falls past, and names the field beside the values it holds. The air temperature runs from the lowest recorded at the Earth's surface[^turner2009] to the 1,330 °C that peak air temperatures exceeded inside a crown fire[^butler2004], the winds reach the world-record gust[^courtney2012] either way, and the heights run from the ground to the top of the dataset's deepest simulations. The smoke tracer and the ground heat flux take no range, so each refuses only a missing value or a value reaching the netCDF fill value, naming the variable beside the values it holds. The table is written from the record declaring the ranges, and `mise render` rewrites it.

<!-- ranges -->

| **Field** | **Default** | **Holds** |
|---|---|---|
| `height_m` | `[0.0, 5000.0]` | The height above the ground a mass point may take, in meters, from the ground to the top of the plume dataset's deepest simulations |
| `temperature_k` | `[183.95, 1603.15]` | The air temperature a mass point may take, in kelvin, from the -89.2 °C recorded at Vostok station, the lowest at the Earth's surface, to the 1,330 °C peak air temperatures exceeded inside a boreal crown fire |
| `wind_m_s` | `[-113.3, 113.3]` | The wind along each axis a mass point may take, in meters per second, either way up to the 113.3 m/s gust recorded on Barrow Island, the world record |

<!-- /ranges -->

## The Parametric Threat

The parametric threat is a front of hot air entering the domain at its western edge as the first snapshot is written and advancing east at a steady speed, laid out under the names and on the grid a simulation writes, so it passes through every conversion above. Ahead of the front, the air holds the potential temperature of a still atmosphere cooling with height at the dry adiabatic lapse rate, and behind it, the air is warmer by the anomaly, which falls toward nothing along a hyperbolic tangent whose scale `edge_m` sets. The smoke tracer follows the same profile, and each cell the front has reached shows burning. The front and five times `edge_m` ahead of it stay at or west of the center of the first column the median at each height reads, through the last snapshot, so that median stays the air ahead of the front.

The graph reads the threat wherever `simulation` names no output, so a fresh clone and the suite run every conversion without the network. The table is written from the record the threat is set in, each field set under `[tool.thermur.threat]`, and `mise render` rewrites it.

The drawing below runs the threat's own formula at ten times its pace, so the hot air behind the front keeps its shape and slides east, and each cell along the ground warms once as the front passes it.

<!-- threat -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `anomaly_k` | `20.0` | above 0 | The potential temperature behind the front in excess of the air ahead of it, in kelvin |
| `edge_m` | `40.0` | above 0 | The scale of the hyperbolic tangent carrying the anomaly to zero ahead of the front, in meters, one grid spacing by default, so 76% of the anomaly falls within this distance on either side of the front |
| `grid.columns` | `100` | at least 1 | The number of mass points from west to east |
| `grid.interval_s` | `15.0` | above 0 | The time between neighboring snapshots, in seconds, the interval each output of the plume dataset is written at |
| `grid.level_m` | `20.0` | above 0 | The depth of each model level, in meters |
| `grid.levels` | `30` | at least 1 | The number of model levels |
| `grid.rows` | `50` | at least 1 | The number of mass points from south to north |
| `grid.snapshots` | `9` | at least 1 | The number of snapshots, which span two minutes at the default interval |
| `grid.spacing_m` | `40.0` | above 0 | The distance between neighboring mass points, in meters, the horizontal spacing of the plume dataset |
| `speed_m_s` | `1.0` | at least 0 | The speed at which the front advances east, in meters per second |

<!-- /threat -->

[^moisseeva2021]: Moisseeva and Stull 2021. "Wildfire smoke-plume rise: a simple energy balance parameterization." *Atmospheric Chemistry and Physics* 21(3): 1407–1425. <https://doi.org/10.5194/acp-21-1407-2021>

[^turner2009]: Turner et al. 2009. "Record low surface air temperature at Vostok station, Antarctica." *Journal of Geophysical Research: Atmospheres* 114(D24): 2009JD012104. <https://doi.org/10.1029/2009JD012104>

[^butler2004]: Butler et al. 2004. "Measurements of radiant emissive power and temperatures in crown fires." *Canadian Journal of Forest Research* 34(8): 1577–1587. <https://doi.org/10.1139/x04-060>

[^courtney2012]: Courtney et al. 2012. "Documentation and verification of the world extreme wind gust record: 113.3 m/s on Barrow Island, Australia, during passage of tropical cyclone Olivia." *Australian Meteorological and Oceanographic Journal* 62(1): 1–9. <https://doi.org/10.22499/2.6201.001>
