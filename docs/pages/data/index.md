# Data

The air *Thermur*'s flock flies through comes from the [WRF-SFIRE LES Synthetic Wildfire Plume Dataset](https://doi.org/10.20383/102.0314), the large-eddy simulations of wildfire plumes from which Moisseeva and Stull derive the height a smoke plume rises to[^moisseeva2021]. They ran each simulation in WRF-SFIRE, which couples a fire spreading over the ground to the Weather Research and Forecasting model (WRF). A clone fetches a slice of one simulation's output as a sample, and wherever no simulation is named, the flock flies through a parametric threat instead, a front of hot air laid out the way a simulation lays out its fields.

## The Simulations

The dataset holds 147 simulations, each over a domain 20 km long and 10 km wide at 40 m horizontal spacing, either 3,000 m deep on 51 levels or 5,000 m deep on 71, and each running 20 or 30 minutes after an hour of spin-up, with the fire spreading on a 4 m mesh beneath the atmosphere. The simulations vary the wind, the sounding the atmosphere starts from, the fuel, and the length of the fireline. The Federated Research Data Repository publishes the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and its files transfer through [Globus](https://www.globus.org/), which takes a Globus account.

## The Sample

The sample is `wrfout_W5F7R4`, the output of one simulation, cut to four snapshots 15 s apart, from 3,615 s to 3,660 s after the simulation starts, and it stays under the dataset's CC BY 4.0. It covers the whole domain on its 500 by 250 grid and the 50 levels of mass points between its 51 staggered levels. The ground heat flux and the smoke tracer are zero across the domain in the first two snapshots.

`mise fetch` runs `thermur fetch sample`, which downloads the archive the table names at the pinned revision of its Hugging Face dataset, checks it against the digest, extracts it under `data/`, and prints the path of the output inside it. The table is written from the record the package pins the sample in, and `mise render` rewrites it.

<!-- sample -->

| **Field** | **Default** | **Bound** | **Holds** |
|---|---|---|---|
| `archive` | `samples.tar.gz` |  | The archive the dataset publishes the sample as |
| `digest` | `6f97d0bd35b56e224720692f81747fb5e1deddfbc3d1397e6b82ddb42518acfa` |  | The SHA-256 digest of the archive |
| `folder` | `data` |  | The folder the archive extracts into, relative to the clone |
| `hub` | `https://huggingface.co` |  | The address of the Hugging Face Hub publishing the dataset |
| `length_m` | `20000.0` | above 0 | The length of the domain the output covers from west to east, in meters |
| `member` | `samples/wrf_sample.nc` |  | The WRF-SFIRE output inside the archive |
| `repository` | `Jybbs/sfire-samples` |  | The Hugging Face dataset publishing the archive |
| `revision` | `e7d4fc2ca1f99e82dfeec3ca2f7ba1c714965913` |  | The commit of the dataset the archive is read at |
| `timeout_s` | `60.0` | above 0 | The longest the download waits on the connection or on one read, in seconds |
| `times_s` | `[3615.0, 3630.0, 3645.0, 3660.0]` |  | The time of each snapshot the output holds, in seconds since the simulation started |

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
