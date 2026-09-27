# Environmental Data Integration for POSEatSea

## ERA5 + GLORYS12V1 for Oil-Spill Drift Hindcasting and Forecasting

## 1. Purpose

This document specifies how POSEatSea should use two environmental
reanalysis sources:

1.  **ERA5 hourly atmospheric reanalysis** --- primarily for 10 m wind
    forcing.
2.  **GLORYS12V1 / Copernicus Marine Global Ocean Physics Reanalysis**
    --- primarily for ocean surface currents.

Together they provide the environmental forcing needed by an oil-spill
drift model such as **OpenDrift/OpenOil**.

For the July 2020 Wakashio demonstration, the recommended combination
is:

``` text
SAR oil-spill detection
        |
        v
spill centroid + detection time
        |
        +--------------------+
        |                    |
        v                    v
ERA5 wind              GLORYS12V1 current
u10, v10               uo, vo
        |                    |
        +---------+----------+
                  |
                  v
          OpenDrift/OpenOil
                  |
       +----------+----------+
       |                     |
       v                     v
 backward hindcast      forward forecast
       |                     |
       v                     v
origin probability       future slick path
       |
       v
AIS trajectory reconstruction
       |
       v
candidate vessel attribution
```

The key design principle is:

> **Do not query the environmental API separately for every AIS point.
> Download/cache a regional spatio-temporal cube once, then interpolate
> locally.**

------------------------------------------------------------------------

# 2. ERA5 API

## 2.1 What ERA5 provides

**ERA5** is ECMWF's global atmospheric reanalysis available from 1940
onward at hourly resolution.

For POSEatSea, the most important variables are:

### Core MVP variables

  Variable                    ERA5 meaning               POSEatSea use
  --------------------------- -------------------------- --------------------------------
  `10m u-component of wind`   Eastward wind component    Wind forcing
  `10m v-component of wind`   Northward wind component   Wind forcing
  `10m wind speed`            Wind magnitude             Quality checks / visualization
  `10m wind direction`        Wind direction             Visualization / validation

The wind components should be retained rather than only storing
speed/direction because numerical drift models work naturally with
vector components.

ERA5 also contains many potentially useful atmospheric and wave
variables, including sea-surface temperature, wave parameters,
precipitation, pressure, air temperature, humidity and other fields.

The official ERA5 single-level dataset provides hourly data from 1940 to
the present.

## 2.2 Access method

ERA5 is accessed programmatically through the **Copernicus Climate Data
Store (CDS) API**.

Official setup:

``` bash
pip install "cdsapi>=0.7.7"
```

The CDS API requires an account/API token and a local configuration
file.

Typical configuration:

``` text
~/.cdsapirc
```

with:

``` yaml
url: https://cds.climate.copernicus.eu/api
key: <PERSONAL-ACCESS-TOKEN>
```

The dataset must also have its terms of use accepted through the CDS
website before programmatic retrieval.

## 2.3 ERA5 dataset

Use:

``` text
reanalysis-era5-single-levels
```

For the Wakashio example:

``` text
Date:
2020-07-09 -> 2020-07-12

Region:
around Mauritius / spill AOI

Variables:
10m u-component of wind
10m v-component of wind
```

A small regional request is preferable to downloading global ERA5.

## 2.4 Example ERA5 Python request

Illustrative implementation:

``` python
import cdsapi

client = cdsapi.Client()

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": ["2020"],
        "month": ["07"],
        "day": ["09", "10", "11", "12"],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "area": [
            -18.0,   # north
            55.0,    # west
            -23.0,   # south
            61.0,    # east
        ],
        "format": "grib",
    },
    "data/wakashio/era5/wind_20200709_20200712.grib",
)
```

The exact API request should preferably be generated from the CDS
dataset's **Show API request code** interface rather than hard-coding
assumptions about request syntax.

## 2.5 ERA5 data processing

Load the downloaded data using xarray/cfgrib or an equivalent GRIB
reader:

``` python
import xarray as xr

ds = xr.open_dataset(
    "data/wakashio/era5/wind_20200709_20200712.grib",
    engine="cfgrib"
)
```

The application should expose a simple interface:

``` python
wind = get_wind(
    lat=-20.320985,
    lon=58.280948,
    timestamp="2020-07-11T04:28:19Z"
)
```

Expected output:

``` python
{
    "u10": ...,
    "v10": ...,
    "speed": ...,
    "direction": ...
}
```

The raw `u10` and `v10` values should remain available.

------------------------------------------------------------------------

# 3. GLORYS12V1 / Copernicus Marine API

## 3.1 What GLORYS12V1 provides

The relevant Copernicus Marine product is:

``` text
Product ID:
GLOBAL_MULTIYEAR_PHY_001_030

Name:
Global Ocean Physics Reanalysis
```

The product is based on the NEMO ocean model and assimilates
observations including:

-   satellite altimetry / sea-level anomaly
-   satellite sea-surface temperature
-   sea-ice concentration
-   in-situ temperature profiles
-   in-situ salinity profiles

It provides a global ocean reanalysis on approximately:

``` text
1/12 degree
~8 km
50 vertical levels
```

The product covers the historical period beginning in 1993.

For the POSEatSea use case, the most important variables are:

  -----------------------------------------------------------------------
  Variable                Meaning                 POSEatSea use
  ----------------------- ----------------------- -----------------------
  `uo`                    Eastward sea-water      Oil drift
                          velocity                

  `vo`                    Northward sea-water     Oil drift
                          velocity                

  `thetao`                Sea-water potential     Optional validation /
                          temperature             oil weathering

  `so`                    Sea-water salinity      Optional

  sea-surface height      Surface elevation       Optional ocean-state
                                                  feature

  mixed-layer thickness   Ocean mixed layer       Optional advanced model

  sea-floor depth         Bathymetry              Coastal / grounding
                                                  logic

  sea-ice variables       Ice state/velocity      Useful for
                                                  high-latitude extension
  -----------------------------------------------------------------------

## 3.2 Important temporal-resolution limitation

For the historical GLORYS12V1 product used for July 2020:

``` text
Daily means
Monthly means
```

are provided.

The daily dataset is:

``` text
cmems_mod_glo_phy_my_0.083deg_P1D-m
```

Therefore:

> **Do not treat GLORYS12V1 daily currents as hourly observations.**

For a first POSEatSea implementation, daily current fields can be
temporally interpolated for the drift model, but the uncertainty
introduced by the daily forcing must be documented.

If an operational/high-frequency current product is required for a
recent event, use the appropriate Copernicus Marine operational current
dataset instead, after checking its historical temporal coverage.

## 3.3 Copernicus Marine Toolbox

The recommended programmatic interface is the **Copernicus Marine
Toolbox**.

Install:

``` bash
pip install copernicusmarine
```

The Toolbox supports:

-   catalogue/metadata discovery
-   spatial subsetting
-   temporal subsetting
-   variable selection
-   depth selection
-   NetCDF output
-   Zarr output
-   downloading original files
-   Python API
-   CLI

Authentication is required for data access.

## 3.4 Dataset to use

For the July 2020 historical case:

``` text
Product:
GLOBAL_MULTIYEAR_PHY_001_030

Dataset:
cmems_mod_glo_phy_my_0.083deg_P1D-m

Variables:
uo
vo
```

## 3.5 Example GLORYS12V1 download

For example:

``` python
import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",

    variables=["uo", "vo"],

    minimum_longitude=55,
    maximum_longitude=61,

    minimum_latitude=-23,
    maximum_latitude=-18,

    start_datetime="2020-07-09",
    end_datetime="2020-07-12",

    minimum_depth=0,
    maximum_depth=5,

    output_directory="data/wakashio/glorys",

    output_filename="glorys_surface_currents.nc",
)
```

For oil-spill drift, the first implementation should focus on the
near-surface layer.

Do not automatically assume that simply selecting `depth=0` is
physically equivalent to every oil particle's actual drift depth. Keep
the environmental depth configuration explicit.

## 3.6 Inspecting the downloaded NetCDF

``` python
import xarray as xr

ds = xr.open_dataset(
    "data/wakashio/glorys/glorys_surface_currents.nc"
)

print(ds)
```

Expected important fields:

``` text
uo
vo
latitude
longitude
depth
time
```

The actual coordinate names and dimensional structure must be inspected
from the downloaded file before hard-coding the reader.

## 3.7 Local interpolation API

POSEatSea should hide the raw Copernicus data format behind an internal
interface:

``` python
current = get_current(
    lat=-20.320985,
    lon=58.280948,
    timestamp="2020-07-11T04:28:19Z"
)
```

Return:

``` python
{
    "uo": ...,
    "vo": ...,
    "speed": ...,
    "direction": ...
}
```

This function should perform:

1.  nearest or bilinear spatial interpolation
2.  temporal interpolation
3.  surface/depth selection
4.  unit normalization
5.  missing-value handling
6.  optional quality flags

------------------------------------------------------------------------

# 4. Unified Environmental API

POSEatSea should expose one abstraction instead of allowing the rest of
the application to know about ERA5 and Copernicus separately.

Recommended interface:

``` python
environment = get_environment(
    lat=-20.320985,
    lon=58.2809483333,
    timestamp="2020-07-11T04:28:19Z"
)
```

Return:

``` python
{
    "timestamp": "...",

    "wind": {
        "u10": ...,
        "v10": ...,
        "speed": ...,
        "direction": ...
    },

    "current": {
        "uo": ...,
        "vo": ...,
        "speed": ...,
        "direction": ...
    }
}
```

Recommended module structure:

``` text
poseatsea/
├── data/
│   ├── era5.py
│   ├── glorys.py
│   ├── ocean_weather.py
│   └── cache.py
│
├── drift/
│   ├── hindcast.py
│   ├── forecast.py
│   └── uncertainty.py
│
└── inference/
    ├── fusion.py
    ├── vessel_crossref.py
    └── loitering.py
```

------------------------------------------------------------------------

# 5. Environmental Data Cache

The application should not depend on live data APIs every time Streamlit
starts.

Recommended structure:

``` text
data/
└── wakashio/
    ├── era5/
    │   └── wind_20200709_20200712.grib
    │
    └── glorys/
        └── glorys_surface_currents_20200709_20200712.nc
```

Recommended workflow:

``` text
User selects incident
        |
        v
Determine AOI
        |
        v
Determine detection + hindcast window
        |
        v
Check local cache
   /             \
exists           missing
 |                 |
 v                 v
load             download
 |                 |
 +-------+---------+
         |
         v
validate + interpolate
         |
         v
OpenDrift/OpenOil
```

For a hackathon/demo deployment, pre-download the Wakashio environmental
window and ship it with the project or provide a setup script.

------------------------------------------------------------------------

# 6. OpenDrift/OpenOil Integration

OpenDrift can use environmental forcing such as currents, wind and waves
through Readers. CF-compliant NetCDF files can be read with the generic
NetCDF reader.

Example:

``` python
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.openoil import OpenOil

wind_reader = reader_netCDF_CF_generic.Reader(
    "data/wakashio/era5/wind.nc"
)

current_reader = reader_netCDF_CF_generic.Reader(
    "data/wakashio/glorys_surface_currents.nc"
)

oil = OpenOil(weathering_model="noaa")

oil.add_reader([
    wind_reader,
    current_reader
])
```

The actual variable mappings must be validated against the downloaded
NetCDF/GRIB structure before final integration.

## 6.1 Forward simulation

Use the detected slick location as the initial condition:

``` python
oil.seed_elements(
    lon=spill_lon,
    lat=spill_lat,
    time=spill_time,
    number=1000,
    radius=1000
)

oil.run(
    duration=timedelta(hours=48),
    time_step=900
)
```

This provides a forecast-style trajectory.

## 6.2 Backward simulation

OpenDrift supports backward simulations using a negative time step.

Conceptually:

``` python
oil.run(
    duration=timedelta(hours=-48),
    time_step=-900
)
```

However, the implementation should follow the OpenDrift
backward-simulation pattern for the selected model/version and verify
the environmental reader's time coverage.

The output should be converted into:

``` text
origin particle cloud
        |
        v
2D density / heatmap
        |
        v
probable release area
```

------------------------------------------------------------------------

# 7. Features We Should Add to POSEatSea

## Priority 1 --- Essential

### 7.1 Automated AOI generation

From:

``` text
spill centroid
spill polygon
detection timestamp
```

automatically create an environmental-data bounding box.

Example:

``` text
spill:
lat = -20.32
lon = 58.28

AOI:
lat = -23 to -18
lon = 55 to 61
```

The AOI should expand automatically for longer hindcast windows.

------------------------------------------------------------------------

### 7.2 Automated time-window selection

Given:

``` text
spill detection time = T
```

generate:

``` text
hindcast:
T - 24 h
T - 48 h
T - 72 h

forecast:
T + 24 h
T + 48 h
T + 72 h
```

The exact duration should be configurable.

------------------------------------------------------------------------

### 7.3 Environmental data downloader

Create:

``` text
poseatsea/data/environment_downloader.py
```

Responsibilities:

-   calculate AOI
-   calculate time window
-   request ERA5
-   request GLORYS
-   save files
-   verify downloads
-   avoid duplicate downloads
-   maintain metadata

------------------------------------------------------------------------

### 7.4 Environmental interpolation layer

Create:

``` text
poseatsea/data/ocean_weather.py
```

API:

``` python
get_environment(lat, lon, timestamp)
```

It should provide:

``` text
u10
v10
wind_speed
wind_direction

uo
vo
current_speed
current_direction
```

------------------------------------------------------------------------

### 7.5 OpenDrift/OpenOil hindcasting

Create:

``` text
poseatsea/drift/hindcast.py
```

Input:

``` text
spill polygon
spill centroid
detection time
wind reader
current reader
```

Output:

``` text
particles
origin candidate region
origin probability raster
confidence
```

------------------------------------------------------------------------

### 7.6 Forward drift prediction

Create:

``` text
poseatsea/drift/forecast.py
```

Output:

``` text
+6 h
+12 h
+24 h
+48 h
+72 h
```

with:

-   predicted slick position
-   predicted area
-   predicted centroid
-   uncertainty
-   coastal intersection risk

------------------------------------------------------------------------

# 8. Advanced Features That Should Be Added

## 8.1 Wind/current uncertainty ensemble

Do not rely on one deterministic trajectory.

Run:

``` text
N = 20–50 simulations
```

with perturbations such as:

``` text
wind ±10–20%
current ±10–20%
detection time ±1–3 h
initial position ± detection uncertainty
```

Then calculate:

``` text
P(origin | observed slick)
```

This is much more useful for vessel attribution than a single backward
trajectory.

------------------------------------------------------------------------

## 8.2 Origin probability heatmap

Instead of showing only one estimated origin:

``` text
single point
```

produce:

``` text
probability density surface
```

For example:

``` text
red   = high probability
yellow = medium probability
blue  = low probability
```

The actual visualization should use a scientifically meaningful
normalized probability scale.

------------------------------------------------------------------------

## 8.3 Environmental confidence score

Calculate a confidence score from:

``` text
wind data availability
current data availability
temporal interpolation distance
spatial interpolation distance
SAR detection confidence
hindcast ensemble agreement
```

Example output:

``` text
Environmental forcing confidence: 0.86
```

Do not confuse this with a calibrated probability unless the score is
statistically calibrated.

------------------------------------------------------------------------

## 8.4 SAR look-alike rejection using wind

Very low wind conditions can produce dark SAR regions that resemble oil.

Add:

``` text
SAR dark region
       +
ERA5 wind
       |
       v
look-alike confidence
```

Use wind as one feature rather than a hard rule.

------------------------------------------------------------------------

## 8.5 Multi-temporal SAR confirmation

If multiple SAR observations exist:

``` text
SAR(t-1)
SAR(t)
SAR(t+1)
```

compare:

``` text
shape
area
centroid
backscatter
texture
orientation
```

This helps distinguish:

``` text
persistent oil
temporary low-wind dark patch
look-alike
```

------------------------------------------------------------------------

## 8.6 Wind/current trajectory agreement

For each candidate vessel, compare:

``` text
vessel position
        |
        v
predicted spill origin
```

and calculate:

``` text
distance to origin probability region
time difference
trajectory compatibility
```

This should become a major input to vessel attribution.

------------------------------------------------------------------------

# 9. Vessel Attribution Features

The environmental model should feed directly into the AIS attribution
pipeline.

## 9.1 Candidate generation

Filter AIS vessels by:

``` text
spatial radius
temporal window
ship type
navigation status
speed
trajectory
```

------------------------------------------------------------------------

## 9.2 Origin proximity score

Calculate:

``` text
minimum distance from vessel trajectory
to high-probability origin region
```

------------------------------------------------------------------------

## 9.3 Temporal compatibility

Check whether:

``` text
vessel was present
```

during the estimated spill-origin interval.

------------------------------------------------------------------------

## 9.4 Trajectory compatibility

Compare:

``` text
vessel heading
vessel speed
vessel trajectory
```

against the estimated release point.

------------------------------------------------------------------------

## 9.5 Behavioural anomaly

Detect:

``` text
unexpected slowdown
course change
loitering
turning
stopping
AIS blackout
```

around the estimated origin window.

------------------------------------------------------------------------

## 9.6 Dark-vessel detection

If SAR identifies a vessel but no corresponding AIS position exists:

``` text
SAR vessel
    |
    +--> AIS match
    |       |
    |       +--> normal candidate
    |
    +--> no AIS match
            |
            +--> dark-vessel flag
```

The system should preserve this as a separate evidence signal rather
than automatically declaring the vessel responsible.

------------------------------------------------------------------------

# 10. Other Environmental Features That MAY Help

These are not required for the first MVP but can improve scientific
realism.

## 10.1 Stokes drift

Wave-induced surface drift can influence floating oil.

Potential use:

``` text
ocean current
+
wind drift
+
Stokes drift
=
more complete surface transport model
```

ERA5 contains wave-related variables that can support an advanced
implementation.

------------------------------------------------------------------------

## 10.2 Significant wave height

Useful for:

-   weathering interpretation
-   dispersion conditions
-   uncertainty estimation
-   operational risk

------------------------------------------------------------------------

## 10.3 Sea-surface temperature

Useful for:

-   oil weathering
-   viscosity-related interpretation
-   environmental context
-   anomaly analysis

------------------------------------------------------------------------

## 10.4 Mixed-layer depth

Useful for advanced dispersion modelling.

GLORYS provides mixed-layer information.

------------------------------------------------------------------------

## 10.5 Bathymetry

Useful for:

-   coastal proximity
-   shallow-water constraints
-   grounding/ship-behaviour interpretation
-   coastal impact assessment

Use the GLORYS/static bathymetry or a dedicated bathymetry dataset.

------------------------------------------------------------------------

## 10.6 Sea-level / mesoscale circulation

Sea-surface height can be used to derive or contextualize mesoscale
circulation.

This is potentially useful when explaining why the slick moved in a
particular direction.

------------------------------------------------------------------------

## 10.7 Salinity and temperature profiles

Mostly an advanced feature.

Potential uses:

-   ocean-state characterization
-   advanced oil fate modelling
-   water-mass analysis

Not necessary for the first demo.

------------------------------------------------------------------------

# 11. Recommended Feature Priority

  Priority   Feature                             Why
  ---------- ----------------------------------- ----------------------------------
  P0         ERA5 u10/v10 ingestion              Required wind forcing
  P0         GLORYS uo/vo ingestion              Required ocean-current forcing
  P0         AOI/time-window downloader          Makes pipeline reproducible
  P0         Local environmental cache           Avoid live API dependency
  P0         Environment interpolation API       Connects data to models
  P0         OpenDrift/OpenOil forward run       Slick trajectory
  P0         OpenDrift/OpenOil backward run      Spill-origin estimation
  P0         Origin probability heatmap          Core SIH output
  P0         AIS + origin cross-reference        Vessel attribution
  P1         Ensemble hindcast                   Uncertainty
  P1         SAR wind look-alike rejection       Reduce false positives
  P1         Dark-vessel detection               SAR-AIS mismatch
  P1         AIS behavioural analysis            Candidate ranking evidence
  P1         Forward spill forecast              Operational usefulness
  P1         Environmental confidence score      Explain model reliability
  P2         Stokes drift                        More realistic surface transport
  P2         Wave height                         Weather/oil-fate context
  P2         SST                                 Weathering/context
  P2         Multi-temporal SAR                  Stronger spill validation
  P2         Bathymetry                          Coastal constraints
  P3         Mixed-layer depth                   Advanced physics
  P3         Salinity/temperature profiles       Advanced ocean modelling
  P3         Transformer trajectory prediction   Optional ML enhancement

------------------------------------------------------------------------

# 12. Recommended POSEatSea End-to-End Pipeline

``` text
                    ┌────────────────────┐
                    │ Sentinel-1 SAR     │
                    └─────────┬──────────┘
                              │
                              v
                    ┌────────────────────┐
                    │ Oil-spill detector │
                    └─────────┬──────────┘
                              │
                  spill polygon + centroid
                              │
                ┌─────────────┴─────────────┐
                │                           │
                v                           v
          ERA5 wind                    GLORYS12V1
          u10 / v10                    uo / vo
                │                           │
                └─────────────┬─────────────┘
                              │
                              v
                    ┌────────────────────┐
                    │ OpenDrift/OpenOil  │
                    └─────────┬──────────┘
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  v                       v
             Backward                   Forward
             hindcast                   forecast
                  │                       │
                  v                       v
         origin probability        future slick path
                  │
                  v
           ┌──────────────┐
           │ Historical   │
           │ AIS traffic  │
           └──────┬───────┘
                  │
                  v
       ┌──────────────────────────┐
       │ Candidate vessel engine  │
       ├──────────────────────────┤
       │ proximity                │
       │ time compatibility       │
       │ trajectory compatibility │
       │ speed/course anomalies   │
       │ loitering                │
       │ AIS blackout             │
       │ SAR-AIS mismatch         │
       └────────────┬─────────────┘
                    │
                    v
          ┌──────────────────────┐
          │ Evidence-based       │
          │ attribution report   │
          └──────────────────────┘
```

------------------------------------------------------------------------

# 13. Recommended Repository Changes

``` text
poseatsea/
│
├── data/
│   ├── era5.py
│   ├── glorys.py
│   ├── ocean_weather.py
│   ├── environment_downloader.py
│   └── cache.py
│
├── drift/
│   ├── hindcast.py
│   ├── forecast.py
│   ├── uncertainty.py
│   └── openoil.py
│
├── inference/
│   ├── fusion.py
│   ├── vessel_crossref.py
│   ├── dark_vessel.py
│   └── loitering.py
│
├── visualization/
│   ├── drift_map.py
│   ├── origin_heatmap.py
│   ├── current_overlay.py
│   └── ais_timeline.py
│
└── reports/
    └── forensic_report.py
```

------------------------------------------------------------------------

# 14. MVP Implementation Order

### Milestone 1 --- Data

Implement:

``` text
ERA5 downloader
GLORYS downloader
local cache
metadata validation
```

Success condition:

``` text
Wakashio AOI
+
2020-07-09 to 2020-07-12
+
wind/current NetCDF/GRIB
```

can be reproduced automatically.

### Milestone 2 --- Environmental API

Implement:

``` python
get_environment(lat, lon, timestamp)
```

and verify values manually at several points.

### Milestone 3 --- Drift

Integrate:

``` text
ERA5 -> wind reader
GLORYS -> current reader
OpenOil -> drift simulation
```

Run:

``` text
24h forward
48h forward
24h backward
48h backward
```

### Milestone 4 --- Origin estimation

Generate:

``` text
particle cloud
origin polygon
origin probability heatmap
```

### Milestone 5 --- AIS attribution

Use:

``` text
origin probability region
+
origin time window
+
AIS trajectories
```

to filter candidates.

### Milestone 6 --- Evidence scoring

Combine:

``` text
SAR evidence
+
drift evidence
+
AIS proximity
+
time compatibility
+
trajectory compatibility
+
behavioural evidence
+
AIS availability
```

into an explainable candidate-evidence table.

### Milestone 7 --- UI

Display:

``` text
SAR spill
+
origin probability
+
wind vectors
+
ocean currents
+
AIS tracks
+
candidate vessels
+
timeline
```

and export a forensic-style report.

------------------------------------------------------------------------

# 15. Important Engineering Rules

1.  **Never download global datasets when a regional subset is
    sufficient.**
2.  **Cache environmental data locally.**
3.  **Never query the API once per AIS point.**
4.  **Keep UTC timestamps throughout the pipeline.**
5.  **Keep raw vector components (`u`, `v`) rather than only
    speed/direction.**
6.  **Record the exact dataset ID and retrieval parameters in
    metadata.**
7.  **Record data resolution and interpolation method.**
8.  **Do not describe daily GLORYS currents as hourly currents.**
9.  **Keep environmental uncertainty separate from AIS uncertainty.**
10. **Do not treat a high attribution score as proof of
    responsibility.**
11. **Keep all intermediate particle trajectories for auditability.**
12. **Make the entire Wakashio demonstration reproducible from a single
    configuration file.**

------------------------------------------------------------------------

# 16. Suggested Scenario Configuration

``` yaml
scenario:
  name: "Wakashio_2020"
  detection_time: "2020-07-11T04:28:19Z"
  center:
    lat: -20.320985
    lon: 58.2809483333

  environment:
    start: "2020-07-09T00:00:00Z"
    end: "2020-07-12T23:00:00Z"

    era5:
      dataset: "reanalysis-era5-single-levels"
      variables:
        - "10m_u_component_of_wind"
        - "10m_v_component_of_wind"

    glorys:
      product: "GLOBAL_MULTIYEAR_PHY_001_030"
      dataset: "cmems_mod_glo_phy_my_0.083deg_P1D-m"
      variables:
        - "uo"
        - "vo"
      depth:
        min: 0
        max: 5

  drift:
    hindcast_hours: 48
    forecast_hours: 48
    timestep_seconds: 900
    ensemble_runs: 30
```

This configuration should become the single source of truth for the
environmental part of the demo.

------------------------------------------------------------------------

# 17. Useful Official Resources

-   ERA5 hourly single levels: Copernicus Climate Data Store
-   CDS API setup and authentication
-   Copernicus Marine Global Ocean Physics Reanalysis
-   Copernicus Marine Toolbox Python API
-   OpenDrift documentation
-   OpenOil documentation

The implementation should always check the live catalogue metadata
before downloading because dataset versions, temporal coverage and
access mechanisms can change.

------------------------------------------------------------------------

# 18. Bottom Line

For the **July 2020 Wakashio demonstration**, the recommended
environmental stack is:

``` text
ERA5
  |
  +-- hourly 10 m wind
  |      u10
  |      v10
  |
  v
OpenDrift/OpenOil
  ^
  |
GLORYS12V1
  |
  +-- historical ocean currents
       uo
       vo
```

Then:

``` text
SAR
  -> spill detection
  -> centroid/time
  -> ERA5 + GLORYS
  -> backward OpenOil
  -> origin probability
  -> historical AIS
  -> candidate vessels
  -> explainable evidence
  -> report
```

This gives POSEatSea the environmental-data layer required to move from
**"we detected an oil-like region"** to **"this is the estimated origin
region and time, and these vessels are the AIS trajectories that are
spatially and temporally compatible with that origin."**
