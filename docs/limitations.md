# Limitations

What to know before relying on a result.

## Data

- **Sentinel-3 starts in 2016** (Sentinel-3B in 2018) and Sentinel-2 L2A is
  complete from about 2018, so time series go back eight to ten summers.
  ESA has reprocessed the temperature product more than once: compare
  differences within a city across years rather than absolute values.
- **Clouds remove data, nothing fills it.** A cloudy cell is empty, and a
  cloudy summer gives few usable passes (Guadalajara and Lagos in August gave
  too few to evaluate).
- **A satellite column is not a surface concentration.** Sentinel-5P and CAMS
  measure the whole atmosphere; any conversion to ground level is a separate,
  explicit and labelled estimate.
- **Sentinel-1**: the `pc_rtc` and `snap` backends are checked on real data;
  `hyp3_rtc` has never run against a real submission and `s1ard` is broken
  by a bug in an external library.

## Downscaling

- `local_trees` was checked against Landsat on two passes over one city:
  better on one, level on the other. More cities and seasons are needed
  before calling it a measured gain.
- Sharpened Landsat at 30 m cannot be checked against a real 30 m thermal
  measurement; the degradation test shows it adds correct detail at 180 to
  270 m, and nothing can show more without a finer thermal sensor.
- The 100 m maps inherit the timing of Sentinel-3 (about 10:00 local time by
  day) and the quality of the matched Sentinel-2 image.
- Prediction intervals are not calibrated on real data yet.

## Areas

- Grids are rectangles over the area's bounding box. A polygon masks the
  results and the statistics, but a long thin area still costs its whole box.
- Areas crossing the 180 degree meridian are not supported, and very wide
  areas are distorted at their edges by the single UTM zone used.

## Running it

Measured on Berlin with `scripts/measure_resources.py` (one process per run,
21 days of daytime Sentinel-3 with the cloud probe, Sentinel-2 and terrain,
downscaling on):

| Run | Area | First run | Same run again | Peak memory | Disk (cache + result) |
|---|---|---|---|---|---|
| Sentinel-3 to 100 m | 474 km2 | 13 to 21 min | 68 s | 0.36 GB | 15 + 26 MB |
| Sentinel-3 to 100 m, whole city | 1,708 km2 | about 8 min* | 117 s | 0.48 GB | 35 + 64 MB |
| Landsat to 30 m (7 weeks) | 474 km2 | 14 min | 28 s | 1.3 to 1.45 GB | 160 + 280 MB |

\* part of its Sentinel-2 was already cached by an interrupted run.

- **A first run is mostly downloading**: Sentinel-3 subsets and cloud
  probes, and Sentinel-2 read from Planetary Computer. Everything read is
  kept as an AOI subset, so a rerun, or another analysis of the same area
  and grid, downloads nothing. Several summers are one request each, about
  15 to 20 minutes the first time and 0.4 GB, whatever the number of years.
- **Memory follows the area and the resolution, not the period**: 100 m
  stays under 0.5 GB even for a whole city; 30 m needs about 3 GB per
  1,000 km2 (an estimate from the run above, a whole-city 30 m run was not
  measured). Requests are checked against a size limit before downloading
  (16 GB in the library, 8 GB in the worker); run one large job at a time on
  a 16 GB machine.
- The worker is for one trusted user on a private network: one shared token,
  no TLS of its own, one job at a time by default.
- Provider credentials expire (the Earthdata token after 60 days);
  `citycube check-credentials` and the worker's `/health/ready` report it.

A complete status of every component, and what comes next, is kept in
[`dev/roadmap.md`](https://github.com/JSempereH/citycube/blob/main/dev/roadmap.md).
