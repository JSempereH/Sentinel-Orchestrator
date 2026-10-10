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
- The 100 m maps inherit the timing of Sentinel-3 (about 10:00 local time by
  day) and the quality of the matched Sentinel-2 image.
- Prediction intervals are not calibrated on real data yet.

## Areas

- Grids are rectangles over the area's bounding box. A polygon masks the
  results and the statistics, but a long thin area still costs its whole box.
- Areas crossing the 180 degree meridian are not supported, and very wide
  areas are distorted at their edges by the single UTM zone used.

## Running it

- Everything is processed in memory. Requests are checked against a size
  limit before downloading (16 GB in the library, 8 GB in the worker), but
  run one large job at a time on a 16 GB machine.
- The worker is for one trusted user on a private network: one shared token,
  no TLS of its own, one job at a time by default.
- Provider credentials expire (the Earthdata token after 60 days);
  `citycube check-credentials` and the worker's `/health/ready` report it.

A complete status of every component, and what comes next, is kept in
[`dev/roadmap.md`](https://github.com/JSempereH/citycube/blob/main/dev/roadmap.md).
