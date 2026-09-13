# Installation and Credentials

## Installation

```bash
uv sync --extra dev --extra auxiliary --extra optical
```

For Sentinel-1 processing, pick one of three `sentinel1_backend` values:

```bash
uv sync --extra sar    # "snap" (the default) - needs SNAP's `gpt` on PATH
uv sync --extra hyp3   # "hyp3_rtc" - no SNAP/native deps, needs Earthdata credentials below
uv sync --extra s1ard  # "s1ard" - confirmed broken, see docs/roadmap.md; kept in case upstream fixes it
```

`"snap"` (`process_s1_grd`) is the default and only needs SNAP's `gpt`
executable on `PATH` - it builds its own GPT XML graph and never touches
`pyroSAR`/`spatialist`. `"hyp3_rtc"` needs neither SNAP nor a local DEM: it
submits the granule name to ASF HyP3 and processes entirely in the cloud,
at the cost of HyP3 processing credits and queue time. `"s1ard"` (the
`pyrosar`-based NRB processor) is confirmed broken end to end against a
real scene as of `spatialist==0.20.1` (an upstream GDAL/`spatialist`
type incompatibility) - `process_s1_ard` raises a `UserWarning` if used.

## `.env`

Start from the template:

```bash
cp .env.example .env
chmod 600 .env
```

### CDSE

Create credentials in [Copernicus Data Space](https://dataspace.copernicus.eu/):

```dotenv
CDSE_CLIENT_ID=...
CDSE_CLIENT_SECRET=...
CDSE_USERNAME=...
CDSE_PASSWORD=...
CDSE_CATALOG_URL=https://catalogue.dataspace.copernicus.eu/odata/v1/Products
CDSE_DOWNLOAD_URL=https://download.dataspace.copernicus.eu/odata/v1/Products
CDSE_TOKEN_URL=https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
```

### ERA5

Register with [CDS](https://cds.climate.copernicus.eu/), create a personal
access token through [CDS API setup](https://cds.climate.copernicus.eu/how-to-api),
accept the dataset license, and create `~/.cdsapirc`:

```yaml
url: https://cds.climate.copernicus.eu/api
key: YOUR_CDS_TOKEN
```

```bash
chmod 600 ~/.cdsapirc
```

### CAMS

Register with [ADS](https://ads.atmosphere.copernicus.eu/), create a token
through [ADS API setup](https://ads.atmosphere.copernicus.eu/how-to-api),
accept the [CAMS EAC4 license](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4?tab=download#manage-licences),
and configure the ADS endpoint in `.env`:

```dotenv
CAMS_API_URL=https://ads.atmosphere.copernicus.eu/api
CAMS_API_KEY=YOUR_ADS_TOKEN
```

Do not confuse `CAMS_API_URL` with the CDS URL used for ERA5.

### OpenAQ

Register with [OpenAQ Explorer](https://explore.openaq.org/register), create a
key in [Account settings](https://explore.openaq.org/account), and add:

```dotenv
OPENAQ_API_KEY=YOUR_OPENAQ_KEY
```

The API endpoint is `https://api.openaq.org/v3`.

### Carbon Mapper

No credentials needed - the `carbon_mapper` auxiliary provider reads one
public, static Zenodo file (the 2020-2021 airborne methane plume catalog,
[10.5281/zenodo.7072824](https://doi.org/10.5281/zenodo.7072824)), not a
live API. It does need the `.xls` parser:

```bash
uv sync --extra carbon_mapper
```

### Earthdata (ECOSTRESS and `sentinel1_backend="hyp3_rtc"`)

Both sit behind the same `urs.earthdata.nasa.gov` login. Generate a
personal token at [Earthdata profile](https://urs.earthdata.nasa.gov/profile)
(Generate Token) and add it to `.env`:

```dotenv
EARTHDATA_BEARER_TOKEN=your_earthdata_user_token
```

`sentinel1_backend="hyp3_rtc"` alone also accepts a plain username/password
instead (`hyp3_sdk` supports both):

```dotenv
EARTHDATA_USERNAME=your_earthdata_username
EARTHDATA_PASSWORD=your_earthdata_password
```

A `~/.netrc` entry alone is not sufficient for ECOSTRESS - NASA's Earthdata
Cloud needs a real OAuth2 flow `~/.netrc` Basic Auth cannot complete.

## Safe Credential Checks

These commands only show whether a credential exists, never its value:

```bash
uv run --extra auxiliary python -c 'import cdsapi; c=cdsapi.Client(quiet=True); print(c.url, bool(c.key))'
uv run python -c 'from sentinel_analysis.providers import CAMSConfig, OpenAQConfig; print(bool(CAMSConfig.from_env().api_key), bool(OpenAQConfig.from_env().api_key))'
```
