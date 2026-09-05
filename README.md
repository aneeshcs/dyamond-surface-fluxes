# dyamond-surface-fluxes

Surface flux analysis of the coupled **GEOS–MITgcm DYAMOND** simulation (c1440–LLC2160):
a 14-month (2020-01-20 to 2021-03-26) global coupled Nature Run with a ~7 km cubed-sphere
GEOS atmosphere (72 levels) and a ~2–4 km LLC2160 MITgcm ocean (90 levels).

This repository computes and maps the surface heat flux terms:

- ocean-side fluxes from the MITgcm diagnostics — net surface heat flux `oceQnet`,
  net shortwave `oceQsw`, and the non-solar residual
  $Q_{ns} = Q_{net} - Q_{sw}$ (latent + sensible + net longwave);
- atmosphere-side components from the GEOS collections (`EFLUX`, `HFLUX`, `SWGNT`,
  `LWGNT`) for the full decomposition
  $Q_{net} = SW_{net} + LW_{net} - LH - SH$ and a cross-grid closure check.

All fluxes are analyzed in a **positive-downward** convention (positive warms the ocean),
in W m⁻².

## Data access (SciServer)

The ~2.5 PB dataset is hosted by the [Poseidon Project](https://www.poseidon-ocean.net/)
and must be analyzed in place on [SciServer](https://www.sciserver.org/) — there is no bulk
download. Following the
[access instructions](https://www.poseidon-ocean.net/access-process-for-the-dyamond-dataset-on-sciserver/):

1. Create a SciServer account and log in.
2. In **Compute**, create a container with:
   - **Domain**: Kraken
   - **Compute Image**: Oceanography if available, otherwise SciServer Essentials works
   - **Data Volume**: Poseidon DYAMOND (ceph) — must be checked at container creation
3. The data live on the dedicated read-only filesystem
   `/home/idies/workspace/poseidon-DYAMOND/C1440-LLC2160_incoming/` (verified 2026-09;
   `dyamond_root()` also tries the older `poseidon_ceph/DYAMOND` location cited by the
   access instructions, now an empty stub).

### Data layout (raw staging format)

```
C1440-LLC2160_incoming/
├── mit/                 # MITgcm ocean: raw MDS binary, one dir per variable
│   ├── oceQnet/         #   oceQnet.<iteration>.data, one file per output step
│   │                    #   (compact LLC: big-endian float, (13·2160, 2160) per level)
│   ├── oceQsw/  oceFWflx/  oceTAUX/  oceTAUY/  SST/ ...
│   ├── grid/            #   XC, YC, RAC, Depth, AngleCS, AngleSN (.data/.meta)
│   └── readme.txt
└── holding/             # GEOS atmosphere: NetCDF, one dir per collection
    ├── tavg_15mn_2d_flx_Mx/   # 2-D surface turbulent fluxes, 15-min averages
    ├── geosgcm_surf/          # surface diagnostics (hourly)
    ├── geos_c1440_lats_lons_2D.nc  # cubed-sphere cell-center coordinates
    └── ... (inst_01hr_3d_*, tavg_01hr_3d_*, ...)
```

Ocean time base: iteration 0 ↔ 2020-01-19 21:00 UTC with the 45 s coupled timestep
(hourly output = 80 iterations); `mds.open_mds_variable` builds the time axis from the
filenames. GEOS file timestamps are parsed directly from the filenames.

The earlier consolidated Zarr described by the official demo notebook
([`hainegroup/Poseidon-share`](https://github.com/hainegroup/Poseidon-share),
`Kraken/DYAMOND_access_demonstration.ipynb`) predates the migration to this staging
area; `catalog.py` still supports Zarr stores for local subsets and any future
re-consolidation.

## Setup

On SciServer (inside the Oceanography container terminal):

```bash
git clone https://github.com/<user>/dyamond-surface-fluxes.git
cd dyamond-surface-fluxes
pip install --user -e ".[maps]"
```

> **Note:** SciServer containers do not persist `pip install --user` (`~/.local`)
> across container restarts — rerun the install (and restart the notebook kernel)
> after creating or restarting a container. As a safety net, each notebook's first
> cell falls back to importing the package directly from the repo's `src/` tree if
> the install is missing.

Locally (for development and unit tests; no data access required):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The data root defaults to the SciServer path; point `DYAMOND_ROOT` at any directory of
Zarr stores (e.g., a small extracted subset) to run elsewhere.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/dyamond_fluxes/mds.py` | Lazy dask-backed reader for raw MITgcm MDS binary (per-variable `.data` files, grid, iteration→time) |
| `src/dyamond_fluxes/geos.py` | GEOS NetCDF collection readers (time-filtered file selection, variable inventory, cubed-sphere coords) |
| `src/dyamond_fluxes/catalog.py` | Zarr store discovery/opening (local subsets; legacy consolidated store) |
| `src/dyamond_fluxes/grid.py` | LLC2160 grid utilities: area-weighted binning to lat-lon, vector rotation via `CS`/`SN`, lat/lon-box face subsetting |
| `src/dyamond_fluxes/fluxes.py` | Sign-convention handling, non-solar flux, GEOS $Q_{net}$ decomposition, area-weighted means |
| `src/dyamond_fluxes/plotting.py` | Global (binned) and regional (native-resolution) flux maps; cartopy optional |
| `notebooks/00_data_inventory.ipynb` | Survey `mit/` variables and `holding/` collections; record flux variable names and conventions |
| `notebooks/01_ocean_fluxes_global.ipynb` | Global snapshot maps of $Q_{net}$, $Q_{sw}$, $Q_{ns}$ |
| `notebooks/02_qnet_decomposition.ipynb` | GEOS component decomposition and closure vs. `oceQnet` on a 1° grid |
| `notebooks/03_regional_zoom.ipynb` | Gulf Stream and Kuroshio zooms at native ~2–4 km resolution |
| `notebooks/04_qsw_validation.ipynb` | July-2020 monthly-mean $Q_{sw}$ vs. CERES EBAF Ed4.2 surface net SW (bias map, zonal means, area-weighted stats) |
| `notebooks/05_qsw_diagnostics.ipynb` | Diagnose raw `oceQsw` content: probe-point hourly series (day/night fingerprint), snapshot statistics and map |

Run the notebooks in order: notebook 00 records the GEOS flux variable names and
collections that notebook 02's `FLUX_SOURCES` mapping must match.

## Physical conventions and caveats

- **Sign conventions differ between components.** Per `mit/readme.txt`, this dataset
  stores the ocean-side fluxes positive **upward** (`oceQnet`/`oceQsw`: ">0 decreases
  theta") — the opposite of the MITgcm diagnostics-package default. The raw MDS files
  carry no metadata, so `mds.py` attaches the readme's descriptions as attributes and
  `fluxes.to_positive_down` reads the direction from them (it refuses to guess when
  ambiguous). GEOS/MERRA-2 conventions: `SWGNT`/`LWGNT` positive down, `EFLUX`/`HFLUX`
  positive up — confirm against the `long_name`s printed by notebook 00.
- **Grids differ.** Cross-grid comparison uses area-weighted bin-averaging of cell-center
  values to a common regular grid — conservative in the mean and dependency-free, adequate
  at 0.25°–1° target resolution.
- **Sea ice.** Under ice, `oceQnet` includes ice–ocean exchange; closure statistics exclude
  latitudes poleward of 60°.

## Citations

- Data descriptor: Menemenlis, D., et al. (2026). *Scientific Data*,
  [doi:10.1038/s41597-026-07349-2](https://www.nature.com/articles/s41597-026-07349-2).
- Dataset hosting: [Poseidon Project](https://www.poseidon-ocean.net/products/datasets/93),
  Johns Hopkins University / SciServer.
- Model code archive: [Zenodo record 15021755](https://zenodo.org/records/15021755).
