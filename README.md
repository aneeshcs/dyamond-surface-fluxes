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

The ~2 PB Zarr dataset is hosted by the [Poseidon Project](https://www.poseidon-ocean.net/)
and must be analyzed in place on [SciServer](https://www.sciserver.org/) — there is no bulk
download. Following the
[access instructions](https://www.poseidon-ocean.net/access-process-for-the-dyamond-dataset-on-sciserver/):

1. Create a SciServer account and log in.
2. In **Compute**, create a container with:
   - **Domain**: Kraken
   - **Compute Image**: Oceanography
   - **Data Volume**: Poseidon DYAMOND (ceph)
3. The data appear at `/home/idies/workspace/poseidon_ceph/DYAMOND` (Zarr v2).

The official access demo lives in
[`hainegroup/Poseidon-share`](https://github.com/hainegroup/Poseidon-share)
(`Kraken/DYAMOND_access_demonstration.ipynb`).

## Setup

On SciServer (inside the Oceanography container terminal):

```bash
git clone https://github.com/<user>/dyamond-surface-fluxes.git
cd dyamond-surface-fluxes
pip install --user -e ".[maps]"
```

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
| `src/dyamond_fluxes/catalog.py` | Zarr store discovery/opening (`list_stores`, `open_store`, `find_stores_with`) |
| `src/dyamond_fluxes/grid.py` | LLC2160 grid utilities: area-weighted binning to lat-lon, vector rotation via `CS`/`SN`, lat/lon-box face subsetting |
| `src/dyamond_fluxes/fluxes.py` | Sign-convention handling, non-solar flux, GEOS $Q_{net}$ decomposition, area-weighted means |
| `src/dyamond_fluxes/plotting.py` | Global (binned) and regional (native-resolution) flux maps; cartopy optional |
| `notebooks/00_data_inventory.ipynb` | Enumerate stores; locate flux variables; record sign conventions |
| `notebooks/01_ocean_fluxes_global.ipynb` | Global snapshot maps of $Q_{net}$, $Q_{sw}$, $Q_{ns}$ |
| `notebooks/02_qnet_decomposition.ipynb` | GEOS component decomposition and closure vs. `oceQnet` on a 1° grid |
| `notebooks/03_regional_zoom.ipynb` | Gulf Stream and Kuroshio zooms at native ~2–4 km resolution |

Run the notebooks in order: notebook 00 establishes which stores and sign conventions the
later notebooks use. If the GEOS flux collections are not mirrored on SciServer, notebook
02 stops with instructions; the GEOS collections are alternatively available from the
[NCCS Dataportal](https://gmao.gsfc.nasa.gov/global_mesoscale/dyamond_phaseII/data_access/).

## Physical conventions and caveats

- **Sign conventions differ between components.** MITgcm diagnostic conventions vary by
  configuration, so `fluxes.to_positive_down` infers direction from variable attributes
  and refuses to guess when metadata is ambiguous. GEOS/MERRA-2 conventions: `SWGNT`/`LWGNT`
  positive down, `EFLUX`/`HFLUX` positive up.
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
