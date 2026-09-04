"""Surface flux analysis of the coupled GEOS-MITgcm DYAMOND (c1440-LLC2160) simulation."""

from .catalog import dyamond_root, find_stores_with, list_stores, open_store
from .fluxes import area_weighted_mean, nonsolar_flux, qnet_from_components, to_positive_down
from .geos import (
    collection_files,
    list_geos_collections,
    load_geos_coords,
    nearest_file,
    open_geos,
    peek_variables,
)
from .grid import bin_to_latlon, faces_in_bbox, rotate_uv_to_east_north, subset_bbox
from .mds import list_mit_variables, open_grid, open_mds_variable, open_ocean_dataset

__version__ = "0.2.0"

__all__ = [
    # catalog (zarr stores / local subsets)
    "dyamond_root",
    "list_stores",
    "open_store",
    "find_stores_with",
    # ocean: raw MITgcm MDS binary
    "list_mit_variables",
    "open_mds_variable",
    "open_grid",
    "open_ocean_dataset",
    # atmosphere: GEOS NetCDF collections
    "list_geos_collections",
    "collection_files",
    "nearest_file",
    "open_geos",
    "peek_variables",
    "load_geos_coords",
    # flux physics
    "to_positive_down",
    "nonsolar_flux",
    "qnet_from_components",
    "area_weighted_mean",
    # LLC grid utilities
    "bin_to_latlon",
    "rotate_uv_to_east_north",
    "faces_in_bbox",
    "subset_bbox",
]
