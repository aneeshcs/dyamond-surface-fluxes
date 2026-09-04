"""Surface flux analysis of the coupled GEOS-MITgcm DYAMOND (c1440-LLC2160) simulation."""

from .catalog import dyamond_root, find_stores_with, list_stores, open_store
from .fluxes import area_weighted_mean, nonsolar_flux, qnet_from_components, to_positive_down
from .grid import bin_to_latlon, faces_in_bbox, rotate_uv_to_east_north, subset_bbox

__version__ = "0.1.0"

__all__ = [
    "dyamond_root",
    "list_stores",
    "open_store",
    "find_stores_with",
    "to_positive_down",
    "nonsolar_flux",
    "qnet_from_components",
    "area_weighted_mean",
    "bin_to_latlon",
    "rotate_uv_to_east_north",
    "faces_in_bbox",
    "subset_bbox",
]
