"""Readers for the GEOS atmosphere collections (NetCDF on the c1440 cubed sphere).

The staging area (``.../C1440-LLC2160_incoming/holding``) holds one directory
per collection with one nc4 file per output time, named
``DYAMOND_c1440_llc2160.<collection>.<YYYYMMDD_HHMM>z.nc4``. Cell-center
latitude/longitude for the native cubed-sphere grid live in the companion file
``geos_c1440_lats_lons_2D.nc``.

Surface heat flux components are expected in ``tavg_15mn_2d_flx_Mx`` (turbulent
fluxes, e.g. EFLUX/HFLUX) and/or ``geosgcm_surf`` (surface diagnostics incl.
radiation) — confirm names with :func:`peek_variables` in notebook 00.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .catalog import dyamond_root

FILE_RE = re.compile(
    r"DYAMOND_c1440_llc2160\.(?P<collection>.+)\.(?P<stamp>\d{8}_\d{4})z\.nc4$"
)
COORD_FILE = "geos_c1440_lats_lons_2D.nc"

__all__ = [
    "holding_dir",
    "list_geos_collections",
    "collection_files",
    "nearest_file",
    "open_geos",
    "peek_variables",
    "load_geos_coords",
]


def holding_dir(root: str | Path | None = None) -> Path:
    """The GEOS output directory (``<root>/holding``, or ``root`` itself if unnested)."""
    root = Path(root) if root is not None else dyamond_root()
    return root / "holding" if (root / "holding").is_dir() else root


def list_geos_collections(root: str | Path | None = None) -> list[str]:
    """Collection directories that contain at least one nc4 file."""
    base = holding_dir(root)
    out = []
    for d in sorted(base.iterdir()):
        try:
            if d.is_dir() and any(d.glob("*.nc4")):
                out.append(d.name)
        except OSError:
            continue
    return out


def _stamp_to_time(stamp: str) -> np.datetime64:
    return np.datetime64(datetime.strptime(stamp, "%Y%m%d_%H%M"))


def collection_files(
    collection: str,
    root: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[Path], np.ndarray]:
    """Sorted nc4 files for one collection with their filename timestamps.

    ``start``/``end`` (ISO strings) restrict the range without touching the
    files — essential when a collection holds tens of thousands of them.
    """
    files, times = [], []
    for f in sorted((holding_dir(root) / collection).glob("*.nc4")):
        m = FILE_RE.match(f.name)
        if m is None:
            continue
        t = _stamp_to_time(m.group("stamp"))
        if start is not None and t < np.datetime64(start):
            continue
        if end is not None and t > np.datetime64(end):
            continue
        files.append(f)
        times.append(t)
    return files, np.array(times, dtype="datetime64[s]")


def nearest_file(collection: str, when: str, root: str | Path | None = None) -> Path:
    """The single file whose filename timestamp is closest to ``when`` (ISO string)."""
    files, times = collection_files(collection, root=root)
    if not files:
        raise FileNotFoundError(f"No nc4 files in collection {collection!r}")
    return files[int(np.abs(times - np.datetime64(when)).argmin())]


def open_geos(
    collection: str,
    root: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    **open_kwargs,
) -> xr.Dataset:
    """Open a (time-restricted) GEOS collection lazily.

    A single file opens with ``open_dataset``; multiple files are concatenated
    along ``time`` with ``open_mfdataset``. Always bound the range with
    ``start``/``end`` for the 15-minute collections.
    """
    files, _ = collection_files(collection, root=root, start=start, end=end)
    if not files:
        raise FileNotFoundError(
            f"No nc4 files for {collection!r} in [{start}, {end}]"
        )
    if len(files) == 1:
        return xr.open_dataset(files[0], chunks="auto", **open_kwargs)
    return xr.open_mfdataset(
        files, combine="nested", concat_dim="time", parallel=True, **open_kwargs
    )


def peek_variables(collection: str, root: str | Path | None = None) -> dict[str, str]:
    """Map variable name -> long_name from the first file of a collection (metadata only)."""
    files, _ = collection_files(collection, root=root)
    if not files:
        return {}
    with xr.open_dataset(files[0]) as ds:
        return {v: str(ds[v].attrs.get("long_name", "")) for v in ds.data_vars}


def load_geos_coords(root: str | Path | None = None) -> xr.Dataset:
    """Cell-center lats/lons of the native c1440 cubed-sphere grid."""
    path = holding_dir(root) / COORD_FILE
    if not path.exists():
        raise FileNotFoundError(f"Coordinate file {path} not found")
    return xr.open_dataset(path)
