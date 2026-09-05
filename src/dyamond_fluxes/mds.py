"""Lazy reader for raw MITgcm MDS binary output (LLC2160 ocean side).

The DYAMOND staging area (``.../C1440-LLC2160_incoming/mit``) holds one
directory per variable containing big-endian binary files named
``<var>.<iteration>.data`` — one file per output step, with no ``.meta``
companions. LLC "compact" layout: the 13 faces are stacked along the first
axis, i.e. a 2-D field is stored as ``(13*N, N)`` and reshaped here to
``(face, j, i)``; 3-D fields carry a leading vertical axis ``(k, face, j, i)``.
The float width is detected from the file size.

Time base: the coupled run starts 2020-01-19 21:00 UTC with a 45 s ocean
timestep, so ``time = T0 + iteration * 45 s`` (hourly output = 80 iterations).
Confirmed by ``mit/readme.txt``: timesteps 0-829200 span
2020-01-19 21:00 to 2021-03-26 18:00 UTC.
"""

from __future__ import annotations

from pathlib import Path

import dask
import dask.array as dsa
import numpy as np
import xarray as xr

from .catalog import dyamond_root

LLC_NFACES = 13
LLC_N = 2160  # points per face side for LLC2160
OCEAN_T0 = np.datetime64("2020-01-19T21:00:00")
OCEAN_DT_SECONDS = 45

# Variable metadata transcribed from mit/readme.txt in the staging area
# (verified 2026-09). Raw .data files carry no attributes, so these supply the
# units and — critically — the sign convention consumed by
# fluxes.to_positive_down. NOTE: unlike the MITgcm diagnostics-package default,
# this dataset stores the surface fluxes positive **upward** (>0 cools/salts
# the ocean).
MITGCM_DIAG_ATTRS: dict[str, dict[str, str]] = {
    "oceQnet": {
        "long_name": "net upward surface heat flux (including shortwave), >0 decreases theta",
        "units": "W m-2",
    },
    "oceQsw": {
        "long_name": "net upward shortwave radiation, >0 decreases theta",
        "units": "W m-2",
        "comment": (
            "WARNING: despite the readme description, this stream carries only "
            "~12% of the true surface net SW (July 2020, vs CERES and vs the "
            "diurnal range of oceQnet) with a smooth zenith-only signature - "
            "consistent with a sub-surface penetrating-SW component, not the "
            "surface flux. Use GEOS SWGNT for surface net shortwave; oceQnet "
            "itself contains the full shortwave and is healthy. See notebook 05."
        ),
    },
    "oceFWflx": {
        "long_name": "net upward freshwater flux, >0 increases salinity",
        "units": "kg m-2 s-1",
    },
    "oceSflux": {
        "long_name": "net upward salt flux, >0 decreases salinity",
        "units": "g m-2 s-1",
    },
    "oceTAUX": {
        "long_name": "zonal (grid-relative) surface wind stress, >0 increases uVel",
        "units": "N m-2",
    },
    "oceTAUY": {
        "long_name": "meridional (grid-relative) surface wind stress, >0 increases vVel",
        "units": "N m-2",
    },
    "Eta": {"long_name": "sea surface height anomaly", "units": "m"},
    "KPPhbl": {"long_name": "KPP mixing layer depth", "units": "m"},
}

# Grid files in mit/grid -> the names the rest of the package expects.
GRID_FILE_MAP = {
    "XC": "XC",
    "YC": "YC",
    "RAC": "rA",
    "Depth": "Depth",
    "AngleCS": "CS",
    "AngleSN": "SN",
}

__all__ = [
    "mit_dir",
    "list_mit_variables",
    "iters_to_time",
    "open_mds_variable",
    "open_grid",
    "open_ocean_dataset",
    "MITGCM_DIAG_ATTRS",
]


def mit_dir(root: str | Path | None = None) -> Path:
    """The MITgcm output directory (``<root>/mit``, or ``root`` itself if unnested)."""
    root = Path(root) if root is not None else dyamond_root()
    return root / "mit" if (root / "mit").is_dir() else root


def list_mit_variables(root: str | Path | None = None) -> list[str]:
    """Variable directories that contain at least one ``.data`` file."""
    base = mit_dir(root)
    out = []
    for d in sorted(base.iterdir()):
        try:
            if d.is_dir() and any(d.glob(f"{d.name}.*.data")):
                out.append(d.name)
        except OSError:
            continue
    return out


def iters_to_time(iterations: np.ndarray) -> np.ndarray:
    """Model iteration numbers -> datetime64, using the 45 s coupled timestep."""
    return OCEAN_T0 + np.asarray(iterations, dtype="int64") * np.timedelta64(
        OCEAN_DT_SECONDS, "s"
    )


def _detect_layout(nbytes: int, n: int) -> tuple[tuple[int, ...], str]:
    """Infer (shape, dtype) of one compact-LLC file from its size."""
    plane = LLC_NFACES * n * n
    for width, dtype in ((4, ">f4"), (8, ">f8")):
        if nbytes == plane * width:
            return (LLC_NFACES, n, n), dtype
        if nbytes % (plane * width) == 0:
            nz = nbytes // (plane * width)
            return (nz, LLC_NFACES, n, n), dtype
    raise ValueError(
        f"File size {nbytes} is not a multiple of a {n}x{n} LLC face plane; "
        "not compact-LLC MDS output?"
    )


def _read_llc(path: Path, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Read one compact file: faces are stacked along the leading row axis on disk."""
    flat = np.fromfile(path, dtype=dtype)
    return flat.reshape(shape)


def open_mds_variable(
    name: str, root: str | Path | None = None, n: int = LLC_N
) -> xr.DataArray:
    """Open every output step of one variable as a lazy dask-backed DataArray.

    Dimensions are ``(time, face, j, i)`` for 2-D fields and
    ``(time, k, face, j, i)`` for 3-D fields; each file is one dask chunk
    (~243 MB for a float32 2-D LLC2160 field), so slicing a single time reads
    exactly one file.
    """
    var_dir = mit_dir(root) / name
    files = sorted(var_dir.glob(f"{name}.*.data"))
    if not files:
        raise FileNotFoundError(f"No {name}.*.data files found in {var_dir}")

    iterations = np.array([int(f.name.split(".")[1]) for f in files])
    shape, dtype = _detect_layout(files[0].stat().st_size, n)

    read = dask.delayed(_read_llc, pure=True)
    arrays = [
        dsa.from_delayed(read(f, shape, dtype), shape=shape, dtype=np.dtype(dtype))
        for f in files
    ]
    data = dsa.stack(arrays)

    space_dims = ("face", "j", "i") if len(shape) == 3 else ("k", "face", "j", "i")
    da = xr.DataArray(
        data,
        dims=("time", *space_dims),
        coords={"time": iters_to_time(iterations), "iteration": ("time", iterations)},
        name=name,
        attrs=dict(MITGCM_DIAG_ATTRS.get(name, {})),
    )
    da.time.attrs["comment"] = (
        f"time = {OCEAN_T0} + iteration * {OCEAN_DT_SECONDS} s; confirmed by mit/readme.txt"
    )
    return da


def open_grid(root: str | Path | None = None, n: int = LLC_N) -> xr.Dataset:
    """Load the horizontal grid descriptors (XC, YC, rA, Depth, CS, SN) eagerly.

    ~243 MB per field at LLC2160 — loaded once and reused across notebooks.
    """
    grid_dir = mit_dir(root) / "grid"
    out = {}
    for fname, target in GRID_FILE_MAP.items():
        path = grid_dir / f"{fname}.data"
        if not path.exists():
            continue
        shape, dtype = _detect_layout(path.stat().st_size, n)
        out[target] = xr.DataArray(
            _read_llc(path, shape, dtype).astype("f4"),
            dims=("face", "j", "i") if len(shape) == 3 else ("k", "face", "j", "i"),
        )
    if not out:
        raise FileNotFoundError(f"No known grid files found in {grid_dir}")
    return xr.Dataset(out)


def open_ocean_dataset(
    variables: list[str], root: str | Path | None = None, n: int = LLC_N
) -> xr.Dataset:
    """Open several MDS variables plus the grid as one Dataset.

    Grid fields (XC, YC, rA, ...) are attached as coordinates, matching the
    layout the flux/plotting utilities expect.
    """
    ds = xr.Dataset({v: open_mds_variable(v, root=root, n=n) for v in variables})
    grid = open_grid(root=root, n=n)
    return ds.assign_coords({k: v for k, v in grid.items()})
