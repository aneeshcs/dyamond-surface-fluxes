"""Discovery and opening of DYAMOND Zarr stores on SciServer.

The coupled GEOS-MITgcm (c1440-LLC2160) DYAMOND output lives on the SciServer
Kraken domain as Zarr v2 stores. The mount point has varied: the access
instructions cite ``poseidon_ceph/DYAMOND``, but containers may instead mount a
dedicated read-only filesystem at ``poseidon-DYAMOND`` — both are tried. Set the
environment variable ``DYAMOND_ROOT`` to point elsewhere (e.g., a small local
test subset).
"""

from __future__ import annotations

import os
from pathlib import Path

import xarray as xr

DEFAULT_ROOTS = (
    "/home/idies/workspace/poseidon-DYAMOND",
    "/home/idies/workspace/poseidon_ceph/DYAMOND",
)

__all__ = [
    "DEFAULT_ROOTS",
    "dyamond_root",
    "list_stores",
    "open_store",
    "find_stores_with",
]


def _has_content(path: Path) -> bool:
    """True if ``path`` is a directory with at least one entry (an unmounted volume
    leaves an empty stub directory behind, which must not count as found)."""
    return path.is_dir() and any(path.iterdir())


def dyamond_root() -> Path:
    """Return the DYAMOND data root.

    ``DYAMOND_ROOT`` (env var) takes precedence; otherwise the known SciServer
    mount points are tried in order, skipping empty stub directories.
    """
    env = os.environ.get("DYAMOND_ROOT")
    if env is not None:
        root = Path(env)
        if not root.exists():
            raise FileNotFoundError(f"DYAMOND_ROOT is set to {root}, which does not exist.")
        return root
    for candidate in DEFAULT_ROOTS:
        root = Path(candidate)
        if _has_content(root):
            return root
    raise FileNotFoundError(
        f"No DYAMOND data found at any of {DEFAULT_ROOTS}. This analysis must run on "
        "SciServer (Kraken domain, Oceanography image, with the Poseidon DYAMOND data "
        "volume attached at container creation), or set DYAMOND_ROOT to a local subset. "
        "See README for access instructions."
    )


def _is_zarr_store(path: Path) -> bool:
    """Detect a Zarr store root: v2 markers (.zgroup/.zmetadata/.zattrs) or v3 (zarr.json)."""
    return any(
        (path / marker).exists() for marker in (".zgroup", ".zmetadata", ".zattrs", "zarr.json")
    )


def list_stores(root: str | Path | None = None, max_depth: int = 3) -> list[Path]:
    """Recursively locate Zarr stores under ``root`` without descending into them.

    Parameters
    ----------
    root : path, optional
        Directory to search; defaults to :func:`dyamond_root`.
    max_depth : int
        Maximum directory depth to search below ``root``.
    """
    root = Path(root) if root is not None else dyamond_root()
    stores: list[Path] = []

    def _walk(d: Path, depth: int) -> None:
        if _is_zarr_store(d):
            stores.append(d)
            return  # do not descend into a store (arrays inside also carry .zattrs)
        if depth >= max_depth:
            return
        try:
            subdirs = sorted(p for p in d.iterdir() if p.is_dir())
        except PermissionError:
            return
        for sub in subdirs:
            _walk(sub, depth + 1)

    _walk(root, 0)
    return stores


def open_store(store: str | Path, chunks: dict | str | None = "auto") -> xr.Dataset:
    """Open one Zarr store lazily with xarray/dask.

    ``store`` may be an absolute path or a name relative to :func:`dyamond_root`.
    """
    path = Path(store)
    if not path.is_absolute():
        path = dyamond_root() / path
    try:
        return xr.open_zarr(path, consolidated=True, chunks=chunks)
    except (KeyError, FileNotFoundError, ValueError):
        # Store without consolidated metadata (zarr v2 raises KeyError, v3 ValueError);
        # slower open but identical result.
        return xr.open_zarr(path, consolidated=False, chunks=chunks)


def find_stores_with(
    variables: list[str], root: str | Path | None = None
) -> dict[Path, list[str]]:
    """Map each discovered store to the requested variables it contains.

    Opens only metadata (lazy), so this is cheap even for petabyte stores. Useful
    for locating, e.g., ocean surface fluxes (``oceQnet``) vs. GEOS atmosphere
    collections (``EFLUX``, ``HFLUX``, ``SWGNT``, ``LWGNT``).
    """
    hits: dict[Path, list[str]] = {}
    for store in list_stores(root):
        try:
            ds = open_store(store, chunks=None)
        except Exception:
            continue
        found = [v for v in variables if v in ds]
        ds.close()
        if found:
            hits[store] = found
    return hits
