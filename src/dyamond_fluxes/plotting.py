"""Plotting helpers: global binned maps and native-resolution regional zooms.

Cartopy is optional (``pip install -e ".[maps]"``); without it, functions fall
back to plain lat/lon axes so the package remains usable in minimal
environments.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .grid import bin_to_latlon, subset_bbox

__all__ = ["FLUX_CMAPS", "plot_global", "plot_region"]

# Diverging colormap for signed fluxes (positive down = ocean warming),
# sequential for positive-definite fields such as shortwave. cmocean is
# optional so the package imports in minimal environments.
try:
    import cmocean

    FLUX_CMAPS: dict[str, Any] = {
        "diverging": cmocean.cm.balance,
        "sequential": cmocean.cm.thermal,
    }
except ImportError:
    FLUX_CMAPS = {"diverging": "RdBu_r", "sequential": "inferno"}


def _get_geoaxes(figsize: tuple[float, float], projection: str | None):
    """Return (fig, ax, transform_kwargs); cartopy if available, else plain axes."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        proj = ccrs.Robinson() if projection == "robinson" else ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
        ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        ax.coastlines(linewidth=0.4, zorder=3)
        return fig, ax, {"transform": ccrs.PlateCarree()}
    except ImportError:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        return fig, ax, {}


def _symmetric_limits(values: np.ndarray, robust_pct: float = 99.0) -> tuple[float, float]:
    """Symmetric color limits from a robust percentile of |values| (NaN-safe)."""
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(np.abs(finite), robust_pct)) if finite.size else 1.0
    return -vmax, vmax


def plot_global(
    da: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    area: xr.DataArray | None = None,
    dlon: float = 0.25,
    dlat: float = 0.25,
    diverging: bool = True,
    title: str | None = None,
    units: str = "W m$^{-2}$",
    figsize: tuple[float, float] = (11, 5.5),
    **pcolor_kw: Any,
):
    """Global map of an LLC field via area-weighted binning onto a lat-lon grid.

    Pass a single time snapshot (2-D field); binning loads that snapshot only.
    Returns ``(fig, ax, binned)`` where ``binned`` is the regridded DataArray.
    """
    binned = bin_to_latlon(da, lon, lat, area=area, dlon=dlon, dlat=dlat)

    fig, ax, tkw = _get_geoaxes(figsize, projection="robinson")
    if diverging:
        vmin, vmax = _symmetric_limits(binned.values)
        pcolor_kw.setdefault("cmap", FLUX_CMAPS["diverging"])
        pcolor_kw.setdefault("vmin", vmin)
        pcolor_kw.setdefault("vmax", vmax)
    else:
        pcolor_kw.setdefault("cmap", FLUX_CMAPS["sequential"])

    mesh = ax.pcolormesh(binned["lon"], binned["lat"], binned, **tkw, **pcolor_kw)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.7, pad=0.05)
    label = da.attrs.get("long_name", da.name or "")
    cb.set_label(f"{label} ({units})")
    if title:
        ax.set_title(title)
    return fig, ax, binned


def plot_region(
    da: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    bbox: tuple[float, float, float, float],
    diverging: bool = True,
    title: str | None = None,
    units: str = "W m$^{-2}$",
    figsize: tuple[float, float] = (9, 6),
    **pcolor_kw: Any,
):
    """Native ~2-4 km resolution regional map, plotted face by face.

    ``bbox`` is (lon_min, lon_max, lat_min, lat_max). Returns ``(fig, ax)``.
    """
    pieces = subset_bbox(da, lon, lat, bbox)
    if not pieces:
        raise ValueError(f"No LLC faces intersect bbox {bbox}.")

    if diverging:
        vals = np.concatenate([np.ravel(np.asarray(p[1].values)) for p in pieces])
        vmin, vmax = _symmetric_limits(vals)
        pcolor_kw.setdefault("cmap", FLUX_CMAPS["diverging"])
        pcolor_kw.setdefault("vmin", vmin)
        pcolor_kw.setdefault("vmax", vmax)
    else:
        pcolor_kw.setdefault("cmap", FLUX_CMAPS["sequential"])

    fig, ax, tkw = _get_geoaxes(figsize, projection="platecarree")
    mesh = None
    for _face, field, lo, la in pieces:
        mesh = ax.pcolormesh(lo.values, la.values, field.values, **tkw, **pcolor_kw)

    if tkw:  # cartopy axes: restrict extent to the box
        ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=tkw["transform"])
    else:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])

    cb = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.03)
    label = da.attrs.get("long_name", da.name or "")
    cb.set_label(f"{label} ({units})")
    if title:
        ax.set_title(title)
    return fig, ax
