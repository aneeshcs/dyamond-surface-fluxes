"""LLC (lat-lon-cap) grid utilities for the LLC2160 ocean output.

The MITgcm LLC2160 grid stores fields as 13 faces of 2160 x 2160 cells with
2-D cell-center coordinates ``XC``/``YC`` (degrees) and grid-rotation cosines/
sines ``CS``/``SN``. These helpers avoid any heavy regridding machinery: global
views use area-weighted binning onto a regular lat-lon grid, and regional views
subset native faces directly.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "bin_to_latlon",
    "rotate_uv_to_east_north",
    "faces_in_bbox",
    "subset_bbox",
]


def bin_to_latlon(
    da: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    area: xr.DataArray | None = None,
    dlon: float = 0.25,
    dlat: float = 0.25,
) -> xr.DataArray:
    """Area-weighted bin-average of an LLC field onto a regular lat-lon grid.

    This is a conservative-in-the-mean aggregation (each model cell contributes
    its full area to the bin containing its center), adequate for global maps
    and cross-grid comparisons at resolutions much coarser than the ~2-4 km
    native grid. NaNs (land) are excluded.

    Parameters
    ----------
    da : DataArray
        2-D field on the native grid (any face/j/i layout; only values are used).
    lon, lat : DataArray
        Cell-center longitude/latitude with the same shape as ``da``.
    area : DataArray, optional
        Cell areas (``rA``) for weighting; uniform weights if omitted.
    dlon, dlat : float
        Output grid spacing in degrees.
    """
    vals = np.ravel(np.asarray(da.values, dtype=np.float64))
    lo = np.ravel(np.asarray(lon.values, dtype=np.float64))
    la = np.ravel(np.asarray(lat.values, dtype=np.float64))
    w = (
        np.ravel(np.asarray(area.values, dtype=np.float64))
        if area is not None
        else np.ones_like(vals)
    )

    good = np.isfinite(vals) & np.isfinite(lo) & np.isfinite(la)
    vals, lo, la, w = vals[good], lo[good], la[good], w[good]
    lo = np.mod(lo + 180.0, 360.0) - 180.0  # wrap to [-180, 180)

    lon_edges = np.arange(-180.0, 180.0 + dlon, dlon)
    lat_edges = np.arange(-90.0, 90.0 + dlat, dlat)

    # Weighted mean per bin: sum(w * v) / sum(w), fully vectorized.
    wsum, _, _ = np.histogram2d(la, lo, bins=[lat_edges, lon_edges], weights=w)
    wvsum, _, _ = np.histogram2d(la, lo, bins=[lat_edges, lon_edges], weights=w * vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = wvsum / wsum
    mean[wsum == 0] = np.nan

    return xr.DataArray(
        mean,
        dims=("lat", "lon"),
        coords={
            "lat": 0.5 * (lat_edges[:-1] + lat_edges[1:]),
            "lon": 0.5 * (lon_edges[:-1] + lon_edges[1:]),
        },
        name=da.name,
        attrs=dict(da.attrs),
    )


def rotate_uv_to_east_north(
    u: xr.DataArray, v: xr.DataArray, cs: xr.DataArray, sn: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate grid-relative vector components to geographic east/north.

    MITgcm stores ``CS``/``SN`` as the cosine/sine of the local grid angle, so
    (u_E, v_N) = (u*CS - v*SN, u*SN + v*CS). Components must already be located
    at cell centers (average neighboring u/v points first if needed).
    """
    u_east = u * cs - v * sn
    v_north = u * sn + v * cs
    u_east.attrs.update(u.attrs, direction="eastward")
    v_north.attrs.update(v.attrs, direction="northward")
    return u_east, v_north


def faces_in_bbox(
    lon: xr.DataArray,
    lat: xr.DataArray,
    bbox: tuple[float, float, float, float],
    face_dim: str = "face",
) -> list[int]:
    """Return indices of LLC faces containing any point inside ``bbox``.

    ``bbox`` is (lon_min, lon_max, lat_min, lat_max) with longitudes in
    [-180, 180]. Boxes crossing the dateline (lon_min > lon_max) are supported.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    lo = np.mod(np.asarray(lon.values) + 180.0, 360.0) - 180.0
    la = np.asarray(lat.values)
    if lon_min <= lon_max:
        in_lon = (lo >= lon_min) & (lo <= lon_max)
    else:  # dateline crossing
        in_lon = (lo >= lon_min) | (lo <= lon_max)
    inside = in_lon & (la >= lat_min) & (la <= lat_max)

    axis = lon.dims.index(face_dim)
    other_axes = tuple(k for k in range(inside.ndim) if k != axis)
    return list(np.nonzero(inside.any(axis=other_axes))[0])


def subset_bbox(
    da: xr.DataArray,
    lon: xr.DataArray,
    lat: xr.DataArray,
    bbox: tuple[float, float, float, float],
    face_dim: str = "face",
) -> list[tuple[int, xr.DataArray, xr.DataArray, xr.DataArray]]:
    """Trim a field to a lat/lon box, face by face, at native resolution.

    Returns a list of ``(face_index, field, lon, lat)`` tuples, one per face
    intersecting the box, each cropped to the minimal index rectangle covering
    the box (values outside the box are masked to NaN). Suitable for direct
    ``pcolormesh`` plotting of regions such as the Gulf Stream or Kuroshio.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    out = []
    for f in faces_in_bbox(lon, lat, bbox, face_dim=face_dim):
        lo_f = lon.isel({face_dim: f})
        la_f = lat.isel({face_dim: f})
        da_f = da.isel({face_dim: f})

        lo_w = np.mod(np.asarray(lo_f.values) + 180.0, 360.0) - 180.0
        la_v = np.asarray(la_f.values)
        if lon_min <= lon_max:
            in_lon = (lo_w >= lon_min) & (lo_w <= lon_max)
        else:
            in_lon = (lo_w >= lon_min) | (lo_w <= lon_max)
        mask = in_lon & (la_v >= lat_min) & (la_v <= lat_max)

        jj, ii = np.nonzero(mask)
        jsl = slice(jj.min(), jj.max() + 1)
        isl = slice(ii.min(), ii.max() + 1)
        jdim, idim = lo_f.dims

        cropped = da_f.isel({jdim: jsl, idim: isl}).where(
            xr.DataArray(mask[jsl, isl], dims=(jdim, idim))
        )
        crop = {jdim: jsl, idim: isl}
        out.append((f, cropped, lo_f.isel(crop), la_f.isel(crop)))
    return out
