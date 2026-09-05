"""Surface flux computations for the coupled GEOS-MITgcm DYAMOND run.

Sign conventions
----------------
All quantities returned by this module use **positive downward** (into the
ocean; a positive net heat flux warms the ocean), in W m^-2.

- MITgcm side: in this dataset ``oceQnet`` (net surface heat flux incl.
  shortwave) and ``oceQsw`` (net shortwave) are stored positive **upward**
  (">0 decreases theta", per ``mit/readme.txt``) — the opposite of the MITgcm
  diagnostics-package default. :func:`to_positive_down` inspects variable
  attributes (attached by ``mds.py`` from the readme) rather than hard-coding
  a sign.
- GEOS/MERRA-2 conventions: ``EFLUX`` (latent) and ``HFLUX`` (sensible) are
  positive **upward** (ocean -> atmosphere); ``SWGNT`` and ``LWGNT`` (net
  surface shortwave/longwave radiation) are positive **downward**.
"""

from __future__ import annotations

import xarray as xr

__all__ = [
    "to_positive_down",
    "nonsolar_flux",
    "qnet_from_components",
    "area_weighted_mean",
]

_UP_HINTS = ("upward", "positive up", "+ up", "up=+", "+=up", ">0 decreases theta")
_DOWN_HINTS = (
    "downward",
    "positive down",
    "+ down",
    "down=+",
    "+=down",  # MITgcm diagnostics-package notation
    "into the ocean",
    ">0 increases theta",  # keep 'theta' — 'oceFWflx: >0 increases salinity' is upward
)


def to_positive_down(da: xr.DataArray, assume_upward: bool | None = None) -> xr.DataArray:
    """Return ``da`` with a positive-downward sign convention.

    The direction is inferred from the ``long_name``/``standard_name``/
    ``comment`` attributes; pass ``assume_upward`` explicitly if the metadata
    is ambiguous (an informative error is raised in that case).
    """
    if assume_upward is None:
        text = " ".join(
            str(da.attrs.get(key, "")) for key in ("long_name", "standard_name", "comment", "units")
        ).lower()
        if any(h in text for h in _DOWN_HINTS):
            assume_upward = False
        elif any(h in text for h in _UP_HINTS):
            assume_upward = True
        else:
            raise ValueError(
                f"Cannot infer sign convention for {da.name!r} from attrs {da.attrs}; "
                "pass assume_upward=True/False explicitly."
            )
    out = -da if assume_upward else da
    out.attrs = dict(da.attrs)
    out.attrs["sign_convention"] = "positive downward (into ocean)"
    return out


def nonsolar_flux(qnet_down: xr.DataArray, qsw_down: xr.DataArray) -> xr.DataArray:
    """Non-solar surface heat flux Q_ns = Q_net - Q_sw (latent + sensible + net longwave).

    Both inputs must already be positive-down (see :func:`to_positive_down`).
    Lazy/dask-friendly: no data is loaded here.
    """
    q_ns = qnet_down - qsw_down
    q_ns.name = "oceQns"
    q_ns.attrs = {
        "long_name": "non-solar surface heat flux (latent + sensible + net longwave)",
        "units": "W m-2",
        "sign_convention": "positive downward (into ocean)",
    }
    return q_ns


def qnet_from_components(
    swgnt: xr.DataArray,
    lwgnt: xr.DataArray,
    eflux: xr.DataArray,
    hflux: xr.DataArray,
) -> xr.Dataset:
    """Reconstruct net downward surface heat flux from GEOS atmosphere diagnostics.

    Q_net(down) = SWGNT + LWGNT - EFLUX - HFLUX, following the GEOS/MERRA-2
    convention (radiative terms positive down, turbulent terms positive up).
    Returns a Dataset holding the four positive-down components and their sum,
    for closure checks against the ocean-side ``oceQnet``.
    """
    ds = xr.Dataset(
        {
            "shortwave": swgnt,
            "longwave": lwgnt,
            "latent": -eflux,
            "sensible": -hflux,
        }
    )
    ds["qnet"] = ds["shortwave"] + ds["longwave"] + ds["latent"] + ds["sensible"]
    for name, da in ds.items():
        da.attrs.update(units="W m-2", sign_convention="positive downward (into ocean)")
        da.attrs.setdefault("long_name", f"{name} surface heat flux")
    return ds


def area_weighted_mean(
    da: xr.DataArray, area: xr.DataArray, mask: xr.DataArray | None = None
) -> xr.DataArray:
    """Area-weighted spatial mean, skipping NaNs (land). Reduces over area's dims."""
    if mask is not None:
        da = da.where(mask)
    # xarray's weighted mean skips NaNs in `da` but requires finite weights.
    return da.weighted(area.fillna(0.0)).mean(dim=list(area.dims))
