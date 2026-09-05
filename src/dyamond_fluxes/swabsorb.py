"""Shortwave penetration in the upper ocean (Paulson & Simpson 1977 two-band).

Mirrors the MITgcm treatment referenced by the DYAMOND data descriptor:

- ``model/src/swfrac.F``: fraction of surface net SW remaining at depth z is
  ``R exp(-z/a1) + (1-R) exp(-z/a2)`` with the Jerlov water-type coefficients
  below (type IA hard-coded as the default), zeroed below 200 m. The fraction
  depends on depth only — not on solar zenith angle or clouds.
- ``model/src/ini_forcing.F``: fractions are evaluated once at the vertical
  level *interfaces* (``SWFracK(k) = rF(k) - rF(1)``).
- ``model/src/apply_forcing.F``: layer k absorbs ``Qsw * (frac(k) - frac(k+1))``.

Because the a1 band is extinct below ~3 m, the flux at any fixed subsurface
interface is a single constant times the surface flux. The DYAMOND ``oceQsw``
stream (empirically ~12% of the surface net SW; notebook 05) is such a
penetrating-SW flux, so dividing by that constant recovers the surface net
shortwave exactly (notebook 06).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "JERLOV_COEFFS",
    "MAX_PENETRATION_M",
    "sw_fraction",
    "infer_reference_depth",
    "interface_fractions",
    "backout_surface_sw",
]

# (R, a1 [m], a2 [m]) per Jerlov water type, exactly as in swfrac.F.
JERLOV_COEFFS: dict[int, tuple[float, float, float]] = {
    1: (0.58, 0.35, 23.0),  # Jerlov I
    2: (0.62, 0.60, 20.0),  # Jerlov IA — swfrac.F hard-coded default
    3: (0.67, 1.00, 17.0),  # Jerlov IB
    4: (0.77, 1.50, 14.0),  # Jerlov II
    5: (0.78, 1.40, 7.9),   # Jerlov III
}
MAX_PENETRATION_M = 200.0  # swfrac.F zeroes the fraction below 200 m


def sw_fraction(depth_m, jerlov_type: int = 2) -> np.ndarray:
    """Fraction of surface net SW remaining at ``depth_m`` (positive down, m)."""
    r, a1, a2 = JERLOV_COEFFS[jerlov_type]
    z = np.asarray(depth_m, dtype=float)
    frac = r * np.exp(-z / a1) + (1.0 - r) * np.exp(-z / a2)
    return np.where(z > MAX_PENETRATION_M, 0.0, frac)


def infer_reference_depth(fraction: float, jerlov_type: int = 2) -> float:
    """Depth (m) at which :func:`sw_fraction` equals ``fraction``.

    The two-band profile is strictly decreasing on [0, 200 m], so the inverse
    is well defined; a dense tabulation + interpolation is exact to ~1 mm.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    zs = np.linspace(0.0, MAX_PENETRATION_M, 200_001)
    fr = sw_fraction(zs, jerlov_type)
    return float(np.interp(fraction, fr[::-1], zs[::-1]))


def interface_fractions(
    drf: np.ndarray, jerlov_type: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """(interface depths, SW fractions) for a column of layer thicknesses ``drf``.

    Interface k (0-based; k=0 is the surface) sits at ``sum(drf[:k])`` —
    mirroring ini_forcing.F's ``SWFracK = rF(k) - rF(1)``. The absorption
    within layer k is ``frac[k] - frac[k+1]`` (apply_forcing.F).
    """
    z = np.concatenate([[0.0], np.cumsum(np.asarray(drf, dtype=float))])
    return z, sw_fraction(z, jerlov_type)


def backout_surface_sw(oceqsw_down: xr.DataArray, fraction: float) -> xr.DataArray:
    """Surface net downward SW recovered from the penetrating-SW stream.

    ``oceqsw_down`` must already be positive-down; ``fraction`` is the constant
    ratio (regressed against an independent surface-SW estimate, or taken from
    :func:`interface_fractions` at the identified interface).
    """
    out = oceqsw_down / float(fraction)
    out.name = "sw_surface"
    out.attrs = {
        "long_name": "surface net shortwave, backed out from the penetrating oceQsw stream",
        "units": "W m-2",
        "sign_convention": "positive downward (into ocean)",
        "method": f"oceQsw / {float(fraction):.6f} (Paulson-Simpson two-band constant)",
    }
    return out
