"""Paulson-Simpson two-band penetration, mirrored against MITgcm's swfrac.F."""

import numpy as np
import pytest
import xarray as xr

from dyamond_fluxes.swabsorb import (
    JERLOV_COEFFS,
    backout_surface_sw,
    infer_reference_depth,
    interface_fractions,
    sw_fraction,
)


def test_surface_fraction_is_one():
    np.testing.assert_allclose(sw_fraction(0.0), 1.0)


def test_hand_computed_values_jerlov_ia():
    # R=0.62, a1=0.6, a2=20 (swfrac.F defaults). At 20 m the a1 band is extinct.
    np.testing.assert_allclose(
        sw_fraction(20.0), 0.62 * np.exp(-20 / 0.6) + 0.38 * np.exp(-1.0), rtol=1e-12
    )
    np.testing.assert_allclose(sw_fraction(20.0), 0.139794, atol=1e-5)
    np.testing.assert_allclose(sw_fraction(1.0), 0.478570, atol=1e-5)


def test_zero_below_200m():
    assert sw_fraction(200.1) == 0.0
    assert sw_fraction(200.0) > 0.0


def test_monotone_decreasing():
    z = np.linspace(0, 199, 500)
    assert np.all(np.diff(sw_fraction(z)) < 0)


@pytest.mark.parametrize("jw", sorted(JERLOV_COEFFS))
def test_infer_reference_depth_roundtrip(jw):
    for z in (0.5, 5.0, 23.0, 80.0):
        c = float(sw_fraction(z, jerlov_type=jw))
        assert abs(infer_reference_depth(c, jerlov_type=jw) - z) < 1e-2


def test_infer_reference_depth_rejects_bad_fraction():
    with pytest.raises(ValueError, match="fraction"):
        infer_reference_depth(1.5)


def test_interface_fractions_layout():
    drf = np.array([1.0, 2.0, 3.0])
    z, fr = interface_fractions(drf)
    np.testing.assert_allclose(z, [0.0, 1.0, 3.0, 6.0])
    np.testing.assert_allclose(fr, sw_fraction(z))
    # apply_forcing.F: per-layer absorption frac[k]-frac[k+1] sums to 1 - frac(bottom)
    np.testing.assert_allclose(np.sum(-np.diff(fr)), 1.0 - fr[-1])


def test_backout_surface_sw():
    qsw = xr.DataArray([12.0, 24.0], dims="x", name="oceQsw")
    sw = backout_surface_sw(qsw, 0.12)
    np.testing.assert_allclose(sw.values, [100.0, 200.0])
    assert sw.attrs["sign_convention"].startswith("positive downward")
