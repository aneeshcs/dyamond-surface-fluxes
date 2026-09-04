"""Flux arithmetic and sign-convention handling (pure array logic, no data access)."""

import numpy as np
import pytest
import xarray as xr

from dyamond_fluxes.fluxes import (
    area_weighted_mean,
    nonsolar_flux,
    qnet_from_components,
    to_positive_down,
)


def _da(values, **attrs) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=float), dims="x", attrs=attrs)


class TestToPositiveDown:
    def test_downward_field_unchanged(self):
        da = _da([5.0], long_name="net surface heat flux, positive down")
        np.testing.assert_allclose(to_positive_down(da).values, [5.0])

    def test_upward_field_flipped(self):
        da = _da([5.0], long_name="net upward surface heat flux")
        np.testing.assert_allclose(to_positive_down(da).values, [-5.0])

    def test_explicit_override_beats_missing_metadata(self):
        da = _da([2.0])
        np.testing.assert_allclose(to_positive_down(da, assume_upward=True).values, [-2.0])
        np.testing.assert_allclose(to_positive_down(da, assume_upward=False).values, [2.0])

    def test_ambiguous_metadata_raises(self):
        with pytest.raises(ValueError, match="sign convention"):
            to_positive_down(_da([1.0], long_name="mystery flux"))

    def test_annotates_convention(self):
        da = _da([1.0], long_name="net upward flux")
        assert "downward" in to_positive_down(da).attrs["sign_convention"]


class TestNonsolarFlux:
    def test_arithmetic(self):
        qnet = _da([100.0, -50.0])
        qsw = _da([180.0, 0.0])
        q_ns = nonsolar_flux(qnet, qsw)
        np.testing.assert_allclose(q_ns.values, [-80.0, -50.0])
        assert q_ns.attrs["units"] == "W m-2"

    def test_lazy_with_dask(self):
        pytest.importorskip("dask")
        qnet = _da(np.arange(10.0)).chunk({"x": 5})
        qsw = _da(np.ones(10)).chunk({"x": 5})
        q_ns = nonsolar_flux(qnet, qsw)
        assert q_ns.chunks is not None  # still lazy
        np.testing.assert_allclose(q_ns.values, np.arange(10.0) - 1.0)


class TestQnetFromComponents:
    def test_geos_sign_convention(self):
        # SWGNT/LWGNT positive down; EFLUX/HFLUX positive up.
        ds = qnet_from_components(
            swgnt=_da([200.0]), lwgnt=_da([-60.0]), eflux=_da([120.0]), hflux=_da([15.0])
        )
        np.testing.assert_allclose(ds["latent"].values, [-120.0])
        np.testing.assert_allclose(ds["sensible"].values, [-15.0])
        np.testing.assert_allclose(ds["qnet"].values, [200.0 - 60.0 - 120.0 - 15.0])

    def test_components_sum_to_qnet(self):
        rng = np.random.default_rng(1)
        parts = [_da(rng.normal(size=8)) for _ in range(4)]
        ds = qnet_from_components(*parts)
        total = ds["shortwave"] + ds["longwave"] + ds["latent"] + ds["sensible"]
        np.testing.assert_allclose(ds["qnet"].values, total.values, rtol=1e-12)


class TestAreaWeightedMean:
    def test_weighting_and_land_nan(self):
        da = xr.DataArray([[1.0, 3.0], [np.nan, 5.0]], dims=("j", "i"))
        area = xr.DataArray([[1.0, 1.0], [10.0, 2.0]], dims=("j", "i"))
        # NaN cell excluded: mean = (1*1 + 3*1 + 5*2) / (1 + 1 + 2) = 3.5
        result = area_weighted_mean(da, area)
        np.testing.assert_allclose(result, 3.5)

    def test_mask_application(self):
        da = xr.DataArray([[1.0, 100.0]], dims=("j", "i"))
        area = xr.DataArray([[1.0, 1.0]], dims=("j", "i"))
        mask = xr.DataArray([[True, False]], dims=("j", "i"))
        np.testing.assert_allclose(area_weighted_mean(da, area, mask=mask), 1.0)

    def test_reduces_only_spatial_dims(self):
        da = xr.DataArray(np.ones((3, 2, 2)), dims=("time", "j", "i"))
        area = xr.DataArray(np.ones((2, 2)), dims=("j", "i"))
        result = area_weighted_mean(da, area)
        assert result.dims == ("time",)
