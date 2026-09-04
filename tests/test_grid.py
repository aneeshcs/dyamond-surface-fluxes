"""Grid utilities tested on a small synthetic 2-face LLC-like grid (no data access)."""

import numpy as np
import pytest
import xarray as xr

from dyamond_fluxes.grid import (
    bin_to_latlon,
    faces_in_bbox,
    rotate_uv_to_east_north,
    subset_bbox,
)


@pytest.fixture
def synthetic_llc() -> xr.Dataset:
    """Two 20x20 'faces': face 0 covers the western Atlantic, face 1 the western Pacific."""
    n = 20
    j = np.arange(n)
    i = np.arange(n)

    def face(lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
        lon2, lat2 = np.meshgrid(lon0 + i * 1.0, lat0 + j * 1.0)
        return lon2, lat2

    lon_a, lat_a = face(-80.0, 20.0)  # Gulf Stream-ish
    lon_p, lat_p = face(140.0, 20.0)  # Kuroshio-ish
    lon = xr.DataArray(np.stack([lon_a, lon_p]), dims=("face", "j", "i"))
    lat = xr.DataArray(np.stack([lat_a, lat_p]), dims=("face", "j", "i"))

    field = xr.DataArray(
        np.stack([np.full((n, n), 10.0), np.full((n, n), -30.0)]),
        dims=("face", "j", "i"),
        name="q",
    )
    area = xr.DataArray(np.ones((2, n, n)), dims=("face", "j", "i"))
    return xr.Dataset({"q": field, "XC": lon, "YC": lat, "rA": area})


def test_bin_to_latlon_preserves_uniform_values(synthetic_llc):
    ds = synthetic_llc
    binned = bin_to_latlon(ds["q"], ds["XC"], ds["YC"], area=ds["rA"], dlon=1.0, dlat=1.0)
    assert binned.dims == ("lat", "lon")
    # Bins covering face 0 must equal 10, face 1 must equal -30 (weighted mean of constants).
    atl = binned.sel(lat=25.5, lon=-75.5)
    pac = binned.sel(lat=25.5, lon=145.5)
    np.testing.assert_allclose(atl, 10.0)
    np.testing.assert_allclose(pac, -30.0)
    # Bins with no source cells are NaN (e.g., the southern ocean here).
    assert np.isnan(binned.sel(lat=-60.5, lon=0.5))


def test_bin_to_latlon_area_weighting():
    # Two cells in one bin with areas 1 and 3: mean = (1*0 + 3*4) / 4 = 3.
    da = xr.DataArray([[0.0, 4.0]], dims=("j", "i"))
    lon = xr.DataArray([[10.1, 10.2]], dims=("j", "i"))
    lat = xr.DataArray([[0.1, 0.2]], dims=("j", "i"))
    area = xr.DataArray([[1.0, 3.0]], dims=("j", "i"))
    binned = bin_to_latlon(da, lon, lat, area=area, dlon=1.0, dlat=1.0)
    np.testing.assert_allclose(binned.sel(lat=0.5, lon=10.5), 3.0)


def test_bin_to_latlon_skips_nan_land():
    da = xr.DataArray([[np.nan, 7.0]], dims=("j", "i"))
    lon = xr.DataArray([[10.1, 10.2]], dims=("j", "i"))
    lat = xr.DataArray([[0.1, 0.2]], dims=("j", "i"))
    binned = bin_to_latlon(da, lon, lat, dlon=1.0, dlat=1.0)
    np.testing.assert_allclose(binned.sel(lat=0.5, lon=10.5), 7.0)


def test_rotate_uv_identity_and_quarter_turn():
    u = xr.DataArray([1.0, 1.0], dims="x")
    v = xr.DataArray([0.0, 0.0], dims="x")
    # Grid aligned with geography (angle 0), then rotated 90 deg.
    cs = xr.DataArray([1.0, 0.0], dims="x")
    sn = xr.DataArray([0.0, 1.0], dims="x")
    u_e, v_n = rotate_uv_to_east_north(u, v, cs, sn)
    np.testing.assert_allclose(u_e.values, [1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(v_n.values, [0.0, 1.0], atol=1e-15)


def test_rotate_uv_preserves_magnitude():
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, 50)
    u = xr.DataArray(rng.normal(size=50), dims="x")
    v = xr.DataArray(rng.normal(size=50), dims="x")
    u_e, v_n = rotate_uv_to_east_north(
        u, v, xr.DataArray(np.cos(ang), dims="x"), xr.DataArray(np.sin(ang), dims="x")
    )
    np.testing.assert_allclose(u_e**2 + v_n**2, u**2 + v**2, rtol=1e-12)


def test_faces_in_bbox(synthetic_llc):
    ds = synthetic_llc
    gulf = faces_in_bbox(ds["XC"], ds["YC"], (-82.0, -55.0, 25.0, 45.0))
    kuroshio = faces_in_bbox(ds["XC"], ds["YC"], (135.0, 165.0, 25.0, 45.0))
    assert gulf == [0]
    assert kuroshio == [1]


def test_faces_in_bbox_dateline(synthetic_llc):
    ds = synthetic_llc
    # Box from 150E across the dateline to 170W catches only the Pacific face.
    faces = faces_in_bbox(ds["XC"], ds["YC"], (150.0, -170.0, 20.0, 40.0))
    assert faces == [1]


def test_subset_bbox_crops_and_masks(synthetic_llc):
    ds = synthetic_llc
    bbox = (-75.0, -70.0, 25.0, 30.0)
    pieces = subset_bbox(ds["q"], ds["XC"], ds["YC"], bbox)
    assert len(pieces) == 1
    face, field, lon, lat = pieces[0]
    assert face == 0
    assert field.shape == lon.shape == lat.shape
    # All unmasked values inside the box, and all equal to the face constant.
    assert np.nanmax(np.abs(field.values - 10.0)) == 0.0
    inside = field.notnull().values
    assert inside.any()
    lo = lon.values[inside]
    la = lat.values[inside]
    assert lo.min() >= bbox[0] and lo.max() <= bbox[1]
    assert la.min() >= bbox[2] and la.max() <= bbox[3]
