"""GEOS collection readers tested against a synthetic holding/ tree of small nc4 files."""

import numpy as np
import pytest
import xarray as xr

from dyamond_fluxes.geos import (
    collection_files,
    list_geos_collections,
    load_geos_coords,
    nearest_file,
    open_geos,
    peek_variables,
)

STAMPS = ["20200119_2107", "20200119_2122", "20200119_2137", "20200120_0007"]


@pytest.fixture
def fake_holding(tmp_path, monkeypatch):
    monkeypatch.setenv("DYAMOND_ROOT", str(tmp_path))
    coll = tmp_path / "holding" / "tavg_15mn_2d_flx_Mx"
    coll.mkdir(parents=True)
    for i, stamp in enumerate(STAMPS):
        t = np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}")
        ds = xr.Dataset(
            {"EFLUX": (("time", "y", "x"), np.full((1, 3, 4), float(i)))},
            coords={"time": [t]},
        )
        ds["EFLUX"].attrs["long_name"] = "total latent energy flux"
        ds.to_netcdf(coll / f"DYAMOND_c1440_llc2160.tavg_15mn_2d_flx_Mx.{stamp}z.nc4")
    (coll / "not_a_collection_file.nc4").touch()  # non-matching name is ignored

    coords = xr.Dataset(
        {
            "lons": (("y", "x"), np.linspace(-180, 180, 12).reshape(3, 4)),
            "lats": (("y", "x"), np.linspace(-60, 60, 12).reshape(3, 4)),
        }
    )
    coords.to_netcdf(tmp_path / "holding" / "geos_c1440_lats_lons_2D.nc")
    return tmp_path


def test_list_geos_collections(fake_holding):
    assert list_geos_collections() == ["tavg_15mn_2d_flx_Mx"]


def test_collection_files_time_filter(fake_holding):
    files, times = collection_files("tavg_15mn_2d_flx_Mx")
    assert len(files) == 4
    files, times = collection_files(
        "tavg_15mn_2d_flx_Mx", start="2020-01-19T21:15", end="2020-01-19T22:00"
    )
    assert [f.name.split(".")[2] for f in files] == ["20200119_2122z", "20200119_2137z"]
    assert times[0] == np.datetime64("2020-01-19T21:22:00")


def test_nearest_file(fake_holding):
    f = nearest_file("tavg_15mn_2d_flx_Mx", "2020-01-19T21:30")
    assert "20200119_2137z" in f.name or "20200119_2122z" in f.name


def test_open_geos_single_and_multi(fake_holding):
    one = open_geos("tavg_15mn_2d_flx_Mx", start="2020-01-20", end="2020-01-21")
    assert one.sizes["time"] == 1
    many = open_geos("tavg_15mn_2d_flx_Mx", end="2020-01-19T22:00")
    assert many.sizes["time"] == 3
    np.testing.assert_allclose(many["EFLUX"].isel(time=2).values, 2.0)


def test_open_geos_empty_range_raises(fake_holding):
    with pytest.raises(FileNotFoundError, match="No nc4 files"):
        open_geos("tavg_15mn_2d_flx_Mx", start="2021-06-01")


def test_peek_variables(fake_holding):
    assert peek_variables("tavg_15mn_2d_flx_Mx") == {"EFLUX": "total latent energy flux"}


def test_load_geos_coords(fake_holding):
    coords = load_geos_coords()
    assert {"lons", "lats"} <= set(coords.data_vars) | set(coords.coords)
