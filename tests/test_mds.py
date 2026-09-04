"""MDS binary reader tested against synthetic compact-LLC files (small face size)."""

import numpy as np
import pytest

from dyamond_fluxes import mds
from dyamond_fluxes.mds import (
    iters_to_time,
    list_mit_variables,
    open_grid,
    open_mds_variable,
    open_ocean_dataset,
)

N = 4  # tiny face size standing in for 2160


@pytest.fixture
def fake_mit(tmp_path, monkeypatch):
    """mit/ tree with a 2-D variable (3 steps), a 3-D variable, and grid files."""
    monkeypatch.setenv("DYAMOND_ROOT", str(tmp_path))
    mit = tmp_path / "mit"

    rng = np.random.default_rng(0)
    truth = {}
    qdir = mit / "oceQnet"
    qdir.mkdir(parents=True)
    for it in (0, 80, 160):
        arr = rng.normal(size=(13, N, N)).astype(">f4")
        arr.tofile(qdir / f"oceQnet.{it:010d}.data")
        truth[it] = arr

    tdir = mit / "Theta"
    tdir.mkdir()
    theta = rng.normal(size=(5, 13, N, N)).astype(">f4")  # 5 vertical levels
    theta.tofile(tdir / "Theta.0000000000.data")

    gdir = mit / "grid"
    gdir.mkdir()
    for fname in ("XC", "YC", "RAC", "Depth", "AngleCS", "AngleSN"):
        rng.normal(size=(13, N, N)).astype(">f4").tofile(gdir / f"{fname}.data")
    (gdir / "DXC.meta").write_text("ignored")

    return truth


def test_list_mit_variables(fake_mit):
    assert list_mit_variables() == ["Theta", "oceQnet"]  # grid/ has no <var>.<iter>.data


def test_open_2d_variable_values_and_time(fake_mit):
    da = open_mds_variable("oceQnet", n=N)
    assert da.dims == ("time", "face", "j", "i")
    assert da.shape == (3, 13, N, N)
    # Lazy: dask-backed until .values
    assert da.chunks is not None
    np.testing.assert_array_equal(da.isel(time=1).values, fake_mit[80])
    # 80 iterations x 45 s = 1 hour after T0
    assert da.time.values[0] == np.datetime64("2020-01-19T21:00:00")
    assert da.time.values[1] - da.time.values[0] == np.timedelta64(3600, "s")
    assert list(da.iteration.values) == [0, 80, 160]
    # Sign convention attrs attached for the flux machinery: MDS fluxes are
    # positive-down, so to_positive_down must infer that and leave values unchanged.
    from dyamond_fluxes.fluxes import to_positive_down

    assert "+=down" in da.attrs["long_name"]
    np.testing.assert_array_equal(to_positive_down(da).isel(time=1).values, fake_mit[80])


def test_open_3d_variable_shape(fake_mit):
    da = open_mds_variable("Theta", n=N)
    assert da.dims == ("time", "k", "face", "j", "i")
    assert da.shape == (1, 5, 13, N, N)


def test_bad_file_size_raises(fake_mit, tmp_path):
    bad = tmp_path / "mit" / "Eta"
    bad.mkdir()
    np.ones(7, dtype=">f4").tofile(bad / "Eta.0000000000.data")
    with pytest.raises(ValueError, match="compact-LLC"):
        open_mds_variable("Eta", n=N)


def test_open_grid_renames(fake_mit):
    grid = open_grid(n=N)
    assert set(grid.data_vars) == {"XC", "YC", "rA", "Depth", "CS", "SN"}
    assert grid["XC"].dims == ("face", "j", "i")


def test_open_ocean_dataset_attaches_grid_coords(fake_mit):
    ds = open_ocean_dataset(["oceQnet"], n=N)
    assert "oceQnet" in ds.data_vars
    assert "XC" in ds.coords and "rA" in ds.coords


def test_iters_to_time_vectorized():
    times = iters_to_time(np.array([0, 1920]))  # 1920 x 45 s = 1 day
    assert times[1] - times[0] == np.timedelta64(86400, "s")


def test_unnested_root_fallback(tmp_path, monkeypatch):
    """A local subset without the mit/ wrapper directory still works."""
    monkeypatch.setenv("DYAMOND_ROOT", str(tmp_path))
    vdir = tmp_path / "Eta"
    vdir.mkdir()
    np.zeros((13, N, N), dtype=">f4").tofile(vdir / "Eta.0000000000.data")
    assert mds.mit_dir() == tmp_path
    assert open_mds_variable("Eta", n=N).shape == (1, 13, N, N)
