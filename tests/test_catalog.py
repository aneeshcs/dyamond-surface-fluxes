"""Store discovery/opening against a tiny on-disk zarr tree standing in for the ceph volume."""

import numpy as np
import pytest
import xarray as xr

from dyamond_fluxes.catalog import dyamond_root, find_stores_with, list_stores, open_store


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Miniature DYAMOND-like layout: one 'ocean' and one 'atmos' zarr store."""
    ocean = xr.Dataset(
        {"oceQnet": (("j", "i"), np.ones((4, 4))), "oceQsw": (("j", "i"), np.ones((4, 4)))}
    )
    atmos = xr.Dataset({"EFLUX": (("y", "x"), np.zeros((4, 4)))})
    ocean.to_zarr(tmp_path / "ocean" / "surface.zarr", consolidated=True)
    atmos.to_zarr(tmp_path / "geos" / "flx.zarr", consolidated=False)
    monkeypatch.setenv("DYAMOND_ROOT", str(tmp_path))
    return tmp_path


def test_dyamond_root_env_override(fake_root):
    assert dyamond_root() == fake_root


def test_dyamond_root_missing_path_raises(monkeypatch):
    monkeypatch.setenv("DYAMOND_ROOT", "/nonexistent/dyamond")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        dyamond_root()


def test_dyamond_root_skips_empty_stub_and_falls_through(tmp_path, monkeypatch):
    """An unmounted volume leaves an empty stub dir; the next candidate must win."""
    from dyamond_fluxes import catalog

    stub = tmp_path / "poseidon_ceph" / "DYAMOND"  # exists but empty
    stub.mkdir(parents=True)
    real = tmp_path / "poseidon-DYAMOND"
    real.mkdir()
    (real / "some_store").mkdir()

    monkeypatch.delenv("DYAMOND_ROOT", raising=False)
    monkeypatch.setattr(catalog, "DEFAULT_ROOTS", (str(stub), str(real)))
    assert dyamond_root() == real

    # With only the empty stub available, a clear error is raised.
    monkeypatch.setattr(catalog, "DEFAULT_ROOTS", (str(stub),))
    with pytest.raises(FileNotFoundError, match="SciServer"):
        dyamond_root()


def test_list_stores_finds_both_without_descending(fake_root):
    stores = sorted(s.name for s in list_stores())
    assert stores == ["flx.zarr", "surface.zarr"]


def test_open_store_relative_and_consolidated_fallback(fake_root):
    ds1 = open_store("ocean/surface.zarr")  # consolidated
    ds2 = open_store("geos/flx.zarr")  # unconsolidated fallback
    assert "oceQnet" in ds1 and "EFLUX" in ds2


def test_find_stores_with(fake_root):
    hits = find_stores_with(["oceQnet", "EFLUX", "HFLUX"])
    by_name = {path.name: vars_ for path, vars_ in hits.items()}
    assert by_name == {"surface.zarr": ["oceQnet"], "flx.zarr": ["EFLUX"]}
