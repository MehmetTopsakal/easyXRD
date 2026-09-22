import numpy as np
import xarray as xr
import pytest
from easyxrd import exrd
from easyxrd.baseline import _optimize_bkg_scale


def test_optimize_bkg_scale():
    # Observed signal with flat background
    true_bkg = np.ones(50) * 10.0
    peaks = 20.0 * np.exp(-0.5 * ((np.linspace(0, 10, 50) - 5.0) / 0.5) ** 2)
    obs = peaks + 1.5 * true_bkg

    scale = _optimize_bkg_scale(obs, true_bkg)
    diff = obs - scale * true_bkg
    assert min(diff) >= -1e-5
    assert scale == pytest.approx(1.5, rel=0.02)

    # Test with initial_scale provided
    scale_with_init = _optimize_bkg_scale(obs, true_bkg, initial_scale=1.0)
    assert scale_with_init == pytest.approx(1.5, rel=0.02)


def test_get_baseline_1d():
    q = np.linspace(1.0, 8.0, 200)
    bkg = 20.0 + 5.0 * np.sin(q)
    peaks = 150.0 * np.exp(-0.5 * ((q - 3.5) / 0.1) ** 2)
    intensity = bkg + peaks
    data = np.column_stack((q, intensity))

    e = exrd()
    e.load_xrd_data(from_i1d_array=data, i1d_array_radial_unit="q", plot=False)
    e.get_baseline(use_iarpls=True, iarpls_lam=1e4, plot=False)

    assert "i1d_baseline" in e.ds
    assert len(e.ds.i1d_baseline) == len(e.ds.i1d)
    # Baseline should be lower than peak maximum
    assert e.ds.i1d_baseline.sel(radial=3.5, method="nearest").values < 50.0


def test_baseline_preserves_raw_i2d_no_inplace_mutation():
    # Construct synthetic 2D cake dataset
    azim = np.linspace(-180, 180, 20, dtype=np.float32)
    rad = np.linspace(1.0, 5.0, 50, dtype=np.float32)

    # Raw 2D grid
    raw_2d = np.ones((len(azim), len(rad)), dtype=np.float32) * 50.0
    # Add a noisy hot pixel
    raw_2d[5, 10] = 9999.0
    original_copy = raw_2d.copy()

    e = exrd()
    e.ds = xr.Dataset()
    e.ds["i2d"] = xr.DataArray(
        data=raw_2d,
        coords=[azim, rad],
        dims=["azimuthal_i2d", "radial_i2d"],
        attrs={"wavelength_in_meter": 1.5406e-10, "xlabel": "q"},
    )
    e.ds["i1d"] = e.ds["i2d"].mean(dim="azimuthal_i2d").rename({"radial_i2d": "radial"})
    e.ds["i1d"].attrs = {"wavelength_in_angst": 1.5406, "xlabel": "q"}

    # Run get_baseline
    e.get_baseline(use_iarpls=True, iarpls_lam=1e5, plot=False)

    # Verify that raw_2d inside e.ds["i2d"] was NOT mutated in-place by median filter!
    np.testing.assert_array_equal(e.ds.i2d.values, original_copy)
    assert e.ds.i2d.values[5, 10] == 9999.0
