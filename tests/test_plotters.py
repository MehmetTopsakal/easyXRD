import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test runner
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pytest
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from easyxrd import exrd
from easyxrd.plotters import _calc_reflection_pattern, exrd_plotter, i2d_plotter


def test_calc_reflection_pattern():
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    xrdc = XRDCalculator(wavelength=1.5406)

    refl_x, refl_y = _calc_reflection_pattern(xrdc, structure, 1.5406, 1.0, 6.0)
    assert len(refl_x) > 0
    assert len(refl_x) == len(refl_y)
    # Check that peaks fall in requested range
    assert min(refl_x) >= 0.8
    assert max(refl_x) <= 6.2


def test_plotter_execution():
    q = np.linspace(1.0, 6.0, 100)
    intensity = 100.0 * np.exp(-0.5 * ((q - 3.0) / 0.1) ** 2) + 10.0

    e = exrd()
    e.load_xrd_data(
        from_i1d_array=np.column_stack((q, intensity)),
        i1d_array_radial_unit="q",
        i1d_array_wavelength_in_angstrom=1.5406,
        plot=False,
    )
    e.get_baseline(use_iarpls=True, iarpls_lam=1e4, plot=False)

    # Test plotting with load_xrd_data hint
    e.plot(plot_hint="load_xrd_data")
    plt.close("all")

    # Test plotting with get_baseline hint
    e.plot(plot_hint="get_baseline")
    plt.close("all")


@pytest.mark.parametrize("with_baseline", [False, True])
@pytest.mark.parametrize("logscale", [False, True])
def test_i2d_plotter_nonpositive_intensities(with_baseline, logscale):
    intensity = np.array([[-1.0, 0.0, np.nan], [1.0, np.e, np.e**2]])
    ds = xr.Dataset(
        {"i2d": (("azimuthal_i2d", "radial_i2d"), intensity.copy())},
        coords={"azimuthal_i2d": [0.0, 90.0], "radial_i2d": [1.0, 2.0, 3.0]},
    )
    if with_baseline:
        ds["i2d_baseline"] = xr.full_like(ds.i2d, 2.0)
        ds["i2d"] = ds.i2d + ds.i2d_baseline
    original = ds.copy(deep=True)

    fig, ax = plt.subplots()
    try:
        with np.errstate(invalid="raise", divide="raise"):
            i2d_plotter(ds, ax, i2d_logscale=logscale, cbar=False)

        displayed = ax.images[0].get_array()
        if logscale:
            np.testing.assert_array_equal(
                np.ma.getmaskarray(displayed),
                [[True, True, True], [False, False, False]],
            )
            np.testing.assert_allclose(displayed[1], [0.0, 1.0, 2.0], atol=1e-12)
        else:
            np.testing.assert_array_equal(
                np.ma.getmaskarray(displayed), np.isnan(intensity)
            )
            np.testing.assert_allclose(displayed.compressed(), intensity[np.isfinite(intensity)])
        xr.testing.assert_identical(ds, original)
    finally:
        plt.close(fig)
