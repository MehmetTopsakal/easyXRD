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
from easyxrd.plotters import _calc_reflection_pattern, exrd_plotter


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
