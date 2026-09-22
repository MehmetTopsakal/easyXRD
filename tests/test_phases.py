import os
import tempfile
import pytest
import numpy as np
import xarray as xr
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from easyxrd import exrd


def _create_sample_cif(filepath):
    # Create simple cubic Si structure
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    structure.to(filename=filepath, fmt="cif")
    return structure


def test_load_phases_from_cif():
    with tempfile.TemporaryDirectory() as tmpdir:
        cif_path = os.path.join(tmpdir, "silicon.cif")
        _create_sample_cif(cif_path)

        e = exrd()
        e.ds = xr.Dataset()
        e.ds["i1d"] = xr.DataArray(
            data=np.ones(50, dtype=np.float32),
            coords=[np.linspace(1.0, 5.0, 50)],
            dims=["radial"],
            attrs={"wavelength_in_angst": 1.5406, "xlabel": "q"},
        )

        phases_dict = [
            {"cif": cif_path, "label": "Si_Cubic", "scale": 1.01}
        ]

        e.load_phases(from_phases_dict=phases_dict, plot=False)

        assert "Si_Cubic" in e.phases
        assert e.ds.attrs["num_phases"] == 1
        assert "PhaseInd_0_cif" in e.ds.attrs
        assert e.ds.attrs["PhaseInd_0_label"] == "Si_Cubic"

        # Check lattice parameter scaled by 1.01
        np.testing.assert_allclose(e.phases["Si_Cubic"].lattice.a, 5.43 * 1.01, rtol=1e-5)


def test_load_phases_empty_dict():
    e = exrd()
    e.ds = xr.Dataset()
    # Empty phase list must not raise UnboundLocalError
    e.load_phases(from_phases_dict=[], plot=False)
    assert e.ds.attrs["num_phases"] == 0
    assert len(e.phases) == 0


def test_missing_materials_project_key_raises_value_error():
    e = exrd()
    e.ds = xr.Dataset()
    phases_dict = [
        {"mp_id": "mp-149", "label": "Si"}
    ]
    # Without valid API key, should raise ValueError rather than blocking on input()
    with pytest.raises(ValueError, match="Materials Project API key"):
        e.load_phases(from_phases_dict=phases_dict, mp_rester_api_key="invalid", plot=False)
