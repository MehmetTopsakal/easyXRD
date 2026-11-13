import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from easyxrd import exrd
from easyxrd.utils import load_gsas_module, refinement_transaction


def test_two_theta_export_requires_wavelength(tmp_path):
    sample = exrd()
    sample.ds = xr.Dataset(
        {"i1d": ("radial", [1.0, 2.0], {})}, coords={"radial": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="wavelength_in_angst"):
        sample.export_i1d_to(to=tmp_path / "out.xy", mode="xy")


def test_invalid_wavelength_attribute_rejected():
    da = xr.DataArray(
        [1.0, 2.0],
        dims="radial",
        coords={"radial": [1.0, 2.0]},
        attrs={"wavelength_in_angst": -1.0},
    )
    with pytest.raises(ValueError, match="wavelength_in_angst"):
        exrd().load_xrd_data(from_da_i1d=da, plot=False)


def test_load_gsas_module_keeps_explicit_path(tmp_path, monkeypatch):
    (tmp_path / "fake_gsas_mod_xyz.py").write_text("VALUE = 1\n")
    before = sys.path[:]
    monkeypatch.setattr(sys, "path", before[:])
    module = load_gsas_module("fake_gsas_mod_xyz", paths=[str(tmp_path)])
    assert module.VALUE == 1
    assert str(tmp_path) in sys.path
    sys.modules.pop("fake_gsas_mod_xyz", None)


def test_transaction_does_not_mask_error_when_save_fails():
    class Sample:
        def __init__(self):
            self.gpx = SimpleNamespace(a=1)

        def gpx_saver(self):
            raise OSError("disk full")

        @refinement_transaction
        def run(self):
            self.gpx.a = 2
            raise RuntimeError("refinement failed")

    sample = Sample()
    with pytest.raises(RuntimeError, match="refinement failed"):
        sample.run()
    assert sample.gpx.a == 1
