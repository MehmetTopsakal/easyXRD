import os
import tempfile
import numpy as np
import pytest
from easyxrd import exrd


def test_load_1d_array_q_unit():
    # Synthetic Q data
    q = np.linspace(1.0, 5.0, 50)
    intensity = 100.0 * np.exp(-0.5 * ((q - 2.5) / 0.1) ** 2)
    data = np.column_stack((q, intensity))

    e = exrd()
    e.load_xrd_data(
        from_i1d_array=data,
        i1d_array_radial_unit="q",
        i1d_array_wavelength_in_angstrom=1.5406,
        plot=False,
    )

    assert hasattr(e, "ds")
    assert "i1d" in e.ds
    assert len(e.ds.i1d) == 50
    np.testing.assert_allclose(e.ds.i1d.radial.values, q, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(e.ds.i1d.values, intensity, rtol=1e-4, atol=1e-4)


def test_load_1d_array_tth_unit_wavelength_fix():
    # 2-theta = 30.0 degrees
    tth = np.array([30.0, 45.0, 60.0])
    intensity = np.array([100.0, 50.0, 20.0])
    data = np.column_stack((tth, intensity))

    wavelength = 1.5406  # Cu-Ka
    # Expected q = (4*pi/lambda) * sin(tth/2 in radians)
    expected_q = ((4 * np.pi) / wavelength) * np.sin(np.deg2rad(tth) / 2)

    e = exrd()
    e.load_xrd_data(
        from_i1d_array=data,
        i1d_array_radial_unit="tth",
        i1d_array_wavelength_in_angstrom=wavelength,
        plot=False,
    )

    assert hasattr(e, "ds")
    assert "i1d" in e.ds
    # Verify coordinates match expected_q calculated with custom wavelength (not default 0.1814)
    np.testing.assert_allclose(e.ds.i1d.radial.values, expected_q, rtol=1e-4)


def test_load_from_txt_and_export():
    tth = np.linspace(10.0, 50.0, 40)
    intensity = 50.0 + 200.0 * np.exp(-0.5 * ((tth - 28.0) / 0.5) ** 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        txt_path = os.path.join(tmpdir, "pattern.xy")
        np.savetxt(txt_path, np.column_stack((tth, intensity)))

        e = exrd()
        e.load_xrd_data(
            from_txt_file=txt_path,
            txt_file_wavelength_in_angstrom=1.5406,
            txt_file_radial_unit="tth",
            plot=False,
        )

        assert "i1d" in e.ds
        assert len(e.ds.i1d) == 40

        # Test export in modes
        for mode in ["xy", "qxy", "d"]:
            out_file = os.path.join(tmpdir, f"export_{mode}.dat")
            e.export_i1d_to(to=out_file, mode=mode, header=True)
            assert os.path.isfile(out_file)
            loaded = np.loadtxt(out_file)
            assert loaded.shape[0] == 40
            assert loaded.shape[1] == 2


def test_export_and_load_netcdf():
    q = np.linspace(1.0, 6.0, 60)
    intensity = 100.0 * np.exp(-0.5 * ((q - 3.0) / 0.2) ** 2)
    data = np.column_stack((q, intensity))

    e1 = exrd()
    e1.load_xrd_data(from_i1d_array=data, i1d_array_radial_unit="q", plot=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        nc_path = os.path.join(tmpdir, "dataset.nc")
        e1.export_ds_to(to=nc_path)
        assert os.path.isfile(nc_path)

        e2 = exrd()
        e2.load_xrd_data(from_nc_file=nc_path, plot=False)
        assert "i1d" in e2.ds
        np.testing.assert_allclose(e2.ds.i1d.values, e1.ds.i1d.values, rtol=1e-5)
