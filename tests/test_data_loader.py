import os
import tempfile

import numpy as np
import pytest
import xarray as xr

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


@pytest.mark.parametrize("use_save_dir", [False, True])
@pytest.mark.parametrize("include_i2d", [False, True])
def test_export_refinement_metadata(tmp_path, use_save_dir, include_i2d):
    """GSAS-II Rvals must export without changing the live analysis dataset."""
    sample = exrd()
    shifts = {":0:Scale": np.float64(6.616707336669345e-05),
              "0::A0": np.float64(-1.026268973268291e-05)}
    sample.ds = xr.Dataset(
        {"i1d": ("radial", [10., 20.]),
         "i1d_refined": ("radial", [9., 19.]),
         "i1d_baseline": ("radial", [1., 1.])},
        coords={"radial": [2., 3.]},
        attrs={"lastShifts": shifts, "converged": np.bool_(True),
               "Aborted": False, "Rwp": np.float64(8.53), "Nobs": 2,
               "PhaseInd_0_cif": "data_Ni\n_cell_length_a 3.52254"},
    )
    if include_i2d:
        sample.ds["i2d"] = (("azimuthal_i2d", "radial_i2d"), [[1., 2.], [3., 4.]])
    sample.ds.i1d.attrs = {"radial_unit": "q_A^-1", "metadata": {"source": "Ni"},
                           "optional": None}
    sample.ds.radial.attrs = {"calibrated": True}
    before = sample.ds.copy(deep=True)
    path = tmp_path / "refined.nc"
    if use_save_dir:
        sample.export_ds_to(save_dir=tmp_path, save_name=path.name)
    else:
        sample.export_ds_to(to=path)

    restored = exrd()
    restored.load_xrd_data(from_nc_file=path, plot=False)
    xr.testing.assert_equal(restored.ds, before)
    assert restored.ds.attrs["lastShifts"] == str(shifts)
    assert restored.ds.attrs["converged"] == "True"
    assert restored.ds.attrs["Aborted"] == "False"
    assert restored.ds.attrs["Rwp"] == 8.53
    assert restored.ds.attrs["Nobs"] == 2
    assert restored.ds.attrs["PhaseInd_0_cif"] == before.attrs["PhaseInd_0_cif"]
    assert restored.ds.i1d.attrs["metadata"] == "{'source': 'Ni'}"
    assert restored.ds.i1d.attrs["optional"] == "None"
    assert restored.ds.radial.attrs["calibrated"] == "True"
    if include_i2d:
        assert restored.ds.i2d.encoding["zlib"]
        assert restored.ds.i2d.encoding["complevel"] == 9
    xr.testing.assert_identical(sample.ds, before)
    assert sample.ds.attrs["lastShifts"] is shifts


@pytest.mark.parametrize("value, expected", [
    ([1, 2], [1, 2]),
    ((1.5, 2.5), [1.5, 2.5]),
    (["Ni", "Cu"], ["Ni", "Cu"]),
    (np.array(["Ni", "Cu"]), ["Ni", "Cu"]),
    (np.array([1., 2.]), [1., 2.]),
    (np.array(3.), 3.),
    (np.array([True, False]), [1, 0]),
    ([[1, 2], [3, 4]], "[[1, 2], [3, 4]]"),
    ([1, [2, 3]], "[1, [2, 3]]"),
    ([{"Ni": 1.}], "[{'Ni': 1.0}]"),
    (np.array([{"Ni": 1.}], dtype=object), "[{'Ni': 1.0}]"),
])
def test_export_attribute_arrays(tmp_path, value, expected):
    sample = exrd()
    sample.ds = xr.Dataset({"i1d": ("radial", [1., 2.])}, attrs={"metadata": value})
    before = sample.ds.copy(deep=True)
    path = tmp_path / "arrays.nc"
    sample.export_ds_to(path)
    with xr.open_dataset(path, engine="h5netcdf") as restored:
        np.testing.assert_equal(restored.attrs["metadata"], expected)
    xr.testing.assert_identical(sample.ds, before)


def test_export_write_errors_propagate(tmp_path):
    sample = exrd()
    sample.ds = xr.Dataset({"i1d": ("radial", [1., 2.])})
    before = sample.ds.copy(deep=True)
    with pytest.raises(OSError):
        sample.export_ds_to(tmp_path / "missing_directory" / "refined.nc")
    xr.testing.assert_identical(sample.ds, before)


def test_load_detector_image_defaults_to_poisson_error_model():
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    ai = AzimuthalIntegrator(
        dist=0.1,
        poni1=0.003,
        poni2=0.004,
        detector=Detector(pixel1=1e-4, pixel2=1e-4),
        wavelength=1e-10,
    )
    y, x = np.indices((64, 64))
    img = (100 + 2 * x + y).astype(float)
    sample = exrd()
    sample.load_xrd_data(
        from_img_array=img,
        ai=ai,
        radial_range=(0.05, 0.3),
        delta_q=0.01,
        plot=False,
    )
    assert sample.ds.i1d.attrs["error_model"] == "poisson"
    assert "i1d_variance" in sample.ds
    assert np.all(sample.ds.i1d_variance > 0)


def test_export_i1d_to_xye(tmp_path):
    q = np.array([1.0, 2.0, 3.0])
    intensity = np.array([10.0, 25.0, 15.0])
    variance = np.array([0.25, 0.64, 0.36])

    sample = exrd()
    sample.load_xrd_data(
        from_i1d_array=np.column_stack((q, intensity)),
        i1d_array_radial_unit="q",
        i1d_array_wavelength_in_angstrom=1.5406,
        i1d_variance=variance,
        plot=False,
    )
    out_file = tmp_path / "out.xye"
    sample.export_i1d_to(to=out_file, mode="xye", header=True)
    exported = np.loadtxt(out_file)
    assert exported.shape == (3, 3)
    expected_tth = np.rad2deg(2 * np.arcsin(q * 1.5406 / (4 * np.pi)))
    np.testing.assert_allclose(exported[:, 0], expected_tth, rtol=1e-4)
    np.testing.assert_allclose(exported[:, 1], intensity, rtol=1e-4)
    np.testing.assert_allclose(exported[:, 2], np.sqrt(variance), rtol=1e-4)

    q_out_file = tmp_path / "out.qxye"
    sample.export_i1d_to(to=q_out_file, mode="qxye", header=True)
    q_exported = np.loadtxt(q_out_file)
    assert q_exported.shape == (3, 3)
    np.testing.assert_allclose(q_exported[:, 0], q, rtol=1e-4)
    np.testing.assert_allclose(q_exported[:, 1], intensity, rtol=1e-4)
    np.testing.assert_allclose(q_exported[:, 2], np.sqrt(variance), rtol=1e-4)

    no_var = exrd()
    no_var.load_xrd_data(
        from_i1d_array=np.column_stack((q, intensity)),
        i1d_array_radial_unit="q",
        i1d_array_wavelength_in_angstrom=1.5406,
        plot=False,
    )
    for mode in ("xye", "qxye"):
        with pytest.raises(ValueError, match="i1d_variance"):
            no_var.export_i1d_to(to=tmp_path / f"missing_var.{mode}", mode=mode)


def test_load_3col_txt_file_automatically(tmp_path):
    tth = np.array([10.0, 20.0, 30.0])
    intensity = np.array([100.0, 200.0, 150.0])
    sigma = np.array([2.0, 3.0, 4.0])
    txt_path = tmp_path / "three_col.xye"
    np.savetxt(txt_path, np.column_stack((tth, intensity, sigma)))

    sample = exrd()
    sample.load_xrd_data(
        from_txt_file=txt_path,
        txt_file_wavelength_in_angstrom=1.5406,
        plot=False,
    )
    assert "i1d_variance" in sample.ds
    np.testing.assert_allclose(sample.ds.i1d_variance.values, sigma**2)
