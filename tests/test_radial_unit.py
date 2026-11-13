"""Display units must keep all plot layers aligned without mutating fit data."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import easyxrd
from easyxrd import exrd
from easyxrd.data_loader import q_to_two_theta


@pytest.fixture
def sample():
    obj = exrd(radial_unit="two-theta", i1d_ylogscale=False)
    q = np.linspace(1., 6., 40)
    obj.load_xrd_data(from_i1d_array=np.column_stack((q, q + 10)),
                      i1d_array_wavelength_in_angstrom=1.54, plot=False)
    obj.ds["i1d_baseline"] = xr.full_like(obj.ds.i1d, 1.)
    obj.ds["i1d_refined"] = obj.ds.i1d + 0.1
    obj.ds["i1d_gsas_background"] = xr.full_like(obj.ds.i1d, 2.)
    obj.ds["i1d_variance"] = xr.full_like(obj.ds.i1d, 4.)
    obj.ds["i1d_phase_profiles"] = xr.concat(
        [obj.ds.i1d * 0.4, obj.ds.i1d * 0.6], dim="phase"
    ).assign_coords(phase=["A", "B"])
    obj.ds.attrs.update(num_phases=0, Rwp=5., GOF=1.,
                        excluded_regions_q=json.dumps([[2., 3.]]))
    obj.ds_previous = obj.ds.copy(deep=True)
    yield obj
    plt.close("all")


def test_constructor_units_and_configurable_default(monkeypatch):
    assert exrd().radial_unit == "q"
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "radial_unit", "two-theta")
    assert exrd().radial_unit == "two-theta"
    assert exrd(radial_unit="q").radial_unit == "q"
    for unit in ("tth", "2theta", "two_theta", "2th_deg", "two-theta"):
        assert exrd(radial_unit=unit).radial_unit == "two-theta"
    with pytest.raises(ValueError, match="radial_unit"):
        exrd(radial_unit="invalid")


@pytest.mark.parametrize("hint", ["load_xrd_data", "get_baseline", "load_phases",
                                  "refine_background", None])
def test_two_theta_profile_layers_and_unchanged_data(sample, hint):
    before = sample.ds.copy(deep=True)
    previous = sample.ds_previous.copy(deep=True)
    sample.plot(plot_hint=hint, normalized_residuals=True)
    axes = {ax.get_label(): ax for ax in plt.gcf().axes}
    ax = axes.get("1", axes.get("C"))
    expected = q_to_two_theta(before.radial.values, 1.54)
    for line in ax.lines:
        if len(line.get_xdata()) == len(expected):
            np.testing.assert_allclose(line.get_xdata(), expected)
    assert any(r"2\theta" in axis.get_xlabel() for axis in axes.values())
    for axis in (ax, axes.get("D")):
        if axis is not None:
            vertices = axis.patches[0].get_path().transformed(
                axis.patches[0].get_patch_transform()).vertices
            np.testing.assert_allclose([vertices[:, 0].min(), vertices[:, 0].max()],
                                       q_to_two_theta([2., 3.], 1.54))
    xr.testing.assert_identical(sample.ds, before)
    xr.testing.assert_identical(sample.ds_previous, previous)


def test_q_default_and_degree_range(sample):
    sample.radial_unit = "q"
    sample.plot(plot_hint="load_xrd_data")
    np.testing.assert_allclose(plt.gca().lines[0].get_xdata(), sample.ds.radial)
    sample.radial_unit = "two-theta"
    sample.plot(i1d_plot_radial_range=(20., 60.))
    for ax in plt.gcf().axes:
        np.testing.assert_allclose(ax.get_xlim(), [20., 60.])


def test_two_theta_cake_uses_nonuniform_bins(sample):
    q = np.linspace(1., 6., 10)
    sample.ds["i2d"] = xr.DataArray(
        np.ones((3, 10)), dims=("azimuthal_i2d", "radial_i2d"),
        coords={"azimuthal_i2d": [-60., 0., 60.], "radial_i2d": q},
        attrs=dict(sample.ds.i1d.attrs),
    )
    sample.ds.i1d.attrs["roi_azimuthal_range"] = [-30., 30.]
    sample.ds.i2d.attrs["roi_azimuthal_range"] = [-30., 30.]
    before = sample.ds.copy(deep=True)
    sample.plot(plot_hint="load_xrd_data")
    ax = next(ax for ax in plt.gcf().axes if ax.get_label() == "B")
    edges = ax.collections[0].get_coordinates()[0, :, 0]
    centers = q_to_two_theta(q, 1.54)
    np.testing.assert_allclose(edges[1:-1], (centers[:-1] + centers[1:]) / 2)
    assert not np.allclose(np.diff(edges), np.diff(edges)[0])
    roi = ax.patches[0]
    np.testing.assert_allclose([roi.get_x(), roi.get_x() + roi.get_width()],
                               q_to_two_theta([1., 6.], 1.54))
    xr.testing.assert_identical(sample.ds, before)


def test_two_theta_phase_markers(sample, monkeypatch):
    from easyxrd.plotters import exrd_plotter
    calls = []

    def pattern(calc, structure, wavelength, low, high):
        calls.append([low, high])
        return np.array([2., 4.]), np.array([50., 100.]), [
            [{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}]]

    monkeypatch.setattr("easyxrd.plotters._calc_reflection_pattern", pattern)
    exrd_plotter(sample.ds, phases={"phase": object()}, plot_hint="load_phases",
                 radial_unit=sample.radial_unit)
    np.testing.assert_allclose(calls, [[1., 6.]])
    ax = next(ax for ax in plt.gcf().axes if ax.get_label() == "P")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), q_to_two_theta([2., 4.], 1.54))


def test_two_theta_requires_wavelength(sample):
    del sample.ds.i1d.attrs["wavelength_in_angst"]
    with pytest.raises(ValueError, match="wavelength_in_angst"):
        sample.plot()


def test_automatic_load_plot():
    sample = exrd(radial_unit="two-theta")
    try:
        sample.load_xrd_data(from_i1d_array=np.array([[1., 10.], [2., 15.]]),
                             i1d_array_wavelength_in_angstrom=1.54)
        np.testing.assert_allclose(plt.gca().lines[0].get_xdata(),
                                   q_to_two_theta([1., 2.], 1.54))
    finally:
        plt.close("all")


def test_direct_refinement_plot_respects_object_unit(sample, monkeypatch):
    monkeypatch.setattr(sample, "_restore_project_checkpoint", lambda checkpoint: None)
    monkeypatch.setattr(sample, "gpx_saver", lambda: None)
    sample.last_refinement_rejection_reason = "invalid_result"
    sample._restore_rejected_refinement(None, "Rejected refinement", plot=True)
    ax = next(ax for ax in plt.gcf().axes if ax.get_label() == "1")
    np.testing.assert_allclose(ax.lines[0].get_xdata(),
                               q_to_two_theta(sample.ds.radial, 1.54))


@pytest.fixture
def integrator():
    """Capture the bounds delivered to both pyFAI integration entry points."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    def profile(npt, **kwargs):
        low, high = kwargs["radial_range"] or (0.1, 0.2)
        return SimpleNamespace(radial=np.linspace(low, high, npt),
                               intensity=np.ones(npt), sigma=None)

    def cake(npt_rad, npt_azim, **kwargs):
        result = profile(npt_rad, **kwargs)
        return SimpleNamespace(radial=result.radial, azimuthal=np.linspace(-180, 180, npt_azim),
                               intensity=np.ones((npt_azim, npt_rad)), sigma=None,
                               sum_normalization=np.ones((npt_azim, npt_rad)))

    return SimpleNamespace(wavelength=1.54e-10, empty=0., dist=0.1, poni1=0., poni2=0.,
                           rot1=0., rot2=0., rot3=0., detector=SimpleNamespace(name="test"),
                           get_config=lambda: {}, integrate1d=Mock(side_effect=profile),
                           integrate2d=Mock(side_effect=cake))


@pytest.mark.parametrize("wavelength", [0.5, 1.54])
@pytest.mark.parametrize("delta_q", [None, 0.01])
def test_detector_range_in_degrees(integrator, wavelength, delta_q):
    integrator.wavelength = wavelength * 1e-10
    obj = exrd(radial_unit="two-theta")
    obj.load_xrd_data(from_img_array=np.ones((2, 2)), ai=integrator,
                      radial_range=(10., 60.), delta_q=delta_q, radial_npts=20,
                      npt_azimuthal=3, plot=False)
    expected = 4 * np.pi / wavelength * np.sin(np.deg2rad([10., 60.]) / 2)
    for method in (integrator.integrate1d, integrator.integrate2d):
        np.testing.assert_allclose(method.call_args.kwargs["radial_range"], expected)
        assert method.call_args.kwargs["unit"] == "q_A^-1"
    npt = 20 if delta_q is None else int(np.ceil(np.diff(expected)[0] / delta_q))
    assert integrator.integrate1d.call_args.kwargs["npt"] == npt
    np.testing.assert_allclose(q_to_two_theta(obj.ds.radial[[0, -1]], wavelength), [10., 60.])
    provenance = json.loads(obj.ds.attrs["integration_provenance"])
    np.testing.assert_allclose(provenance["radial_range"], expected)
    assert provenance["radial_range_unit"] == "q_A^-1"


def test_range_defaults_none_and_q_compatibility(integrator, monkeypatch):
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "load_xrd_data_radial_range", (10., 60.))
    obj = exrd(radial_unit="two-theta")
    kwargs = dict(from_img_array=np.ones((2, 2)), ai=integrator, integrate2d=False,
                  delta_q=None, radial_npts=20, plot=False)
    obj.load_xrd_data(**kwargs)
    np.testing.assert_allclose(q_to_two_theta(obj.ds.radial[[0, -1]], 1.54), [10., 60.])
    obj.load_xrd_data(**kwargs, radial_range=None)
    assert integrator.integrate1d.call_args.kwargs["radial_range"] is None
    exrd().load_xrd_data(**dict(kwargs, delta_q=0.1), radial_range=(1.05, 1.29))
    np.testing.assert_allclose(integrator.integrate1d.call_args.kwargs["radial_range"],
                               [1.05, 1.35])


@pytest.mark.parametrize("bounds", [(80., 20.), (-1., 10.), (0., 181.),
                                    (np.nan, 10.), [10.], 4.])
def test_invalid_degree_bounds_do_not_integrate(integrator, bounds):
    obj = exrd(radial_unit="two-theta")
    with pytest.raises(ValueError, match="radial_range"):
        obj.load_xrd_data(from_img_array=np.ones((2, 2)), ai=integrator,
                          radial_range=bounds, plot=False)
    integrator.integrate1d.assert_not_called()
    assert not hasattr(obj, "ds")


@pytest.mark.parametrize("input_unit", ["tth", "q"])
@pytest.mark.parametrize("wavelength", [0.5, 1.54])
def test_text_range_in_degrees(tmp_path, input_unit, wavelength):
    degrees = np.array([0., 20., 40., 60., 80.])
    x = (degrees if input_unit == "tth" else
         4 * np.pi / wavelength * np.sin(np.deg2rad(degrees) / 2))
    path = tmp_path / "pattern.xy"
    np.savetxt(path, np.column_stack((x, np.arange(5.) + 10)))
    obj = exrd(radial_unit="two-theta")
    obj.load_xrd_data(from_txt_file=path, txt_file_radial_unit=input_unit,
                      txt_file_wavelength_in_angstrom=wavelength,
                      radial_range=(20., 60.), plot=False)
    np.testing.assert_allclose(q_to_two_theta(obj.ds.radial, wavelength), [20., 40., 60.])
    np.testing.assert_array_equal(obj.ds.i1d.values, [11., 12., 13.])


def test_real_detector_degree_range_matches_q_integration():
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import Detector

    ai = AzimuthalIntegrator(dist=.1, poni1=.003, poni2=.004,
                            detector=Detector(pixel1=1e-4, pixel2=1e-4), wavelength=1e-10)
    kwargs = dict(from_img_array=np.ones((100, 100)), ai=ai, plot=False,
                  delta_q=None, radial_npts=20, npt_azimuthal=10)
    degrees = (0.5, 3.)
    q_bounds = 4 * np.pi * np.sin(np.deg2rad(degrees) / 2)
    in_degrees, in_q = exrd(radial_unit="two-theta"), exrd()
    in_degrees.load_xrd_data(**kwargs, radial_range=degrees)
    in_q.load_xrd_data(**kwargs, radial_range=q_bounds)
    xr.testing.assert_identical(in_degrees.ds, in_q.ds)
