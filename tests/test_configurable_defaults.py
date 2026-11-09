"""Behavior of session defaults and explicit per-call overrides."""
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import easyxrd
from easyxrd import exrd
from easyxrd.plotters import exrd_plotter


@pytest.fixture
def profile():
    q = np.linspace(1., 5., 20)
    sample = exrd()
    sample.load_xrd_data(from_i1d_array=np.column_stack((q, q + 10)), plot=False)
    return sample


def test_new_objects_use_current_defaults_and_keep_independent_settings(monkeypatch):
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "plot_figsize", [10, 7])
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "plot_i1d_ylogscale", False)
    first = exrd()
    first.figsize[0] = 12
    assert easyxrd.easyxrd_defaults["plot_figsize"] == [10, 7]
    assert exrd().figsize == [10, 7]
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "plot_figsize", (6, 4))
    assert first.figsize == [12, 7]
    assert exrd().figsize == (6, 4)
    assert first.i1d_ylogscale is False
    assert exrd(figsize=(9, 5), i1d_ylogscale=True).figsize == (9, 5)
    assert exrd(i1d_ylogscale=True).i1d_ylogscale is True


def test_plot_defaults_after_import_and_explicit_overrides(monkeypatch, profile):
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "plot_figsize", (10, 9))
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "plot_dpi", 160)
    try:
        exrd_plotter(profile.ds, plot_hint="load_xrd_data")
        np.testing.assert_allclose(plt.gcf().get_size_inches(), [10, 6])
        assert plt.gcf().dpi == 160
        profile.plot(figsize=(12, 6), dpi=200)
        np.testing.assert_allclose(plt.gcf().get_size_inches(), [12, 4])
        assert plt.gcf().dpi == 200
    finally:
        plt.close("all")


def test_loader_uses_configured_wavelength_and_explicit_value(monkeypatch):
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "load_xrd_data_i1d_array_wavelength_in_angstrom", 1.54)
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "load_xrd_data_plot", False)
    plot = Mock()
    sample = exrd()
    monkeypatch.setattr(sample, "plot", plot)
    data = np.array([[1., 10.], [2., 20.], [3., 15.]])
    sample.load_xrd_data(from_i1d_array=data)
    assert sample.ds.i1d.attrs["wavelength_in_angst"] == 1.54
    plot.assert_not_called()
    sample.load_xrd_data(from_i1d_array=data, i1d_array_wavelength_in_angstrom=0.5, plot=True)
    assert sample.ds.i1d.attrs["wavelength_in_angst"] == 0.5
    plot.assert_called_once()


def test_baseline_uses_configured_lambda(monkeypatch, profile):
    values = np.zeros(profile.ds.i1d.size)
    fit = Mock(return_value=(values, {}))
    monkeypatch.setattr("easyxrd.baseline.pybaselines.Baseline", lambda **kw: SimpleNamespace(iarpls=fit))
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "get_baseline_iarpls_lam", 1e6)
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "get_baseline_plot", False)
    profile.get_baseline()
    assert fit.call_args.kwargs["lam"] == 1e6
    profile.get_baseline(iarpls_lam=1e3)
    assert fit.call_args.kwargs["lam"] == 1e3


def test_refinement_setter_reads_config_and_respects_false(monkeypatch):
    sample = exrd()
    sample.gpx = SimpleNamespace(set_refinement=Mock())
    sample.gpx_saver = Mock()
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "set_background_refinement_set_num_coeffs_to", 15)
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "set_background_refinement_save_gpx", True)
    sample.set_background_refinement(save_gpx=False)
    settings = sample.gpx.set_refinement.call_args.args[0]
    assert settings["set"]["Background"]["no. coeffs"] == 15
    sample.gpx_saver.assert_not_called()
    sample.set_background_refinement(set_num_coeffs_to=4)
    assert sample.gpx.set_refinement.call_args.args[0]["set"]["Background"]["no. coeffs"] == 4
    sample.gpx_saver.assert_called_once()


def test_explicit_none_disables_configured_refinement_threshold(monkeypatch):
    sample = exrd()
    monkeypatch.setitem(easyxrd.easyxrd_defaults, "refine_min_gof_improvement", 3.)
    # Stop at the engine boundary to inspect the resolved threshold.
    class Stop(Exception):
        pass
    seen = []
    def prepare(threshold):
        seen.append(threshold)
        raise Stop
    monkeypatch.setattr(sample, "_prepare_refinement", prepare)
    with pytest.raises(Stop):
        sample.refine()
    with pytest.raises(Stop):
        sample.refine(min_gof_improvement=None)
    assert seen == [3., None]
