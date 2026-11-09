"""Parameter labeling, uncertainty export, provenance and logging regressions."""
import io
import copy
import json
import logging

import numpy as np
import pytest
import xarray as xr

from easyxrd import configure_logging, exrd
from easyxrd.results import fit_result_dataset, clear_fit_results
from tests.test_review_regressions import real_sample
from tests.test_refinement import configurable_sample, Project


def covariance():
    return dict(varyList=[":0:Scale", "0::A0"], variables=[2., 0.08],
                sig=[0.2, 0.03], covMatrix=[[0.04, 0.003], [0.003, 0.0009]],
                Rvals={"Rwp": 4., "GOF": 1.2, "converged": True, "SVD0": 0})


def test_parameter_order_covariance_and_netcdf(tmp_path):
    source = covariance()
    result = fit_result_dataset(source)
    assert result.gsas_parameter.values.tolist() == source["varyList"]
    assert result.gsas_parameter_cov.values.tolist() == source["varyList"]
    np.testing.assert_allclose(result.gsas_correlation, [[1., 0.5], [0.5, 1.]])
    np.testing.assert_array_equal(result.gsas_parameter_su.values, source["sig"])
    assert result.gsas_fit_statistic.sel(gsas_statistic="Rwp") == 4.
    assert result.gsas_converged == 1
    result.to_netcdf(tmp_path / "result.nc")
    xr.testing.assert_identical(result, xr.load_dataset(tmp_path / "result.nc"))
    result.gsas_covariance.values[0, 0] = 999
    assert source["covMatrix"][0][0] == 0.04


def test_missing_covariance_and_zero_variance_are_not_certainty():
    source = covariance()
    source.pop("covMatrix")
    source["sig"] = [None, None]
    source["Rvals"]["converged"] = None
    result = fit_result_dataset(source)
    assert result.gsas_covariance_status == "unavailable"
    assert np.isnan(result.gsas_covariance).all()
    assert np.isnan(result.gsas_parameter_su).all()
    assert result.gsas_converged == -1
    source["covMatrix"] = [[0., 0.], [0., 1.]]
    result = fit_result_dataset(source)
    assert np.isnan(result.gsas_correlation[0]).all()
    assert result.gsas_correlation[1, 1] == 1


@pytest.mark.parametrize("key,value", [("varyList", ["a", "a"]),
    ("variables", [1.]), ("sig", [-1., 1.]), ("covMatrix", [[1.]])])
def test_inconsistent_parameter_results_raise(key, value):
    source = covariance(); source[key] = value
    with pytest.raises(ValueError, match="GSAS-II"):
        fit_result_dataset(source)


def test_parameter_set_replacement_does_not_align_to_old_names():
    s = exrd()
    s.ds = xr.merge([xr.Dataset({"i1d": ("radial", [1., 2.])}), fit_result_dataset(covariance())])
    source = dict(varyList=["0::A1"], variables=[3.], sig=[0.1], covMatrix=[[0.01]])
    s.gpx = {"Covariance": {"data": source}}
    result = s._store_fit_results(s.ds)
    assert result.sizes["gsas_parameter"] == 1
    assert result.gsas_parameter.item() == "0::A1"
    assert result.gsas_parameter_value.item() == 3.
    assert clear_fit_results(result).identical(s.ds[["i1d"]].drop_attrs())


def test_empty_parameter_result():
    result = fit_result_dataset(dict(varyList=[], variables=0, sig=[], covMatrix=[]))
    assert result.sizes["gsas_parameter"] == 0
    assert result.gsas_covariance_status == "no varied parameters"


def test_logging_is_configurable_and_does_not_modify_root():
    logger = logging.getLogger("easyxrd")
    state = logger.handlers[:], logger.level, logger.propagate
    root_handlers = logging.getLogger().handlers[:]
    stream = io.StringIO()
    try:
        configure_logging("INFO", stream)
        configure_logging("INFO", stream)
        child = logging.getLogger("easyxrd.refinement")
        child.info("accepted test step")
        assert stream.getvalue().count("accepted test step") == 1
        configure_logging("ERROR", stream)
        child.warning("rejected test step")
        assert "rejected test step" not in stream.getvalue()
        assert logging.getLogger().handlers == root_handlers
    finally:
        for handler in logger.handlers:
            if handler not in state[0]:
                handler.close()
        logger.handlers, logger.level, logger.propagate = state


def test_real_gsas_parameter_export_and_rejection_retains_results(tmp_path):
    s = real_sample(tmp_path)
    s.setup_gsas2_refiner(plot=False)
    s.set_LeBail(False, plot=False)
    for enabled in (True, False):
        s.set_instrument_parameters_refinement(("U", "V"), set_refine_to=enabled, save_gpx=False)
        instrument = s.gpx.histograms()[0].data["Instrument Parameters"][0]
        assert all(instrument[key][2] == enabled for key in ("U", "V"))
        assert not instrument["W"][2]
    s.refine_background(num_coeffs=3, min_gof_improvement=None, plot=False)
    source = s.gpx["Covariance"]["data"]
    assert len(source["varyList"]) > 0
    assert s.ds.gsas_parameter.values.tolist() == source["varyList"]
    np.testing.assert_allclose(s.ds.gsas_parameter_value, source["variables"])
    np.testing.assert_allclose(s.ds.gsas_parameter_su, source["sig"])
    np.testing.assert_allclose(s.ds.gsas_covariance, source["covMatrix"])
    provenance = json.loads(s.ds.attrs["refinement_provenance"])
    assert provenance["weighting"] == "measurement_variance"
    assert provenance["instrument_source"] is None
    s.export_ds_to(tmp_path / "fit.nc")
    loaded = xr.load_dataset(tmp_path / "fit.nc")
    for name in ("gsas_covariance", "gsas_parameter_su", "gsas_parameter_value", "gsas_converged"):
        xr.testing.assert_identical(loaded[name], s.ds[name])
    before = s.ds[["gsas_covariance", "gsas_parameter_su", "gsas_parameter_value"]].copy(deep=True)
    step = s.ds.attrs["gsas_result_step"]
    s.refine_background(num_coeffs=5, min_gof_improvement=1e9, plot=False)
    assert not s.last_refinement_accepted
    for name in before:
        xr.testing.assert_identical(s.ds[name], before[name])
    assert s.ds.attrs["gsas_result_step"] == step
    s.get_baseline(plot=False)
    assert "gsas_parameter" not in s.ds.dims
    assert "gsas_converged" not in s.ds
    assert "gsas_result_step" not in s.ds.attrs


@pytest.mark.parametrize("wrapper,expected_copies", [(False, 1), (True, 2)])
def test_accepted_step_reuses_initial_transaction_checkpoint(tmp_path, monkeypatch, wrapper, expected_copies):
    s = configurable_sample(tmp_path)
    copies = []
    original = copy.deepcopy

    def counted(value, *args, **kwargs):
        if isinstance(value, Project):
            copies.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", counted)
    refine = s.refine

    def without_result_extraction(**kwargs):
        kwargs.update(update_ds=False, update_ds_phases=False, update_phases=False)
        return refine(**kwargs)

    monkeypatch.setattr(s, "refine", without_result_extraction)
    if wrapper:
        s.refine_background(plot=False)
    else:
        s.refine()
    assert s.last_refinement_accepted
    assert len(copies) == expected_copies
    assert not hasattr(s, "_transaction_checkpoint")


def test_rejection_checkpoint_is_protected_from_failed_save(tmp_path, monkeypatch):
    s = configurable_sample(tmp_path)
    s.gpx.outcome_gof = 1.999
    original = copy.deepcopy(s.gpx.data)
    save = s.gpx_saver
    attempts = []

    def fail_first_save():
        attempts.append(1)
        if len(attempts) == 1:
            s.gpx.data["Covariance"]["data"]["Rvals"]["GOF"] = 999.
            raise RuntimeError("save failed after mutation")
        save()

    monkeypatch.setattr(s, "gpx_saver", fail_first_save)
    with pytest.raises(RuntimeError, match="save failed"):
        s.refine(min_gof_improvement=1.)
    assert s.gpx.data == original
    assert not hasattr(s, "_transaction_checkpoint")
