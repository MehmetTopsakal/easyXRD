"""Acceptance and rollback tests without running the GSAS-II optimizer."""

import copy
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from easyxrd import exrd


@pytest.fixture(autouse=True)
def stub_lattice_for_wrapper_unit_tests(monkeypatch):
    original = __import__("easyxrd.refinement", fromlist=["load_gsas_module"]).load_gsas_module
    def module(name, *args):
        if name == "GSASIIlattice":
            return SimpleNamespace(GenSHCoeffT=lambda symmetry, sample, order, flag: ["C(2,0)"])
        return original(name, *args)
    monkeypatch.setattr("easyxrd.refinement.load_gsas_module", module)


def phase(name="Ni", uiso=0.01, *, ct=1, cia=9, mode="I"):
    atom = [0] * (cia + 2)
    atom[ct - 1] = "Ni1"
    atom[cia] = mode
    atom[cia + 1] = uiso
    return SimpleNamespace(
        name=name,
        data={"General": {"AtomPtrs": [ct + 2, ct, cia - 2, cia]},
              "Atoms": [atom]},
    )


class Histogram:
    def __init__(self, name="PWDR data.xy"):
        self.name = name
        self.data = {
            "Instrument Parameters": [{
                **{name: [1., 1., False] for name in ("U", "V", "W", "X", "Y", "Z")},
                "Type": ["PXC", "PXC", False],
            }, {}],
            "Limits": [[5., 60.], [5., 60.]],
        }

    def __getitem__(self, key):
        return self.data[key]


class Project:
    """Model GSAS-II's mutation of both its project and its saved file."""

    def __init__(self, filename, phases, outcomes):
        self.filename = filename
        self._phases = phases
        self._histograms = [Histogram()]
        self.outcomes = outcomes
        self.data = {"Covariance": {"data": {"Rvals": {"Rwp": 12., "GOF": 2.}}}}

    def phases(self):
        return self._phases

    def histograms(self):
        return self._histograms

    def __getitem__(self, key):
        return self.data[key]

    def set_refinement(self, settings, **kwargs):
        self.data.setdefault("settings", []).append((settings, kwargs))

    def save(self):
        self.filename.write_text(json.dumps({
            "phases": [p.data for p in self._phases], "data": self.data,
            "histograms": [{"name": h.name, "data": h.data} for h in self._histograms],
        }))

    def refine(self):
        # GSAS-II may mutate nested fit statistics already referenced by an
        # easyXRD dataset, even when dataset updates are disabled.
        self.data["Covariance"]["data"]["Rvals"].get("lastShifts", {}).update(
            {"scale": 0.25}
        )
        for p, values in zip(self._phases, self.outcomes):
            cia = p.data["General"]["AtomPtrs"][3]
            for atom, value in zip(p.data["Atoms"], values):
                atom[cia + 1] = value
        for histogram, values in zip(self._histograms, getattr(self, "instrument_outcomes", [])):
            for name, value in values.items():
                histogram["Instrument Parameters"][0][name][1] = value
        self.data["Covariance"]["data"]["Rvals"] = {
            "Rwp": 1., "GOF": getattr(self, "outcome_gof", 1.)
        }
        self.save()


def sample(tmp_path, phases, outcomes):
    result = exrd()
    result.gpx = Project(tmp_path / "gsas.gpx", phases, outcomes)
    result.gpx.save()
    result.ds = xr.Dataset({"i1d": ("radial", [1., 2.]),
                            "i1d_refined": ("radial", [0.9, 1.9])},
                           attrs={"Rwp": 12., "PhaseInd_0_cif": "accepted CIF"})
    result.phases = {"Ni": "accepted structure"}
    result.ds_previous = object()
    result.gpx_previous = object()
    result.phases_previous = object()
    return result


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("update_previous", [False, True])
def test_negative_uiso_rolls_back_entire_result(tmp_path, caplog, verbose, update_previous):
    other = phase("other", ct=3, cia=11)
    # The negative site is the second atom of the second phase, using shifted
    # offsets; the first atom and all atoms in the first phase are valid.
    other.data["Atoms"].append(copy.deepcopy(other.data["Atoms"][0]))
    other.data["Atoms"][1][2] = "Ni2"
    s = sample(tmp_path, [phase(), other], [[0.02], [0.03, -0.001]])
    old_ds = s.ds.copy(deep=True)
    old_phases = copy.deepcopy(s.phases)
    old_histories = (s.ds_previous, s.gpx_previous, s.phases_previous)
    old_file = s.gpx.filename.read_text()

    # Default update flags would require histogram/CIF access. Rejection must
    # return before any such access, as well as before replacing the histories.
    message = s.refine(verbose=verbose, update_previous_ds=update_previous,
                       update_previous_gpx=update_previous,
                       update_previous_phases=update_previous)

    assert not s.last_refinement_accepted
    assert "discarded" in message and "other/Ni2: Uiso=-0.001" in message
    assert s.gpx.filename.read_text() == old_file
    assert s.gpx.phases()[0].data["Atoms"][0][10] == 0.01
    assert s.gpx["Covariance"]["data"]["Rvals"]["Rwp"] == 12.
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), old_ds)
    assert s.ds.refinement_status.item() == "rejected"
    assert s.phases == old_phases
    assert (s.ds_previous, s.gpx_previous, s.phases_previous) == old_histories
    assert "discarded" in caplog.text


@pytest.mark.parametrize("uiso", [0., 0.02, -1.e-15])
def test_uiso_boundary_and_disabled_updates(tmp_path, uiso):
    s = sample(tmp_path, [phase()], [[uiso]])
    message = s.refine(update_ds=False, update_ds_phases=False, update_phases=False,
                       update_previous_ds=False, update_previous_gpx=False,
                       update_previous_phases=False)
    assert s.last_refinement_accepted == (uiso >= 0)
    if uiso < 0:
        assert "discarded" in message
        assert s.gpx.phases()[0].data["Atoms"][0][10] == 0.01
    else:
        assert "Rwp/GoF" in message
        assert s.gpx.phases()[0].data["Atoms"][0][10] == uiso


def test_unused_anisotropic_uiso_is_ignored(tmp_path):
    s = sample(tmp_path, [phase(mode="A")], [[-1.]])
    assert s._refine_gpx(copy.deepcopy(s.gpx)) is None
    assert s.last_refinement_accepted


def test_rejection_does_not_block_later_valid_refinement(tmp_path):
    s = sample(tmp_path, [phase()], [[-0.01]])
    assert s._refine_gpx(copy.deepcopy(s.gpx)) is not None
    s.gpx.outcomes = [[0.02]]
    assert s._refine_gpx(copy.deepcopy(s.gpx)) is None
    assert s.last_refinement_accepted
    assert s.gpx.phases()[0].data["Atoms"][0][10] == 0.02


def test_rejection_restores_shared_nested_fit_metadata(tmp_path):
    s = sample(tmp_path, [phase()], [[-0.01]])
    shared_shifts = {"scale": 0.001}
    s.gpx["Covariance"]["data"]["Rvals"]["lastShifts"] = shared_shifts
    s.ds.attrs["lastShifts"] = shared_shifts
    s.ds_previous = s.ds.copy(deep=False)
    before_ds = s.ds.copy(deep=True)
    before_history = s.ds_previous.copy(deep=True)
    history = s.ds_previous
    s.refine(update_ds=False, update_ds_phases=False, update_phases=False,
             update_previous_ds=False, update_previous_gpx=False,
             update_previous_phases=False)
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), before_ds)
    assert s.ds.refinement_status.item() == "rejected"
    xr.testing.assert_identical(s.ds_previous, before_history)
    assert s.ds_previous is history


def test_rejected_plot_uses_restored_dataset(tmp_path, monkeypatch):
    s = sample(tmp_path, [phase()], [[-0.01]])
    plotted = []
    monkeypatch.setattr("easyxrd.refinement.exrd_plotter", lambda **kw: plotted.append(kw))
    message = s.refine(plot=True)
    assert plotted[0]["ds"] is s.ds
    assert plotted[0]["title_str"] == message
    assert "discarded" in s._report_refinement("Site is refined", message)


@pytest.mark.parametrize("method", ["fine_tune_gpx", "replace_gpx_with"])
def test_direct_entry_points_restore_original_project(tmp_path, monkeypatch, method):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gsasii_run_directory = str(tmp_path)
    s.gsasii_lib_path = str(tmp_path)
    before = s.gpx.filename.read_text()
    candidate = Project(s.gpx.filename, [phase(uiso=0.03)], [[-0.01]])
    monkeypatch.setattr("easyxrd.refinement.load_gsas_module",
                        lambda *a: SimpleNamespace(G2Project=lambda **kw: candidate))
    monkeypatch.setattr("easyxrd.refinement.subprocess.check_call", lambda *a: None)
    if method == "fine_tune_gpx":
        s.fine_tune_gpx()
    else:
        source = tmp_path / "candidate.gpx"
        source.write_text("candidate project")
        s.replace_gpx_with(source)
    assert not s.last_refinement_accepted
    assert s.gpx.phases()[0].data["Atoms"][0][10] == 0.01
    assert s.gpx.filename.read_text() == before


@pytest.mark.parametrize("gof", [2., 2.1, 1.9999, float("nan"), float("inf")])
def test_insufficient_improvement_preserves_all_state(tmp_path, gof):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.outcome_gof = gof
    shared = {"scale": 0.001}
    s.gpx["Covariance"]["data"]["Rvals"]["lastShifts"] = shared
    s.ds.attrs["lastShifts"] = shared
    s.ds_previous = s.ds.copy(deep=True)
    s.gpx.save()
    before_file = s.gpx.filename.read_bytes()
    before_ds = s.ds.copy(deep=True)
    before_previous = s.ds_previous.copy(deep=True)
    histories = (s.ds_previous, s.gpx_previous, s.phases_previous)
    phases = copy.deepcopy(s.phases)

    message = s.refine(min_gof_improvement=1.)

    assert not s.last_refinement_accepted
    assert "insufficient GoF improvement" in message
    assert s.gpx.filename.read_bytes() == before_file
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), before_ds)
    assert s.ds.refinement_status.item() == "rejected"
    xr.testing.assert_identical(s.ds_previous, before_previous)
    assert s.phases == phases
    assert all(a is b for a, b in zip(
        histories, (s.ds_previous, s.gpx_previous, s.phases_previous)))


@pytest.mark.parametrize("minimum,gof,accepted", [
    (1., 1.98, True), (1., 1.99, False), (0., 1.9999, True),
    (0., 2., False), (None, 2.1, True),
])
def test_improvement_threshold(tmp_path, minimum, gof, accepted):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.outcome_gof = gof
    s.refine(min_gof_improvement=minimum, update_ds=False,
             update_ds_phases=False, update_phases=False,
             update_previous_ds=False, update_previous_gpx=False,
             update_previous_phases=False)
    assert s.last_refinement_accepted is accepted


class SitePhase:
    def __init__(self):
        self.name = "Ni"
        self.data = phase().data
        self.data["Atoms"][0][1] = "X"

    def __getitem__(self, key):
        return self.data[key]

    def atom(self, label):
        return SiteAtom(self.data["Atoms"][0])


class SiteAtom:
    def __init__(self, data):
        self.data = data

    @property
    def refinement_flags(self):
        return self.data[1]

    @refinement_flags.setter
    def refinement_flags(self, value):
        self.data[1] = value


@pytest.mark.parametrize("clear_flags", [False, True])
def test_site_rejection_restores_original_flags(tmp_path, clear_flags):
    s = sample(tmp_path, [SitePhase()], [[0.02]])
    s.gpx.outcome_gof = 1.9999
    before = s.gpx.filename.read_bytes()
    old_ds = s.ds.copy(deep=True)
    histories = (s.ds_previous, s.gpx_previous, s.phases_previous)
    s.refine_site_property(refinement_flags="U",
                           set_to_false_after_refinement=clear_flags)
    assert not s.last_refinement_accepted
    assert s.gpx.phases()[0].atom("Ni1").refinement_flags == "X"
    assert s.gpx.filename.read_bytes() == before
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), old_ds)
    assert s.ds.refinement_status.item() == "rejected"
    assert all(a is b for a, b in zip(
        histories, (s.ds_previous, s.gpx_previous, s.phases_previous)))


@pytest.mark.parametrize("minimum", [-1., float("nan"), float("inf")])
def test_site_invalid_threshold_does_not_change_flags(tmp_path, minimum):
    s = sample(tmp_path, [SitePhase()], [[0.02]])
    before = s.gpx.filename.read_bytes()
    with pytest.raises(ValueError, match="min_gof_improvement"):
        s.refine_site_property(refinement_flags="U", min_gof_improvement=minimum)
    assert s.gpx.phases()[0].atom("Ni1").refinement_flags == "X"
    assert s.gpx.filename.read_bytes() == before


REFINEMENT_METHODS = [
    "refine_background", "refine_instrument_parameters", "refine_cell_parameters",
    "refine_strain_broadening", "refine_size_broadening", "refine_phase_fractions",
    "refine_preferred_orientation", "refine_site_property", "set_LeBail",
]


def configurable_sample(tmp_path):
    p = SitePhase()
    p.data["General"]["SGData"] = {"SGLaue": "-1"}
    p.data["Histograms"] = {"PWDR data.xy": {
        "Scale": [1., False],
        "Pref.Ori.": ["MD", 1., False, [0, 0, 1], 0, {}, [], 0.1],
    }}
    s = sample(tmp_path, [p], [[0.02]])
    s.gpx.data["Phases"] = {p.name: p.data}
    s.gpx.data["PWDR data.xy"] = {"Sample Parameters": {"Scale": [1., True]}}
    s.gpx.save()
    return s


@pytest.mark.parametrize("method", REFINEMENT_METHODS)
@pytest.mark.parametrize("minimum,gof", [(1., 1.99), (2., 1.97)])
@pytest.mark.parametrize("plot", [False, True])
def test_all_wrappers_restore_full_state(tmp_path, monkeypatch, caplog, method, minimum, gof, plot):
    s = configurable_sample(tmp_path)
    s.gpx.outcome_gof = gof
    before_file = s.gpx.filename.read_bytes()
    before_ds = s.ds.copy(deep=True)
    before_phases = copy.deepcopy(s.phases)
    history = (s.gpx_previous, s.ds_previous, s.phases_previous)
    plotted = []
    monkeypatch.setattr("easyxrd.refinement.exrd_plotter", lambda **kw: plotted.append(kw))
    # Explicit thresholds keep this check independent of user-configured defaults.
    result = getattr(s, method)(plot=plot, min_gof_improvement=minimum)
    assert result is None  # No second copy for Jupyter's result display hook.
    assert not s.last_refinement_accepted
    printed = caplog.text
    assert printed.count("Refinement discarded") == 1
    assert "insufficient GoF improvement" in printed
    assert s.gpx.filename.read_bytes() == before_file
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), before_ds)
    assert s.ds.refinement_status.item() == "rejected"
    assert s.phases == before_phases
    assert all(a is b for a, b in zip(history,
        (s.gpx_previous, s.ds_previous, s.phases_previous)))
    assert s.ds.refinement_label.item() == ("set_LeBail:True" if method == "set_LeBail" else method)
    assert plotted == []
    assert s.last_refinement_rejection_reason == "insufficient_gof_improvement"


@pytest.mark.parametrize("minimum,gof", [(1., 1.99), (0., 2.), (1., float("nan"))])
def test_direct_refine_skips_plot_for_insufficient_improvement(tmp_path, monkeypatch, minimum, gof):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.outcome_gof = gof
    plotted = []
    monkeypatch.setattr("easyxrd.refinement.exrd_plotter", lambda **kw: plotted.append(kw))
    message = s.refine(plot=True, min_gof_improvement=minimum)
    assert "insufficient GoF improvement" in message
    assert not s.last_refinement_accepted
    assert plotted == []


def test_accepted_refinement_still_plots_after_rejection(tmp_path, monkeypatch):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.outcome_gof = 1.99
    plotted = []
    monkeypatch.setattr("easyxrd.refinement.exrd_plotter", lambda **kw: plotted.append(kw))
    s.refine(plot=True, min_gof_improvement=1.)
    assert plotted == []
    s.gpx.outcome_gof = 1.5
    s.refine(plot=True, min_gof_improvement=1., update_ds=False,
             update_ds_phases=False, update_phases=False)
    assert s.last_refinement_accepted
    assert s.last_refinement_rejection_reason is None
    assert len(plotted) == 1


@pytest.mark.parametrize("method", REFINEMENT_METHODS)
def test_invalid_threshold_precedes_wrapper_mutations(tmp_path, method):
    s = configurable_sample(tmp_path)
    before = copy.deepcopy(s.gpx.data)
    with pytest.raises(ValueError, match="min_gof_improvement"):
        getattr(s, method)(min_gof_improvement=-1.)
    assert s.gpx.data == before


@pytest.mark.parametrize("method", REFINEMENT_METHODS)
@pytest.mark.parametrize("minimum", [None, 0., 0.1])
def test_wrappers_forward_override_on_accepted_path(tmp_path, monkeypatch, method, minimum):
    s = configurable_sample(tmp_path)
    calls = []
    def accepted(**kwargs):
        calls.append(kwargs)
        s.last_refinement_accepted = True
        return "accepted"
    monkeypatch.setattr(s, "refine", accepted)
    getattr(s, method)(min_gof_improvement=minimum)
    assert len(calls) == 1
    assert calls[0]["min_gof_improvement"] == minimum


@pytest.mark.parametrize("method", ["fine_tune_gpx", "replace_gpx_with"])
def test_direct_entry_points_reject_insufficient_improvement(tmp_path, monkeypatch, method):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gsasii_run_directory = str(tmp_path)
    s.gsasii_lib_path = str(tmp_path)
    before = s.gpx.filename.read_bytes()
    candidate = Project(s.gpx.filename, [phase(uiso=0.03)], [[0.04]])
    candidate.outcome_gof = 1.99
    monkeypatch.setattr("easyxrd.refinement.load_gsas_module",
                        lambda *a: SimpleNamespace(G2Project=lambda **kw: candidate))
    monkeypatch.setattr("easyxrd.refinement.subprocess.check_call", lambda *a: None)
    if method == "fine_tune_gpx":
        s.fine_tune_gpx(min_gof_improvement=1.)
    else:
        source = tmp_path / "candidate.gpx"
        source.write_text("candidate project")
        s.replace_gpx_with(source, min_gof_improvement=1.)
    assert not s.last_refinement_accepted
    assert s.gpx.filename.read_bytes() == before
    assert s.ds.refinement_label.item() == ("set_LeBail:True" if method == "set_LeBail" else method)


def refine_without_updates(s, **kwargs):
    return s.refine(update_ds=False, update_ds_phases=False, update_phases=False,
                    update_previous_ds=False, update_previous_gpx=False,
                    update_previous_phases=False, **kwargs)


def test_rwp_history_records_retained_and_attempted_values(tmp_path):
    s = sample(tmp_path, [phase()], [[-0.01]])
    before = s.ds.copy(deep=True)
    refine_without_updates(s, log_label="trial Uiso")
    s.gpx.outcomes = [[0.02]]
    refine_without_updates(s, log_label="valid Uiso")
    refine_without_updates(s)

    np.testing.assert_equal(s.ds.refinement_step.values, [1, 2, 3])
    np.testing.assert_equal(s.ds.rwp_history.values, [12., 1., 1.])
    np.testing.assert_equal(s.ds.rwp_attempted_history.values, [1., 1., 1.])
    assert s.ds.refinement_status.values.tolist() == ["rejected", "accepted", "accepted"]
    assert s.ds.refinement_reason.values.tolist() == ["negative_uiso", "", ""]
    assert s.ds.refinement_label.values.tolist() == ["trial Uiso", "valid Uiso", "refine"]
    assert s.ds.rwp_history.attrs["units"] == "%"
    times = [datetime.fromisoformat(t) for t in s.ds.refinement_timestamp.values]
    assert times == sorted(times)
    assert all(t.utcoffset().total_seconds() == 0 for t in times)
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), before)


def test_rwp_history_logs_optimizer_error_and_reraises(tmp_path, monkeypatch):
    s = sample(tmp_path, [phase()], [[0.02]])
    def fail():
        raise RuntimeError("optimizer failed")
    monkeypatch.setattr(s.gpx, "refine", fail)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        refine_without_updates(s)
    assert np.isnan(s.ds.rwp_history.item())
    assert np.isnan(s.ds.rwp_attempted_history.item())
    assert s.ds.refinement_status.item() == "error"
    assert s.ds.refinement_reason.item() == "RuntimeError: optimizer failed"


@pytest.mark.parametrize("initial_status", [None, "accepted", "rejected", "error"])
def test_rwp_history_netcdf_round_trip_and_continue(tmp_path, monkeypatch, initial_status):
    s = sample(tmp_path, [phase()], [[0.02]])
    s._reset_rwp_history()
    if initial_status == "rejected":
        s.gpx.outcomes = [[-0.01]]
    if initial_status == "error":
        with monkeypatch.context() as patch:
            def fail():
                raise RuntimeError("optimizer failed")
            patch.setattr(s.gpx, "refine", fail)
            with pytest.raises(RuntimeError, match="optimizer failed"):
                refine_without_updates(s)
    elif initial_status is not None:
        refine_without_updates(s)
    s.gpx.outcomes = [[0.02]]
    before = s.ds.copy(deep=True)
    path = tmp_path / "history.nc"
    s.export_ds_to(path)
    s.load_xrd_data(from_nc_file=path, plot=False)
    xr.testing.assert_identical(s.ds, before)
    refine_without_updates(s)
    count = int(initial_status is not None) + 1
    assert s.ds.sizes["refinement_step"] == count
    np.testing.assert_equal(s.ds.refinement_step.values, np.arange(1, count + 1))
    assert s.ds.rwp_history.values[-1] == 1.
    if initial_status is not None:
        assert s.ds.refinement_status.values[0] == initial_status


def test_rwp_history_reset_preserves_profile_and_is_per_instance(tmp_path):
    s = sample(tmp_path, [phase()], [[0.02]])
    other = sample(tmp_path, [phase()], [[0.02]])
    before = s.ds.copy(deep=True)
    refine_without_updates(s)
    assert "rwp_history" not in other.ds
    s._reset_rwp_history()
    assert s.ds.sizes["refinement_step"] == 0
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), before)
    refine_without_updates(s)
    assert s.ds.refinement_step.item() == 1


VALID_PROFILE = dict(U=1., V=0., W=1., X=1., Y=0., Z=0.)


@pytest.mark.parametrize("component", ["Gaussian variance", "Lorentzian width"])
@pytest.mark.parametrize("value", [-1., -1.e-15])
@pytest.mark.parametrize("minimum", [None, 1.])
def test_negative_peak_width_restores_entire_result(tmp_path, component, value, minimum):
    s = sample(tmp_path, [phase()], [[0.02]])
    bad = {"U": 0., "V": 0., "W": value} if component == "Gaussian variance" else {"X": 0., "Y": 0., "Z": value}
    s.gpx.instrument_outcomes = [VALID_PROFILE | bad]
    shared_shifts = {"scale": 0.001}
    s.gpx["Covariance"]["data"]["Rvals"]["lastShifts"] = shared_shifts
    s.ds.attrs["lastShifts"] = shared_shifts
    s.ds_previous = s.ds.copy(deep=True)
    s.gpx.save()
    old_file = s.gpx.filename.read_bytes()
    old_ds = s.ds.copy(deep=True)
    old_previous = s.ds_previous.copy(deep=True)
    old_phases = copy.deepcopy(s.phases)
    histories = (s.ds_previous, s.gpx_previous, s.phases_previous)

    message = s.refine(min_gof_improvement=minimum)

    assert not s.last_refinement_accepted
    assert s.last_refinement_rejection_reason == "invalid_peak_width"
    assert f"PWDR data.xy/{component}={value:.6g}" in message
    assert "2theta=" in message
    assert s.gpx.filename.read_bytes() == old_file
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), old_ds)
    xr.testing.assert_identical(s.ds_previous, old_previous)
    assert s.phases == old_phases
    assert all(a is b for a, b in zip(histories,
        (s.ds_previous, s.gpx_previous, s.phases_previous)))
    assert s.ds.refinement_status.item() == "rejected"
    assert s.ds.refinement_reason.item() == "invalid_peak_width"
    assert s.ds.rwp_history.item() == 12.
    assert s.ds.rwp_attempted_history.item() == 1.


@pytest.mark.parametrize("changes", [
    {"U": -0.1}, {"V": -1.}, {"W": -0.001},
    {"X": -0.1, "Z": 1.}, {"Y": -0.1}, {"Z": -0.1},
])
def test_negative_coefficients_with_valid_widths_are_accepted(tmp_path, changes):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.instrument_outcomes = [VALID_PROFILE | changes]
    refine_without_updates(s)
    assert s.last_refinement_accepted
    assert s.last_refinement_rejection_reason is None


@pytest.mark.parametrize("zero_terms", [
    {"U": 0., "V": 0., "W": 0.}, {"X": 0., "Y": 0., "Z": 0.},
])
def test_zero_width_components_are_accepted(tmp_path, zero_terms):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.instrument_outcomes = [VALID_PROFILE | zero_terms]
    refine_without_updates(s)
    assert s.last_refinement_accepted


@pytest.mark.parametrize("changes,component", [
    ({"U": 1., "V": -0.4, "W": 0.0399}, "Gaussian variance"),
    ({"X": 1., "Y": -0.5, "Z": -0.8661}, "Lorentzian width"),
])
def test_negative_interior_minimum_is_rejected_even_with_positive_endpoints(tmp_path, changes, component):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.histograms()[0].data["Limits"] = [[5., 100.], [5., 100.]]
    profile = VALID_PROFILE | changes
    theta = np.deg2rad(np.array([5., 100.]) / 2)
    if component == "Gaussian variance":
        endpoints = profile["U"] * np.tan(theta)**2 + profile["V"] * np.tan(theta) + profile["W"]
    else:
        endpoints = profile["X"] / np.cos(theta) + profile["Y"] * np.tan(theta) + profile["Z"]
    assert (endpoints > 0).all()
    s.gpx.instrument_outcomes = [profile]
    message = refine_without_updates(s)
    assert not s.last_refinement_accepted
    assert component in message


def test_negative_width_outside_fitted_limits_does_not_reject(tmp_path):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.histograms()[0].data["Limits"] = [[1., 150.], [5., 10.]]
    # This Gaussian is positive at 5–10 degrees but negative near 53 degrees.
    s.gpx.instrument_outcomes = [VALID_PROFILE | {"U": 1., "V": -1., "W": 0.1}]
    refine_without_updates(s)
    assert s.last_refinement_accepted


@pytest.mark.parametrize("changes", [{"W": float("nan")}, {"X": float("inf")}])
def test_nonfinite_width_is_rejected(tmp_path, changes):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.instrument_outcomes = [VALID_PROFILE | changes]
    message = refine_without_updates(s)
    assert not s.last_refinement_accepted
    assert "nonfinite" in message
    assert s.last_refinement_rejection_reason == "invalid_peak_width"


def test_checks_all_histograms_and_reports_both_width_components(tmp_path):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx._histograms.append(Histogram("PWDR second"))
    s.gpx.instrument_outcomes = [{}, dict(U=0., V=0., W=-0.2, X=0., Y=0., Z=-0.3)]
    s.gpx.save()
    old_file = s.gpx.filename.read_bytes()
    message = refine_without_updates(s)
    assert "PWDR second/Gaussian variance=-0.2" in message
    assert "PWDR second/Lorentzian width=-0.3" in message
    assert s.gpx.filename.read_bytes() == old_file


def test_checks_current_values_and_ignores_initial_values(tmp_path):
    s = sample(tmp_path, [phase()], [[0.02]])
    parameters = s.gpx.histograms()[0]["Instrument Parameters"][0]
    parameters["W"][0] = -1.
    refine_without_updates(s)
    assert s.last_refinement_accepted
    # Fixed parameters still determine the resulting width.
    parameters = s.gpx.histograms()[0]["Instrument Parameters"][0]
    parameters["W"][1] = -10.
    message = refine_without_updates(s)
    assert "Gaussian variance" in message
    assert not s.last_refinement_accepted


def test_z_is_optional_for_older_cw_instruments(tmp_path):
    s = sample(tmp_path, [phase()], [[0.02]])
    del s.gpx.histograms()[0]["Instrument Parameters"][0]["Z"]
    refine_without_updates(s)
    assert s.last_refinement_accepted


@pytest.mark.parametrize("histogram_type", ["PNT", "PXE"])
def test_other_profile_types_do_not_use_cw_width_equations(tmp_path, histogram_type):
    s = sample(tmp_path, [phase()], [[0.02]])
    parameters = s.gpx.histograms()[0]["Instrument Parameters"][0]
    parameters["Type"] = [histogram_type, histogram_type, False]
    s.gpx.instrument_outcomes = [{"W": -10.}]
    refine_without_updates(s)
    assert s.last_refinement_accepted


def test_histograms_without_cw_profile_parameters_are_ignored(tmp_path):
    s = sample(tmp_path, [phase()], [[0.02]])
    single_crystal = Histogram("HKLF second")
    single_crystal.data = {}
    s.gpx._histograms.append(single_crystal)
    # Also allow a non-CW powder instrument that lacks U, V, W.
    s.gpx.histograms()[0]["Instrument Parameters"][0].clear()
    refine_without_updates(s)
    assert s.last_refinement_accepted


@pytest.mark.parametrize("method", ["refine_instrument_parameters", "refine_background"])
def test_peak_width_guard_restores_wrapper_flags(tmp_path, method):
    s = configurable_sample(tmp_path)
    s.gpx.instrument_outcomes = [{"W": -10.}]
    old_file = s.gpx.filename.read_bytes()
    old_ds = s.ds.copy(deep=True)
    getattr(s, method)(min_gof_improvement=None)
    assert not s.last_refinement_accepted
    assert s.gpx.filename.read_bytes() == old_file
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), old_ds)
    assert s.ds.refinement_reason.item() == "invalid_peak_width"


def test_valid_refinement_can_follow_invalid_peak_width(tmp_path):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gpx.instrument_outcomes = [{"W": -10.}]
    refine_without_updates(s)
    s.gpx.instrument_outcomes = [VALID_PROFILE | {"V": -1.}]
    refine_without_updates(s)
    assert s.last_refinement_accepted
    assert s.last_refinement_rejection_reason is None
    assert s.ds.refinement_status.values.tolist() == ["rejected", "accepted"]


@pytest.mark.parametrize("method", ["fine_tune_gpx", "replace_gpx_with"])
def test_direct_entry_points_reject_invalid_peak_width(tmp_path, monkeypatch, method):
    s = sample(tmp_path, [phase()], [[0.02]])
    s.gsasii_run_directory = str(tmp_path)
    s.gsasii_lib_path = str(tmp_path)
    old_file = s.gpx.filename.read_bytes()
    old_ds = s.ds.copy(deep=True)
    candidate = Project(s.gpx.filename, [phase(uiso=0.03)], [[0.04]])
    candidate.instrument_outcomes = [{"W": -10.}]
    monkeypatch.setattr("easyxrd.refinement.load_gsas_module",
                        lambda *a: SimpleNamespace(G2Project=lambda **kw: candidate))
    monkeypatch.setattr("easyxrd.refinement.subprocess.check_call", lambda *a: None)
    if method == "fine_tune_gpx":
        s.fine_tune_gpx(min_gof_improvement=None)
    else:
        source = tmp_path / "candidate.gpx"
        source.write_text("candidate project")
        s.replace_gpx_with(source, min_gof_improvement=None)
    assert not s.last_refinement_accepted
    assert s.gpx.filename.read_bytes() == old_file
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), old_ds)
    assert s.ds.refinement_reason.item() == "invalid_peak_width"
