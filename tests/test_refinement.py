"""Acceptance and rollback tests without running the GSAS-II optimizer."""

import copy
import json
from types import SimpleNamespace

import pytest
import xarray as xr

from easyxrd import exrd


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


class Project:
    """Model GSAS-II's mutation of both its project and its saved file."""

    def __init__(self, filename, phases, outcomes):
        self.filename = filename
        self._phases = phases
        self.outcomes = outcomes
        self.data = {"Covariance": {"data": {"Rvals": {"Rwp": 12., "GOF": 2.}}}}

    def phases(self):
        return self._phases

    def __getitem__(self, key):
        return self.data[key]

    def set_refinement(self, settings, **kwargs):
        self.data.setdefault("settings", []).append((settings, kwargs))

    def save(self):
        self.filename.write_text(json.dumps({
            "phases": [p.data for p in self._phases], "data": self.data,
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
def test_negative_uiso_rolls_back_entire_result(tmp_path, capsys, verbose, update_previous):
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
    xr.testing.assert_identical(s.ds, old_ds)
    assert s.phases == old_phases
    assert (s.ds_previous, s.gpx_previous, s.phases_previous) == old_histories
    assert "discarded" in capsys.readouterr().out


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
    xr.testing.assert_identical(s.ds, before_ds)
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
    monkeypatch.setitem(__import__("sys").modules, "GSASIIscriptable",
                        SimpleNamespace(G2Project=lambda **kw: candidate))
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
    xr.testing.assert_identical(s.ds, before_ds)
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
    xr.testing.assert_identical(s.ds, old_ds)
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
def test_all_wrappers_restore_full_state(tmp_path, monkeypatch, capsys, method, minimum, gof, plot):
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
    printed = capsys.readouterr().out
    assert printed.count("Refinement discarded") == 1
    assert "insufficient GoF improvement" in printed
    assert s.gpx.filename.read_bytes() == before_file
    xr.testing.assert_identical(s.ds, before_ds)
    assert s.phases == before_phases
    assert all(a is b for a, b in zip(history,
        (s.gpx_previous, s.ds_previous, s.phases_previous)))
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
    monkeypatch.setitem(__import__("sys").modules, "GSASIIscriptable",
                        SimpleNamespace(G2Project=lambda **kw: candidate))
    monkeypatch.setattr("easyxrd.refinement.subprocess.check_call", lambda *a: None)
    if method == "fine_tune_gpx":
        s.fine_tune_gpx()
    else:
        source = tmp_path / "candidate.gpx"
        source.write_text("candidate project")
        s.replace_gpx_with(source)
    assert not s.last_refinement_accepted
    assert s.gpx.filename.read_bytes() == before
