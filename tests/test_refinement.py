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
        self.data["Covariance"]["data"]["Rvals"] = {"Rwp": 1., "GOF": 1.}
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
