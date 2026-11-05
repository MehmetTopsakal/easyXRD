"""Independent lattice starts, winner retention and failure recovery."""

import copy

import numpy as np
import pytest
import xarray as xr

from tests.test_refinement import Project, phase, sample


def trial_sample(tmp_path, monkeypatch, outcomes):
    phases = [phase("first"), phase("second")]
    for i, p in enumerate(phases):
        p.data["General"]["Cell"] = [False, 4. + i, 5., 6., 80., 90., 100., 120.]
    s = sample(tmp_path, phases, [[0.02], [0.02]])
    s.phases = {p.name: tuple(p.data["General"]["Cell"][1:7]) for p in phases}
    s.gpx.save()
    starts = []

    def optimizer(project):
        cells = [copy.deepcopy(p.data["General"]["Cell"]) for p in project.phases()]
        starts.append(cells)
        # Also mutate an unrelated setting: later trials must not inherit it.
        assert "trial_marker" not in project.data
        project.data["trial_marker"] = len(starts)
        outcome = outcomes[len(starts) - 1]
        if isinstance(outcome, Exception):
            project.save()
            raise outcome
        rwp, gof, uiso = outcome[:3]
        for p in project.phases():
            p.data["Atoms"][0][10] = uiso
        if len(outcome) == 4:
            for index, value in outcome[3].items():
                project.phases()[0].data["General"]["Cell"][index] = value
        project.data["Covariance"]["data"]["Rvals"] = {"Rwp": rwp, "GOF": gof}
        project.save()

    monkeypatch.setattr(Project, "refine", optimizer)

    def refine(**kwargs):
        # Exercise the real optimizer acceptance and history path. Profile/CIF
        # synchronization is represented here; the real GSAS-II run covers it.
        previous = copy.deepcopy(s.gpx)
        old_ds = s.ds.copy(deep=True)
        old_phases = copy.deepcopy(s.phases)
        message = s._refine_gpx(previous, **kwargs)
        if message is not None:
            return message
        s.ds_previous, s.gpx_previous, s.phases_previous = old_ds, previous, old_phases
        rwp = s._project_rwp(s.gpx)
        s.ds["i1d_refined"] = ("radial", [rwp, rwp])
        s.ds.attrs["Rwp"] = rwp
        s.phases = {p.name: tuple(p.data["General"]["Cell"][1:7]) for p in s.gpx.phases()}
        return f"Rwp={rwp:g}"

    monkeypatch.setattr(s, "refine", refine)
    return s, starts


@pytest.mark.parametrize("phase_ind", ["all", None, 1])
@pytest.mark.parametrize("clear_flags", [True, False])
def test_scales_are_independent_and_best_result_is_restored(
    tmp_path, monkeypatch, phase_ind, clear_flags
):
    s, starts = trial_sample(tmp_path, monkeypatch,
                            [(4., 1., .02), (2., .8, .02), (3., .9, .02)])
    baseline = copy.deepcopy(s.gpx)
    baseline_ds = s.ds.copy(deep=True)
    plotted = []
    monkeypatch.setattr("easyxrd.refinement.exrd_plotter", lambda **kw: plotted.append(kw))

    s.refine_cell_parameters(scale_trials=[.98, .99, 1.01], phase_ind=phase_ind,
                             set_to_false_after_refinement=clear_flags, plot=True)

    assert s.cell_scale_best == .99
    assert s.last_refinement_accepted
    assert len(starts) == 3
    for cells, scale in zip(starts, [.98, .99, 1.01]):
        for index, cell in enumerate(cells):
            original = baseline.phases()[index].data["General"]["Cell"]
            factor = scale if phase_ind in (None, "all", index) else 1.
            np.testing.assert_allclose(cell[1:4], np.array(original[1:4]) * factor)
            assert cell[4:7] == original[4:7]
            assert cell[7] == pytest.approx(original[7] * factor ** 3)
    assert s.ds.attrs["Rwp"] == s._project_rwp(s.gpx) == 2.
    np.testing.assert_equal(s.ds.i1d_refined.values, [2., 2.])
    assert s.phases == {p.name: tuple(p.data["General"]["Cell"][1:7]) for p in s.gpx.phases()}
    assert s.gpx.data["trial_marker"] == 2
    assert s.gpx.filename.read_text().find('"Rwp": 2.0') >= 0
    assert s.gpx_previous.data == baseline.data
    xr.testing.assert_identical(s.ds_previous.drop_dims("refinement_step"), baseline_ds)
    assert s.ds.attrs["PhaseInd_0_cell_a_previous"] == 4.
    np.testing.assert_equal(s.ds.rwp_attempted_history.values, [4., 2., 3.])
    np.testing.assert_equal(s.ds.rwp_history.values, [4., 2., 2.])
    assert list(s.ds.refinement_status.values) == ["accepted", "accepted", "rejected"]
    assert s.ds.refinement_reason.values[-1] == "no_rwp_improvement"
    assert "scale=0.99" in s.ds.refinement_label.values[1]
    assert s.gpx.data["settings"][-1][0]["set"]["Cell"] is (not clear_flags)
    assert len(plotted) == 1 and plotted[0]["ds"] is s.ds


@pytest.mark.parametrize("outcomes", [
    [(1., 2.1, .02), (2., 1., -.01)],  # GoF and Uiso guards
    [(13., 1., .02), (12., 1., .02)],  # no Rwp decrease
    [RuntimeError("bad start"), ValueError("singular matrix")],
    [(float("nan"), 1., .02), (float("inf"), 1., .02)],
])
def test_no_winner_restores_original_fit_and_comparison_snapshots(tmp_path, monkeypatch, outcomes):
    s, _ = trial_sample(tmp_path, monkeypatch, outcomes)
    before_ds = s.ds.copy(deep=True)
    before_file = s.gpx.filename.read_bytes()
    previous = (s.ds_previous, s.gpx_previous, s.phases_previous)
    plotted = []
    monkeypatch.setattr("easyxrd.refinement.exrd_plotter", lambda **kw: plotted.append(kw))

    s.refine_cell_parameters(scale_trials=[.99, 1.01], plot=True)

    assert s.cell_scale_best is None
    assert not s.last_refinement_accepted
    assert s.gpx.filename.read_bytes() == before_file
    xr.testing.assert_identical(s.ds.drop_dims("refinement_step"), before_ds)
    assert all(a is b for a, b in zip(previous,
        (s.ds_previous, s.gpx_previous, s.phases_previous)))
    assert s.ds.sizes["refinement_step"] == 2
    assert plotted == []
    assert all(row["status"] != "accepted" for row in s.cell_scale_trials)


def test_later_exception_cannot_replace_winner_and_history_can_continue(tmp_path, monkeypatch):
    s, _ = trial_sample(tmp_path, monkeypatch,
                        [(3., 1., .02), RuntimeError("singular matrix"), (2., .8, .02)])
    s.refine_cell_parameters(scale_trials=iter([.99, 1.01]))
    assert s.cell_scale_best == .99
    assert s._project_rwp(s.gpx) == 3.
    assert list(s.ds.refinement_status.values) == ["accepted", "error"]
    assert "RuntimeError: singular matrix" in s.ds.refinement_reason.values[-1]
    assert s.gpx.data["trial_marker"] == 1
    # A subsequent call must append rather than reset the existing audit log.
    # The stand-in optimizer expects independent states within each search.
    s.gpx.data.pop("trial_marker")
    s.refine_cell_parameters(scale_trials=[1.])
    assert s.ds.sizes["refinement_step"] == 3
    np.testing.assert_equal(s.ds.rwp_attempted_history.values, [3., np.nan, 2.])


def test_disabling_gof_guard_still_requires_rwp_improvement(tmp_path, monkeypatch):
    s, _ = trial_sample(tmp_path, monkeypatch, [(15., 3., .02)])
    s.refine_cell_parameters(scale_trials=[1.], min_gof_improvement=None)
    assert s.cell_scale_best is None
    assert s._project_rwp(s.gpx) == 12.


@pytest.mark.parametrize("bad_cell", [{1: -1.}, {2: np.nan}, {4: 200.}, {7: 0.}])
def test_invalid_cell_cannot_replace_valid_winner(tmp_path, monkeypatch, bad_cell):
    s, _ = trial_sample(tmp_path, monkeypatch,
                        [(3., 1., .02), (.1, .1, .02, bad_cell)])
    s.refine_cell_parameters(scale_trials=[.99, 1.01])
    assert s.cell_scale_best == .99
    assert s._project_rwp(s.gpx) == 3.
    assert s.cell_scale_trials[-1]["reason"] == "invalid_cell_or_rwp"
    assert s.ds.refinement_reason.values[-1] == "invalid_cell_or_rwp"
    np.testing.assert_equal(s.ds.rwp_history.values, [3., 3.])


@pytest.mark.parametrize("scales", [[], [0.], [-.1], [np.nan], [np.inf], [[1.]], 1., "abc"])
def test_invalid_scales_precede_all_mutations(tmp_path, monkeypatch, scales):
    s, starts = trial_sample(tmp_path, monkeypatch, [])
    before = s.gpx.filename.read_bytes()
    before_ds = s.ds.copy(deep=True)
    with pytest.raises(ValueError, match="scale_trials"):
        s.refine_cell_parameters(scale_trials=scales)
    assert s.gpx.filename.read_bytes() == before
    xr.testing.assert_identical(s.ds, before_ds)
    assert starts == []
