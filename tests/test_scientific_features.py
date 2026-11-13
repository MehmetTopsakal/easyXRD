"""Tests for the 6 scientific XRD features:
1. Propagated physical uncertainties (unit-cell parameters, volume, weight fractions)
2. Excluded regions & sample displacement / transparency / absorption corrections
3. Sequential / parametric series refinement (refine_series)
4. Amorphous content & internal-standard QPA (compute_amorphous_fraction)
5. Automated candidate phase screening (screen_phases)
6. Per-phase deconvolved Bragg profiles & normalized residuals in plots
"""

import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from easyxrd import easyxrd_defaults, exrd
from easyxrd.results import (
    clear_fit_results,
    compute_cell_uncertainties,
    compute_internal_standard_qpa,
    compute_weight_fraction_uncertainties,
    format_value_with_su,
)


def _cell_to_A(cell):
    a, b, c, alpha, beta, gamma = [float(x) for x in cell]
    ca = np.cos(np.deg2rad(alpha))
    cb = np.cos(np.deg2rad(beta))
    cg = np.cos(np.deg2rad(gamma))
    G = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ],
        dtype=float,
    )
    g = np.linalg.inv(G)
    return np.array([g[0, 0], g[1, 1], g[2, 2], 2.0 * g[0, 1], 2.0 * g[0, 2], 2.0 * g[1, 2]])


def _A_to_cell(A):
    g = np.array(
        [
            [A[0], A[3] / 2.0, A[4] / 2.0],
            [A[3] / 2.0, A[1], A[5] / 2.0],
            [A[4] / 2.0, A[5] / 2.0, A[2]],
        ],
        dtype=float,
    )
    G = np.linalg.inv(g)
    a = np.sqrt(G[0, 0])
    b = np.sqrt(G[1, 1])
    c = np.sqrt(G[2, 2])
    alpha = np.rad2deg(np.arccos(G[1, 2] / (b * c)))
    beta = np.rad2deg(np.arccos(G[0, 2] / (a * c)))
    gamma = np.rad2deg(np.arccos(G[0, 1] / (a * b)))
    vol = np.sqrt(np.linalg.det(G))
    return np.array([a, b, c, alpha, beta, gamma, vol])


# =============================================================================
# Feature 1: Propagated Physical Uncertainties
# =============================================================================


def test_format_value_with_su():
    assert format_value_with_su(5.43123, 0.00014, precision=4) == "5.43123(14)"
    assert format_value_with_su(5.43123, 0.00024, precision=4) == "5.4312(2)"
    assert format_value_with_su(5.43123, 0.0014, precision=4) == "5.4312(14)"
    assert format_value_with_su(90.0, 0.0, precision=2) == "90.00"
    assert format_value_with_su(90.0, np.nan, precision=2) == "90.00"
    assert format_value_with_su(np.nan, 0.01, precision=3) == "nan"


@pytest.mark.parametrize(
    "sg_laue,cell,vary_names,cov_sub",
    [
        ("m3m", [5.431, 5.431, 5.431, 90.0, 90.0, 90.0], ["0::A0"], [[(2e-6) ** 2]]),
        (
            "4/mmm",
            [4.593, 4.593, 2.959, 90.0, 90.0, 90.0],
            ["0::A0", "0::A2"],
            [[(3e-6) ** 2, 1e-12], [1e-12, (5e-6) ** 2]],
        ),
        (
            "6/mmm",
            [4.759, 4.759, 12.991, 90.0, 90.0, 120.0],
            ["0::A0", "0::A2"],
            [[(4e-6) ** 2, -2e-12], [-2e-12, (1e-6) ** 2]],
        ),
        (
            "mmm",
            [4.5, 5.5, 6.5, 90.0, 90.0, 90.0],
            ["0::A0", "0::A1", "0::A2"],
            np.diag([(3e-6) ** 2, (2e-6) ** 2, (1.5e-6) ** 2]),
        ),
        (
            "2/m",
            [5.1, 6.2, 7.3, 90.0, 105.0, 90.0],
            ["0::A0", "0::A1", "0::A2", "0::A4"],
            np.diag([(3e-6) ** 2, (2e-6) ** 2, (1.5e-6) ** 2, (4e-6) ** 2]),
        ),
        (
            "-1",
            [5.0, 6.0, 7.0, 85.0, 95.0, 102.0],
            ["0::A0", "0::A1", "0::A2", "0::A3", "0::A4", "0::A5"],
            np.diag([(3e-6) ** 2, (2.5e-6) ** 2, (2e-6) ** 2, (4e-6) ** 2, (3.5e-6) ** 2, (3e-6) ** 2]),
        ),
    ],
)
def test_compute_cell_uncertainties_matches_monte_carlo(sg_laue, cell, vary_names, cov_sub):
    cov_data = {"varyList": vary_names, "covMatrix": np.asarray(cov_sub, dtype=float)}
    su = compute_cell_uncertainties(cell, sg_laue, "0::", cov_data)
    assert su is not None

    # Compare against Monte Carlo propagation
    rng = np.random.default_rng(12345)
    A0 = _cell_to_A(cell)
    draws = rng.multivariate_normal(np.zeros(len(vary_names)), cov_data["covMatrix"], size=80000)
    samples = np.zeros((len(draws), 7), dtype=float)
    for idx, d in enumerate(draws):
        A = A0.copy()
        if sg_laue in ("m3m", "m3"):
            A[0] = A[1] = A[2] = A0[0] + d[0]
        elif sg_laue in ("4/mmm", "4/m"):
            A[0] = A[1] = A0[0] + d[0]
            A[2] = A0[2] + d[1]
        elif sg_laue in ("6/mmm", "6/m", "-3m", "-3", "3m1", "31m", "3"):
            A[0] = A[1] = A[3] = A0[0] + d[0]
            A[2] = A0[2] + d[1]
        elif sg_laue == "mmm":
            A[0], A[1], A[2] = A0[0] + d[0], A0[1] + d[1], A0[2] + d[2]
        elif sg_laue == "2/m":
            A[0], A[1], A[2], A[4] = A0[0] + d[0], A0[1] + d[1], A0[2] + d[2], A0[4] + d[3]
        else:
            A = A0 + d
        samples[idx] = _A_to_cell(A)

    mc_std = samples.std(axis=0)
    keys = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
    for k, expected in zip(keys, mc_std):
        np.testing.assert_allclose(su[k], expected, rtol=1.5e-2, atol=1e-7)


def test_weight_fraction_uncertainties_and_amorphous_qpa():
    unit_masses = np.array([100.0, 200.0, 150.0])
    scales = np.array([2.0, 1.0, 1.5])
    scale_names = ["0:0:Scale", "1:0:Scale", "2:0:Scale"]
    cov = np.array(
        [
            [0.004, -0.001, 0.0005],
            [-0.001, 0.002, -0.0003],
            [0.0005, -0.0003, 0.003],
        ]
    )
    cov_data = {"varyList": scale_names, "covMatrix": cov}
    wt_su = compute_weight_fraction_uncertainties(unit_masses, scales, scale_names, cov_data)
    assert wt_su is not None and wt_su.shape == (3,)
    assert np.all(wt_su > 0)

    # Single-phase weight fraction has zero uncertainty
    su_single = compute_weight_fraction_uncertainties(
        [100.0], [1.5], ["0:0:Scale"], {"varyList": ["0:0:Scale"], "covMatrix": [[0.01]]}
    )
    np.testing.assert_allclose(su_single, [0.0])

    # Internal-standard QPA with known 20 wt% spike of phase 0
    # Masses are m_0*s_0 = 200, m_1*s_1 = 200, m_2*s_2 = 225.
    # Since phase 0 is 20 wt% in the spiked sample:
    # spiked_wt = [20.0, 20.0, 22.5] -> sum = 62.5% -> amorphous_spiked = 37.5%
    # unspiked_norm = 0.80 -> original_wt = [0.0, 25.0, 28.125], amorphous_wt = 46.875%
    qpa = compute_internal_standard_qpa(
        unit_masses=unit_masses,
        scales=scales,
        scale_names=scale_names,
        covariance=cov_data,
        std_index=0,
        std_wt_fraction=20.0,
        std_wt_fraction_su=0.2,
    )
    np.testing.assert_allclose(qpa["spiked_wt_fractions"], [20.0, 20.0, 22.5])
    assert qpa["amorphous_spiked_wt_fraction"] == pytest.approx(37.5)
    np.testing.assert_allclose(qpa["original_wt_fractions"], [0.0, 25.0, 28.125])
    assert qpa["amorphous_wt_fraction"] == pytest.approx(46.875)
    assert qpa["spiked_wt_fractions_su"][0] == pytest.approx(0.2)
    assert np.all(np.isfinite(qpa["original_wt_fractions_su"]))
    assert qpa["amorphous_wt_fraction_su"] > 0


# =============================================================================
# Feature 2: Excluded Regions & Sample Parameters (Unit + Real GSAS-II)
# =============================================================================


def _two_phase_real_sample(tmp_path):
    pytest.importorskip("GSASII.GSASIIscriptable")
    easyxrd_defaults["easyxrd_scratch_path"] = str(tmp_path)
    s = exrd()
    q = np.linspace(1.5, 5.5, 400)
    # Ni peak near 3.09 A^-1 and Si peak near 2.00 and 3.27 A^-1
    y = (
        100.0
        + 1200.0 * np.exp(-((q - 3.09) / 0.05) ** 2)
        + 900.0 * np.exp(-((q - 3.57) / 0.05) ** 2)
        + 800.0 * np.exp(-((q - 2.00) / 0.05) ** 2)
        + 600.0 * np.exp(-((q - 3.27) / 0.05) ** 2)
    )
    s.load_xrd_data(
        from_i1d_array=np.column_stack([q, y, np.sqrt(y)]),
        i1d_array_wavelength_in_angstrom=1.5406,
        plot=False,
    )
    s.ds["i1d_baseline"] = xr.full_like(s.ds.i1d, 100.0)
    ni = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.52), ["Ni"], [[0, 0, 0]])
    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])
    ni_cif = tmp_path / "Ni.cif"
    si_cif = tmp_path / "Si.cif"
    ni.to(filename=ni_cif, fmt="cif")
    si.to(filename=si_cif, fmt="cif")
    s.load_phases(
        from_phases_dict=[
            {"cif": str(ni_cif), "label": "Ni"},
            {"cif": str(si_cif), "label": "Si"},
        ],
        plot=False,
    )
    return s


def test_excluded_regions_and_sample_parameters_real_gsas(tmp_path):
    s = _two_phase_real_sample(tmp_path)
    # Inject a huge parasitic peak at Q = 2.50 A^-1
    q = s.ds.i1d.radial.values
    s.ds["i1d"] = s.ds.i1d + 5000.0 * np.exp(-((q - 2.50) / 0.03) ** 2)

    s.setup_gsas2_refiner(
        excluded_regions=[(2.35, 2.65)],
        excluded_regions_unit="q",
        plot=False,
    )
    assert "excluded_regions_q" in s.ds.attrs
    assert "excluded_regions_tth" in s.ds.attrs
    excl_q = json.loads(s.ds.attrs["excluded_regions_q"])
    assert len(excl_q) == 1 and excl_q[0] == pytest.approx([2.35, 2.65])
    assert len(s.gpx.histograms()[0].data["Limits"]) == 3

    # Switch to Rietveld, refine cell parameters, sample parameters, and phase fractions
    s.set_LeBail(False, plot=False)
    s.refine_cell_parameters(min_gof_improvement=None, report=True, plot=False)
    assert np.isfinite(s.ds.attrs["PhaseInd_0_cell_a_su"])
    assert s.ds.attrs["PhaseInd_0_cell_a_su"] > 0
    assert np.isfinite(s.ds.attrs["PhaseInd_0_cell_volume"])
    assert np.isfinite(s.ds.attrs["PhaseInd_0_cell_volume_su"])

    # Sample displacement refinement (Bragg-Brentano Shift)
    s.refine_sample_parameters(
        sample_pars_to_refine=["Shift"],
        min_gof_improvement=None,
        plot=False,
    )
    assert s.ds.attrs.get("gsasii_sample_prm_Type") == "Bragg-Brentano"
    assert np.isfinite(s.ds.attrs.get("gsasii_sample_prm_Shift", np.nan))
    # Flag should be cleared afterwards by default
    assert s.gpx[s._histogram_name]["Sample Parameters"]["Shift"][1] is False

    # Mixing Bragg-Brentano and Debye-Scherrer parameters must raise ValueError
    with pytest.raises(ValueError, match="Cannot mix"):
        s.set_sample_parameters_refinement(["Shift", "DisplaceX"], save_gpx=False)

    # Refining sample displacement while instrument Zero is active warns
    s.set_instrument_parameters_refinement(["Zero"], True, save_gpx=False)
    with pytest.warns(UserWarning, match="Zero"):
        s.refine_sample_parameters(["Shift"], min_gof_improvement=None, plot=False)
    s.clear_instrument_parameters_refinement(save_gpx=False)

    # Refine phase fractions with internal standard (Ni = 25 wt%)
    s.refine_phase_fractions(
        min_gof_improvement=None,
        internal_standard_phase="Ni",
        standard_wt_fraction=25.0,
        standard_wt_fraction_su=0.5,
        plot=False,
    )
    assert np.isfinite(s.ds.attrs["PhaseInd_0_wt_fraction_su"])
    assert np.isfinite(s.ds.attrs["PhaseInd_1_wt_fraction_su"])
    assert s.ds.attrs["internal_standard_phase"] == "Ni"
    assert np.isfinite(s.ds.attrs["amorphous_wt_fraction"])
    assert np.isfinite(s.ds.attrs["amorphous_wt_fraction_su"])

    # Per-phase profiles must exist and sum to net Bragg intensity
    assert "i1d_phase_profiles" in s.ds
    assert list(s.ds.i1d_phase_profiles.coords["phase"].values) == ["Ni", "Si"]
    net_bragg = (s.ds.i1d_refined - s.ds.i1d_baseline - s.ds.i1d_gsas_background).values
    sum_phases = s.ds.i1d_phase_profiles.sum(dim="phase").values
    np.testing.assert_allclose(sum_phases, net_bragg, rtol=1e-3, atol=1e-2)

    # Plotting with phase profiles, excluded regions, normalized residuals, and s.u. labels
    s.plot(
        plot_hint=None,
        show_phase_profiles=True,
        normalized_residuals=True,
        close=False,
    )
    fig = plt.gcf()
    ax_d = [ax for ax in fig.axes if ax.get_ylabel() == r"$\Delta / \sigma$"]
    assert len(ax_d) == 1
    plt.close("all")

    # Clearing excluded regions unmasks data and removes attributes
    s.clear_excluded_regions(save_gpx=False)
    assert "excluded_regions_q" not in s.ds.attrs
    assert len(s.gpx.histograms()[0].data["Limits"]) == 2


def test_excluded_regions_validation(tmp_path):
    s = _two_phase_real_sample(tmp_path)
    s.setup_gsas2_refiner(do_1st_refinement=False, plot=False)
    with pytest.raises(ValueError, match="low < high"):
        s.set_excluded_regions([(3.0, 2.0)], save_gpx=False)
    with pytest.raises(ValueError, match="unit must be"):
        s.set_excluded_regions([(2.0, 2.5)], unit="nm", save_gpx=False)
    with pytest.raises(ValueError, match="does not overlap"):
        s.set_excluded_regions([(0.01, 0.02)], unit="q", save_gpx=False)


# =============================================================================
# Feature 3: Sequential / Parametric Series Refinement (refine_series)
# =============================================================================


def test_refine_series_chaining_and_error_recovery(tmp_path):
    s = _two_phase_real_sample(tmp_path)
    s.setup_gsas2_refiner(plot=False)
    s.set_LeBail(False, plot=False)

    q = s.ds.i1d.radial.values
    temps = [300.0, 400.0, 500.0]
    scans = []
    for idx, temp in enumerate(temps):
        shift = 0.005 * idx
        y = (
            100.0
            + 1200.0 * np.exp(-((q - (3.09 - shift)) / 0.05) ** 2)
            + 900.0 * np.exp(-((q - (3.57 - shift)) / 0.05) ** 2)
            + 800.0 * np.exp(-((q - 2.00) / 0.05) ** 2)
            + 600.0 * np.exp(-((q - 3.27) / 0.05) ** 2)
        )
        ds_scan = xr.Dataset(
            {
                "i1d": xr.DataArray(
                    y,
                    dims=("radial",),
                    coords={"radial": q},
                    attrs={"wavelength_in_angst": 1.5406, "xlabel": "Q"},
                ),
                "i1d_variance": xr.DataArray(y, dims=("radial",), coords={"radial": q}),
                "i1d_baseline": xr.DataArray(
                    np.full_like(q, 100.0), dims=("radial",), coords={"radial": q}
                ),
            }
        )
        scans.append(ds_scan)

    series_ds = s.refine_series(
        scans,
        coord_values=temps,
        coord_name="temperature",
        refine_steps=["background", "cell", "phase_fractions"],
        min_gof_improvement=None,
    )
    assert series_ds is s.series_ds
    assert list(series_ds.coords["temperature"].values) == temps
    assert np.all(series_ds.converged.values)
    assert series_ds.i1d_refined.shape == (3, q.size)
    assert series_ds.i1d_phase_profiles.shape == (3, 2, q.size)
    assert np.all(np.isfinite(series_ds.Rwp.values))
    assert np.all(np.isfinite(series_ds.PhaseInd_0_cell_a.values))
    assert np.all(np.isfinite(series_ds.PhaseInd_0_wt_fraction.values))

    # Also test 2D DataArray input with a bad (NaN) scan and stop_on_error=False
    da_2d = xr.DataArray(
        np.vstack([scans[0].i1d.values, np.full_like(q, np.nan), scans[2].i1d.values]),
        dims=("scan", "radial"),
        coords={"scan": [0, 1, 2], "radial": q},
        attrs={"wavelength_in_angst": 1.5406},
    )
    series_recovered = s.refine_series(
        da_2d,
        coord_name="scan",
        refine_steps=["background"],
        stop_on_error=False,
    )
    assert series_recovered.converged.values.tolist() == [True, False, True]


# =============================================================================
# Feature 5: Automated Candidate Phase Screening (screen_phases)
# =============================================================================


def test_screen_phases_local_structures_and_residual(tmp_path):
    s = exrd()
    q = np.linspace(1.5, 5.5, 500)
    ni = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.52), ["Ni"], [[0, 0, 0]])
    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])
    au = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.078), ["Au"], [[0, 0, 0]])

    # Build a pattern with strong Si peaks + weaker Ni peaks, no Au peaks
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xrdc = XRDCalculator(wavelength=1.5406)
    y = np.full_like(q, 50.0)
    for st, scale in ((si, 10.0), (ni, 2.5)):
        pat = xrdc.get_pattern(st, scaled=True, two_theta_range=(10.0, 90.0))
        q_peaks = 4.0 * np.pi * np.sin(np.deg2rad(pat.x / 2.0)) / 1.5406
        for qp, yp in zip(q_peaks, pat.y):
            if 1.5 <= qp <= 5.5:
                y += scale * yp * np.exp(-0.5 * ((q - qp) / 0.018) ** 2)

    s.load_xrd_data(
        from_i1d_array=np.column_stack([q, y]),
        i1d_array_wavelength_in_angstrom=1.5406,
        plot=False,
    )
    s.ds["i1d_baseline"] = xr.full_like(s.ds.i1d, 50.0)

    ranked = s.screen_phases(
        candidate_structures={"Si": si, "Ni": ni, "Au": au},
        tolerance_q=0.04,
    )
    assert len(ranked) == 3
    # Si and Ni should rank above Au
    top_names = [r["name"] for r in ranked[:2]]
    assert "Si" in top_names and "Ni" in top_names
    assert ranked[-1]["name"] == "Au"
    assert s.screened_phases == ranked

    # Now simulate that Si is already fitted in i1d_refined, and screen the residual for Ni vs Au
    y_si_calc = np.full_like(q, 50.0)
    pat_si = xrdc.get_pattern(si, scaled=True, two_theta_range=(10.0, 90.0))
    q_si = 4.0 * np.pi * np.sin(np.deg2rad(pat_si.x / 2.0)) / 1.5406
    for qp, yp in zip(q_si, pat_si.y):
        if 1.5 <= qp <= 5.5:
            y_si_calc += 10.0 * yp * np.exp(-0.5 * ((q - qp) / 0.018) ** 2)
    s.ds["i1d_refined"] = xr.DataArray(y_si_calc, dims=("radial",), coords={"radial": q})

    residual_ranked = s.screen_phases(
        candidate_structures={"Ni": ni, "Au": au},
        use_residual=True,
        tolerance_q=0.04,
    )
    assert residual_ranked[0]["name"] == "Ni"
    assert residual_ranked[0]["fom"] > residual_ranked[1]["fom"]


def test_screen_phases_mp_chemsys_mocked(monkeypatch):
    s = exrd()
    q = np.linspace(1.5, 5.0, 300)
    ni = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.52), ["Ni"], [[0, 0, 0]])
    y = 10.0 + 500.0 * np.exp(-0.5 * ((q - 3.09) / 0.02) ** 2) + 250.0 * np.exp(-0.5 * ((q - 3.57) / 0.02) ** 2)
    s.load_xrd_data(
        from_i1d_array=np.column_stack([q, y]),
        i1d_array_wavelength_in_angstrom=1.5406,
        plot=False,
    )

    doc_stable = SimpleNamespace(
        material_id="mp-23",
        formula_pretty="Ni",
        energy_above_hull=0.0,
        structure=ni,
    )
    doc_unstable = SimpleNamespace(
        material_id="mp-999999",
        formula_pretty="Ni",
        energy_above_hull=0.5,
        structure=ni,
    )

    class FakeRester:
        def __init__(self, api_key=None):
            self.materials = SimpleNamespace(
                summary=SimpleNamespace(search=lambda **kw: [doc_stable, doc_unstable])
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import sys
    monkeypatch.setitem(sys.modules, "mp_api", SimpleNamespace(client=SimpleNamespace(MPRester=FakeRester)))
    monkeypatch.setitem(sys.modules, "mp_api.client", SimpleNamespace(MPRester=FakeRester))

    results = s.screen_phases(
        chemsys="Ni",
        e_above_hull_max=0.05,
        mp_rester_api_key="DUMMY_KEY_FOR_TEST",
    )
    assert len(results) == 1
    assert results[0]["material_id"] == "mp-23"
    assert results[0]["fom"] > 0.5


def test_clear_fit_results_removes_phase_profiles_and_preserves_other_vars():
    q = np.linspace(1.0, 4.0, 20)
    ds = xr.Dataset(
        {
            "i1d": ("radial", np.ones_like(q)),
            "i1d_phase_profiles": (("phase", "radial"), np.ones((2, q.size))),
        },
        coords={"radial": q, "phase": ["Ni", "Si"]},
    )
    cleaned = clear_fit_results(ds)
    assert "i1d_phase_profiles" not in cleaned
    assert "phase" not in cleaned.dims
    assert "i1d" in cleaned
