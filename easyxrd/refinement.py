from . import DEFAULT, resolve_default

import os
import sys
import subprocess
import shutil
import random
import string
import fnmatch
import time
import copy
import warnings
import logging
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime, timezone
import numpy as np
import xarray as xr
from pymatgen.core.structure import Structure

from .utils import HiddenPrints, load_gsas_module, refinement_transaction
from .data_loader import q_to_two_theta, validate_profile
from .results import (
    clear_fit_results,
    compute_cell_uncertainties,
    compute_cell_volume,
    compute_internal_standard_qpa,
    compute_weight_fraction_uncertainties,
    fit_result_dataset,
    format_value_with_su,
)
from .provenance import file_record, software_versions, provenance_json
from . import easyxrd_defaults
from .plotters import exrd_plotter, refinement_plotter


logger = logging.getLogger(__name__)


BRAGG_SAMPLE_PARS = ("Shift", "Transparency", "SurfRoughA", "SurfRoughB")
DEBYE_SAMPLE_PARS = ("DisplaceX", "DisplaceY", "Absorption")
ALL_SAMPLE_PARS = BRAGG_SAMPLE_PARS + DEBYE_SAMPLE_PARS + ("Scale",)


INSTPRM_PRESETS = {
    "CuKa_lab": {
        "U": 22.75,
        "V": -17.6,
        "W": 10.6,
        "X": 1.58,
        "Y": 5.78,
        "SH/L": 0.002,
        "Zero": 0.0,
        "Polariz.": 0.0,
        "Azimuth": 0.0,
        "Z": 0.0,
    },
    "synchrotron": {
        "U": 1.0,
        "V": -0.5,
        "W": 0.1,
        "X": 0.5,
        "Y": 0.1,
        "SH/L": 0.0001,
        "Zero": 0.0,
        "Polariz.": 0.99,
        "Azimuth": 0.0,
        "Z": 0.0,
    },
}


class RefinementMixin:
    """
    Mixin class providing GSAS-II refinement setup, execution, and parameter controls.

    Refinement helpers use configurable minimum GoF improvement percentages. Pass
    min_gof_improvement to override the percentage, 0 to accept any strictly
    positive decrease, or None to disable the improvement check. Setup creates
    the initial fit and defaults to no improvement check. Negative Uiso
    or negative/nonfinite calculated CW instrument peak widths always
    discard the entire result. Individual profile coefficients may be
    negative; zero width components are allowed.
    """
    instprm_presets = INSTPRM_PRESETS
    @staticmethod
    def _normalization_peak(values):
        peak = np.nanmax(values)
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("Normalization requires a finite positive intensity maximum")
        return peak

    @property
    def _histogram_name(self):
        histograms = self.gpx.histograms()
        if len(histograms) != 1:
            raise ValueError("easyXRD requires exactly one powder histogram per analysis")
        return histograms[0].name

    def _phase_mode(self):
        modes = [p.data.get("Histograms", {}).get(self._histogram_name, {}).get("LeBail", False)
                 for p in self.gpx.phases()]
        return "LeBail" if any(modes) else "Rietveld"

    def _set_po(
        self,
        phase_ind,
        order,
        enabled,
        model="SH",
        md_axis=(0, 0, 1),
        md_ratio=1.0,
    ):
        model_clean = str(model).upper().strip()
        if model_clean not in ("SH", "MD"):
            raise ValueError(
                f"Preferred orientation model must be 'SH' (Spherical Harmonics) or 'MD' (March-Dollase), got {model!r}"
            )
        phases = self.gpx.phases()
        if phase_ind in ("all", None):
            selected = list(range(len(phases)))
        elif isinstance(phase_ind, (int, np.integer)):
            selected = [int(phase_ind)]
        else:
            selected = [i for i, p in enumerate(phases) if p.name == phase_ind]
            if not selected:
                raise ValueError(f"Phase {phase_ind!r} not found in project phases")

        if model_clean == "MD":
            try:
                axis = [int(x) for x in md_axis]
            except (TypeError, ValueError) as exc:
                raise ValueError("March-Dollase axis must be a sequence of 3 integers") from exc
            if len(axis) != 3 or all(x == 0 for x in axis):
                raise ValueError("March-Dollase axis must be a sequence of 3 integers with at least one nonzero component")
            for index in selected:
                phase = phases[index]
                hap = phase.data["Histograms"][self._histogram_name]
                previous = hap.get("Pref.Ori.", [None] * 4)
                if md_ratio is not None:
                    try:
                        ratio = float(md_ratio)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("March-Dollase ratio must be a finite positive number") from exc
                    if not np.isfinite(ratio) or ratio <= 0:
                        raise ValueError("March-Dollase ratio must be a finite positive number")
                else:
                    ratio = float(previous[1]) if (previous and previous[0] == "MD" and np.isfinite(previous[1])) else 1.0
                hap["Pref.Ori."] = ["MD", ratio, bool(enabled), list(axis), 0, {}, [""], 0.1]
        else:
            if not isinstance(order, (int, np.integer)) or order < 0 or order % 2:
                raise ValueError("Spherical harmonic order must be a nonnegative even integer")
            lattice = load_gsas_module("GSASIIlattice")
            for index in selected:
                phase = phases[index]
                symmetry = phase.data["General"]["SGData"]["SGLaue"]
                hap = phase.data["Histograms"][self._histogram_name]
                previous = hap.get("Pref.Ori.", [None] * 6)
                old = previous[5] if previous[0] == "SH" else {}
                names = lattice.GenSHCoeffT(symmetry, '0', order, False)
                coefficients = {name: old.get(name, 0.) for name in names}
                hap["Pref.Ori."] = ["SH", 1., bool(enabled), [0, 0, 1], order, coefficients, [""], 0.1]

    def _reset_rwp_history(self):
        """Start an empty dataset history for a new GSAS-II engine session."""
        if "refinement_step" in self.ds.dims:
            self.ds = self.ds.drop_dims("refinement_step")
        history = xr.Dataset(
            {
                "rwp_history": ("refinement_step", np.array([], dtype=float)),
                "rwp_attempted_history": ("refinement_step", np.array([], dtype=float)),
            },
            coords={
                "refinement_step": np.array([], dtype=int),
                **{name: ("refinement_step", np.array([], dtype=str)) for name in (
                    "refinement_status", "refinement_label", "refinement_timestamp",
                    "refinement_reason",
                )},
            },
        )
        history.rwp_history.attrs = {
            "long_name": "Retained Rwp after each refinement attempt",
            "units": "%",
        }
        history.rwp_attempted_history.attrs = {
            "long_name": "Optimizer Rwp before acceptance or rollback",
            "units": "%",
        }
        history.refinement_timestamp.attrs["timezone"] = "UTC"
        self.ds = xr.merge([self.ds, history], join="exact")

    @staticmethod
    def _project_rwp(project):
        """A new GSAS-II project may not have fit statistics yet."""
        try:
            return float(project["Covariance"]["data"]["Rvals"]["Rwp"])
        except (KeyError, TypeError, ValueError):
            return np.nan

    def _record_rwp(self, attempted_rwp, status, label, reason=""):
        """Append independently of dataset/phase update flags and rollback."""
        if "rwp_history" not in self.ds:
            self._reset_rwp_history()
        previous = self.ds[["rwp_history", "rwp_attempted_history"]]
        entry = xr.Dataset(
            {
                "rwp_history": ("refinement_step", [
                    self._project_rwp(self.gpx) if status != "error" else np.nan
                ]),
                "rwp_attempted_history": ("refinement_step", [attempted_rwp]),
            },
            coords={
                "refinement_step": [previous.sizes["refinement_step"] + 1],
                "refinement_status": ("refinement_step", [status]),
                "refinement_label": ("refinement_step", [label]),
                "refinement_timestamp": ("refinement_step", [
                    datetime.now(timezone.utc).isoformat()
                ]),
                "refinement_reason": ("refinement_step", [reason]),
            },
        )
        history = xr.concat([previous, entry], dim="refinement_step")
        # Assigning into the old dataset would align with its old step index
        # and silently discard the appended row. Merge against the profile data.
        self.ds = xr.merge([
            self.ds.drop_dims("refinement_step"), history
        ], join="exact")

    @staticmethod
    def _validate_gof_improvement(min_gof_improvement):
        if min_gof_improvement is not None and (
            not np.isfinite(min_gof_improvement) or min_gof_improvement < 0
        ):
            raise ValueError("min_gof_improvement must be a finite, nonnegative percentage")

    def _prepare_refinement(self, min_gof_improvement):
        """Validate before changing any flags and preserve the accepted GPX."""
        self._validate_gof_improvement(min_gof_improvement)
        original, checkpoint = getattr(self, "_transaction_checkpoint", (None, None))
        if getattr(self, "_transaction_checkpoint_available", False) and self.gpx is original:
            self._transaction_checkpoint_available = False
            return checkpoint
        return copy.deepcopy(self.gpx)

    def _restore_project_checkpoint(self, previous):
        """Keep the transaction's exception recovery copy isolated on rejection."""
        checkpoint = getattr(self, "_transaction_checkpoint", (None, None))[1]
        self.gpx = copy.deepcopy(previous) if previous is checkpoint else previous

    @staticmethod
    def _invalid_instrument_peak_widths(project):
        """Find negative/nonfinite CW profile widths within each fit's 2theta limits.

        Use the current GSAS-II coefficients: sigma^2 = U*tan(theta)^2 +
        V*tan(theta) + W, gamma = X/cos(theta) + Y*tan(theta) + Z.
        Endpoints and stationary points cover the entire continuous interval,
        including minima between measured bins. Negative coefficients are valid
        when these combined functions remain nonnegative. Other profile types
        use different equations and are not checked here.
        """
        invalid = []
        for histogram in project.histograms():
            instrument = histogram.data.get("Instrument Parameters")
            if not instrument:
                continue
            parameters = instrument[0]
            if "C" not in parameters.get("Type", [""])[0]:
                continue
            if not all(name in parameters for name in ("U", "V", "W", "X", "Y")):
                continue
            u, v, w, x, y = (float(parameters[name][1])
                              for name in ("U", "V", "W", "X", "Y"))
            z = float(parameters["Z"][1]) if "Z" in parameters else 0.
            lower, upper = histogram.data["Limits"][1]
            positions = [lower, upper]

            # Gaussian stationary point: tan(theta) = -V/(2U).
            if u != 0:
                position = np.rad2deg(2 * np.arctan(-v / (2 * u)))
                if lower < position < upper:
                    positions.append(position)
            # Lorentzian derivative: (X*sin(theta) + Y) / cos(theta)^2.
            if x != 0 and -1 <= -y / x <= 1:
                position = np.rad2deg(2 * np.arcsin(-y / x))
                if lower < position < upper:
                    positions.append(position)

            theta = np.deg2rad(np.asarray(positions) / 2)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                gaussian = u * np.tan(theta)**2 + v * np.tan(theta) + w
                lorentzian = x / np.cos(theta) + y * np.tan(theta) + z
            for label, widths in (("Gaussian variance", gaussian),
                                  ("Lorentzian width", lorentzian)):
                bad = np.flatnonzero(~np.isfinite(widths) | (widths < 0))
                if bad.size:
                    i = bad[0]
                    invalid.append(
                        f"{histogram.name}/{label}={widths[i]:.6g} "
                        f"at 2theta={positions[i]:.6g} deg"
                    )
        return invalid

    def _restore_rejected_refinement(self, gpx_previous, message, plot=False,
                                     plot_hint="refine"):
        """Restore wrapper changes as well as the optimizer's rejected result."""
        self._restore_project_checkpoint(gpx_previous)
        self.gpx_saver()
        if plot and self.last_refinement_rejection_reason != "insufficient_gof_improvement":
            exrd_plotter(
                ds=self.ds,
                ds_previous=getattr(self, "ds_previous", None),
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint=plot_hint,
                title_str=message,
            )
        # High-level helpers report via logging; returning the diagnostic here
        # would make Jupyter echo it again as the cell result.
        return None

    @refinement_transaction
    def _refine_gpx(self, gpx_previous, verbose=False, min_gof_improvement=None,
                    log_label="refine"):
        """Run GSAS-II and restore invalid or insufficiently improved results.

        Returns a rejection message, or None when the result is accepted. This
        check must run before copying results into datasets or CIF structures.
        """
        # Rvals can contain nested dictionaries shared with dataset attributes
        # (e.g. lastShifts). GSAS-II mutates these even before results are copied
        # below, so preserve metadata independently of the update/history flags.
        dataset_attrs_previous = [
            (dataset, copy.deepcopy(dataset.attrs))
            for dataset in (getattr(self, "ds", None),
                            getattr(self, "ds_previous", None))
            if isinstance(dataset, xr.Dataset)
        ]
        try:
            if self.verbose or verbose:
                self.gpx.refine()
            else:
                with HiddenPrints():
                    self.gpx.refine()
        except (Exception, KeyboardInterrupt) as exc:
            self.last_refinement_accepted = False
            self.last_refinement_rejection_reason = "error"
            self._record_rwp(np.nan, "error", log_label,
                             f"{type(exc).__name__}: {exc}")
            raise
        attempted_rwp = self._project_rwp(self.gpx)

        negative_sites = []
        for phase in self.gpx.phases():
            # Atom offsets differ between GSAS-II phase types. cia points to
            # the I/A flag; the following entry is Uiso. Anisotropic atoms have
            # an unused Uiso slot, which must not determine acceptance.
            _, ct, _, cia = phase.data["General"]["AtomPtrs"]
            for atom in phase.data["Atoms"]:
                if atom[cia] == "I" and (not np.isfinite(atom[cia + 1]) or atom[cia + 1] < 0):
                    negative_sites.append(
                        f"{phase.name}/{atom[ct - 1]}: Uiso={atom[cia + 1]:.6g}"
                    )

        invalid_peak_widths = self._invalid_instrument_peak_widths(self.gpx)

        reason = None
        self.last_refinement_rejection_reason = None
        if negative_sites:
            self.last_refinement_rejection_reason = "negative_uiso"
            reason = "negative Uiso (" + "; ".join(negative_sites) + ")"
        elif invalid_peak_widths:
            self.last_refinement_rejection_reason = "invalid_peak_width"
            reason = (
                "negative or nonfinite instrument peak widths ("
                + "; ".join(invalid_peak_widths) + ")"
            )
        elif min_gof_improvement is None and (
            not np.isfinite(attempted_rwp) or attempted_rwp < 0
            or not np.isfinite(self.gpx["Covariance"]["data"]["Rvals"].get("GOF", np.nan))
        ):
            self.last_refinement_rejection_reason = "invalid_fit_statistics"
            reason = "nonfinite or invalid fit statistics"
        elif min_gof_improvement is not None:
            old_gof = gpx_previous["Covariance"]["data"]["Rvals"]["GOF"]
            new_gof = self.gpx["Covariance"]["data"]["Rvals"]["GOF"]
            improvement = 100 * (old_gof - new_gof) / old_gof if old_gof > 0 else np.nan
            if not np.isfinite(improvement) or improvement <= 0 or improvement < min_gof_improvement:
                self.last_refinement_rejection_reason = "insufficient_gof_improvement"
                reason = (
                    f"insufficient GoF improvement ({improvement:.6g}%; "
                    f"requires a decrease of at least {min_gof_improvement:g}%)"
                )

        self.last_refinement_accepted = reason is None
        if reason is None:
            self._record_rwp(attempted_rwp, "accepted", log_label)
            return None

        self._restore_project_checkpoint(gpx_previous)
        for dataset, attrs in dataset_attrs_previous:
            dataset.attrs = attrs
        # G2Project.refine() writes its result before returning. Replace that
        # rejected result on disk as well as in memory.
        self.gpx_saver()
        self._record_rwp(attempted_rwp, "rejected", log_label,
                         self.last_refinement_rejection_reason)
        message = (
            "Refinement discarded because of " + reason
            + ". \n      Previous project restored; profile data and phases unchanged."
        )
        print(" ❌--" + message)
        logger.warning("%s", message)
        return message

    def _report_refinement(self, title_str, ref_str):
        """Keep helper messages and plot titles consistent with acceptance."""
        if not self.last_refinement_accepted:
            return ref_str
        print(" ✅--" + title_str)
        logger.info("%s", title_str)
        return title_str

    @refinement_transaction
    def refine(
        self,
        update_ds=DEFAULT,
        update_ds_phases=DEFAULT,
        update_phases=DEFAULT,
        update_previous_ds=DEFAULT,
        update_previous_gpx=DEFAULT,
        update_previous_phases=DEFAULT,
        verbose=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
        log_label="refine",
    ):
        """Refine, discarding negative Uiso or invalid CW instrument peak widths.

        Rejection restores the pre-refinement GPX in memory and on disk, leaves
        profile data, structures and previous-result snapshots unchanged, and returns
        a diagnostic string. Negative profile coefficients are allowed when
        the resulting Gaussian variance and Lorentzian width are finite and
        nonnegative throughout each CW histogram's fitted 2theta interval.
        This check also applies with the GoF improvement check disabled.
        Zero Uiso and zero width components are accepted. When min_gof_improvement is
        set, also require a positive GoF decrease of at least that percentage
        (disabled by default). Pass None to disable the improvement check.
        Results discarded for insufficient GoF improvement never plot, even
        when plot=True.

        Each optimizer attempt appends to ds.rwp_history, even with update_ds
        disabled or when the result is rejected. ds.rwp_attempted_history and
        refinement_status distinguish trial values from retained fits. Use
        log_label to describe a manually configured refinement step.
        """
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_min_gof_improvement")
        update_ds = resolve_default(update_ds, "refine_update_ds")
        update_ds_phases = resolve_default(update_ds_phases, "refine_update_ds_phases")
        update_phases = resolve_default(update_phases, "refine_update_phases")
        update_previous_ds = resolve_default(update_previous_ds, "refine_update_previous_ds")
        update_previous_gpx = resolve_default(update_previous_gpx, "refine_update_previous_gpx")
        update_previous_phases = resolve_default(update_previous_phases, "refine_update_previous_phases")
        verbose = resolve_default(verbose, "refine_verbose")
        plot = resolve_default(plot, "refine_plot")

        gpx_previous = self._prepare_refinement(min_gof_improvement)
        if update_previous_ds:
            vars_to_copy = [
                k
                for k in [
                    "i1d",
                    "i1d_refined",
                    "i1d_gsas_background",
                    "i1d_baseline",
                ]
                if k in self.ds
            ]
            ds_previous = self.ds[vars_to_copy].copy(deep=True)
            ds_previous.attrs = self.ds.attrs.copy()
        else:
            ds_previous = None

        if update_previous_phases:
            phases_previous = copy.deepcopy(self.phases)
        else:
            phases_previous = None

        rejection = None if getattr(self, "_sync_only", False) else self._refine_gpx(
            gpx_previous, verbose=verbose, min_gof_improvement=min_gof_improvement,
            log_label=log_label,
        )
        if rejection is not None:
            if plot and self.last_refinement_rejection_reason != "insufficient_gof_improvement":
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=getattr(self, "ds_previous", None),
                    figsize=self.figsize,
                    plot_hint="refine",
                    title_str=rejection,
                )
            return rejection

        if update_ds:
            prev_ds_attrs = dict(self.ds.attrs)
            histogram = self.gpx.histograms()[0]
            multiplier = self.ds.i1d.attrs.get("normalization_multiplier", 1.)
            offset = (self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"]
                      if "i1d_baseline" in self.ds and "normalized_to" in self.ds.i1d.attrs else 0.)
            raw_ycalc = np.asarray(np.ma.getdata(histogram.getdata("ycalc")), dtype=float)
            raw_ybkg = np.asarray(np.ma.getdata(histogram.getdata("background")), dtype=float)
            calculated = (raw_ycalc - offset) / multiplier
            background = (raw_ybkg - offset) / multiplier
            self.ds["i1d_refined"] = self.ds.i1d.copy(data=calculated)
            if "i1d_baseline" in self.ds:
                self.ds["i1d_refined"] += self.ds.i1d_baseline
            self.ds["i1d_gsas_background"] = self.ds.i1d.copy(data=background)
            self.ds.attrs.update(copy.deepcopy(self.gpx["Covariance"]["data"]["Rvals"]))
            self.ds.attrs["refinement_mode"] = self._phase_mode()

            self.ds = self._store_fit_results(self.ds)
            self.ds["i1d_phase_profiles"] = self._compute_phase_profiles(
                histogram, calculated - background, multiplier
            )
            self.ds.attrs.update(self._phase_result_attrs(self.gpx, previous_attrs=prev_ds_attrs))

            unit_masses = np.array([
                p.data["General"]["Mass"] for p in self.gpx.phases()
            ], dtype=float)
            scales = np.array([
                p.data["Histograms"][self._histogram_name]["Scale"][0]
                for p in self.gpx.phases()
            ], dtype=float)
            masses = unit_masses * scales
            valid_fractions = (self._phase_mode() == "Rietveld" and np.all(np.isfinite(masses))
                               and np.all(masses >= 0) and masses.sum() > 0)
            self.ds.attrs["weight_fractions_valid"] = bool(valid_fractions)
            for index, mass in enumerate(masses):
                self.ds.attrs[f"PhaseInd_{index}_wt_fraction"] = (100 * mass / masses.sum()
                                                                 if valid_fractions else np.nan)
            if not valid_fractions:
                for index in range(len(masses)):
                    self.ds.attrs[f"PhaseInd_{index}_wt_fraction_su"] = np.nan
                self._clear_internal_standard_attrs()
            else:
                h_id = self._histogram_id(histogram)
                scale_names = [
                    f"{p.data.get('pId', idx)}:{h_id}:Scale"
                    for idx, p in enumerate(self.gpx.phases())
                ]
                wt_su = compute_weight_fraction_uncertainties(
                    unit_masses, scales, scale_names, self.gpx["Covariance"]["data"]
                )
                if wt_su is not None:
                    for index, su_val in enumerate(wt_su):
                        self.ds.attrs[f"PhaseInd_{index}_wt_fraction_su"] = float(su_val)
                else:
                    same_wt = all(
                        np.isclose(
                            self.ds.attrs.get(f"PhaseInd_{idx}_wt_fraction", np.nan),
                            prev_ds_attrs.get(f"PhaseInd_{idx}_wt_fraction", np.nan),
                            rtol=1e-7, atol=1e-7,
                        )
                        for idx in range(len(masses))
                    )
                    for index in range(len(masses)):
                        key_su = f"PhaseInd_{index}_wt_fraction_su"
                        if same_wt and key_su in prev_ds_attrs:
                            self.ds.attrs[key_su] = float(prev_ds_attrs[key_su])
                        elif len(masses) == 1:
                            self.ds.attrs[key_su] = 0.0
                        else:
                            self.ds.attrs[key_su] = np.nan
                if (
                    "internal_standard_phase" in prev_ds_attrs
                    and "internal_standard_wt_fraction" in prev_ds_attrs
                ):
                    try:
                        self.compute_amorphous_fraction(
                            prev_ds_attrs["internal_standard_phase"],
                            prev_ds_attrs["internal_standard_wt_fraction"],
                            prev_ds_attrs.get("internal_standard_wt_fraction_su", 0.0),
                        )
                    except Exception:
                        self._clear_internal_standard_attrs()

            self.ds.attrs.update(self._phase_result_attrs(gpx_previous, suffix="_previous"))

            inst_prm_dict = self.gpx[self._histogram_name]["Instrument Parameters"][0]
            inst_prm_dict_clean = {}
            for i in inst_prm_dict:
                inst_prm_dict_clean["gsasii_inst_prm_" + i] = inst_prm_dict[i][1]
            self.ds.attrs = self.ds.attrs | inst_prm_dict_clean

            sample_prm_dict = self._sample_parameters_dict()
            if isinstance(sample_prm_dict, dict):
                sample_prm_clean = {}
                if "Type" in sample_prm_dict:
                    sample_prm_clean["gsasii_sample_prm_Type"] = sample_prm_dict["Type"]
                for key in ALL_SAMPLE_PARS:
                    if (
                        key in sample_prm_dict
                        and isinstance(sample_prm_dict[key], (list, tuple))
                        and len(sample_prm_dict[key]) >= 1
                    ):
                        sample_prm_clean["gsasii_sample_prm_" + key] = float(sample_prm_dict[key][0])
                self.ds.attrs = self.ds.attrs | sample_prm_clean

        if update_phases or update_ds_phases:
            for e, p in enumerate(self.gpx.phases()):
                cif_path = "%s/%s_refined.cif" % (self.gsasii_run_directory, p.name)

                p.export_CIF(outputname=cif_path)
                with open(cif_path, "r") as ciffile:
                    ciffile_content = ciffile.read()
                if update_ds_phases:
                    self.ds.attrs["PhaseInd_%d_cif" % (e)] = ciffile_content
                if update_phases:
                    self.phases[p.name] = Structure.from_str(ciffile_content, fmt="cif")

        if update_previous_gpx:
            self.gpx_previous = gpx_previous
        if update_previous_ds:
            self.ds_previous = ds_previous
        if update_previous_phases:
            self.phases_previous = phases_previous

        try:
            gof_change = (
                100
                * (
                    self.gpx["Covariance"]["data"]["Rvals"]["GOF"]
                    - self.gpx_previous["Covariance"]["data"]["Rvals"]["GOF"]
                )
                / self.gpx_previous["Covariance"]["data"]["Rvals"]["GOF"]
            )
            if gof_change < -10:
                gof_symbol = "*"
            elif gof_change > -1:
                gof_symbol = "!"
            else:
                gof_symbol = ""
            refinement_str = "Rwp/GoF is now %.3f/%.3f (was %.3f(%.2f%%)/%.3f(%.2f%%%s))" % (
                self.gpx["Covariance"]["data"]["Rvals"]["Rwp"],
                self.gpx["Covariance"]["data"]["Rvals"]["GOF"],
                # self.gpx["Covariance"]["data"]["Rvals"]["Nvars"],
                self.gpx_previous["Covariance"]["data"]["Rvals"]["Rwp"],
                100
                * (
                    self.gpx["Covariance"]["data"]["Rvals"]["Rwp"]
                    - self.gpx_previous["Covariance"]["data"]["Rvals"]["Rwp"]
                )
                / self.gpx_previous["Covariance"]["data"]["Rvals"]["Rwp"],
                self.gpx_previous["Covariance"]["data"]["Rvals"]["GOF"],
                gof_change,
                gof_symbol,
            )
        except Exception:
            refinement_str = "Rwp/GoF is %.3f/%.3f" % (
                self.gpx["Covariance"]["data"]["Rvals"]["Rwp"],
                self.gpx["Covariance"]["data"]["Rvals"]["GOF"],
                # self.gpx["Covariance"]["data"]["Rvals"]["Nvars"],
            )

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                plot_hint="refine",
                title_str=refinement_str.replace("✨", "").replace("❗", ""),
            )

        return refinement_str

    @staticmethod
    def _histogram_id(histogram):
        """Return the GSAS-II integer histogram id (defaulting to 0)."""
        try:
            return int(histogram.data.get("hId", histogram.data["data"][0].get("hId", 0)))
        except Exception:
            return 0

    def _clear_internal_standard_attrs(self):
        """Remove internal-standard and amorphous QPA attributes when invalidated."""
        for key in list(self.ds.attrs.keys()):
            if (
                key.startswith("internal_standard_")
                or key.startswith("amorphous_")
                or key.endswith("_spiked_wt_fraction")
                or key.endswith("_spiked_wt_fraction_su")
                or key.endswith("_original_wt_fraction")
                or key.endswith("_original_wt_fraction_su")
            ):
                self.ds.attrs.pop(key, None)

    def _compute_phase_profiles(self, histogram, net_bragg, multiplier):
        """Compute deconvolved per-phase Bragg profiles on the dataset intensity scale."""
        phases = self.gpx.phases() if hasattr(self, "gpx") and self.gpx is not None else []
        phase_names = [str(p.name) for p in phases]
        n_phases = len(phases)
        n_pts = self.ds.i1d.size
        net_bragg_arr = np.asarray(net_bragg, dtype=float)
        if n_phases == 0:
            profiles = np.empty((0, n_pts), dtype=float)
        elif n_phases == 1:
            profiles = net_bragg_arr[None, :].copy()
        else:
            profiles = None
            ref_lists = getattr(histogram, "data", {}).get("Reflection Lists")
            inst_list = getattr(histogram, "data", {}).get("Instrument Parameters")
            if isinstance(ref_lists, dict) and inst_list:
                try:
                    G2pwd = load_gsas_module("GSASIIpwd", [getattr(self, "gsasii_lib_path", "")])
                    x = np.asarray(np.ma.getdata(histogram.getdata("x")), dtype=float)
                    inst = inst_list[0]
                    shl = max(float(inst.get("SH/L", [0.002, 0.002])[1]), 0.002)
                    raw_profiles = np.zeros((n_phases, n_pts), dtype=float)
                    for idx, p in enumerate(phases):
                        rdict = ref_lists.get(p.name, {})
                        im = 1 if p.data.get("General", {}).get("Modulated", False) else 0
                        for refl in rdict.get("RefList", []):
                            pos = float(refl[5 + im])
                            sig = float(refl[6 + im])
                            gam = float(refl[7 + im])
                            _, fmin, fmax = G2pwd.getWidthsCW(pos, sig, gam, shl)
                            i_beg = int(np.searchsorted(x, pos - fmin))
                            i_fin = int(np.searchsorted(x, pos + fmax))
                            if 0 <= i_beg < i_fin <= n_pts:
                                fp = G2pwd.getFCJVoigt3(pos, sig, gam, shl, x[i_beg:i_fin])[0]
                                raw_profiles[idx, i_beg:i_fin] += (
                                    float(refl[11 + im]) * float(refl[9 + im]) * fp
                                )
                    ycalc_ma = histogram.getdata("ycalc")
                    if np.ma.is_masked(ycalc_ma):
                        raw_profiles[:, np.ma.getmaskarray(ycalc_ma)] = 0.0
                    profiles = raw_profiles / float(multiplier)
                except Exception:
                    profiles = None
            if profiles is None:
                try:
                    weights = np.array([
                        float(p.data["General"]["Mass"])
                        * float(p.data["Histograms"][self._histogram_name]["Scale"][0])
                        for p in phases
                    ], dtype=float)
                    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                        weights = np.ones(n_phases, dtype=float)
                except Exception:
                    weights = np.ones(n_phases, dtype=float)
                weights = weights / weights.sum()
                profiles = weights[:, None] * net_bragg_arr[None, :]

        return xr.DataArray(
            profiles,
            dims=("phase", "radial"),
            coords={
                "phase": np.asarray(phase_names, dtype=str),
                "radial": self.ds.i1d.radial,
            },
            attrs={
                "long_name": "Calculated Bragg intensity profile per phase (excluding background)",
                "includes_background": "False",
            },
        )

    @staticmethod
    def _phase_result_attrs(project, suffix="", previous_attrs=None):
        """Extract cell/profile attributes and propagated cell s.u. consistently."""
        hist_name = project.histograms()[0].name
        try:
            cov_data = project["Covariance"]["data"]
        except Exception:
            cov_data = {}
        attrs = {}
        cell_keys = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
        for index, phase in enumerate(project.phases()):
            general = phase.data["General"]
            prefix = f"PhaseInd_{index}_"
            if not suffix:
                for key in ("SGSys", "SpGrp"):
                    attrs[prefix + key] = general["SGData"][key]
            cell_vals = general["Cell"][1:7]
            for name, value in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell_vals):
                attrs[f"{prefix}cell_{name}{suffix}"] = value
            vol = (
                float(general["Cell"][7])
                if len(general["Cell"]) > 7
                else compute_cell_volume(cell_vals)
            )
            attrs[f"{prefix}cell_volume{suffix}"] = vol
            p_id = phase.data.get("pId", getattr(phase, "id", index))
            sg_laue = general.get("SGData", {}).get("SGLaue", "-1")
            cell_su = compute_cell_uncertainties(cell_vals, sg_laue, f"{p_id}::", cov_data)
            if cell_su is not None:
                for name in cell_keys:
                    attrs[f"{prefix}cell_{name}_su{suffix}"] = cell_su[name]
            elif previous_attrs is not None:
                same_cell = all(
                    np.isclose(
                        attrs[f"{prefix}cell_{name}{suffix}"],
                        previous_attrs.get(f"{prefix}cell_{name}", np.nan),
                        rtol=1e-9, atol=1e-9,
                    )
                    for name in ("a", "b", "c", "alpha", "beta", "gamma")
                )
                for name in cell_keys:
                    prev_key = f"{prefix}cell_{name}_su"
                    attrs[f"{prefix}cell_{name}_su{suffix}"] = (
                        float(previous_attrs[prev_key])
                        if same_cell and prev_key in previous_attrs
                        else np.nan
                    )
            else:
                for name in cell_keys:
                    attrs[f"{prefix}cell_{name}_su{suffix}"] = np.nan
            hap = phase.data["Histograms"][hist_name]
            attrs[f"{prefix}use_lebail{suffix}"] = bool(hap.get("LeBail", False))
            for key, kind, field in (("Size", "size", "size"), ("Mustrain", "strain", "mustrain")):
                attrs[f"{prefix}{kind}_broadening_type{suffix}"] = hap[key][0]
                for component, value in enumerate(hap[key][1][:3]):
                    attrs[f"{prefix}{field}_{component}{suffix}"] = value
            po = hap.get("Pref.Ori.")
            if po and len(po) >= 4:
                attrs[f"{prefix}pref_ori_model{suffix}"] = po[0]
                if po[0] == "MD":
                    attrs[f"{prefix}pref_ori_ratio{suffix}"] = float(po[1])
                    attrs[f"{prefix}pref_ori_axis{suffix}"] = tuple(po[3])
                elif po[0] == "SH" and len(po) >= 5:
                    attrs[f"{prefix}pref_ori_order{suffix}"] = int(po[4])
        return attrs

    def _store_fit_results(self, ds):
        result = fit_result_dataset(self.gpx["Covariance"]["data"])
        result.attrs["gsas_result_step"] = ds.sizes.get("refinement_step", 0)
        clean = clear_fit_results(ds)
        merged = xr.merge([clean, result], join="exact")
        merged.attrs.update(result.attrs)
        return merged

    def gpx_saver(self):
        if self.verbose:
            self.gpx.save()
        else:
            with HiddenPrints():
                self.gpx.save()

    @refinement_transaction
    def setup_gsas2_refiner(
        self,
        gsasii_lib_path=None,
        instprm_from_gpx=None,
        instprm_from_nc=None,
        instprm_file=None,
        instprm_preset=DEFAULT,
        instprm_Polariz=DEFAULT,
        instprm_Azimuth=DEFAULT,
        instprm_Zero=DEFAULT,
        instprm_U=DEFAULT,
        instprm_V=DEFAULT,
        instprm_W=DEFAULT,
        instprm_X=DEFAULT,
        instprm_Y=DEFAULT,
        instprm_Z=DEFAULT,
        instprm_SHL=DEFAULT,
        do_1st_refinement=DEFAULT,
        yshift_multiplier=DEFAULT,
        normalize=DEFAULT,
        normalize_to=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
        excluded_regions=None,
        excluded_regions_unit=DEFAULT,
    ):
        """Create the initial fit; optionally gate the second, LeBail pass.

        The first pass establishes the baseline GoF and is always ungated.
        min_gof_improvement defaults to None during setup.
        Starts a new ds.rwp_history on the refinement_step dimension. Setup's
        initial optimizer passes are logged too; do_1st_refinement=False leaves
        an empty history until the first refinement attempt.
        Instrument parameters can be loaded from a .gpx project (instprm_from_gpx),
        a NetCDF file (instprm_from_nc), a GSAS-II .instprm file (instprm_file),
        or a standard preset such as 'CuKa_lab' or 'synchrotron' (instprm_preset).
        """
        instprm_preset = resolve_default(instprm_preset, "setup_gsas2_refiner_instprm_preset")
        min_gof_improvement = resolve_default(min_gof_improvement, "setup_gsas2_refiner_min_gof_improvement")
        instprm_Polariz = resolve_default(instprm_Polariz, "setup_gsas2_refiner_instprm_Polariz")
        instprm_Azimuth = resolve_default(instprm_Azimuth, "setup_gsas2_refiner_instprm_Azimuth")
        instprm_Zero = resolve_default(instprm_Zero, "setup_gsas2_refiner_instprm_Zero")
        instprm_U = resolve_default(instprm_U, "setup_gsas2_refiner_instprm_U")
        instprm_V = resolve_default(instprm_V, "setup_gsas2_refiner_instprm_V")
        instprm_W = resolve_default(instprm_W, "setup_gsas2_refiner_instprm_W")
        instprm_X = resolve_default(instprm_X, "setup_gsas2_refiner_instprm_X")
        instprm_Y = resolve_default(instprm_Y, "setup_gsas2_refiner_instprm_Y")
        instprm_Z = resolve_default(instprm_Z, "setup_gsas2_refiner_instprm_Z")
        instprm_SHL = resolve_default(instprm_SHL, "setup_gsas2_refiner_instprm_SHL")
        do_1st_refinement = resolve_default(do_1st_refinement, "setup_gsas2_refiner_do_1st_refinement")
        yshift_multiplier = resolve_default(yshift_multiplier, "setup_gsas2_refiner_yshift_multiplier")
        normalize = resolve_default(normalize, "setup_gsas2_refiner_normalize")
        normalize_to = resolve_default(normalize_to, "setup_gsas2_refiner_normalize_to")
        plot = resolve_default(plot, "setup_gsas2_refiner_plot")
        excluded_regions_unit = resolve_default(excluded_regions_unit, "setup_gsas2_refiner_excluded_regions_unit")

        self._validate_gof_improvement(min_gof_improvement)
        validate_profile(self.ds.i1d)
        q_to_two_theta(self.ds.i1d.radial.values, self.ds.i1d.attrs["wavelength_in_angst"])
        if not getattr(self, "phases", None):
            raise ValueError("Load at least one phase before setting up refinement")
        file_sources = [s for s in (instprm_from_gpx, instprm_from_nc, instprm_file) if s is not None]
        if len(file_sources) > 1 or (len(file_sources) == 1 and instprm_preset is not None):
            raise ValueError("Specify only one instrument parameter source or preset")
        if instprm_from_gpx is not None and not os.path.isfile(instprm_from_gpx):
            raise FileNotFoundError(instprm_from_gpx)
        if instprm_from_nc is not None and not os.path.isfile(instprm_from_nc):
            raise FileNotFoundError(instprm_from_nc)
        if instprm_file is not None and not os.path.isfile(instprm_file):
            raise FileNotFoundError(instprm_file)
        if instprm_preset is not None and instprm_preset not in INSTPRM_PRESETS:
            raise ValueError(
                f"Unknown instprm_preset '{instprm_preset}'. "
                f"Available presets: {list(INSTPRM_PRESETS.keys())}"
            )
        self.ds = clear_fit_results(self.ds)
        self.ds.attrs.pop("excluded_regions_q", None)
        self.ds.attrs.pop("excluded_regions_tth", None)
        self._clear_internal_standard_attrs()
        for attr in ("normalized_to", "normalization_multiplier"):
            self.ds.i1d.attrs.pop(attr, None)

        for k in ["i1d_refined", "i1d_gsas_background"]:
            if k in self.ds.keys():
                del self.ds[k]

        if hasattr(self, "gsasii_lib_path"):
            del self.gsasii_lib_path
        if hasattr(self, "gpx"):
            del self.gpx

        self.yshift_multiplier = yshift_multiplier


        # Resolve GSAS-II library
        candidate_paths = []
        if gsasii_lib_path:
            candidate_paths.append(gsasii_lib_path)
        if os.environ.get("GSASII_PATH"):
            candidate_paths.append(os.environ["GSASII_PATH"])
        default_lib = easyxrd_defaults.get("gsasii_lib_path")
        if default_lib and default_lib not in ("none", "not found") and os.path.isdir(default_lib):
            candidate_paths.append(default_lib)
        candidate_paths.append(
            os.path.join(os.path.expanduser("~"), "g2full","GSAS-II")
        )
        candidate_paths.append(
            os.path.join(os.path.expanduser("~"), "g2main","GSAS-II")
        )
        candidate_paths.append(
            os.path.join(os.path.expanduser("~"), "G2")
        )
        candidate_paths.append(
            os.path.join(os.path.expanduser("~"), ".easyxrd_scratch","GSAS-II")
        )
        candidate_paths.append(
            os.path.join(os.path.expanduser("~"), ".easyxrd_scratch","g2main","GSAS-II")
        )
        candidate_paths.append(
            os.path.join(os.path.expanduser("~"), ".easyxrd_scratch","g2full","GSAS-II")
        )
        candidate_paths.append(
            os.path.join(os.path.expanduser("~"), ".easyxrd_scratch","G2")
        )

        G2sc = load_gsas_module("GSASIIscriptable", candidate_paths)
        self.gsasii_lib_path = str(Path(G2sc.__file__).resolve().parent)

        self.easyxrd_scratch_directory = easyxrd_defaults["easyxrd_scratch_path"]

        randstr = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))

        self.gsasii_run_directory = "%s/%d_%s.gsastmp" % (
            self.easyxrd_scratch_directory,
            int(time.time()),
            randstr,
        )

        os.makedirs(self.gsasii_run_directory, exist_ok=True)

        if normalize:
            if not np.isfinite(normalize_to) or normalize_to <= 0:
                raise ValueError("normalize_to must be finite and positive")
            # find normalization scale from i1d
            if "i1d_baseline" in self.ds.keys():
                da_baseline_sub = self.ds.i1d - self.ds.i1d_baseline
                normalization_multiplier = normalize_to * (
                    1 / self._normalization_peak(da_baseline_sub.values)
                )
            else:
                da = self.ds.i1d
                normalization_multiplier = normalize_to * (1 / self._normalization_peak(da.values))

            self.ds.i1d.attrs["normalization_multiplier"] = normalization_multiplier
            self.ds.i1d.attrs["normalized_to"] = normalize_to

        data_x = q_to_two_theta(self.ds.i1d.radial.values, self.ds.i1d.attrs["wavelength_in_angst"])
        if "i1d_baseline" in self.ds.keys():
            if "normalized_to" in self.ds.i1d.attrs:
                data_y = (
                    self.ds.i1d.attrs["normalization_multiplier"]
                    * (self.ds.i1d - self.ds.i1d_baseline).values
                    + self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"]
                )
            else:
                data_y = (self.ds.i1d - self.ds.i1d_baseline).values
                # data_y = data_y + max(data_y)*self.yshift_multiplier
        else:
            if "normalized_to" in self.ds.i1d.attrs:
                data_y = (
                    self.ds.i1d.attrs["normalization_multiplier"] * (self.ds.i1d).values
                )
            else:
                data_y = self.ds.i1d.values

        if not np.all(np.isfinite(data_y)):
            raise ValueError("Refinement intensities must be finite")
        if "i1d_variance" in self.ds:
            measurement_variance = self.ds.i1d_variance.values.copy()
        else:
            warnings.warn("No measurement variance supplied: using unit weights; GoF and uncertainties lack a counting-statistics interpretation", UserWarning)
            measurement_variance = np.ones_like(data_y)
        if "i1d_baseline_variance" in self.ds:
            measurement_variance += self.ds.i1d_baseline_variance.values
        if not np.all(np.isfinite(measurement_variance)) or np.any(measurement_variance <= 0):
            raise ValueError("Refinement variance must be finite and positive")
        multiplier = self.ds.i1d.attrs.get("normalization_multiplier", 1.)
        data_variance = measurement_variance * multiplier**2
        # The GSAS-II text reader clips nonpositive y even with a sigma column.
        # Restore the original signed observations and explicit weights below.
        np.savetxt(f"{self.gsasii_run_directory}/data.xy",
                   np.column_stack((data_x, data_y, np.sqrt(data_variance))), fmt="%.12e")

        instrument = dict(zip(
            ("Polariz.", "Azimuth", "Zero", "U", "V", "W", "X", "Y", "Z", "SH/L"),
            (instprm_Polariz, instprm_Azimuth, instprm_Zero, instprm_U, instprm_V,
             instprm_W, instprm_X, instprm_Y, instprm_Z, instprm_SHL)))
        if instprm_preset is not None:
            instrument.update(INSTPRM_PRESETS[instprm_preset])
        elif instprm_file is not None:
            with open(instprm_file, "r") as stream:
                for line in stream:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k, v = k.strip(), v.strip()
                        if k in instrument:
                            try:
                                instrument[k] = float(v)
                            except ValueError:
                                instrument[k] = v
        elif instprm_from_gpx is not None:
            project = G2sc.G2Project(gpxfile=str(instprm_from_gpx))
            histograms = project.histograms()
            if not histograms:
                raise ValueError("Instrument source has no powder histograms")
            source = histograms[0].data["Instrument Parameters"][0]
            instrument.update({key: source[key][1] for key in instrument})
        elif instprm_from_nc is not None:
            with xr.open_dataset(instprm_from_nc) as source:
                instrument.update({key: source.attrs["gsasii_inst_prm_" + key] for key in instrument})
        instrument_source = instprm_from_gpx if instprm_from_gpx is not None else (
            instprm_from_nc if instprm_from_nc is not None else instprm_file
        )
        self.ds.attrs["refinement_provenance"] = provenance_json({
            "schema": 1, "software": software_versions("easyxrd", "GSAS-II"),
            "instrument_source": (
                {"preset": instprm_preset}
                if instprm_preset is not None
                else file_record(instrument_source)
            ),
            "instrument_parameters": instrument,
            "wavelength_angstrom": self.ds.i1d.attrs["wavelength_in_angst"],
            "weighting": "measurement_variance" if "i1d_variance" in self.ds else "unit_weights",
            "measured_background_variance": "i1d_baseline_variance" in self.ds,
            "normalization_multiplier": multiplier,
            "uncertainty_limitations": "Fixed baseline/scale; pixel-splitting covariance omitted",
        })
        with open(f"{self.gsasii_run_directory}/gsas.instprm", "w") as stream:
            stream.write("#GSAS-II instrument parameter file; do not add/delete items!\nType:PXC\nBank:1.0\n")
            stream.write(f"Lam:{self.ds.i1d.attrs['wavelength_in_angst']}\n")
            for key, value in instrument.items():
                stream.write(f"{key}:{value}\n")

        with nullcontext() if self.verbose else HiddenPrints():
            self.gpx = G2sc.G2Project(newgpx=f"{self.gsasii_run_directory}/gsas.gpx")
            self.gpx.data["Controls"]["data"]["max cyc"] = 100
            hist = self.gpx.add_powder_histogram(f"{self.gsasii_run_directory}/data.xy",
                                               f"{self.gsasii_run_directory}/gsas.instprm")
            hist.data["data"][1][1][:] = data_y
            hist.data["data"][1][2][:] = 1 / data_variance
            self.export_phases(export_to=self.gsasii_run_directory, export_extension=".cif")
            for phase in self.phases:
                self.gpx.add_phase(f"{self.gsasii_run_directory}/{phase}.cif", phasename=phase,
                                   histograms=[hist], fmthint="CIF")
        pwdr_name = self._histogram_name
        if "i1d_baseline" in self.ds.keys():
            if "normalized_to" in self.ds.i1d.attrs:
                self.gpx[pwdr_name]["Background"][0] = [
                    "chebyschev-1",
                    False,
                    1,
                    self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"],
                ]
            else:
                self.gpx[pwdr_name]["Background"][0] = ["chebyschev-1", False, 1, 0]
        else:
            if "normalized_to" in self.ds.i1d.attrs:
                self.gpx[pwdr_name]["Background"][0] = [
                    "chebyschev-1",
                    False,
                    1,
                    self.ds.i1d.attrs["normalization_multiplier"]
                    * np.nanmin(self.ds.i1d.values),
                ]
            else:
                self.gpx[pwdr_name]["Background"][0] = [
                    "chebyschev-1",
                    False,
                    1,
                    np.nanmin(self.ds.i1d.values),
                ]

        self._reset_rwp_history()
        if excluded_regions is not None:
            self.set_excluded_regions(
                excluded_regions=excluded_regions,
                unit=excluded_regions_unit,
                save_gpx=False,
            )
        if do_1st_refinement:

            _ = self.refine(
                update_ds=False,
                update_ds_phases=False,
                update_phases=False,
                update_previous_ds=True,
                update_previous_gpx=False,
                update_previous_phases=False,
                verbose=False,
                min_gof_improvement=None,
                log_label="setup_initial",
            )

            self.gpx.set_refinement(
                {
                    "set": {
                        "Background": {
                            "refine": False,
                            "type": "chebyschev-1",
                            "no. coeffs": 1,
                        }
                    }
                }
            )
            self.gpx.set_refinement({"set": {"LeBail": True}}, phase="all")

            ref_str = self.refine(
                update_ds=True,
                update_ds_phases=True,
                update_phases=False,
                update_previous_ds=True,
                update_previous_gpx=False,
                update_previous_phases=False,
                verbose=False,
                min_gof_improvement=min_gof_improvement,
                log_label="setup_LeBail",
            )

            title_str = self._report_refinement(
                "1st refinement with LeBail is completed. %s" % ref_str, ref_str
            )

            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=None,
                    phases=self.phases,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    title_str=title_str,
                    export_fig_as=None,
                    plot_hint="1st_refinement",
                )

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_background(
        self,
        num_coeffs=DEFAULT,
        background_type=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
    ):
        """Refine the background, requiring the specified percent GoF decrease."""
        num_coeffs = resolve_default(num_coeffs, "refine_background_num_coeffs")
        background_type = resolve_default(background_type, "refine_background_background_type")
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_background_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_background_plot")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_background_min_gof_improvement")

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self.gpx.set_refinement(
            {
                "set": {
                    "Background": {
                        "refine": True,
                        "type": background_type,
                        "no. coeffs": num_coeffs,
                    }
                }
            }
        )

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=False,
            min_gof_improvement=min_gof_improvement,
            log_label="refine_background",
        )
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_background",
            )
        title_str = "Background with %d coeffs is refined. %s" % (num_coeffs, ref_str)
        title_str = self._report_refinement(title_str, ref_str)

        if set_to_false_after_refinement:
            self.gpx.set_refinement({"set": {"Background": {"refine": False}}})
        self.gpx_saver()

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="refine_background",
                title_str=title_str.replace("✨", "").replace("❗", ""),
            )

    def set_background_refinement(
        self,
        set_num_coeffs_to=DEFAULT,
        set_background_type_to=DEFAULT,
        set_refine_to=DEFAULT,
        save_gpx=DEFAULT,
    ):
        """Configure the background model and refinement flag; optionally save without fitting."""
        set_num_coeffs_to = resolve_default(set_num_coeffs_to, "set_background_refinement_set_num_coeffs_to")
        set_background_type_to = resolve_default(set_background_type_to, "set_background_refinement_set_background_type_to")
        set_refine_to = resolve_default(set_refine_to, "set_background_refinement_set_refine_to")
        save_gpx = resolve_default(save_gpx, "set_background_refinement_save_gpx")

        self.gpx.set_refinement(
            {
                "set": {
                    "Background": {
                        "refine": set_refine_to,
                        "type": set_background_type_to,
                        "no. coeffs": set_num_coeffs_to,
                    }
                }
            }
        )

        if save_gpx:
            self.gpx_saver()

    def clear_background_refinement(self, save_gpx=DEFAULT):
        """Disable background refinement and optionally save the project."""
        save_gpx = resolve_default(save_gpx, "clear_background_refinement_save_gpx")

        self.gpx.set_refinement(
            {
                "set": {
                    "Background": {
                        "refine": False,
                    }
                }
            }
        )

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_instrument_parameters(
        self,
        inst_pars_to_refine=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
    ):
        """
        Refine instrument parameters with a minimum percent GoF decrease.
        inst_pars_to_refine=['U', 'V', 'W',   'X', 'Y', 'Z', 'Zero', 'SH/L']
        """
        inst_pars_to_refine = resolve_default(inst_pars_to_refine, "refine_instrument_parameters_inst_pars_to_refine")
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_instrument_parameters_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_instrument_parameters_plot")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_instrument_parameters_min_gof_improvement")

        sample_broadening_active = False
        if hasattr(self, "gpx") and self.gpx is not None:
            hist_name = getattr(self, "_histogram_name", "PWDR data.xy")
            try:
                phases = self.gpx.phases() if hasattr(self.gpx, "phases") else []
                for p in phases:
                    hap = getattr(p, "data", {}).get("Histograms", {}).get(hist_name, {})
                    size_data = hap.get("Size")
                    mustrain_data = hap.get("Mustrain")
                    size_active = bool(
                        size_data
                        and (
                            (len(size_data) > 2 and any(size_data[2]) if isinstance(size_data[2], (list, tuple)) else bool(size_data[2]))
                            or (len(size_data) > 5 and any(size_data[5]) if isinstance(size_data[5], (list, tuple)) else bool(size_data[5]))
                        )
                    )
                    mustrain_active = bool(
                        mustrain_data
                        and (
                            (len(mustrain_data) > 2 and any(mustrain_data[2]) if isinstance(mustrain_data[2], (list, tuple)) else bool(mustrain_data[2]))
                            or (len(mustrain_data) > 5 and any(mustrain_data[5]) if isinstance(mustrain_data[5], (list, tuple)) else bool(mustrain_data[5]))
                        )
                    )
                    if size_active or mustrain_active:
                        sample_broadening_active = True
                        break
            except Exception:
                pass

        if sample_broadening_active:
            warnings.warn(
                "Refining instrument parameters simultaneously with sample size or strain broadening "
                "causes severe parameter correlation and singular covariance. "
                "Instrument parameters should be calibrated using a standard reference material "
                "(SRM, e.g. NIST 660 LaB6) with sample broadening fixed.",
                UserWarning,
                stacklevel=2,
            )

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self.gpx.set_refinement({"set": {"Instrument Parameters": inst_pars_to_refine}})

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=False,
            min_gof_improvement=min_gof_improvement,
            log_label="refine_instrument_parameters",
        )
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_instrument_parameters",
            )
        title_str = "Instrument parameter %s is refined. %s" % (
            inst_pars_to_refine,
            ref_str,
        )
        title_str = self._report_refinement(title_str, ref_str)

        if set_to_false_after_refinement:
            ParDict = {
                "clear": {
                    "Instrument Parameters": [
                        "X",
                        "Y",
                        "Z",
                        "Zero",
                        "SH/L",
                        "U",
                        "V",
                        "W",
                    ]
                }
            }
            self.gpx.set_refinement(ParDict)
        self.gpx_saver()

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="refine_instrument_parameters",
                title_str=title_str.replace("✨", "").replace("❗", ""),
            )

    def set_instrument_parameters_refinement(
        self, set_inst_pars_to_refine=DEFAULT, set_refine_to=DEFAULT, save_gpx=DEFAULT
    ):
        """Enable or disable the named instrument parameters; optionally save without fitting."""
        set_inst_pars_to_refine = resolve_default(set_inst_pars_to_refine, "set_instrument_parameters_refinement_set_inst_pars_to_refine")
        set_refine_to = resolve_default(set_refine_to, "set_instrument_parameters_refinement_set_refine_to")
        save_gpx = resolve_default(save_gpx, "set_instrument_parameters_refinement_save_gpx")

        self.gpx.set_refinement(
            {"set" if set_refine_to else "clear": {"Instrument Parameters": set_inst_pars_to_refine}}
        )

        if save_gpx:
            self.gpx_saver()

    def clear_instrument_parameters_refinement(self, save_gpx=DEFAULT):
        """Disable CW profile, zero-shift and axial-divergence refinement flags."""
        save_gpx = resolve_default(save_gpx, "clear_instrument_parameters_refinement_save_gpx")

        self.gpx.set_refinement(
            {
                "clear": {
                    "Instrument Parameters": [
                        "X",
                        "Y",
                        "Z",
                        "Zero",
                        "SH/L",
                        "U",
                        "V",
                        "W",
                    ]
                }
            }
        )

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    def set_excluded_regions(
        self, excluded_regions=None, unit=DEFAULT, save_gpx=DEFAULT
    ):
        """Set excluded regions [(low, high), ...] in Q (Å^-1) or 2theta (deg).

        Excluded regions are masked out of GSAS-II residual and covariance
        evaluations via ``histogram.data["Limits"][2:]`` and stored in
        ``ds.attrs["excluded_regions_q"]`` and ``ds.attrs["excluded_regions_tth"]``.
        Passing ``None`` or an empty sequence clears all excluded regions.
        """
        unit = resolve_default(unit, "set_excluded_regions_unit")
        save_gpx = resolve_default(save_gpx, "set_excluded_regions_save_gpx")

        if not hasattr(self, "gpx") or self.gpx is None or not self.gpx.histograms():
            raise ValueError("Set up GSAS-II refiner before configuring excluded regions")
        if not excluded_regions:
            return self.clear_excluded_regions(save_gpx=save_gpx)

        unit_norm = str(unit).strip().lower()
        if unit_norm not in ("q", "tth", "2theta", "two_theta"):
            raise ValueError("unit must be 'q' or 'tth' ('2theta')")

        wavelength = float(self.ds.i1d.attrs["wavelength_in_angst"])
        histogram = self.gpx.histograms()[0]
        limits = histogram.data["Limits"]
        tth_min, tth_max = float(limits[0][0]), float(limits[0][1])

        q_pairs = []
        tth_pairs = []
        for region in excluded_regions:
            if not isinstance(region, (list, tuple, np.ndarray)) or len(region) != 2:
                raise ValueError("Each excluded region must be a (low, high) pair")
            low, high = float(region[0]), float(region[1])
            if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
                raise ValueError(
                    f"Excluded region ({low}, {high}) must have finite bounds with low < high"
                )
            if unit_norm == "q":
                if low < 0:
                    raise ValueError("Q excluded region bounds must be nonnegative")
                tth_bounds = q_to_two_theta(np.asarray([low, high], dtype=float), wavelength)
                tth_low, tth_high = float(tth_bounds[0]), float(tth_bounds[1])
                q_low, q_high = low, high
            else:
                if low < 0 or high >= 180.0:
                    raise ValueError("2theta excluded region bounds must be in [0, 180) degrees")
                tth_low, tth_high = low, high
                q_low = float(4.0 * np.pi * np.sin(np.deg2rad(tth_low / 2.0)) / wavelength)
                q_high = float(4.0 * np.pi * np.sin(np.deg2rad(tth_high / 2.0)) / wavelength)

            if tth_high <= tth_min or tth_low >= tth_max:
                raise ValueError(
                    f"Excluded region ({low}, {high}) does not overlap the histogram range"
                )
            q_pairs.append([q_low, q_high])
            tth_pairs.append([tth_low, tth_high])

        data_arr = histogram.data.get("data", [None, None])[1]
        if isinstance(data_arr, np.ma.MaskedArray):
            data_arr.mask = np.ma.nomask
        limits[2:] = tth_pairs
        self.ds.attrs["excluded_regions_q"] = provenance_json(q_pairs)
        self.ds.attrs["excluded_regions_tth"] = provenance_json(tth_pairs)

        if save_gpx:
            self.gpx_saver()

    def clear_excluded_regions(self, save_gpx=DEFAULT):
        """Clear all excluded regions from the active GSAS-II histogram."""
        save_gpx = resolve_default(save_gpx, "clear_excluded_regions_save_gpx")

        if not hasattr(self, "gpx") or self.gpx is None or not self.gpx.histograms():
            raise ValueError("Set up GSAS-II refiner before clearing excluded regions")
        histogram = self.gpx.histograms()[0]
        data_arr = histogram.data.get("data", [None, None])[1]
        if isinstance(data_arr, np.ma.MaskedArray):
            data_arr.mask = np.ma.nomask
        histogram.data["Limits"][2:] = []
        self.ds.attrs.pop("excluded_regions_q", None)
        self.ds.attrs.pop("excluded_regions_tth", None)

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    def _sample_parameters_dict(self):
        """Return the mutable Sample Parameters dictionary for the active histogram."""
        if not hasattr(self, "gpx") or self.gpx is None:
            return {}
        hist_name = self._histogram_name
        try:
            hist_entry = self.gpx[hist_name]
            if isinstance(hist_entry, dict) and "Sample Parameters" in hist_entry:
                return hist_entry["Sample Parameters"]
        except Exception:
            pass
        try:
            hist = self.gpx.histograms()[0]
            if "Sample Parameters" not in hist.data:
                hist.data["Sample Parameters"] = {"Type": "Bragg-Brentano", "Scale": [1.0, True]}
            return hist.data["Sample Parameters"]
        except Exception:
            return {}

    def _configure_sample_parameters(self, sample_pars, refine_flag):
        """Validate and set GSAS-II Sample Parameters and goniometer Type."""
        if isinstance(sample_pars, str):
            pars = [sample_pars]
        else:
            try:
                pars = [str(p) for p in sample_pars]
            except TypeError as exc:
                raise ValueError("sample_pars_to_refine must be a parameter name or sequence of names") from exc
        if not pars:
            raise ValueError("sample_pars_to_refine must not be empty")
        unknown = [p for p in pars if p not in ALL_SAMPLE_PARS]
        if unknown:
            raise ValueError(
                f"Unknown sample parameter(s) {unknown}. "
                f"Supported parameters: {list(ALL_SAMPLE_PARS)}"
            )
        has_bragg = any(p in BRAGG_SAMPLE_PARS for p in pars)
        has_debye = any(p in DEBYE_SAMPLE_PARS for p in pars)
        if has_bragg and has_debye:
            raise ValueError(
                "Cannot mix Bragg-Brentano ('Shift', 'Transparency', 'SurfRoughA', 'SurfRoughB') "
                "and Debye-Scherrer ('DisplaceX', 'DisplaceY', 'Absorption') sample parameters"
            )
        sample_dict = self._sample_parameters_dict()
        if "Scale" not in sample_dict or not isinstance(sample_dict["Scale"], list) or len(sample_dict["Scale"]) < 2:
            sample_dict["Scale"] = [1.0, True]
        if has_bragg:
            sample_dict["Type"] = "Bragg-Brentano"
            for bragg_par in BRAGG_SAMPLE_PARS:
                if (
                    bragg_par not in sample_dict
                    or not isinstance(sample_dict[bragg_par], list)
                    or len(sample_dict[bragg_par]) < 2
                ):
                    sample_dict[bragg_par] = [0.0, False]
            if refine_flag:
                for other in DEBYE_SAMPLE_PARS:
                    if isinstance(sample_dict.get(other), list) and len(sample_dict[other]) > 1:
                        sample_dict[other][1] = False
        elif has_debye:
            sample_dict["Type"] = "Debye-Scherrer"
            for debye_par in DEBYE_SAMPLE_PARS:
                if (
                    debye_par not in sample_dict
                    or not isinstance(sample_dict[debye_par], list)
                    or len(sample_dict[debye_par]) < 2
                ):
                    sample_dict[debye_par] = [0.0, False]
            if refine_flag:
                for other in BRAGG_SAMPLE_PARS:
                    if isinstance(sample_dict.get(other), list) and len(sample_dict[other]) > 1:
                        sample_dict[other][1] = False
        for par in pars:
            default_val = 1.0 if par == "Scale" else 0.0
            if par not in sample_dict or not isinstance(sample_dict[par], list) or len(sample_dict[par]) < 2:
                sample_dict[par] = [default_val, bool(refine_flag)]
            else:
                sample_dict[par][1] = bool(refine_flag)
        return pars

    @refinement_transaction
    def refine_sample_parameters(
        self,
        sample_pars_to_refine=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
    ):
        """Refine sample displacement/transparency/absorption parameters.

        Supports Bragg-Brentano ('Shift', 'Transparency', 'SurfRoughA', 'SurfRoughB')
        and Debye-Scherrer ('DisplaceX', 'DisplaceY', 'Absorption') parameters,
        automatically configuring the histogram Sample Parameters 'Type'.
        """
        sample_pars_to_refine = resolve_default(
            sample_pars_to_refine, "refine_sample_parameters_sample_pars_to_refine"
        )
        set_to_false_after_refinement = resolve_default(
            set_to_false_after_refinement, "refine_sample_parameters_set_to_false_after_refinement"
        )
        plot = resolve_default(plot, "refine_sample_parameters_plot")
        min_gof_improvement = resolve_default(
            min_gof_improvement, "refine_sample_parameters_min_gof_improvement"
        )

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        pars = self._configure_sample_parameters(sample_pars_to_refine, True)

        try:
            inst_dict = self.gpx.histograms()[0].data.get("Instrument Parameters", [{}])[0]
            zero_entry = inst_dict.get("Zero")
            zero_active = bool(isinstance(zero_entry, (list, tuple)) and len(zero_entry) > 2 and zero_entry[2])
        except Exception:
            zero_active = False
        if zero_active and any(p in ("Shift", "Transparency", "DisplaceX", "DisplaceY") for p in pars):
            warnings.warn(
                "Refining sample displacement/transparency parameters simultaneously with instrument 'Zero' "
                "causes severe parameter correlation with unit-cell parameters. "
                "Calibrate and fix instrument 'Zero' before refining sample displacement.",
                UserWarning,
                stacklevel=2,
            )

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=False,
            min_gof_improvement=min_gof_improvement,
            log_label="refine_sample_parameters",
        )
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_sample_parameters",
            )
        title_str = "Sample parameter %s is refined. %s" % (pars, ref_str)
        title_str = self._report_refinement(title_str, ref_str)

        if set_to_false_after_refinement:
            self.clear_sample_parameters_refinement(save_gpx=False)
        self.gpx_saver()

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="refine_sample_parameters",
                title_str=title_str.replace("✨", "").replace("❗", ""),
            )

    def set_sample_parameters_refinement(
        self, set_sample_pars_to_refine=DEFAULT, set_refine_to=DEFAULT, save_gpx=DEFAULT
    ):
        """Enable or disable the named sample parameters; optionally save without fitting."""
        set_sample_pars_to_refine = resolve_default(
            set_sample_pars_to_refine, "set_sample_parameters_refinement_set_sample_pars_to_refine"
        )
        set_refine_to = resolve_default(
            set_refine_to, "set_sample_parameters_refinement_set_refine_to"
        )
        save_gpx = resolve_default(save_gpx, "set_sample_parameters_refinement_save_gpx")

        self._configure_sample_parameters(set_sample_pars_to_refine, set_refine_to)
        if save_gpx:
            self.gpx_saver()

    def clear_sample_parameters_refinement(self, save_gpx=DEFAULT):
        """Disable all sample displacement, transparency, roughness, and absorption flags."""
        save_gpx = resolve_default(save_gpx, "clear_sample_parameters_refinement_save_gpx")

        sample_dict = self._sample_parameters_dict()
        for par in BRAGG_SAMPLE_PARS + DEBYE_SAMPLE_PARS:
            if isinstance(sample_dict.get(par), list) and len(sample_dict[par]) > 1:
                sample_dict[par][1] = False

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def set_LeBail(
        self, to=DEFAULT, phase_ind=DEFAULT, refine=DEFAULT, plot=DEFAULT,
        min_gof_improvement=DEFAULT,
    ):
        """Set LeBail mode; when refining, require the minimum percent GoF decrease."""
        to = resolve_default(to, "set_LeBail_to")
        min_gof_improvement = resolve_default(min_gof_improvement, "set_LeBail_min_gof_improvement")
        phase_ind = resolve_default(phase_ind, "set_LeBail_phase_ind")
        refine = resolve_default(refine, "set_LeBail_refine")
        plot = resolve_default(plot, "set_LeBail_plot")

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement({"set": {"LeBail": to}})
        else:
            self.gpx.set_refinement({"set": {"LeBail": to}}, phase=phase_ind)

        if refine:
            ref_str = self.refine(
                update_ds=True,
                update_ds_phases=False,
                update_phases=False,
                update_previous_ds=True,
                update_previous_gpx=True,
                update_previous_phases=False,
                min_gof_improvement=min_gof_improvement,
                log_label="set_LeBail:True" if to else "set_LeBail:False",
            )
            if not self.last_refinement_accepted:
                return self._restore_rejected_refinement(
                    gpx_before_refinement, ref_str, plot=plot,
                    plot_hint="set_LeBail",
                )
            if to:
                title_str = "After setting LeBail refinement to True, %s" % (ref_str)
                title_str = self._report_refinement(title_str, ref_str)

            else:
                title_str = "After setting LeBail refinement to False, %s" % (ref_str)
                title_str = self._report_refinement(title_str, ref_str)
        else:
            pass

        self.gpx_saver()

        if plot and refine:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="set_LeBail",
                title_str=title_str.replace("✨", "").replace("❗", ""),
            )

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def _refine_cell_scale_trials(self, scales, phase_ind, min_gof_improvement):
        """Refine independent lattice starts, retaining the best accepted fit."""
        indices = (list(range(len(self.gpx.phases()))) if phase_ind == "all"
                   else [phase_ind])
        # Resolve the selection before changing any state.
        for index in indices:
            self.gpx.phases()[index]
        if "rwp_history" not in self.ds:
            self._reset_rwp_history()
        names = ("gpx", "ds", "phases", "gpx_previous", "ds_previous",
                 "phases_previous")
        original = {name: getattr(self, name) for name in names if hasattr(self, name)}

        def restore(state):
            for name in names:
                if name in state:
                    setattr(self, name, state[name])
                elif hasattr(self, name):
                    delattr(self, name)

        def with_history(dataset, history):
            return xr.merge([dataset.drop_dims("refinement_step"), history], join="exact")

        history = self.ds[["rwp_history", "rwp_attempted_history"]].copy(deep=True)
        original_rwp = self._project_rwp(self.gpx)
        best_rwp = original_rwp if np.isfinite(original_rwp) else np.inf
        best = None
        best_scale = None
        best_message = "No lattice scale trial improved Rwp; previous fit retained."
        results = []
        try:
            for scale in scales:
                restore(copy.deepcopy(original))
                self.ds = with_history(self.ds, history)
                label = f"refine_cell_parameters(scale={scale:g})"
                previous_steps = self.ds.sizes["refinement_step"]
                try:
                    for index in indices:
                        cell = self.gpx.phases()[index].data["General"]["Cell"]
                        cell[1:4] = [length * scale for length in cell[1:4]]
                        cell[7] *= scale ** 3
                    self.gpx.set_refinement({"set": {"Cell": True}}, phase=phase_ind)
                    message = self.refine(
                        min_gof_improvement=min_gof_improvement, log_label=label,
                    )
                    rwp = float(self.ds.rwp_attempted_history.values[-1])
                    reason = self.last_refinement_rejection_reason or ""
                    valid_cells = all(
                        np.all(np.isfinite(p.data["General"]["Cell"][1:8]))
                        and np.all(np.asarray(p.data["General"]["Cell"][1:4]) > 0)
                        and np.all(np.asarray(p.data["General"]["Cell"][4:7]) > 0)
                        and np.all(np.asarray(p.data["General"]["Cell"][4:7]) < 180)
                        and p.data["General"]["Cell"][7] > 0
                        for p in self.gpx.phases()
                    )
                    if not np.isfinite(rwp) or rwp < 0 or not valid_cells:
                        reason = "invalid_cell_or_rwp"
                    elif self.last_refinement_accepted:
                        if rwp < best_rwp:
                            best_rwp, best_scale = rwp, float(scale)
                            best_message = f"Best lattice scale {scale:g}. {message}"
                            best = copy.deepcopy({
                                name: getattr(self, name) for name in names
                                if hasattr(self, name)
                            })
                        else:
                            reason = "no_rwp_improvement"
                    status = "rejected" if reason else "accepted"
                except Exception as exc:
                    # Numerical failures at one start must not hide a valid fit
                    # at another. Keep a diagnostic for every attempted scale.
                    status, reason, rwp = "error", f"{type(exc).__name__}: {exc}", np.nan
                    if self.ds.sizes["refinement_step"] == previous_steps:
                        self._record_rwp(np.nan, status, label, reason)

                history = self.ds[["rwp_history", "rwp_attempted_history"]].copy(deep=True)
                # Assigning through .values truncates longer strings, so replace
                # the coordinate using a new array with an adequate string dtype.
                for key, value in (("refinement_status", status),
                                   ("refinement_reason", reason)):
                    values = history[key].values.tolist()
                    values[-1] = value
                    history = history.assign_coords({key: ("refinement_step", values)})
                history.rwp_history.values[-1] = (
                    np.nan if status == "error" else
                    best_rwp if np.isfinite(best_rwp) else original_rwp
                )
                results.append({"scale": float(scale), "rwp": rwp,
                                "status": status, "reason": reason})
                logger.info("Lattice scale %g: Rwp=%.6g (%s)", scale, rwp, status)
        finally:
            # The saved GPX, profile, CIF structures and comparison snapshots
            # must all describe the same winner, including after a failed trial.
            restore(best if best is not None else original)
            self.ds = with_history(self.ds, history)
            self.cell_scale_trials = results
            self.cell_scale_best = best_scale
            self.last_refinement_accepted = best is not None
            self.last_refinement_rejection_reason = (
                None if best is not None else "no_scale_improvement"
            )
            if best is not None:
                self.gpx_previous = copy.deepcopy(original["gpx"])
                for index, p in enumerate(original["gpx"].phases()):
                    cell_orig = p.data["General"]["Cell"]
                    for offset, parameter in enumerate(
                        ("a", "b", "c", "alpha", "beta", "gamma"), start=1
                    ):
                        self.ds.attrs[f"PhaseInd_{index}_cell_{parameter}_previous"] = (
                            cell_orig[offset]
                        )
                    self.ds.attrs[f"PhaseInd_{index}_cell_volume_previous"] = (
                        float(cell_orig[7])
                        if len(cell_orig) > 7
                        else compute_cell_volume(cell_orig[1:7])
                    )
            self.gpx_saver()
        if best is None:
            print(" ❌--" + best_message)
            logger.warning("%s", best_message)
        return best_message

    @refinement_transaction
    def refine_cell_parameters(
        self,
        phase_ind=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        report=DEFAULT,
        min_gof_improvement=DEFAULT,
        scale_trials=DEFAULT,
    ):
        """Refine unit cells, optionally trying several initial lattice scales.

        scale_trials is a nonempty sequence of positive, finite factors, e.g.
        np.linspace(0.98, 1.01, 5). Each factor multiplies a, b and c of the
        selected phase(s), relative to the cells at entry; angles are unchanged.
        With phase_ind='all' (or None), the same factor applies to every phase.
        Each trial starts from the same project. Keep the lowest finite Rwp
        that improves the original fit and passes the Uiso and specified GoF
        checks. Failed trials are logged and skipped; if none improves the fit,
        restore the original project. Only the retained result is plotted.

        cell_scale_trials contains per-scale diagnostics and cell_scale_best
        the winning factor (None if no improvement). Trials also appear in
        ds.rwp_attempted_history with their scale in refinement_label. Omitting
        scale_trials preserves the usual single-refinement behavior.
        """
        scale_trials = resolve_default(scale_trials, "refine_cell_parameters_scale_trials")
        phase_ind = resolve_default(phase_ind, "refine_cell_parameters_phase_ind")
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_cell_parameters_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_cell_parameters_plot")
        report = resolve_default(report, "refine_cell_parameters_report")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_cell_parameters_min_gof_improvement")

        if scale_trials is not None:
            try:
                scales = np.asarray(list(scale_trials), dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("scale_trials must be a nonempty 1D sequence of positive finite factors") from exc
            if (scales.ndim != 1 or not scales.size or
                    not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
                raise ValueError("scale_trials must be a nonempty 1D sequence of positive finite factors")

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        if phase_ind is None:
            phase_ind = "all"
        if scale_trials is not None:
            ref_str = self._refine_cell_scale_trials(scales, phase_ind, min_gof_improvement)
        else:
            self.gpx.set_refinement({"set": {"Cell": True}}, phase=phase_ind)

            ref_str = self.refine(
                update_ds=True,
                update_ds_phases=True,
                update_phases=True,
                update_previous_ds=True,
                update_previous_gpx=True,
                update_previous_phases=True,
                min_gof_improvement=min_gof_improvement,
                log_label="refine_cell_parameters",
            )
        if not self.last_refinement_accepted:
            if scale_trials is not None:
                return None
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_cell_parameters",
            )
        if (phase_ind == "all") or (phase_ind == None):
            if len(self.phases) > 1:
                title_str = "Cell parameters of all phases are refined. %s" % (ref_str)
            else:
                title_str = "Cell parameters are refined. %s" % (ref_str)
            title_str = self._report_refinement(title_str, ref_str)
            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=self.ds_previous,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    plot_hint="refine_cell_parameters",
                    title_str=title_str.replace("✨", "").replace("❗", ""),
                )
        else:
            title_str = "Cell parameters of %s phase are refined. %s" % (
                self.gpx.phases()[phase_ind].name,
                ref_str,
            )
            title_str = self._report_refinement(title_str, ref_str)
            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=self.ds_previous,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    plot_hint="refine_cell_parameters",
                    title_str=title_str.replace("✨", "").replace("❗", ""),
                )

        if report:
            for e, si in enumerate(range(self.ds.attrs["num_phases"])):
                site_ind = si

                site_label = self.ds.attrs["PhaseInd_%d_label" % site_ind]
                site_SGSys = self.ds.attrs["PhaseInd_%d_SGSys" % site_ind]
                site_SpGrp = self.ds.attrs["PhaseInd_%d_SpGrp" % site_ind]

                site_a = self.ds.attrs["PhaseInd_%d_cell_a" % site_ind]
                site_b = self.ds.attrs["PhaseInd_%d_cell_b" % site_ind]
                site_c = self.ds.attrs["PhaseInd_%d_cell_c" % site_ind]
                site_alpha = self.ds.attrs["PhaseInd_%d_cell_alpha" % site_ind]
                site_beta = self.ds.attrs["PhaseInd_%d_cell_beta" % site_ind]
                site_gamma = self.ds.attrs["PhaseInd_%d_cell_gamma" % site_ind]

                site_a_previous = self.ds.attrs[
                    "PhaseInd_%d_cell_a_previous" % site_ind
                ]
                site_b_previous = self.ds.attrs[
                    "PhaseInd_%d_cell_b_previous" % site_ind
                ]
                site_c_previous = self.ds.attrs[
                    "PhaseInd_%d_cell_c_previous" % site_ind
                ]
                site_alpha_previous = self.ds.attrs[
                    "PhaseInd_%d_cell_alpha_previous" % site_ind
                ]
                site_beta_previous = self.ds.attrs[
                    "PhaseInd_%d_cell_beta_previous" % site_ind
                ]
                site_gamma_previous = self.ds.attrs[
                    "PhaseInd_%d_cell_gamma_previous" % site_ind
                ]

                site_a_str = format_value_with_su(
                    site_a, self.ds.attrs.get("PhaseInd_%d_cell_a_su" % site_ind), precision=5
                )
                site_b_str = format_value_with_su(
                    site_b, self.ds.attrs.get("PhaseInd_%d_cell_b_su" % site_ind), precision=5
                )
                site_c_str = format_value_with_su(
                    site_c, self.ds.attrs.get("PhaseInd_%d_cell_c_su" % site_ind), precision=5
                )
                site_alpha_str = format_value_with_su(
                    site_alpha, self.ds.attrs.get("PhaseInd_%d_cell_alpha_su" % site_ind), precision=2
                )
                site_beta_str = format_value_with_su(
                    site_beta, self.ds.attrs.get("PhaseInd_%d_cell_beta_su" % site_ind), precision=2
                )
                site_gamma_str = format_value_with_su(
                    site_gamma, self.ds.attrs.get("PhaseInd_%d_cell_gamma_su" % site_ind), precision=2
                )

                report_str = (
                    "\n%s-phase\n\n%s (%s)\n\n     refined (old) \
                    \n a=%s (%.5f) \n b=%s (%.5f) \n c=%s (%.5f) \
                    \n \\alpha=%s (%.2f) \n \\beta=%s (%.2f) \n \\gamma=%s (%.2f) "
                    % (
                        site_label,
                        site_SpGrp.replace(" ", ""),
                        site_SGSys,
                        site_a_str,
                        site_a_previous,
                        site_b_str,
                        site_b_previous,
                        site_c_str,
                        site_c_previous,
                        site_alpha_str,
                        site_alpha_previous,
                        site_beta_str,
                        site_beta_previous,
                        site_gamma_str,
                        site_gamma_previous,
                    )
                )
                if f"PhaseInd_{site_ind}_cell_volume" in self.ds.attrs:
                    vol_str = format_value_with_su(
                        self.ds.attrs[f"PhaseInd_{site_ind}_cell_volume"],
                        self.ds.attrs.get(f"PhaseInd_{site_ind}_cell_volume_su"),
                        precision=3,
                    )
                    vol_prev = self.ds.attrs.get(
                        f"PhaseInd_{site_ind}_cell_volume_previous", np.nan
                    )
                    report_str += f"\n V={vol_str} ({vol_prev:.3f}) "

                logger.info("%s", report_str)

        if set_to_false_after_refinement:
            self.gpx.set_refinement({"set": {"Cell": False}}, phase=phase_ind)
        self.gpx_saver()

    def set_cell_parameters_refinement(
        self, set_refine_to=DEFAULT, phase_ind=DEFAULT, save_gpx=DEFAULT
    ):
        """Set cell-refinement flags for a phase index or all phases; do not run a fit."""
        set_refine_to = resolve_default(set_refine_to, "set_cell_parameters_refinement_set_refine_to")
        phase_ind = resolve_default(phase_ind, "set_cell_parameters_refinement_phase_ind")
        save_gpx = resolve_default(save_gpx, "set_cell_parameters_refinement_save_gpx")

        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement({"set": {"Cell": set_refine_to}})
        else:
            self.gpx.set_refinement({"set": {"Cell": set_refine_to}}, phase=phase_ind)

        if save_gpx:
            self.gpx_saver()

    def clear_cell_parameters_refinement(self, save_gpx=DEFAULT):
        """Disable cell refinement for all phases and optionally save."""
        save_gpx = resolve_default(save_gpx, "clear_cell_parameters_refinement_save_gpx")

        self.gpx.set_refinement({"set": {"Cell": False}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_strain_broadening(
        self,
        phase_ind=DEFAULT,
        model_type=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        report=DEFAULT,
        min_gof_improvement=DEFAULT,
        type=None,
    ):
        """Refine strain broadening, requiring the specified percent GoF decrease."""
        phase_ind = resolve_default(phase_ind, "refine_strain_broadening_phase_ind")
        model_type = resolve_default(model_type, "refine_strain_broadening_model_type")
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_strain_broadening_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_strain_broadening_plot")
        report = resolve_default(report, "refine_strain_broadening_report")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_strain_broadening_min_gof_improvement")

        actual_type = type if type is not None else model_type
        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self.gpx.set_refinement(
            {"set": {"Mustrain": {"refine": True, "type": actual_type}}}, phase=phase_ind
        )

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=True,
            min_gof_improvement=min_gof_improvement,
            log_label="refine_strain_broadening",
        )
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_strain_broadening",
            )
        if (phase_ind == "all") or (phase_ind == None):
            title_str = "Strain broadening of all phases are refined. %s" % (ref_str)
            title_str = self._report_refinement(title_str, ref_str)
            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=self.ds_previous,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    plot_hint="refine_strain_broadening",
                    title_str=title_str.replace("✨", "").replace("❗", ""),
                )
        else:
            title_str = "Strain broadening of %s phase is refined. %s" % (
                self.gpx.phases()[phase_ind].name,
                ref_str,
            )
            title_str = self._report_refinement(title_str, ref_str)
            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=self.ds_previous,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    plot_hint="refine_strain_broadening",
                    title_str=title_str.replace("✨", "").replace("❗", ""),
                )

        if report:
            for e, si in enumerate(range(self.ds.attrs["num_phases"])):
                site_ind = si

                site_label = self.ds.attrs["PhaseInd_%d_label" % site_ind]
                site_SGSys = self.ds.attrs["PhaseInd_%d_SGSys" % site_ind]
                site_SpGrp = self.ds.attrs["PhaseInd_%d_SpGrp" % site_ind]

                site_strain_broadening_type = self.ds.attrs[
                    "PhaseInd_%d_strain_broadening_type" % site_ind
                ]
                site_mustrain_0 = self.ds.attrs["PhaseInd_%d_mustrain_0" % site_ind]
                site_mustrain_1 = self.ds.attrs["PhaseInd_%d_mustrain_1" % site_ind]
                site_mustrain_2 = self.ds.attrs["PhaseInd_%d_mustrain_2" % site_ind]

                site_strain_broadening_type_previous = self.ds.attrs[
                    "PhaseInd_%d_strain_broadening_type_previous" % site_ind
                ]
                site_mustrain_0_previous = self.ds.attrs[
                    "PhaseInd_%d_mustrain_0_previous" % site_ind
                ]
                site_mustrain_1_previous = self.ds.attrs[
                    "PhaseInd_%d_mustrain_1_previous" % site_ind
                ]
                site_mustrain_2_previous = self.ds.attrs[
                    "PhaseInd_%d_mustrain_2_previous" % site_ind
                ]

                report_str = (
                    "\n%s-phase\n\n%s (%s)\n\n     refined (old) \
                    \n type=%s (%s) \n mustrain_0=%.5f (%.5f) \n mustrain_1=%.5f (%.5f) \n mustrain_2=%.5f (%.5f) "
                    % (
                        site_label,
                        site_SpGrp.replace(" ", ""),
                        site_SGSys,
                        site_strain_broadening_type,
                        site_strain_broadening_type_previous,
                        site_mustrain_0,
                        site_mustrain_0_previous,
                        site_mustrain_1,
                        site_mustrain_1_previous,
                        site_mustrain_2,
                        site_mustrain_2_previous,
                    )
                )

                logger.info("%s", report_str)

        if set_to_false_after_refinement:
            self.gpx.set_refinement(
                {"set": {"Mustrain": {"refine": False}}}, phase=phase_ind
            )
        self.gpx_saver()

    def set_strain_broadening_refinement(
        self, set_refine_to=DEFAULT, phase_ind=DEFAULT, model_type=DEFAULT, save_gpx=DEFAULT, type=None
    ):
        """Select a microstrain model and flag for one or all phases; do not run a fit."""
        set_refine_to = resolve_default(set_refine_to, "set_strain_broadening_refinement_set_refine_to")
        phase_ind = resolve_default(phase_ind, "set_strain_broadening_refinement_phase_ind")
        model_type = resolve_default(model_type, "set_strain_broadening_refinement_model_type")
        save_gpx = resolve_default(save_gpx, "set_strain_broadening_refinement_save_gpx")

        actual_type = type if type is not None else model_type
        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement(
                {"set": {"Mustrain": {"refine": set_refine_to, "type": actual_type}}}
            )
        else:
            self.gpx.set_refinement(
                {"set": {"Mustrain": {"refine": set_refine_to, "type": actual_type}}},
                phase=phase_ind,
            )

        if save_gpx:
            self.gpx_saver()

    def clear_strain_broadening_refinement(self, save_gpx=DEFAULT):
        """Disable microstrain refinement for all phases and optionally save."""
        save_gpx = resolve_default(save_gpx, "clear_strain_broadening_refinement_save_gpx")

        self.gpx.set_refinement({"set": {"Mustrain": {"refine": False}}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_size_broadening(
        self,
        phase_ind=DEFAULT,
        model_type=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        report=DEFAULT,
        min_gof_improvement=DEFAULT,
        type=None,
    ):
        """Refine size broadening, requiring the specified percent GoF decrease."""
        phase_ind = resolve_default(phase_ind, "refine_size_broadening_phase_ind")
        model_type = resolve_default(model_type, "refine_size_broadening_model_type")
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_size_broadening_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_size_broadening_plot")
        report = resolve_default(report, "refine_size_broadening_report")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_size_broadening_min_gof_improvement")

        actual_type = type if type is not None else model_type
        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self.gpx.set_refinement(
            {"set": {"Size": {"refine": True, "type": actual_type}}}, phase=phase_ind
        )

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=True,
            min_gof_improvement=min_gof_improvement,
            log_label="refine_size_broadening",
        )
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_size_broadening",
            )
        if (phase_ind == "all") or (phase_ind == None):
            title_str = "Size broadening of all phases are refined. %s" % (ref_str)
            title_str = self._report_refinement(title_str, ref_str)
            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=self.ds_previous,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    plot_hint="refine_size_broadening",
                    title_str=title_str.replace("✨", "").replace("❗", ""),
                )
        else:
            title_str = "Size broadening of %s phase is refined. %s" % (
                self.gpx.phases()[phase_ind].name,
                ref_str,
            )
            title_str = self._report_refinement(title_str, ref_str)
            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=self.ds_previous,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    plot_hint="refine_size_broadening",
                    title_str=title_str.replace("✨", "").replace("❗", ""),
                )

        if report:
            for e, si in enumerate(range(self.ds.attrs["num_phases"])):
                site_ind = si

                site_label = self.ds.attrs["PhaseInd_%d_label" % site_ind]
                site_SGSys = self.ds.attrs["PhaseInd_%d_SGSys" % site_ind]
                site_SpGrp = self.ds.attrs["PhaseInd_%d_SpGrp" % site_ind]

                site_strain_broadening_type = self.ds.attrs[
                    "PhaseInd_%d_size_broadening_type" % site_ind
                ]
                site_size_0 = self.ds.attrs["PhaseInd_%d_size_0" % site_ind]
                site_size_1 = self.ds.attrs["PhaseInd_%d_size_1" % site_ind]
                site_size_2 = self.ds.attrs["PhaseInd_%d_size_2" % site_ind]

                site_strain_broadening_type_previous = self.ds.attrs[
                    "PhaseInd_%d_size_broadening_type_previous" % site_ind
                ]
                site_size_0_previous = self.ds.attrs[
                    "PhaseInd_%d_size_0_previous" % site_ind
                ]
                site_size_1_previous = self.ds.attrs[
                    "PhaseInd_%d_size_1_previous" % site_ind
                ]
                site_size_2_previous = self.ds.attrs[
                    "PhaseInd_%d_size_2_previous" % site_ind
                ]

                report_str = (
                    "\n%s-phase\n\n%s (%s)\n\n     refined (old) \
                    \n type=%s (%s) \n size_0=%.5f (%.5f) \n size_1=%.5f (%.5f) \n size_2=%.5f (%.5f) "
                    % (
                        site_label,
                        site_SpGrp.replace(" ", ""),
                        site_SGSys,
                        site_strain_broadening_type,
                        site_strain_broadening_type_previous,
                        site_size_0,
                        site_size_0_previous,
                        site_size_1,
                        site_size_1_previous,
                        site_size_2,
                        site_size_2_previous,
                    )
                )

                logger.info("%s", report_str)

        if set_to_false_after_refinement:
            self.gpx.set_refinement(
                {"set": {"Size": {"refine": False}}}, phase=phase_ind
            )
        self.gpx_saver()

    def set_size_broadening_refinement(
        self, set_refine_to=DEFAULT, phase_ind=DEFAULT, model_type=DEFAULT, save_gpx=DEFAULT, type=None
    ):
        """Select a crystallite-size model and flag for one or all phases; do not run a fit."""
        set_refine_to = resolve_default(set_refine_to, "set_size_broadening_refinement_set_refine_to")
        phase_ind = resolve_default(phase_ind, "set_size_broadening_refinement_phase_ind")
        model_type = resolve_default(model_type, "set_size_broadening_refinement_model_type")
        save_gpx = resolve_default(save_gpx, "set_size_broadening_refinement_save_gpx")

        actual_type = type if type is not None else model_type
        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement(
                {"set": {"Size": {"refine": set_refine_to, "type": actual_type}}}
            )
        else:
            self.gpx.set_refinement(
                {"set": {"Size": {"refine": set_refine_to, "type": actual_type}}},
                phase=phase_ind,
            )

        if save_gpx:
            self.gpx_saver()

    def clear_size_broadening_refinement(self, save_gpx=DEFAULT):
        """Disable crystallite-size refinement for all phases and optionally save."""
        save_gpx = resolve_default(save_gpx, "clear_size_broadening_refinement_save_gpx")

        self.gpx.set_refinement({"set": {"Size": {"refine": False}}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def _resolve_phase_index(self, phase_selector):
        """Resolve an integer index or phase name to a 0-based phase index."""
        n_phases = int(
            self.ds.attrs.get(
                "num_phases",
                len(self.gpx.phases()) if hasattr(self, "gpx") and self.gpx is not None else len(getattr(self, "phases", {})),
            )
        )
        if isinstance(phase_selector, (int, np.integer)) and not isinstance(phase_selector, bool):
            idx = int(phase_selector)
            if idx < 0 or idx >= n_phases:
                raise ValueError(f"Phase index {idx} out of range (0..{n_phases - 1})")
            return idx
        name_str = str(phase_selector).strip()
        for idx in range(n_phases):
            if self.ds.attrs.get(f"PhaseInd_{idx}_label") == name_str:
                return idx
        if hasattr(self, "gpx") and self.gpx is not None:
            for idx, p in enumerate(self.gpx.phases()):
                if p.name == name_str:
                    return idx
        if hasattr(self, "phases") and isinstance(self.phases, dict):
            for idx, k in enumerate(self.phases.keys()):
                if k == name_str:
                    return idx
        raise ValueError(f"Unknown phase '{phase_selector}'")

    @refinement_transaction
    def refine_phase_fractions(
        self,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
        internal_standard_phase=None,
        standard_wt_fraction=None,
        standard_wt_fraction_su=0.0,
    ):
        """Refine phase fractions, requiring the specified percent GoF decrease.

        Optionally compute amorphous content and renormalized weight fractions
        via the internal-standard method when ``internal_standard_phase`` and
        ``standard_wt_fraction`` (wt% in the spiked mixture) are supplied.
        """
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_phase_fractions_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_phase_fractions_plot")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_phase_fractions_min_gof_improvement")

        if self._phase_mode() != "Rietveld":
            raise ValueError("Quantitative phase fractions require LeBail=False for all phases")
        if (internal_standard_phase is None) != (standard_wt_fraction is None):
            raise ValueError(
                "Specify both internal_standard_phase and standard_wt_fraction, or neither"
            )
        if internal_standard_phase is not None:
            self._resolve_phase_index(internal_standard_phase)
            w_std_val = float(standard_wt_fraction)
            if not np.isfinite(w_std_val) or w_std_val <= 0 or w_std_val >= 100:
                raise ValueError(
                    "standard_wt_fraction must be finite and strictly between 0 and 100 wt%"
                )
            w_std_su_val = float(standard_wt_fraction_su) if standard_wt_fraction_su is not None else 0.0
            if not np.isfinite(w_std_su_val) or w_std_su_val < 0:
                raise ValueError("standard_wt_fraction_su must be finite and nonnegative")

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self.gpx[self._histogram_name]["Sample Parameters"]["Scale"][1] = False
        for e, p in enumerate(self.phases):
            self.gpx["Phases"][p]["Histograms"][self._histogram_name]["Scale"][1] = True

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=True,
            min_gof_improvement=min_gof_improvement,
            log_label="refine_phase_fractions",
        )
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_phase_fractions",
            )
        if internal_standard_phase is not None:
            self.compute_amorphous_fraction(
                internal_standard_phase=internal_standard_phase,
                standard_wt_fraction=standard_wt_fraction,
                standard_wt_fraction_su=standard_wt_fraction_su,
            )
        title_str = "Phase fractions of all phases are refined. %s" % (ref_str)
        title_str = self._report_refinement(title_str, ref_str)

        if set_to_false_after_refinement:
            self.gpx[self._histogram_name]["Sample Parameters"]["Scale"][1] = True
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"][self._histogram_name]["Scale"][1] = False

        self.gpx_saver()

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="refine_phase_fractions",
                title_str=title_str.replace("✨", "").replace("❗", ""),
            )

    def compute_amorphous_fraction(
        self,
        internal_standard_phase,
        standard_wt_fraction,
        standard_wt_fraction_su=0.0,
    ):
        """Compute amorphous and renormalized crystalline weight fractions (wt%) via an internal standard.

        Parameters
        ----------
        internal_standard_phase : int or str
            Phase index or phase name of the spiked internal standard (e.g. 'Al2O3' or 0).
        standard_wt_fraction : float
            Known weight percentage (0 < wt% < 100) of the internal standard in the spiked mixture.
        standard_wt_fraction_su : float, optional
            Standard uncertainty on ``standard_wt_fraction`` (in wt%, default 0.0).

        Returns
        -------
        dict
            Dictionary with ``spiked_wt_fractions``, ``spiked_wt_fractions_su``,
            ``amorphous_spiked_wt_fraction``, ``amorphous_spiked_wt_fraction_su``,
            ``original_wt_fractions``, ``original_wt_fractions_su``,
            ``amorphous_wt_fraction``, and ``amorphous_wt_fraction_su``.
        """
        if not hasattr(self, "ds") or self.ds is None:
            raise ValueError("Dataset 'ds' is not initialized")
        if not self.ds.attrs.get("weight_fractions_valid", False):
            raise ValueError(
                "Valid Rietveld phase weight fractions are required before computing amorphous content"
            )
        std_idx = self._resolve_phase_index(internal_standard_phase)
        n_phases = int(
            self.ds.attrs.get(
                "num_phases",
                len(self.gpx.phases()) if hasattr(self, "gpx") and self.gpx is not None else len(getattr(self, "phases", {})),
            )
        )
        std_label = self.ds.attrs.get(
            f"PhaseInd_{std_idx}_label",
            self.gpx.phases()[std_idx].name if hasattr(self, "gpx") and self.gpx is not None else str(std_idx),
        )

        if hasattr(self, "gpx") and self.gpx is not None and self.gpx.phases():
            histogram = self.gpx.histograms()[0]
            h_id = self._histogram_id(histogram)
            unit_masses = np.array(
                [p.data["General"]["Mass"] for p in self.gpx.phases()], dtype=float
            )
            scales = np.array(
                [p.data["Histograms"][self._histogram_name]["Scale"][0] for p in self.gpx.phases()],
                dtype=float,
            )
            scale_names = [
                f"{p.data.get('pId', idx)}:{h_id}:Scale"
                for idx, p in enumerate(self.gpx.phases())
            ]
            cov_data = self.gpx["Covariance"]["data"]
        else:
            unit_masses = np.ones(n_phases, dtype=float)
            scales = np.array(
                [float(self.ds.attrs[f"PhaseInd_{i}_wt_fraction"]) for i in range(n_phases)],
                dtype=float,
            )
            scale_names = [f"{i}:0:Scale" for i in range(n_phases)]
            cov_data = {}

        qpa = compute_internal_standard_qpa(
            unit_masses=unit_masses,
            scales=scales,
            scale_names=scale_names,
            covariance=cov_data,
            std_index=std_idx,
            std_wt_fraction=standard_wt_fraction,
            std_wt_fraction_su=standard_wt_fraction_su,
        )

        self.ds.attrs["internal_standard_phase"] = str(std_label)
        self.ds.attrs["internal_standard_phase_ind"] = int(std_idx)
        self.ds.attrs["internal_standard_wt_fraction"] = float(standard_wt_fraction)
        self.ds.attrs["internal_standard_wt_fraction_su"] = float(
            0.0 if standard_wt_fraction_su is None else standard_wt_fraction_su
        )
        self.ds.attrs["amorphous_spiked_wt_fraction"] = qpa["amorphous_spiked_wt_fraction"]
        self.ds.attrs["amorphous_spiked_wt_fraction_su"] = qpa["amorphous_spiked_wt_fraction_su"]
        self.ds.attrs["amorphous_wt_fraction"] = qpa["amorphous_wt_fraction"]
        self.ds.attrs["amorphous_wt_fraction_su"] = qpa["amorphous_wt_fraction_su"]
        for i in range(n_phases):
            self.ds.attrs[f"PhaseInd_{i}_spiked_wt_fraction"] = float(qpa["spiked_wt_fractions"][i])
            self.ds.attrs[f"PhaseInd_{i}_spiked_wt_fraction_su"] = float(
                qpa["spiked_wt_fractions_su"][i]
            )
            self.ds.attrs[f"PhaseInd_{i}_original_wt_fraction"] = float(
                qpa["original_wt_fractions"][i]
            )
            self.ds.attrs[f"PhaseInd_{i}_original_wt_fraction_su"] = float(
                qpa["original_wt_fractions_su"][i]
            )
        return qpa

    def set_phase_fractions_refinement(self, set_refine_to=DEFAULT, save_gpx=DEFAULT):
        """Toggle phase scales against the histogram scale; use Rietveld mode for fractions."""
        set_refine_to = resolve_default(set_refine_to, "set_phase_fractions_refinement_set_refine_to")
        save_gpx = resolve_default(save_gpx, "set_phase_fractions_refinement_save_gpx")

        if set_refine_to:
            if self._phase_mode() != "Rietveld":
                raise ValueError("Quantitative phase fractions require LeBail=False for all phases")
            self.gpx[self._histogram_name]["Sample Parameters"]["Scale"][1] = False
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"][self._histogram_name]["Scale"][1] = True
        else:
            self.gpx[self._histogram_name]["Sample Parameters"]["Scale"][1] = True
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"][self._histogram_name]["Scale"][1] = False
        if save_gpx:
            self.gpx_saver()

    def clear_phase_fractions_refinement(self, save_gpx=DEFAULT):
        """Disable phase scales and enable the histogram scale; optionally save."""
        save_gpx = resolve_default(save_gpx, "clear_phase_fractions_refinement_save_gpx")

        self.gpx[self._histogram_name]["Sample Parameters"]["Scale"][1] = True
        for e, p in enumerate(self.phases):
            self.gpx["Phases"][p]["Histograms"][self._histogram_name]["Scale"][1] = False
        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_preferred_orientation(
        self,
        phase_ind=DEFAULT,
        harmonics_order=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
        model=DEFAULT,
        md_axis=DEFAULT,
        md_ratio=DEFAULT,
    ):
        """Refine preferred orientation with a minimum percent GoF decrease.

        Parameters
        ----------
        phase_ind : int, str, or None, optional
            Index or name of the phase to refine, or 'all'/None for all phases (default 'all').
        harmonics_order : int, optional
            Even integer spherical harmonic order when model='SH' (default 4).
        set_to_false_after_refinement : bool, optional
            Whether to disable refinement flag after refinement (default True).
        plot : bool, optional
            Whether to plot result after refinement (default False).
        min_gof_improvement : float, optional
            Minimum percent GoF improvement required to accept result (default 1.0).
        model : {'SH', 'MD'}, optional
            Preferred orientation model: 'SH' for Spherical Harmonics, 'MD' for
            March-Dollase (default 'SH').
        md_axis : sequence of 3 ints, optional
            Crystallographic preferred orientation axis [h, k, l] when model='MD'
            (default (0, 0, 1)).
        md_ratio : float, optional
            Initial March-Dollase ratio r when model='MD' (default 1.0).
        """
        phase_ind = resolve_default(phase_ind, "refine_preferred_orientation_phase_ind")
        harmonics_order = resolve_default(harmonics_order, "refine_preferred_orientation_harmonics_order")
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_preferred_orientation_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_preferred_orientation_plot")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_preferred_orientation_min_gof_improvement")
        model = resolve_default(model, "refine_preferred_orientation_model")
        md_axis = resolve_default(md_axis, "refine_preferred_orientation_md_axis")
        md_ratio = resolve_default(md_ratio, "refine_preferred_orientation_md_ratio")

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self._set_po(
            phase_ind,
            harmonics_order,
            True,
            model=model,
            md_axis=md_axis,
            md_ratio=md_ratio,
        )

        ref_str = self.refine(update_ds=True, update_ds_phases=False, update_phases=False,
                              min_gof_improvement=min_gof_improvement,
                              log_label="refine_preferred_orientation")
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(gpx_before_refinement, ref_str, plot=plot,
                                                     plot_hint="refine_preferred_orientation")
        desc = (
            f"March-Dollase preferred orientation (axis={list(md_axis)})"
            if str(model).upper().strip() == "MD"
            else f"Spherical harmonics preferred orientation (order={harmonics_order})"
        )
        title_str = self._report_refinement(f"{desc} refined. {ref_str}", ref_str)

        if set_to_false_after_refinement:
            phases = self.gpx.phases()
            selected = (
                range(len(phases))
                if phase_ind in ("all", None)
                else (
                    [phase_ind]
                    if isinstance(phase_ind, (int, np.integer))
                    else [i for i, p in enumerate(phases) if p.name == phase_ind]
                )
            )
            for index in selected:
                po = phases[index].data["Histograms"][self._histogram_name].get("Pref.Ori.")
                if po and len(po) > 2:
                    po[2] = False

        self.gpx_saver()

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="refine_preferred_orientation",
                title_str=title_str.replace("✨", "").replace("❗", ""),
            )

    def set_preferred_orientation_refinement(
        self,
        set_refine_to=DEFAULT,
        phase_ind=DEFAULT,
        harmonics_order_to=DEFAULT,
        model=DEFAULT,
        md_axis=DEFAULT,
        md_ratio=DEFAULT,
        save_gpx=DEFAULT,
    ):
        """Configure preferred orientation (Spherical Harmonics or March-Dollase) and set refinement flag."""
        set_refine_to = resolve_default(set_refine_to, "set_preferred_orientation_refinement_set_refine_to")
        phase_ind = resolve_default(phase_ind, "set_preferred_orientation_refinement_phase_ind")
        harmonics_order_to = resolve_default(harmonics_order_to, "set_preferred_orientation_refinement_harmonics_order_to")
        model = resolve_default(model, "set_preferred_orientation_refinement_model")
        md_axis = resolve_default(md_axis, "set_preferred_orientation_refinement_md_axis")
        md_ratio = resolve_default(md_ratio, "set_preferred_orientation_refinement_md_ratio")
        save_gpx = resolve_default(save_gpx, "set_preferred_orientation_refinement_save_gpx")

        self._set_po(
            phase_ind,
            harmonics_order_to,
            set_refine_to,
            model=model,
            md_axis=md_axis,
            md_ratio=md_ratio,
        )

        if save_gpx:
            self.gpx_saver()

    def clear_preferred_orientation_refinement(self, phase_ind=DEFAULT, save_gpx=DEFAULT):
        """Disable preferred-orientation refinement for selected or all phases and optionally save."""
        phase_ind = resolve_default(phase_ind, "clear_preferred_orientation_refinement_phase_ind")
        save_gpx = resolve_default(save_gpx, "clear_preferred_orientation_refinement_save_gpx")

        phases = self.gpx.phases()
        selected = (
            range(len(phases))
            if phase_ind in ("all", None)
            else (
                [phase_ind]
                if isinstance(phase_ind, (int, np.integer))
                else [i for i, p in enumerate(phases) if p.name == phase_ind]
            )
        )
        for index in selected:
            po = phases[index].data["Histograms"][self._histogram_name].get("Pref.Ori.")
            if po and len(po) > 2:
                po[2] = False

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_site_property(
        self,
        phase_ind=DEFAULT,
        site_ind=DEFAULT,
        refinement_flags=DEFAULT,
        set_to_false_after_refinement=DEFAULT,
        plot=DEFAULT,
        min_gof_improvement=DEFAULT,
    ):
        """Refine a site, accepting only a sufficient percentage decrease in GoF.

        Rejected trials preserve the project, site flags, structures, dataset,
        and previous-result history. The default requires at least a 0.1% decrease
        in GoF. Set min_gof_improvement=None to disable
        the improvement check, or 0 to accept any strictly positive decrease.
        """
        phase_ind = resolve_default(phase_ind, "refine_site_property_phase_ind")
        site_ind = resolve_default(site_ind, "refine_site_property_site_ind")
        refinement_flags = resolve_default(refinement_flags, "refine_site_property_refinement_flags")
        set_to_false_after_refinement = resolve_default(set_to_false_after_refinement, "refine_site_property_set_to_false_after_refinement")
        plot = resolve_default(plot, "refine_site_property_plot")
        min_gof_improvement = resolve_default(min_gof_improvement, "refine_site_property_min_gof_improvement")

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        site_label = self.gpx.phases()[phase_ind]["Atoms"][site_ind][0]
        self.gpx.phases()[phase_ind].atom(
            site_label
        ).refinement_flags = refinement_flags

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=True,
            update_phases=True,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=True,
            min_gof_improvement=min_gof_improvement,
            log_label="refine_site_property",
        )
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(
                gpx_before_refinement, ref_str, plot=plot,
                plot_hint="refine_site_property",
            )
        title_str = "%s property of %s site of %s phase is refined. %s" % (
            refinement_flags,
            site_label,
            self.gpx.phases()[phase_ind].name,
            ref_str,
        )
        title_str = self._report_refinement(title_str, ref_str)

        if set_to_false_after_refinement:
            self.gpx.phases()[phase_ind].atom(site_label).refinement_flags = ""

        self.gpx_saver()

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=self.ds_previous,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="refine_site_property",
                title_str=title_str.replace("✨", "").replace("❗", ""),
            )

    def set_site_property_refinement(
        self, phase_ind=DEFAULT, site_ind=DEFAULT, refinement_flags=DEFAULT, save_gpx=DEFAULT
    ):
        """Set native GSAS-II atom refinement flags for one site; do not run a fit."""
        phase_ind = resolve_default(phase_ind, "set_site_property_refinement_phase_ind")
        site_ind = resolve_default(site_ind, "set_site_property_refinement_site_ind")
        refinement_flags = resolve_default(refinement_flags, "set_site_property_refinement_refinement_flags")
        save_gpx = resolve_default(save_gpx, "set_site_property_refinement_save_gpx")

        site_label = self.gpx.phases()[phase_ind]["Atoms"][site_ind][0]
        self.gpx.phases()[phase_ind].atom(
            site_label
        ).refinement_flags = refinement_flags
        if save_gpx:
            self.gpx_saver()

    def clear_site_property_refinement(self, phase_ind=DEFAULT, site_ind=DEFAULT, save_gpx=DEFAULT):
        """Clear all refinement flags for one atom and optionally save the project."""
        phase_ind = resolve_default(phase_ind, "clear_site_property_refinement_phase_ind")
        site_ind = resolve_default(site_ind, "clear_site_property_refinement_site_ind")
        save_gpx = resolve_default(save_gpx, "clear_site_property_refinement_save_gpx")

        site_label = self.gpx.phases()[phase_ind]["Atoms"][site_ind][0]
        self.gpx.phases()[phase_ind].atom(site_label).refinement_flags = ""

        if save_gpx:
            self.gpx_saver()

    @refinement_transaction
    def fine_tune_gpx(self, min_gof_improvement=DEFAULT):
        """Open the installed GSAS-II GUI, then synchronize the accepted result."""
        min_gof_improvement = resolve_default(min_gof_improvement, "fine_tune_gpx_min_gof_improvement")

        self._validate_gof_improvement(min_gof_improvement)
        filename = f"{self.gsasii_run_directory}/gsas.gpx"
        self.gpx_saver()
        subprocess.check_call([sys.executable, "-m", "GSASII", filename])
        return self._replace_project(filename, min_gof_improvement, "fine_tune_gpx")

    @refinement_transaction
    def replace_gpx_with(self, newgpx_to_replace, min_gof_improvement=DEFAULT):
        """Replace a compatible single-histogram project and synchronize results."""
        min_gof_improvement = resolve_default(min_gof_improvement, "replace_gpx_with_min_gof_improvement")

        return self._replace_project(newgpx_to_replace, min_gof_improvement, "replace_gpx_with")

    def _replace_project(self, source, minimum, label):
        previous = self._prepare_refinement(minimum)
        original_ds, original_phases = self.ds.copy(deep=True), copy.deepcopy(self.phases)
        target = f"{self.gsasii_run_directory}/gsas.gpx"
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copyfile(source, target)
        module = load_gsas_module("GSASIIscriptable", [getattr(self, "gsasii_lib_path", "")])
        self.gpx = module.G2Project(gpxfile=target)
        rejection = self._refine_gpx(previous, min_gof_improvement=minimum, log_label=label)
        if rejection:
            return rejection
        # Result synchronization must not invoke the optimizer a second time.
        self._sync_only = True
        try:
            histogram = self.gpx.histograms()[0]
            expected = q_to_two_theta(self.ds.i1d.radial, self.ds.i1d.attrs["wavelength_in_angst"])
            observed = np.asarray(np.ma.getdata(histogram.getdata("x")), dtype=float)
            if observed.shape != expected.shape or not np.allclose(observed, expected, rtol=1e-7, atol=1e-7):
                raise ValueError("Replacement histogram must match the loaded profile coordinates")
            multiplier = self.ds.i1d.attrs.get("normalization_multiplier", 1.)
            expected_y = self.ds.i1d.values.copy()
            if "i1d_baseline" in self.ds:
                expected_y -= self.ds.i1d_baseline.values
            expected_y *= multiplier
            if "i1d_baseline" in self.ds and "normalized_to" in self.ds.i1d.attrs:
                expected_y += self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"]
            if not np.allclose(np.asarray(np.ma.getdata(histogram.getdata("yobs")), dtype=float), expected_y):
                raise ValueError("Replacement histogram observations differ from the loaded data")
            self.phases = {phase.name: None for phase in self.gpx.phases()}
            self.ds.attrs = {key: value for key, value in self.ds.attrs.items() if not key.startswith("PhaseInd_")}
            self.ds.attrs["num_phases"] = len(self.phases)
            for index, name in enumerate(self.phases):
                self.ds.attrs[f"PhaseInd_{index}_label"] = name
            result = self.refine()
        finally:
            self._sync_only = False
        self.gpx_previous, self.ds_previous, self.phases_previous = previous, original_ds, original_phases
        return result

    def export_gpx_to(self, to="gsas.gpx"):
        """ """
        shutil.copy("%s/gsas.gpx" % self.gsasii_run_directory, to)

    ###############################################################################################
    def _extract_series_scans(self, series_sources, coord_values, coord_name, load_kwargs):
        """Normalize series_sources into a list of xr.Dataset objects and coordinate values."""
        base_ds = getattr(self, "ds", None)
        if not isinstance(base_ds, xr.Dataset) or "i1d" not in base_ds:
            raise ValueError("Initial dataset 'ds' with 'i1d' must be loaded before refine_series")
        wavelength = float(base_ds.i1d.attrs["wavelength_in_angst"])
        radial_ref = np.asarray(base_ds.i1d.radial.values, dtype=float)

        scans = []
        inferred_coords = None
        if isinstance(series_sources, xr.DataArray):
            if series_sources.ndim != 2 or "radial" not in series_sources.dims:
                raise ValueError(
                    "When series_sources is a DataArray, it must be 2D and include the 'radial' dimension"
                )
            other_dim = [d for d in series_sources.dims if d != "radial"][0]
            if coord_values is None and other_dim in series_sources.coords:
                inferred_coords = np.asarray(series_sources.coords[other_dim].values)
            for idx in range(series_sources.sizes[other_dim]):
                da_slice = series_sources.isel({other_dim: idx}).drop_vars(
                    other_dim, errors="ignore"
                )
                ds_item = xr.Dataset({"i1d": da_slice})
                ds_item.i1d.attrs["wavelength_in_angst"] = float(
                    da_slice.attrs.get("wavelength_in_angst", wavelength)
                )
                scans.append(ds_item)
        else:
            try:
                items = list(series_sources)
            except TypeError as exc:
                raise ValueError(
                    "series_sources must be a non-empty sequence of datasets/arrays/files or a 2D DataArray"
                ) from exc
            for item in items:
                if isinstance(item, xr.Dataset):
                    ds_item = item.copy(deep=True)
                    if "i1d" not in ds_item:
                        raise ValueError("Each Dataset in series_sources must contain 'i1d'")
                    ds_item.i1d.attrs.setdefault("wavelength_in_angst", wavelength)
                    scans.append(ds_item)
                elif isinstance(item, xr.DataArray):
                    da_item = item.copy(deep=True)
                    da_item.attrs.setdefault("wavelength_in_angst", wavelength)
                    scans.append(xr.Dataset({"i1d": da_item}))
                elif isinstance(item, dict) and "i1d" in item:
                    da_i1d = (
                        item["i1d"].copy(deep=True)
                        if isinstance(item["i1d"], xr.DataArray)
                        else xr.DataArray(
                            np.asarray(item["i1d"], dtype=float),
                            dims=("radial",),
                            coords={"radial": radial_ref},
                        )
                    )
                    da_i1d.attrs.setdefault("wavelength_in_angst", wavelength)
                    ds_item = xr.Dataset({"i1d": da_i1d})
                    for extra_k in ("i1d_variance", "i1d_baseline", "i1d_baseline_variance"):
                        if extra_k in item and item[extra_k] is not None:
                            ds_item[extra_k] = (
                                item[extra_k].copy(deep=True)
                                if isinstance(item[extra_k], xr.DataArray)
                                else xr.DataArray(
                                    np.asarray(item[extra_k], dtype=float),
                                    dims=("radial",),
                                    coords={"radial": radial_ref},
                                )
                            )
                    scans.append(ds_item)
                elif isinstance(item, (str, Path)):
                    path_obj = Path(item)
                    if not path_obj.is_file():
                        raise FileNotFoundError(str(path_obj))
                    if path_obj.suffix.lower() in (".nc", ".netcdf"):
                        with xr.open_dataset(path_obj) as loaded:
                            ds_item = loaded.load().copy(deep=True)
                        ds_item.i1d.attrs.setdefault("wavelength_in_angst", wavelength)
                        scans.append(ds_item)
                    else:
                        kw = dict(load_kwargs or {})
                        kw["plot"] = False
                        saved_ds = self.ds.copy(deep=True)
                        try:
                            self.load_xrd_data(
                                from_1d_file=str(path_obj),
                                wavelength_in_angst=kw.pop("wavelength_in_angst", wavelength),
                                **kw,
                            )
                            scans.append(self.ds.copy(deep=True))
                        finally:
                            self.ds = saved_ds
                else:
                    raise ValueError(
                        f"Unsupported series_sources element type: {type(item).__name__}"
                    )

        if not scans:
            raise ValueError("series_sources must contain at least one scan")

        for idx, ds_item in enumerate(scans):
            r_vals = np.asarray(ds_item.i1d.radial.values, dtype=float)
            if r_vals.shape != radial_ref.shape or not np.allclose(
                r_vals, radial_ref, rtol=1e-6, atol=1e-6
            ):
                raise ValueError(
                    f"Scan {idx} radial grid does not match the initialized histogram radial grid"
                )

        if coord_values is not None:
            coords = np.asarray(list(coord_values))
            if coords.ndim != 1 or coords.size != len(scans):
                raise ValueError(
                    f"coord_values length ({coords.size}) must match number of scans ({len(scans)})"
                )
        elif inferred_coords is not None and inferred_coords.size == len(scans):
            coords = inferred_coords
        else:
            coords = np.arange(len(scans), dtype=int)

        return scans, coords

    def _update_histogram_observations(self, scan_ds):
        """Update the active GSAS-II histogram observations and weights for a new scan."""
        prev_norm_to = self.ds.i1d.attrs.get("normalized_to")
        preserved_attrs = {
            k: copy.deepcopy(v)
            for k, v in self.ds.attrs.items()
            if k == "num_phases"
            or k.startswith("PhaseInd_")
            or k.startswith("excluded_regions_")
            or k.startswith("internal_standard_")
            or k == "refinement_provenance"
        }

        new_ds = scan_ds.copy(deep=True)
        wavelength = float(
            new_ds.i1d.attrs.get("wavelength_in_angst", self.ds.i1d.attrs["wavelength_in_angst"])
        )
        new_ds.i1d.attrs["wavelength_in_angst"] = wavelength
        new_ds.attrs.update(preserved_attrs)

        if prev_norm_to is not None:
            normalize_to = float(prev_norm_to)
            if "i1d_baseline" in new_ds:
                peak_scale = self._normalization_peak((new_ds.i1d - new_ds.i1d_baseline).values)
            else:
                peak_scale = self._normalization_peak(new_ds.i1d.values)
            norm_mult = normalize_to / peak_scale
            new_ds.i1d.attrs["normalization_multiplier"] = norm_mult
            new_ds.i1d.attrs["normalized_to"] = normalize_to
        else:
            norm_mult = float(new_ds.i1d.attrs.get("normalization_multiplier", 1.0))

        yshift_mult = float(getattr(self, "yshift_multiplier", 0.0))
        if "i1d_baseline" in new_ds:
            if "normalized_to" in new_ds.i1d.attrs:
                data_y = (
                    norm_mult * (new_ds.i1d - new_ds.i1d_baseline).values
                    + yshift_mult * new_ds.i1d.attrs["normalized_to"]
                )
            else:
                data_y = (new_ds.i1d - new_ds.i1d_baseline).values
        else:
            data_y = norm_mult * new_ds.i1d.values if "normalized_to" in new_ds.i1d.attrs else new_ds.i1d.values

        data_y = np.asarray(data_y, dtype=float)
        if not np.all(np.isfinite(data_y)):
            raise ValueError("Series scan intensities must be finite")

        if "i1d_variance" in new_ds:
            measurement_variance = np.asarray(new_ds.i1d_variance.values, dtype=float).copy()
        else:
            measurement_variance = np.ones_like(data_y)
        if "i1d_baseline_variance" in new_ds:
            measurement_variance = measurement_variance + np.asarray(
                new_ds.i1d_baseline_variance.values, dtype=float
            )
        if not np.all(np.isfinite(measurement_variance)) or np.any(measurement_variance <= 0):
            raise ValueError("Series scan variance must be finite and positive")

        data_variance = measurement_variance * (norm_mult ** 2)
        data_x = q_to_two_theta(new_ds.i1d.radial.values, wavelength)

        if hasattr(self, "gsasii_run_directory") and os.path.isdir(self.gsasii_run_directory):
            np.savetxt(
                f"{self.gsasii_run_directory}/data.xy",
                np.column_stack((data_x, data_y, np.sqrt(data_variance))),
                fmt="%.12e",
            )

        hist = self.gpx.histograms()[0]
        if "data" in hist.data and len(hist.data["data"]) > 1 and hist.data["data"][1] is not None:
            arr = hist.data["data"][1]
            saved_mask = np.ma.getmask(arr).copy() if isinstance(arr, np.ma.MaskedArray) else None
            if isinstance(arr, np.ma.MaskedArray):
                arr.mask = np.ma.nomask
            arr[0][:] = data_x
            arr[1][:] = data_y
            arr[2][:] = 1.0 / data_variance
            if saved_mask is not None and saved_mask is not np.ma.nomask:
                arr.mask = saved_mask

        self.ds = new_ds
        self._reset_rwp_history()

    def _run_series_step(self, step, min_gof_improvement):
        """Execute a single sequential refinement step specification."""
        step_aliases = {
            "background": "refine_background",
            "refine_background": "refine_background",
            "cell": "refine_cell_parameters",
            "refine_cell_parameters": "refine_cell_parameters",
            "phase_fractions": "refine_phase_fractions",
            "refine_phase_fractions": "refine_phase_fractions",
            "sample": "refine_sample_parameters",
            "refine_sample_parameters": "refine_sample_parameters",
            "strain": "refine_strain_broadening",
            "refine_strain_broadening": "refine_strain_broadening",
            "size": "refine_size_broadening",
            "refine_size_broadening": "refine_size_broadening",
            "instrument": "refine_instrument_parameters",
            "refine_instrument_parameters": "refine_instrument_parameters",
            "preferred_orientation": "refine_preferred_orientation",
            "refine_preferred_orientation": "refine_preferred_orientation",
            "refine": "refine",
        }
        if callable(step):
            return step(self)
        if isinstance(step, (tuple, list)) and len(step) == 2 and isinstance(step[1], dict):
            name_raw, step_kwargs = str(step[0]).strip(), dict(step[1])
        elif isinstance(step, str):
            name_raw, step_kwargs = step.strip(), {}
        else:
            raise ValueError(
                f"Invalid refine_steps entry {step!r}. Expected step name, (name, kwargs), or callable."
            )

        method_name = step_aliases.get(name_raw, name_raw)
        if not hasattr(self, method_name):
            raise ValueError(f"Unknown refinement step '{name_raw}'")
        if method_name == "refine_phase_fractions" and self._phase_mode() != "Rietveld":
            return None

        step_kwargs.setdefault("plot", False)
        step_kwargs.setdefault("min_gof_improvement", min_gof_improvement)
        if method_name in (
            "refine_cell_parameters",
            "refine_strain_broadening",
            "refine_size_broadening",
        ):
            step_kwargs.setdefault("report", False)
        return getattr(self, method_name)(**step_kwargs)

    def refine_series(
        self,
        series_sources,
        coord_values=None,
        coord_name=DEFAULT,
        load_kwargs=None,
        refine_steps=DEFAULT,
        min_gof_improvement=DEFAULT,
        stop_on_error=DEFAULT,
    ):
        """Sequentially refine a series of 1D patterns (e.g. in-situ T, t, V, P scans).

        Chains the accepted GSAS-II project state from scan k to scan k + 1,
        compiling all 1D profiles, per-phase profiles, R-factors, unit-cell
        parameters (+ s.u.), and weight fractions (+ s.u.) into ``self.series_ds``
        along ``coord_name``.

        Parameters
        ----------
        series_sources : sequence of (xr.Dataset, xr.DataArray, dict, str, Path) or 2D xr.DataArray
            Ordered sequence of 1D patterns or a 2D DataArray containing the ``radial`` dimension.
        coord_values : sequence, optional
            Coordinate values along the series dimension (e.g. temperatures or timestamps).
        coord_name : str, optional
            Name of the series coordinate dimension (default ``'series_index'``).
        load_kwargs : dict, optional
            Extra keyword arguments forwarded to ``load_xrd_data`` when loading file paths.
        refine_steps : sequence of (str, (str, dict), or callable), optional
            Sequence of refinement steps executed on each scan (default
            ``('background', 'cell', 'phase_fractions')``).
        min_gof_improvement : float or None, optional
            Minimum percent GoF improvement required per step (default ``None``).
        stop_on_error : bool, optional
            If True, re-raise exceptions immediately; if False (default), restore the
            last converged state, mark ``converged=False`` for the failed scan, and continue.

        Returns
        -------
        xr.Dataset
            Combined multi-scan dataset ``self.series_ds`` indexed along ``coord_name``.
        """
        coord_name = resolve_default(coord_name, "refine_series_coord_name")
        refine_steps = resolve_default(refine_steps, "refine_series_refine_steps")
        min_gof_improvement = resolve_default(
            min_gof_improvement, "refine_series_min_gof_improvement"
        )
        stop_on_error = resolve_default(stop_on_error, "refine_series_stop_on_error")

        self._validate_gof_improvement(min_gof_improvement)
        if not hasattr(self, "gpx") or self.gpx is None or not self.gpx.histograms():
            raise ValueError("Set up GSAS-II refiner on an initial dataset before running refine_series")
        steps_list = list(refine_steps) if refine_steps is not None else []
        scans, coords = self._extract_series_scans(
            series_sources, coord_values, coord_name, load_kwargs
        )

        n_phases = len(self.gpx.phases())
        phase_names = [str(p.name) for p in self.gpx.phases()]
        radial_coord = self.ds.i1d.radial.copy(deep=True)
        n_radial = radial_coord.size
        n_scans = len(scans)

        i1d_arr = np.full((n_scans, n_radial), np.nan, dtype=float)
        i1d_ref_arr = np.full((n_scans, n_radial), np.nan, dtype=float)
        i1d_bkg_arr = np.full((n_scans, n_radial), np.nan, dtype=float)
        phase_prof_arr = np.full((n_scans, n_phases, n_radial), np.nan, dtype=float)

        rwp_arr = np.full(n_scans, np.nan, dtype=float)
        gof_arr = np.full(n_scans, np.nan, dtype=float)
        converged_arr = np.zeros(n_scans, dtype=bool)

        scalar_keys = []
        for idx in range(n_phases):
            prefix = f"PhaseInd_{idx}_"
            for cell_k in ("a", "b", "c", "alpha", "beta", "gamma", "volume"):
                scalar_keys.append(f"{prefix}cell_{cell_k}")
                scalar_keys.append(f"{prefix}cell_{cell_k}_su")
            scalar_keys.extend([
                f"{prefix}wt_fraction",
                f"{prefix}wt_fraction_su",
                f"{prefix}size_0",
                f"{prefix}mustrain_0",
            ])
        scalar_series = {k: np.full(n_scans, np.nan, dtype=float) for k in scalar_keys}

        last_good_state = {
            "gpx": copy.deepcopy(self.gpx),
            "ds": self.ds.copy(deep=True),
            "phases": copy.deepcopy(self.phases),
        }

        for scan_idx, scan_ds in enumerate(scans):
            try:
                self._update_histogram_observations(scan_ds)
                i1d_arr[scan_idx, :] = self.ds.i1d.values
                self.refine(
                    update_ds=True,
                    update_ds_phases=False,
                    update_phases=False,
                    update_previous_ds=True,
                    update_previous_gpx=True,
                    update_previous_phases=False,
                    verbose=False,
                    plot=False,
                    min_gof_improvement=None,
                    log_label=f"series[{scan_idx}]:init",
                )
                for step in steps_list:
                    self._run_series_step(step, min_gof_improvement)

                if "i1d_refined" in self.ds:
                    i1d_ref_arr[scan_idx, :] = self.ds.i1d_refined.values
                if "i1d_gsas_background" in self.ds:
                    i1d_bkg_arr[scan_idx, :] = self.ds.i1d_gsas_background.values
                if "i1d_phase_profiles" in self.ds:
                    phase_prof_arr[scan_idx, :, :] = self.ds.i1d_phase_profiles.values

                rwp_arr[scan_idx] = float(self.ds.attrs.get("Rwp", np.nan))
                gof_arr[scan_idx] = float(self.ds.attrs.get("GOF", np.nan))
                converged_arr[scan_idx] = bool(np.isfinite(rwp_arr[scan_idx]))
                for k in scalar_keys:
                    if k in self.ds.attrs:
                        scalar_series[k][scan_idx] = float(self.ds.attrs[k])

                if converged_arr[scan_idx]:
                    last_good_state = {
                        "gpx": copy.deepcopy(self.gpx),
                        "ds": self.ds.copy(deep=True),
                        "phases": copy.deepcopy(self.phases),
                    }
            except Exception:
                if stop_on_error:
                    raise
                self.gpx = copy.deepcopy(last_good_state["gpx"])
                self.ds = last_good_state["ds"].copy(deep=True)
                self.phases = copy.deepcopy(last_good_state["phases"])
                self.gpx_saver()
                converged_arr[scan_idx] = False

        data_vars = {
            "i1d": ((coord_name, "radial"), i1d_arr),
            "i1d_refined": ((coord_name, "radial"), i1d_ref_arr),
            "i1d_gsas_background": ((coord_name, "radial"), i1d_bkg_arr),
            "i1d_phase_profiles": ((coord_name, "phase", "radial"), phase_prof_arr),
            "Rwp": ((coord_name,), rwp_arr),
            "GOF": ((coord_name,), gof_arr),
            "converged": ((coord_name,), converged_arr),
        }
        for k, arr in scalar_series.items():
            data_vars[k] = ((coord_name,), arr)

        series_ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                coord_name: coords,
                "radial": radial_coord,
                "phase": np.asarray(phase_names, dtype=str),
            },
            attrs={
                "num_phases": n_phases,
                "coord_name": str(coord_name),
                **{
                    f"PhaseInd_{idx}_label": name
                    for idx, name in enumerate(phase_names)
                },
            },
        )
        self.series_ds = series_ds
        return series_ds

    ###############################################################################################
    def plot_refinement_history(
        self,
        figsize=DEFAULT,
        dpi=DEFAULT,
        min_step_detail=DEFAULT,
        export_fig_as=None,
        show=DEFAULT,
        return_fig=DEFAULT,
    ):
        """
        Plot comprehensive refinement convergence history showing accepted values and rollbacks.
        """
        figsize = resolve_default(figsize, "plot_figsize")
        dpi = resolve_default(dpi, "plot_refinement_history_dpi")
        min_step_detail = resolve_default(min_step_detail, "plot_refinement_history_min_step_detail")
        show = resolve_default(show, "plot_refinement_history_show")
        return_fig = resolve_default(return_fig, "plot_refinement_history_return_fig")

        ds = getattr(self, "ds", None)
        return refinement_plotter(
            ds,
            figsize=figsize,
            dpi=dpi,
            min_step_detail=min_step_detail,
            export_fig_as=export_fig_as,
            show=show,
            return_fig=return_fig,
        )

    # Backward-compatible alias
    plot_refinement = plot_refinement_history
