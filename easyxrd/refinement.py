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
from .results import clear_fit_results, fit_result_dataset
from .provenance import file_record, software_versions, provenance_json
from . import easyxrd_defaults
from .plotters import exrd_plotter, refinement_plotter


logger = logging.getLogger(__name__)


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

    def _set_po(self, phase_ind, order, enabled):
        if not isinstance(order, (int, np.integer)) or order < 0 or order % 2:
            raise ValueError("Spherical harmonic order must be a nonnegative even integer")
        lattice = load_gsas_module("GSASIIlattice")
        phases = self.gpx.phases()
        selected = range(len(phases)) if phase_ind in ("all", None) else [phase_ind]
        for index in selected:
            phase = phases[index]
            symmetry = phase.data["General"]["SGData"]["SGLaue"]
            hap = phase.data["Histograms"][self._histogram_name]
            previous = hap.get("Pref.Ori.", [None] * 6)
            old = previous[5] if previous[0] == "SH" else {}
            names = lattice.GenSHCoeffT(symmetry, '0', order, False)
            coefficients = {name: old.get(name, 0.) for name in names}
            hap["Pref.Ori."] = ["SH", 1., enabled, [0, 0, 1], order, coefficients, [""], .1]

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
        logger.warning("%s", message)
        return message

    def _report_refinement(self, title_str, ref_str):
        """Keep helper messages and plot titles consistent with acceptance."""
        if not self.last_refinement_accepted:
            return ref_str
        logger.info("%s", title_str)
        return title_str

    @refinement_transaction
    def refine(
        self,
        update_ds=True,
        update_ds_phases=True,
        update_phases=True,
        update_previous_ds=True,
        update_previous_gpx=True,
        update_previous_phases=True,
        verbose=False,
        plot=False,
        min_gof_improvement=None,
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
            histogram = self.gpx.histograms()[0]
            multiplier = self.ds.i1d.attrs.get("normalization_multiplier", 1.)
            offset = (self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"]
                      if "i1d_baseline" in self.ds and "normalized_to" in self.ds.i1d.attrs else 0.)
            calculated = (histogram.getdata("ycalc") - offset) / multiplier
            background = (histogram.getdata("background") - offset) / multiplier
            self.ds["i1d_refined"] = self.ds.i1d.copy(data=calculated)
            if "i1d_baseline" in self.ds:
                self.ds["i1d_refined"] += self.ds.i1d_baseline
            self.ds["i1d_gsas_background"] = self.ds.i1d.copy(data=background)
            self.ds.attrs.update(copy.deepcopy(self.gpx["Covariance"]["data"]["Rvals"]))
            self.ds.attrs["refinement_mode"] = self._phase_mode()

            self.ds = self._store_fit_results(self.ds)
            self.ds.attrs.update(self._phase_result_attrs(self.gpx))

            masses = np.array([
                p.data["General"]["Mass"] * p.data["Histograms"][self._histogram_name]["Scale"][0]
                for p in self.gpx.phases()
            ], dtype=float)
            valid_fractions = (self._phase_mode() == "Rietveld" and np.all(np.isfinite(masses))
                               and np.all(masses >= 0) and masses.sum() > 0)
            self.ds.attrs["weight_fractions_valid"] = bool(valid_fractions)
            for index, mass in enumerate(masses):
                self.ds.attrs[f"PhaseInd_{index}_wt_fraction"] = (100 * mass / masses.sum()
                                                                 if valid_fractions else np.nan)

            self.ds.attrs.update(self._phase_result_attrs(gpx_previous, suffix="_previous"))

            inst_prm_dict = self.gpx[self._histogram_name]["Instrument Parameters"][0]
            inst_prm_dict_clean = {}
            for i in inst_prm_dict:
                inst_prm_dict_clean["gsasii_inst_prm_" + i] = inst_prm_dict[i][1]
            self.ds.attrs = self.ds.attrs | inst_prm_dict_clean

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
    def _phase_result_attrs(project, suffix=""):
        """Extract cell/profile attributes consistently for current/previous fits."""
        hist_name = project.histograms()[0].name
        attrs = {}
        for index, phase in enumerate(project.phases()):
            general = phase.data["General"]
            prefix = f"PhaseInd_{index}_"
            if not suffix:
                for key in ("SGSys", "SpGrp"):
                    attrs[prefix + key] = general["SGData"][key]
            for name, value in zip(("a", "b", "c", "alpha", "beta", "gamma"), general["Cell"][1:7]):
                attrs[f"{prefix}cell_{name}{suffix}"] = value
            hap = phase.data["Histograms"][hist_name]
            for key, kind, field in (("Size", "size", "size"), ("Mustrain", "strain", "mustrain")):
                attrs[f"{prefix}{kind}_broadening_type{suffix}"] = hap[key][0]
                for component, value in enumerate(hap[key][1][:3]):
                    attrs[f"{prefix}{field}_{component}{suffix}"] = value
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
        instprm_Polariz=0,
        instprm_Azimuth=0,
        instprm_Zero=0,
        instprm_U=100,
        instprm_V=5,
        instprm_W=0.5,
        instprm_X=0,
        instprm_Y=0,
        instprm_Z=0,
        instprm_SHL=0.002,
        do_1st_refinement=True,
        yshift_multiplier=0.01,
        normalize=False,
        normalize_to=100,
        plot=True,
        min_gof_improvement=None,
    ):
        """Create the initial fit; optionally gate the second, LeBail pass.

        The first pass establishes the baseline GoF and is always ungated.
        min_gof_improvement defaults to None during setup.
        Starts a new ds.rwp_history on the refinement_step dimension. Setup's
        initial optimizer passes are logged too; do_1st_refinement=False leaves
        an empty history until the first refinement attempt.
        """
        self._validate_gof_improvement(min_gof_improvement)
        validate_profile(self.ds.i1d)
        q_to_two_theta(self.ds.i1d.radial.values, self.ds.i1d.attrs["wavelength_in_angst"])
        if not getattr(self, "phases", None):
            raise ValueError("Load at least one phase before setting up refinement")
        if instprm_from_gpx is not None and not os.path.isfile(instprm_from_gpx):
            raise FileNotFoundError(instprm_from_gpx)
        if instprm_from_gpx is not None and instprm_from_nc is not None:
            raise ValueError("Specify only one instrument parameter source")
        self.ds = clear_fit_results(self.ds)
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
        if instprm_from_gpx is not None:
            project = G2sc.G2Project(gpxfile=str(instprm_from_gpx))
            histograms = project.histograms()
            if not histograms:
                raise ValueError("Instrument source has no powder histograms")
            source = histograms[0].data["Instrument Parameters"][0]
            instrument.update({key: source[key][1] for key in instrument})
        elif instprm_from_nc is not None:
            with xr.open_dataset(instprm_from_nc) as source:
                instrument.update({key: source.attrs["gsasii_inst_prm_" + key] for key in instrument})
        instrument_source = instprm_from_gpx if instprm_from_gpx is not None else instprm_from_nc
        self.ds.attrs["refinement_provenance"] = provenance_json({
            "schema": 1, "software": software_versions("easyxrd", "GSAS-II"),
            "instrument_source": file_record(instrument_source),
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
        num_coeffs=10,
        background_type="chebyschev-1",
        set_to_false_after_refinement=True,
        plot=False,
        min_gof_improvement=1.0,
    ):
        """Refine the background, requiring the specified percent GoF decrease."""

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
        set_num_coeffs_to=10,
        set_background_type_to="chebyschev-1",
        set_refine_to=True,
        save_gpx=True,
    ):
        """Configure the background model and refinement flag; optionally save without fitting."""

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

    def clear_background_refinement(self, save_gpx=True):
        """Disable background refinement and optionally save the project."""

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
        inst_pars_to_refine=("U", "V", "W"),
        set_to_false_after_refinement=True,
        plot=False,
        min_gof_improvement=1.0,
    ):
        """
        Refine instrument parameters with a minimum percent GoF decrease.
        inst_pars_to_refine=['U', 'V', 'W',   'X', 'Y', 'Z', 'Zero', 'SH/L']
        """

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
        self, set_inst_pars_to_refine=("U", "V", "W"), set_refine_to=True, save_gpx=True
    ):
        """Enable or disable the named instrument parameters; optionally save without fitting."""

        self.gpx.set_refinement(
            {"set" if set_refine_to else "clear": {"Instrument Parameters": set_inst_pars_to_refine}}
        )

        if save_gpx:
            self.gpx_saver()

    def clear_instrument_parameters_refinement(self, save_gpx=True):
        """Disable CW profile, zero-shift and axial-divergence refinement flags."""

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
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def set_LeBail(
        self, to=True, phase_ind="all", refine=True, plot=False,
        min_gof_improvement=None,
    ):
        """Set LeBail mode; when refining, require the minimum percent GoF decrease."""

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
                    for offset, parameter in enumerate(
                        ("a", "b", "c", "alpha", "beta", "gamma"), start=1
                    ):
                        self.ds.attrs[f"PhaseInd_{index}_cell_{parameter}_previous"] = (
                            p.data["General"]["Cell"][offset]
                        )
            self.gpx_saver()
        if best is None:
            logger.warning("%s", best_message)
        return best_message

    @refinement_transaction
    def refine_cell_parameters(
        self,
        phase_ind="all",
        set_to_false_after_refinement=True,
        plot=False,
        report=False,
        min_gof_improvement=0.1,
        scale_trials=None,
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

                report_str = (
                    "\n%s-phase\n\n%s (%s)\n\n     refined (old) \
                    \n a=%.5f (%.5f) \n b=%.5f (%.5f) \n c=%.5f (%.5f) \
                    \n \\alpha=%.2f (%.2f) \n \\beta=%.2f (%.2f) \n \\gamma=%.2f (%.2f) "
                    % (
                        site_label,
                        site_SpGrp.replace(" ", ""),
                        site_SGSys,
                        site_a,
                        site_a_previous,
                        site_b,
                        site_b_previous,
                        site_c,
                        site_c_previous,
                        site_alpha,
                        site_alpha_previous,
                        site_beta,
                        site_beta_previous,
                        site_gamma,
                        site_gamma_previous,
                    )
                )

                logger.info("%s", report_str)

        if set_to_false_after_refinement:
            self.gpx.set_refinement({"set": {"Cell": False}}, phase=phase_ind)
        self.gpx_saver()

    def set_cell_parameters_refinement(
        self, set_refine_to=True, phase_ind="all", save_gpx=True
    ):
        """Set cell-refinement flags for a phase index or all phases; do not run a fit."""

        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement({"set": {"Cell": set_refine_to}})
        else:
            self.gpx.set_refinement({"set": {"Cell": set_refine_to}}, phase=phase_ind)

        if save_gpx:
            self.gpx_saver()

    def clear_cell_parameters_refinement(self, save_gpx=True):
        """Disable cell refinement for all phases and optionally save."""

        self.gpx.set_refinement({"set": {"Cell": False}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_strain_broadening(
        self,
        phase_ind="all",
        type="isotropic",
        set_to_false_after_refinement=True,
        plot=False,
        report=False,
        min_gof_improvement=1.0,
    ):
        """Refine strain broadening, requiring the specified percent GoF decrease."""

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self.gpx.set_refinement(
            {"set": {"Mustrain": {"refine": True, "type": type}}}, phase=phase_ind
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
        self, set_refine_to=True, phase_ind="all", type="isotropic", save_gpx=True
    ):
        """Select a microstrain model and flag for one or all phases; do not run a fit."""
        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement(
                {"set": {"Mustrain": {"refine": set_refine_to, "type": type}}}
            )
        else:
            self.gpx.set_refinement(
                {"set": {"Mustrain": {"refine": set_refine_to, "type": type}}},
                phase=phase_ind,
            )

        if save_gpx:
            self.gpx_saver()

    def clear_strain_broadening_refinement(self, save_gpx=True):
        """Disable microstrain refinement for all phases and optionally save."""

        self.gpx.set_refinement({"set": {"Mustrain": {"refine": False}}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_size_broadening(
        self,
        phase_ind="all",
        type="isotropic",
        set_to_false_after_refinement=True,
        plot=False,
        report=False,
        min_gof_improvement=1.0,
    ):
        """Refine size broadening, requiring the specified percent GoF decrease."""

        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self.gpx.set_refinement(
            {"set": {"Size": {"refine": True, "type": type}}}, phase=phase_ind
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
        self, set_refine_to=True, phase_ind="all", type="isotropic", save_gpx=True
    ):
        """Select a crystallite-size model and flag for one or all phases; do not run a fit."""
        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement(
                {"set": {"Size": {"refine": set_refine_to, "type": type}}}
            )
        else:
            self.gpx.set_refinement(
                {"set": {"Size": {"refine": set_refine_to, "type": type}}},
                phase=phase_ind,
            )

        if save_gpx:
            self.gpx_saver()

    def clear_size_broadening_refinement(self, save_gpx=True):
        """Disable crystallite-size refinement for all phases and optionally save."""

        self.gpx.set_refinement({"set": {"Size": {"refine": False}}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_phase_fractions(
        self, set_to_false_after_refinement=True, plot=False,
        min_gof_improvement=0.1,
    ):
        """Refine phase fractions, requiring the specified percent GoF decrease."""

        if self._phase_mode() != "Rietveld":
            raise ValueError("Quantitative phase fractions require LeBail=False for all phases")
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

    def set_phase_fractions_refinement(self, set_refine_to=True, save_gpx=True):
        """Toggle phase scales against the histogram scale; use Rietveld mode for fractions."""

        if set_refine_to:
            self.gpx[self._histogram_name]["Sample Parameters"]["Scale"][1] = False
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"][self._histogram_name]["Scale"][1] = True
        else:
            self.gpx[self._histogram_name]["Sample Parameters"]["Scale"][1] = True
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"][self._histogram_name]["Scale"][1] = False
        if save_gpx:
            self.gpx_saver()

    def clear_phase_fractions_refinement(self, save_gpx=True):
        """Disable phase scales and enable the histogram scale; optionally save."""
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
        phase_ind="all",
        harmonics_order=4,
        set_to_false_after_refinement=True,
        plot=False,
        min_gof_improvement=1.0,
    ):
        """Refine preferred orientation with a minimum percent GoF decrease."""
        gpx_before_refinement = self._prepare_refinement(min_gof_improvement)
        self._set_po(phase_ind, harmonics_order, True)

        ref_str = self.refine(update_ds=True, update_ds_phases=False, update_phases=False,
                              min_gof_improvement=min_gof_improvement,
                              log_label="refine_preferred_orientation")
        if not self.last_refinement_accepted:
            return self._restore_rejected_refinement(gpx_before_refinement, ref_str, plot=plot,
                                                     plot_hint="refine_preferred_orientation")
        title_str = self._report_refinement("Preferred orientation refined. " + ref_str, ref_str)

        if set_to_false_after_refinement:
            # self.gpx['Phases'][st]['Histograms']['PWDR data.xy']['Pref.Ori.'][2] = False
            for e, st in enumerate(self.phases):
                if (phase_ind == "all") or (phase_ind == None):
                    self.gpx["Phases"][st]["Histograms"][self._histogram_name]["Pref.Ori."][
                        2
                    ] = False
                else:
                    if e == phase_ind:
                        self.gpx["Phases"][st]["Histograms"][self._histogram_name][
                            "Pref.Ori."
                        ][2] = False

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
        self, set_refine_to=True, phase_ind=0, harmonics_order_to=4, save_gpx=True
    ):

        """Configure complete spherical harmonics of an even order and set their refinement flag."""
        self._set_po(phase_ind, harmonics_order_to, set_refine_to)

        if save_gpx:
            self.gpx_saver()

    def clear_preferred_orientation_refinement(self, save_gpx=True):

        """Disable preferred-orientation refinement for every phase and optionally save."""
        for st in self.phases:
            self.gpx["Phases"][st]["Histograms"][self._histogram_name]["Pref.Ori."][2] = False

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    @refinement_transaction
    def refine_site_property(
        self,
        phase_ind=0,
        site_ind=0,
        refinement_flags="",
        set_to_false_after_refinement=True,
        plot=False,
        min_gof_improvement=0.1,
    ):
        """Refine a site, accepting only a sufficient percentage decrease in GoF.

        Rejected trials preserve the project, site flags, structures, dataset,
        and previous-result history. The default requires at least a 0.1% decrease
        in GoF. Set min_gof_improvement=None to disable
        the improvement check, or 0 to accept any strictly positive decrease.
        """

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
        self, phase_ind=0, site_ind=0, refinement_flags="", save_gpx=True
    ):
        """Set native GSAS-II atom refinement flags for one site; do not run a fit."""

        site_label = self.gpx.phases()[phase_ind]["Atoms"][site_ind][0]
        self.gpx.phases()[phase_ind].atom(
            site_label
        ).refinement_flags = refinement_flags
        if save_gpx:
            self.gpx_saver()

    def clear_site_property_refinement(self, phase_ind=0, site_ind=0, save_gpx=True):
        """Clear all refinement flags for one atom and optionally save the project."""

        site_label = self.gpx.phases()[phase_ind]["Atoms"][site_ind][0]
        self.gpx.phases()[phase_ind].atom(site_label).refinement_flags = ""

        if save_gpx:
            self.gpx_saver()

    @refinement_transaction
    def fine_tune_gpx(self, min_gof_improvement=1.0):
        """Open the installed GSAS-II GUI, then synchronize the accepted result."""
        self._validate_gof_improvement(min_gof_improvement)
        filename = f"{self.gsasii_run_directory}/gsas.gpx"
        self.gpx_saver()
        subprocess.check_call([sys.executable, "-m", "GSASII", filename])
        return self._replace_project(filename, min_gof_improvement, "fine_tune_gpx")

    @refinement_transaction
    def replace_gpx_with(self, newgpx_to_replace, min_gof_improvement=1.0):
        """Replace a compatible single-histogram project and synchronize results."""
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
            observed = histogram.getdata("x")
            if observed.shape != expected.shape or not np.allclose(observed, expected, rtol=1e-7, atol=1e-7):
                raise ValueError("Replacement histogram must match the loaded profile coordinates")
            multiplier = self.ds.i1d.attrs.get("normalization_multiplier", 1.)
            expected_y = self.ds.i1d.values.copy()
            if "i1d_baseline" in self.ds:
                expected_y -= self.ds.i1d_baseline.values
            expected_y *= multiplier
            if "i1d_baseline" in self.ds and "normalized_to" in self.ds.i1d.attrs:
                expected_y += self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"]
            if not np.allclose(histogram.getdata("yobs"), expected_y):
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
    def plot_refinement_history(
        self,
        figsize=(8, 6),
        dpi=128,
        min_step_detail=4,
        export_fig_as=None,
        show=False,
        return_fig=False,
    ):
        """
        Plot comprehensive refinement convergence history showing accepted values and rollbacks.
        """
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
