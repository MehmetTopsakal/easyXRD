import math
import os
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import xarray as xr
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.signal import find_peaks

from . import DEFAULT, easyxrd_defaults, resolve_default
from .plotters import _calc_reflection_pattern
from .results import clear_fit_results
from .utils import load_gsas_module


def _resolve_mp_api_key(mp_rester_api_key=None):
    """Resolve and validate the Materials Project API key from argument, env, or scratch."""
    key = (
        mp_rester_api_key
        or os.environ.get("MP_API_KEY")
        or easyxrd_defaults.get("mp_api_key")
    )
    if not key or str(key).lower() in ("invalid", "none", "not found"):
        key_file = Path(easyxrd_defaults["easyxrd_scratch_path"]) / "mp_api_key.dat"
        if key_file.is_file() and mp_rester_api_key is None:
            key = key_file.read_text().strip()
    if not key or str(key).lower() in ("invalid", "none", "not found"):
        raise ValueError("A valid Materials Project API key is required")
    return key


class PhasesMixin:
    def load_phases(
        self,
        from_phases_dict=None,
        from_gpx=None,
        from_nc=None,
        from_ds=None,
        mp_rester_api_key=None,
        plot=DEFAULT,
    ):
        """Load one phase source atomically, clearing superseded phase metadata."""
        plot = resolve_default(plot, "load_phases_plot")

        if (
            sum(x is not None for x in (from_phases_dict, from_gpx, from_nc, from_ds))
            != 1
        ):
            raise ValueError("Provide exactly one phase source")
        if not hasattr(self, "ds"):
            raise ValueError("Load diffraction data before phases")
        entries = []
        if from_phases_dict is not None:
            for item in from_phases_dict:
                label = item["label"]
                if not isinstance(label, str) or not label or Path(label).name != label:
                    raise ValueError(
                        "Phase labels must be nonempty filenames without directory components"
                    )
                mp_id = item.get("mp_id", "none")
                if mp_id.lower() == "none":
                    structure = Structure.from_file(item["cif"])
                else:
                    key = _resolve_mp_api_key(mp_rester_api_key)
                    from mp_api.client import MPRester

                    with MPRester(key) as client:
                        structure = client.get_structure_by_material_id(
                            mp_id, final=True, conventional_unit_cell=True
                        )
                lengths = [
                    length * item.get("scale", 1) * item.get("scale_" + axis, 1)
                    for axis, length in zip("abc", structure.lattice.abc)
                ]
                if any(not math.isfinite(v) or v <= 0 for v in lengths):
                    raise ValueError(
                        "Lattice scale factors must be finite and positive"
                    )
                structure.lattice = Lattice.from_parameters(
                    *lengths, *structure.lattice.angles
                )
                entries.append(
                    (label, structure, str(CifWriter(structure, symprec=0.01)))
                )
        elif from_gpx is not None:
            g2 = load_gsas_module("GSASIIscriptable")
            project = g2.G2Project(gpxfile=str(from_gpx))
            with tempfile.TemporaryDirectory(prefix="easyxrd-phases-") as directory:
                for index, phase in enumerate(project.phases()):
                    path = Path(directory) / f"{index}.cif"
                    phase.export_CIF(outputname=str(path))
                    cif = path.read_text()
                    entries.append(
                        (phase.name, Structure.from_str(cif, fmt="cif"), cif)
                    )
        else:
            source = xr.load_dataset(from_nc) if from_nc is not None else from_ds
            for index in range(source.attrs.get("num_phases", 0)):
                cif = source.attrs[f"PhaseInd_{index}_cif"]
                entries.append(
                    (
                        source.attrs[f"PhaseInd_{index}_label"],
                        Structure.from_str(cif, fmt="cif"),
                        cif,
                    )
                )
        if len({label for label, _, _ in entries}) != len(entries):
            raise ValueError("Phase labels must be unique")
        ds = self.ds.copy(deep=False)
        ds.attrs = {k: v for k, v in ds.attrs.items() if not k.startswith("PhaseInd_")}
        ds.attrs["num_phases"] = len(entries)
        for index, (label, _, cif) in enumerate(entries):
            ds.attrs[f"PhaseInd_{index}_label"] = label
            ds.attrs[f"PhaseInd_{index}_cif"] = cif
        for key in ("i1d_refined", "i1d_gsas_background"):
            if key in ds:
                del ds[key]
        self.ds = clear_fit_results(ds)
        self.phases = {label: structure for label, structure, _ in entries}
        if hasattr(self, "gpx"):
            del self.gpx
        if plot:
            self.plot(plot_hint="load_phases")

    def screen_phases(
        self,
        chemsys=None,
        candidate_structures=None,
        e_above_hull_max=DEFAULT,
        max_candidates=DEFAULT,
        peak_prominence=DEFAULT,
        tolerance_q=DEFAULT,
        use_residual=DEFAULT,
        mp_rester_api_key=None,
    ):
        """Screen candidate crystal structures against observed or residual XRD peaks.

        Candidates can be queried from the Materials Project by chemical system
        (e.g. ``chemsys='Fe-Cr-O'`` or ``['Fe', 'Cr', 'O']``, which searches all
        elemental/binary/ternary subsystems up to ``e_above_hull_max``) or supplied
        directly via ``candidate_structures`` (a dict mapping labels to
        ``pymatgen.core.Structure`` objects or CIF paths, or a sequence of
        structures/dicts).

        Each candidate's Bragg pattern is simulated at the dataset wavelength
        and scored against experimental (or unindexed residual) peaks within
        ``tolerance_q`` (in inverse Angstrom), returning a ranked list of matches
        with an estimated isotropic lattice scale factor ``suggested_scale``.
        """
        e_above_hull_max = resolve_default(
            e_above_hull_max, "screen_phases_e_above_hull_max"
        )
        max_candidates = resolve_default(max_candidates, "screen_phases_max_candidates")
        peak_prominence = resolve_default(
            peak_prominence, "screen_phases_peak_prominence"
        )
        tolerance_q = resolve_default(tolerance_q, "screen_phases_tolerance_q")
        use_residual = resolve_default(use_residual, "screen_phases_use_residual")

        if not hasattr(self, "ds") or "i1d" not in self.ds or not self.ds.i1d.size:
            raise ValueError("Load diffraction data before screening phases")
        if (chemsys is None) == (candidate_structures is None):
            raise ValueError("Provide exactly one of chemsys or candidate_structures")
        if not np.isfinite(tolerance_q) or tolerance_q <= 0:
            raise ValueError("tolerance_q must be finite and positive")
        if not np.isfinite(peak_prominence) or peak_prominence <= 0:
            raise ValueError("peak_prominence must be finite and positive")
        if not isinstance(max_candidates, (int, np.integer)) or max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        if e_above_hull_max is not None and (
            not np.isfinite(e_above_hull_max) or e_above_hull_max < 0
        ):
            raise ValueError("e_above_hull_max must be nonnegative or None")

        wavelength = float(self.ds.i1d.attrs["wavelength_in_angst"])
        q_grid = np.asarray(self.ds.i1d.radial.values, dtype=float)
        q_min, q_max = float(q_grid[0]), float(q_grid[-1])

        if use_residual and "i1d_refined" in self.ds:
            signal = np.asarray(
                self.ds.i1d.values - self.ds.i1d_refined.values, dtype=float
            )
        elif "i1d_baseline" in self.ds:
            signal = np.asarray(
                self.ds.i1d.values - self.ds.i1d_baseline.values, dtype=float
            )
        else:
            raw = np.asarray(self.ds.i1d.values, dtype=float)
            signal = raw - float(np.nanmin(raw))
        signal = np.nan_to_num(np.maximum(0.0, signal), nan=0.0)
        peak_max = float(np.max(signal)) if signal.size else 0.0
        if peak_max <= 0:
            self.phase_candidates = []
            return []
        norm_signal = 100.0 * signal / peak_max
        peak_indices, _ = find_peaks(norm_signal, prominence=float(peak_prominence))
        if peak_indices.size == 0:
            peak_indices = np.array([int(np.argmax(norm_signal))], dtype=int)
        exp_q = q_grid[peak_indices]
        exp_i = norm_signal[peak_indices]
        total_exp_i = float(np.sum(exp_i))

        candidates_raw = []
        if candidate_structures is not None:
            if isinstance(candidate_structures, dict):
                items = [
                    {"label": str(k), "structure": v, "mp_id": "none"}
                    for k, v in candidate_structures.items()
                ]
            else:
                items = list(candidate_structures)
            for idx, entry in enumerate(items):
                if isinstance(entry, Structure):
                    st = entry
                    label = f"{st.composition.reduced_formula}_{idx}"
                    mp_id = "none"
                elif isinstance(entry, (str, Path)):
                    st = Structure.from_file(str(entry))
                    label = Path(entry).stem
                    mp_id = "none"
                elif isinstance(entry, dict):
                    raw_st = (
                        entry.get("structure")
                        if "structure" in entry
                        else entry.get("cif")
                    )
                    if isinstance(raw_st, Structure):
                        st = raw_st
                    elif raw_st is not None:
                        st = Structure.from_file(str(raw_st))
                    else:
                        raise ValueError(
                            "Candidate dict must contain 'structure' or 'cif'"
                        )
                    mp_id = str(entry.get("mp_id", "none"))
                    label = str(
                        entry.get("label", f"{st.composition.reduced_formula}_{idx}")
                    )
                else:
                    raise ValueError(
                        f"Unsupported candidate entry type: {type(entry).__name__}"
                    )
                candidates_raw.append((label, mp_id, st))
        else:
            if isinstance(chemsys, str):
                elements = sorted(
                    {
                        el.strip()
                        for el in chemsys.replace(",", "-").split("-")
                        if el.strip()
                    }
                )
            else:
                elements = sorted(
                    {str(el).strip() for el in chemsys if str(el).strip()}
                )
            if not elements:
                raise ValueError("chemsys must specify at least one chemical element")
            subsystems = [
                "-".join(combo)
                for r in range(1, len(elements) + 1)
                for combo in combinations(elements, r)
            ]
            key = _resolve_mp_api_key(mp_rester_api_key)
            from mp_api.client import MPRester

            with MPRester(key) as client:
                search_kwargs = {"chemsys": subsystems}
                if e_above_hull_max is not None:
                    search_kwargs["energy_above_hull"] = (0.0, float(e_above_hull_max))
                docs = client.materials.summary.search(
                    **search_kwargs,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "energy_above_hull",
                        "structure",
                        "symmetry",
                    ],
                )
            for doc in docs:
                e_hull = getattr(doc, "energy_above_hull", None)
                if (
                    e_above_hull_max is not None
                    and e_hull is not None
                    and float(e_hull) > float(e_above_hull_max)
                ):
                    continue
                mp_id = str(getattr(doc, "material_id", "none"))
                formula = str(getattr(doc, "formula_pretty", "Phase"))
                st = getattr(doc, "structure", None)
                if st is None:
                    continue
                try:
                    st = SpacegroupAnalyzer(
                        st, symprec=0.01
                    ).get_conventional_standard_structure()
                except Exception:
                    pass
                label = f"{formula}_{mp_id.replace('-', '_')}"
                candidates_raw.append((label, mp_id, st))

        xrdc = XRDCalculator(wavelength=wavelength)
        scored = []
        for label, mp_id, st in candidates_raw:
            refl_q, refl_i, _ = _calc_reflection_pattern(
                xrdc, st, wavelength, q_min, q_max
            )
            refl_q = np.asarray(refl_q, dtype=float)
            refl_i = np.asarray(refl_i, dtype=float)
            significant = refl_i >= 2.0
            refl_q = refl_q[significant]
            refl_i = refl_i[significant]
            if refl_q.size == 0:
                continue
            total_cand_i = float(np.sum(refl_i))
            matched_cand_weight = 0.0
            matched_exp_indices = set()
            scale_ratios = []
            scale_weights = []
            matched_count = 0
            for cq, ci in zip(refl_q, refl_i):
                diffs = np.abs(exp_q - cq)
                best_idx = int(np.argmin(diffs))
                dist = float(diffs[best_idx])
                if dist <= tolerance_q:
                    matched_count += 1
                    matched_exp_indices.add(best_idx)
                    proximity = 1.0 - 0.5 * ((dist / tolerance_q) ** 2)
                    matched_cand_weight += ci * proximity
                    if exp_q[best_idx] > 0:
                        scale_ratios.append(cq / float(exp_q[best_idx]))
                        scale_weights.append(ci * proximity)

            if matched_count == 0:
                continue
            recall = matched_cand_weight / total_cand_i if total_cand_i > 0 else 0.0
            explained_exp_i = (
                float(np.sum(exp_i[list(matched_exp_indices)]))
                if matched_exp_indices
                else 0.0
            )
            precision = explained_exp_i / total_exp_i if total_exp_i > 0 else 0.0
            score = 100.0 * (0.7 * recall + 0.3 * precision)
            suggested_scale = (
                float(np.average(scale_ratios, weights=scale_weights))
                if scale_ratios
                else 1.0
            )
            try:
                spg_symbol = st.get_space_group_info()[0]
            except Exception:
                spg_symbol = "unknown"
            scored.append(
                {
                    "label": label,
                    "name": label,
                    "mp_id": mp_id,
                    "material_id": mp_id,
                    "formula": st.composition.reduced_formula,
                    "space_group": spg_symbol,
                    "match_score": float(score),
                    "fom": float(score) / 100.0,
                    "matched_peaks": int(matched_count),
                    "total_candidate_peaks": int(refl_q.size),
                    "suggested_scale": suggested_scale,
                    "structure": st,
                }
            )

        scored.sort(
            key=lambda item: (item["match_score"], item["matched_peaks"]), reverse=True
        )
        self.phase_candidates = scored[: int(max_candidates)]
        self.screened_phases = self.phase_candidates
        return self.phase_candidates

    def export_phases(
        self,
        phase_ind=None,  # should start from 0. -1 is not allowed
        export_to=".",
        export_extension="_exported.cif",
    ):
        """Export loaded phase CIFs to files."""
        Path(export_to).mkdir(parents=True, exist_ok=True)
        num_phases = self.ds.attrs.get("num_phases", 0)
        if (phase_ind is None) or (phase_ind == "all"):
            indices = list(range(num_phases))
        elif isinstance(phase_ind, (int, np.integer)):
            if phase_ind < 0 or phase_ind >= num_phases:
                raise IndexError(
                    f"Phase index {phase_ind} out of range (0 to {num_phases - 1})"
                )
            indices = [phase_ind]
        elif isinstance(phase_ind, str):
            indices = [
                p
                for p in range(num_phases)
                if self.ds.attrs.get(f"PhaseInd_{p}_label") == phase_ind
            ]
            if not indices:
                raise ValueError(f"Phase '{phase_ind}' not found in loaded phases")
        else:
            indices = [p for p in range(num_phases) if p == phase_ind]

        for p in indices:
            label = self.ds.attrs[f"PhaseInd_{p}_label"]
            out_path = Path(export_to) / f"{label}{export_extension}"
            out_path.write_text(str(self.ds.attrs[f"PhaseInd_{p}_cif"]))
