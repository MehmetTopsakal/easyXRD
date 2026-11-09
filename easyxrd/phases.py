from .results import clear_fit_results
import os
import math
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from . import easyxrd_defaults
from .utils import load_gsas_module


class PhasesMixin:
    def load_phases(self, from_phases_dict=None, from_gpx=None, from_nc=None,
                    from_ds=None, mp_rester_api_key=None, plot=True):
        """Load one phase source atomically, clearing superseded phase metadata."""
        if sum(x is not None for x in (from_phases_dict, from_gpx, from_nc, from_ds)) != 1:
            raise ValueError("Provide exactly one phase source")
        if not hasattr(self, "ds"):
            raise ValueError("Load diffraction data before phases")
        entries = []
        if from_phases_dict is not None:
            for item in from_phases_dict:
                label = item["label"]
                if not isinstance(label, str) or not label or Path(label).name != label:
                    raise ValueError("Phase labels must be nonempty filenames without directory components")
                mp_id = item.get("mp_id", "none")
                if mp_id.lower() == "none":
                    structure = Structure.from_file(item["cif"])
                else:
                    key = mp_rester_api_key or os.environ.get("MP_API_KEY") or easyxrd_defaults.get("mp_api_key")
                    if not key or key.lower() in ("invalid", "none", "not found"):
                        key_file = Path(easyxrd_defaults["easyxrd_scratch_path"]) / "mp_api_key.dat"
                        if key_file.is_file() and mp_rester_api_key is None:
                            key = key_file.read_text().strip()
                    if not key or key.lower() in ("invalid", "none", "not found"):
                        raise ValueError("A valid Materials Project API key is required")
                    from mp_api.client import MPRester
                    with MPRester(key) as client:
                        structure = client.get_structure_by_material_id(mp_id, final=True, conventional_unit_cell=True)
                lengths = [length * item.get("scale", 1) * item.get("scale_" + axis, 1)
                           for axis, length in zip("abc", structure.lattice.abc)]
                if any(not math.isfinite(v) or v <= 0 for v in lengths):
                    raise ValueError("Lattice scale factors must be finite and positive")
                structure.lattice = Lattice.from_parameters(*lengths, *structure.lattice.angles)
                entries.append((label, structure, str(CifWriter(structure, symprec=0.01))))
        elif from_gpx is not None:
            g2 = load_gsas_module("GSASIIscriptable")
            project = g2.G2Project(gpxfile=str(from_gpx))
            with tempfile.TemporaryDirectory(prefix="easyxrd-phases-") as directory:
                for index, phase in enumerate(project.phases()):
                    path = Path(directory) / f"{index}.cif"
                    phase.export_CIF(outputname=str(path))
                    cif = path.read_text()
                    entries.append((phase.name, Structure.from_str(cif, fmt="cif"), cif))
        else:
            source = xr.load_dataset(from_nc) if from_nc is not None else from_ds
            for index in range(source.attrs.get("num_phases", 0)):
                cif = source.attrs[f"PhaseInd_{index}_cif"]
                entries.append((source.attrs[f"PhaseInd_{index}_label"], Structure.from_str(cif, fmt="cif"), cif))
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
                raise IndexError(f"Phase index {phase_ind} out of range (0 to {num_phases - 1})")
            indices = [phase_ind]
        elif isinstance(phase_ind, str):
            indices = [
                p for p in range(num_phases)
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

