import os
import random
import string
import copy
import numpy as np
import xarray as xr
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter

from . import easyxrd_defaults
from .plotters import exrd_plotter


class PhasesMixin:
    """
    Mixin class providing crystal structure and phase loading/exporting capabilities.
    """
    def load_phases(
        self,
        from_phases_dict=None,
        from_gpx=None,
        from_nc=None,
        from_ds=None,
        mp_rester_api_key=None,
        plot=True,
    ):

        for k in ["i1d_refined", "i1d_gsas_background"]:
            if k in self.ds.keys():
                del self.ds[k]

        if mp_rester_api_key is None:
            mp_rester_api_key = os.environ.get("MP_API_KEY") or easyxrd_defaults.get(
                "mp_api_key", "none"
            )

        self.easyxrd_scratch_directory = easyxrd_defaults["easyxrd_scratch_path"]

        if from_phases_dict is not None:

            self.phases = {}
            for e, p in enumerate(from_phases_dict):

                mp_id = p.get("mp_id", "none")

                if mp_id.lower() == "none":
                    st = Structure.from_file(p["cif"])

                    scale = p.get("scale", 1)
                    scale_a = p.get("scale_a", 1)
                    scale_b = p.get("scale_b", 1)
                    scale_c = p.get("scale_c", 1)

                    st.lattice = Lattice.from_parameters(
                        a=st.lattice.abc[0] * scale * scale_a,
                        b=st.lattice.abc[1] * scale * scale_b,
                        c=st.lattice.abc[2] * scale * scale_c,
                        alpha=st.lattice.angles[0],
                        beta=st.lattice.angles[1],
                        gamma=st.lattice.angles[2],
                    )
                    self.phases[p["label"]] = st

                else:

                    if (not mp_rester_api_key) or mp_rester_api_key.lower() in (
                        "not found",
                        "invalid",
                        "none",
                    ):
                        raise ValueError(
                            "A valid Materials Project API key is needed to retrieve crystal structures. "
                            "Please provide `mp_rester_api_key`, set the MP_API_KEY environment variable, "
                            "or save your key to ~/.easyxrd_scratch/mp_api_key.dat. "
                            "API keys can be obtained from: https://profile.materialsproject.org/"
                        )

                    from mp_api.client import MPRester

                    try:
                        mpr = MPRester(mp_rester_api_key)
                    except Exception as exc:
                        raise ValueError(
                            f"The Materials Project API key is not valid or connection failed: {exc}. "
                            "Please check your API key from https://profile.materialsproject.org/"
                        )

                    st = mpr.get_structure_by_material_id(mp_id, final=False)[0]

                    scale = p.get("scale", 1)
                    scale_a = p.get("scale_a", 1)
                    scale_b = p.get("scale_b", 1)
                    scale_c = p.get("scale_c", 1)

                    st.lattice = Lattice.from_parameters(
                        a=st.lattice.abc[0] * scale * scale_a,
                        b=st.lattice.abc[1] * scale * scale_b,
                        c=st.lattice.abc[2] * scale * scale_c,
                        alpha=st.lattice.angles[0],
                        beta=st.lattice.angles[1],
                        gamma=st.lattice.angles[2],
                    )
                    self.phases[p["label"]] = st

                randstr = "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=7)
                )
                CifWriter(st, symprec=0.01).write_file("%s.cif" % randstr)
                # read cif
                with open("%s.cif" % randstr, "r") as ciffile:
                    ciffile_content = ciffile.read()
                    self.ds.attrs["PhaseInd_%d_cif" % (e)] = ciffile_content
                self.ds.attrs["PhaseInd_%d_label" % (e)] = p["label"]
                os.remove("%s.cif" % randstr)

            self.ds.attrs["num_phases"] = len(from_phases_dict)

        elif from_gpx is not None:
            import GSASII.GSASIIscriptable as G2sc

            phases_gpx = G2sc.G2Project(gpxfile=from_gpx)
            self.phases = {}
            for e, p in enumerate(phases_gpx.phases()):
                tmp_cif = "%s/tmp_%d.cif" % (self.easyxrd_scratch_directory, e)
                p.export_CIF(outputname=tmp_cif)
                with open(tmp_cif, "r") as ciffile:
                    ciffile_content = ciffile.read()
                try:
                    os.remove(tmp_cif)
                except OSError:
                    pass
                st = Structure.from_str(ciffile_content, fmt="cif")
                self.phases[p.name] = st
                self.ds.attrs["PhaseInd_%d_cif" % (e)] = ciffile_content
                self.ds.attrs["PhaseInd_%d_label" % (e)] = p.name
            self.ds.attrs["num_phases"] = len(self.phases)

        elif from_nc is not None:

            with xr.open_dataset(from_nc) as ds_nc:

                self.phases = {}
                num_p = ds_nc.attrs.get("num_phases", 0)
                for p in range(num_p):
                    cif_str = ds_nc.attrs["PhaseInd_%d_cif" % p]
                    label = ds_nc.attrs["PhaseInd_%d_label" % p]
                    self.phases[label] = Structure.from_str(cif_str, fmt="cif")
                    self.ds.attrs["PhaseInd_%d_label" % p] = label
                    self.ds.attrs["PhaseInd_%d_cif" % p] = cif_str

                self.ds.attrs["num_phases"] = num_p

        elif from_ds is not None:

            self.phases = {}
            num_p = from_ds.attrs.get("num_phases", 0)
            for p in range(num_p):
                cif_str = from_ds.attrs["PhaseInd_%d_cif" % p]
                label = from_ds.attrs["PhaseInd_%d_label" % p]
                self.phases[label] = Structure.from_str(cif_str, fmt="cif")
                self.ds.attrs["PhaseInd_%d_label" % p] = label
                self.ds.attrs["PhaseInd_%d_cif" % p] = cif_str

            self.ds.attrs["num_phases"] = num_p

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=None,
                phases=self.phases,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                title_str=None,
                export_fig_as=None,
                plot_hint="load_phases",
            )

    def export_phases(
        self,
        phase_ind=None,  # should start from 0. -1 is not allowed
        export_to=".",
        export_extension="_exported.cif",
    ):
        if (phase_ind is None) or (phase_ind == "all"):
            for p in range(self.ds.attrs["num_phases"]):
                with open(
                    "%s/%s%s"
                    % (
                        export_to,
                        self.ds.attrs["PhaseInd_%d_label" % p],
                        export_extension,
                    ),
                    "w",
                ) as ciffile:
                    ciffile.write("%s" % self.ds.attrs["PhaseInd_%d_cif" % p])
        else:
            for p in range(self.ds.attrs["num_phases"]):
                if p == phase_ind:
                    with open(
                        "%s/%s%s"
                        % (
                            export_to,
                            self.ds.attrs["PhaseInd_%d_label" % p],
                            export_extension,
                        ),
                        "w",
                    ) as ciffile:
                        ciffile.write("%s" % self.ds.attrs["PhaseInd_%d_cif" % p])

