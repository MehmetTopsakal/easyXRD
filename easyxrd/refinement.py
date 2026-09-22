import os
import sys
import subprocess
import shutil
import random
import string
import fnmatch
import time
import copy
import numpy as np
import xarray as xr
from pymatgen.core.structure import Structure

from .utils import HiddenPrints
from . import easyxrd_defaults
from .plotters import exrd_plotter


class RefinementMixin:
    """
    Mixin class providing GSAS-II refinement setup, execution, and parameter controls.
    """
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
    ):

        gpx_previous = copy.deepcopy(self.gpx)
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

        if self.verbose or verbose:
            print("\n\n\n\n\n")
            self.gpx.refine()
        else:
            with HiddenPrints():
                self.gpx.refine()

        if update_ds:
            histogram = self.gpx.histograms()[0]

            if "i1d_baseline" in self.ds.keys():
                if "normalized_to" in self.ds.i1d.attrs:
                    Ycalc = (
                        histogram.getdata("ycalc").astype("float32")
                        - self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"]
                    )  # this includes gsas background
                    Ybkg = (
                        histogram.getdata("Background").astype("float32")
                        - self.yshift_multiplier * self.ds.i1d.attrs["normalized_to"]
                    )
                    self.ds["i1d_refined"] = xr.DataArray(
                        data=(
                            self.ds.i1d_baseline.values
                            + Ycalc / self.ds.i1d.attrs["normalization_multiplier"]
                        ),
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds["i1d_gsas_background"] = xr.DataArray(
                        data=Ybkg / self.ds.i1d.attrs["normalization_multiplier"],
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds.attrs = (
                        self.ds.attrs | self.gpx["Covariance"]["data"]["Rvals"]
                    )
                    try:
                        self.ds.attrs["converged"] = str(self.ds.attrs["converged"])
                        self.ds.attrs["Aborted"] = str(self.ds.attrs["Aborted"])
                    except:
                        pass
                else:
                    Ycalc = histogram.getdata("ycalc").astype("float32")
                    Ybkg = histogram.getdata("Background").astype("float32")
                    self.ds["i1d_refined"] = xr.DataArray(
                        data=(self.ds.i1d_baseline.values + Ycalc),
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds["i1d_gsas_background"] = xr.DataArray(
                        data=Ybkg,
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds.attrs = (
                        self.ds.attrs | self.gpx["Covariance"]["data"]["Rvals"]
                    )
                    try:
                        self.ds.attrs["converged"] = str(self.ds.attrs["converged"])
                        self.ds.attrs["Aborted"] = str(self.ds.attrs["Aborted"])
                    except:
                        pass
            else:
                if "normalized_to" in self.ds.i1d.attrs:
                    Ycalc = histogram.getdata("ycalc").astype("float32")
                    Ybkg = histogram.getdata("Background").astype("float32")
                    self.ds["i1d_refined"] = xr.DataArray(
                        data=(Ycalc / self.ds.i1d.attrs["normalization_multiplier"]),
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds["i1d_gsas_background"] = xr.DataArray(
                        data=Ybkg / self.ds.i1d.attrs["normalization_multiplier"],
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds.attrs = (
                        self.ds.attrs | self.gpx["Covariance"]["data"]["Rvals"]
                    )
                    try:
                        self.ds.attrs["converged"] = str(self.ds.attrs["converged"])
                        self.ds.attrs["Aborted"] = str(self.ds.attrs["Aborted"])
                    except:
                        pass
                else:
                    Ycalc = histogram.getdata("ycalc").astype("float32")
                    Ybkg = histogram.getdata("Background").astype("float32")
                    self.ds["i1d_refined"] = xr.DataArray(
                        data=(Ycalc),
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds["i1d_gsas_background"] = xr.DataArray(
                        data=Ybkg,
                        dims=["radial"],
                        coords={"radial": self.ds.i1d.radial},
                    )
                    self.ds.attrs = (
                        self.ds.attrs | self.gpx["Covariance"]["data"]["Rvals"]
                    )
                    try:
                        self.ds.attrs["converged"] = str(self.ds.attrs["converged"])
                        self.ds.attrs["Aborted"] = str(self.ds.attrs["Aborted"])
                    except:
                        pass

            for e, p in enumerate(self.gpx.phases()):
                self.ds.attrs["PhaseInd_%d_SGSys" % (e)] = self.gpx["Phases"][p.name][
                    "General"
                ]["SGData"]["SGSys"]
                self.ds.attrs["PhaseInd_%d_SpGrp" % (e)] = self.gpx["Phases"][p.name][
                    "General"
                ]["SGData"]["SpGrp"]

                self.ds.attrs["PhaseInd_%d_cell_a" % (e)] = self.gpx["Phases"][p.name][
                    "General"
                ]["Cell"][1]
                self.ds.attrs["PhaseInd_%d_cell_b" % (e)] = self.gpx["Phases"][p.name][
                    "General"
                ]["Cell"][2]
                self.ds.attrs["PhaseInd_%d_cell_c" % (e)] = self.gpx["Phases"][p.name][
                    "General"
                ]["Cell"][3]
                self.ds.attrs["PhaseInd_%d_cell_alpha" % (e)] = self.gpx["Phases"][
                    p.name
                ]["General"]["Cell"][4]
                self.ds.attrs["PhaseInd_%d_cell_beta" % (e)] = self.gpx["Phases"][
                    p.name
                ]["General"]["Cell"][5]
                self.ds.attrs["PhaseInd_%d_cell_gamma" % (e)] = self.gpx["Phases"][
                    p.name
                ]["General"]["Cell"][6]

                self.ds.attrs["PhaseInd_%d_size_broadening_type" % (e)] = self.gpx[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Size"][0]
                self.ds.attrs["PhaseInd_%d_size_0" % (e)] = self.gpx["Phases"][p.name][
                    "Histograms"
                ]["PWDR data.xy"]["Size"][1][0]
                self.ds.attrs["PhaseInd_%d_size_1" % (e)] = self.gpx["Phases"][p.name][
                    "Histograms"
                ]["PWDR data.xy"]["Size"][1][1]
                self.ds.attrs["PhaseInd_%d_size_2" % (e)] = self.gpx["Phases"][p.name][
                    "Histograms"
                ]["PWDR data.xy"]["Size"][1][2]
                self.ds.attrs["PhaseInd_%d_strain_broadening_type" % (e)] = self.gpx[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Mustrain"][0]
                self.ds.attrs["PhaseInd_%d_mustrain_0" % (e)] = self.gpx["Phases"][
                    p.name
                ]["Histograms"]["PWDR data.xy"]["Mustrain"][1][0]
                self.ds.attrs["PhaseInd_%d_mustrain_1" % (e)] = self.gpx["Phases"][
                    p.name
                ]["Histograms"]["PWDR data.xy"]["Mustrain"][1][1]
                self.ds.attrs["PhaseInd_%d_mustrain_2" % (e)] = self.gpx["Phases"][
                    p.name
                ]["Histograms"]["PWDR data.xy"]["Mustrain"][1][2]

            wtSum = 0.0
            for e, p in enumerate(self.phases):
                mass = self.gpx["Phases"][p]["General"]["Mass"]
                phFr = self.gpx["Phases"][p]["Histograms"]["PWDR data.xy"]["Scale"][0]
                wtSum += mass * phFr
            for e, p in enumerate(self.phases):
                weightFr = (
                    self.gpx["Phases"][p]["Histograms"]["PWDR data.xy"]["Scale"][0]
                    * self.gpx["Phases"][p]["General"]["Mass"]
                    / wtSum
                )
                self.ds.attrs["PhaseInd_%d_wt_fraction" % (e)] = 100 * weightFr

            for e, p in enumerate(gpx_previous.phases()):
                self.ds.attrs["PhaseInd_%d_cell_a_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["General"]["Cell"][1]
                self.ds.attrs["PhaseInd_%d_cell_b_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["General"]["Cell"][2]
                self.ds.attrs["PhaseInd_%d_cell_c_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["General"]["Cell"][3]
                self.ds.attrs["PhaseInd_%d_cell_alpha_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["General"]["Cell"][4]
                self.ds.attrs["PhaseInd_%d_cell_beta_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["General"]["Cell"][5]
                self.ds.attrs["PhaseInd_%d_cell_gamma_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["General"]["Cell"][6]

                self.ds.attrs["PhaseInd_%d_size_broadening_type_previous" % (e)] = (
                    gpx_previous["Phases"][p.name]["Histograms"]["PWDR data.xy"][
                        "Size"
                    ][0]
                )
                self.ds.attrs["PhaseInd_%d_size_0_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Size"][1][0]
                self.ds.attrs["PhaseInd_%d_size_1_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Size"][1][1]
                self.ds.attrs["PhaseInd_%d_size_2_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Size"][1][2]
                self.ds.attrs["PhaseInd_%d_strain_broadening_type_previous" % (e)] = (
                    gpx_previous["Phases"][p.name]["Histograms"]["PWDR data.xy"][
                        "Mustrain"
                    ][0]
                )
                self.ds.attrs["PhaseInd_%d_mustrain_0_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Mustrain"][1][0]
                self.ds.attrs["PhaseInd_%d_mustrain_1_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Mustrain"][1][1]
                self.ds.attrs["PhaseInd_%d_mustrain_2_previous" % (e)] = gpx_previous[
                    "Phases"
                ][p.name]["Histograms"]["PWDR data.xy"]["Mustrain"][1][2]

            inst_prm_dict = self.gpx["PWDR data.xy"]["Instrument Parameters"][0]
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
                gof_symbol = "✨"  # https://www.compart.com/en/unicode/category/So
            elif gof_change > -1:
                gof_symbol = "❗"
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
        except:
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

    def gpx_saver(self):
        if self.verbose:
            self.gpx.save()
        else:
            with HiddenPrints():
                self.gpx.save()

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
    ):

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
            os.path.join(os.path.expanduser("~"), "g2full/GSAS-II/GSASII")
        )

        G2sc = None
        for p in candidate_paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
            try:
                import GSASII.GSASIIscriptable as G2sc
                import GSASII.GSASIIlattice as G2lat

                self.gsasii_lib_path = p
                break
            except ImportError:
                try:
                    import GSASIIscriptable as G2sc
                    import GSASIIlattice as G2lat

                    self.gsasii_lib_path = p
                    break
                except ImportError:
                    continue

        if G2sc is None:
            try:
                import GSASII.GSASIIscriptable as G2sc
                import GSASII.GSASIIlattice as G2lat

                self.gsasii_lib_path = "system"
            except ImportError:
                raise ImportError(
                    "Unable to import GSASIIscriptable. Please ensure GSAS-II is installed and accessible. "
                    "You can specify the GSAS-II directory via `setup_gsas2_refiner(gsasii_lib_path=...)` "
                    "or by setting the GSASII_PATH environment variable. "
                    "Installation notes: https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html"
                )

        self.easyxrd_scratch_directory = easyxrd_defaults["easyxrd_scratch_path"]

        randstr = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))

        self.gsasii_run_directory = "%s/%d_%s.gsastmp" % (
            self.easyxrd_scratch_directory,
            int(time.time()),
            randstr,
        )

        os.makedirs(self.gsasii_run_directory, exist_ok=True)

        if normalize:
            # find normalization scale from i1d
            if "i1d_baseline" in self.ds.keys():
                da_baseline_sub = self.ds.i1d - self.ds.i1d_baseline
                normalization_multiplier = normalize_to * (
                    1 / max(da_baseline_sub.values)
                )
            else:
                da = self.ds.i1d
                normalization_multiplier = normalize_to * (1 / max(da.values))

            self.ds.i1d.attrs["normalization_multiplier"] = normalization_multiplier
            self.ds.i1d.attrs["normalized_to"] = normalize_to

        data_x = np.rad2deg(
            2
            * np.arcsin(
                self.ds.i1d.radial.values
                * ((self.ds.i1d.attrs["wavelength_in_angst"]) / (4 * np.pi))
            )
        )
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

        np.savetxt(
            "%s/data.xy" % self.gsasii_run_directory,
            fmt="%.7e",
            X=np.column_stack((data_x, data_y)),
        )

        if instprm_from_gpx is not None:
            if os.path.isfile(instprm_from_gpx):
                gpx_instprm = G2sc.G2Project(gpxfile=instprm_from_gpx)
                for n in gpx_instprm.names:
                    l = n
                    pattern = "PWDR *"
                    matching = fnmatch.filter(l, pattern)
                    if matching != []:
                        pwdr_name = matching[0]
                instprm_dict = gpx_instprm[pwdr_name]["Instrument Parameters"][0]

                with open("%s/gsas.instprm" % self.gsasii_run_directory, "w") as f:
                    f.write(
                        "#GSAS-II instrument parameter file; do not add/delete items!\n"
                    )
                    f.write("Type:PXC\n")
                    f.write("Bank:1.0\n")
                    f.write("Lam:%s\n" % (self.ds.i1d.attrs["wavelength_in_angst"]))
                    f.write("Polariz.:%s\n" % (instprm_dict["Polariz."][1]))
                    f.write("Azimuth:%s\n" % (instprm_dict["Azimuth"][1]))
                    f.write("Zero:%s\n" % (instprm_dict["Zero"][1]))
                    f.write("U:%s\n" % (instprm_dict["U"][1]))
                    f.write("V:%s\n" % (instprm_dict["V"][1]))
                    f.write("W:%s\n" % (instprm_dict["W"][1]))
                    f.write("X:%s\n" % (instprm_dict["X"][1]))
                    f.write("Y:%s\n" % (instprm_dict["Y"][1]))
                    f.write("Z:%s\n" % (instprm_dict["Z"][1]))
                    f.write("SH/L:%s\n" % (instprm_dict["SH/L"][1]))
            else:
                print(
                    "gpx file for reading instrument parameters do net exist. Please check the path"
                )
                # return

        elif instprm_from_nc is not None:
            with xr.open_dataset(instprm_from_nc) as ds_inst_prm:
                with open("%s/gsas.instprm" % self.gsasii_run_directory, "w") as f:
                    f.write(
                        "#GSAS-II instrument parameter file; do not add/delete items!\n"
                    )
                    f.write("Type:PXC\n")
                    f.write("Bank:1.0\n")
                    f.write("Lam:%s\n" % (self.ds.i1d.attrs["wavelength_in_angst"]))
                    f.write(
                        "Polariz.:%s\n"
                        % (ds_inst_prm.attrs["gsasii_inst_prm_Polariz."])
                    )
                    f.write(
                        "Azimuth:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_Azimuth"])
                    )
                    f.write("Zero:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_Zero"]))
                    f.write("U:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_U"]))
                    f.write("V:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_V"]))
                    f.write("W:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_W"]))
                    f.write("X:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_X"]))
                    f.write("Y:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_Y"]))
                    f.write("Z:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_Z"]))
                    f.write("SH/L:%s\n" % (ds_inst_prm.attrs["gsasii_inst_prm_SH/L"]))

        else:
            with open("%s/gsas.instprm" % self.gsasii_run_directory, "w") as f:
                f.write(
                    "#GSAS-II instrument parameter file; do not add/delete items!\n"
                )
                f.write("Type:PXC\n")
                f.write("Bank:1.0\n")
                f.write("Lam:%s\n" % (self.ds.i1d.attrs["wavelength_in_angst"]))
                f.write("Polariz.:%s\n" % (instprm_Polariz))
                f.write("Azimuth:%s\n" % (instprm_Azimuth))
                f.write("Zero:%s\n" % (instprm_Zero))
                f.write("U:%s\n" % (instprm_U))
                f.write("V:%s\n" % (instprm_V))
                f.write("W:%s\n" % (instprm_W))
                f.write("X:%s\n" % (instprm_X))
                f.write("Y:%s\n" % (instprm_Y))
                f.write("Z:%s\n" % (instprm_Z))
                f.write("SH/L:%s\n" % (instprm_SHL))

        if self.verbose:

            self.gpx = G2sc.G2Project(newgpx="%s/gsas.gpx" % self.gsasii_run_directory)
            self.gpx.data["Controls"]["data"]["max cyc"] = 100
            self.gpx.add_powder_histogram(
                "%s/data.xy" % self.gsasii_run_directory,
                "%s/gsas.instprm" % self.gsasii_run_directory,
            )
            self.export_phases(
                export_to=self.gsasii_run_directory, export_extension=".cif"
            )
            hist = self.gpx.histograms()[0]
            for p in self.phases:
                self.gpx.add_phase(
                    "%s/%s.cif" % (self.gsasii_run_directory, p),
                    phasename=p,
                    histograms=[hist],
                    fmthint="CIF",
                )

        else:
            with HiddenPrints():
                self.gpx = G2sc.G2Project(
                    newgpx="%s/gsas.gpx" % self.gsasii_run_directory
                )
                self.gpx.data["Controls"]["data"]["max cyc"] = 100
                self.gpx.add_powder_histogram(
                    "%s/data.xy" % self.gsasii_run_directory,
                    "%s/gsas.instprm" % self.gsasii_run_directory,
                )
                self.export_phases(
                    export_to=self.gsasii_run_directory, export_extension=".cif"
                )
                hist = self.gpx.histograms()[0]
                for p in self.phases:
                    self.gpx.add_phase(
                        "%s/%s.cif" % (self.gsasii_run_directory, p),
                        phasename=p,
                        histograms=[hist],
                        fmthint="CIF",
                    )

        for n in self.gpx.names:
            l = n
            pattern = "PWDR *"
            matching = fnmatch.filter(l, pattern)
            if matching != []:
                pwdr_name = matching[0]

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
                    * min(self.ds.i1d.values),
                ]
            else:
                self.gpx[pwdr_name]["Background"][0] = [
                    "chebyschev-1",
                    False,
                    1,
                    min(self.ds.i1d.values),
                ]

        if do_1st_refinement:

            _ = self.refine(
                update_ds=False,
                update_ds_phases=False,
                update_phases=False,
                update_previous_ds=True,
                update_previous_gpx=False,
                update_previous_phases=False,
                verbose=False,
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
            )

            print("\n ⏩--1st refinement with LeBail is completed. %s \n" % (ref_str))

            if plot:
                exrd_plotter(
                    ds=self.ds,
                    ds_previous=None,
                    phases=self.phases,
                    figsize=self.figsize,
                    i2d_robust=self.i2d_robust,
                    i2d_logscale=self.i2d_logscale,
                    i1d_ylogscale=self.i1d_ylogscale,
                    title_str="1st refinement with LeBail is completed. %s "
                    % (ref_str),
                    export_fig_as=None,
                    plot_hint="1st_refinement",
                )

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def refine_background(
        self,
        num_coeffs=10,
        background_type="chebyschev-1",
        set_to_false_after_refinement=True,
        plot=False,
    ):
        """ """

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
        )
        title_str = "Background with %d coeffs is refined. %s" % (num_coeffs, ref_str)
        print(" ✅--" + title_str)

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
        """ """

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
        """ """

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
    def refine_instrument_parameters(
        self,
        inst_pars_to_refine=["U", "V", "W"],
        set_to_false_after_refinement=True,
        plot=False,
    ):
        """
        inst_pars_to_refine=['U', 'V', 'W',   'X', 'Y', 'Z', 'Zero', 'SH/L']
        """

        self.gpx.set_refinement({"set": {"Instrument Parameters": inst_pars_to_refine}})

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=False,
        )
        title_str = "Instrument parameter %s is refined. %s" % (
            inst_pars_to_refine,
            ref_str,
        )
        print(" ✅--" + title_str)

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
        self, set_inst_pars_to_refine=["U", "V", "W"], set_refine_to=True, save_gpx=True
    ):
        """ """

        self.gpx.set_refinement(
            {"set": {"Instrument Parameters": set_inst_pars_to_refine}}
        )

        if save_gpx:
            self.gpx_saver()

    def clear_instrument_parameters_refinement(self, save_gpx=True):
        """ """

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
    def set_LeBail(self, to=True, phase_ind="all", refine=True, plot=False):
        """ """

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
            )
            if to:
                title_str = "After setting LeBail refinement to True, %s" % (ref_str)
                print("\n ✅--" + title_str)

            else:
                title_str = "After setting LeBail refinement to False, %s" % (ref_str)
                print("\n ✅--" + title_str)
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
    def refine_cell_parameters(
        self,
        phase_ind="all",
        set_to_false_after_refinement=True,
        plot=False,
        report=False,
    ):
        """ """

        self.gpx.set_refinement({"set": {"Cell": True}}, phase=phase_ind)

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=True,
            update_phases=True,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=True,
        )
        if (phase_ind == "all") or (phase_ind == None):
            if len(self.phases) > 1:
                title_str = "Cell parameters of all phases are refined. %s" % (ref_str)
            else:
                title_str = "Cell parameters are refined. %s" % (ref_str)
            print(" ✅--" + title_str)
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
            print(" ✅--" + title_str)
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

                print(report_str + "\n")

        if set_to_false_after_refinement:
            self.gpx.set_refinement({"set": {"Cell": False}}, phase=phase_ind)
        self.gpx_saver()

    def set_cell_parameters_refinement(
        self, set_refine_to=True, phase_ind="all", save_gpx=True
    ):
        """ """

        if (phase_ind == "all") or (phase_ind == None):
            self.gpx.set_refinement({"set": {"Cell": set_refine_to}})
        else:
            self.gpx.set_refinement({"set": {"Cell": set_refine_to}}, phase=phase_ind)

        if save_gpx:
            self.gpx_saver()

    def clear_cell_parameters_refinement(self, save_gpx=True):
        """ """

        self.gpx.set_refinement({"set": {"Cell": False}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def refine_strain_broadening(
        self,
        phase_ind="all",
        type="isotropic",
        set_to_false_after_refinement=True,
        plot=False,
        report=False,
    ):
        """ """

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
        )
        if (phase_ind == "all") or (phase_ind == None):
            title_str = "Strain broadening of all phases are refined. %s" % (ref_str)
            print(" ✅--" + title_str)
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
            print(" ✅--" + title_str)
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

                print(report_str + "\n")

        if set_to_false_after_refinement:
            self.gpx.set_refinement(
                {"set": {"Mustrain": {"refine": False}}}, phase=phase_ind
            )
        self.gpx_saver()

    def set_strain_broadening_refinement(
        self, set_refine_to=True, phase_ind="all", type="isotropic", save_gpx=True
    ):
        """ """
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
        """ """

        self.gpx.set_refinement({"set": {"Mustrain": {"refine": False}}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def refine_size_broadening(
        self,
        phase_ind="all",
        type="isotropic",
        set_to_false_after_refinement=True,
        plot=False,
        report=False,
    ):
        """ """

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
        )
        if (phase_ind == "all") or (phase_ind == None):
            title_str = "Size broadening of all phases are refined. %s" % (ref_str)
            print(" ✅--" + title_str)
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
            print(" ✅--" + title_str)
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

                print(report_str + "\n")

        if set_to_false_after_refinement:
            self.gpx.set_refinement(
                {"set": {"Size": {"refine": False}}}, phase=phase_ind
            )
        self.gpx_saver()

    def set_size_broadening_refinement(
        self, set_refine_to=True, phase_ind="all", type="isotropic", save_gpx=True
    ):
        """ """
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
        """ """

        self.gpx.set_refinement({"set": {"Size": {"refine": False}}})

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def refine_phase_fractions(self, set_to_false_after_refinement=True, plot=False):
        """ """

        self.gpx["PWDR data.xy"]["Sample Parameters"]["Scale"][1] = False
        for e, p in enumerate(self.phases):
            self.gpx["Phases"][p]["Histograms"]["PWDR data.xy"]["Scale"][1] = True

        ref_str = self.refine(
            update_ds=True,
            update_ds_phases=False,
            update_phases=False,
            update_previous_ds=True,
            update_previous_gpx=True,
            update_previous_phases=True,
        )
        title_str = "Phase fractions of all phases are refined. %s" % (ref_str)
        print(" ✅--" + title_str)

        if set_to_false_after_refinement:
            self.gpx["PWDR data.xy"]["Sample Parameters"]["Scale"][1] = True
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"]["PWDR data.xy"]["Scale"][1] = False

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
        """ """

        if set_refine_to:
            self.gpx["PWDR data.xy"]["Sample Parameters"]["Scale"][1] = False
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"]["PWDR data.xy"]["Scale"][1] = True
        else:
            self.gpx["PWDR data.xy"]["Sample Parameters"]["Scale"][1] = True
            for e, p in enumerate(self.phases):
                self.gpx["Phases"][p]["Histograms"]["PWDR data.xy"]["Scale"][1] = False
        if save_gpx:
            self.gpx_saver()

    def clear_phase_fractions_refinement(self, save_gpx=True):
        self.gpx["PWDR data.xy"]["Sample Parameters"]["Scale"][1] = True
        for e, p in enumerate(self.phases):
            self.gpx["Phases"][p]["Histograms"]["PWDR data.xy"]["Scale"][1] = False
        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def refine_preferred_orientation(
        self,
        phase_ind="all",
        harmonics_order=4,
        set_to_false_after_refinement=True,
        plot=False,
    ):
        """ """
        import GSASIIlattice as G2lat

        L = harmonics_order
        for e, st in enumerate(self.phases):
            if (phase_ind == "all") or (phase_ind == None):
                coef_dict = {}
                sytsym = self.gpx["Phases"][st]["General"]["SGData"]["SGLaue"]
                for l in range(2, L + 1):
                    coeffs = G2lat.GenShCoeff(sytsym=sytsym, L=l)
                    try:
                        cst = coeffs[0][0][:6]
                        coef_dict[cst] = 0.0
                    except:
                        pass
                self.gpx["Phases"][st]["Histograms"]["PWDR data.xy"]["Pref.Ori."] = [
                    "SH",
                    1.0,
                    True,
                    [0, 0, 1],
                    L,
                    coef_dict,
                    [""],
                    0.1,
                ]
            else:
                if e == phase_ind:
                    coef_dict = {}
                    sytsym = self.gpx["Phases"][st]["General"]["SGData"]["SGLaue"]
                    for l in range(2, L + 1):
                        coeffs = G2lat.GenShCoeff(sytsym=sytsym, L=l)
                        try:
                            cst = coeffs[0][0][:6]
                            coef_dict[cst] = 0.0
                        except:
                            pass
                    self.gpx["Phases"][st]["Histograms"]["PWDR data.xy"][
                        "Pref.Ori."
                    ] = ["SH", 1.0, True, [0, 0, 1], L, coef_dict, [""], 0.1]

        if (phase_ind == "all") or (phase_ind == None):
            ref_str = self.refine(
                update_ds=True,
                update_ds_phases=False,
                update_phases=False,
                update_previous_ds=True,
                update_previous_gpx=True,
                update_previous_phases=True,
            )
            title_str = "Preferred orientation for all phases are refined. %s" % (
                ref_str
            )
            print(" ✅--" + title_str)
        else:
            ref_str = self.refine(
                update_ds=True,
                update_ds_phases=False,
                update_phases=False,
                update_previous_ds=True,
                update_previous_gpx=True,
                update_previous_phases=True,
            )
            title_str = "Preferred orientation for %s phase is refined. %s" % (
                self.gpx.phases()[phase_ind].name,
                ref_str,
            )
            print(" ✅--" + title_str)

        if set_to_false_after_refinement:
            # self.gpx['Phases'][st]['Histograms']['PWDR data.xy']['Pref.Ori.'][2] = False
            for e, st in enumerate(self.phases):
                if (phase_ind == "all") or (phase_ind == None):
                    self.gpx["Phases"][st]["Histograms"]["PWDR data.xy"]["Pref.Ori."][
                        2
                    ] = False
                else:
                    if e == phase_ind:
                        self.gpx["Phases"][st]["Histograms"]["PWDR data.xy"][
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

        import GSASIIlattice as G2lat

        L = harmonics_order_to

        for e, st in enumerate(self.phases):
            if (phase_ind == "all") or (phase_ind == None):
                coef_dict = {}
                sytsym = self.gpx["Phases"][st]["General"]["SGData"]["SGLaue"]
                for l in range(2, L + 1):
                    coeffs = G2lat.GenShCoeff(sytsym=sytsym, L=l)
                    try:
                        cst = coeffs[0][0][:6]
                        coef_dict[cst] = 0.0
                    except:
                        pass
                self.gpx["Phases"][st]["Histograms"]["PWDR data.xy"]["Pref.Ori."] = [
                    "SH",
                    1.0,
                    set_refine_to,
                    [0, 0, 1],
                    L,
                    coef_dict,
                    [""],
                    0.1,
                ]
            else:
                if e == phase_ind:
                    coef_dict = {}
                    sytsym = self.gpx["Phases"][st]["General"]["SGData"]["SGLaue"]
                    for l in range(2, L + 1):
                        coeffs = G2lat.GenShCoeff(sytsym=sytsym, L=l)
                        try:
                            cst = coeffs[0][0][:6]
                            coef_dict[cst] = 0.0
                        except:
                            pass
                    self.gpx["Phases"][st]["Histograms"]["PWDR data.xy"][
                        "Pref.Ori."
                    ] = ["SH", 1.0, True, [0, 0, 1], L, coef_dict, [""], 0.1]

        if save_gpx:
            self.gpx_saver()

    def clear_preferred_orientation_refinement(self, save_gpx=True):

        for st in self.phases:
            self.gpx["Phases"][st]["Histograms"]["PWDR data.xy"]["Pref.Ori."][2] = False

        if save_gpx:
            self.gpx_saver()

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def refine_site_property(
        self,
        phase_ind=0,
        site_ind=0,
        refinement_flags="",
        set_to_false_after_refinement=True,
        plot=False,
    ):
        """ """

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
        )
        title_str = "%s property of %s site of %s phase is refined. %s" % (
            refinement_flags,
            site_label,
            self.gpx.phases()[phase_ind].name,
            ref_str,
        )
        print(" ✅--" + title_str)

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
        """ """

        site_label = self.gpx.phases()[phase_ind]["Atoms"][site_ind][0]
        self.gpx.phases()[phase_ind].atom(
            site_label
        ).refinement_flags = refinement_flags
        if save_gpx:
            self.gpx_saver()

    def clear_site_property_refinement(self, phase_ind=0, site_ind=0, save_gpx=True):
        """ """

        site_label = self.gpx.phases()[phase_ind]["Atoms"][site_ind][0]
        self.gpx.phases()[phase_ind].atom(site_label).refinement_flags = ""

        if save_gpx:
            self.gpx_saver()

    def fine_tune_gpx(self):
        """ """
        subprocess.check_call(
            [
                "%s/../../RunGSASII.sh" % self.gsasii_lib_path,
                "%s/gsas.gpx" % self.gsasii_run_directory,
            ]
        )
        import GSASIIscriptable as G2sc

        self.gpx = G2sc.G2Project(gpxfile="%s/gsas.gpx" % self.gsasii_run_directory)
        self.gpx.refine()

    ###############################################################################################
    def replace_gpx_with(self, newgpx_to_replace):
        """ """
        shutil.copy(newgpx_to_replace, "%s/gsas.gpx" % self.gsasii_run_directory)
        import GSASIIscriptable as G2sc

        self.gpx = G2sc.G2Project(gpxfile="%s/gsas.gpx" % self.gsasii_run_directory)
        self.gpx.refine()

    ###############################################################################################
    def export_gpx_to(self, to="gsas.gpx"):
        """ """
        shutil.copy("%s/gsas.gpx" % self.gsasii_run_directory, to)

    ###############################################################################################
