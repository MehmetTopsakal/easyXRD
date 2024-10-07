from scipy.signal import savgol_filter
import pybaselines

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifWriter

from IPython.display import clear_output

import subprocess

import shutil

import random, string
import fnmatch

import time
import copy
from copy import deepcopy


import os, sys


import numpy as np
import xarray as xr

import fabio
import pyFAI


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt

plt.rcParams.update({"figure.max_open_warning": 0})


from scipy.ndimage import median_filter


from . import easyxrd_defaults
from .plotters import *


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class exrd:
    def __init__(
        self,
        verbose=False,
        figsize=(8, 6),
        i2d_robust=True,
        i1d_ylogscale=True,
        i2d_logscale=True,
    ):

        self.verbose = verbose
        self.figsize = figsize
        self.i2d_robust = i2d_robust
        self.i1d_ylogscale = i1d_ylogscale
        self.i2d_logscale = i2d_logscale

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
        ds_previous = copy.deepcopy(self.ds)
        phases_previous = copy.deepcopy(self.phases)

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
            inst_prm_dict_clean["gsasii_inst_prm_" + i] = inst_prm_dict[i][0]
        self.ds.attrs = self.ds.attrs | inst_prm_dict_clean

        if update_phases or update_ds_phases:
            for e, p in enumerate(self.gpx.phases()):
                p.export_CIF(
                    outputname="%s/%s_refined.cif" % (self.gsasii_run_directory, p.name)
                )
                if update_ds_phases:
                    with open(
                        "%s/%s_refined.cif" % (self.gsasii_run_directory, p.name), "r"
                    ) as ciffile:
                        ciffile_content = ciffile.read()
                        self.ds.attrs["PhaseInd_%d_cif" % (e)] = ciffile_content
                if update_phases:
                    st = Structure.from_file(
                        "%s/%s_refined.cif" % (self.gsasii_run_directory, p.name)
                    )
                    self.phases[p.name] = st

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
            refinement_str = (
                "Rwp/GoF is now %.3f/%.3f (was %.3f(%.2f%%)/%.3f(%.2f%%%s))"
                % (
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




    def load_xrd_data(
        self,
        from_img_array=None,
        from_tiff_file=None,
        ai=None,
        poni_file=None,
        mask=None,
        mask_file=None,
        median_filter_size=2,
        from_nc_file=None,
        from_txt_file=None,
        txt_file_wavelength_in_angstrom=0.1814,
        txt_file_comments="#",
        txt_file_skiprows=0,
        txt_file_usecols=(0, 1),
        txt_file_radial_unit="tth",
        radial_range=[0.1, 11.1],
        radial_npts=1000,
        plot=True,
    ):

        if (from_img_array is None) and (from_tiff_file is not None):
            img_array = median_filter(
                fabio.open(from_tiff_file).data, size=median_filter_size
            )
        elif (from_img_array is not None) and (from_tiff_file is None):
            img_array = from_img_array
        else:
            img_array = None

        if img_array is not None:

            self.ds = xr.Dataset()

            if radial_range is not None:
                delta_q = 0.0010
                npt = int(np.ceil((radial_range[1] - radial_range[0]) / delta_q))
                radial_range = [radial_range[0], radial_range[0] + delta_q * npt]
            else:
                npt, radial_range = None, None
            


            if (mask is None) and (mask_file is None):
                pass
            elif (mask_file is not None) and (mask is None):
                mask = fabio.open(mask_file).data
            elif (mask_file is not None) and (mask is not None):
                print('\nmask is provided. Ignoring mask_file\n')
                
            if (ai is None) and (poni_file is None):
                print('\n\nERROR: Valid a poni file or ai object is needed\n')
                return
            elif (poni_file is not None) and (ai is None):
                ai = pyFAI.load(poni_file)
            elif (poni_file is not None) and (ai is not None):
                print('\nAzimuthal integrator (ai) is provided. Ignoring poni_file\n')


            # integrate
            i2d = ai.integrate2d(
                data=img_array,
                npt_rad=npt,
                npt_azim=360,
                filename=None,
                correctSolidAngle=True,
                variance=None,
                error_model=None,
                radial_range=radial_range,
                azimuth_range=None,
                mask=mask,
                dummy=np.NaN,
                delta_dummy=None,
                polarization_factor=None,
                dark=None,
                flat=None,
                method="bbox",
                unit="q_A^-1",
                safe=True,
                normalization_factor=1.0,
                metadata=None,
            )

            self.ds["i2d"] = xr.DataArray(
                data=i2d.intensity.astype("float32"),
                coords=[i2d.azimuthal.astype("float32"), i2d.radial.astype("float32")],
                dims=["azimuthal_i2d", "radial_i2d"],
                attrs={
                    "radial_unit": "q_A^-1",
                    "xlabel": r"Scattering vector $q$ ($\AA^{-1}$)",
                    "ylabel": r"Azimuthal angle $\chi$ ($^{o}$)",
                    "detector_name": ai.__dict__["detector"].name,
                    "wavelength_in_meter": ai.__dict__["_wavelength"],
                    "detector_dist": ai.__dict__["_dist"],
                    "detector_poni1": ai.__dict__["_poni1"],
                    "detector_poni2": ai.__dict__["_poni2"],
                    "detector_rot1": ai.__dict__["_rot1"],
                    "detector_rot2": ai.__dict__["_rot2"],
                    "detector_rot3": ai.__dict__["_rot3"],
                },
            )

            da_i1d = xr.DataArray(
                data=self.ds["i2d"].mean(dim="azimuthal_i2d").astype("float32"),
                coords=[self.ds["i2d"].radial_i2d],
                dims=["radial"],
                attrs={
                    "radial_unit": "q_A^-1",
                    "xlabel": r"Scattering vector $q$ ($\AA^{-1}$)",
                    "ylabel": r"Intensity (a.u.)",
                    "wavelength_in_angst": ai.__dict__["_wavelength"] * 10e9,
                },
            )
            self.ds["i1d"] = da_i1d.dropna(dim="radial")

        elif (img_array is None) and (from_txt_file is not None):
            if os.path.isfile(from_txt_file):
                try:
                    X, Y = np.loadtxt(
                        from_txt_file,
                        comments=txt_file_comments,
                        skiprows=txt_file_skiprows,
                        usecols=txt_file_usecols,
                        unpack=True,
                    )
                    if txt_file_radial_unit.lower()[0] == "t":
                        X = (
                            (4 * np.pi) / (txt_file_wavelength_in_angstrom)
                        ) * np.sin(np.deg2rad(X) / 2)
                    elif txt_file_radial_unit.lower()[0] == "q":
                        pass
                    else:
                        print(
                            "Unable to determine radial unit. Check the radial_unit\n\n"
                        )
                        return
                    self.ds = xr.Dataset()
                    self.ds["i1d"] = xr.DataArray(
                        data=Y.astype("float32"),
                        coords=[X],
                        dims=["radial"],
                        attrs={
                            "radial_unit": "q_A^-1",
                            "xlabel": "Scattering vector $q$ ($\AA^{-1}$)",
                            "ylabel": "Intensity (a.u.)",
                            "wavelength_in_angst": txt_file_wavelength_in_angstrom,
                            "i1d_from": from_txt_file,
                        },
                    )

                    if radial_range is not None:
                        self.ds = self.ds.sel(radial=slice(radial_range[0],radial_range[1]))
                except Exception as exc:
                    print(
                        "Unable to read %s \nPlease check %s is a valid plain text file\n\n"
                        % (from_txt_file, from_txt_file)
                    )
                    print("Error msg from np.loadtxt:\n%s" % exc)
                    return
            else:
                print("%s does not exist. Please check the file path." % from_txt_file)
                return

        elif ((from_img_array is None) and (from_txt_file is None)) and (
            from_nc_file is not None
        ):
            with xr.open_dataset(from_nc_file) as self.ds:
                pass

        if plot:
            exrd_plotter(
                self.ds,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                plot_hint="load_xrd_data",
            )

    def get_baseline(
        self,
        input_bkg=None,
        use_iarpls=True,
        iarpls_lam=1e5,
        plot=True,
        get_i2d_baseline=False,
        use_i2d_baseline=False,
        roi_radial_range=None,
        roi_azimuthal_range=None,
        include_baseline_in_ds=True,
        spotty_data_correction=False,
        spotty_data_correction_threshold=1,
    ):
        


        for k in ["i1d_refined", "i1d_gsas_background"]:
            if k in self.ds.keys():
                del self.ds[k]


        if (input_bkg is None) and (use_iarpls is False):
            print(
                "\n\nYou did not provide input_bkg and use_iarpls is set to False. Nothing to do here. baseline is not calculated...\n\n"
            )
            plot = False

        else:

            if input_bkg is not None:
                if (("i2d" in self.ds.keys())) and ("i2d" in input_bkg.ds.keys()):
                    # check if they have same radial and azimuthal
                    if (
                        np.array_equal(input_bkg.ds.radial_i2d, self.ds.radial_i2d)
                    ) and (
                        np.array_equal(
                            input_bkg.ds.azimuthal_i2d, self.ds.azimuthal_i2d
                        )
                    ):
                        for k in ["radial", "i1d", "i1d_baseline", "i2d_baseline"]:
                            if k in self.ds.keys():
                                del self.ds[k]

                        if roi_azimuthal_range is not None:
                            da_i2d = self.ds.i2d.sel(
                                azimuthal_i2d=slice(
                                    roi_azimuthal_range[0], roi_azimuthal_range[1]
                                )
                            )
                            da_i2d.values = median_filter(da_i2d.values, size=3)
                            da_i2d_bkg = input_bkg.ds.i2d.sel(
                                azimuthal_i2d=slice(
                                    roi_azimuthal_range[0], roi_azimuthal_range[1]
                                )
                            )
                            da_i2d_bkg.values = median_filter(da_i2d_bkg.values, size=3)
                            da_i1d = da_i2d.mean(dim="azimuthal_i2d").dropna(
                                dim="radial_i2d"
                            )
                            da_i1d_bkg = da_i2d_bkg.mean(dim="azimuthal_i2d").dropna(
                                dim="radial_i2d"
                            )
                        else:
                            da_i2d = self.ds.i2d
                            da_i2d.values = median_filter(da_i2d.values, size=3)
                            da_i2d_bkg = input_bkg.ds.i2d
                            da_i2d_bkg.values = median_filter(da_i2d_bkg.values, size=3)
                            da_i1d = self.ds.i2d.mean(dim="azimuthal_i2d").dropna(
                                dim="radial_i2d"
                            )
                            da_i1d_bkg = input_bkg.ds.i2d.mean(
                                dim="azimuthal_i2d"
                            ).dropna(dim="radial_i2d")

                        if roi_radial_range is not None:
                            bkg_scale = 1
                            diff_now = (
                                da_i1d.sel(
                                    radial_i2d=slice(
                                        roi_radial_range[0], roi_radial_range[-1]
                                    )
                                )
                                - bkg_scale
                                * da_i1d_bkg.sel(
                                    radial_i2d=slice(
                                        roi_radial_range[0], roi_radial_range[-1]
                                    )
                                )
                            ).values
                            if min(diff_now) > 0:
                                while min(diff_now) > 0:
                                    bkg_scale = bkg_scale * 1.01
                                    diff_now = (
                                        da_i1d.sel(
                                            radial_i2d=slice(
                                                roi_radial_range[0],
                                                roi_radial_range[-1],
                                            )
                                        )
                                        - bkg_scale
                                        * da_i1d_bkg.sel(
                                            radial_i2d=slice(
                                                roi_radial_range[0],
                                                roi_radial_range[-1],
                                            )
                                        )
                                    ).values
                            else:
                                while min(diff_now) < 0:
                                    bkg_scale = bkg_scale * 0.99
                                    diff_now = (
                                        da_i1d.sel(
                                            radial_i2d=slice(
                                                roi_radial_range[0],
                                                roi_radial_range[-1],
                                            )
                                        )
                                        - bkg_scale
                                        * da_i1d_bkg.sel(
                                            radial_i2d=slice(
                                                roi_radial_range[0],
                                                roi_radial_range[-1],
                                            )
                                        )
                                    ).values
                        else:
                            bkg_scale = 1
                            diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values
                            if min(diff_now) > 0:
                                while min(diff_now) > 0:
                                    bkg_scale = bkg_scale * 1.01
                                    diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values
                            else:
                                while min(diff_now) < 0:
                                    bkg_scale = bkg_scale * 0.99
                                    diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values

                        if use_iarpls:
                            if roi_azimuthal_range is not None:
                                da_i2d_diff = self.ds.i2d.sel(
                                    azimuthal_i2d=slice(
                                        roi_azimuthal_range[0], roi_azimuthal_range[1]
                                    )
                                ) - bkg_scale * input_bkg.ds.i2d.sel(
                                    azimuthal_i2d=slice(
                                        roi_azimuthal_range[0], roi_azimuthal_range[1]
                                    )
                                )
                            else:
                                da_i2d_diff = self.ds.i2d - bkg_scale * input_bkg.ds.i2d
                            if get_i2d_baseline:
                                da_i2d_diff_baseline = deepcopy(da_i2d_diff)
                                # serial version (can be speed-up using threads)
                                for a_ind in range(da_i2d_diff.shape[0]):
                                    #
                                    da_now = da_i2d_diff_baseline.isel(
                                        azimuthal_i2d=a_ind
                                    )
                                    da_now_dropna = da_now.dropna(dim="radial_i2d")
                                    try:
                                        baseline_now, params = pybaselines.Baseline(
                                            x_data=da_now_dropna.radial_i2d.values
                                        ).iarpls(da_now_dropna.values, lam=iarpls_lam)
                                        # create baseline da by copying
                                        da_now_dropna_baseline = copy.deepcopy(
                                            da_now_dropna
                                        )
                                        da_now_dropna_baseline.values = baseline_now
                                        # now interpolate baseline da to original i2d radial range
                                        da_now_dropna_baseline_interpolated = (
                                            da_now_dropna_baseline.interp(
                                                radial_i2d=da_i2d_diff.radial_i2d
                                            )
                                        )
                                        da_i2d_diff_baseline[a_ind, :] = (
                                            da_now_dropna_baseline_interpolated
                                        )
                                    except:
                                        # da_now.values[:] = np.nan
                                        da_i2d_diff_baseline[a_ind, :] = da_now
                                if roi_azimuthal_range is not None:
                                    self.ds["i2d_baseline"] = da_i2d_diff_baseline + (
                                        bkg_scale
                                        * input_bkg.ds.i2d.sel(
                                            azimuthal_i2d=slice(
                                                roi_azimuthal_range[0],
                                                roi_azimuthal_range[1],
                                            )
                                        )
                                    )
                                else:
                                    self.ds["i2d_baseline"] = da_i2d_diff_baseline + (
                                        bkg_scale * input_bkg.ds.i2d
                                    )
                                self.ds["i2d_baseline"].attrs[
                                    "baseline_note"
                                ] = "baseline is from provided input_bkg and iarpls is used"
                                self.ds["i2d_baseline"].attrs["iarpls_lam"] = iarpls_lam

                                if use_i2d_baseline:
                                    self.ds["i1d_baseline"] = (
                                        self.ds["i2d_baseline"]
                                        .mean(dim="azimuthal_i2d")
                                        .rename({"radial_i2d": "radial"})
                                    )
                                    self.ds["i1d_baseline"].attrs[
                                        "baseline_note"
                                    ] = "baseline is from i2d_baseline as available in this dataset. iarpls is used"
                                    self.ds["i1d_baseline"].attrs[
                                        "iarpls_lam"
                                    ] = iarpls_lam
                                else:
                                    da_for_baseline = da_i2d_diff.mean(
                                        dim="azimuthal_i2d"
                                    ).dropna(dim="radial_i2d")
                                    diff_baseline, params = pybaselines.Baseline(
                                        x_data=da_for_baseline.radial_i2d.values
                                    ).iarpls(da_for_baseline.values, lam=iarpls_lam)
                                    if roi_azimuthal_range is not None:
                                        self.ds["i1d_baseline"] = (
                                            xr.DataArray(
                                                data=(
                                                    diff_baseline
                                                    + bkg_scale
                                                    * input_bkg.ds.i2d.sel(
                                                        azimuthal_i2d=slice(
                                                            roi_azimuthal_range[0],
                                                            roi_azimuthal_range[1],
                                                        )
                                                    )
                                                    .mean(dim="azimuthal_i2d")
                                                    .dropna(dim="radial_i2d")
                                                    .values
                                                ),
                                                dims=["radial_i2d"],
                                                coords={
                                                    "radial_i2d": da_for_baseline.radial_i2d.values
                                                },
                                                attrs={"iarpls_lam": iarpls_lam},
                                            )
                                            .interp(radial_i2d=self.ds.i2d.radial_i2d)
                                            .rename({"radial_i2d": "radial"})
                                        )
                                    else:
                                        self.ds["i1d_baseline"] = (
                                            xr.DataArray(
                                                data=(
                                                    diff_baseline
                                                    + bkg_scale
                                                    * input_bkg.ds.i2d.mean(
                                                        dim="azimuthal_i2d"
                                                    )
                                                    .dropna(dim="radial_i2d")
                                                    .values
                                                ),
                                                dims=["radial_i2d"],
                                                coords={
                                                    "radial_i2d": da_for_baseline.radial_i2d.values
                                                },
                                                attrs={"iarpls_lam": iarpls_lam},
                                            )
                                            .interp(radial_i2d=self.ds.i2d.radial_i2d)
                                            .rename({"radial_i2d": "radial"})
                                        )
                                    self.ds["i1d_baseline"].attrs[
                                        "baseline_note"
                                    ] = "baseline is from provided input_bkg and iarpls is used"
                                    self.ds["i1d_baseline"].attrs[
                                        "iarpls_lam"
                                    ] = iarpls_lam

                            else:
                                da_for_baseline = da_i2d_diff.mean(
                                    dim="azimuthal_i2d"
                                ).dropna(dim="radial_i2d")
                                diff_baseline, params = pybaselines.Baseline(
                                    x_data=da_for_baseline.radial_i2d.values
                                ).iarpls(da_for_baseline.values, lam=iarpls_lam)
                                if roi_azimuthal_range is not None:
                                    self.ds["i1d_baseline"] = (
                                        xr.DataArray(
                                            data=(
                                                diff_baseline
                                                + bkg_scale
                                                * input_bkg.ds.i2d.sel(
                                                    azimuthal_i2d=slice(
                                                        roi_azimuthal_range[0],
                                                        roi_azimuthal_range[1],
                                                    )
                                                )
                                                .mean(dim="azimuthal_i2d")
                                                .dropna(dim="radial_i2d")
                                                .values
                                            ),
                                            dims=["radial_i2d"],
                                            coords={
                                                "radial_i2d": da_for_baseline.radial_i2d.values
                                            },
                                            attrs={"iarpls_lam": iarpls_lam},
                                        )
                                        .interp(radial_i2d=self.ds.i2d.radial_i2d)
                                        .rename({"radial_i2d": "radial"})
                                    )
                                else:
                                    self.ds["i1d_baseline"] = (
                                        xr.DataArray(
                                            data=(
                                                diff_baseline
                                                + bkg_scale
                                                * input_bkg.ds.i2d.mean(
                                                    dim="azimuthal_i2d"
                                                )
                                                .dropna(dim="radial_i2d")
                                                .values
                                            ),
                                            dims=["radial_i2d"],
                                            coords={
                                                "radial_i2d": da_for_baseline.radial_i2d.values
                                            },
                                            attrs={"iarpls_lam": iarpls_lam},
                                        )
                                        .interp(radial_i2d=self.ds.i2d.radial_i2d)
                                        .rename({"radial_i2d": "radial"})
                                    )
                                self.ds["i1d_baseline"].attrs[
                                    "baseline_note"
                                ] = "baseline is from provided input_bkg and iarpls is used"
                                self.ds["i1d_baseline"].attrs["iarpls_lam"] = iarpls_lam
                        else:
                            if roi_azimuthal_range is not None:
                                self.ds["i2d_baseline"] = deepcopy(
                                    bkg_scale
                                    * input_bkg.ds.i2d.sel(
                                        azimuthal_i2d=slice(
                                            roi_azimuthal_range[0],
                                            roi_azimuthal_range[1],
                                        )
                                    )
                                )
                            else:
                                self.ds["i2d_baseline"] = deepcopy(
                                    bkg_scale * input_bkg.ds.i2d
                                )
                            self.ds["i2d_baseline"].attrs[
                                "baseline_note"
                            ] = "baseline is from provided input_bkg. iarpls is not used"
                            self.ds["i1d_baseline"] = (
                                self.ds["i2d_baseline"]
                                .mean(dim="azimuthal_i2d")
                                .rename({"radial_i2d": "radial"})
                            )
                            self.ds["i1d_baseline"].attrs[
                                "baseline_note"
                            ] = "baseline is from i2d_baseline as available in this dataset. iarpls is not used"

                    else:
                        # TODO
                        print(
                            "dimensions do not match.... ignoring input_bkg and getting baseline via iarpls"
                        )

                elif (("i2d" in self.ds.keys())) and ("i1d" in input_bkg.ds.keys()):

                    if roi_azimuthal_range is not None:
                        da_i2d = self.ds.i2d.sel(
                            azimuthal_i2d=slice(
                                roi_azimuthal_range[0], roi_azimuthal_range[1]
                            )
                        ).rename({"radial_i2d": "radial"})
                        da_i2d.values = median_filter(da_i2d.values, size=3)
                        da_i1d = da_i2d.mean(dim="azimuthal_i2d").dropna(dim="radial")
                        da_i1d_bkg = input_bkg.ds.i1d
                    else:
                        da_i2d = self.ds.i2d.rename({"radial_i2d": "radial"})
                        da_i2d.values = median_filter(da_i2d.values, size=3)
                        da_i1d = da_i2d.mean(dim="azimuthal_i2d").dropna(dim="radial")
                        da_i1d_bkg = input_bkg.ds.i1d

                    if roi_radial_range is not None:
                        bkg_scale = 1
                        diff_now = (
                            da_i1d.sel(
                                radial=slice(roi_radial_range[0], roi_radial_range[-1])
                            )
                            - bkg_scale
                            * da_i1d_bkg.sel(
                                radial=slice(roi_radial_range[0], roi_radial_range[-1])
                            )
                        ).values
                        if min(diff_now) > 0:
                            while min(diff_now) > 0:
                                bkg_scale = bkg_scale * 1.01
                                diff_now = (
                                    da_i1d.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                    - bkg_scale
                                    * da_i1d_bkg.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                ).values
                        else:
                            while min(diff_now) < 0:
                                bkg_scale = bkg_scale * 0.99
                                diff_now = (
                                    da_i1d.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                    - bkg_scale
                                    * da_i1d_bkg.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                ).values
                    else:
                        bkg_scale = 1
                        diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values
                        if min(diff_now) > 0:
                            while min(diff_now) > 0:
                                bkg_scale = bkg_scale * 1.01
                                diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values
                        else:
                            while min(diff_now) < 0:
                                bkg_scale = bkg_scale * 0.99
                                diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values

                    if use_iarpls:

                        # pass

                        if roi_azimuthal_range is not None:
                            da_i1d_diff = (
                                self.ds.i2d.sel(
                                    azimuthal_i2d=slice(
                                        roi_azimuthal_range[0], roi_azimuthal_range[1]
                                    )
                                )
                                .mean(dim="azimuthal_i2d")
                                .rename({"radial_i2d": "radial"})
                                - bkg_scale * input_bkg.ds.i1d
                            )
                        else:
                            da_i1d_diff = (
                                self.ds.i2d.mean(dim="azimuthal_i2d").rename(
                                    {"radial_i2d": "radial"}
                                )
                                - bkg_scale * input_bkg.ds.i1d
                            )

                        da_for_baseline = da_i1d_diff  # .dropna(dim='radial')
                        diff_baseline, params = pybaselines.Baseline(
                            x_data=da_for_baseline.radial.values
                        ).iarpls(da_for_baseline.values, lam=iarpls_lam)

                        self.ds["i1d_baseline"] = xr.DataArray(
                            data=(diff_baseline + bkg_scale * input_bkg.ds.i1d.values),
                            dims=["radial"],
                            coords={"radial": input_bkg.ds.i1d.radial.values},
                            attrs={"iarpls_lam": iarpls_lam},
                        )
                        self.ds["i1d_baseline"].attrs[
                            "baseline_note"
                        ] = "baseline is from provided input_bkg i1d and iarpls is used"
                    else:
                        self.ds["i1d_baseline"] = deepcopy(bkg_scale * input_bkg.ds.i1d)
                        self.ds["i1d_baseline"].attrs[
                            "baseline_note"
                        ] = "baseline is from provided input_bkg. i1d iarpls is not used"

                else:

                    da_i1d = self.ds.i1d
                    da_i1d_bkg = input_bkg.ds.i1d

                    if roi_radial_range is not None:
                        bkg_scale = 1
                        diff_now = (
                            da_i1d.sel(
                                radial=slice(roi_radial_range[0], roi_radial_range[-1])
                            )
                            - bkg_scale
                            * da_i1d_bkg.sel(
                                radial=slice(roi_radial_range[0], roi_radial_range[-1])
                            )
                        ).values
                        if min(diff_now) > 0:
                            while min(diff_now) > 0:
                                bkg_scale = bkg_scale * 1.01
                                diff_now = (
                                    da_i1d.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                    - bkg_scale
                                    * da_i1d_bkg.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                ).values
                        else:
                            while min(diff_now) < 0:
                                bkg_scale = bkg_scale * 0.99
                                diff_now = (
                                    da_i1d.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                    - bkg_scale
                                    * da_i1d_bkg.sel(
                                        radial=slice(
                                            roi_radial_range[0], roi_radial_range[-1]
                                        )
                                    )
                                ).values
                    else:
                        bkg_scale = 1
                        diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values
                        if min(diff_now) > 0:
                            while min(diff_now) > 0:
                                bkg_scale = bkg_scale * 1.01
                                diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values
                        else:
                            while min(diff_now) < 0:
                                bkg_scale = bkg_scale * 0.99
                                diff_now = (da_i1d - bkg_scale * da_i1d_bkg).values

                    if use_iarpls:

                        da_i1d_diff = self.ds.i1d - bkg_scale * input_bkg.ds.i1d

                        da_for_baseline = da_i1d_diff  # .dropna(dim='radial')
                        diff_baseline, params = pybaselines.Baseline(
                            x_data=da_for_baseline.radial.values
                        ).iarpls(da_for_baseline.values, lam=iarpls_lam)

                        self.ds["i1d_baseline"] = xr.DataArray(
                            data=(diff_baseline + bkg_scale * input_bkg.ds.i1d.values),
                            dims=["radial"],
                            coords={"radial": input_bkg.ds.i1d.radial.values},
                            attrs={"iarpls_lam": iarpls_lam},
                        )
                        self.ds["i1d_baseline"].attrs[
                            "baseline_note"
                        ] = "baseline is from provided input_bkg i1d and iarpls is used"
                    else:
                        self.ds["i1d_baseline"] = deepcopy(bkg_scale * input_bkg.ds.i1d)
                        self.ds["i1d_baseline"].attrs[
                            "baseline_note"
                        ] = "baseline is from provided input_bkg. i1d iarpls is not used"

            else:

                if "i2d" in self.ds.keys():

                    for k in ["radial", "i1d", "i1d_baseline", "i2d_baseline"]:
                        if k in self.ds.keys():
                            del self.ds[k]

                    if use_iarpls:

                        if roi_azimuthal_range is not None:
                            da_i2d = self.ds.i2d.sel(
                                azimuthal_i2d=slice(
                                    roi_azimuthal_range[0], roi_azimuthal_range[1]
                                )
                            )
                        else:
                            da_i2d = self.ds.i2d

                        if get_i2d_baseline:
                            da_i2d_baseline = deepcopy(da_i2d)
                            # serial version (can be speed-up using threads)
                            for a_ind in range(da_i2d.shape[0]):
                                #
                                da_now = da_i2d_baseline.isel(azimuthal_i2d=a_ind)
                                da_now_dropna = da_now.dropna(dim="radial_i2d")
                                try:
                                    baseline_now, params = pybaselines.Baseline(
                                        x_data=da_now_dropna.radial_i2d.values
                                    ).iarpls(da_now_dropna.values, lam=iarpls_lam)
                                    # create baseline da by copying
                                    da_now_dropna_baseline = copy.deepcopy(
                                        da_now_dropna
                                    )
                                    da_now_dropna_baseline.values = baseline_now
                                    # now interpolate baseline da to original i2d radial range
                                    da_now_dropna_baseline_interpolated = (
                                        da_now_dropna_baseline.interp(
                                            radial_i2d=da_i2d.radial_i2d
                                        )
                                    )
                                    da_i2d_baseline[a_ind, :] = (
                                        da_now_dropna_baseline_interpolated
                                    )
                                except:
                                    # da_now.values[:] = np.nan
                                    da_i2d_baseline[a_ind, :] = da_now
                            self.ds["i2d_baseline"] = da_i2d_baseline
                            self.ds["i2d_baseline"].attrs[
                                "baseline_note"
                            ] = "baseline is estimated with iarpls"
                            self.ds["i2d_baseline"].attrs["iarpls_lam"] = iarpls_lam

                            if use_i2d_baseline:
                                self.ds["i1d_baseline"] = (
                                    self.ds["i2d_baseline"]
                                    .mean(dim="azimuthal_i2d")
                                    .rename({"radial_i2d": "radial"})
                                )
                                self.ds["i1d_baseline"].attrs[
                                    "baseline_note"
                                ] = "baseline is from i2d_baseline as available in this dataset. iarpls is used"
                                self.ds["i1d_baseline"].attrs["iarpls_lam"] = iarpls_lam
                            else:
                                da_for_baseline = da_i2d.mean(
                                    dim="azimuthal_i2d"
                                ).dropna(dim="radial_i2d")
                                baseline, params = pybaselines.Baseline(
                                    x_data=da_for_baseline.radial_i2d.values
                                ).iarpls(da_for_baseline.values, lam=iarpls_lam)
                                self.ds["i1d_baseline"] = (
                                    xr.DataArray(
                                        data=(baseline),
                                        dims=["radial_i2d"],
                                        coords={
                                            "radial_i2d": da_for_baseline.radial_i2d.values
                                        },
                                        attrs={"iarpls_lam": iarpls_lam},
                                    )
                                    .interp(radial_i2d=da_i2d.radial_i2d)
                                    .rename({"radial_i2d": "radial"})
                                )
                                self.ds["i1d_baseline"].attrs[
                                    "baseline_note"
                                ] = "baseline is estimated with iarpls"
                                self.ds["i1d_baseline"].attrs["iarpls_lam"] = iarpls_lam

                        else:
                            da_for_baseline = da_i2d.mean(dim="azimuthal_i2d").dropna(
                                dim="radial_i2d"
                            )
                            baseline, params = pybaselines.Baseline(
                                x_data=da_for_baseline.radial_i2d.values
                            ).iarpls(da_for_baseline.values, lam=iarpls_lam)
                            self.ds["i1d_baseline"] = (
                                xr.DataArray(
                                    data=(baseline),
                                    dims=["radial_i2d"],
                                    coords={
                                        "radial_i2d": da_for_baseline.radial_i2d.values
                                    },
                                    attrs={"iarpls_lam": iarpls_lam},
                                )
                                .interp(radial_i2d=da_i2d.radial_i2d)
                                .rename({"radial_i2d": "radial"})
                            )
                            self.ds["i1d_baseline"].attrs[
                                "baseline_note"
                            ] = "baseline is estimated with iarpls"
                            self.ds["i1d_baseline"].attrs["iarpls_lam"] = iarpls_lam

                    else:
                        self.ds["i1d_baseline"] = (
                            self.ds.i2d.mean(dim="azimuthal_i2d") * 0
                        )
                        bkg_scale = da_i1d.values[0] / da_i1d_bkg.values[0]
                        while min((da_i1d.values - bkg_scale * da_i1d_bkg.values)) < 0:
                            bkg_scale = bkg_scale * 0.99


                else:

                    if iarpls_lam:
                        baseline, params = pybaselines.Baseline(
                            x_data=self.ds.i1d.radial.values
                        ).iarpls(self.ds.i1d.values, lam=iarpls_lam)                 
                        self.ds["i1d_baseline"] = (
                            xr.DataArray(
                                data=(baseline),
                                dims=["radial"],
                                coords={
                                    "radial": self.ds.i1d.radial.values
                                },
                                attrs={"iarpls_lam": iarpls_lam},
                            )
                        )
                        self.ds["i1d_baseline"].attrs[
                            "baseline_note"
                        ] = "baseline is estimated with iarpls"
                        self.ds["i1d_baseline"].attrs["iarpls_lam"] = iarpls_lam



        if "i2d" in self.ds.keys():
            if roi_azimuthal_range is not None:
                self.ds["i1d"] = (
                    self.ds["i2d"]
                    .sel(
                        azimuthal_i2d=slice(
                            roi_azimuthal_range[0], roi_azimuthal_range[1]
                        )
                    )
                    .mean(dim="azimuthal_i2d")
                    .rename({"radial_i2d": "radial"})
                )
                self.ds["i1d"].attrs = {
                    "radial_unit": "q_A^-1",
                    "xlabel": "Scattering vector $q$ ($\AA^{-1}$)",
                    "ylabel": "Intensity (a.u.)",
                    "wavelength_in_angst": self.ds["i2d"].attrs["wavelength_in_meter"]
                    * 10e9,
                    "roi_azimuthal_range": roi_azimuthal_range,
                }
                self.ds["i2d"].attrs["roi_azimuthal_range"] = roi_azimuthal_range
            else:
                self.ds["i1d"] = (
                    self.ds["i2d"]
                    .mean(dim="azimuthal_i2d")
                    .rename({"radial_i2d": "radial"})
                )
                self.ds["i1d"].attrs = {
                    "radial_unit": "q_A^-1",
                    "xlabel": "Scattering vector $q$ ($\AA^{-1}$)",
                    "ylabel": "Intensity (a.u.)",
                    "wavelength_in_angst": self.ds["i2d"].attrs["wavelength_in_meter"]
                    * 10e9,
                }

        if roi_radial_range is not None:
            self.ds = self.ds.sel(
                radial=slice(roi_radial_range[0], roi_radial_range[-1])
            ).dropna(dim="radial")
        else:
            self.ds = self.ds.dropna(dim="radial")

        if spotty_data_correction:
            da_diff = self.ds.i2d - self.ds.i2d_baseline

            self.ds["i2d"] = (self.ds["i2d"]).where(
                da_diff >= spotty_data_correction_threshold
            )

            i1d_attrs = copy.deepcopy(self.ds.i1d.attrs)
            self.ds["i1d"] = (
                (
                    (
                        (self.ds["i2d"])
                        .where(da_diff >= spotty_data_correction_threshold)
                        .mean(dim="azimuthal_i2d")
                    )
                    - spotty_data_correction_threshold
                )
                .rename({"radial_i2d": "radial"})
                .fillna(0)
            )
            self.ds["i1d"].attrs = i1d_attrs

        if plot:
            exrd_plotter(
                ds=self.ds,
                ds_previous=None,
                phases=None,
                figsize=self.figsize,
                i2d_robust=self.i2d_robust,
                i2d_logscale=self.i2d_logscale,
                i1d_ylogscale=self.i1d_ylogscale,
                title_str=None,
                export_fig_as=None,
                plot_hint="get_baseline",
            )

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
            try:
                mp_rester_api_key=easyxrd_defaults["mp_api_key"]
            except:
                mp_rester_api_key='none'

        self.easyxrd_scratch_directory = easyxrd_defaults["easyxrd_scratch_path"]

        if from_phases_dict is not None:






            self.phases = {}
            for e, p in enumerate(from_phases_dict):

                try:
                    mp_id = p["mp_id"]
                except:
                    mp_id = 'none'

                if mp_id.lower() == "none":
                    st = Structure.from_file(p["cif"])

                    try:
                        scale = p["scale"]
                    except:
                        scale = 1
                    try:
                        scale_a = p["scale_a"]
                    except:
                        scale_a = 1
                    try:
                        scale_b = p["scale_b"]
                    except:
                        scale_b = 1
                    try:
                        scale_c = p["scale_c"]
                    except:
                        scale_c = 1

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

                    if mp_rester_api_key.lower() == 'none':
                        mp_rester_api_key = input(
                                            "\nIn order to retrieve structural information from Materials Project, api_ket is needed. \nPlease enter your 32 character key it here:\n"
                                        )
                        easyxrd_defaults["mp_api_key"] = mp_rester_api_key

                    from mp_api.client import MPRester

                    mpr = MPRester(mp_rester_api_key)
                    st = mpr.get_structure_by_material_id(mp_id, final=False)[0]

                    try:
                        scale = p["scale"]
                    except:
                        scale = 1
                    try:
                        scale_a = p["scale_a"]
                    except:
                        scale_a = 1
                    try:
                        scale_b = p["scale_b"]
                    except:
                        scale_b = 1
                    try:
                        scale_c = p["scale_c"]
                    except:
                        scale_c = 1

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

            self.ds.attrs["num_phases"] = e + 1

        elif from_gpx is not None:
            import GSASIIscriptable as G2sc

            phases_gpx = G2sc.G2Project(gpxfile=from_gpx)
            self.phases = {}
            for e, p in enumerate(phases_gpx.phases()):
                p.export_CIF(outputname="tmp.cif")
                st = Structure.from_file("tmp.cif")
                self.phases[p.name] = st
                with open("tmp.cif", "r") as ciffile:
                    ciffile_content = ciffile.read()
                    self.ds.attrs["PhaseInd_%d_cif" % (e)] = ciffile_content
                self.ds.attrs["PhaseInd_%d_label" % (e)] = p.name
                os.remove("tmp.cif")
            self.ds.attrs["num_phases"] = e + 1

        elif from_nc is not None:

            with xr.open_dataset(from_nc) as ds_nc:

                self.phases = {}
                for p in range(ds_nc.attrs["num_phases"]):

                    randstr = "".join(
                        random.choices(string.ascii_uppercase + string.digits, k=7)
                    )
                    with open(
                        "%s/%s.cif" % (self.easyxrd_scratch_directory, randstr), "w"
                    ) as ciffile:
                        ciffile.write("%s" % ds_nc.attrs["PhaseInd_%d_cif" % p])
                    st = Structure.from_file(
                        "%s/%s.cif" % (self.easyxrd_scratch_directory, randstr)
                    )
                    self.phases[ds_nc.attrs["PhaseInd_%d_label" % p]] = st
                    os.remove("%s/%s.cif" % (self.easyxrd_scratch_directory, randstr))

                    self.ds.attrs["PhaseInd_%d_label" % (p)] = ds_nc.attrs[
                        "PhaseInd_%d_label" % (p)
                    ]
                    self.ds.attrs["PhaseInd_%d_cif" % (p)] = ds_nc.attrs[
                        "PhaseInd_%d_cif" % (p)
                    ]

                self.ds.attrs["num_phases"] = p + 1

        elif from_ds is not None:

            self.phases = {}
            for p in range(from_ds.attrs["num_phases"]):

                randstr = "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=7)
                )
                with open(
                    "%s/%s.cif" % (self.easyxrd_scratch_directory, randstr), "w"
                ) as ciffile:
                    ciffile.write("%s" % from_ds.attrs["PhaseInd_%d_cif" % p])
                st = Structure.from_file(
                    "%s/%s.cif" % (self.easyxrd_scratch_directory, randstr)
                )
                self.phases[from_ds.attrs["PhaseInd_%d_label" % p]] = st
                os.remove("%s/%s.cif" % (self.easyxrd_scratch_directory, randstr))

                self.ds.attrs["PhaseInd_%d_label" % (p)] = from_ds.attrs[
                    "PhaseInd_%d_label" % (p)
                ]
                self.ds.attrs["PhaseInd_%d_cif" % (p)] = from_ds.attrs[
                    "PhaseInd_%d_cif" % (p)
                ]

            self.ds.attrs["num_phases"] = p + 1

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

    def setup_gsas2_refiner(
        self,
        gsasii_lib_directory=None,
        gsasii_scratch_directory=None,
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

        try:
            del self.gsasii_lib_directory
        except:
            pass
        try:
            del self.gpx
        except:
            pass

        for k in ["i1d_refined", "i1d_gsas_background"]:
            if k in self.ds.keys():
                del self.ds[k]

        self.yshift_multiplier = yshift_multiplier

        if easyxrd_defaults["gsasii_lib_path"] == "not found":
            try:
                default_install_path = os.path.join(
                    os.path.expanduser("~"), "g2full/GSAS-II/GSASII"
                )
                sys.path += [default_install_path]
                import GSASIIscriptable as G2sc
                import GSASIIlattice as G2lat

                self.gsasii_lib_directory = default_install_path
            except:
                user_loc = input(
                    "Enter location of GSASII directory on your GSAS-II installation."
                )
                sys.path += [user_loc]
                try:
                    import GSASIIscriptable as G2sc
                    import GSASIIlattice as G2lat

                    self.gsasii_lib_directory = user_loc
                except:
                    try:
                        user_loc = input(
                            "\nUnable to import GSASIIscriptable. Please re-enter GSASII directory on your GSAS-II installation\n"
                        )
                        sys.path += [user_loc]
                        import GSASIIscriptable as G2sc
                        import GSASIIlattice as G2lat

                        self.gsasii_lib_directory = user_loc
                    except:
                        print(
                            "\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html"
                        )
        else:
            if os.path.isdir(easyxrd_defaults["gsasii_lib_path"]):
                sys.path += [easyxrd_defaults["gsasii_lib_path"]]
                try:
                    import GSASIIscriptable as G2sc
                    import GSASIIlattice as G2lat

                    self.gsasii_lib_directory = easyxrd_defaults["gsasii_lib_path"]
                except:
                    try:
                        gsasii_lib_directory = input(
                            "\nUnable to import GSASIIscriptable. Please enter GSASII directory on your GSAS-II installation\n"
                        )
                        sys.path += [gsasii_lib_directory]
                        import GSASIIscriptable as G2sc
                        import GSASIIlattice as G2lat

                        self.gsasii_lib_directory = gsasii_lib_directory
                    except:
                        gsasii_lib_directory = print(
                            "\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html"
                        )
            else:
                print("%s does NOT exist. Please check!" % gsasii_lib_directory)

        self.easyxrd_scratch_directory = easyxrd_defaults["easyxrd_scratch_path"]

        randstr = "test"
        self.gsasii_run_directory = "%s/%.2f_%s.gsastmp" % (
            self.easyxrd_scratch_directory,
            10.0,
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
        title_str = "Background with %d coeffs is refined. %s" % (num_coeffs,ref_str)
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
                title_str = "After setting LeBail refinement to True, %s" % (ref_str)
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




    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def plot(
        self,
        plot_hint=None,
        figsize=None,
        i2d_robust=None,
        i2d_logscale=None,
        i1d_ylogscale=None,
        export_fig_as=None,
        i1d_plot_radial_range=None,
        i1d_plot_bottom=None,
        i1d_plot_top=None,
        title=None,
        site_str_x=0.4,
        site_str_y=0.8,
        show_wt_fractions=False,
    ):

        if figsize is None:
            figsize = self.figsize

        if i2d_robust is None:
            i2d_robust = self.i2d_robust

        if i2d_logscale is None:
            i2d_logscale = self.i2d_logscale

        if i1d_ylogscale is None:
            i1d_ylogscale = self.i1d_ylogscale

        try:
            ds=self.ds
        except:
            ds=None

        try:
            ds_previous=self.ds_previous
        except:
            ds_previous=None


        exrd_plotter(
            ds=ds,
            ds_previous=ds_previous,
            figsize=figsize,
            i2d_robust=i2d_robust,
            i2d_logscale=i2d_logscale,
            i1d_ylogscale=i1d_ylogscale,
            plot_hint=plot_hint,
            export_fig_as=export_fig_as,
            i1d_plot_radial_range=i1d_plot_radial_range,
            i1d_plot_bottom=i1d_plot_bottom,
            i1d_plot_top=i1d_plot_top,
            title=title,
            site_str_x=site_str_x,
            site_str_y=site_str_y,
            show_wt_fractions=show_wt_fractions,
        )

















    ###############################################################################################
    def export_ds_to(self, to=None, save_dir=None, save_name=None):
        """ """

        if to is None:
            try:
                self.ds.to_netcdf(
                    "%s/%s" % (save_dir, save_name),
                    engine="h5netcdf",
                    encoding={"i2d": {"zlib": True, "complevel": 9}},
                )  # pip install h5netcdf
            except:
                self.ds.to_netcdf(
                    "%s/%s" % (save_dir, save_name),
                    engine="h5netcdf",
                )  # pip install h5netcdf
        else:
            try:
                self.ds.to_netcdf(
                    "%s" % (to),
                    engine="h5netcdf",
                    encoding={"i2d": {"zlib": True, "complevel": 9}},
                )  # pip install h5netcdf
            except:
                self.ds.to_netcdf(
                    "%s" % (to),
                    engine="h5netcdf",
                )  # pip install h5netcdf


    ###############################################################################################
    def fine_tune_gpx(self):
        """ """
        subprocess.check_call(
            [
                "%s/../../RunGSASII.sh" % self.gsasii_lib_directory,
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
