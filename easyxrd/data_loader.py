import os
import numpy as np
import xarray as xr
import fabio
import pyFAI
from scipy.signal import medfilt2d

from .plotters import exrd_plotter


class DataLoaderMixin:
    """
    Mixin class providing XRD data loading (1D and 2D) and data export capabilities.
    """

    def load_xrd_data(
        self,
        integrate2d=True,
        from_img_array=None,
        from_tiff_file=None,
        ai=None,
        poni_file=None,
        mask=None,
        mask_file=None,
        median_filter_kernel_size=None,
        median_filter_on_i2d=True,
        from_da_i2d=None,
        from_i1d_array=None,
        i1d_array_wavelength_in_angstrom=0.1814,
        i1d_array_radial_unit="q",
        from_da_i1d=None,
        from_nc_file=None,
        from_txt_file=None,
        txt_file_wavelength_in_angstrom=0.1814,
        txt_file_comments="#",
        txt_file_skiprows=0,
        txt_file_usecols=(0, 1),
        txt_file_radial_unit="tth",
        radial_range=[0.1, 11.1],
        radial_npts=1000,
        delta_q=0.0010,
        npt_azimuthal=91,
        plot=True,
        ds_attrs=None,
        method=("bbox", "csr", "cython"),
    ):

        if (from_img_array is None) and (from_tiff_file is not None):
            if (median_filter_kernel_size is not None) and (
                median_filter_on_i2d is False
            ):
                img_array = medfilt2d(
                    fabio.open(from_tiff_file).data.astype("float32"),
                    kernel_size=median_filter_kernel_size,
                )
            else:
                img_array = fabio.open(from_tiff_file).data.astype("float32")
        elif (from_img_array is not None) and (from_tiff_file is None):
            if (median_filter_kernel_size is not None) and (
                median_filter_on_i2d is False
            ):
                img_array = medfilt2d(
                    from_img_array.astype("float32"),
                    kernel_size=median_filter_kernel_size,
                )
            else:
                img_array = from_img_array
        else:
            img_array = None

        if img_array is not None:

            self.ds = xr.Dataset()

            try:
                self.ds.attrs = ds_attrs
            except Exception as exc:
                print(exc)
                print("Unable to include ds_attrs in self.ds")

            if radial_range is not None:
                npt = int(np.ceil((radial_range[1] - radial_range[0]) / delta_q))
                radial_range = [radial_range[0], radial_range[0] + delta_q * npt]
            else:
                npt, radial_range = None, None

            if (mask is None) and (mask_file is None):
                pass
            elif (mask_file is not None) and (mask is None):
                mask = fabio.open(mask_file).data
            elif (mask_file is not None) and (mask is not None):
                print("\nmask is provided. Ignoring mask_file\n")

            if (ai is None) and (poni_file is None):
                print("\n\nERROR: Valid a poni file or ai object is needed\n")
                return
            elif (poni_file is not None) and (ai is None):
                ai = pyFAI.load(poni_file)
            elif (poni_file is not None) and (ai is not None):
                print("\nAzimuthal integrator (ai) is provided. Ignoring poni_file\n")

            if integrate2d:
                ai.empty = np.nan
                i2d = ai.integrate2d(
                    data=img_array,
                    npt_rad=npt,
                    npt_azim=npt_azimuthal,
                    filename=None,
                    correctSolidAngle=True,
                    variance=None,
                    error_model=None,
                    radial_range=radial_range,
                    azimuth_range=None,
                    mask=mask,
                    dummy=np.nan,
                    delta_dummy=None,
                    polarization_factor=None,
                    dark=None,
                    flat=None,
                    method=method,
                    unit="q_A^-1",
                    safe=True,
                    normalization_factor=1.0,
                    metadata=None,
                )

                if (median_filter_kernel_size is not None) and (median_filter_on_i2d):
                    data_i2d = medfilt2d(
                        i2d.intensity.astype("float32"),
                        kernel_size=median_filter_kernel_size,
                    )
                else:
                    data_i2d = i2d.intensity.astype("float32")

                self.ds["i2d"] = xr.DataArray(
                    data=data_i2d,
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
                        "2dintegration_method_split": method[0],
                        "2dintegration_method_algorithm": method[1],
                        "2dintegration_method_implementation": method[2],
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

            else:
                ai.empty = np.nan
                i1d = ai.integrate1d(
                    data=img_array,
                    npt=npt,
                    filename=None,
                    correctSolidAngle=True,
                    variance=None,
                    error_model=None,
                    radial_range=radial_range,
                    azimuth_range=None,
                    mask=mask,
                    dummy=np.nan,
                    delta_dummy=None,
                    polarization_factor=None,
                    dark=None,
                    flat=None,
                    method=method,
                    unit="q_A^-1",
                    safe=True,
                    normalization_factor=1.0,
                    metadata=None,
                )

                da_i1d = xr.DataArray(
                    data=i1d.intensity.astype("float32"),
                    coords=[i1d.radial.astype("float32")],
                    dims=["radial"],
                    attrs={
                        "radial_unit": "q_A^-1",
                        "xlabel": r"Scattering vector $q$ ($\AA^{-1}$)",
                        "ylabel": r"Intensity (a.u.)",
                        "wavelength_in_angst": ai.__dict__["_wavelength"] * 10e9,
                        "detector_name": ai.__dict__["detector"].name,
                        "wavelength_in_meter": ai.__dict__["_wavelength"],
                        "detector_dist": ai.__dict__["_dist"],
                        "detector_poni1": ai.__dict__["_poni1"],
                        "detector_poni2": ai.__dict__["_poni2"],
                        "detector_rot1": ai.__dict__["_rot1"],
                        "detector_rot2": ai.__dict__["_rot2"],
                        "detector_rot3": ai.__dict__["_rot3"],
                        "2dintegration_method_split": method[0],
                        "2dintegration_method_algorithm": method[1],
                        "2dintegration_method_implementation": method[2],
                    },
                )
                self.ds["i1d"] = da_i1d.dropna(dim="radial")

        elif ((img_array is None) and (from_txt_file is None)) and (
            from_i1d_array is not None
        ):

            self.ds = xr.Dataset()

            X, Y = from_i1d_array[:, 0], from_i1d_array[:, 1]
            if i1d_array_radial_unit.lower()[0] == "t":
                X = ((4 * np.pi) / (txt_file_wavelength_in_angstrom)) * np.sin(
                    np.deg2rad(X) / 2
                )
            elif i1d_array_radial_unit.lower()[0] == "q":
                pass
            else:
                print("Unable to determine radial unit. Check the radial_unit\n\n")
                return

            da_i1d = xr.DataArray(
                data=Y.astype("float32"),
                coords=[X.astype("float32")],
                dims=["radial"],
                attrs={
                    "radial_unit": "q_A^-1",
                    "xlabel": r"Scattering vector $q$ ($\AA^{-1}$)",
                    "ylabel": r"Intensity (a.u.)",
                    "wavelength_in_angst": i1d_array_wavelength_in_angstrom,
                },
            )
            self.ds["i1d"] = da_i1d.dropna(dim="radial")

        elif ((img_array is None) and (from_txt_file is None)) and (
            from_da_i1d is not None
        ):

            self.ds = xr.Dataset()
            self.ds["i1d"] = from_da_i1d.dropna(dim="radial")

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
                        X = ((4 * np.pi) / (txt_file_wavelength_in_angstrom)) * np.sin(
                            np.deg2rad(X) / 2
                        )
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
                            "xlabel": r"Scattering vector $q$ ($\AA^{-1}$)",
                            "ylabel": "Intensity (a.u.)",
                            "wavelength_in_angst": txt_file_wavelength_in_angstrom,
                            "i1d_from": from_txt_file,
                        },
                    )

                    if radial_range is not None:
                        self.ds = self.ds.sel(
                            radial=slice(radial_range[0], radial_range[1])
                        )
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

        elif (
            ((from_img_array is None) and (from_txt_file is None))
            and (from_nc_file is None)
            and (from_da_i2d is not None)
        ):

            self.ds = xr.Dataset()

            self.ds["i2d"] = from_da_i2d

            da_i1d = xr.DataArray(
                data=self.ds["i2d"].mean(dim="azimuthal_i2d").astype("float32"),
                coords=[self.ds["i2d"].radial_i2d],
                dims=["radial"],
                attrs=from_da_i2d.attrs,
            )
            self.ds["i1d"] = da_i1d.dropna(dim="radial")

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

    def export_ds_to(self, to=None, save_dir=None, save_name=None):
        """Export xarray dataset to NetCDF format."""

        if to is None:
            try:
                self.ds.to_netcdf(
                    "%s/%s" % (save_dir, save_name),
                    engine="h5netcdf",
                    encoding={"i2d": {"zlib": True, "complevel": 9}},
                )
            except:
                self.ds.to_netcdf(
                    "%s/%s" % (save_dir, save_name),
                    engine="h5netcdf",
                )
        else:
            try:
                self.ds.to_netcdf(
                    "%s" % (to),
                    engine="h5netcdf",
                    encoding={"i2d": {"zlib": True, "complevel": 9}},
                )
            except:
                self.ds.to_netcdf(
                    "%s" % (to),
                    engine="h5netcdf",
                )

    def export_i1d_to(
        self, to="data.dat", mode="xy", subtract_baseline=False, fmt="%.4e %.4e", header=False,
    ):
        """Export integrated 1D XRD data to file."""

        if subtract_baseline and ("i1d_baseline" in self.ds.keys()):
            data_y = self.ds.i1d.values - self.ds.i1d_baseline.values
        elif subtract_baseline and ("i1d_baseline" not in self.ds.keys()):
            data_y = self.ds.i1d.values
            print("\n....baseline is not subtracted as it is not available!")
        else:
            data_y = self.ds.i1d.values

        if mode == "qxy":
            data_x = self.ds.i1d.radial.values
            header_str = "q(Angst.^-1) Intensity(a.u.)"
        elif mode == "d":
            data_x = (2 * np.pi) / self.ds.i1d.radial.values
            header_str = "d Intensity(a.u.)"
        elif mode == "xy":
            data_x = np.rad2deg(
                2
                * np.arcsin(
                    self.ds.i1d.radial
                    * ((self.ds.i1d.attrs["wavelength_in_angst"]) / (4 * np.pi))
                )
            )
            header_str = "TwoTheta(Deg.) Intensity(a.u.)"
        else:
            data_x = np.rad2deg(
                2
                * np.arcsin(
                    self.ds.i1d.radial
                    * ((self.ds.i1d.attrs["wavelength_in_angst"]) / (4 * np.pi))
                )
            )
            header_str = "Q_inv(A-1) Intensity(a.u.)"

        out = np.column_stack((data_x, data_y))

        if header:
            np.savetxt(to, out, fmt=fmt, header=header_str)
        else:
            np.savetxt(to, out, fmt=fmt)
