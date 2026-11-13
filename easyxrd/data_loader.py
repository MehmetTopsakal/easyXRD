from . import DEFAULT, resolve_default

import os
import warnings
import logging
import numpy as np
import xarray as xr
import fabio
import pyFAI
from scipy.signal import medfilt2d

from .plotters import _normalize_radial_unit
from .provenance import file_record, software_versions, provenance_json

logger = logging.getLogger(__name__)


def _netcdf_attr_value(value):
    """Keep NetCDF-compatible metadata and represent other values as text."""
    # h5netcdf rejects booleans even though Python considers them numbers.
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (str, bytes, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, (np.ndarray, list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return str(value)
        if array.ndim <= 1 and array.dtype.kind in "biufSU":
            if array.dtype.kind == "b":
                return array.astype("int8")
            if array.dtype.kind == "U":
                # h5py accepts Python strings, but not NumPy Unicode arrays.
                return array.tolist()
            return value
    # Includes GSAS-II dictionaries (e.g. lastShifts), None, nested lists,
    # and object arrays, which cannot be written as NetCDF attributes.
    return str(value)


def q_to_two_theta(q, wavelength):
    """Convert inverse-Angstrom q to degrees, rejecting impossible coordinates."""
    q = np.asarray(q, dtype=float)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive in Angstrom")
    argument = q * wavelength / (4 * np.pi)
    if not np.all(np.isfinite(argument)) or np.any(argument < 0) or np.any(argument > 1.0 + 1e-7):
        raise ValueError("q must be finite and satisfy 0 <= q*wavelength/(4*pi) <= 1")
    return np.rad2deg(2 * np.arcsin(np.clip(argument, 0.0, 1.0)))


def _radial_range_to_q(radial_range, radial_unit, wavelength):
    """Translate loader bounds from the object's selected unit to analysis q."""
    if radial_range is None:
        return None
    bounds = np.asarray(radial_range, dtype=float)
    if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
            or bounds[0] < 0 or bounds[1] <= bounds[0]):
        raise ValueError("radial_range must be an increasing nonnegative pair")
    if radial_unit == "two-theta":
        if bounds[1] > 180:
            raise ValueError("two-theta radial_range must be between 0 and 180 degrees")
        bounds = 4 * np.pi / wavelength * np.sin(np.deg2rad(bounds) / 2)
    return tuple(bounds)


def validate_profile(da):
    if da.dims != ("radial",) or not da.size:
        raise ValueError("i1d must be a nonempty one-dimensional radial profile")
    q = np.asarray(da.radial, dtype=float)
    if not np.all(np.isfinite(q)) or np.any(q < 0) or np.any(np.diff(q) <= 0):
        raise ValueError("radial coordinates must be finite, nonnegative and strictly increasing")
    if not np.all(np.isfinite(da.values)):
        raise ValueError("i1d intensities must be finite")


def cake_profile(ds, azimuthal_range=None):
    """Reduce a cake with its normalization weights when available.

    Summed variances omit inter-bin covariance introduced by pixel splitting.
    Imported cakes without accumulators explicitly use equal-sector averaging.
    """
    cake = ds.i2d
    if azimuthal_range is not None:
        cake = cake.sel(azimuthal_i2d=slice(*azimuthal_range))
    if not cake.sizes["azimuthal_i2d"]:
        raise ValueError("azimuthal range contains no bins")
    if "i2d_normalization" in ds:
        norm = ds.i2d_normalization.sel(azimuthal_i2d=cake.azimuthal_i2d)
        norm = norm.where(np.isfinite(cake) & (norm > 0), 0)
        denominator = norm.sum("azimuthal_i2d")
        profile = (cake.fillna(0) * norm).sum("azimuthal_i2d") / denominator.where(denominator > 0)
        variance = None
        if "i2d_variance" in ds:
            v = ds.i2d_variance.sel(azimuthal_i2d=cake.azimuthal_i2d)
            variance = (v.fillna(0) * norm**2).sum("azimuthal_i2d") / denominator.where(denominator > 0)**2
    else:
        warnings.warn("Cake has no normalization accumulators; using equal-sector averaging", UserWarning, stacklevel=2)
        profile = cake.mean("azimuthal_i2d")
        variance = None
    attrs = dict(cake.attrs)
    if "wavelength_in_meter" in attrs:
        attrs["wavelength_in_angst"] = attrs["wavelength_in_meter"] * 1e10
    attrs.setdefault("xlabel", r"Scattering vector $q$ ($\AA^{-1}$)")
    attrs["radial_unit"] = "q_A^-1"
    profile = profile.rename({"radial_i2d": "radial"})
    profile.attrs = attrs
    if variance is not None:
        variance = variance.rename({"radial_i2d": "radial"})
    return profile, variance


class DataLoaderMixin:
    """Load XRD observations and preserve their measurement variance."""

    def load_xrd_data(
        self, integrate2d=DEFAULT, from_img_array=None, from_tiff_file=None,
        ai=None, poni_file=None, mask=None, mask_file=None,
        median_filter_kernel_size=DEFAULT, median_filter_on_i2d=DEFAULT,
        from_da_i2d=None, from_i1d_array=None,
        i1d_array_wavelength_in_angstrom=DEFAULT, i1d_array_radial_unit=DEFAULT,
        from_da_i1d=None, from_nc_file=None, from_txt_file=None,
        txt_file_wavelength_in_angstrom=DEFAULT, txt_file_comments=DEFAULT,
        txt_file_skiprows=DEFAULT, txt_file_usecols=DEFAULT, txt_file_radial_unit=DEFAULT,
        radial_range=DEFAULT, radial_npts=DEFAULT, delta_q=DEFAULT,
        npt_azimuthal=DEFAULT, plot=DEFAULT, ds_attrs=None,
        method=DEFAULT, variance=None, error_model=DEFAULT,
        i1d_variance=None, polarization_factor=DEFAULT, dark=None, flat=None,
        normalization_factor=DEFAULT,
    ):
        """Load exactly one source, committing state only after validation.

        Arrays may contain q/2theta, intensity, and optional standard deviation.
        Alternatively supply i1d_variance. Detector variance/error_model is passed
        to pyFAI; no Poisson assumption is made for arbitrary detector values.
        The quantitative i1d always uses integrate1d, even when a cake is saved.
        Median filtering produces a separate diagnostic image, not fit data.
        radial_range uses the object's radial_unit: q in inverse Angstrom,
        or two-theta in degrees (0 to 180), for detector and text sources.
        Configured default bounds use the same unit; None disables the bounds.
        delta_q sets image bin spacing in q even for two-theta bounds;
        set it to None to use radial_npts. Two-theta bounds are kept exact,
        with q bin spacing at most delta_q.
        Disabled polarization is recorded as the string "None" in NetCDF-safe
        metadata; pyFAI still receives None to disable the correction.
        """
        median_filter_kernel_size = resolve_default(median_filter_kernel_size, "load_xrd_data_median_filter_kernel_size")
        error_model = resolve_default(error_model, "load_xrd_data_error_model")
        polarization_factor = resolve_default(polarization_factor, "load_xrd_data_polarization_factor")
        integrate2d = resolve_default(integrate2d, "load_xrd_data_integrate2d")
        median_filter_on_i2d = resolve_default(median_filter_on_i2d, "load_xrd_data_median_filter_on_i2d")
        i1d_array_wavelength_in_angstrom = resolve_default(i1d_array_wavelength_in_angstrom, "load_xrd_data_i1d_array_wavelength_in_angstrom")
        i1d_array_radial_unit = resolve_default(i1d_array_radial_unit, "load_xrd_data_i1d_array_radial_unit")
        txt_file_wavelength_in_angstrom = resolve_default(txt_file_wavelength_in_angstrom, "load_xrd_data_txt_file_wavelength_in_angstrom")
        txt_file_comments = resolve_default(txt_file_comments, "load_xrd_data_txt_file_comments")
        txt_file_skiprows = resolve_default(txt_file_skiprows, "load_xrd_data_txt_file_skiprows")
        txt_file_usecols = resolve_default(txt_file_usecols, "load_xrd_data_txt_file_usecols")
        txt_file_radial_unit = resolve_default(txt_file_radial_unit, "load_xrd_data_txt_file_radial_unit")
        radial_range = resolve_default(radial_range, "load_xrd_data_radial_range")
        radial_unit = _normalize_radial_unit(getattr(self, "radial_unit", "q"))
        radial_npts = resolve_default(radial_npts, "load_xrd_data_radial_npts")
        delta_q = resolve_default(delta_q, "load_xrd_data_delta_q")
        npt_azimuthal = resolve_default(npt_azimuthal, "load_xrd_data_npt_azimuthal")
        plot = resolve_default(plot, "load_xrd_data_plot")
        method = resolve_default(method, "load_xrd_data_method")
        normalization_factor = resolve_default(normalization_factor, "load_xrd_data_normalization_factor")

        sources = [from_img_array, from_tiff_file, from_da_i2d, from_i1d_array,
                   from_da_i1d, from_nc_file, from_txt_file]
        if sum(source is not None for source in sources) != 1:
            raise ValueError("Provide exactly one XRD data source")
        ds = xr.Dataset(attrs=dict(ds_attrs or {}))
        if from_nc_file is not None:
            ds = xr.load_dataset(from_nc_file)
        elif from_img_array is not None or from_tiff_file is not None:
            geometry_from_file = ai is None
            mask_from_file = mask is None and mask_file is not None
            if ai is None:
                if poni_file is None:
                    raise ValueError("Detector data require ai or poni_file")
                ai = pyFAI.load(poni_file)
            if ai.wavelength is None or not np.isfinite(ai.wavelength) or ai.wavelength <= 0:
                raise ValueError("The integrator must have a positive wavelength")
            if from_tiff_file is not None:
                with fabio.open(str(from_tiff_file)) as frame:
                    img = frame.data.copy()
            else:
                img = np.asarray(from_img_array)
            if img.ndim != 2:
                raise ValueError("Detector image must be two-dimensional")
            if mask is None and mask_file is not None:
                with fabio.open(str(mask_file)) as frame:
                    mask = frame.data.copy()
            radial_range = _radial_range_to_q(radial_range, radial_unit, ai.wavelength * 1e10)
            npt = radial_npts
            if radial_range is not None and delta_q is not None:
                if not np.isfinite(delta_q) or delta_q <= 0:
                    raise ValueError("delta_q must be finite and positive")
                npt = int(np.ceil((radial_range[1] - radial_range[0]) / delta_q))
                if radial_unit == "q":
                    radial_range = (radial_range[0], radial_range[0] + delta_q * npt)
            if not isinstance(npt, (int, np.integer)) or npt <= 0:
                raise ValueError("radial_npts must be a positive integer")
            common = dict(data=img, correctSolidAngle=True, variance=variance,
                          error_model=error_model, radial_range=radial_range, mask=mask,
                          dummy=np.nan, polarization_factor=polarization_factor,
                          dark=dark, flat=flat, method=method, unit="q_A^-1",
                          normalization_factor=normalization_factor)
            ds.attrs["integration_provenance"] = provenance_json({
                "schema": 1, "software": software_versions("easyxrd", "pyFAI"),
                "geometry": ai.get_config(),
                "poni_file": file_record(poni_file) if geometry_from_file else None,
                "image_source": str(from_tiff_file) if from_tiff_file is not None else "array",
                "mask_file": file_record(mask_file) if mask_from_file else None,
                "mask_supplied": mask is not None,
                "dark_supplied": dark is not None, "flat_supplied": flat is not None,
                "variance_supplied": variance is not None, "error_model": str(error_model),
                "method_requested": str(method), "correct_solid_angle": True,
                "polarization_factor": polarization_factor,
                "normalization_factor": normalization_factor,
                "radial_range": radial_range, "radial_npts": npt,
                "radial_range_unit": "q_A^-1",
                "npt_azimuthal": npt_azimuthal if integrate2d else None,
            })
            old_empty = ai.empty
            try:
                ai.empty = np.nan
                result = ai.integrate1d(npt=npt, **common)
                cake = ai.integrate2d(npt_rad=npt, npt_azim=npt_azimuthal, **common) if integrate2d else None
            finally:
                ai.empty = old_empty
            attrs = dict(radial_unit="q_A^-1", xlabel=r"Scattering vector $q$ ($\AA^{-1}$)",
                         wavelength_in_angst=ai.wavelength * 1e10,
                         wavelength_in_meter=ai.wavelength, detector_name=ai.detector.name,
                         detector_dist=ai.dist, detector_poni1=ai.poni1, detector_poni2=ai.poni2,
                         detector_rot1=ai.rot1, detector_rot2=ai.rot2, detector_rot3=ai.rot3,
                         integration_method=str(method), error_model=str(error_model),
                         polarization_factor=_netcdf_attr_value(polarization_factor),
                         normalization_factor=normalization_factor)
            ds["i1d"] = xr.DataArray(result.intensity, dims="radial", coords={"radial": result.radial}, attrs=attrs)
            if result.sigma is not None:
                ds["i1d_variance"] = ("radial", np.asarray(result.sigma)**2)
            subset = ["i1d", "i1d_variance"] if "i1d_variance" in ds else ["i1d"]
            ds = ds.dropna("radial", subset=subset)
            if cake is not None:
                dims = ("azimuthal_i2d", "radial_i2d")
                coords = dict(azimuthal_i2d=cake.azimuthal, radial_i2d=cake.radial)
                ds["i2d"] = xr.DataArray(cake.intensity, dims=dims, coords=coords, attrs=attrs)
                ds["i2d_normalization"] = (dims, cake.sum_normalization)
                if cake.sigma is not None:
                    ds["i2d_variance"] = (dims, np.asarray(cake.sigma)**2)
            if median_filter_kernel_size is not None:
                warnings.warn("Median filtering is diagnostic only; quantitative observations are preserved", UserWarning)
                if median_filter_on_i2d and cake is not None:
                    ds["i2d_filtered"] = ds.i2d.copy(data=medfilt2d(np.asarray(cake.intensity, dtype=float), median_filter_kernel_size))
                else:
                    ds["image_filtered"] = (("pixel_y", "pixel_x"), medfilt2d(img.astype(float), median_filter_kernel_size))
        elif from_da_i2d is not None:
            ds["i2d"] = from_da_i2d.copy(deep=True)
            ds["i1d"], _ = cake_profile(ds)
            ds = ds.dropna("radial", subset=["i1d"])
        elif from_da_i1d is not None:
            ds["i1d"] = from_da_i1d.copy(deep=True).dropna("radial")
        else:
            is_text = from_txt_file is not None
            data = (np.loadtxt(from_txt_file, comments=txt_file_comments, skiprows=txt_file_skiprows,
                               usecols=txt_file_usecols, ndmin=2) if is_text else np.asarray(from_i1d_array))
            if data.ndim != 2 or data.shape[1] not in (2, 3):
                raise ValueError("1D arrays require two columns (x, I), or three (x, I, sigma)")
            x, y = np.asarray(data[:, 0], float), np.asarray(data[:, 1], float)
            wavelength = txt_file_wavelength_in_angstrom if is_text else i1d_array_wavelength_in_angstrom
            unit = (txt_file_radial_unit if is_text else i1d_array_radial_unit).lower()
            if not np.isfinite(wavelength) or wavelength <= 0:
                raise ValueError("wavelength must be finite and positive")
            if unit in ("tth", "2theta", "two_theta", "2th_deg"):
                if not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 180):
                    raise ValueError("2theta must be finite and between 0 and 180 degrees")
                x = 4 * np.pi / wavelength * np.sin(np.deg2rad(x) / 2)
            elif unit not in ("q", "q_a^-1"):
                raise ValueError("radial unit must be q (inverse Angstrom) or tth (degrees)")
            ds["i1d"] = xr.DataArray(y, dims="radial", coords={"radial": x},
                attrs=dict(radial_unit="q_A^-1", wavelength_in_angst=wavelength,
                           xlabel=r"Scattering vector $q$ ($\AA^{-1}$)", ylabel="Intensity (a.u.)"))
            if data.shape[1] == 3:
                if i1d_variance is not None:
                    raise ValueError("Supply sigma column or i1d_variance, not both")
                sigma = np.asarray(data[:, 2], float)
                if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
                    raise ValueError("sigma must be finite and positive")
                ds["i1d_variance"] = ("radial", sigma**2)
            if is_text:
                ds.i1d.attrs["i1d_from"] = str(from_txt_file)
                if radial_range is not None:
                    radial_range = _radial_range_to_q(radial_range, radial_unit, wavelength)
                    ds = ds.sel(radial=slice(*radial_range))
        if i1d_variance is not None:
            values = np.asarray(i1d_variance, float)
            if values.shape != ds.i1d.shape:
                raise ValueError("i1d_variance must match the loaded profile")
            ds["i1d_variance"] = ("radial", values)
        validate_profile(ds.i1d)
        if "i1d_variance" in ds and (not np.all(np.isfinite(ds.i1d_variance)) or np.any(ds.i1d_variance <= 0)):
            raise ValueError("i1d_variance must be finite and positive")
        same_profile = (from_nc_file is not None and hasattr(self, "ds")
                        and self.ds.identical(ds))
        self.ds = ds
        for name in ("gpx", "gpx_previous", "ds_previous", "phases_previous", "phases"):
            if not same_profile and hasattr(self, name):
                delattr(self, name)
        if plot:
            self.plot(plot_hint="load_xrd_data")

    def export_ds_to(self, to=None, save_dir=None, save_name=None):
        """Export the dataset to NetCDF without changing in-memory metadata.

        Unsupported dataset, variable, and coordinate attributes (including
        GSAS-II dictionaries and boolean scalars) are stored as text. Boolean
        attribute arrays are stored as integers. Numeric and string metadata
        retain their native representation. The i2d variable is compressed
        when present.
        """
        ds_out = self.ds.copy(deep=False)
        ds_out.attrs = {k: _netcdf_attr_value(v) for k, v in self.ds.attrs.items()}
        for name in ds_out.variables:
            ds_out[name].attrs = {
                k: _netcdf_attr_value(v) for k, v in self.ds[name].attrs.items()
            }

        if to is None and (save_dir is None or save_name is None):
            raise ValueError("Provide to, or both save_dir and save_name")
        path = to if to is not None else os.path.join(save_dir, save_name)
        encoding = {"i2d": {"zlib": True, "complevel": 9}} if "i2d" in ds_out else {}
        ds_out.to_netcdf(path, engine="h5netcdf", encoding=encoding)

    def export_i1d_to(
        self, to="data.dat", mode=DEFAULT, subtract_baseline=DEFAULT, fmt=DEFAULT, header=DEFAULT,
    ):
        """Export integrated 1D XRD data to file."""
        mode = resolve_default(mode, "export_i1d_to_mode")
        subtract_baseline = resolve_default(subtract_baseline, "export_i1d_to_subtract_baseline")
        fmt = resolve_default(fmt, "export_i1d_to_fmt")
        header = resolve_default(header, "export_i1d_to_header")

        if subtract_baseline and ("i1d_baseline" in self.ds.keys()):
            data_y = self.ds.i1d.values - self.ds.i1d_baseline.values
        elif subtract_baseline and ("i1d_baseline" not in self.ds.keys()):
            data_y = self.ds.i1d.values
            logger.warning("Baseline was not subtracted because no baseline is available")
        else:
            data_y = self.ds.i1d.values

        if mode == "qxy":
            data_x = self.ds.i1d.radial.values
            header_str = "q(Angst.^-1) Intensity(a.u.)"
        elif mode == "d":
            if np.any(self.ds.i1d.radial.values <= 0):
                raise ValueError("d-spacing export requires strictly positive q")
            data_x = 2 * np.pi / self.ds.i1d.radial.values
            header_str = "d(Angstrom) Intensity(a.u.)"
        elif mode == "xy":
            data_x = q_to_two_theta(self.ds.i1d.radial.values, self.ds.i1d.attrs["wavelength_in_angst"])
            header_str = "TwoTheta(Deg.) Intensity(a.u.)"
        else:
            raise ValueError("mode must be 'xy', 'qxy', or 'd'")
        np.savetxt(to, np.column_stack((data_x, data_y)), fmt=fmt,
                   header=header_str if header else "")
