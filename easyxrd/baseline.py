import os
import numpy as np
import xarray as xr
import pybaselines
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from concurrent.futures import ThreadPoolExecutor
import copy

from .plotters import exrd_plotter

def _optimize_bkg_scale(y_obs, y_bkg, initial_scale=None, max_iter=100):
    """
    Vectorized calculation of background scaling factor to ensure non-negative subtraction.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    y_bkg = np.asarray(y_bkg, dtype=np.float64)
    valid = np.isfinite(y_obs) & np.isfinite(y_bkg)
    if not np.any(valid):
        return 1.0

    # Boolean indexing copies the entire input, so skip it for finite data.
    if np.all(valid):
        y_o = y_obs.ravel()
        y_b = y_bkg.ravel()
    else:
        y_o = y_obs[valid]
        y_b = y_bkg[valid]
    max_b = np.nanmax(y_b)
    if max_b <= 0:
        return 1.0

    if initial_scale is None:
        bkg_scale = float(y_o[0] / max_b)
    else:
        bkg_scale = float(initial_scale)

    # Reuse one work array throughout the search, retaining the same floating
    # point operations and stopping rules as the allocation-based expression.
    diff = np.empty_like(y_o)
    np.multiply(y_b, bkg_scale, out=diff)
    np.subtract(y_o, diff, out=diff)
    min_diff = np.nanmin(diff)
    c = 0
    if min_diff > 0:
        while min_diff > 0 and c < max_iter:
            bkg_scale *= 1.01
            np.multiply(y_b, bkg_scale, out=diff)
            np.subtract(y_o, diff, out=diff)
            min_diff = np.nanmin(diff)
            c += 1
    elif min_diff < 0:
        while min_diff < 0 and c < max_iter:
            bkg_scale *= 0.99
            np.multiply(y_b, bkg_scale, out=diff)
            np.subtract(y_o, diff, out=diff)
            min_diff = np.nanmin(diff)
            c += 1
    return bkg_scale


def _compute_i2d_baseline(da_2d, iarpls_lam=1e5, max_workers=None):
    """
    Parallelized 2D baseline calculation across azimuthal slices.
    """
    radial_vals = da_2d.radial_i2d.values
    data_vals = da_2d.values
    n_azim, n_rad = data_vals.shape
    out_vals = np.empty_like(data_vals)

    def _fit_row(idx):
        y = data_vals[idx]
        valid = np.isfinite(y)
        if np.count_nonzero(valid) < 3:
            out_vals[idx] = y
            return
        x_valid = radial_vals[valid]
        y_valid = y[valid]
        try:
            b, _ = pybaselines.Baseline(x_data=x_valid).iarpls(y_valid, lam=iarpls_lam)
            if np.all(valid):
                out_vals[idx] = b
            else:
                out_vals[idx] = np.interp(radial_vals, x_valid, b, left=np.nan, right=np.nan)
        except Exception:
            out_vals[idx] = y

    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4))

    if n_azim <= 1 or max_workers <= 1:
        for i in range(n_azim):
            _fit_row(i)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_fit_row, range(n_azim)))

    return xr.DataArray(
        data=out_vals,
        coords=da_2d.coords,
        dims=da_2d.dims,
        attrs=da_2d.attrs.copy(),
    )



class BaselineMixin:
    """
    Mixin class providing baseline calculation for 1D and 2D XRD data.
    """
    def get_baseline(
        self,
        input_bkg=None,
        use_iarpls=True,
        radial_rolling=-1,
        iarpls_lam=1e5,
        plot=True,
        get_i2d_baseline=False,
        use_i2d_baseline=False,
        roi_radial_range=None,
        roi_azimuthal_range=None,
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
                            ).copy()
                            da_i2d.values = median_filter(da_i2d.values, size=3)
                            da_i2d_bkg = input_bkg.ds.i2d.sel(
                                azimuthal_i2d=slice(
                                    roi_azimuthal_range[0], roi_azimuthal_range[1]
                                )
                            ).copy()
                            da_i2d_bkg.values = median_filter(da_i2d_bkg.values, size=3)
                            da_i1d = da_i2d.mean(dim="azimuthal_i2d").dropna(
                                dim="radial_i2d"
                            )
                            da_i1d_bkg = da_i2d_bkg.mean(dim="azimuthal_i2d").dropna(
                                dim="radial_i2d"
                            )
                        else:
                            da_i2d = self.ds.i2d.copy()
                            da_i2d.values = median_filter(da_i2d.values, size=3)
                            da_i2d_bkg = input_bkg.ds.i2d.copy()
                            da_i2d_bkg.values = median_filter(da_i2d_bkg.values, size=3)
                            da_i1d = da_i2d.mean(dim="azimuthal_i2d").dropna(
                                dim="radial_i2d"
                            )
                            da_i1d_bkg = da_i2d_bkg.mean(
                                dim="azimuthal_i2d"
                            ).dropna(dim="radial_i2d")

                        if roi_radial_range is not None:
                            y_obs = da_i1d.sel(
                                radial_i2d=slice(
                                    roi_radial_range[0], roi_radial_range[-1]
                                )
                            ).values
                            y_bkg = da_i1d_bkg.sel(
                                radial_i2d=slice(
                                    roi_radial_range[0], roi_radial_range[-1]
                                )
                            ).values
                        else:
                            y_obs = da_i1d.values
                            y_bkg = da_i1d_bkg.values
                        bkg_scale = _optimize_bkg_scale(y_obs, y_bkg)

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
                                da_i2d_diff_baseline = _compute_i2d_baseline(
                                    da_i2d_diff, iarpls_lam=iarpls_lam
                                )
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
                                self.ds["i2d_baseline"] = copy.deepcopy(
                                    bkg_scale
                                    * input_bkg.ds.i2d.sel(
                                        azimuthal_i2d=slice(
                                            roi_azimuthal_range[0],
                                            roi_azimuthal_range[1],
                                        )
                                    )
                                )
                            else:
                                self.ds["i2d_baseline"] = copy.deepcopy(
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
                        da_i2d = (
                            self.ds.i2d.sel(
                                azimuthal_i2d=slice(
                                    roi_azimuthal_range[0], roi_azimuthal_range[1]
                                )
                            )
                            .rename({"radial_i2d": "radial"})
                            .copy()
                        )
                        da_i2d.values = median_filter(da_i2d.values, size=3)
                        da_i1d = da_i2d.mean(dim="azimuthal_i2d").dropna(dim="radial")
                        da_i1d_bkg = input_bkg.ds.i1d
                    else:
                        da_i2d = (
                            self.ds.i2d.rename({"radial_i2d": "radial"}).copy()
                        )
                        da_i2d.values = median_filter(da_i2d.values, size=3)
                        da_i1d = da_i2d.mean(dim="azimuthal_i2d").dropna(dim="radial")
                        da_i1d_bkg = input_bkg.ds.i1d

                    if roi_radial_range is not None:
                        y_obs = da_i1d.sel(
                            radial=slice(roi_radial_range[0], roi_radial_range[-1])
                        ).values
                        y_bkg = da_i1d_bkg.sel(
                            radial=slice(roi_radial_range[0], roi_radial_range[-1])
                        ).values
                    else:
                        y_obs = da_i1d.values
                        y_bkg = da_i1d_bkg.values
                    bkg_scale = _optimize_bkg_scale(y_obs, y_bkg)

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
                        self.ds["i1d_baseline"] = copy.deepcopy(bkg_scale * input_bkg.ds.i1d)
                        self.ds["i1d_baseline"].attrs[
                            "baseline_note"
                        ] = "baseline is from provided input_bkg. i1d iarpls is not used"

                else:

                    da_i1d = self.ds.i1d
                    da_i1d_bkg = input_bkg.ds.i1d

                    if roi_radial_range is not None:
                        y_obs = da_i1d.sel(
                            radial=slice(roi_radial_range[0], roi_radial_range[-1])
                        ).values
                        y_bkg = da_i1d_bkg.sel(
                            radial=slice(roi_radial_range[0], roi_radial_range[-1])
                        ).values
                    else:
                        y_obs = da_i1d.values
                        y_bkg = da_i1d_bkg.values
                    bkg_scale = _optimize_bkg_scale(y_obs, y_bkg)

                    if use_iarpls:

                        da_i1d_diff = self.ds.i1d - bkg_scale * input_bkg.ds.i1d

                        da_for_baseline = da_i1d_diff  # .dropna(dim='radial')
                        # diff_baseline, params = pybaselines.Baseline(
                        #     x_data=da_for_baseline.radial.values
                        # ).iarpls(da_for_baseline.values, lam=iarpls_lam)

                        if radial_rolling < 1:
                            diff_baseline, params = pybaselines.Baseline(
                                x_data=da_for_baseline.radial.values
                            ).iarpls(da_for_baseline.values, lam=iarpls_lam)
                        else:
                            diff_baseline, params = pybaselines.Baseline(
                                x_data=da_for_baseline.radial.values
                            ).iarpls(
                                da_for_baseline.rolling(
                                    radial=radial_rolling, center=True
                                )
                                .mean()
                                .interpolate_na(
                                    dim="radial",
                                    method="nearest",
                                    fill_value="extrapolate",
                                )
                                .values,
                                lam=iarpls_lam,
                            )

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
                        self.ds["i1d_baseline"] = copy.deepcopy(bkg_scale * input_bkg.ds.i1d)
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
                            da_i2d_baseline = _compute_i2d_baseline(
                                da_i2d, iarpls_lam=iarpls_lam
                            )
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

                    if use_iarpls:

                        if radial_rolling < 1:
                            baseline, params = pybaselines.Baseline(
                                x_data=self.ds.i1d.radial.values
                            ).iarpls(self.ds.i1d.values, lam=iarpls_lam)
                        else:
                            baseline, params = pybaselines.Baseline(
                                x_data=self.ds.i1d.radial.values
                            ).iarpls(
                                self.ds.i1d.rolling(radial=radial_rolling, center=True)
                                .mean()
                                .interpolate_na(
                                    dim="radial",
                                    method="nearest",
                                    fill_value="extrapolate",
                                )
                                .values,
                                lam=iarpls_lam,
                            )

                        self.ds["i1d_baseline"] = xr.DataArray(
                            data=(baseline),
                            dims=["radial"],
                            coords={"radial": self.ds.i1d.radial.values},
                            attrs={"iarpls_lam": iarpls_lam},
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
                    "xlabel": r"Scattering vector $q$ ($\AA^{-1}$)",
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
                wl_meter = self.ds["i2d"].attrs.get("wavelength_in_meter", None)
                if wl_meter is not None:
                    wl_angst = wl_meter * 1e10
                else:
                    wl_angst = self.ds["i1d"].attrs.get("wavelength_in_angst", 0.1814)
                self.ds["i1d"].attrs = {
                    "radial_unit": "q_A^-1",
                    "xlabel": r"Scattering vector $q$ ($\AA^{-1}$)",
                    "ylabel": "Intensity (a.u.)",
                    "wavelength_in_angst": wl_angst,
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

