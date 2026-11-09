from .results import clear_fit_results
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xarray as xr
import pybaselines
from .data_loader import cake_profile, validate_profile


def _optimize_bkg_scale(y_obs, y_bkg, initial_scale=None, max_iter=100):
    """Largest nonnegative scale satisfying I-s*B >= 0 on finite bins.

    initial_scale/max_iter remain accepted for compatibility. For signed noise
    no feasible nonnegative scale may exist; return zero and report that case.
    """
    obs, bkg = np.asarray(y_obs, float), np.asarray(y_bkg, float)
    if obs.shape != bkg.shape:
        raise ValueError("Observed and background arrays must have equal shapes")
    valid = np.isfinite(obs) & np.isfinite(bkg)
    if not valid.any():
        raise ValueError("No finite observations overlap the background")
    obs, bkg = obs[valid], bkg[valid]
    positive, negative = bkg > 0, bkg < 0
    upper = np.min(obs[positive] / bkg[positive]) if positive.any() else np.inf
    lower = max(0., np.max(obs[negative] / bkg[negative])) if negative.any() else 0.
    if np.any((bkg == 0) & (obs < 0)) or upper < lower:
        warnings.warn("No nonnegative background scale makes all residuals nonnegative; using zero", UserWarning, stacklevel=2)
        return 0.
    return float(upper if np.isfinite(upper) else max(1., lower))


def _compute_i2d_baseline(da_2d, iarpls_lam=1e5, max_workers=None):
    """Fit each sector; report failures instead of removing its entire signal."""
    radial = da_2d.radial_i2d.values
    def fit(y):
        valid = np.isfinite(y)
        if valid.sum() < 3:
            warnings.warn("Insufficient finite points for sector baseline", UserWarning)
            return np.full(y.shape, np.nan)
        try:
            b, _ = pybaselines.Baseline(x_data=radial[valid]).iarpls(y[valid], lam=iarpls_lam)
        except Exception as exc:
            raise RuntimeError("Azimuthal-sector baseline fit failed") from exc
        return np.interp(radial, radial[valid], b, left=np.nan, right=np.nan)
    with ThreadPoolExecutor(max_workers=max_workers or  min(16, len(da_2d))) as pool:
        result = np.asarray(list(pool.map(fit, da_2d.values)))
    return da_2d.copy(data=result)


def _same_grid(a, b, rtol=1e-5, atol=1e-7):
    if a.dims != b.dims or a.shape != b.shape or any(
        not np.allclose(a.coords[d].values, b.coords[d].values, rtol=rtol, atol=atol) for d in a.dims
    ):
        raise ValueError("Sample and background grids must match exactly; explicitly interpolate the background first")
    if a.attrs.get("radial_unit", "q_A^-1") != b.attrs.get("radial_unit", "q_A^-1"):
        raise ValueError("Sample and background radial units differ")


class BaselineMixin:
    def get_baseline(self, input_bkg=None, use_iarpls=True, radial_rolling=-1,
                     iarpls_lam=1e5, plot=True, get_i2d_baseline=False,
                     use_i2d_baseline=False, roi_radial_range=None,
                     roi_azimuthal_range=None, spotty_data_correction=False,
                     spotty_data_correction_threshold=1):
        """Estimate a baseline without mutating raw images or losing coordinates.

        Backgrounds must be explicitly interpolated onto the sample grid before
        calling. Independent measured-background variance is propagated with the
        fitted scale treated as fixed. IarPLS and fitted-scale uncertainty are
        not estimated. Spot selection is a separate diagnostic product, never
        the quantitative observation supplied to refinement.
        """
        if input_bkg is None and not use_iarpls:
            return
        if use_i2d_baseline and not get_i2d_baseline:
            raise ValueError("use_i2d_baseline requires get_i2d_baseline=True")
        if (get_i2d_baseline or roi_azimuthal_range is not None or spotty_data_correction) and "i2d" not in self.ds:
            raise ValueError("The requested operation requires i2d")
        if spotty_data_correction and not get_i2d_baseline:
            raise ValueError("spotty_data_correction requires get_i2d_baseline=True")
        # Work on independent state; every failure leaves the accepted data intact.
        ds = self.ds.copy(deep=True)
        obs = ds.i1d
        if roi_azimuthal_range is not None:
            obs, var = cake_profile(ds, roi_azimuthal_range)
            if "radial" in ds.dims:
                ds = ds.drop_dims("radial")
            ds["i1d"] = obs
            if var is not None:
                ds["i1d_variance"] = var
            ds.i1d.attrs["roi_azimuthal_range"] = roi_azimuthal_range
            ds.i2d.attrs["roi_azimuthal_range"] = roi_azimuthal_range
        if roi_radial_range is not None:
            ds = ds.sel(radial=slice(*roi_radial_range))
        obs = ds.i1d
        validate_profile(obs)
        bkg = xr.zeros_like(obs)
        bkg2d = xr.zeros_like(ds.i2d) if "i2d" in ds else None
        scale = 0.
        external_var = None
        if input_bkg is not None:
            other = input_bkg.ds
            if "i2d" in ds and "i2d" in other:
                _same_grid(ds.i2d, other.i2d)
                bkg2d = other.i2d.assign_coords({d: ds.i2d.coords[d] for d in ds.i2d.dims})
            if roi_azimuthal_range is not None and "i2d" in other:
                bkg, external_var = cake_profile(other, roi_azimuthal_range)
            else:
                bkg = other.i1d
                external_var = other.get("i1d_variance")
            if roi_radial_range is not None:
                bkg = bkg.sel(radial=slice(*roi_radial_range))
                if external_var is not None:
                    external_var = external_var.sel(radial=slice(*roi_radial_range))
            _same_grid(obs, bkg)
            bkg = bkg.assign_coords(radial=obs.radial)
            if external_var is not None:
                external_var = external_var.assign_coords(radial=obs.radial)
            if get_i2d_baseline and "i2d" not in other:
                full = other.i1d.rename({"radial": "radial_i2d"})
                if not np.allclose(full.radial_i2d.values, ds.radial_i2d.values, rtol=1e-5, atol=1e-7):
                    raise ValueError("Background radial grid must match the entire cake")
                bkg2d = full.assign_coords(radial_i2d=ds.radial_i2d).broadcast_like(ds.i2d)
            scale = _optimize_bkg_scale(obs.values, bkg.values)
        baseline = scale * bkg
        if use_iarpls:
            fit_data = obs - baseline
            if radial_rolling >= 1:
                fit_data = fit_data.rolling(radial=radial_rolling, center=True, min_periods=1).mean()
            values, _ = pybaselines.Baseline(x_data=obs.radial.values).iarpls(fit_data.values, lam=iarpls_lam)
            baseline = baseline + values
        if get_i2d_baseline:
            cake = ds.i2d
            if roi_azimuthal_range is not None:
                cake = cake.sel(azimuthal_i2d=slice(*roi_azimuthal_range))
                bkg2d = bkg2d.sel(azimuthal_i2d=cake.azimuthal_i2d)
            cake_baseline = scale * bkg2d
            if use_iarpls:
                cake_baseline = cake_baseline + _compute_i2d_baseline(cake - cake_baseline, iarpls_lam)
            ds["i2d_baseline"] = cake_baseline
            if use_i2d_baseline:
                temp = ds.copy(deep=False)
                temp["i2d"] = ds.i2d_baseline
                baseline, _ = cake_profile(temp, roi_azimuthal_range)
                baseline = baseline.sel(radial=obs.radial)
        ds["i1d_baseline"] = baseline
        ds.i1d_baseline.attrs = dict(baseline_note="measured background and/or iarpls", background_scale=scale,
                                     iarpls_lam=iarpls_lam, uncertainty_note="Fitted scale and empirical baseline uncertainty not included")
        if "i1d_baseline_variance" in ds:
            del ds["i1d_baseline_variance"]
        if external_var is not None:
            ds["i1d_baseline_variance"] = scale**2 * external_var
        if not np.all(np.isfinite(ds.i1d_baseline)):
            raise ValueError("Baseline has unsupported or nonfinite bins; observations were preserved")
        if spotty_data_correction:
            selected = (ds.i2d - ds.i2d_baseline) >= spotty_data_correction_threshold
            ds["i2d_spot_mask"] = selected.astype("int8")
            temp = ds.copy(deep=False)
            temp["i2d"] = ds.i2d.where(selected)
            diagnostic, _ = cake_profile(temp, roi_azimuthal_range)
            ds["i1d_spot_corrected"] = diagnostic
            warnings.warn("Spot-selected intensities are stored separately in i1d_spot_corrected; raw observations are unchanged", UserWarning)
        for key in ("i1d_refined", "i1d_gsas_background"):
            if key in ds:
                del ds[key]
        self.ds = clear_fit_results(ds)
        # A changed baseline invalidates the engine's observations.
        if hasattr(self, "gpx"):
            del self.gpx
        if plot:
            self.plot(plot_hint="get_baseline")
