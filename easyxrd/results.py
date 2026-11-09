"""Labeled, serializable results from the most recent accepted GSAS-II fit."""
import numpy as np
import xarray as xr


RESULT_DIMS = ("gsas_parameter", "gsas_parameter_cov", "gsas_statistic")
RESULT_SCALARS = ("gsas_converged",)
RESULT_ATTRS = ("gsas_result_schema", "gsas_covariance_status", "gsas_result_step")


def clear_fit_results(ds):
    """Remove the current parameter result without removing the attempt history."""
    result = ds.drop_dims(RESULT_DIMS, errors="ignore").drop_vars(RESULT_SCALARS, errors="ignore")
    result.attrs = {k: v for k, v in result.attrs.items() if k not in RESULT_ATTRS}
    return result


def fit_result_dataset(covariance):
    """Copy native GSAS-II variables, s.u. and covariance in varyList order.

    These are the directly varied parameters, possibly in constrained or
    reciprocal-cell coordinates, not derived lattice constants or fractions.
    Values retain GSAS-II units and scaling. Missing covariance stays missing;
    no diagonal approximation or extra GoF scaling is applied.
    """
    names = np.asarray(covariance.get("varyList", []), dtype=str)
    if names.ndim != 1 or len(set(names)) != names.size:
        raise ValueError("GSAS-II varyList must contain unique parameter names")
    n = names.size

    def vector(key):
        value = covariance.get(key)
        if value is None or (n == 0 and np.asarray(value).size <= 1):
            return np.full(n, np.nan)
        array = np.asarray(value, dtype=float)
        if array.shape != (n,):
            raise ValueError(f"GSAS-II {key} does not match varyList")
        return array.copy()

    values, su = vector("variables"), vector("sig")
    if np.any(su < 0):
        raise ValueError("GSAS-II standard uncertainties must be nonnegative")
    raw = covariance.get("covMatrix")
    if raw is None or np.asarray(raw).size == 0:
        matrix = np.full((n, n), np.nan)
        status = "unavailable" if n else "no varied parameters"
    else:
        matrix = np.asarray(raw, dtype=float).copy()
        if matrix.shape != (n, n):
            raise ValueError("GSAS-II covariance dimensions do not match varyList")
        status = "available" if np.all(np.isfinite(matrix)) else "nonfinite"
    correlation = np.full_like(matrix, np.nan)
    diagonal = np.diag(matrix)
    valid = np.isfinite(diagonal) & (diagonal > 0)
    scale = np.sqrt(np.where(valid, diagonal, np.nan))
    np.divide(matrix, scale[:, None] * scale[None, :], out=correlation,
              where=valid[:, None] & valid[None, :])
    rvals = covariance.get("Rvals", {})
    statistic_names = ("Rwp", "GOF", "Nobs", "Nvars", "chisq", "SVD0", "Max shft/sig")
    statistics = [np.nan if rvals.get(k) is None else float(rvals[k]) for k in statistic_names]
    converged = rvals.get("converged")
    result = xr.Dataset(
        {
            "gsas_parameter_value": ("gsas_parameter", values),
            "gsas_parameter_su": ("gsas_parameter", su),
            "gsas_covariance": (("gsas_parameter", "gsas_parameter_cov"), matrix),
            "gsas_correlation": (("gsas_parameter", "gsas_parameter_cov"), correlation),
            "gsas_fit_statistic": ("gsas_statistic", statistics),
            "gsas_converged": np.int8(-1 if converged is None else bool(converged)),
        },
        coords={"gsas_parameter": names, "gsas_parameter_cov": names.copy(),
                "gsas_statistic": list(statistic_names)},
        attrs={"gsas_result_schema": "1", "gsas_covariance_status": status},
    )
    result.gsas_parameter_value.attrs["convention"] = "Native GSAS-II directly varied parameters and units"
    result.gsas_parameter_su.attrs["long_name"] = "GSAS-II reported standard uncertainty"
    result.gsas_covariance.attrs["convention"] = "Native GSAS-II covariance; no additional GoF scaling"
    result.gsas_correlation.attrs["units"] = "1"
    result.gsas_converged.attrs["codes"] = "-1 unknown; 0 not converged; 1 converged"
    return result
