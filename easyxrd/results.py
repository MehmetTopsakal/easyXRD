"""Labeled, serializable results from the most recent accepted GSAS-II fit."""
import math
import numpy as np
import xarray as xr


RESULT_DIMS = ("gsas_parameter", "gsas_parameter_cov", "gsas_statistic", "phase")
RESULT_SCALARS = ("gsas_converged", "i1d_phase_profiles")
RESULT_ATTRS = ("gsas_result_schema", "gsas_covariance_status", "gsas_result_step")


def clear_fit_results(ds):
    """Remove the current parameter result without removing the attempt history."""
    result = ds.drop_dims(RESULT_DIMS, errors="ignore").drop_vars(RESULT_SCALARS, errors="ignore")
    result.attrs = {k: v for k, v in result.attrs.items() if k not in RESULT_ATTRS}
    return result


def format_value_with_su(value, su=None, precision=4):
    """Format a physical quantity with crystallographic parenthetical uncertainty.

    Follows the IUCr rule (1-2 uncertainty digits in parentheses matching the
    least significant digits of the rounded value). When ``su`` is None, NaN,
    or nonpositive, falls back to fixed-point formatting with ``precision``
    decimal places.
    """
    if value is None or not np.isfinite(value):
        return "nan"
    val = float(value)
    if su is None or not np.isfinite(su) or su <= 0:
        return f"{val:.{precision}f}"
    err = float(su)
    exponent = int(math.floor(math.log10(err)))
    two_digit = int(round(err / (10.0 ** (exponent - 1))))
    if two_digit >= 100:
        exponent += 1
        two_digit = int(round(err / (10.0 ** (exponent - 1))))
    if two_digit < 20:
        decimals = -(exponent - 1)
        su_digits = two_digit
    else:
        decimals = -exponent
        su_digits = int(round(err / (10.0 ** exponent)))
        if su_digits >= 10:
            decimals -= 1
            su_digits = 1
    if decimals > 0:
        rounded = round(val, decimals)
        if rounded == 0.0:
            rounded = 0.0
        return f"{rounded:.{decimals}f}({su_digits})"
    scale = 10.0 ** (-decimals)
    rounded_int = int(round(val / scale) * scale)
    su_int = int(round(err / scale) * scale)
    return f"{rounded_int}({su_int})"


def compute_cell_volume(cell):
    """Compute unit-cell volume (Angstrom^3) from [a, b, c, alpha, beta, gamma]."""
    a, b, c, alpha, beta, gamma = [float(x) for x in cell[:6]]
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    term = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    return float(a * b * c * np.sqrt(max(0.0, term)))


def compute_cell_uncertainties(cell, sg_laue, pfx, covariance):
    """Propagate GSAS-II reciprocal metric tensor covariance to unit-cell s.u.

    Parameters
    ----------
    cell : sequence of 6 floats
        Unit-cell parameters ``[a, b, c, alpha, beta, gamma]`` (Angstrom, deg).
    sg_laue : str
        GSAS-II Laue symmetry symbol (e.g. ``'m3m'``, ``'4/mmm'``, ``'-1'``).
    pfx : str
        GSAS-II phase parameter prefix, e.g. ``'0::'``.
    covariance : dict
        GSAS-II ``project['Covariance']['data']`` dictionary.

    Returns
    -------
    dict or None
        Dictionary mapping ``('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume')``
        to standard uncertainties, or ``None`` if no cell parameters for ``pfx``
        were varied in ``covariance['varyList']``.
    """
    keys = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
    if not isinstance(covariance, dict):
        return None
    vary_list = [str(x) for x in covariance.get("varyList", [])]
    rm_names = [f"{pfx}A{i}" for i in range(6)]
    if not any(name in vary_list for name in rm_names):
        return None
    raw_cov = covariance.get("covMatrix")
    if raw_cov is None or np.asarray(raw_cov).size == 0:
        return {k: np.nan for k in keys}
    cov = np.asarray(raw_cov, dtype=float)
    n = len(vary_list)
    if cov.shape != (n, n):
        return {k: np.nan for k in keys}

    idx_map = {name: vary_list.index(name) for name in rm_names if name in vary_list}
    vcov = np.zeros((6, 6), dtype=float)
    for i, name_i in enumerate(rm_names):
        if name_i in idx_map:
            for j, name_j in enumerate(rm_names):
                if name_j in idx_map:
                    vcov[i, j] = cov[idx_map[name_i], idx_map[name_j]]
    if not np.all(np.isfinite(vcov)):
        return {k: np.nan for k in keys}

    laue = str(sg_laue or "-1").strip()
    if laue in ("3", "3m1", "31m", "6/m", "6/mmm"):
        vcov[1, 1] = vcov[3, 3] = vcov[0, 1] = vcov[1, 0] = vcov[0, 0]
        vcov[1, 3] = vcov[3, 1] = vcov[0, 3] = vcov[3, 0] = vcov[0, 0]
        vcov[1, 2] = vcov[2, 1] = vcov[2, 3] = vcov[3, 2] = vcov[0, 2]
        vcov[2, 0] = vcov[0, 2]
    elif laue in ("m3", "m3m"):
        vcov[0:3, 0:3] = vcov[0, 0]
    elif laue in ("4/m", "4/mmm"):
        vcov[0:2, 0:2] = vcov[0, 0]
        vcov[1, 2] = vcov[2, 1] = vcov[2, 0] = vcov[0, 2]
    elif laue in ("3R", "3mR"):
        vcov[0:3, 0:3] = vcov[0, 0]
        vcov[3:6, 3:6] = vcov[3, 3]
        vcov[0:3, 3:6] = vcov[0, 3]
        vcov[3:6, 0:3] = vcov[3, 0]

    a, b, c, alpha, beta, gamma = [float(x) for x in cell[:6]]
    if not np.all(np.isfinite([a, b, c, alpha, beta, gamma])) or min(a, b, c) <= 0:
        return {k: np.nan for k in keys}
    ca, cb, cg = np.cos(np.deg2rad([alpha, beta, gamma]))
    sa, sb, sg = np.sin(np.deg2rad([alpha, beta, gamma]))
    g = np.array([
        [a * a, a * b * cg, a * c * cb],
        [a * b * cg, b * b, b * c * ca],
        [a * c * cb, b * c * ca, c * c],
    ], dtype=float)
    det_g = float(np.linalg.det(g))
    if not np.isfinite(det_g) or det_g <= 0:
        return {k: np.nan for k in keys}
    vol = math.sqrt(det_g)

    dG_basis = (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
    )
    J = np.zeros((7, 6), dtype=float)
    rad_to_deg = 180.0 / np.pi
    for i, dG in enumerate(dG_basis):
        dg = -g @ dG @ g
        da = dg[0, 0] / (2.0 * a)
        db = dg[1, 1] / (2.0 * b)
        dc = dg[2, 2] / (2.0 * c)
        d_ca = (dg[1, 2] - ca * (b * dc + c * db)) / (b * c)
        d_cb = (dg[0, 2] - cb * (a * dc + c * da)) / (a * c)
        d_cg = (dg[0, 1] - cg * (a * db + b * da)) / (a * b)
        d_alpha = -rad_to_deg * d_ca / sa if abs(sa) > 1e-12 else 0.0
        d_beta = -rad_to_deg * d_cb / sb if abs(sb) > 1e-12 else 0.0
        d_gamma = -rad_to_deg * d_cg / sg if abs(sg) > 1e-12 else 0.0
        d_vol = -0.5 * vol * float(np.trace(g @ dG))
        J[:, i] = (da, db, dc, d_alpha, d_beta, d_gamma, d_vol)

    cov_cell = J @ vcov @ J.T
    variances = np.diag(cov_cell)
    if not np.all(np.isfinite(variances)):
        return {k: np.nan for k in keys}
    su = np.sqrt(np.maximum(0.0, variances))
    if laue in ("3", "3m1", "31m", "6/m", "6/mmm", "m3", "m3m", "4/m", "4/mmm", "mmm"):
        su[3:6] = 0.0
    elif laue == "2/m":
        su[3] = 0.0
        su[5] = 0.0
    return {k: float(su[idx]) for idx, k in enumerate(keys)}


def extract_scale_covariance(scale_names, covariance):
    """Extract the P x P phase-scale covariance submatrix from GSAS-II covariance."""
    p_count = len(scale_names)
    if not isinstance(covariance, dict) or p_count == 0:
        return None, False, False
    vary_list = [str(x) for x in covariance.get("varyList", [])]
    any_varied = any(name in vary_list for name in scale_names)
    if not any_varied:
        return None, False, False
    raw_cov = covariance.get("covMatrix")
    if raw_cov is None or np.asarray(raw_cov).size == 0:
        return None, True, False
    cov = np.asarray(raw_cov, dtype=float)
    n = len(vary_list)
    if cov.shape != (n, n):
        return None, True, False
    sigma_s = np.zeros((p_count, p_count), dtype=float)
    idx_map = {name: vary_list.index(name) for name in scale_names if name in vary_list}
    for i, name_i in enumerate(scale_names):
        if name_i in idx_map:
            for j, name_j in enumerate(scale_names):
                if name_j in idx_map:
                    sigma_s[i, j] = cov[idx_map[name_i], idx_map[name_j]]
    if not np.all(np.isfinite(sigma_s)) or np.any(np.diag(sigma_s) < 0):
        return None, True, False
    return sigma_s, True, True


def compute_weight_fraction_uncertainties(unit_masses, scales, scale_names, covariance):
    """Propagate GSAS-II phase-scale covariance to weight-fraction s.u. (wt%).

    Returns
    -------
    np.ndarray or None
        1D array of standard uncertainties (in wt%, 0-100 scale) for each phase,
        or ``None`` if no phase scale factor was varied in ``covariance['varyList']``.
    """
    unit_masses = np.asarray(unit_masses, dtype=float)
    scales = np.asarray(scales, dtype=float)
    p_count = unit_masses.size
    sigma_s, any_varied, valid_cov = extract_scale_covariance(scale_names, covariance)
    if not any_varied:
        return None
    if not valid_cov:
        return np.full(p_count, np.nan)
    masses = unit_masses * scales
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0 or np.any(masses < 0):
        return np.full(p_count, np.nan)
    if p_count == 1:
        return np.array([0.0], dtype=float)
    # J[j, k] = d(w_j) / d(s_k), where w_j = 100 * m_j * s_j / sum_i(m_i * s_i)
    J = np.zeros((p_count, p_count), dtype=float)
    for j in range(p_count):
        for k in range(p_count):
            delta = 1.0 if j == k else 0.0
            J[j, k] = 100.0 * (
                delta * unit_masses[j] / total_mass
                - (masses[j] * unit_masses[k]) / (total_mass * total_mass)
            )
    cov_w = J @ sigma_s @ J.T
    var_w = np.diag(cov_w)
    if not np.all(np.isfinite(var_w)):
        return np.full(p_count, np.nan)
    return np.sqrt(np.maximum(0.0, var_w))


def compute_internal_standard_qpa(
    unit_masses,
    scales,
    scale_names,
    covariance,
    std_index,
    std_wt_fraction,
    std_wt_fraction_su=0.0,
):
    """Compute crystalline and amorphous weight fractions (wt%) via an internal standard.

    Propagates phase-scale covariance and optional uncertainty on the spiked
    standard weight fraction ``std_wt_fraction`` to both the spiked mixture
    and the original (unspiked) sample.
    """
    unit_masses = np.asarray(unit_masses, dtype=float)
    scales = np.asarray(scales, dtype=float)
    p_count = unit_masses.size
    if std_index < 0 or std_index >= p_count:
        raise IndexError(f"Internal standard phase index {std_index} out of range")
    w_std = float(std_wt_fraction)
    if not np.isfinite(w_std) or w_std <= 0 or w_std >= 100:
        raise ValueError("standard_wt_fraction must be finite and strictly between 0 and 100 wt%")
    w_std_su = float(std_wt_fraction_su) if std_wt_fraction_su is not None else 0.0
    if not np.isfinite(w_std_su) or w_std_su < 0:
        raise ValueError("standard_wt_fraction_su must be finite and nonnegative")

    masses = unit_masses * scales
    if not np.all(np.isfinite(masses)) or np.any(masses < 0) or masses[std_index] <= 0:
        raise ValueError("Internal standard phase must have a finite, positive refined mass/scale")

    m_s = unit_masses[std_index]
    s_s = scales[std_index]
    ms_prod = masses[std_index]
    unspiked_norm = 1.0 - w_std / 100.0

    # Point estimates in the spiked mixture: w_{i,spiked} = W_std * (m_i * s_i) / (m_s * s_s)
    spiked_wt = w_std * (masses / ms_prod)
    amorphous_spiked = 100.0 - float(np.sum(spiked_wt))

    # Point estimates in the original (unspiked) sample
    original_wt = np.zeros(p_count, dtype=float)
    for i in range(p_count):
        if i != std_index:
            original_wt[i] = spiked_wt[i] / unspiked_norm
    amorphous_original = amorphous_spiked / unspiked_norm

    sigma_s, any_varied, valid_cov = extract_scale_covariance(scale_names, covariance)
    if not any_varied or not valid_cov:
        spiked_su = np.full(p_count, np.nan)
        original_su = np.full(p_count, np.nan)
        amorphous_spiked_su = np.nan
        amorphous_original_su = np.nan
    else:
        # Augmented parameter vector: [s_0, ..., s_{P-1}, W_std]
        sigma_aug = np.zeros((p_count + 1, p_count + 1), dtype=float)
        sigma_aug[:p_count, :p_count] = sigma_s
        sigma_aug[p_count, p_count] = w_std_su * w_std_su

        # Jacobian for spiked_wt[i] = W_std * (m_i * s_i) / (m_s * s_s)
        J_spiked = np.zeros((p_count, p_count + 1), dtype=float)
        for i in range(p_count):
            if i == std_index:
                J_spiked[i, p_count] = 1.0
            else:
                J_spiked[i, i] = w_std * unit_masses[i] / ms_prod
                J_spiked[i, std_index] = -w_std * masses[i] / (m_s * s_s * s_s)
                J_spiked[i, p_count] = masses[i] / ms_prod
        cov_spiked = J_spiked @ sigma_aug @ J_spiked.T
        spiked_su = np.sqrt(np.maximum(0.0, np.diag(cov_spiked)))

        # amorphous_spiked = 100 - sum_i w_{i,spiked}
        j_amorph_spiked = -np.sum(J_spiked, axis=0)
        var_amorph_spiked = float(j_amorph_spiked @ sigma_aug @ j_amorph_spiked.T)
        amorphous_spiked_su = math.sqrt(max(0.0, var_amorph_spiked))

        # Jacobian for original_wt[i] = (100 / (100 - W_std)) * W_std * (m_i * s_i) / (m_s * s_s) for i != s
        J_orig = np.zeros((p_count, p_count + 1), dtype=float)
        d_factor_dw = 100.0 / ((100.0 - w_std) ** 2)
        for i in range(p_count):
            if i != std_index:
                J_orig[i, :p_count] = J_spiked[i, :p_count] / unspiked_norm
                J_orig[i, p_count] = d_factor_dw * (masses[i] / ms_prod) * 100.0
        cov_orig = J_orig @ sigma_aug @ J_orig.T
        original_su = np.sqrt(np.maximum(0.0, np.diag(cov_orig)))

        # amorphous_original = 100 - sum_{i != s} original_wt[i]
        j_amorph_orig = -np.sum(J_orig, axis=0)
        var_amorph_orig = float(j_amorph_orig @ sigma_aug @ j_amorph_orig.T)
        amorphous_original_su = math.sqrt(max(0.0, var_amorph_orig))

    return {
        "spiked_wt_fractions": spiked_wt,
        "spiked_wt_fractions_su": spiked_su,
        "amorphous_spiked_wt_fraction": float(amorphous_spiked),
        "amorphous_spiked_wt_fraction_su": float(amorphous_spiked_su),
        "original_wt_fractions": original_wt,
        "original_wt_fractions_su": original_su,
        "amorphous_wt_fraction": float(amorphous_original),
        "amorphous_wt_fraction_su": float(amorphous_original_su),
    }



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
