import logging
import os
import sys
from pathlib import Path

from .utils import HiddenPrints, install_gsas2, read_tiff_as_DataArray

logger = logging.getLogger("easyxrd")
logger.addHandler(logging.NullHandler())


def configure_logging(level=logging.INFO, stream=None):
    """Enable easyXRD progress messages without configuring the root logger.

    Repeated calls replace only the handler installed by this function.
    Use level="WARNING" for rejections only or "ERROR" for quiet batch runs.
    Third-party GSAS-II stdout remains controlled by sample.verbose.
    """
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler._easyxrd_owned = True
    logger.setLevel(level)
    for existing in list(logger.handlers):
        if getattr(existing, "_easyxrd_owned", False):
            logger.removeHandler(existing)
            existing.close()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Configurable keyword defaults. Resolved when a call omits the argument.
easyxrd_defaults = {
    # core
    "verbose": False,
    "radial_unit": "q",
    "plot_figsize": (8, 6),
    "plot_i2d_robust": True,
    "plot_i1d_ylogscale": True,
    "plot_i2d_logscale": True,
    "plot_site_str_x": 0.3,
    "plot_site_str_y": 0.8,
    "plot_close": False,
    "plot_show_wt_fractions": False,
    "plot_show_phase_profiles": True,
    "plot_normalized_residuals": False,
    # plotters
    "i1d_plotter_xlabel": True,
    "i1d_plotter_return_da": False,
    "i1d_plotter_title_str": "",
    "i1d_plotter_ncol": 1,
    "i1d_plotter_show_Ybkg_old": False,
    "i1d_plotter_show_phase_profiles": True,
    "i2d_plotter_xlabel": False,
    "i2d_plotter_cbar": True,
    "i2d_plotter_cmap": "Greys",
    "i2d_plotter_title_str": "",
    "i2d_plotter_annotate": False,
    "phases_plotter_phase_label_x": 0.8,
    "phases_plotter_phase_label_y": 0.75,
    "phases_plotter_phase_label_yshift": -0.2,
    "phases_plotter_show_hkls": True,
    "plot_dpi": 128,
    "refinement_plotter_dpi": 150,
    "refinement_plotter_min_step_detail": 4,
    "refinement_plotter_show": False,
    "refinement_plotter_return_fig": False,
    # Optional processing and refinement settings
    "load_xrd_data_median_filter_kernel_size": None,
    "load_xrd_data_error_model": "poisson",
    "load_xrd_data_polarization_factor": None,
    "refine_min_gof_improvement": None,
    "setup_gsas2_refiner_instprm_preset": None,
    "setup_gsas2_refiner_min_gof_improvement": None,
    "set_LeBail_to": True,
    "set_LeBail_min_gof_improvement": None,
    "refine_cell_parameters_scale_trials": None,
    # data_loader
    "load_xrd_data_integrate2d": True,
    "load_xrd_data_median_filter_on_i2d": True,
    "load_xrd_data_i1d_array_wavelength_in_angstrom": 0.1814,
    "load_xrd_data_i1d_array_radial_unit": "q",
    "load_xrd_data_txt_file_wavelength_in_angstrom": 0.1814,
    "load_xrd_data_txt_file_comments": "#",
    "load_xrd_data_txt_file_skiprows": 0,
    "load_xrd_data_txt_file_usecols": None,
    "load_xrd_data_txt_file_radial_unit": "tth",
    "load_xrd_data_radial_range": (0.1, 11.1),
    "load_xrd_data_radial_npts": 1000,
    "load_xrd_data_delta_q": 0.001,
    "load_xrd_data_npt_azimuthal": 91,
    "load_xrd_data_plot": True,
    "load_xrd_data_method": ("bbox", "csr", "cython"),
    "load_xrd_data_normalization_factor": 1.0,
    "export_i1d_to_mode": "xye",
    "export_i1d_to_subtract_baseline": False,
    "export_i1d_to_fmt": "%.4e %.4e",
    "export_i1d_to_header": False,
    # baseline
    "get_baseline_use_iarpls": True,
    "get_baseline_radial_rolling": -1,
    "get_baseline_iarpls_lam": 100000.0,
    "get_baseline_plot": True,
    "get_baseline_get_i2d_baseline": False,
    "get_baseline_use_i2d_baseline": False,
    "get_baseline_spotty_data_correction": False,
    "get_baseline_spotty_data_correction_threshold": 1,
    # phases
    "load_phases_plot": True,
    "screen_phases_e_above_hull_max": 0.05,
    "screen_phases_max_candidates": 15,
    "screen_phases_peak_prominence": 5.0,
    "screen_phases_tolerance_q": 0.03,
    "screen_phases_use_residual": True,
    # refinement
    "refine_update_ds": True,
    "refine_update_ds_phases": True,
    "refine_update_phases": True,
    "refine_update_previous_ds": True,
    "refine_update_previous_gpx": True,
    "refine_update_previous_phases": True,
    "refine_verbose": False,
    "refine_plot": False,
    "setup_gsas2_refiner_instprm_Polariz": 0,
    "setup_gsas2_refiner_instprm_Azimuth": 0,
    "setup_gsas2_refiner_instprm_Zero": 0,
    "setup_gsas2_refiner_instprm_U": 100,
    "setup_gsas2_refiner_instprm_V": 5,
    "setup_gsas2_refiner_instprm_W": 0.5,
    "setup_gsas2_refiner_instprm_X": 0,
    "setup_gsas2_refiner_instprm_Y": 0,
    "setup_gsas2_refiner_instprm_Z": 0,
    "setup_gsas2_refiner_instprm_SHL": 0.002,
    "setup_gsas2_refiner_do_1st_refinement": True,
    "setup_gsas2_refiner_yshift_multiplier": 0.01,
    "setup_gsas2_refiner_normalize": False,
    "setup_gsas2_refiner_normalize_to": 100,
    "setup_gsas2_refiner_plot": True,
    "refine_background_num_coeffs": 10,
    "refine_background_background_type": "chebyschev-1",
    "refine_background_set_to_false_after_refinement": True,
    "refine_background_plot": False,
    "refine_background_min_gof_improvement": 1.0,
    "set_background_refinement_set_num_coeffs_to": 10,
    "set_background_refinement_set_background_type_to": "chebyschev-1",
    "set_background_refinement_set_refine_to": True,
    "set_background_refinement_save_gpx": True,
    "clear_background_refinement_save_gpx": True,
    "refine_instrument_parameters_inst_pars_to_refine": ("U", "V", "W"),
    "refine_instrument_parameters_set_to_false_after_refinement": True,
    "refine_instrument_parameters_plot": False,
    "refine_instrument_parameters_min_gof_improvement": 1.0,
    "set_instrument_parameters_refinement_set_inst_pars_to_refine": ("U", "V", "W"),
    "set_instrument_parameters_refinement_set_refine_to": True,
    "set_instrument_parameters_refinement_save_gpx": True,
    "clear_instrument_parameters_refinement_save_gpx": True,
    "set_LeBail_phase_ind": "all",
    "set_LeBail_refine": True,
    "set_LeBail_plot": False,
    "refine_cell_parameters_phase_ind": "all",
    "refine_cell_parameters_set_to_false_after_refinement": True,
    "refine_cell_parameters_plot": False,
    "refine_cell_parameters_report": False,
    "refine_cell_parameters_min_gof_improvement": 0.1,
    "set_cell_parameters_refinement_set_refine_to": True,
    "set_cell_parameters_refinement_phase_ind": "all",
    "set_cell_parameters_refinement_save_gpx": True,
    "clear_cell_parameters_refinement_save_gpx": True,
    "refine_strain_broadening_phase_ind": "all",
    "refine_strain_broadening_model_type": "isotropic",
    "refine_strain_broadening_set_to_false_after_refinement": True,
    "refine_strain_broadening_plot": False,
    "refine_strain_broadening_report": False,
    "refine_strain_broadening_min_gof_improvement": 1.0,
    "set_strain_broadening_refinement_set_refine_to": True,
    "set_strain_broadening_refinement_phase_ind": "all",
    "set_strain_broadening_refinement_model_type": "isotropic",
    "set_strain_broadening_refinement_save_gpx": True,
    "clear_strain_broadening_refinement_save_gpx": True,
    "refine_size_broadening_phase_ind": "all",
    "refine_size_broadening_model_type": "isotropic",
    "refine_size_broadening_set_to_false_after_refinement": True,
    "refine_size_broadening_plot": False,
    "refine_size_broadening_report": False,
    "refine_size_broadening_min_gof_improvement": 1.0,
    "set_size_broadening_refinement_set_refine_to": True,
    "set_size_broadening_refinement_phase_ind": "all",
    "set_size_broadening_refinement_model_type": "isotropic",
    "set_size_broadening_refinement_save_gpx": True,
    "clear_size_broadening_refinement_save_gpx": True,
    "refine_phase_fractions_set_to_false_after_refinement": True,
    "refine_phase_fractions_plot": False,
    "refine_phase_fractions_min_gof_improvement": 0.1,
    "set_phase_fractions_refinement_set_refine_to": True,
    "set_phase_fractions_refinement_save_gpx": True,
    "clear_phase_fractions_refinement_save_gpx": True,
    "refine_preferred_orientation_phase_ind": "all",
    "refine_preferred_orientation_harmonics_order": 4,
    "refine_preferred_orientation_set_to_false_after_refinement": True,
    "refine_preferred_orientation_plot": False,
    "refine_preferred_orientation_min_gof_improvement": 1.0,
    "refine_preferred_orientation_model": "SH",
    "refine_preferred_orientation_md_axis": (0, 0, 1),
    "refine_preferred_orientation_md_ratio": 1.0,
    "set_preferred_orientation_refinement_set_refine_to": True,
    "set_preferred_orientation_refinement_phase_ind": 0,
    "set_preferred_orientation_refinement_harmonics_order_to": 4,
    "set_preferred_orientation_refinement_model": "SH",
    "set_preferred_orientation_refinement_md_axis": (0, 0, 1),
    "set_preferred_orientation_refinement_md_ratio": 1.0,
    "set_preferred_orientation_refinement_save_gpx": True,
    "clear_preferred_orientation_refinement_phase_ind": "all",
    "clear_preferred_orientation_refinement_save_gpx": True,
    "refine_site_property_phase_ind": 0,
    "refine_site_property_site_ind": 0,
    "refine_site_property_refinement_flags": "",
    "refine_site_property_set_to_false_after_refinement": True,
    "refine_site_property_plot": False,
    "refine_site_property_min_gof_improvement": 0.1,
    "set_site_property_refinement_phase_ind": 0,
    "set_site_property_refinement_site_ind": 0,
    "set_site_property_refinement_refinement_flags": "",
    "set_site_property_refinement_save_gpx": True,
    "clear_site_property_refinement_phase_ind": 0,
    "clear_site_property_refinement_site_ind": 0,
    "clear_site_property_refinement_save_gpx": True,
    "fine_tune_gpx_min_gof_improvement": 1.0,
    "replace_gpx_with_min_gof_improvement": 1.0,
    "setup_gsas2_refiner_excluded_regions_unit": "q",
    "set_excluded_regions_unit": "q",
    "set_excluded_regions_save_gpx": True,
    "clear_excluded_regions_save_gpx": True,
    "refine_sample_parameters_sample_pars_to_refine": ("Shift",),
    "refine_sample_parameters_set_to_false_after_refinement": True,
    "refine_sample_parameters_plot": False,
    "refine_sample_parameters_min_gof_improvement": 0.1,
    "set_sample_parameters_refinement_set_sample_pars_to_refine": ("Shift",),
    "set_sample_parameters_refinement_set_refine_to": True,
    "set_sample_parameters_refinement_save_gpx": True,
    "clear_sample_parameters_refinement_save_gpx": True,
    "refine_series_coord_name": "scan",
    "refine_series_refine_steps": ("refine",),
    "refine_series_min_gof_improvement": None,
    "refine_series_stop_on_error": False,
    "plot_refinement_history_dpi": 128,
    "plot_refinement_history_min_step_detail": 4,
    "plot_refinement_history_show": False,
    "plot_refinement_history_return_fig": False,
}
user_home = os.path.expanduser("~")

# Setting up easyxrd_scratch folder
scratch_dir = os.environ.get(
    "EASYXRD_SCRATCH", os.path.join(user_home, ".easyxrd_scratch")
)
easyxrd_defaults["easyxrd_scratch_path"] = scratch_dir

# GSAS-II is imported on first refinement use, not during package import.
easyxrd_defaults["gsasii_lib_path"] = os.environ.get("GSASII_PATH", "not found")

# Check Materials Project API key from env or scratch folder
mp_api_key = os.environ.get("MP_API_KEY")
easyxrd_defaults["mp_api_key"] = mp_api_key or "not found"


class _Default:
    def __repr__(self):
        return "<easyxrd default>"


DEFAULT = _Default()


def resolve_default(value, name):
    """Read configuration at call time and isolate mutable configured values."""
    if value is not DEFAULT:
        return value
    from copy import deepcopy

    return deepcopy(easyxrd_defaults[name])


def set_defaults(name, val):
    """Set a session default; existing objects retain their plotting settings.

    Use print_defaults() to list supported settings. Explicit call arguments
    override these values. Custom keys remain supported for compatibility.
    """
    global easyxrd_defaults
    easyxrd_defaults[name] = val


def print_defaults():
    """Print current configuration defaults."""
    for key, val in easyxrd_defaults.items():
        if key != "mp_api_key":
            print("%s : %s" % (key, val))
        else:
            masked = "......" if val not in (None, "", "not found", "invalid") else val
            print("%s : %s" % (key, masked))


def show_config():
    """Print environment versions and configuration, with the API key masked."""
    import platform
    from importlib import metadata

    versions = {
        "Platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    for package in [
        "easyxrd",
        "pyFAI",
        "fabio",
        "numpy",
        "scipy",
        "pymatgen",
        "GSAS-II",
    ]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed or local"

    try:
        with HiddenPrints():
            from GSASII import GSASIIscriptable
        versions["GSASII_module"] = GSASIIscriptable.__file__
    except Exception:
        versions["GSASII_module"] = "not found"

    print("=" * 40)
    print("easyXRD environment and configurations")
    print("=" * 40)
    for key, value in versions.items():
        print(f"{key:<20}: {value}")
    for key, value in easyxrd_defaults.items():
        if key == "mp_api_key" and value not in (None, "", "not found", "invalid"):
            value = "......"
        print(f"{key:<20}: {value}")
    print("-" * 40)
    print()


# Re-export exrd for convenient package imports
from .core import exrd

__all__ = [
    "HiddenPrints",
    "configure_logging",
    "easyxrd_defaults",
    "exrd",
    "install_gsas2",
    "print_defaults",
    "read_tiff_as_DataArray",
    "set_defaults",
    "show_config",
]
