from . import DEFAULT, resolve_default
from .results import format_value_with_su

import logging
import warnings
import json
from collections import OrderedDict
import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator



def i1d_plotter(
    ds,
    ax,
    ds_previous=None,
    i1d_ylogscale=DEFAULT,
    xlabel=DEFAULT,
    return_da=DEFAULT,
    title_str=DEFAULT,
    ncol=DEFAULT,
    show_Ybkg_old=DEFAULT,
    show_phase_profiles=DEFAULT,
):
    """Plot 1D XRD intensity profile (observed, calculated, and background)."""
    i1d_ylogscale = resolve_default(i1d_ylogscale, "plot_i1d_ylogscale")
    xlabel = resolve_default(xlabel, "i1d_plotter_xlabel")
    return_da = resolve_default(return_da, "i1d_plotter_return_da")
    title_str = resolve_default(title_str, "i1d_plotter_title_str")
    ncol = resolve_default(ncol, "i1d_plotter_ncol")
    show_Ybkg_old = resolve_default(show_Ybkg_old, "i1d_plotter_show_Ybkg_old")
    show_phase_profiles = resolve_default(show_phase_profiles, "i1d_plotter_show_phase_profiles")

    da_Y_calc = da_Y_bkg = None
    if "i1d_baseline" in ds.keys():
        da_Y_obs = ds.i1d - ds.i1d_baseline
        if "i1d_refined" in ds.keys():
            da_Y_calc = ds.i1d_refined - ds.i1d_baseline
            da_Y_bkg = ds.i1d_gsas_background
        ylabel = "Intensity"
    else:
        da_Y_obs = ds.i1d
        if "i1d_refined" in ds.keys():
            da_Y_calc = ds.i1d_refined
            da_Y_bkg = ds.i1d_gsas_background
        ylabel = "Intensity"

    if i1d_ylogscale:
        if "i1d_refined" in ds.keys():
            min_obs = np.nanmin(da_Y_obs.values)
            min_calc = np.nanmin(da_Y_calc.values)
            min_bkg = np.nanmin(da_Y_bkg.values)
            min_all = min([min_obs, min_calc, min_bkg])
        else:
            min_all = np.nanmin(da_Y_obs.values)

        if min_all < 1:
            extra_yshift = 1 - min_all
            ylabel = "%s +%.2f" % (ylabel, extra_yshift)
        else:
            extra_yshift = 0

        da_Y_obs = da_Y_obs + extra_yshift
        if "i1d_refined" in ds.keys():
            da_Y_calc = da_Y_calc + extra_yshift
            da_Y_bkg = da_Y_bkg + extra_yshift
    else:
        extra_yshift = 0

    da_Y_obs.plot(ax=ax, color="k", linewidth=2, label="Y$_{obs.}$")
    if "i1d_refined" in ds.keys():
        da_Y_calc.plot(ax=ax, alpha=0.9, linewidth=1, color="y", label="Y$_{calc.}$")
        da_Y_bkg.plot(ax=ax, alpha=0.9, linewidth=1, color="r", label="Y$_{bkg.}$")
        if (
            show_phase_profiles
            and "i1d_phase_profiles" in ds
            and ds.i1d_phase_profiles.sizes.get("phase", 0) > 1
        ):
            for p_idx, p_name in enumerate(ds.i1d_phase_profiles.coords["phase"].values):
                da_phase = (
                    ds.i1d_gsas_background
                    + ds.i1d_phase_profiles.isel(phase=p_idx)
                    + extra_yshift
                )
                da_phase.plot(
                    ax=ax,
                    alpha=0.75,
                    linewidth=0.9,
                    linestyle="-.",
                    color="C%d" % p_idx,
                    label="Y$_{calc.}$ (%s)" % p_name,
                )

    if "excluded_regions_q" in ds.attrs:
        try:
            excl_q = json.loads(ds.attrs["excluded_regions_q"])
            for idx_reg, (q_low, q_high) in enumerate(excl_q):
                ax.axvspan(
                    float(q_low),
                    float(q_high),
                    color="0.85",
                    alpha=0.45,
                    zorder=0,
                    label="Excluded" if idx_reg == 0 else None,
                )
        except Exception:
            pass

    if ds_previous is not None and "i1d_refined" in ds_previous:
        if "i1d_baseline" in ds_previous.keys():
            if "i1d_refined" in ds_previous.keys():
                da_Y_calc_previous = ds_previous.i1d_refined - ds_previous.i1d_baseline
                da_Y_bkg_previous = ds_previous.i1d_gsas_background
        else:
            if "i1d_refined" in ds_previous.keys():
                da_Y_calc_previous = ds_previous.i1d_refined
                da_Y_bkg_previous = ds_previous.i1d_gsas_background

        if "i1d_refined" in ds_previous.keys():
            da_Y_calc_previous = da_Y_calc_previous + extra_yshift
            da_Y_bkg_previous = da_Y_bkg_previous + extra_yshift

        da_Y_calc_previous.plot(
            ax=ax,
            alpha=0.9,
            linewidth=1.2,
            linestyle="--",
            color="y",
            label="Y$_{calc.}$ (old)",
        )

        if show_Ybkg_old:
            da_Y_bkg_previous.plot(
                ax=ax,
                alpha=0.9,
                linewidth=1.2,
                linestyle="--",
                color="r",
                label="Y$_{bkg.}$ (old)",
            )

    ax.set_ylabel(ylabel)
    ax.set_title(title_str, fontsize=8, color="r")
    ax.legend(loc="upper right", fontsize=8, ncol=ncol)
    ax.set_xlim([ds.i1d.radial[0], ds.i1d.radial[-1]])

    if xlabel:
        ax.set_xlabel(ds.i1d.attrs["xlabel"])
    else:
        ax.set_xlabel(None)

    if i1d_ylogscale:
        ax.set_yscale("log")

    if return_da:
        return [da_Y_obs, da_Y_calc, da_Y_bkg]


def i2d_plotter(
    ds,
    ax,
    i2d_robust=DEFAULT,
    i2d_logscale=DEFAULT,
    xlabel=DEFAULT,
    cbar=DEFAULT,
    cmap=DEFAULT,
    title_str=DEFAULT,
    annotate=DEFAULT,
):
    """Plot 2D caked XRD pattern (azimuthal vs radial)."""
    i2d_robust = resolve_default(i2d_robust, "plot_i2d_robust")
    i2d_logscale = resolve_default(i2d_logscale, "plot_i2d_logscale")
    xlabel = resolve_default(xlabel, "i2d_plotter_xlabel")
    cbar = resolve_default(cbar, "i2d_plotter_cbar")
    cmap = resolve_default(cmap, "i2d_plotter_cmap")
    title_str = resolve_default(title_str, "i2d_plotter_title_str")
    annotate = resolve_default(annotate, "i2d_plotter_annotate")

    if ("i2d_baseline" in ds.keys()) and ("roi_azimuthal_range" in ds.i2d.attrs):
        da_i2d = ds.i2d
        vmin = 0
        i2d_str = "i2d"
    elif "i2d_baseline" in ds.keys():
        da_i2d = ds.i2d - ds.i2d_baseline
        vmin = 0
        i2d_str = "i2d-i2d_baseline"
    else:
        da_i2d = ds.i2d
        vmin = 0
        i2d_str = "i2d"

    if i2d_logscale:
        da_i2d = np.log10(da_i2d.where(da_i2d > 0))
        vmin = None
        i2d_str = "Log$_{10}$(%s)" % (i2d_str)

    if cbar:
        da_i2d.plot.imshow(
            ax=ax,
            robust=i2d_robust,
            add_colorbar=cbar,
            cbar_kwargs=dict(orientation="vertical", pad=0.02, shrink=0.8, label=None),
            cmap=cmap,
            vmin=vmin,
        )
    else:
        da_i2d.plot.imshow(
            ax=ax, robust=i2d_robust, add_colorbar=cbar, cmap=cmap, vmin=vmin
        )

    if annotate:
        ax.annotate(
            "%s (i2d_robust=%s)" % (i2d_str, i2d_robust),
            xy=(0.01, 0.9),
            xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points",
            color="r",
            fontsize=8,
            rotation=0,
        )

    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_facecolor("#FFF7D9")
    if xlabel:
        ax.set_xlabel(ds.i2d.attrs["xlabel"])
    else:
        ax.set_xlabel(None)

    if ("roi_azimuthal_range" in ds.i2d.attrs) and ("i1d" in ds.keys()):
        roi_xy = [ds.i1d.radial.values[0], ds.i1d.attrs["roi_azimuthal_range"][0]]
        roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
        roi_height = (
            ds.i1d.attrs["roi_azimuthal_range"][1]
            - ds.i1d.attrs["roi_azimuthal_range"][0]
        )
        rect = matplotlib.patches.Rectangle(
            xy=roi_xy, width=roi_width, height=roi_height, color="r", alpha=0.1
        )
        ax.add_patch(rect)

    ax.set_title(title_str, fontsize=8, color="r")


_PATTERN_CACHE = OrderedDict()
_PATTERN_CACHE_MAXSIZE = 128


def _calc_reflection_pattern(xrdc, st, wavelength, radial_min, radial_max):
    """Calculate or retrieve cached Bragg peak positions and intensities."""

    arg0 = np.clip(radial_min * (wavelength / (4 * np.pi)), -1.0, 1.0)
    arg1 = np.clip(radial_max * (wavelength / (4 * np.pi)), -1.0, 1.0)
    tth0 = float(np.rad2deg(2 * np.arcsin(arg0)))
    tth1 = float(np.rad2deg(2 * np.arcsin(arg1)))
    two_theta_range = (min(tth0, tth1), max(tth0, tth1))

    cache_key = (json.dumps(st.as_dict(), sort_keys=True),
                 repr(sorted(xrdc.__dict__.items())), float(wavelength), two_theta_range)
    if cache_key not in _PATTERN_CACHE:
        ps = xrdc.get_pattern(st, scaled=True, two_theta_range=two_theta_range)
        refl_x = 4 * np.pi / wavelength * np.sin(np.deg2rad(ps.x) / 2)
        _PATTERN_CACHE[cache_key] = (refl_x, ps.y, ps.hkls)
        if len(_PATTERN_CACHE) > _PATTERN_CACHE_MAXSIZE:
            _PATTERN_CACHE.popitem(last=False)
    _PATTERN_CACHE.move_to_end(cache_key)
    import copy
    return copy.deepcopy(_PATTERN_CACHE[cache_key])


def phases_plotter(
    ds,
    ax_main,
    phases=None,
    line_axes=None,
    phase_label_x=DEFAULT,
    phase_label_y=DEFAULT,
    phase_label_yshift=DEFAULT,
    show_hkls = DEFAULT,
):
    """Plot Bragg reflection lines and phase labels."""
    phase_label_x = resolve_default(phase_label_x, "phases_plotter_phase_label_x")
    phase_label_y = resolve_default(phase_label_y, "phases_plotter_phase_label_y")
    phase_label_yshift = resolve_default(phase_label_yshift, "phases_plotter_phase_label_yshift")
    show_hkls = resolve_default(show_hkls, "phases_plotter_show_hkls")

    wavelength = ds.i1d.attrs["wavelength_in_angst"]
    xrdc = XRDCalculator(wavelength=wavelength)

    line_axes = [] if line_axes is None else line_axes
    if phases is None:
        ds_phases = {}
        num_phases = ds.attrs.get("num_phases", 0)
        for aa in range(num_phases):
            cif_key = "PhaseInd_%d_cif" % aa
            label_key = "PhaseInd_%d_label" % aa
            if cif_key in ds.attrs:
                try:
                    st = Structure.from_str(ds.attrs[cif_key], fmt="cif")
                    label = ds.attrs.get(label_key, "Phase_%d" % aa)
                    ds_phases[label] = st
                except Exception as exc:
                    logging.getLogger(__name__).warning("Could not parse CIF for phase %s: %s", aa, exc)
        phases = ds_phases

    radial_min = ds.i1d.radial.values[0]
    radial_max = ds.i1d.radial.values[-1]

    if isinstance(phases, dict):
        phase_items = list(phases.items())
    elif isinstance(phases, (list, tuple)):
        phase_items = [(getattr(st, "formula", f"Phase_{idx}"), st) for idx, st in enumerate(phases)]
    else:
        phase_items = []

    for e, (st_name, st) in enumerate(phase_items):
        refl_X, refl_Y, refl_HKLs = _calc_reflection_pattern(
            xrdc, st, wavelength, radial_min, radial_max
        )

        if len(refl_X) > 0:
            color = "C%d" % e
            ax_main.vlines(
                refl_X, 0, 1, transform=ax_main.get_xaxis_transform(),
                lw=0.3, linestyle="--", colors=color
            )
            for a in line_axes:
                a.vlines(
                    refl_X, 0, 1, transform=a.get_xaxis_transform(),
                    lw=0.3, linestyle="--", colors=color
                )

            markerline, stemlines, stem_baseline = ax_main.stem(
                refl_X, refl_Y, markerfmt="."
            )
            plt.setp(stemlines, linewidth=0.5, color=color)
            plt.setp(markerline, color=color)

            if show_hkls:
                for ee,y in enumerate(refl_Y):
                    if y > 10:
                        ax_main.annotate(refl_HKLs[ee][0]['hkl'],
                                        xy = (refl_X[ee],refl_Y[ee]),
                                        rotation=30,
                                        fontsize=4,
                                        ha='center',
                                        va='bottom',
                                        color=color
                        )


        ax_main.text(
            phase_label_x,
            phase_label_y + e * phase_label_yshift,
            st_name,
            color="C%d" % e,
            transform=ax_main.transAxes,
        )

    ax_main.set_xlabel(ds.i1d.attrs["xlabel"])
    ax_main.set_ylim(bottom=1, top=130)
    ax_main.set_yticks([])


def exrd_plotter(
    ds,
    ds_previous=None,
    phases=None,
    figsize=DEFAULT,
    i2d_robust=DEFAULT,
    i2d_logscale=DEFAULT,
    i1d_ylogscale=DEFAULT,
    plot_hint="1st_loaded_data",
    title_str=None,
    export_fig_as=None,
    i1d_plot_radial_range=None,
    i1d_plot_bottom=None,
    i1d_plot_top=None,
    title=None,
    site_str_x=DEFAULT,
    site_str_y=DEFAULT,
    show_wt_fractions=DEFAULT,
    show_phase_profiles=DEFAULT,
    normalized_residuals=DEFAULT,
    dpi=DEFAULT,
):
    """Main plotting coordinator for easyXRD workflow stages."""
    figsize = resolve_default(figsize, "plot_figsize")
    i2d_robust = resolve_default(i2d_robust, "plot_i2d_robust")
    i2d_logscale = resolve_default(i2d_logscale, "plot_i2d_logscale")
    i1d_ylogscale = resolve_default(i1d_ylogscale, "plot_i1d_ylogscale")
    site_str_x = resolve_default(site_str_x, "plot_site_str_x")
    site_str_y = resolve_default(site_str_y, "plot_site_str_y")
    show_wt_fractions = resolve_default(show_wt_fractions, "plot_show_wt_fractions")
    show_phase_profiles = resolve_default(show_phase_profiles, "plot_show_phase_profiles")
    normalized_residuals = resolve_default(normalized_residuals, "plot_normalized_residuals")
    dpi = resolve_default(dpi, "plot_dpi")

    if plot_hint in ("refinement_history", "plot_refinement_history", "refinement", "plot_refinement"):
        return refinement_plotter(ds, figsize=figsize, export_fig_as=export_fig_as)
    if ds is None or "i1d" not in ds or not ds.i1d.size:
        raise ValueError("Load a nonempty diffraction profile before plotting")
    if plot_hint is None and "i1d_refined" not in ds:
        plot_hint = "load_phases" if ds.attrs.get("num_phases", 0) else "load_xrd_data"
    has_i2d = "i2d" in ds.keys()

    if plot_hint == "load_xrd_data":
        fig = plt.figure(figsize=(figsize[0], figsize[1] / 1.5), dpi=dpi)
        if has_i2d:
            mosaic = """
                        B
                        C
                        C
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            i2d_plotter(
                ds,
                ax=ax_dict["B"],
                cbar=True,
                i2d_robust=i2d_robust,
                i2d_logscale=i2d_logscale,
                annotate=True,
                title_str=title,
            )
            i1d_plotter(
                ds,
                ax=ax_dict["C"],
                ds_previous=None,
                i1d_ylogscale=i1d_ylogscale,
                xlabel=True,
                return_da=False,
                title_str="",
            )
        else:
            ax_dict = fig.subplot_mosaic("C", sharex=True)
            i1d_plotter(
                ds,
                ax=ax_dict["C"],
                ds_previous=None,
                i1d_ylogscale=i1d_ylogscale,
                xlabel=True,
                return_da=False,
                title_str=title,
            )

    elif plot_hint == "get_baseline":
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if has_i2d:
            mosaic = """
                        AA222
                        AA222
                        AA111
                        AA111
                        AA111
                        AA111
                        """
        else:
            mosaic = """
                        AA111
                        AA111
                        AA111
                        AA111
                        """
        ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
        ax_a = ax_dict["A"]
        ax_a.set_xlim([ds.i1d.radial[0], ds.i1d.radial[-1]])

        label_i1d = "i1d (norm.)" if "normalized_to" in ds.i1d.attrs else "i1d"
        ds.i1d.plot(ax=ax_a, label=label_i1d)

        if "i1d_baseline" in ds.keys():
            label_bkg = "i1d_baseline (norm.)" if "normalized_to" in ds.i1d.attrs else "i1d_baseline"
            ds.i1d_baseline.plot(ax=ax_a, label=label_bkg)

        ax_a.set_yscale("log")
        ax_a.set_xlabel(ds.i1d.attrs["xlabel"])
        ax_a.set_ylabel("Intensity (a.u.)")
        ax_a.legend(fontsize=6)
        ax_a.set_title(title)

        if has_i2d:
            i2d_plotter(
                ds,
                ax=ax_dict["2"],
                cbar=True,
                i2d_robust=i2d_robust,
                i2d_logscale=i2d_logscale,
                annotate=True,
            )

        i1d_plotter(
            ds,
            ax=ax_dict["1"],
            ds_previous=None,
            i1d_ylogscale=i1d_ylogscale,
            xlabel=True,
            return_da=False,
            title_str="",
        )

    elif plot_hint == "load_phases":
        if has_i2d:
            fig = plt.figure(figsize=(figsize[0], figsize[1]), dpi=dpi)
            mosaic = """
                        2
                        2
                        1
                        1
                        1
                        1
                        P
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax_1 = ax_dict["1"]
            i1d_plotter(
                ds,
                ax=ax_1,
                ds_previous=None,
                i1d_ylogscale=i1d_ylogscale,
                xlabel=True,
                return_da=False,
                title_str="",
            )
            ax_1.set_xlabel(None)
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["1"], ax_dict["2"], ax_dict["P"]],
            )
            i2d_plotter(
                ds,
                ax=ax_dict["2"],
                cbar=True,
                i2d_robust=i2d_robust,
                i2d_logscale=i2d_logscale,
                annotate=True,
            )
            ax_dict["2"].set_title(title)
        else:
            fig = plt.figure(figsize=(figsize[0], figsize[1] / 1.5), dpi=dpi)
            mosaic = """
                        1
                        1
                        1
                        1
                        1
                        P
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax_1 = ax_dict["1"]
            i1d_plotter(
                ds,
                ax=ax_1,
                ds_previous=None,
                i1d_ylogscale=i1d_ylogscale,
                xlabel=True,
                return_da=False,
                title_str=title,
            )
            ax_1.set_xlabel(None)
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["1"], ax_dict["P"]],
            )

    elif plot_hint in ("refinement_history", "plot_refinement_history", "refinement", "plot_refinement"):
        return refinement_plotter(
            ds,
            figsize=figsize,
            export_fig_as=export_fig_as,
        )

    elif plot_hint is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if has_i2d:
            mosaic = """
                        2
                        2
                        1
                        1
                        1
                        1
                        1
                        1
                        D
                        P
                    """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            i2d_plotter(
                ds,
                ax=ax_dict["2"],
                cbar=False,
                i2d_robust=i2d_robust,
                i2d_logscale=i2d_logscale,
                title_str=title_str,
                annotate=False,
            )
            ax_dict["2"].set_title(title)
            ax_dict["2"].set_yticks([])
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["2"], ax_dict["1"], ax_dict["D"], ax_dict["P"]],
            )
            ax_dict["P"].set_yticks([])
        else:
            mosaic = """
                        1
                        1
                        1
                        1
                        1
                        1
                        D
                        P
                    """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["1"], ax_dict["D"], ax_dict["P"]],
            )
            ax_dict["P"].set_yticks([])

        [da_Y_obs, da_Y_calc, da_Y_bkg] = i1d_plotter(
            ds,
            ax=ax_dict["1"],
            ds_previous=None,
            xlabel=False,
            i1d_ylogscale=i1d_ylogscale,
            return_da=True,
            ncol=1,
            show_phase_profiles=show_phase_profiles,
        )

        fit_str = "Rwp/GoF = %.3f/%.3f" % (ds.attrs["Rwp"], ds.attrs["GOF"])
        ax_dict["1"].annotate(
            "%s" % (fit_str),
            xy=(0.75, 0.95),
            xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points",
            color="k",
            fontsize=10,
            rotation=0,
        )

        for e, si in enumerate(range(ds.attrs["num_phases"])):
            site_ind = si
            site_label = ds.attrs["PhaseInd_%d_label" % site_ind]
            site_SGSys = ds.attrs["PhaseInd_%d_SGSys" % site_ind]
            site_SpGrp = ds.attrs["PhaseInd_%d_SpGrp" % site_ind]

            site_a = ds.attrs["PhaseInd_%d_cell_a" % site_ind]
            site_b = ds.attrs["PhaseInd_%d_cell_b" % site_ind]
            site_c = ds.attrs["PhaseInd_%d_cell_c" % site_ind]
            site_alpha = ds.attrs["PhaseInd_%d_cell_alpha" % site_ind]
            site_beta = ds.attrs["PhaseInd_%d_cell_beta" % site_ind]
            site_gamma = ds.attrs["PhaseInd_%d_cell_gamma" % site_ind]

            site_a_str = format_value_with_su(
                site_a, ds.attrs.get("PhaseInd_%d_cell_a_su" % site_ind), precision=4
            )
            site_b_str = format_value_with_su(
                site_b, ds.attrs.get("PhaseInd_%d_cell_b_su" % site_ind), precision=4
            )
            site_c_str = format_value_with_su(
                site_c, ds.attrs.get("PhaseInd_%d_cell_c_su" % site_ind), precision=4
            )
            site_alpha_str = format_value_with_su(
                site_alpha, ds.attrs.get("PhaseInd_%d_cell_alpha_su" % site_ind), precision=2
            )
            site_beta_str = format_value_with_su(
                site_beta, ds.attrs.get("PhaseInd_%d_cell_beta_su" % site_ind), precision=2
            )
            site_gamma_str = format_value_with_su(
                site_gamma, ds.attrs.get("PhaseInd_%d_cell_gamma_su" % site_ind), precision=2
            )

            site_size_broadening_type = ds.attrs[
                "PhaseInd_%d_size_broadening_type" % site_ind
            ][:3]
            site_size0 = ds.attrs["PhaseInd_%d_size_0" % site_ind]
            if site_size0 == 1.0:
                size_str = ""
            else:
                size_str = "| Size: %.3f$\\mu$ (%s.) " % (
                    site_size0,
                    site_size_broadening_type,
                )

            site_strain_broadening_type = ds.attrs[
                "PhaseInd_%d_strain_broadening_type" % site_ind
            ][:3]
            site_mustrain0 = ds.attrs["PhaseInd_%d_mustrain_0" % site_ind]
            if site_mustrain0 == 1000.0:
                strain_str = ""
            else:
                strain_str = "| Strain: %.3f (%s.)" % (
                    site_mustrain0,
                    site_strain_broadening_type,
                )

            site_wt_fraction = ds.attrs["PhaseInd_%d_wt_fraction" % site_ind]
            if not np.isfinite(site_wt_fraction) or site_wt_fraction == 100.0 or not show_wt_fractions:
                str_fraction = ""
            else:
                wt_su = ds.attrs.get("PhaseInd_%d_wt_fraction_su" % site_ind, np.nan)
                wt_str = format_value_with_su(site_wt_fraction, wt_su, precision=2)
                str_fraction = "| wt%%=%s" % wt_str
                if (
                    site_ind == 0
                    and "amorphous_wt_fraction" in ds.attrs
                    and np.isfinite(ds.attrs["amorphous_wt_fraction"])
                ):
                    am_str = format_value_with_su(
                        ds.attrs["amorphous_wt_fraction"],
                        ds.attrs.get("amorphous_wt_fraction_su"),
                        precision=2,
                    )
                    str_fraction += f" (Amorph.={am_str}%%)"

            is_lebail = (
                ds.attrs.get("refinement_mode") == "LeBail"
                or ds.attrs.get("PhaseInd_%d_use_lebail" % site_ind, False)
                or not ds.attrs.get("weight_fractions_valid", True)
            )
            mode_tag = " [Le Bail]" if is_lebail else ""

            site_str = (
                "\n%s %s phase (%s)%s  %s %s %s \nLattice: a/b/c=%s/%s/%s ($\\alpha$/$\\beta$/$\\gamma$=%s/%s/%s) \n"
                % (
                    site_SpGrp.replace(" ", ""),
                    site_SGSys,
                    site_label,
                    mode_tag,
                    str_fraction,
                    size_str,
                    strain_str,
                    site_a_str,
                    site_b_str,
                    site_c_str,
                    site_alpha_str,
                    site_beta_str,
                    site_gamma_str,
                )
            )

            ax_dict["1"].annotate(
                "%s" % (site_str),
                xy=(site_str_x, site_str_y - 0.1 * e),
                xycoords="axes fraction",
                xytext=(0, 0),
                textcoords="offset points",
                color="C%d" % e,
                fontsize=8,
                rotation=0,
            )

        residual = da_Y_obs - da_Y_calc
        if normalized_residuals and "i1d_variance" in ds:
            var_tot = ds.i1d_variance
            if "i1d_baseline_variance" in ds:
                var_tot = var_tot + ds.i1d_baseline_variance
            sigma = np.sqrt(var_tot.where(var_tot > 0))
            (residual / sigma).plot(ax=ax_dict["D"], color="b")
            ax_dict["D"].set_ylabel(r"$\Delta / \sigma$", fontsize=8)
        else:
            residual.plot(ax=ax_dict["D"], color="b")
            ax_dict["D"].set_ylabel(r"$\Delta Y$", fontsize=8)
        if "excluded_regions_q" in ds.attrs:
            try:
                for q_low, q_high in json.loads(ds.attrs["excluded_regions_q"]):
                    ax_dict["D"].axvspan(
                        float(q_low), float(q_high), color="0.85", alpha=0.45, zorder=0
                    )
            except Exception:
                pass
        ax_dict["D"].axhline(y=0, linestyle="--", color="k", lw=0.5)
        ax_dict["D"].set_xlabel(None)

        if i1d_plot_radial_range is not None:
            xlims = [i1d_plot_radial_range[0], i1d_plot_radial_range[-1]]
            ax_dict["1"].set_xlim(xlims)
            if has_i2d:
                ax_dict["2"].set_xlim(xlims)
            ax_dict["P"].set_xlim(xlims)
            ax_dict["D"].set_xlim(xlims)

        ax_dict["1"].set_ylim(bottom=i1d_plot_bottom, top=i1d_plot_top)
        ax_dict["1"].legend(loc="upper left", fontsize=8, ncol=1)

        if not has_i2d:
            ax_dict["1"].set_title(title)

    else:
        # Generic refinement step plotting (covers "1st_refinement", "refine_background", "refine_cell_parameters", etc.)
        show_Ybkg_old = (plot_hint == "refine_background")

        if has_i2d:
            fig = plt.figure(figsize=(figsize[0], figsize[1]), dpi=dpi)
            mosaic = """
                        2
                        1
                        1
                        1
                        1
                        P
                    """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            i2d_plotter(
                ds,
                ax=ax_dict["2"],
                cbar=False,
                i2d_robust=i2d_robust,
                i2d_logscale=i2d_logscale,
                title_str=title_str,
                annotate=False,
            )
            i1d_plotter(
                ds,
                ax=ax_dict["1"],
                ds_previous=ds_previous,
                xlabel=False,
                i1d_ylogscale=i1d_ylogscale,
                return_da=False,
                show_Ybkg_old=show_Ybkg_old,
                show_phase_profiles=show_phase_profiles,
            )
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["2"], ax_dict["1"], ax_dict["P"]],
            )
        else:
            fig = plt.figure(figsize=(figsize[0], figsize[1] / 1.5), dpi=dpi)
            mosaic = """
                        1
                        1
                        1
                        1
                        1
                        P
                    """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            i1d_plotter(
                ds,
                ax=ax_dict["1"],
                ds_previous=ds_previous,
                xlabel=False,
                i1d_ylogscale=i1d_ylogscale,
                return_da=False,
                show_Ybkg_old=show_Ybkg_old,
                show_phase_profiles=show_phase_profiles,
            )
            ax_dict["1"].set_title(title_str, fontsize=8)
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["1"], ax_dict["P"]],
            )

    if export_fig_as is not None:
        fig.savefig(export_fig_as, dpi=dpi)


def refinement_plotter(
    ds,
    figsize=DEFAULT,
    dpi=DEFAULT,
    min_step_detail=DEFAULT,
    export_fig_as=None,
    show=DEFAULT,
    return_fig=DEFAULT,
):
    """
    Plot comprehensive refinement convergence history showing full progression
    and zoomed-in micro-refinements with accepted vs. rejected attempts.
    """
    figsize = resolve_default(figsize, "plot_figsize")
    dpi = resolve_default(dpi, "refinement_plotter_dpi")
    min_step_detail = resolve_default(min_step_detail, "refinement_plotter_min_step_detail")
    show = resolve_default(show, "refinement_plotter_show")
    return_fig = resolve_default(return_fig, "refinement_plotter_return_fig")

    if ds is None or "rwp_history" not in ds or ds.sizes.get("refinement_step", 0) == 0:
        raise ValueError(
            "No refinement history found in dataset. Run a refinement step first."
        )

    steps = np.asarray(ds.refinement_step.values)
    rwp = np.asarray(ds.rwp_history.values)
    rwp_att = np.asarray(ds.rwp_attempted_history.values)
    status = np.asarray(ds.refinement_status.values)
    labels = np.asarray(ds.refinement_label.values)

    # Clean parameter labels
    def format_label(l):
        l_str = str(l).replace("refine_", "").replace("setup_", "")
        mapping = {
            "initial": "Init",
            "LeBail": "Le Bail",
            "background": "Bkg",
            "cell_parameters": "Cell",
            "size_broadening": "Size",
            "strain_broadening": "Strain",
            "site_property": "Site U",
            "phase_fractions": "Phase Frac",
            "preferred_orientation": "Pref Orient",
            "instrument_parameters": "Inst Prm",
        }
        return mapping.get(l_str, l_str.replace("_", " "))

    step_labels = [format_label(l) for l in labels]

    # Detect boundary where Le Bail was switched off
    lebail_off_indices = [i for i, l in enumerate(labels) if l == "set_LeBail:False" and status[i] == "accepted"]
    has_stage_split = len(lebail_off_indices) > 0
    if has_stage_split:
        split_idx = lebail_off_indices[0]
        split_step = steps[split_idx] - 0.5
    else:
        split_step = None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, gridspec_kw={"height_ratios": [1, 1.35]}
    )

    if not np.any(np.isfinite(rwp)):
        for ax in (ax1, ax2):
            ax.text(.5, .5, "No finite retained Rwp; inspect refinement_status and refinement_reason",
                    ha="center", va="center", transform=ax.transAxes, wrap=True)
        if export_fig_as is not None:
            fig.savefig(export_fig_as, dpi=dpi)
        if show:
            plt.show()
        return (fig, (ax1, ax2)) if return_fig else None

    # --- Panel 1: Full Refinement Overview ---
    y_min_val = max(0, float(np.nanmin(rwp)) - 1)
    y_max_val = float(np.nanmax(rwp)) + (float(np.nanmax(rwp)) - y_min_val) * 0.15

    if split_step is not None and 1 < split_idx < len(steps):
        ax1.axvspan(0.5, split_step, color="#f1f5f9", alpha=0.8, zorder=1)
        ax1.axvspan(split_step, len(steps) + 0.5, color="#fefce8", alpha=0.6, zorder=1)
        ax1.axvline(split_step, color="#94a3b8", linestyle="--", lw=1, zorder=2)

        text_y = y_max_val - (y_max_val - y_min_val) * 0.15
        ax1.text(
            split_step / 2,
            text_y,
            "Le Bail Stage",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#475569",
        )
        ax1.text(
            (split_step + len(steps)) / 2,
            text_y,
            "Rietveld Stage",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#854d0e",
        )

    ax1.plot(
        steps,
        rwp,
        color="#0284c7",
        lw=2.2,
        label=r"Accepted $R_{\mathrm{wp}}$",
        zorder=3,
    )
    ax1.scatter(
        steps[status == "accepted"],
        rwp[status == "accepted"],
        color="#16a34a",
        edgecolor="white",
        s=55,
        lw=1.2,
        zorder=5,
        label="Accepted step",
    )
    if np.any(status == "rejected"):
        ax1.scatter(
            steps[status == "rejected"],
            rwp_att[status == "rejected"],
            color="#dc2626",
            marker="x",
            s=60,
            lw=2,
            zorder=5,
            label="Rejected attempt",
        )

    ax1.set_xlabel("Refinement Step", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"$R_{\mathrm{wp}}$ [%]", fontsize=11, fontweight="bold")
    ax1.set_title("Full Refinement History", fontsize=12, fontweight="bold", pad=10)
    tick_step = 2 if len(steps) > 8 else 1
    ax1.set_xticks(range(1, len(steps) + 1, tick_step))
    ax1.set_xlim(0.5, len(steps) + 0.5)
    ax1.set_ylim(y_min_val, y_max_val)
    ax1.grid(True, linestyle="--", alpha=0.4, color="#cbd5e1")
    ax1.legend(
        loc="center right",
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.95,
    )

    # --- Panel 2: Detailed Micro-refinements ---
    effective_min_step = min_step_detail if len(steps) >= min_step_detail + 2 else 1
    sub_mask = steps >= effective_min_step
    s_steps = steps[sub_mask]
    s_rwp = rwp[sub_mask]
    s_att = rwp_att[sub_mask]
    s_status = status[sub_mask]
    s_labels = [step_labels[int(s) - 1] for s in s_steps]

    if split_step is not None and effective_min_step < split_step:
        ax2.axvspan(
            effective_min_step - 0.5,
            split_step,
            color="#f1f5f9",
            alpha=0.8,
            zorder=1,
        )
        ax2.axvspan(
            split_step,
            len(steps) + 0.5,
            color="#fefce8",
            alpha=0.6,
            zorder=1,
        )
        ax2.axvline(split_step, color="#94a3b8", linestyle="--", lw=1, zorder=2)

    ax2.plot(s_steps, s_rwp, color="#0284c7", lw=2.2, zorder=3)
    acc_m = s_status == "accepted"
    rej_m = s_status == "rejected"

    ax2.scatter(
        s_steps[acc_m],
        s_rwp[acc_m],
        color="#16a34a",
        edgecolor="white",
        s=65,
        lw=1.2,
        zorder=5,
        label="Accepted",
    )
    if np.any(rej_m):
        ax2.scatter(
            s_steps[rej_m],
            s_att[rej_m],
            color="#dc2626",
            marker="x",
            s=75,
            lw=2,
            zorder=5,
            label="Rejected attempt",
        )
        for s, att, acc in zip(s_steps[rej_m], s_att[rej_m], s_rwp[rej_m]):
            ax2.plot(
                [s, s], [acc, att], color="#dc2626", linestyle=":", lw=1.4, zorder=4
            )

    ax2.set_xticks(s_steps)
    ax2.set_xticklabels(
        [f"{s}\n{lbl}" for s, lbl in zip(s_steps, s_labels)],
        fontsize=8.5,
        rotation=30,
        ha="right",
    )
    ax2.set_xlabel("Step & Refined Parameter", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"$R_{\mathrm{wp}}$ [%]", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Detailed Convergence & Rollbacks (Steps {s_steps[0]}–{s_steps[-1]})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlim(effective_min_step - 0.5, len(steps) + 0.5)

    detail_values = np.concatenate((s_rwp, s_att))
    detail_values = detail_values[np.isfinite(detail_values)]
    detail_min = float(detail_values.min()) if detail_values.size else 0.
    detail_max = float(detail_values.max()) if detail_values.size else 1.
    ax2.set_ylim(max(0, detail_min - 0.25), detail_max + 0.35)
    ax2.grid(True, linestyle="--", alpha=0.4, color="#cbd5e1")

    if has_stage_split:
        switch_step = steps[split_idx]
        if switch_step in s_steps:
            rwp_switch = rwp[split_idx]
            ax2.annotate(
                "Switch to Rietveld\n(Le Bail = False)",
                xy=(switch_step, rwp_switch),
                xytext=(switch_step + 1.0, min(detail_max + 0.15, rwp_switch + 0.15)),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2),
                fontsize=9,
                fontweight="normal",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#fefce8",
                    edgecolor="#ca8a04",
                    alpha=0.9,
                ),
            )

    ax2.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.95,
    )

    if fig.get_layout_engine() is None:
        fig.tight_layout()

    if export_fig_as is not None:
        fig.savefig(export_fig_as, dpi=dpi)

    if show:
        plt.show()

    if return_fig:
        return fig, (ax1, ax2)


__all__ = [
    "i1d_plotter",
    "i2d_plotter",
    "phases_plotter",
    "exrd_plotter",
    "refinement_plotter",
    "_calc_reflection_pattern",
]
