import warnings
import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.max_open_warning": 0})


def i1d_plotter(
    ds,
    ax,
    ds_previous=None,
    i1d_ylogscale=True,
    xlabel=True,
    return_da=False,
    title_str="",
    ncol=1,
    show_Ybkg_old=False,
):
    """Plot 1D XRD intensity profile (observed, calculated, and background)."""

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
            min_obs = min(da_Y_obs.values)
            min_calc = min(da_Y_calc.values)
            min_bkg = min(da_Y_bkg.values)
            min_all = min([min_obs, min_calc, min_bkg])
        else:
            min_all = min(da_Y_obs.values)

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

    if ds_previous is not None:
        if "i1d_baseline" in ds_previous.keys():
            if "i1d_refined" in ds_previous.keys():
                da_Y_calc_previous = ds_previous.i1d_refined - ds_previous.i1d_baseline
                da_Y_bkg_previous = ds_previous.i1d_gsas_background
        else:
            if "i1d_refined" in ds_previous.keys():
                da_Y_calc_previous = ds_previous.i1d_refined
                da_Y_bkg_previous = ds_previous.i1d_gsas_background

        if "i1d_refined" in ds.keys():
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
    i2d_robust=True,
    i2d_logscale=True,
    xlabel=False,
    cbar=True,
    cmap="Greys",
    title_str="",
    annotate=False,
):
    """Plot 2D caked XRD pattern (azimuthal vs radial)."""

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
        da_i2d = np.log(da_i2d)
        vmin = 1
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


_PATTERN_CACHE = {}


def _calc_reflection_pattern(xrdc, st, wavelength, radial_min, radial_max):
    """Calculate or retrieve cached Bragg peak positions and intensities."""

    arg0 = np.clip(radial_min * (wavelength / (4 * np.pi)), -1.0, 1.0)
    arg1 = np.clip(radial_max * (wavelength / (4 * np.pi)), -1.0, 1.0)
    tth0 = float(np.rad2deg(2 * np.arcsin(arg0)))
    tth1 = float(np.rad2deg(2 * np.arcsin(arg1)))
    two_theta_range = (min(tth0, tth1), max(tth0, tth1))

    try:
        lat_tuple = tuple(np.round(st.lattice.matrix.flat, 4))
        comp_str = st.composition.reduced_formula
    except Exception:
        lat_tuple = ()
        comp_str = str(getattr(st, "formula", ""))

    cache_key = (
        comp_str,
        lat_tuple,
        round(float(wavelength), 6),
        round(two_theta_range[0], 4),
        round(two_theta_range[1], 4),
    )

    if cache_key in _PATTERN_CACHE:
        return _PATTERN_CACHE[cache_key]

    try:
        ps = xrdc.get_pattern(st, scaled=True, two_theta_range=two_theta_range)
        if len(ps.x) > 0:
            refl_X = ((4 * np.pi) / wavelength) * np.sin(np.deg2rad(ps.x) / 2)
            refl_Y = ps.y
        else:
            refl_X, refl_Y = np.array([]), np.array([])
    except Exception:
        refl_X, refl_Y = np.array([]), np.array([])

    _PATTERN_CACHE[cache_key] = (refl_X, refl_Y)
    return refl_X, refl_Y


def phases_plotter(
    ds,
    ax_main,
    phases=None,
    line_axes=[],
    phase_label_x=0.9,
    phase_label_y=0.75,
    phase_label_yshift=-0.2,
):
    """Plot Bragg reflection lines and phase labels."""

    wavelength = ds.i1d.attrs["wavelength_in_angst"]
    xrdc = XRDCalculator(wavelength=wavelength)

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
                    print(f"Warning: could not parse CIF for phase {aa}: {exc}")
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
        refl_X, refl_Y = _calc_reflection_pattern(
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

        ax_main.text(
            phase_label_x,
            phase_label_y + e * phase_label_yshift,
            st_name,
            color="C%d" % e,
            transform=ax_main.transAxes,
        )

    ax_main.set_xlabel(ds.i1d.attrs["xlabel"])
    ax_main.set_ylim(bottom=1, top=120)
    ax_main.set_yticks([])


def exrd_plotter(
    ds,
    ds_previous=None,
    phases=None,
    figsize=(8, 6),
    i2d_robust=True,
    i2d_logscale=True,
    i1d_ylogscale=True,
    plot_hint="1st_loaded_data",
    title_str=None,
    export_fig_as=None,
    i1d_plot_radial_range=None,
    i1d_plot_bottom=None,
    i1d_plot_top=None,
    title=None,
    site_str_x=0.4,
    site_str_y=0.8,
    show_wt_fractions=False,
):
    """Main plotting coordinator for easyXRD workflow stages."""

    has_i2d = "i2d" in ds.keys()

    if plot_hint == "load_xrd_data":
        fig = plt.figure(figsize=(figsize[0], figsize[1] / 1.5), dpi=128)
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
        fig = plt.figure(figsize=figsize, dpi=128)
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
            fig = plt.figure(figsize=(figsize[0], figsize[1]), dpi=128)
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
            fig = plt.figure(figsize=(figsize[0], figsize[1] / 1.5), dpi=128)
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

    elif plot_hint is None:
        fig = plt.figure(figsize=figsize, dpi=128)
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
                "PhaseInd_%d_size_broadening_type" % site_ind
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
            if site_wt_fraction == 100.0 or not show_wt_fractions:
                str_fraction = ""
            else:
                str_fraction = "| wt%%=%.2f" % site_wt_fraction

            site_str = (
                "\n%s %s phase (%s)  %s %s %s \nLattice: a/b/c=%.4f/%.4f/%.4f ($\\alpha$/$\\beta$/$\\gamma$=%.2f/%.2f/%.2f) \n"
                % (
                    site_SpGrp.replace(" ", ""),
                    site_SGSys,
                    site_label,
                    str_fraction,
                    size_str,
                    strain_str,
                    site_a,
                    site_b,
                    site_c,
                    site_alpha,
                    site_beta,
                    site_gamma,
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

        (da_Y_obs - da_Y_calc).plot(ax=ax_dict["D"], color="b")
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
            fig = plt.figure(figsize=(figsize[0], figsize[1]), dpi=128)
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
            )
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["2"], ax_dict["1"], ax_dict["P"]],
            )
        else:
            fig = plt.figure(figsize=(figsize[0], figsize[1] / 1.5), dpi=128)
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
            )
            ax_dict["1"].set_title(title_str, fontsize=8)
            phases_plotter(
                ds,
                ax_main=ax_dict["P"],
                phases=phases,
                line_axes=[ax_dict["1"], ax_dict["P"]],
            )

    if export_fig_as is not None:
        plt.savefig(export_fig_as, dpi=128)


__all__ = [
    "i1d_plotter",
    "i2d_plotter",
    "phases_plotter",
    "exrd_plotter",
    "_calc_reflection_pattern",
]
