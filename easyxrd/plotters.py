import warnings
import numpy as np
import matplotlib.patches
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator



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
        da_i2d = np.log(da_i2d.where(da_i2d > 0))
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
            refl_HKLs = ps.hkls
        else:
            refl_X, refl_Y, refl_HKLs = np.array([]), np.array([]), np.array([])
    except Exception:
        refl_X, refl_Y, refl_HKLs = np.array([]), np.array([]), np.array([])

    _PATTERN_CACHE[cache_key] = (refl_X, refl_Y, refl_HKLs)
    return refl_X, refl_Y, refl_HKLs


def phases_plotter(
    ds,
    ax_main,
    phases=None,
    line_axes=[],
    phase_label_x=0.9,
    phase_label_y=0.75,
    phase_label_yshift=-0.2,
    show_hkls = True,
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

    elif plot_hint in ("refinement_history", "plot_refinement_history", "refinement", "plot_refinement"):
        return refinement_plotter(
            ds,
            figsize=figsize,
            export_fig_as=export_fig_as,
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


def refinement_plotter(
    ds,
    figsize=(8, 6),
    dpi=150,
    min_step_detail=4,
    export_fig_as=None,
    show=False,
    return_fig=False,
):
    """
    Plot comprehensive refinement convergence history showing full progression
    and zoomed-in micro-refinements with accepted vs. rejected attempts.
    """
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
    lebail_off_indices = [i for i, l in enumerate(labels) if l == "set_LeBail"]
    has_stage_split = len(lebail_off_indices) > 0
    if has_stage_split:
        split_idx = lebail_off_indices[0]
        split_step = steps[split_idx] - 0.5
    else:
        split_step = None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, gridspec_kw={"height_ratios": [1, 1.35]}
    )

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

    detail_min = min(float(np.nanmin(s_rwp)), float(np.nanmin(s_att)))
    detail_max = max(float(np.nanmax(s_rwp)), float(np.nanmax(s_att)))
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
        plt.savefig(export_fig_as, dpi=dpi)

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
