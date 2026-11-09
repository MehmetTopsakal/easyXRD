from . import DEFAULT, resolve_default

import warnings
import matplotlib.pyplot as plt

from .utils import HiddenPrints
from .data_loader import DataLoaderMixin
from .baseline import BaselineMixin, _optimize_bkg_scale, _compute_i2d_baseline
from .phases import PhasesMixin
from .refinement import RefinementMixin
from .plotters import exrd_plotter



class exrd(DataLoaderMixin, BaselineMixin, PhasesMixin, RefinementMixin):
    """
    Main easyXRD analysis class coordinating data loading, baseline calculation,
    phase identification, GSAS-II refinement, and visualization.
    """

    def __init__(
        self,
        verbose=DEFAULT,
        figsize=DEFAULT,
        i2d_robust=DEFAULT,
        i1d_ylogscale=DEFAULT,
        i2d_logscale=DEFAULT,
    ):
        verbose = resolve_default(verbose, "verbose")
        figsize = resolve_default(figsize, "plot_figsize")
        i2d_robust = resolve_default(i2d_robust, "plot_i2d_robust")
        i1d_ylogscale = resolve_default(i1d_ylogscale, "plot_i1d_ylogscale")
        i2d_logscale = resolve_default(i2d_logscale, "plot_i2d_logscale")

        self.verbose = verbose
        self.figsize = figsize
        self.i2d_robust = i2d_robust
        self.i1d_ylogscale = i1d_ylogscale
        self.i2d_logscale = i2d_logscale

    def plot(
        self,
        plot_hint=None,
        figsize=None,
        i2d_robust=None,
        i2d_logscale=None,
        i1d_ylogscale=None,
        export_fig_as=None,
        i1d_plot_radial_range=None,
        i1d_plot_bottom=None,
        i1d_plot_top=None,
        title=None,
        site_str_x=DEFAULT,
        site_str_y=DEFAULT,
        close=DEFAULT,
        show_wt_fractions=DEFAULT,
        dpi=DEFAULT,
    ):
        """Plot XRD data, baselines, phases, and refinement results."""
        dpi = resolve_default(dpi, "plot_dpi")
        site_str_x = resolve_default(site_str_x, "plot_site_str_x")
        site_str_y = resolve_default(site_str_y, "plot_site_str_y")
        close = resolve_default(close, "plot_close")
        show_wt_fractions = resolve_default(show_wt_fractions, "plot_show_wt_fractions")

        if figsize is None:
            figsize = self.figsize

        if i2d_robust is None:
            i2d_robust = self.i2d_robust

        if i2d_logscale is None:
            i2d_logscale = self.i2d_logscale

        if i1d_ylogscale is None:
            i1d_ylogscale = self.i1d_ylogscale

        ds = getattr(self, "ds", None)
        ds_previous = getattr(self, "ds_previous", None)

        before = set(plt.get_fignums())
        exrd_plotter(
            ds=ds,
            ds_previous=ds_previous,
            figsize=figsize,
            i2d_robust=i2d_robust,
            i2d_logscale=i2d_logscale,
            i1d_ylogscale=i1d_ylogscale,
            plot_hint=plot_hint,
            export_fig_as=export_fig_as,
            i1d_plot_radial_range=i1d_plot_radial_range,
            i1d_plot_bottom=i1d_plot_bottom,
            i1d_plot_top=i1d_plot_top,
            title=title,
            site_str_x=site_str_x,
            site_str_y=site_str_y,
            show_wt_fractions=show_wt_fractions,
            dpi=dpi,
        )

        if close:
            for number in set(plt.get_fignums()) - before:
                plt.close(number)


__all__ = [
    "exrd",
    "HiddenPrints",
    "_optimize_bkg_scale",
    "_compute_i2d_baseline",
    "DataLoaderMixin",
    "BaselineMixin",
    "PhasesMixin",
    "RefinementMixin",
]
