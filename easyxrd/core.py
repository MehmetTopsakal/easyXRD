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
        verbose=False,
        figsize=(8, 6),
        i2d_robust=True,
        i1d_ylogscale=True,
        i2d_logscale=True,
    ):
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
        site_str_x=0.4,
        site_str_y=0.8,
        show_wt_fractions=False,
    ):
        """Plot XRD data, baselines, phases, and refinement results."""

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
        )


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
