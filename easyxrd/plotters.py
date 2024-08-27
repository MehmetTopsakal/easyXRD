from scipy.signal import savgol_filter
import pybaselines

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifWriter

from IPython.display import clear_output


import random, string
import fnmatch

import time
import copy

import os,sys


import numpy as np
import xarray as xr


import matplotlib 

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})













def ds_plotter(ds, gpx=None, phases=None, plot_hint = '1st_loaded_data'):


    if plot_hint == '1st_loaded_data':
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # ds.i1d.plot(ax=ax)
            # ax.set_xlabel(ds.i1d.attrs['xlabel'])
            # ax.set_ylabel(ds.i1d.attrs['ylabel'])
        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=(8,4),dpi=128)
            mosaic = """
                        B
                        C
                        C
                        C
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["B"]
            np.log(ds.i2d).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
            ax.set_xlabel(None)
            ax.set_ylabel('Azimuthal')

            try:
                roi_xy = [ds.i1d.radial.values[0],ds.i1d.attrs['roi_azimuthal_range'][0]]
                roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
                roi_height = ds.i1d.attrs['roi_azimuthal_range'][1] - ds.i1d.attrs['roi_azimuthal_range'][0]
                rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
                ax.add_patch(rect)
            except:
                pass

            ax =  ax_dict["C"]
            np.log(ds.i2d.mean(dim='azimuthal_i2d')).plot(ax=ax,color='k')
            ax.set_xlabel(ds.i2d.attrs['xlabel'])
            ax.set_ylabel(ds.i2d.attrs['ylabel'])


        else:
            fig = plt.figure(figsize=(8,5),dpi=128)
            mosaic = """
                        C
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

            ax = ax_dict["C"]
            np.log(ds.i1d).plot(ax=ax,color='k')
            ax.set_xlabel(ds.i1d.attrs['xlabel'])
            ax.set_ylabel(ds.i1d.attrs['ylabel'])


        










    if plot_hint == 'get_baseline':
        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=(8,6),dpi=128)
            mosaic = """
                        AABBB
                        AABBB
                        AACCC
                        AACCC
                        AACCC
                        AACCC
                        AADDD
                        """
        else:
            fig = plt.figure(figsize=(8,5),dpi=128)
            mosaic = """
                        AACCC
                        AACCC
                        AACCC
                        AACCC
                        AADDD
                        """
        ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

        ax = ax_dict["A"]
        (ds.i1d).plot(ax=ax,label='data')
        (ds.i1d_baseline).plot(ax=ax,label='baseline')
        ax.set_yscale('log')
        ax.set_xlabel(ds.i1d.attrs['xlabel'])
        ax.set_ylabel(ds.i1d.attrs['ylabel'])
        ax.legend()
        ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])

        if 'i2d' in ds.keys():
            ax = ax = ax_dict["B"]
            np.log(ds.i2d-ds.i2d_baseline+0.01*ds.i2d.attrs['normalized_to']).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys')
            ax.set_xlabel(None)
            ax.set_ylabel('Azimuthal')

            try:
                roi_xy = [ds.i1d.radial.values[0],ds.i1d.attrs['roi_azimuthal_range'][0]]
                roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
                roi_height = ds.i1d.attrs['roi_azimuthal_range'][1] - ds.i1d.attrs['roi_azimuthal_range'][0]
                rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
                ax.add_patch(rect)
            except:
                pass

        ax = ax = ax_dict["C"]
        np.log(ds.i1d-ds.i1d_baseline+0.01*ds.i2d.attrs['normalized_to']).plot(ax=ax,color='k')
        # ax.fill_between(ds.i1d.radial.values, ds.i1d.radial.values*0+np.log(0.01*ds.i1d.attrs['normalized_to']),alpha=0.2)
        ax.set_xlabel(None)
        ax.set_ylabel('Log$_{10}$(data-baseline+%d) (a.u.)'%(0.01*ds.i1d.attrs['normalized_to']))
        ax.set_ylim(bottom=np.log(0.01*0.9*ds.i2d.attrs['normalized_to']))
        
        ax = ax = ax_dict["D"]
        (ds.i1d-ds.i1d_baseline).plot(ax=ax,color='k')
        ax.axhline(y=0,alpha=0.5,color='y')
        ax.set_xlabel(ds.i1d.attrs['xlabel'])
        ax.set_ylim([-0.02,0.02])





    if plot_hint == 'load_phases':

        plot_label_x = 0.9,
        plot_label_y = 0.8,
        plot_label_y_shift = -0.2

        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=(8,6),dpi=128)
            mosaic = """
                        A
                        A
                        B
                        B
                        B
                        B
                        C
                        """
        else:
            fig = plt.figure(figsize=(8,4),dpi=128)
            mosaic = """
                        B
                        B
                        B
                        B
                        C
                        """

        ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

        if 'i2d' in ds.keys():
            ax = ax = ax_dict["A"]
            np.log(ds.i2d-ds.i2d_baseline+0.01*ds.i2d.attrs['normalized_to']).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys')
            ax.set_xlabel(None)
            ax.set_ylabel('Azimuthal')
            ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])
            

            try:
                roi_xy = [ds.i1d.radial.values[0],ds.i1d.attrs['roi_azimuthal_range'][0]]
                roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
                roi_height = ds.i1d.attrs['roi_azimuthal_range'][1] - ds.i1d.attrs['roi_azimuthal_range'][0]
                rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
                ax.add_patch(rect)
            except:
                pass

        ax = ax_dict["B"]
        np.log(ds.i1d-ds.i1d_baseline+0.01*ds.i2d.attrs['normalized_to']).plot(ax=ax,color='k')
        ax.fill_between(ds.i1d.radial.values, ds.i1d.radial.values*0+np.log(0.01*ds.i1d.attrs['normalized_to']),alpha=0.2)
        ax.set_xlabel(None)
        ax.set_ylabel('Log$_{10}$(data-baseline+%d) (a.u.)'%(0.01*ds.i1d.attrs['normalized_to']))
        ax.set_ylim(bottom=-0.02)
        ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])

        xrdc = XRDCalculator(wavelength=ds.i1d.attrs['wavelength_in_angst'])

        for e,st in enumerate(phases):
            ps = xrdc.get_pattern(phases[st],
                                scaled=True,
                                two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([ds.i1d.radial.values[0],ds.i1d.radial.values[-1]]) * ( (ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) )
                                )
            refl_X, refl_Y = ((4 * np.pi) / (ds.i1d.attrs['wavelength_in_angst'])) * np.sin(np.deg2rad(ps.x) / 2), ps.y

            for i in refl_X:
                if 'i2d' in ds.keys():
                    ax_dict["A"].axvline(x=i,lw=0.3,color='C%d'%e)
                ax_dict["B"].axvline(x=i,lw=0.3,color='C%d'%e)
                ax_dict["C"].axvline(x=i,lw=0.3,color='C%d'%e)

            markerline, stemlines, baseline = ax_dict["C"].stem(refl_X,refl_Y,markerfmt=".")
            ax_dict["C"].set_ylim(bottom=0)

            plt.setp(stemlines, linewidth=0.5, color='C%d'%e)
            plt.setp(markerline, color='C%d'%e)

            # print([label_x[0]])
            ax_dict["C"].text(plot_label_x[0],plot_label_y[0]+e*plot_label_y_shift,st,color='C%d'%e,transform=ax_dict["C"].transAxes)

        ax_dict["C"].set_xlabel(ds.i1d.attrs['xlabel'])
        ax_dict["C"].set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])

