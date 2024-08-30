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



import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



class HiddenPrints:
    """
    This class hides print outputs from functions. It is useful for processes like refinement which produce a lot of text prints.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout








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
            ax.set_facecolor('#FFF7D9')

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
                        AADDD
                        AADDD
                        AADDD
                        AAEEE
                        AAEEE
                        BBEEE
                        BBEEE
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["A"]
            ax.set_xlim([ds.i2d.radial_i2d[0],ds.i2d.radial_i2d[-1]])
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
            ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])
        

        
        (ds.i1d).plot(ax=ax,label='data')
        (ds.i1d_baseline).plot(ax=ax,label='baseline')
        ax.set_yscale('log')
        ax.set_xlabel(None)
        ax.set_ylabel('Normalized Intensity (a.u.)')
        ax.legend()
        

        if 'i2d' in ds.keys():


            ax = ax_dict["D"]
            (ds.i2d-ds.i2d_baseline).plot.imshow(ax=ax,
                                                 robust=True,
                                                 add_colorbar=True,
                                                 cbar_kwargs=dict(orientation="vertical", 
                                                                pad=0.02, 
                                                                shrink=0.8, 
                                                                label=None),
                                                                cmap='Greys',
                                                                vmin=0
                                                 )
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_facecolor('#FFF7D9')

            try:
                roi_xy = [ds.i1d.radial.values[0],ds.i1d.attrs['roi_azimuthal_range'][0]]
                roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
                roi_height = ds.i1d.attrs['roi_azimuthal_range'][1] - ds.i1d.attrs['roi_azimuthal_range'][0]
                rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
                ax.add_patch(rect)
            except:
                pass

        ax = ax_dict["E"]
        np.log(ds.i1d-ds.i1d_baseline+0.01*ds.i1d.attrs['normalized_to']).plot(ax=ax,color='k')
        # ax.fill_between(ds.i1d.radial.values, ds.i1d.radial.values*0+np.log(0.01*ds.i1d.attrs['normalized_to']),alpha=0.2)
        ax.set_xlabel(ds.i1d.attrs['xlabel'])
        ax.set_ylabel('Log$_{10}$(data-baseline+%d) (a.u.)'%(0.01*ds.i1d.attrs['normalized_to']))
        ax.set_ylim(bottom=np.log(0.01*0.9*ds.i2d.attrs['normalized_to']))
        
        ax = ax_dict["B"]
        (ds.i1d-ds.i1d_baseline).plot(ax=ax,color='k',label='data-baseline')
        ax.axhline(y=0,alpha=0.5,color='y')
        ax.set_xlabel(ds.i1d.attrs['xlabel'])
        ax.set_ylim([-0.01,0.1])
        ax.legend()
























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
            if 'i2d_baseline' in ds.keys():
                np.log(ds.i2d-ds.i2d_baseline+0.01*ds.i1d.attrs['normalized_to']).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
            else:
                np.log(ds.i2d).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
            ax.set_xlabel(None)
            ax.set_ylabel('Azimuthal')
            ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])
            ax.set_facecolor('#FFF7D9')
            

            try:
                roi_xy = [ds.i1d.radial.values[0],ds.i1d.attrs['roi_azimuthal_range'][0]]
                roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
                roi_height = ds.i1d.attrs['roi_azimuthal_range'][1] - ds.i1d.attrs['roi_azimuthal_range'][0]
                rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
                ax.add_patch(rect)
            except:
                pass

        ax = ax_dict["B"]
        np.log(ds.i1d-ds.i1d_baseline+0.01*ds.i1d.attrs['normalized_to']).plot(ax=ax,color='k')
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
        ax_dict["C"].set_ylim(bottom=1,top=120)
        ax_dict["C"].set_yscale('log')


















    if plot_hint == 'ds_with_refinement_info':


        label_x = 0.85,
        label_y = 0.8,
        label_y_shift = -0.2,
        ylogscale=True
        yshift_multiplier = 0.01




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
            if ylogscale:
                np.log(ds.i2d-ds.i2d_baseline+yshift_multiplier*ds.i2d.attrs['normalized_to']).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=yshift_multiplier*ds.i2d.attrs['normalized_to'])
            else:
                (ds.i2d-ds.i2d_baseline+yshift_multiplier*ds.i2d.attrs['normalized_to']).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=yshift_multiplier*ds.i2d.attrs['normalized_to'])
            ax.set_xlabel(None)
            ax.set_ylabel('Azimuthal')
            ax.set_facecolor('#FFF7D9')

            try:
                roi_xy = [ds.i1d.radial.values[0],ds.i1d.attrs['roi_azimuthal_range'][0]]
                roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
                roi_height = ds.i1d.attrs['roi_azimuthal_range'][1] - ds.i1d.attrs['roi_azimuthal_range'][0]
                rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
                ax.add_patch(rect)
            except:
                pass

        ax = ax_dict["B"]
        if ylogscale:
            np.log(ds.i1d-ds.i1d_baseline+yshift_multiplier*ds.i2d.attrs['normalized_to']).plot(ax=ax,color='k',label='Yobs.')
            np.log(ds.i1d_refined+yshift_multiplier*ds.i2d.attrs['normalized_to']).plot(ax=ax, alpha=0.9, linewidth=1, color='y',label='Ycalc. (Rwp=%.3f,GoF=%.3f)'%(ds.attrs['Rwp'],ds.attrs['GOF'])) 
            np.log(ds.i1d_gsas_background+yshift_multiplier*ds.i2d.attrs['normalized_to']).plot(ax=ax, alpha=0.9, linewidth=1, color='r',label='Ybkg.')  
            # ax.fill_between(ds.i1d.radial.values,
            #                 ds.i1d.radial.values*0+np.log(yshift_multiplier*ds.i2d.attrs['normalized_to']),
            #                 alpha=0.2,
            #                 color='C7',
            #                 )
            
            # ax.fill_between(ds.i1d.radial.values, 
            #                 y1=np.log((yshift_multiplier*ds.i2d.attrs['normalized_to']+ds.i1d_gsas_background).values),
            #                 y2=np.log((yshift_multiplier*ds.i2d.attrs['normalized_to'])),
            #                 alpha=0.2,
            #                 color='C9',
            #                 label='Ybkg.'
            #                 )
            ax.set_ylabel('Log$_{10}$(data+10) (a.u.)')

        else:
            (ds.i1d-ds.i1d_baseline+yshift_multiplier*ds.i2d.attrs['normalized_to']).plot(ax=ax,color='k',label='Yobs.')
            (ds.i1d_refined).plot(ax=ax, alpha=0.9, linewidth=1, color='y',label='Ycalc. (Rwp=%.3f,GoF=%.3f)'%(ds.attrs['Rwp'],ds.attrs['GOF']))   
            ax.fill_between(ds.i1d.radial.values,
                            ds.i1d.radial.values*0+yshift_multiplier*ds.i2d.attrs['normalized_to'],
                            alpha=0.2,
                            color='C7',
                            )
            
            ax.fill_between(ds.i1d.radial.values, 
                            y1=(yshift_multiplier*ds.i2d.attrs['normalized_to']+ds.i1d_gsas_background).values,
                            y2=yshift_multiplier*ds.i2d.attrs['normalized_to'],
                            alpha=0.2,
                            color='C9',
                            label='Ybkg.'
                            )
            ax.set_ylabel('data+10 (a.u.)')





        ax.set_xlabel(None)


        ax.legend(loc='upper right',fontsize=8)
        ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])



        xrdc = XRDCalculator(wavelength=ds.i1d.attrs['wavelength_in_angst'])

        ds_phases = {}
        for a in ds.attrs.keys():
            for aa in range(ds.attrs['num_phases']):
                    if a == 'PhaseInd_%d_cif'%aa:
                        with open("tmp.cif", "w") as cif_file:
                            cif_file.write(ds.attrs[a])
                        st = Structure.from_file('tmp.cif')
                        ds_phases[ds.attrs['PhaseInd_%d_label'%aa]] = st
        os.remove('tmp.cif')
        for e,st in enumerate(ds_phases):
            ps = xrdc.get_pattern(ds_phases[st],
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
            plt.setp(stemlines, linewidth=0.5, color='C%d'%e)
            plt.setp(markerline, color='C%d'%e)



        ds_phases_refined = {}
        for a in ds.attrs.keys():
            for aa in range(ds.attrs['num_phases']):

                if a == 'PhaseInd_%d_refined_cif'%aa:
                    with open("tmp.cif", "w") as cif_file:
                        cif_file.write(ds.attrs[a])
                    st = Structure.from_file('tmp.cif')
                    ds_phases_refined[ds.attrs['PhaseInd_%d_label'%aa]] = st
        os.remove('tmp.cif')
        for e,st in enumerate(ds_phases_refined):
            ps = xrdc.get_pattern(ds_phases_refined[st],
                                scaled=True,
                                two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([ds.i1d.radial.values[0],ds.i1d.radial.values[-1]]) * ( (ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) )
                                )
            refl_X, refl_Y = ((4 * np.pi) / (ds.i1d.attrs['wavelength_in_angst'])) * np.sin(np.deg2rad(ps.x) / 2), ps.y
            for i in refl_X:
                if 'i2d' in ds.keys():
                    ax_dict["A"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
                ax_dict["B"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
                ax_dict["C"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
            markerline, stemlines, baseline = ax_dict["C"].stem(refl_X,refl_Y,markerfmt="+")
            plt.setp(stemlines, linewidth=0.5,linestyle='--', color='C%d'%e)
            plt.setp(markerline, color='C%d'%e)




        ax_dict["C"].set_xlabel(ds.i1d.attrs['xlabel'])
        ax_dict["C"].set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])
        ax_dict["C"].set_ylim(bottom=1,top=120)
        ax_dict["C"].set_yscale('log')





        # wtSum = 0.0
        # for e,p in enumerate(self.phases):
        #     mass = self.gpx['Phases'][p]['General']['Mass']
        #     phFr = self.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][0]
        #     wtSum += mass*phFr
        # for e,p in enumerate(sample.phases):
            # weightFr = sample.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][0]*sample.gpx['Phases'][p]['General']['Mass']/wtSum


            # # weightFr = self.gpx['Phases'][st]['Histograms']['PWDR data.xy']['Scale'][0]*self.gpx['Phases'][st]['General']['Mass']/wtSum
            # # ax_dict["C"].text(label_x,label_y+e*label_y_shift,'%s (%.3f)'%(st,weightFr),color='C%d'%e,transform=ax_dict["C"].transAxes)