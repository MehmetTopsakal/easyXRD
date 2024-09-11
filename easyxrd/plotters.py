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
warnings.filterwarnings("ignore", category=UserWarning)



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




def i1d_plotter(ds,
                ax,
                ds_previous=None,
                i1d_ylogscale=True,
                xlabel=True,
                return_da = False,
                title_str='',
                ncol = 2
                ):
    

    # label_x = 0.85,
    # label_y = 0.8,
    # label_y_shift = -0.2,
    yshift_multiplier = 0.01


    # if 'i1d_baseline' in ds.keys():
    #     if 'normalized_to' in ds.i1d.attrs:
    #         da_Y_obs = ds.i1d-ds.i1d_baseline+yshift_multiplier*ds.i1d.attrs['normalized_to']
    #         if 'i1d_refined' in ds.keys():
    #             da_Y_calc= ds.i1d_refined-ds.i1d_baseline+yshift_multiplier*ds.i1d.attrs['normalized_to']
    #             da_Y_bkg = ds.i1d_gsas_background+yshift_multiplier*ds.i1d.attrs['normalized_to']
    #     else:
    #         da_Y_obs = ds.i1d-ds.i1d_baseline
    #         if 'i1d_refined' in ds.keys():
    #             da_Y_calc= ds.i1d_refined-ds.i1d_baseline
    #             da_Y_bkg = ds.i1d_gsas_background
    # else:
    #     if 'normalized_to' in ds.i1d.attrs:
    #         da_Y_obs = ds.i1d+yshift_multiplier*ds.i1d.attrs['normalized_to']
    #         if 'i1d_refined' in ds.keys():
    #             da_Y_calc= ds.i1d_refined+yshift_multiplier*ds.i1d.attrs['normalized_to']
    #             da_Y_bkg = ds.i1d_gsas_background+yshift_multiplier*ds.i1d.attrs['normalized_to']
    #     else:
    #         da_Y_obs = ds.i1d
    #         if 'i1d_refined' in ds.keys():
    #             da_Y_calc= ds.i1d_refined
    #             da_Y_bkg = ds.i1d_gsas_background


    if 'i1d_baseline' in ds.keys():
        da_Y_obs = ds.i1d-ds.i1d_baseline
        if 'i1d_refined' in ds.keys():
            da_Y_calc= ds.i1d_refined-ds.i1d_baseline
            da_Y_bkg = ds.i1d_gsas_background
        ylabel = 'i1d - i1d_baseline'
    else:
        da_Y_obs = ds.i1d
        if 'i1d_refined' in ds.keys():
            da_Y_calc= ds.i1d_refined
            da_Y_bkg = ds.i1d_gsas_background
        ylabel = 'i1d'

    if i1d_ylogscale:
        # check negative data boundaries
        if 'i1d_refined' in ds.keys():
            min_obs = min(da_Y_obs.values)
            min_calc = min(da_Y_calc.values)
            min_bkg = min(da_Y_bkg.values)
            min_all = min([min_obs,min_calc,min_bkg])
        else:
            min_all = min(da_Y_obs.values)

        if min_all < 1:
            extra_yshift = 1-min_all
            ylabel = '%s +%.2f'%(ylabel,extra_yshift)
        else:
            extra_yshift = 0

        da_Y_obs = da_Y_obs+extra_yshift
        if 'i1d_refined' in ds.keys():
            da_Y_calc = da_Y_calc+extra_yshift
            da_Y_bkg  = da_Y_bkg+extra_yshift

        da_Y_obs =  np.log(da_Y_obs)
        if 'i1d_refined' in ds.keys():
            da_Y_calc=  np.log(da_Y_calc)
            da_Y_bkg =  np.log(da_Y_bkg)

        ylabel = 'Log$_{10}$(%s)'%ylabel
    else:
        extra_yshift = 0



    da_Y_obs.plot(ax=ax,color='k',linewidth=2, label='Y$_{obs.}$')
    if 'i1d_refined' in ds.keys():
        da_Y_calc.plot(ax=ax, alpha=0.9, linewidth=1, color='y',label='Y$_{calc.}$')
        da_Y_bkg.plot(ax=ax, alpha=0.9, linewidth=1, color='r',label='Y$_{bkg.}$')

    if ds_previous is not None:
        if 'i1d_baseline' in ds_previous.keys():
            if 'i1d_refined' in ds_previous.keys():
                da_Y_calc_previous= ds_previous.i1d_refined-ds_previous.i1d_baseline
                da_Y_bkg_previous = ds_previous.i1d_gsas_background
        else:
            if 'i1d_refined' in ds_previous.keys():
                da_Y_calc_previous= ds_previous.i1d_refined
                da_Y_bkg_previous = ds_previous.i1d_gsas_background

        if 'i1d_refined' in ds.keys():
            da_Y_calc_previous= da_Y_calc_previous+extra_yshift
            da_Y_bkg_previous  = da_Y_bkg_previous+extra_yshift

        if i1d_ylogscale:
            da_Y_calc_previous=  np.log(da_Y_calc_previous)
            da_Y_bkg_previous =  np.log(da_Y_bkg_previous)

        da_Y_calc_previous.plot(ax=ax, alpha=0.9, linewidth=1.2, linestyle='--', color='y',label='Y$_{calc.}$ (old)')
        da_Y_bkg_previous.plot(ax=ax, alpha=0.9, linewidth=1.2, linestyle='--', color='r',label='Y$_{bkg.}$ (old)')




    # if i1d_ylogscale:
    #     if 'i1d_baseline' in ds.keys():
    #         if 'normalized_to' in ds.i1d.attrs:
    #             ax.set_ylabel('Log$_{10}$(i1d-i1d_baseline+%.3f) (a.u.)'%(extra_yshift+yshift_multiplier*ds.i1d.attrs['normalized_to']),fontsize=8)
    #         else:
    #             if extra_yshift > 0:
    #                 ax.set_ylabel('Log$_{10}$(i1d-i1d_baseline+%.3f) (a.u.)'%extra_yshift,fontsize=8)
    #             else:
    #                 ax.set_ylabel('Log$_{10}$(i1d-i1d_baseline) (a.u.)',fontsize=8)
    #     else:
    #         if 'normalized_to' in ds.i1d.attrs:
    #             if extra_yshift > 0:
    #                 ax.set_ylabel('Log$_{10}$(i1d+%.3f) (a.u.)'%(extra_yshift),fontsize=8)  
    #             else:
    #                 ax.set_ylabel('Log$_{10}$(i1d) (a.u.)',fontsize=8)  
    #         else:
    #             if extra_yshift > 0:
    #                 ax.set_ylabel('Log$_{10}$(i1d+%.3f) (a.u.)'%extra_yshift,fontsize=8)
    #             else:
    #                 ax.set_ylabel('Log$_{10}$(i1d) (a.u.)',fontsize=8)
    # else:
    #     if 'i1d_baseline' in ds.keys():
    #         if 'normalized_to' in ds.i1d.attrs:
    #             ax.set_ylabel('(i1d-i1d_baseline+%.3f) (a.u.)'%(extra_yshift+yshift_multiplier*ds.i1d.attrs['normalized_to']),fontsize=8)
    #         else:
    #             if extra_yshift > 0:
    #                 ax.set_ylabel('(i1d-i1d_baseline+%.3f) (a.u.)'%extra_yshift,fontsize=8)
    #             else:
    #                 ax.set_ylabel('(i1d-i1d_baseline) (a.u.)',fontsize=8)
    #     else:
    #         if 'normalized_to' in ds.i1d.attrs:
    #             if extra_yshift > 0:
    #                 ax.set_ylabel('(i1d+%.3f) (a.u.)'%(extra_yshift),fontsize=8)  
    #             else:
    #                 ax.set_ylabel('(i1d) (a.u.)',fontsize=8)
    #         else:
    #             if extra_yshift > 0:
    #                 ax.set_ylabel('(i1d+%.3f) (a.u.)'%extra_yshift,fontsize=8)
    #             else:
    #                 ax.set_ylabel('(i1d) (a.u.)',fontsize=8)


    ax.set_ylabel(ylabel)
    ax.set_title(title_str,fontsize=8,color='r')


    ax.legend(loc='upper right',fontsize=8,ncol=ncol)
    ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])

    if xlabel:
        ax.set_xlabel(ds.i1d.attrs['xlabel'])
    else:
        ax.set_xlabel(None)

    if return_da:
        return [da_Y_obs,da_Y_calc,da_Y_bkg]




def i2d_plotter(ds,
                ax,
                i2d_robust=True,
                i2d_logscale=True,
                xlabel=False,
                cbar=True,
                cmap='Greys',
                title_str='',
                annotate=False,
                ):

    if ('i2d_baseline' in ds.keys()) and ('roi_azimuthal_range' in ds.i2d.attrs):
        da_i2d = (ds.i2d)
        vmin = 0
        i2d_str = 'i2d'
    elif ('i2d_baseline' in ds.keys()):
        da_i2d = (ds.i2d-ds.i2d_baseline)
        vmin = 0
        i2d_str = 'i2d-i2d_baseline'

    else:
        da_i2d = (ds.i2d)
        vmin = 0
        i2d_str = 'i2d'

    if i2d_logscale:
        da_i2d = np.log(da_i2d)
        vmin = 1
        i2d_str = 'Log$_{10}$(%s)'%(i2d_str)

    if cbar:
        da_i2d.plot.imshow(ax=ax,
                            robust=i2d_robust,
                            add_colorbar=cbar,
                            cbar_kwargs=dict(orientation="vertical",
                                            pad=0.02,
                                            shrink=0.8,
                                            label=None
                                            ),
                            cmap=cmap,
                            vmin=vmin
                                            )
    else:
        da_i2d.plot.imshow(ax=ax,
                            robust=i2d_robust,
                            add_colorbar=cbar,
                            cmap=cmap,
                            vmin=vmin
                            )


    if annotate:
        ax.annotate('%s (i2d_robust=%s)'%(i2d_str, i2d_robust),
                    xy=(0.01,0.93),
                    xycoords='axes fraction',
                    xytext=(0,0),
                    textcoords='offset points',
                    color='r',fontsize=8,
                    rotation=0,
                ) 



    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_facecolor('#FFF7D9')
    if xlabel:
        ax.set_xlabel(ds.i2d.attrs['xlabel'])
    else:
        ax.set_xlabel(None)


    if ('roi_azimuthal_range' in ds.i2d.attrs) and ('i1d' in ds.keys()):
        roi_xy = [ds.i1d.radial.values[0],ds.i1d.attrs['roi_azimuthal_range'][0]]
        roi_width = ds.i1d.radial.values[-1] - ds.i1d.radial.values[0]
        roi_height = ds.i1d.attrs['roi_azimuthal_range'][1] - ds.i1d.attrs['roi_azimuthal_range'][0]
        rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
        ax.add_patch(rect)

    ax.set_title(title_str,fontsize=8,color='r')








def phases_plotter(
        ds,
        ax_main,
        phases=None,
        line_axes=[],
        phase_label_x=0.9,
        phase_label_y=0.8,
        phase_label_yshift=-0.2,
        ):

        xrdc = XRDCalculator(wavelength=ds.i1d.attrs['wavelength_in_angst'])



        if phases is not None:
            for e,st in enumerate(phases):
                ps = xrdc.get_pattern(phases[st],
                                    scaled=True,
                                    two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([ds.i1d.radial.values[0],ds.i1d.radial.values[-1]]) * ( (ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) )
                                    )
                refl_X, refl_Y = ((4 * np.pi) / (ds.i1d.attrs['wavelength_in_angst'])) * np.sin(np.deg2rad(ps.x) / 2), ps.y
                for i in refl_X:
                    ax_main.axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
                    for a in line_axes:
                        a.axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)

                markerline, stemlines, stem_baseline = ax_main.stem(refl_X,refl_Y,markerfmt=".")
                plt.setp(stemlines, linewidth=0.5, color='C%d'%e)
                plt.setp(markerline, color='C%d'%e)

                ax_main.text(phase_label_x,phase_label_y+e*phase_label_yshift,st,color='C%d'%e,transform=ax_main.transAxes)

        else:
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
                    ax_main.axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
                    for a in line_axes:
                        a.axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)

                markerline, stemlines, stem_baseline = ax_main.stem(refl_X,refl_Y,markerfmt=".")
                plt.setp(stemlines, linewidth=0.5, color='C%d'%e)
                plt.setp(markerline, color='C%d'%e)

                ax_main.text(phase_label_x,phase_label_y+e*phase_label_yshift,st,color='C%d'%e,transform=ax_main.transAxes)



        ax_main.set_xlabel(ds.i1d.attrs['xlabel'])
        ax_main.set_ylim(bottom=1,top=120)



















def exrd_plotter(ds, 
                 ds_previous=None, 
                 phases=None,
                 gpx=None, 
                 gpx_previous=None, 
                 figsize=(8,6), 
                 i2d_robust=True,
                 i2d_logscale=True,
                 i1d_ylogscale=True,
                 plot_hint = '1st_loaded_data', 
                 title_str=None,
                 export_fig_as = None,
                 i1d_plot_radial_range = None,
                 i1d_plot_bottom = None,
                 i1d_plot_top = None,
                 title = None,
                 ):







#############################################################################
#############################################################################
#############################################################################
    if plot_hint == 'load_xrd_data':
        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=figsize,dpi=128)
            mosaic = """
                        B
                        C
                        C
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["B"]

            i2d_plotter(ds,ax,cbar=True,i2d_robust=i2d_robust,i2d_logscale=i2d_logscale,annotate=True)


            ax =  ax_dict["C"]
            # np.log(ds.i2d.mean(dim='azimuthal_i2d')).plot(ax=ax,color='k')
            i1d_plotter(ds,
                ax,
                ds_previous=None,
                i1d_ylogscale=i1d_ylogscale,
                xlabel=True,
                return_da = False,
                title_str=''
                )
            
            # ax.set_xlabel(ds.i1d.attrs['xlabel'])
            # ax.set_ylabel(ds.i1d.attrs['ylabel'])
            # ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')


        else:
            fig = plt.figure(figsize=figsize,dpi=128)
            mosaic = """
                        C
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

            ax = ax_dict["C"]
            # np.log(ds.i1d).plot(ax=ax,color='k')
            i1d_plotter(ds,
                ax,
                ds_previous=None,
                i1d_ylogscale=i1d_ylogscale,
                xlabel=True,
                return_da = False,
                title_str=''
                )




#############################################################################
#############################################################################
#############################################################################
    elif plot_hint == 'get_baseline':
        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=figsize,dpi=128)
            mosaic = """
                        AA222
                        AA222
                        AA111
                        AA111
                        AA111
                        AA111
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["A"]
            ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])
        else:
            fig = plt.figure(figsize=figsize,dpi=128)
            mosaic = """
                        AA111
                        AA111
                        AA111
                        AA111
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["A"]
            ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])
        


        if ('normalized_to' in ds.i1d.attrs):
            (ds.i1d).plot(ax=ax,label='i1d (norm.)')
        else:
            (ds.i1d).plot(ax=ax,label='i1d')
        if ('i1d_baseline' in ds.keys()) and ('normalized_to' in ds.i1d.attrs):
            (ds.i1d_baseline).plot(ax=ax,label='i1d_baseline (norm.)')
        elif ('i1d_baseline' in ds.keys()):
            (ds.i1d_baseline).plot(ax=ax,label='i1d_baseline')

        ax.set_yscale('log')
        ax.set_xlabel(ds.i1d.attrs['xlabel'])
        ax.set_ylabel('Intensity (a.u.)')
        ax.legend(fontsize=6)
        

        if 'i2d' in ds.keys():
            ax = ax_dict["2"]
            i2d_plotter(ds,ax,cbar=True,i2d_robust=i2d_robust,i2d_logscale=i2d_logscale,annotate=True)


        ax = ax_dict["1"]
        i1d_plotter(ds,
                        ax,
                        ds_previous=None,
                        i1d_ylogscale=i1d_ylogscale,
                        xlabel=True,
                        return_da = False,
                        title_str=''
                        )




#############################################################################
#############################################################################
#############################################################################
    elif plot_hint == 'load_phases':

        # plot_label_x = 0.9,
        # plot_label_y = 0.8,
        # plot_label_y_shift = -0.2

        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=figsize,dpi=128)
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
            ax = ax_dict["1"]
            i1d_plotter(ds,
                            ax,
                            ds_previous=None,
                            i1d_ylogscale=i1d_ylogscale,
                            xlabel=True,
                            return_da = False,
                            title_str=''
                            )
            ax.set_xlabel(None)
            phases_plotter(ds,
                        ax_main=ax_dict["P"],
                        phases=phases,
                        line_axes=[ax_dict["1"],ax_dict["2"],ax_dict["P"]],
                        )
            ax = ax_dict["2"]
            i2d_plotter(ds,
                        ax,
                        cbar=True,
                        i2d_robust=i2d_robust,
                        i2d_logscale=i2d_logscale,
                        annotate=True
                        )


        else:
            fig = plt.figure(figsize=figsize,dpi=128)
            mosaic = """
                        1
                        1
                        1
                        1
                        P
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["1"]
            i1d_plotter(ds,
                            ax,
                            ds_previous=None,
                            i1d_ylogscale=i1d_ylogscale,
                            xlabel=True,
                            return_da = False,
                            title_str=''
                            )
            ax.set_xlabel(None)
            phases_plotter(ds,
                        ax_main=ax_dict["P"],
                        phases=phases,
                        line_axes=[ax_dict["1"],ax_dict["P"]],
                        )




        # 
        












#############################################################################
#############################################################################
#############################################################################
    elif plot_hint == 'plot-type-0':
        fig = plt.figure(figsize=figsize,dpi=128)



        if 'i2d' in ds.keys():
            mosaic = """
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

            i2d_plotter(ds,ax=ax_dict["2"],
                        cbar=False,
                        i2d_robust=i2d_robust,
                        i2d_logscale=i2d_logscale,
                        title_str=title_str,
                        annotate=False,
                        )
            ax_dict["2"].set_title(title)
            ax_dict["2"].set_yticks([])  

            phases_plotter(ds,
                        ax_main=ax_dict["P"],
                        phases=phases,
                        line_axes=[ax_dict["2"],ax_dict["1"],ax_dict["D"],ax_dict["P"]],
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

            phases_plotter(ds,
                        ax_main=ax_dict["P"],
                        phases=phases,
                        line_axes=[ax_dict["1"],ax_dict["D"],ax_dict["P"]],
                        )
            ax_dict["P"].set_yticks([])




        [da_Y_obs,da_Y_calc,da_Y_bkg] = i1d_plotter(ds,
                    ax=ax_dict["1"],
                    ds_previous=None,
                    xlabel=False,
                    i1d_ylogscale=i1d_ylogscale,
                    return_da = True,
                    ncol = 1
                    )

        

        fit_str = 'Rwp/GoF = %.3f/%.3f'%(ds.attrs['Rwp'],ds.attrs['GOF'])
        ax_dict["1"].annotate('%s'%(fit_str),
                xy=(0.4,0.95),
                xycoords='axes fraction',
                xytext=(0,0),
                textcoords='offset points',
                color='k',fontsize=10,
                rotation=0,
            )        

        for e,si in enumerate(range(ds.attrs['num_phases'])):
            site_ind  =  si

            site_label = ds.attrs['PhaseInd_%d_label'%site_ind]
            site_SGSys = ds.attrs['PhaseInd_%d_SGSys'%site_ind]
            site_SpGrp = ds.attrs['PhaseInd_%d_SpGrp'%site_ind]

            site_a = ds.attrs['PhaseInd_%d_cell_a'%site_ind]
            site_b = ds.attrs['PhaseInd_%d_cell_b'%site_ind]
            site_c = ds.attrs['PhaseInd_%d_cell_c'%site_ind]
            site_alpha = ds.attrs['PhaseInd_%d_cell_alpha'%site_ind]
            site_beta = ds.attrs['PhaseInd_%d_cell_beta'%site_ind]
            site_gamma = ds.attrs['PhaseInd_%d_cell_gamma'%site_ind]


            site_size_broadening_type = ds.attrs['PhaseInd_%d_size_broadening_type'%site_ind]
            site_size0 = ds.attrs['PhaseInd_%d_size_0'%site_ind]       
            site_size1 = ds.attrs['PhaseInd_%d_size_1'%site_ind] 
            site_size2 = ds.attrs['PhaseInd_%d_size_2'%site_ind] 
            if site_size0 == 1.0:
                size_str = ''
            else:
                size_str = 'Size: %.3f$\\mu$ (%s) '%(site_size0,site_size_broadening_type)
            site_strain_broadening_type = ds.attrs['PhaseInd_%d_size_broadening_type'%site_ind]
            site_mustrain0 = ds.attrs['PhaseInd_%d_mustrain_0'%site_ind]       
            site_mustrain1 = ds.attrs['PhaseInd_%d_mustrain_1'%site_ind] 
            site_mustrain2 = ds.attrs['PhaseInd_%d_mustrain_2'%site_ind] 
            if site_mustrain0 == 1000.0:
                strain_str = ''
            else:
                strain_str = '| Strain: %.3f (%s)'%(site_mustrain0,site_strain_broadening_type)

            site_wt_fraction = ds.attrs['PhaseInd_%d_wt_fraction'%site_ind]



            site_str = '\n%s-phase:   %s (%s)  | weight_fraction=%.3f \nLattice: a/b/c=%.4f/%.4f/%.4f ($\\alpha$/$\\beta$/$\\gamma$=%.2f/%.2f/%.2f) \n%s %s'%(
                site_label,
                site_SpGrp.replace(' ',''),
                site_SGSys,
                site_wt_fraction,
                site_a,
                site_b,
                site_c,
                site_alpha,
                site_beta,
                site_gamma,
                size_str,
                strain_str         
                )
            


            ax_dict["1"].annotate('%s'%(site_str),
                        xy=(0.4,0.75-0.15*e),
                        xycoords='axes fraction',
                        xytext=(0,0),
                        textcoords='offset points',
                        color='C%d'%e,fontsize=8,
                        rotation=0,
                    ) 



        (da_Y_obs-da_Y_calc).plot(ax=ax_dict["D"],color='b')
        ax_dict["D"].axhline(y=0,linestyle='--',color='k',lw=0.5)


        ax_dict["D"].set_xlabel(None)



        if i1d_plot_radial_range is not None:
            ax_dict["1"].set_xlim([i1d_plot_radial_range[0],i1d_plot_radial_range[-1]])
            ax_dict["2"].set_xlim([i1d_plot_radial_range[0],i1d_plot_radial_range[-1]])
            ax_dict["P"].set_xlim([i1d_plot_radial_range[0],i1d_plot_radial_range[-1]])
            ax_dict["D"].set_xlim([i1d_plot_radial_range[0],i1d_plot_radial_range[-1]])


        ax_dict["1"].set_ylim(bottom = i1d_plot_bottom, top = i1d_plot_top)



        if export_fig_as is not None:
            plt.savefig(export_fig_as)








#############################################################################
#############################################################################
#############################################################################
    else:
        
        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=figsize,dpi=128)
            mosaic = """
                        2
                        1
                        1
                        1
                        1
                        P
                    """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

            i2d_plotter(ds,ax=ax_dict["2"],
                        cbar=False,
                        i2d_robust=i2d_robust,
                        i2d_logscale=i2d_logscale,
                        title_str=title_str,
                        annotate=False,
                        
                        )
            i1d_plotter(ds,
                        ax=ax_dict["1"],
                        ds_previous=ds_previous,
                        xlabel=False,
                        i1d_ylogscale=i1d_ylogscale,
                        return_da = False,
                        )
            phases_plotter(ds,
                        ax_main=ax_dict["P"],
                        phases=phases,
                        line_axes=[ax_dict["2"],ax_dict["1"],ax_dict["P"]],
                        )


        else:
            fig = plt.figure(figsize=figsize,dpi=128)
            mosaic = """
                        1
                        1
                        1
                        1
                        P
                    """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

            i1d_plotter(ds,
                        ax=ax_dict["1"],
                        ds_previous=ds_previous,
                        xlabel=False,
                        i1d_ylogscale=i1d_ylogscale,
                        return_da = False,
                        )
            ax_dict["1"].set_title(title_str)
            phases_plotter(ds,
                        ax_main=ax_dict["P"],
                        phases=phases,
                        line_axes=[ax_dict["1"],ax_dict["P"]],
                        )
























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
        # ax.set_ylabel('Log$_{10}$(data+10) (a.u.)')





        # wtSum = 0.0
        # for e,p in enumerate(self.phases):
        #     mass = self.gpx['Phases'][p]['General']['Mass']
        #     phFr = self.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][0]
        #     wtSum += mass*phFr
        # for e,p in enumerate(sample.phases):
            # weightFr = sample.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][0]*sample.gpx['Phases'][p]['General']['Mass']/wtSum


            # # weightFr = self.gpx['Phases'][st]['Histograms']['PWDR data.xy']['Scale'][0]*self.gpx['Phases'][st]['General']['Mass']/wtSum
            # # ax_dict["C"].text(label_x,label_y+e*label_y_shift,'%s (%.3f)'%(st,weightFr),color='C%d'%e,transform=ax_dict["C"].transAxes)
