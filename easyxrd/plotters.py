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
                ylogscale=True,
                xlabel=True,
                return_da = False
                ):
    

    label_x = 0.85,
    label_y = 0.8,
    label_y_shift = -0.2,
    yshift_multiplier = 0.01


    if 'i1d_baseline' in ds.keys():
        if 'normalized_to' in ds.i1d.attrs:
            da_Y_obs = ds.i1d-ds.i1d_baseline+yshift_multiplier*ds.i1d.attrs['normalized_to']
            da_Y_calc= ds.i1d_refined-ds.i1d_baseline+yshift_multiplier*ds.i1d.attrs['normalized_to']
            da_Y_bkg = ds.i1d_gsas_background+yshift_multiplier*ds.i1d.attrs['normalized_to']
        else:
            da_Y_obs = ds.i1d-ds.i1d_baseline+10
            da_Y_calc= ds.i1d_refined-ds.i1d_baseline+10
            da_Y_bkg = ds.i1d_gsas_background+10
    else:
        if 'normalized_to' in ds.i1d.attrs:
            da_Y_obs = ds.i1d+yshift_multiplier*ds.i1d.attrs['normalized_to']
            da_Y_calc= ds.i1d_refined+yshift_multiplier*ds.i1d.attrs['normalized_to']
            da_Y_bkg = ds.i1d_gsas_background+yshift_multiplier*ds.i1d.attrs['normalized_to']
        else:
            da_Y_obs = ds.i1d
            da_Y_calc= ds.i1d_refined
            da_Y_bkg = ds.i1d_gsas_background


    if ylogscale:
            da_Y_obs =  np.log(da_Y_obs )
            da_Y_calc=  np.log(da_Y_calc)
            da_Y_bkg =  np.log(da_Y_bkg )



    da_Y_obs.plot(ax=ax,color='k',label='Y$_{obs.}$')
    da_Y_calc.plot(ax=ax, alpha=0.9, linewidth=1, color='y',label='Y$_{calc.}$')
    da_Y_bkg.plot(ax=ax, alpha=0.9, linewidth=1, color='r',label='Y$_{bkg.}$')




    if ylogscale:
        if 'i1d_baseline' in ds.keys():
            if 'normalized_to' in ds.i1d.attrs:
                ax.set_ylabel('Log$_{10}$(i1d - i1d_baseline + %.f) (norm.)'%(0.01*ds.i1d.attrs['normalized_to']),fontsize=8)
            else:
                ax.set_ylabel('Log$_{10}$(i1d - i1d_baseline) (norm.)',fontsize=8)
        else:
            if 'normalized_to' in ds.i1d.attrs:
                ax.set_ylabel('Log$_{10}$(i1d + %.f) (a.u.)'%(0.01*ds.i1d.attrs['normalized_to']),fontsize=8)  
            else:
                ax.set_ylabel('Log$_{10}$(i1d) (a.u.)',fontsize=8)

    else:
        if 'i1d_baseline' in ds.keys():
            if 'normalized_to' in ds.i1d.attrs:
                ax.set_ylabel('(i1d - i1d_baseline + %.f) (norm.)'%(0.01*ds.i1d.attrs['normalized_to']),fontsize=8)
            else:
                ax.set_ylabel('(i1d - i1d_baseline) (a.u.)',fontsize=8)
        else:
            if 'normalized_to' in ds.i1d.attrs:
                ax.set_ylabel('(i1d + %.f) (norm.)'%(0.01*ds.i1d.attrs['normalized_to']),fontsize=8)  
            else:
                ax.set_ylabel('(i1d) (a.u.)',fontsize=8)


    if ds_previous is not None:
        ax.set_title('Rwp/GoF = %.3f/%.3f (was %.3f/%.3f, %.3f %% change in GoF)'%(
            ds.attrs['Rwp'],
            ds.attrs['GOF'],
            ds_previous.attrs['Rwp'],
            ds_previous.attrs['GOF'],
            100*(ds.attrs['GOF']-ds_previous.attrs['GOF'])/ds_previous.attrs['GOF'],
            ),
            fontsize=8,
            )


    ax.legend(loc='upper right',fontsize=8)
    ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])

    if xlabel:
        ax.set_xlabel(ds.i1d.attrs['xlabel'])
    else:
        ax.set_xlabel(None)

    if return_da:
        return [da_Y_obs,da_Y_calc,da_Y_bkg]





def i2d_plotter(ds,
                ax,
                vmin=None,
                logscale=False,
                robust=True,
                xlabel=False,
                cbar=True,
                cmap='Greys'
                ):

    if ('i2d_baseline' in ds.keys()) and ('roi_azimuthal_range' in ds.i2d.attrs):
        da_i2d = (ds.i2d)
    elif ('i2d_baseline' in ds.keys()):
        da_i2d = (ds.i2d-ds.i2d_baseline)
    else:
        da_i2d = (ds.i2d)

    if logscale:
        da_i2d = np.log(da_i2d)


    if cbar:
        da_i2d.plot.imshow(ax=ax,
                            robust=robust,
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
                            robust=robust,
                            add_colorbar=cbar,
                            cmap=cmap,
                            vmin=vmin
                            )    


    ax.set_xlabel(None)
    ax.set_ylabel('Azimuthal')
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








def phases_plotter(ds,
                   ax_main,
                   line_axes=[],
                   phase_label_x=0.9,
                   phase_label_y=0.8,
                   phase_label_yshift=-0.2,
                   ):

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
                ax_main.axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)

                for a in line_axes:
                    a.axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)

            markerline, stemlines, stem_baseline = ax_main.stem(refl_X,refl_Y,markerfmt=".")
            plt.setp(stemlines, linewidth=0.5, color='C%d'%e)
            plt.setp(markerline, color='C%d'%e)

            ax_main.text(phase_label_x,phase_label_y+e*phase_label_yshift,st,color='C%d'%e,transform=ax_main.transAxes)

            ax_main.set_xlabel(ds.i1d.attrs['xlabel'])



















def exrd_plotter(ds, ds_previous=None, gpx=None, gpx_previous=None, phases=None, plot_hint = '1st_loaded_data'):


    if plot_hint == '1st_loaded_data':
        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=(6,4),dpi=128)
            mosaic = """
                        B
                        C
                        C
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["B"]

            i2d_plotter(ds,ax,cbar=True)


            ax =  ax_dict["C"]
            np.log(ds.i2d.mean(dim='azimuthal_i2d')).plot(ax=ax,color='k')
            ax.set_xlabel(ds.i1d.attrs['xlabel'])
            ax.set_ylabel(ds.i1d.attrs['ylabel'])
            ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')


        else:
            fig = plt.figure(figsize=(8,5),dpi=128)
            mosaic = """
                        C
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

            ax = ax_dict["C"]
            np.log(ds.i1d).plot(ax=ax,color='k')
            ax.set_xlabel(ds.i1d.attrs['xlabel'])
            ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')


        
























    if plot_hint == 'get_baseline':
        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=(6,6),dpi=128)
            mosaic = """
                        AADDD
                        AADDD
                        AADDD
                        AAEEE
                        AAEEE
                        AAEEE
                        AAEEE
                        """
            ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
            ax = ax_dict["A"]
            ax.set_xlim([ds.i1d.radial[0],ds.i1d.radial[-1]])
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


            ax = ax_dict["D"]

            i2d_plotter(ds,ax,cbar=True,vmin=0)


        ax = ax_dict["E"]

        if ('i1d_baseline' in ds.keys()) and ('normalized_to' in ds.i1d.attrs):
            np.log(ds.i1d-ds.i1d_baseline+0.01*ds.i1d.attrs['normalized_to']).plot(ax=ax,color='k',label='i1d - i1d_baseline + %.f'%(0.01*ds.i1d.attrs['normalized_to']))
            ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')
            ax.set_ylim(bottom=np.log(0.01*0.9*ds.i2d.attrs['normalized_to']))
            ax.legend(fontsize=8)
        elif ('i1d_baseline' in ds.keys()):
            np.log(ds.i1d-ds.i1d_baseline +1).plot(ax=ax,color='k',label='i1d - i1d_baseline + 1')
            ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')
            ax.legend(fontsize=8)
        ax.set_xlabel(ds.i1d.attrs['xlabel'])


































    if plot_hint == 'load_phases':

        plot_label_x = 0.9,
        plot_label_y = 0.8,
        plot_label_y_shift = -0.2

        if 'i2d' in ds.keys():
            fig = plt.figure(figsize=(6,6),dpi=128)
            mosaic = """
                        D
                        D
                        E
                        E
                        E
                        E
                        C
                        """
        else:
            fig = plt.figure(figsize=(8,4),dpi=128)
            mosaic = """
                        E
                        E
                        E
                        E
                        C
                        """

        ax_dict = fig.subplot_mosaic(mosaic, sharex=True)


        if 'i2d' in ds.keys():


            ax = ax_dict["D"]
            i2d_plotter(ds,ax,cbar=True,vmin=0)



        ax = ax_dict["E"]

        if ('i1d_baseline' in ds.keys()) and ('normalized_to' in ds.i1d.attrs):
            np.log(ds.i1d-ds.i1d_baseline+0.01*ds.i1d.attrs['normalized_to']).plot(ax=ax,color='k',label='i1d - i1d_baseline + %.f  (norm.)'%(0.01*ds.i1d.attrs['normalized_to']))
            ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')
            ax.set_ylim(bottom=np.log(0.01*0.9*ds.i2d.attrs['normalized_to']))
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=-0.02)
        elif ('i1d_baseline' in ds.keys()):
            np.log(ds.i1d-ds.i1d_baseline +1).plot(ax=ax,color='k',label='i1d - i1d_baseline + 1')
            ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=-0.02)
        else:
            np.log(ds.i1d).plot(ax=ax,color='k',label='i1d')
            ax.set_ylabel('Log$_{10}$(Intensity) (a.u.)')
            ax.legend(fontsize=8)
            # ax.set_ylim(bottom=-0.02)

        ax.set_xlabel(ds.i1d.attrs['xlabel'])

        
        ax.set_xlabel(None)
        
        
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
                    ax_dict["D"].axvline(x=i,lw=0.3,color='C%d'%e)
                ax_dict["E"].axvline(x=i,lw=0.3,color='C%d'%e)
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












    if plot_hint == '1st_refinement':
        fig = plt.figure(figsize=(6,4),dpi=128)
        mosaic = """
                    A
                    """
        ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

        [da_Y_obs,da_Y_calc,da_Y_bkg] = i1d_plotter(ds,
                    ax=ax_dict["A"],
                    ds_previous=ds_previous,
                    xlabel=True,
                    return_da = True,
                    ylogscale=False
                    )
        ax_dict["A"].set_xlabel(None)

        # da = i1d_plotter(ds,
        #             ax=ax_dict["B"],
        #             ds_previous=ds_previous,
        #             xlabel=True,
        #             return_da = True
        #             )        
        # ax_dict["B"].set_ylim(top=np.log(max(da.values)/20))
        # ax_dict["B"].set_title(None)








    if plot_hint == 'refine_background':
        fig = plt.figure(figsize=(6,4),dpi=128)
        mosaic = """
                    A
                    B
                    B
                    """
        ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

        i1d_plotter(ds,
                    ax=ax_dict["A"],
                    ds_previous=ds_previous,
                    xlabel=True,
                    ylogscale=False,
                    return_da = False
                    )
        ax_dict["A"].set_xlabel(None)
        ax_dict["A"].set_ylabel(None)
        ax_dict["A"].get_legend().remove()

        [da_Y_obs,da_Y_calc,da_Y_bkg] = i1d_plotter(ds,
                    ax=ax_dict["B"],
                    ds_previous=ds_previous,
                    xlabel=True,
                    ylogscale=False,
                    return_da = True,
                    )  

        bkg_limits =  min(da_Y_bkg.values),max(da_Y_bkg.values)  
        if bkg_limits[1] < 0:
            ax_dict["B"].set_ylim(bottom=bkg_limits[0]*2, top=bkg_limits[1]*2)
        else:
            ax_dict["B"].set_ylim(bottom=bkg_limits[0]*0.5, top=bkg_limits[1]*2)
        ax_dict["B"].set_title(None)






    if plot_hint == 'refine_cell_params':
        fig = plt.figure(figsize=(6,4),dpi=128)
        mosaic = """
                    A
                    A
                    A
                    A
                    B
                    """
        ax_dict = fig.subplot_mosaic(mosaic, sharex=True)
        i1d_plotter(ds,ax=ax_dict["A"],ds_previous=ds_previous,xlabel=False)
        phases_plotter(ds,ax_main=ax_dict["B"],line_axes=[ax_dict["A"]])







































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