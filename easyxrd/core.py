from scipy.signal import savgol_filter
import pybaselines

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifWriter

from IPython.display import clear_output

import subprocess

import shutil

import random, string
import fnmatch

import time
import copy


import os,sys


import numpy as np
import xarray as xr

import fabio
import pyFAI




import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})


from scipy.ndimage import median_filter


from .plotters import *
 



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





class exrd():
    def __init__(self, verbose=False):
        self.verbose = verbose







    def gpx_refiner(self, save_previous = True):

        if save_previous:
            self.gpx_previous = copy.deepcopy(self.gpx)
            rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
        else:
            rwp_previous = None

        if self.verbose:
            print('\n\n\n\n\n')
            self.gpx.refine()
        else:
            with HiddenPrints():
                self.gpx.refine()
        rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']

        return rwp_new, rwp_previous
    


    def gpx_saver(self):
            if self.verbose:
                self.gpx.save()
            else:
                with HiddenPrints():
                    self.gpx.save()



    def load_xrd_data(self,
                      
                from_img_array=None,
                from_tiff_file=None,
                ai=None,
                mask=None,
                median_filter_size=2,


                from_nc_file=None,

                from_txt_file=None,
                from_txt_file_wavelength_in_angst=0.1814,
                from_txt_file_comments='#',
                from_txt_file_skiprows=0,
                from_txt_file_usecols=(0,1),
                from_txt_file_radial_unit='tth',

                radial_range=[0.1,11.1],


                plot=True,
                ):
        



        if (from_img_array is None) and (from_tiff_file is not None):
            from_img_array = median_filter(fabio.open(from_tiff_file).data,size=median_filter_size)




        if (from_img_array is not None) and (ai is not None):



            self.ds = xr.Dataset()
            delta_q = 0.0025
            npt = int(np.ceil((radial_range[1] - radial_range[0]) / delta_q ))
            radial_range = [radial_range[0],radial_range[0]+delta_q*npt]

            #integrate
            i2d = ai.integrate2d(
                data=from_img_array,
                npt_rad=npt,
                npt_azim=360,
                filename=None,
                correctSolidAngle=True,
                variance=None,
                error_model=None,
                radial_range=radial_range,
                azimuth_range=None,
                mask=mask,
                dummy=np.NaN,
                delta_dummy=None,
                polarization_factor=None,
                dark=None,
                flat=None,
                method='bbox',
                unit='q_A^-1',
                safe=True,
                normalization_factor=1.0,
                metadata=None,
            )

            self.ds['i2d'] = xr.DataArray(
                data=i2d.intensity.astype('float32'),
                coords=[i2d.azimuthal.astype('float32'),i2d.radial.astype('float32')],
                dims=['azimuthal_i2d','radial_i2d'],
                attrs={
                    'radial_unit'         :'q_A^-1',
                    'xlabel'              :r'Scattering vector $q$ ($\AA^{-1}$)',
                    'ylabel'              :r"Azimuthal angle $\chi$ ($^{o}$)",
                    'detector_name'       :ai.__dict__['detector'].name,
                    'wavelength_in_meter' :ai.__dict__['_wavelength'],
                    'detector_dist'       :ai.__dict__['_dist'],
                    'detector_poni1'      :ai.__dict__['_poni1'],
                    'detector_poni2'      :ai.__dict__['_poni2'],
                    'detector_rot1'       :ai.__dict__['_rot1'],
                    'detector_rot2'       :ai.__dict__['_rot2'],
                    'detector_rot3'       :ai.__dict__['_rot3'],
                }
            )

            da_i1d = xr.DataArray(
                data=self.ds['i2d'].mean(dim='azimuthal_i2d').astype('float32'),
                coords=[self.ds['i2d'].radial_i2d],
                dims=['radial'],
                attrs={
                    'radial_unit'         :'q_A^-1',
                    'xlabel'              :r'Scattering vector $q$ ($\AA^{-1}$)',
                    'ylabel'              :r"Intensity (a.u.)",
                    'wavelength_in_angst' :ai.__dict__['_wavelength']*10e9,
                }
            )
            self.ds['i1d'] = da_i1d.dropna(dim='radial')
            # if roi_azimuthal_range is not None:
            #     self.ds['i2d'].attrs['roi_azimuthal_range'] = roi_azimuthal_range
            #     self.ds['i1d'].attrs['roi_azimuthal_range'] = roi_azimuthal_range

            # cropping ds['i2d'] to non NaN range
            # self.ds['i2d'] = da_i2d.sel(radial=slice(self.ds['i1d'].radial[0],self.ds['i1d'].radial[-1]))




        if (from_img_array is None) and (from_txt_file is not None):
            if os.path.isfile(from_txt_file):
                try:
                    X,Y = np.loadtxt(from_txt_file,comments=from_txt_file_comments,skiprows=from_txt_file_skiprows,usecols=from_txt_file_usecols,unpack=True)
                    if from_txt_file_radial_unit.lower()[0] == 't':
                        X = ((4 * np.pi) / (from_txt_file_wavelength_in_angst)) * np.sin(np.deg2rad(X) / 2)
                    elif from_txt_file_radial_unit.lower()[0] == 'q':
                        pass
                    else:
                        print('Unable to determine radial unit. Check the radial_unit in txt file\n\n')
                        return
                    self.ds = xr.Dataset()
                    self.ds['i1d_from_txt_file'] = xr.DataArray(data=Y.astype('float32'),
                                                    coords=[X],
                                                    dims=['radial'],
                                                    attrs={'radial_unit':'q_A^-1',
                                                        'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
                                                        'ylabel':'Intensity (a.u.)',
                                                        'wavelength_in_angst':from_txt_file_wavelength_in_angst,
                                                        'i1d_from':from_txt_file,
                                                            })
                except Exception as exc:
                    print('Unable to read %s \nPlease check %s is a valid plain text file\n\n'%(from_txt_file,from_txt_file))
                    print('Error msg from np.loadtxt:\n%s'%exc)
                    return
            else:
                print('%s does not exist. Please check the file path.'%from_txt_file)
                return
            



        if ((from_img_array is None) and (from_txt_file is None)) and (from_nc_file is not None):
            with xr.open_dataset(from_nc_file) as self.ds:
                pass



        # else:
        #     if (from_nc_file is None) and (from_txt_file is None):
        #         print('Please enter a valid from_nc_file or txt file path to read data')
        #         return
        #     elif (from_nc_file is not None) and (from_txt_file is None)  :
        #         if os.path.isfile(from_nc_file):
        #             try:
        #                 with xr.open_dataset(from_nc_file) as self.ds:
        #                     pass
        #             except Exception as exc:
        #                 print('Unable to read %s \nPlease check %s is a valid xarray nc file\n\n'%(from_nc_file,from_nc_file))
        #                 print('Error msg from xarray:\n%s'%exc)
        #                 return
        #         else:
        #             print('%s does not exist. Please check the file path. '%from_nc_file)
        #             return
        #     elif (from_nc_file is None) and (from_txt_file is not None)  :
        #         if os.path.isfile(from_txt_file):
        #             try:
        #                 X,Y = np.loadtxt(from_txt_file,comments=from_txt_file_comments,skiprows=from_txt_file_skiprows,usecols=from_txt_file_usecols,unpack=True)
        #                 if from_txt_file_radial_unit.lower()[0] == 't':
        #                     X = ((4 * np.pi) / (from_txt_file_wavelength_in_angst)) * np.sin(np.deg2rad(X) / 2)
        #                 elif from_txt_file_radial_unit.lower()[0] == 'q':
        #                     pass
        #                 else:
        #                     print('Unable to determine radial unit. Check the radial_unit in txt file\n\n')
        #                     return
        #                 self.ds = xr.Dataset()
        #                 self.ds['i1d'] = xr.DataArray(data=Y.astype('float32'),
        #                                                 coords=[X],
        #                                                 dims=['radial'],
        #                                                 attrs={'radial_unit':'q_A^-1',
        #                                                     'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
        #                                                     'ylabel':'Intensity (a.u.)',
        #                                                     'wavelength_in_angst':from_txt_file_wavelength_in_angst,
        #                                                     'i1d_from':from_txt_file,
        #                                                         })
        #             except Exception as exc:
        #                 print('Unable to read %s \nPlease check %s is a valid plain text file\n\n'%(from_txt_file,from_txt_file))
        #                 print('Error msg from np.loadtxt:\n%s'%exc)
        #                 return
        #         else:
        #             print('%s does not exist. Please check the file path.'%from_txt_file)
        #             return








        if plot:
            ds_plotter(self.ds, plot_hint = '1st_loaded_data') # type: ignore








    def get_baseline(self,
                     i1d_bkg=None,
                     use_arpls=True,
                     arpls_lam=1e5,
                     plot=True,
                     roi_radial_range=None,
                     roi_azimuthal_range=None,
                     include_baseline_in_ds = True,
                     ):
        

        # refresh i1d
        if ('i2d' in self.ds.keys()):

            for k in ['radial','i1d','i1d_baseline','i1d_refined']:
                if k in self.ds.keys():
                    del self.ds[k]



            # get i2d_baseline

            i2d_baseline = copy.deepcopy(self.ds.i2d)

            # serial version (can be speed-up using threads)
            for a_ind in range(self.ds.i2d.shape[0]):

                # 
                da_now = self.ds.i2d.isel(azimuthal_i2d=a_ind)
                da_now_dropna = da_now.dropna(dim='radial_i2d')
                baseline_now, params = pybaselines.Baseline(x_data=da_now_dropna.radial_i2d.values).arpls(da_now_dropna.values,lam=1e5)

                # create baseline da by copying
                da_now_dropna_baseline = copy.deepcopy(da_now_dropna)
                da_now_dropna_baseline.values = baseline_now

                # now interpolate baseline da to original i2d radial range
                da_now_dropna_baseline_interpolated = da_now_dropna_baseline.interp(radial_i2d=self.ds.i2d.radial_i2d)

                i2d_baseline[a_ind,:] = da_now_dropna_baseline_interpolated


            self.ds['i2d_baseline'] = i2d_baseline




            if roi_azimuthal_range is not None:
                self.ds['i1d'] = self.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[-1])).mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').rename({'radial_i2d': 'radial'})
                self.ds['i1d'].attrs =  {
                                            'radial_unit':'q_A^-1',
                                            'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
                                            'ylabel':'Intensity (a.u.)',
                                            'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
                                            'roi_azimuthal_range':roi_azimuthal_range
                                        }
                self.ds['i1d_baseline'] = self.ds.i2d_baseline.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[-1])).mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').rename({'radial_i2d': 'radial'})
                self.ds['i1d_baseline'].attrs =  {
                                            'radial_unit':'q_A^-1',
                                            'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
                                            'ylabel':'Intensity (a.u.)',
                                            'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
                                            'roi_azimuthal_range':roi_azimuthal_range,
                                            'arpls_lam':arpls_lam,
                                        }
            else:
                self.ds['i1d'] = self.ds.i2d.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').rename({'radial_i2d': 'radial'})
                self.ds['i1d'].attrs =  {
                                            'radial_unit':'q_A^-1',
                                            'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
                                            'ylabel':'Intensity (a.u.)',
                                            'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
                                        }
                self.ds['i1d_baseline'] = self.ds.i2d_baseline.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').rename({'radial_i2d': 'radial'})         
                self.ds['i1d_baseline'].attrs =  {
                                            'radial_unit':'q_A^-1',
                                            'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
                                            'ylabel':'Intensity (a.u.)',
                                            'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
                                            'arpls_lam':arpls_lam,
                                        }
                


            if not include_baseline_in_ds:
                del self.ds['i2d_baseline']

                    
        #     if roi_azimuthal_range is not None:
        #         da_here = self.ds['i2d'].sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1])).mean(dim='azimuthal_i2d')
        #         da_i1d = xr.DataArray(data=da_here.dropna(dim='radial_i2d').values,
        #                               dims = ['radial'],
        #                               coords = [da_here.dropna(dim='radial_i2d').radial_i2d],
        #                               attrs = {
        #                                   'radial_unit':'q_A^-1',
        #                                   'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
        #                                   'ylabel':'Intensity (a.u.)',
        #                                   'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
        #                                   'roi_azimuthal_range':roi_azimuthal_range
        #                               }
        #         )
        #         self.ds['i1d'] = da_i1d

        #     else:
        #         da_here = self.ds['i2d'].mean(dim='azimuthal_i2d')
        #         da_i1d = xr.DataArray(data=da_here.dropna(dim='radial_i2d').values,
        #                               dims = ['radial'],
        #                               coords = [da_here.dropna(dim='radial_i2d').radial_i2d],
        #                               attrs = {
        #                                   'radial_unit':'q_A^-1',
        #                                   'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
        #                                   'ylabel':'Intensity (a.u.)',
        #                                   'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
        #                               }
        #         )
        #         self.ds['i1d'] = da_i1d

        #     if roi_radial_range is not None:
        #         self.ds = self.ds.sel(radial=slice(roi_radial_range[0],roi_radial_range[1]))

        # if ('i1d_from_txt_file' in self.ds.keys()):
        #     self.ds['i1d'] = self.ds.i1d_from_txt_file


        # if roi_radial_range is not None:
        #     self.ds = self.ds.sel(radial=slice(roi_radial_range[0],roi_radial_range[1]))

        if roi_radial_range is not None:
            self.ds = self.ds.sel(radial=slice(roi_radial_range[0],roi_radial_range[1]))


              
        if i1d_bkg is not None:
            # check the limits
            if not np.array_equal(i1d_bkg.radial,self.ds.i1d.radial):
                if (i1d_bkg.radial.values[0] > self.ds.i1d.radial.values[0]) or (i1d_bkg.radial.values[-1] < self.ds.i1d.radial.values[-1]):
                    print('i1d_bkg is not useable. i1d_bkg is ignored and using arpls to find the baseline\n\n')
                    i1d_bkg_useable = False
                else:
                    i1d_bkg = i1d_bkg.interp(radial=self.ds.i1d.radial)
                    i1d_bkg_useable = True

            else:
                i1d_bkg_useable = True

            if i1d_bkg_useable:
                bkg_scale = self.ds['i1d'].values[0]/i1d_bkg.values[0]
                while (min((self.ds['i1d'].values-bkg_scale*i1d_bkg.values)) < 0):
                    bkg_scale = bkg_scale*0.99
                if use_arpls:
                    baseline = pybaselines.Baseline(x_data=self.ds['i1d'].radial.values).arpls((self.ds['i1d']-bkg_scale*i1d_bkg).values, 
                                                                                               lam=arpls_lam)[0]
                    self.ds['i1d_baseline'] = xr.DataArray(data=(baseline+bkg_scale*i1d_bkg.values),
                                                           dims=['radial'],
                                                           coords={'radial':self.ds.i1d.radial},
                                                           attrs={'arpls_lam':arpls_lam}
                                                           )
                else:
                    baseline = bkg_scale*i1d_bkg
                    self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),
                                                           dims=['radial'],
                                                           coords={'radial':self.ds.i1d.radial},
                                                           attrs={'arpls_lam':arpls_lam}
                                                           )
            else:
                baseline = pybaselines.Baseline(x_data=self.ds['i1d'].radial.values).arpls((self.ds['i1d']).values, 
                                                                                           lam=arpls_lam)[0]
                self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),dims=['radial'],
                                                       coords={'radial':self.ds.i1d.radial},
                                                       attrs={'arpls_lam':arpls_lam}
                                                       )
        # else:
        #     print('i1d_bkg is not provided. Using arpls to find the baseline\n\n')
        #     baseline = pybaselines.Baseline(x_data=self.ds['i1d'].radial.values).arpls((self.ds['i1d']).values, 
        #                                                                                lam=arpls_lam)[0]
        #     self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),dims=['radial'],
        #                                            coords={'radial':self.ds.i1d.radial},
        #                                            attrs={'arpls_lam':arpls_lam}
        #                                            )

        # if smoothen:
        #     self.ds['i1d_bkg'] = xr.DataArray(data=savgol_filter(i1d_bkg.interp(radial=self.ds.i1d.radial).values, window_length=savgol_filter_window_length, polyorder=2),
        #                                     coords=[self.ds.i1d.radial],
        #                                     dims=['radial'],
        #                                     attrs={'savgol_filter_window_length': savgol_filter_window_length,
        #                                             'savgol_filter_polyorder': 2,
        #                                     } | i1d_bkg.attrs)
        # else:
        #     self.ds['i1d_bkg'] = i1d_bkg.interp(radial=self.ds.i1d.radial)


        if plot:
            ds_plotter(ds=self.ds,  plot_hint = 'get_baseline') # type: ignore












    def load_phases(self,
                    phases,
                    mp_rester_api_key='dHgNQRNYSpuizBPZYYab75iJNMJYCklB',
                    plot = True,
                    ):
        
        self.phases = {}
        for p in phases:
            if p['mp_id'].lower() == 'none':
                st = Structure.from_file(p['cif'])
                st.lattice = Lattice.from_parameters(a=st.lattice.abc[0]*p['scale']*p['scale_a'],
                                                     b=st.lattice.abc[1]*p['scale']*p['scale_b'],
                                                     c=st.lattice.abc[2]*p['scale']*p['scale_c'],
                                                     alpha=st.lattice.angles[0],
                                                     beta =st.lattice.angles[1],
                                                     gamma=st.lattice.angles[2]
                                                    )
                self.phases[p['label']] = st
            else:
                from mp_api.client import MPRester
                mpr = MPRester(mp_rester_api_key)
                st = mpr.get_structure_by_material_id(p['mp_id'],final=False)[0]
                st.lattice = Lattice.from_parameters(a=st.lattice.abc[0]*p['scale']*p['scale_a'],
                                                     b=st.lattice.abc[1]*p['scale']*p['scale_b'],
                                                     c=st.lattice.abc[2]*p['scale']*p['scale_c'],
                                                     alpha=st.lattice.angles[0],
                                                     beta =st.lattice.angles[1],
                                                     gamma=st.lattice.angles[2]
                                                    )
                self.phases[p['label']] = st
                
        if plot:
            ds_plotter(ds=self.ds, phases=self.phases,  plot_hint = 'load_phases') # type: ignore





    def export_phases(self,phase_ind=None, # should start from 0. -1 is not allowed
                            export_to='.',
                            export_extension='_exported.cif'
                            ):
        if phase_ind is None:
            for e,st in enumerate(self.phases):
                CifWriter(self.phases[st],symprec=0.01).write_file("%s/%s%s"%(export_to,st,export_extension))
        else:
            for e,st in enumerate(self.phases):
                if e == phase_ind:
                    CifWriter(self.phases[st],symprec=0.01).write_file("%s/%s%s"%(export_to,st,export_extension))
            






    def setup_gsas2_calculator(self,
                               gsasii_lib_directory=None,
                               gsasii_scratch_directory=None,
                               instprm_from_gpx=None,
                               instprm_Polariz=0,
                               instprm_Azimuth=0,
                               instprm_Zero=-0.0006,                                                                   
                               instprm_U=118.313,                                  
                               instprm_V=4.221,                                 
                               instprm_W=0.747,   
                               instprm_X=0, 
                               instprm_Y=-7.148, 
                               instprm_Z=0, 
                               instprm_SHL=0.002,   
                               do_1st_refinement=True,
                        ):
        




        
        if gsasii_lib_directory is None:

            try:
                default_install_path = os.path.join(os.path.expanduser('~'),'g2full/GSAS-II/GSASII')
                sys.path += [default_install_path]
                import GSASIIscriptable as G2sc
                self.gsasii_lib_directory = default_install_path
            except:
                user_loc = input("Enter location of GSASII directory on your GSAS-II installation.")
                sys.path += [user_loc]
                try:
                    import GSASIIscriptable as G2sc
                    self.gsasii_lib_directory = user_loc
                except:
                    try:
                        gsasii_lib_directory = input("\nUnable to import GSASIIscriptable. Please re-enter GSASII directory on your GSAS-II installation\n")
                        sys.path += [user_loc]
                        import GSASIIscriptable as G2sc
                        self.gsasii_lib_directory = user_loc
                    except:
                        gsasii_lib_directory = input("\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html")
                else:
                    #clear_output()
                    self.gsasii_lib_directory = gsasii_lib_directory
        else:
            if os.path.isdir(gsasii_lib_directory):
                sys.path += [gsasii_lib_directory]
                try:
                    import GSASIIscriptable as G2sc
                except:
                    try:
                        gsasii_lib_directory = input("\nUnable to import GSASIIscriptable. Please enter GSASII directory on your GSAS-II installation\n")
                        sys.path += [gsasii_lib_directory]
                        import GSASIIscriptable as G2sc
                        self.gsasii_lib_directory = gsasii_lib_directory
                    except:
                        gsasii_lib_directory = input("\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html")
                    else:
                        #clear_output()
                        self.gsasii_lib_directory = gsasii_lib_directory
                else:
                    #clear_output()
                    self.gsasii_lib_directory = gsasii_lib_directory
            else:
                print('%s does NOT exist. Please check!'%gsasii_lib_directory)




        if gsasii_scratch_directory is None:
            user_home = os.path.expanduser('~')
            if not os.path.isdir(os.path.join(user_home,'.gsasii_scratch')):
                os.mkdir(os.path.join(user_home,'.gsasii_scratch'))
            self.gsasii_scratch_directory = os.path.join(user_home,'.gsasii_scratch')
        else:
            try:
                os.makedirs(gsasii_scratch_directory,exist_ok=True)
            except Exception as exc:
                print(exc)
                print('Unable to creat or use gsasii_scratch_directory. Please check.')
                return
            else:
                self.gsasii_scratch_directory = gsasii_scratch_directory

        # randstr = ''.join(random.choices(string.ascii_uppercase+string.digits, k=7))
        # self.gsasii_run_directory = '%s/%.2f_%s.gsastmp'%(self.gsasii_scratch_directory,time.time(),randstr)
        randstr ='test'
        self.gsasii_run_directory = '%s/%.2f_%s.gsastmp'%(self.gsasii_scratch_directory,10.0,randstr)

        os.makedirs(self.gsasii_run_directory,exist_ok=True)




        # np.savetxt('%s/data.xy'%self.gsasii_run_directory,
        #            fmt='%.7e',
        #            X=np.column_stack( (self.ds.i1d.radial.values, (self.ds.i1d-self.ds.i1d_baseline).values) ))
        np.savetxt('%s/data.xy'%self.gsasii_run_directory,
                   fmt='%.7e',
                   X=np.column_stack( (np.rad2deg( 2 * np.arcsin( self.ds.i1d.radial.values * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) ), (self.ds.i1d-self.ds.i1d_baseline).values+10 ) ))        


        if instprm_from_gpx is not None:
            if os.path.isfile(instprm_from_gpx):
                gpx_instprm = G2sc.G2Project(gpxfile=instprm_from_gpx)
                for n in gpx_instprm.names:
                    l = n
                    pattern = 'PWDR *'
                    matching = fnmatch.filter(l, pattern)
                    if matching != []:
                        pwdr_name = matching[0]
                instprm_dict = gpx_instprm[pwdr_name]['Instrument Parameters'][0]

                with open('%s/gsas.instprm'%self.gsasii_run_directory, 'w') as f:
                    f.write('#GSAS-II instrument parameter file; do not add/delete items!\n')
                    f.write('Type:PXC\n')
                    f.write('Bank:1.0\n')
                    # f.write('Lam:%s\n'%(instprm_dict['Lam'][1]))
                    f.write('Lam:%s\n'%(self.ds.i1d.attrs['wavelength_in_angst']))
                    f.write('Polariz.:%s\n'%(instprm_dict['Polariz.'][1]))
                    f.write('Azimuth:%s\n'%(instprm_dict['Azimuth'][1]))
                    f.write('Zero:%s\n'%(instprm_dict['Zero'][1]))
                    f.write('U:%s\n'%(instprm_dict['U'][1]))
                    f.write('V:%s\n'%(instprm_dict['V'][1]))
                    f.write('W:%s\n'%(instprm_dict['W'][1]))
                    f.write('X:%s\n'%(instprm_dict['X'][1]))
                    f.write('Y:%s\n'%(instprm_dict['Y'][1]))
                    f.write('Z:%s\n'%(instprm_dict['Z'][1]))
                    f.write('SH/L:%s\n'%(instprm_dict['SH/L'][1]))
            else:
                print('gpx file for reading instrument parameters do net exist. Please check the path')
                # return

        else:
            with open('%s/gsas.instprm'%self.gsasii_run_directory, 'w') as f:
                f.write('#GSAS-II instrument parameter file; do not add/delete items!\n')
                f.write('Type:PXC\n')
                f.write('Bank:1.0\n')
                f.write('Lam:%s\n'%(self.ds.i1d.attrs['wavelength_in_angst']))
                f.write('Polariz.:%s\n'%(instprm_Polariz))
                f.write('Azimuth:%s\n'%(instprm_Azimuth))
                f.write('Zero:%s\n'%(instprm_Zero))
                f.write('U:%s\n'%(instprm_U))
                f.write('V:%s\n'%(instprm_V))
                f.write('W:%s\n'%(instprm_W))
                f.write('X:%s\n'%(instprm_X))
                f.write('Y:%s\n'%(instprm_Y))
                f.write('Z:%s\n'%(instprm_Z))
                f.write('SH/L:%s\n'%(instprm_SHL))





        if self.verbose:

            self.gpx = G2sc.G2Project(newgpx='%s/gsas.gpx'%self.gsasii_run_directory)
            self.gpx.data['Controls']['data']['max cyc'] = 100
            self.gpx.add_powder_histogram('%s/data.xy'%self.gsasii_run_directory,'%s/gsas.instprm'%self.gsasii_run_directory)
            self.export_phases(export_to=self.gsasii_run_directory,export_extension='.cif')
            hist = self.gpx.histograms()[0]
            for p in self.phases:
                self.gpx.add_phase('%s/%s.cif'%(self.gsasii_run_directory,p),phasename=p,histograms=[hist],fmthint='CIF')

        else:
            with HiddenPrints():
                self.gpx = G2sc.G2Project(newgpx='%s/gsas.gpx'%self.gsasii_run_directory)
                self.gpx.data['Controls']['data']['max cyc'] = 100
                self.gpx.add_powder_histogram('%s/data.xy'%self.gsasii_run_directory,'%s/gsas.instprm'%self.gsasii_run_directory)
                self.export_phases(export_to=self.gsasii_run_directory,export_extension='.cif')
                hist = self.gpx.histograms()[0]
                for p in self.phases:
                    self.gpx.add_phase('%s/%s.cif'%(self.gsasii_run_directory,p),phasename=p,histograms=[hist],fmthint='CIF')




        for n in self.gpx.names:
            l = n
            pattern = 'PWDR *'
            matching = fnmatch.filter(l, pattern)
            if matching != []:
                pwdr_name = matching[0]
        self.gpx[pwdr_name]['Background'][0] = ['chebyschev-1', True, 3, 10, 0.0, 0.0]


        if do_1st_refinement:
            ParDict = {'set': {'Background': {'refine': False,
                                            'type': 'chebyschev-1',
                                            'no. coeffs': 1
                                            }}}
            self.gpx.set_refinement(ParDict)

            rwp_new, _ = self.gpx_refiner(save_previous=False)

            print('\nRwp from 1st refinement is = %.3f \n '%(rwp_new))










    def set_LeBail(self, set_to=True, phase_ind=None, refine=False):
        if phase_ind is None:
            self.gpx.set_refinement({"set":{'LeBail': set_to}})
        else:
            for e,p in enumerate(self.phases):
                if e == phase_ind:
                    self.gpx['Phases'][p]['Histograms']['PWDR data.xy']['LeBail'] = set_to
        if refine:
            self.gpx_previous = copy.deepcopy(self.gpx)
            rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
            self.gpx_refiner()
            rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']
            print('set_LeBail output:\nRwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))
        self.gpx_saver()








    def refine_background(self, num_coeffs=10, background_type='chebyschev-1', set_to_false_after_refinement=True):
        ParDict = {'set': {'Background': {'refine': True,
                                        'type': background_type,
                                        'no. coeffs': num_coeffs
                                        }}}
        self.gpx.set_refinement(ParDict)

        rwp_new, rwp_previous = self.gpx_refiner(self)
        print('Background is refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            self.gpx.set_refinement({'set': {'Background': {'refine': False}}})
        self.gpx_saver()


                

    def refine_cell_params(self, phase_ind=None, set_to_false_after_refinement=True, export_refined_phases=True):

        for e,p in enumerate(self.gpx.phases()):
            if phase_ind is None:
                self.gpx['Phases'][p.name]['General']['Cell'][0]= True
            else:
                if e == phase_ind:
                    self.gpx['Phases'][p.name]['General']['Cell'][0]= True


        rwp_new, rwp_previous = self.gpx_refiner(self)
        if phase_ind is None:
            print('Cell parameters of all phases are refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))
        else:
            print('Cell parameters of %s phase is refined. Rwp is now %.3f (was %.3f)'%(p.name,rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            phases = self.gpx.phases()
            for p in phases:
                self.gpx['Phases'][p.name]['General']['Cell'][0]= False
        self.gpx_saver()

        # if export_refined_phases:
        #     for p in self.gpx.phases():
        #         p.export_CIF(outputname='%s/%s_refined.cif'%(self.gsasii_run_directory,p.name))



    def refine_strain_broadening(self,set_to_false_after_refinement=True):

        ParDict = {'set': {'Mustrain': {'type': 'isotropic',
                                        'refine': True
                                        }}}
        self.gpx.set_refinement(ParDict)

        rwp_new, rwp_previous = self.gpx_refiner(self)
        print('Strain broadening of all phases are refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            ParDict = {'set': {'Mustrain': {'type': 'isotropic',
                                'refine': False
                                }}}
            self.gpx.set_refinement(ParDict)
        self.gpx_saver()



    def refine_size_broadening(self,set_to_false_after_refinement=True):


        ParDict = {'set': {'Size': {'type': 'isotropic',
                                        'refine': True
                                        }}}
        self.gpx.set_refinement(ParDict)

        rwp_new, rwp_previous = self.gpx_refiner(self)
        print('Size broadening of all phases are refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            ParDict = {'set': {'Size': {'type': 'isotropic',
                                'refine': False
                                }}}
            self.gpx.set_refinement(ParDict)
        self.gpx_saver()




    def refine_inst_parameters(self,inst_pars_to_refine=['U', 'V', 'W'],set_to_false_after_refinement=True,verbose=False):
        """
        inst_pars_to_refine=['U', 'V', 'W',   'X', 'Y', 'Z', 'Zero', 'SH/L']
        """

        self.gpx.set_refinement({"set": {'Instrument Parameters': inst_pars_to_refine}})

        rwp_new, rwp_previous = self.gpx_refiner(self)
        print('Instrument parameters %s are refined. Rwp is now %.3f (was %.3f)'%(inst_pars_to_refine,rwp_new,rwp_previous))
     
        if set_to_false_after_refinement:
            ParDict = {"clear": {'Instrument Parameters': ['X', 'Y', 'Z', 'Zero', 'SH/L', 'U', 'V', 'W']}}
            self.gpx.set_refinement(ParDict)
        self.gpx_saver()
            




    def export_ds(self,save_dir='.',save_name='ds.nc'):
        self.ds.to_netcdf('%s/%s'%(save_dir,save_name),
                     engine="h5netcdf",
                     encoding={'i2d': {'zlib': True, 'complevel': 9}}) # pip install h5netcdf






    def fine_tune_gpx(self):
        subprocess.check_call(['%s/../../RunGSASII.sh'%self.gsasii_lib_directory, '%s/gsas.gpx'%self.gsasii_run_directory])
        import GSASIIscriptable as G2sc
        self.gpx = G2sc.G2Project(gpxfile='%s/gsas.gpx'%self.gsasii_run_directory)
        self.gpx.refine()


    def replace_gpx_with(self,newgpx_to_replace):
        shutil.copy(newgpx_to_replace,'%s/gsas.gpx'%self.gsasii_lib_directory)
        import GSASIIscriptable as G2sc
        self.gpx = G2sc.G2Project(gpxfile='%s/gsas.gpx'%self.gsasii_run_directory)
        self.gpx.refine()














    def plot_refinement(self,
                    label_x = 0.9,
                    label_y = 0.8,
                    label_y_shift = -0.2
                    ):

        histogram = self.gpx.histograms()[0]
        Ycalc     = histogram.getdata('ycalc').astype('float32')
        Ybkg      = histogram.getdata('Background').astype('float32')
        self.ds['i1d_refined'] = xr.DataArray(data=Ycalc,dims=['radial'],coords={'radial':self.ds.i1d.radial})
        
        for p in self.gpx.phases():
            p.export_CIF(outputname='%s/%s_refined.cif'%(self.gsasii_run_directory,p.name))

        if 'i2d' in self.ds.keys():
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

        if 'i2d' in self.ds.keys():
            ax = ax = ax_dict["A"]
            np.log(self.ds.i2d).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
            # self.ds.i2d.plot.imshow(ax=ax,robust=False,add_colorbar=False,cmap='viridis',vmin=0)
            ax.set_xlabel(None)
            ax.set_ylabel('Azimuthal')

            try:
                roi_xy = [self.ds.i1d.radial.values[0],self.ds.i1d.attrs['roi_azimuthal_range'][0]]
                roi_width = self.ds.i1d.radial.values[-1] - self.ds.i1d.radial.values[0]
                roi_height = self.ds.i1d.attrs['roi_azimuthal_range'][1] - self.ds.i1d.attrs['roi_azimuthal_range'][0]
                rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
                ax.add_patch(rect)
            except:
                pass

        ax = ax_dict["B"]
        np.log(self.ds.i1d-self.ds.i1d_baseline+10).plot(ax=ax,color='k',label='Yobs.')
        np.log(self.ds.i1d_refined).plot(ax=ax, alpha=0.9, linewidth=1, color='y',label='Ycalc. (Rwp=%.3f)'%self.gpx['Covariance']['data']['Rvals']['Rwp'])       
        ax.fill_between(self.ds.i1d.radial.values, self.ds.i1d.radial.values*0+np.log(10),alpha=0.2)
        ax.set_xlabel(None)
        ax.set_ylabel('Log$_{10}$(data-baseline+10) (a.u.)')
        ax.set_ylim(bottom=np.log(8))
        ax.legend()
        ax.set_xlim([self.ds.i1d.radial[0],self.ds.i1d.radial[-1]])

        xrdc = XRDCalculator(wavelength=self.ds.i1d.attrs['wavelength_in_angst'])








        # supplied phases
        for e,st in enumerate(self.phases):
            ps = xrdc.get_pattern(self.phases[st],
                                scaled=True,
                                two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([self.ds.i1d.radial.values[0],self.ds.i1d.radial.values[-1]]) * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) )
                                )
            refl_X, refl_Y = ((4 * np.pi) / (self.ds.i1d.attrs['wavelength_in_angst'])) * np.sin(np.deg2rad(ps.x) / 2), ps.y

            for i in refl_X:
                if 'i2d' in self.ds.keys():
                    ax_dict["A"].axvline(x=i,lw=0.3,color='C%d'%e)
                ax_dict["B"].axvline(x=i,lw=0.3,color='C%d'%e)
                ax_dict["C"].axvline(x=i,lw=0.3,color='C%d'%e)

            markerline, stemlines, baseline = ax_dict["C"].stem(refl_X,refl_Y,markerfmt=".")
            ax_dict["C"].set_ylim(bottom=0)

            plt.setp(stemlines, linewidth=0.5, color='C%d'%e)
            plt.setp(markerline, color='C%d'%e)

            ax_dict["C"].text(label_x,label_y+e*label_y_shift,st,color='C%d'%e,transform=ax_dict["C"].transAxes)






        # refined phases
        self.refined_phases = {}
        for p in self.phases:
                st = Structure.from_file('%s/%s_refined.cif'%(self.gsasii_run_directory,p))
                self.refined_phases[p] = st

        for e,st in enumerate(self.refined_phases):
            ps = xrdc.get_pattern(self.refined_phases[st],
                                scaled=True,
                                two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([self.ds.i1d.radial.values[0],self.ds.i1d.radial.values[-1]]) * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) )
                                )
            refl_X, refl_Y = ((4 * np.pi) / (self.ds.i1d.attrs['wavelength_in_angst'])) * np.sin(np.deg2rad(ps.x) / 2), ps.y

            for i in refl_X:
                if 'i2d' in self.ds.keys():
                    ax_dict["A"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
                ax_dict["B"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
                ax_dict["C"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)

            markerline, stemlines, baseline = ax_dict["C"].stem(refl_X,refl_Y,markerfmt="+")
            ax_dict["C"].set_ylim(bottom=0)

            plt.setp(stemlines, linewidth=0.5, linestyle='--', color='C%d'%e)
            plt.setp(markerline, color='C%d'%e)

        ax_dict["C"].set_xlabel(self.ds.i1d.attrs['xlabel'])


