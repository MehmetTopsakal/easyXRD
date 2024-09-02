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
from copy import deepcopy


import os,sys


import numpy as np
import xarray as xr

import fabio
import pyFAI


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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


        # super(exrd, self).__init__()







    def gpx_refiner(self, 
                    remember_previous_ds = True, 
                    remember_previous_gpx = True, 
                    update_ds=True
                    ):

        if remember_previous_gpx:
            self.gpx_previous = copy.deepcopy(self.gpx)
            rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
        else:
            rwp_previous = None


        if remember_previous_ds:
            self.ds_previous = copy.deepcopy(self.ds)


        if self.verbose:
            print('\n\n\n\n\n')
            self.gpx.refine()
        else:
            with HiddenPrints():
                self.gpx.refine()
        rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']

        if update_ds:
            histogram = self.gpx.histograms()[0]
            if 'normalized_to' in self.ds.i1d.attrs:
                Ycalc     = histogram.getdata('ycalc').astype('float32')-self.yshift_multiplier*self.ds.i2d.attrs['normalized_to'] # this includes gsas background
                Ybkg      = histogram.getdata('Background').astype('float32')-self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']
            else:
                Ycalc     = histogram.getdata('ycalc').astype('float32') -10 # this includes gsas background
                Ybkg      = histogram.getdata('Background').astype('float32') -10
            self.ds['i1d_refined'] = xr.DataArray(data=Ycalc,dims=['radial'],coords={'radial':self.ds.i1d.radial})
            self.ds['i1d_gsas_background'] = xr.DataArray(data=Ybkg,dims=['radial'],coords={'radial':self.ds.i1d.radial})
            self.ds.attrs = self.ds.attrs | self.gpx['Covariance']['data']['Rvals']

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
                    self.ds['i1d'] = xr.DataArray(data=Y.astype('float32'),
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
                     input_bkg=None,
                     use_arpls=True,
                     arpls_lam=1e5,
                     plot=True,
                     get_i2d_baseline = False,
                     use_i2d_baseline = False,
                     roi_radial_range=None,
                     roi_azimuthal_range=None,
                     include_baseline_in_ds = True,
                     normalize = True,
                     normalize_to = 100,
                     spotty_data_correction=False,
                     spotty_data_correction_threshold=1,
                     ):
        


        if (input_bkg is None) and  (use_arpls is False):
            print('\n\nYou did not provide input_bkg and use_arpls is set to False. Nothing to do here. baseline is not calculated...\n\n')
            plot=False

        else:

            if input_bkg is not None:
                if (('i2d' in self.ds.keys())) and ('i2d' in input_bkg.ds.keys()):
                    #check if they have same radial and azimuthal
                    if (np.array_equal(input_bkg.ds.radial_i2d,self.ds.radial_i2d)) and (np.array_equal(input_bkg.ds.azimuthal_i2d,self.ds.azimuthal_i2d)) :
                        for k in ['radial','i1d','i1d_baseline','i2d_baseline']:
                            if k in self.ds.keys():
                                del self.ds[k]

                        if roi_azimuthal_range is not None:
                            # da_i2d = (self.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1]))) 
                            da_i1d     = self.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1])).mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')
                            da_i1d_bkg = input_bkg.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1])).mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')
                        else:
                            da_i1d     = self.ds.i2d.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')
                            da_i1d_bkg = input_bkg.ds.i2d.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')

                        da_i1d.values = median_filter(da_i1d.values,size=3)
                        da_i1d_bkg.values = median_filter(da_i1d_bkg.values,size=3)
                        bkg_scale = da_i1d.values[0]/da_i1d_bkg.values[0]
                        while (min((da_i1d.values-bkg_scale*da_i1d_bkg.values)) < 0):
                            bkg_scale = bkg_scale*0.99
                        if use_arpls:
                            if roi_azimuthal_range is not None:
                                da_i2d_diff = (self.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1]))-bkg_scale*input_bkg.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1]))) 
                            else:
                                da_i2d_diff = (self.ds.i2d-bkg_scale*input_bkg.ds.i2d) 
                            if get_i2d_baseline:
                                da_i2d_diff_baseline = deepcopy(da_i2d_diff)
                                # serial version (can be speed-up using threads)
                                for a_ind in range(da_i2d_diff.shape[0]):
                                    # 
                                    da_now = da_i2d_diff_baseline.isel(azimuthal_i2d=a_ind)
                                    da_now_dropna = da_now.dropna(dim='radial_i2d')
                                    try:
                                        baseline_now, params = pybaselines.Baseline(x_data=da_now_dropna.radial_i2d.values).arpls(da_now_dropna.values,lam=arpls_lam)
                                        # create baseline da by copying
                                        da_now_dropna_baseline = copy.deepcopy(da_now_dropna)
                                        da_now_dropna_baseline.values = baseline_now
                                        # now interpolate baseline da to original i2d radial range
                                        da_now_dropna_baseline_interpolated = da_now_dropna_baseline.interp(radial_i2d=da_i2d_diff.radial_i2d)
                                        da_i2d_diff_baseline[a_ind,:] = da_now_dropna_baseline_interpolated
                                    except:
                                        # da_now.values[:] = np.nan
                                        da_i2d_diff_baseline[a_ind,:] = da_now
                                if roi_azimuthal_range is not None:
                                    self.ds['i2d_baseline'] = (da_i2d_diff_baseline+(bkg_scale*input_bkg.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1]))))
                                else:
                                    self.ds['i2d_baseline'] = (da_i2d_diff_baseline+(bkg_scale*input_bkg.ds.i2d))
                                self.ds['i2d_baseline'].attrs['baseline_note'] = 'baseline is from provided input_bkg and arpls is used'
                                self.ds['i2d_baseline'].attrs['arpls_lam'] = arpls_lam

                                if use_i2d_baseline:
                                    self.ds['i1d_baseline'] = self.ds['i2d_baseline'].mean(dim='azimuthal_i2d').rename({'radial_i2d': 'radial'})
                                    self.ds['i1d_baseline'].attrs['baseline_note'] = 'baseline is from i2d_baseline as available in this dataset. arpls is used'
                                    self.ds['i1d_baseline'].attrs['arpls_lam'] = arpls_lam
                                else:
                                    da_for_baseline = da_i2d_diff.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')
                                    diff_baseline, params = pybaselines.Baseline(x_data=da_for_baseline.radial_i2d.values).arpls(da_for_baseline.values,lam=arpls_lam)
                                    if roi_azimuthal_range is not None:
                                        self.ds['i1d_baseline'] = xr.DataArray(data=(diff_baseline+bkg_scale*input_bkg.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1])).mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').values),dims=['radial_i2d'],
                                                                                coords={'radial_i2d':da_for_baseline.radial_i2d.values},
                                                                                attrs={'arpls_lam':arpls_lam}
                                                                                ).interp(radial_i2d=self.ds.i2d.radial_i2d).rename({'radial_i2d': 'radial'})
                                    else:
                                        self.ds['i1d_baseline'] = xr.DataArray(data=(diff_baseline+bkg_scale*input_bkg.ds.i2d.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').values),dims=['radial_i2d'],
                                                                                coords={'radial_i2d':da_for_baseline.radial_i2d.values},
                                                                                attrs={'arpls_lam':arpls_lam}
                                                                                ).interp(radial_i2d=self.ds.i2d.radial_i2d).rename({'radial_i2d': 'radial'})
                                    self.ds['i1d_baseline'].attrs['baseline_note'] = 'baseline is from provided input_bkg and arpls is used'
                                    self.ds['i1d_baseline'].attrs['arpls_lam'] = arpls_lam


                            else:
                                da_for_baseline = da_i2d_diff.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')
                                diff_baseline, params = pybaselines.Baseline(x_data=da_for_baseline.radial_i2d.values).arpls(da_for_baseline.values,lam=arpls_lam)
                                if roi_azimuthal_range is not None:
                                    self.ds['i1d_baseline'] = xr.DataArray(data=(diff_baseline+bkg_scale*input_bkg.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1])).mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').values),dims=['radial_i2d'],
                                                                            coords={'radial_i2d':da_for_baseline.radial_i2d.values},
                                                                            attrs={'arpls_lam':arpls_lam}
                                                                            ).interp(radial_i2d=self.ds.i2d.radial_i2d).rename({'radial_i2d': 'radial'})
                                else:
                                    self.ds['i1d_baseline'] = xr.DataArray(data=(diff_baseline+bkg_scale*input_bkg.ds.i2d.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d').values),dims=['radial_i2d'],
                                                                            coords={'radial_i2d':da_for_baseline.radial_i2d.values},
                                                                            attrs={'arpls_lam':arpls_lam}
                                                                            ).interp(radial_i2d=self.ds.i2d.radial_i2d).rename({'radial_i2d': 'radial'})
                                self.ds['i1d_baseline'].attrs['baseline_note'] = 'baseline is from provided input_bkg and arpls is used'
                                self.ds['i1d_baseline'].attrs['arpls_lam'] = arpls_lam
                        else:
                            if roi_azimuthal_range is not None:
                                self.ds['i2d_baseline'] = deepcopy(bkg_scale*input_bkg.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1])))
                            else:
                                self.ds['i2d_baseline'] = deepcopy(bkg_scale*input_bkg.ds.i2d)
                            self.ds['i2d_baseline'].attrs['baseline_note'] = 'baseline is from provided input_bkg. arpls is not used'
                            self.ds['i1d_baseline'] = self.ds['i2d_baseline'].mean(dim='azimuthal_i2d').rename({'radial_i2d': 'radial'})
                            self.ds['i1d_baseline'].attrs['baseline_note'] = 'baseline is from i2d_baseline as available in this dataset. arpls is not used'

                    else:
                        #TODO
                        print('dimensions do not match.... ignoring input_bkg and getting baseline via arpls')



            else:

                if ('i2d' in self.ds.keys()):

                    for k in ['radial','i1d','i1d_baseline','i2d_baseline']:
                        if k in self.ds.keys():
                            del self.ds[k]

                    if use_arpls:

                        if roi_azimuthal_range is not None:
                            da_i2d = (self.ds.i2d.sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1]))) 
                        else:
                            da_i2d = (self.ds.i2d) 

                        if get_i2d_baseline:
                            da_i2d_baseline = deepcopy(da_i2d)
                            # serial version (can be speed-up using threads)
                            for a_ind in range(da_i2d.shape[0]):
                                # 
                                da_now = da_i2d_baseline.isel(azimuthal_i2d=a_ind)
                                da_now_dropna = da_now.dropna(dim='radial_i2d')
                                try:
                                    baseline_now, params = pybaselines.Baseline(x_data=da_now_dropna.radial_i2d.values).arpls(da_now_dropna.values,lam=arpls_lam)
                                    # create baseline da by copying
                                    da_now_dropna_baseline = copy.deepcopy(da_now_dropna)
                                    da_now_dropna_baseline.values = baseline_now
                                    # now interpolate baseline da to original i2d radial range
                                    da_now_dropna_baseline_interpolated = da_now_dropna_baseline.interp(radial_i2d=da_i2d.radial_i2d)
                                    da_i2d_baseline[a_ind,:] = da_now_dropna_baseline_interpolated
                                except:
                                    # da_now.values[:] = np.nan
                                    da_i2d_baseline[a_ind,:] = da_now
                            self.ds['i2d_baseline'] = (da_i2d_baseline)
                            self.ds['i2d_baseline'].attrs['baseline_note'] = 'baseline is estimated with arpls'
                            self.ds['i2d_baseline'].attrs['arpls_lam'] = arpls_lam

                            if use_i2d_baseline:
                                self.ds['i1d_baseline'] = self.ds['i2d_baseline'].mean(dim='azimuthal_i2d').rename({'radial_i2d': 'radial'})
                                self.ds['i1d_baseline'].attrs['baseline_note'] = 'baseline is from i2d_baseline as available in this dataset. arpls is used'
                                self.ds['i1d_baseline'].attrs['arpls_lam'] = arpls_lam
                            else:
                                da_for_baseline = da_i2d.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')
                                baseline, params = pybaselines.Baseline(x_data=da_for_baseline.radial_i2d.values).arpls(da_for_baseline.values,lam=arpls_lam)
                                self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),dims=['radial_i2d'],
                                                                        coords={'radial_i2d':da_for_baseline.radial_i2d.values},
                                                                        attrs={'arpls_lam':arpls_lam}
                                                                        ).interp(radial_i2d=da_i2d.radial_i2d).rename({'radial_i2d': 'radial'})
                                self.ds['i1d_baseline'].attrs['baseline_note'] = 'baseline is estimated with arpls'
                                self.ds['i1d_baseline'].attrs['arpls_lam'] = arpls_lam

                        else:
                            da_for_baseline = da_i2d.mean(dim='azimuthal_i2d').dropna(dim='radial_i2d')
                            baseline, params = pybaselines.Baseline(x_data=da_for_baseline.radial_i2d.values).arpls(da_for_baseline.values,lam=arpls_lam)
                            self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),dims=['radial_i2d'],
                                                                    coords={'radial_i2d':da_for_baseline.radial_i2d.values},
                                                                    attrs={'arpls_lam':arpls_lam}
                                                                    ).interp(radial_i2d=da_i2d.radial_i2d).rename({'radial_i2d': 'radial'})
                            self.ds['i1d_baseline'].attrs['baseline_note'] = 'baseline is estimated with arpls'
                            self.ds['i1d_baseline'].attrs['arpls_lam'] = arpls_lam


                    else:
                        self.ds['i1d_baseline'] = (self.ds.i2d.mean(dim='azimuthal_i2d')*0)



                        



        if ('i2d' in self.ds.keys()):
            if roi_azimuthal_range is not None:
                self.ds['i1d'] = self.ds['i2d'].sel(azimuthal_i2d=slice(roi_azimuthal_range[0],roi_azimuthal_range[1])).mean(dim='azimuthal_i2d').rename({'radial_i2d': 'radial'})
                self.ds['i1d'].attrs =  {
                            'radial_unit':'q_A^-1',
                            'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
                            'ylabel':'Intensity (a.u.)',
                            'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
                            'roi_azimuthal_range':roi_azimuthal_range,
                        }
                self.ds['i2d'].attrs['roi_azimuthal_range'] = roi_azimuthal_range
            else:
                self.ds['i1d'] = self.ds['i2d'].mean(dim='azimuthal_i2d').rename({'radial_i2d': 'radial'})
                self.ds['i1d'].attrs =  {
                            'radial_unit':'q_A^-1',
                            'xlabel':'Scattering vector $q$ ($\AA^{-1}$)',
                            'ylabel':'Intensity (a.u.)',
                            'wavelength_in_angst':self.ds['i2d'].attrs['wavelength_in_meter']*10e9,
                        }





        if roi_radial_range is not None:
            self.ds = self.ds.sel(radial=slice(roi_radial_range[0],roi_radial_range[-1])).dropna(dim='radial')
        else:
            self.ds = self.ds.dropna(dim='radial')





        if normalize:

            # find normalization scale from i1d
            if ('i1d_baseline' in self.ds.keys()):
                da_baseline_sub = (self.ds.i1d-self.ds.i1d_baseline)
                normalization_multiplier = normalize_to*(1/max(da_baseline_sub.values))
                self.ds.i1d_baseline.values = self.ds.i1d_baseline.values*normalization_multiplier
                if ('i2d_baseline' in self.ds.keys()):
                    self.ds.i2d_baseline.values = self.ds.i2d_baseline.values*normalization_multiplier
                    self.ds.i2d_baseline.attrs['normalization_multiplier'] = normalization_multiplier
            else:  
                da = (self.ds.i1d)
                normalization_multiplier = normalize_to*(1/max(da.values))

            self.ds.i1d.values = self.ds.i1d.values*normalization_multiplier
            self.ds.i1d.attrs['normalization_multiplier'] = normalization_multiplier
            self.ds.i1d.attrs['normalized_to'] = normalize_to
            


            if ('i2d' in self.ds.keys()):
                self.ds.i2d.values = self.ds.i2d.values*normalization_multiplier  
                self.ds.i2d.attrs['normalization_multiplier'] = normalization_multiplier
                self.ds.i2d.attrs['normalized_to'] = normalize_to
        



        # if spotty_data_correction:
        #     da_diff = self.ds.i2d-self.ds.i2d_baseline 

        #     self.ds['i2d'] = (self.ds['i2d']).where(da_diff>=spotty_data_correction_threshold)

        #     i1d_attrs = copy.deepcopy(self.ds.i1d.attrs)
        #     self.ds['i1d'] = (((self.ds['i2d']).where(da_diff>=spotty_data_correction_threshold).mean(dim='azimuthal_i2d'))-spotty_data_correction_threshold).rename({'radial_i2d': 'radial'}).fillna(0)   
        #     self.ds['i1d'].attrs = i1d_attrs



        if plot:
            ds_plotter(ds=self.ds,  plot_hint = 'get_baseline') # type: ignore































































    def load_phases(self,
                    phases,
                    mp_rester_api_key='dHgNQRNYSpuizBPZYYab75iJNMJYCklB',
                    plot = True,
                    ):
        
        self.phases = {}
        for e,p in enumerate(phases):
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


            randstr = ''.join(random.choices(string.ascii_uppercase+string.digits, k=7)) 
            CifWriter(st,symprec=0.01).write_file('%s.cif'%randstr)
            # read cif
            with open('%s.cif'%randstr, 'r') as ciffile:
                ciffile_content = ciffile.read()
            self.ds.attrs['PhaseInd_%d_cif'%(e)] = ciffile_content
            self.ds.attrs['PhaseInd_%d_label'%(e)] = p['label']
            os.remove('%s.cif'%randstr)

        self.ds.attrs['num_phases'] = e+1
            
                
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
                               yshift_multiplier=0.01,
                        ):
        

        try:
            del self.gsasii_lib_directory
        except:
            pass
        try:
            del self.gpx
        except:
            pass

        for k in ['i1d_refined','i1d_gsas_background']:
            if k in self.ds.keys():
                del self.ds[k]

        self.yshift_multiplier = yshift_multiplier

        if gsasii_lib_directory is None:

            try:
                default_install_path = os.path.join(os.path.expanduser('~'),'g2full/GSAS-II/GSASII')
                sys.path += [default_install_path]
                import GSASIIscriptable as G2sc
                import GSASIIlattice as G2lat
                self.gsasii_lib_directory = default_install_path
            except:
                user_loc = input("Enter location of GSASII directory on your GSAS-II installation.")
                sys.path += [user_loc]
                try:
                    import GSASIIscriptable as G2sc
                    import GSASIIlattice as G2lat
                    self.gsasii_lib_directory = user_loc
                except:
                    try:
                        user_loc = input("\nUnable to import GSASIIscriptable. Please re-enter GSASII directory on your GSAS-II installation\n")
                        sys.path += [user_loc]
                        import GSASIIscriptable as G2sc
                        import GSASIIlattice as G2lat
                        self.gsasii_lib_directory = user_loc
                    except:
                        print("\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html")
                # else:
                    #clear_output()
                    # self.gsasii_lib_directory = gsasii_lib_directory
        else:
            if os.path.isdir(gsasii_lib_directory):
                sys.path += [gsasii_lib_directory]
                try:
                    import GSASIIscriptable as G2sc
                    import GSASIIlattice as G2lat
                    self.gsasii_lib_directory = gsasii_lib_directory
                except:
                    try:
                        gsasii_lib_directory = input("\nUnable to import GSASIIscriptable. Please enter GSASII directory on your GSAS-II installation\n")
                        sys.path += [gsasii_lib_directory]
                        import GSASIIscriptable as G2sc
                        import GSASIIlattice as G2lat
                        self.gsasii_lib_directory = gsasii_lib_directory
                    except:
                        gsasii_lib_directory = print("\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html")
                    # else:
                    #     #clear_output()
                    #     self.gsasii_lib_directory = gsasii_lib_directory
                # else:
                #     #clear_output()
                #     self.gsasii_lib_directory = gsasii_lib_directory
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

        if 'i1d_baseline' in self.ds.keys():
            if 'normalized_to' in self.ds.i1d.attrs:
                np.savetxt('%s/data.xy'%self.gsasii_run_directory,
                        fmt='%.7e',
                        X=np.column_stack( (np.rad2deg( 2 * np.arcsin( self.ds.i1d.radial.values * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) ), (self.ds.i1d-self.ds.i1d_baseline).values+self.yshift_multiplier*self.ds.i1d.attrs['normalized_to'] ) ))   
            else:
                np.savetxt('%s/data.xy'%self.gsasii_run_directory,
                        fmt='%.7e',
                        X=np.column_stack( (np.rad2deg( 2 * np.arcsin( self.ds.i1d.radial.values * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) ), (self.ds.i1d-self.ds.i1d_baseline).values+10 ) ))  
        else:
            if 'normalized_to' in self.ds.i1d.attrs:
                np.savetxt('%s/data.xy'%self.gsasii_run_directory,
                        fmt='%.7e',
                        X=np.column_stack( (np.rad2deg( 2 * np.arcsin( self.ds.i1d.radial.values * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) ), (self.ds.i1d).values+self.yshift_multiplier*self.ds.i1d.attrs['normalized_to'] ) ))   
            else:
                np.savetxt('%s/data.xy'%self.gsasii_run_directory,
                        fmt='%.7e',
                        X=np.column_stack( (np.rad2deg( 2 * np.arcsin( self.ds.i1d.radial.values * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) ), (self.ds.i1d).values ) ))  


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
        if 'normalized_to' in self.ds.i1d.attrs:
            self.gpx[pwdr_name]['Background'][0] = ['chebyschev-1', True, 3, self.yshift_multiplier*self.ds.i1d.attrs['normalized_to'], 0.0, 0.0]
        else:
            self.gpx[pwdr_name]['Background'][0] = ['chebyschev-1', True, 3, 0.0, 0.0, 0.0]


        if do_1st_refinement:
            ParDict = {'set': {'Background': {'refine': False,'type': 'chebyschev-1','no. coeffs': 1},
                               }
                        }
            self.gpx.set_refinement(ParDict)
            rwp_new, _ = self.gpx_refiner(remember_previous_ds=False,remember_previous_gpx=False)

            self.gpx.set_refinement({"set":{'LeBail': True}},phase='all')
            rwp_new, _ = self.gpx_refiner(remember_previous_ds=False,remember_previous_gpx=False)

            print('\nRwp from 1st refinement is = %.3f \n '%(rwp_new))







































    def refine_background(self,
                          num_coeffs=10,
                          background_type='chebyschev-1',
                          set_to_false_after_refinement=True,
                          plot=False,
                          ):
        """
        """
        ParDict = {'set': {'Background': {'refine': True,
                                        'type': background_type,
                                        'no. coeffs': num_coeffs
                                        }}}
        self.gpx.set_refinement(ParDict)

        rwp_new, rwp_previous = self.gpx_refiner()
        print('Background is refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            self.gpx.set_refinement({'set': {'Background': {'refine': False}}})
        self.gpx_saver()

        if plot:
            ds_plotter(ds=self.ds, ds_previous=self.ds_previous, plot_hint = 'refine_background') # type: ignore





    def refine_inst_parameters(self,
                               inst_pars_to_refine=['U', 'V', 'W'],
                               set_to_false_after_refinement=True,
                               verbose=False
                               ):
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



    def set_LeBail(self,
                   set_to=True,
                   phase='all',
                   refine=False,
                   
                   ):
        """
        """
        self.gpx.set_refinement({"set":{'LeBail': set_to}},phase=phase)
        if refine:
            rwp_new, rwp_previous = self.gpx_refiner()
            if set_to:
                print('After setting LeBail refinement to True, Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))
            else:
                print('After setting LeBail refinement to False, Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))
        else:
            pass
        self.gpx_saver()






                

    def refine_cell_params(self,
                           phase='all',
                           set_to_false_after_refinement=True,
                           update_ds_phases = True,
                           update_phases = True,
                           plot=False,
                           ):
        """
        """

        self.gpx.set_refinement({"set":{'Cell': True}},phase=phase)
        rwp_new, rwp_previous = self.gpx_refiner(self)

        if (phase=='all') or (phase==None):
            print('Cell parameters of all phases are refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))
        else:
            print('Cell parameters of %s phase is refined. Rwp is now %.3f (was %.3f)'%(self.gpx.phases()[phase].name,rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            self.gpx.set_refinement({"set":{'Cell': False}},phase=phase)

        self.gpx_saver()


        if update_ds_phases or update_phases:
            for e,p in enumerate(self.gpx.phases()):
                p.export_CIF(outputname='%s/%s_refined.cif'%(self.gsasii_run_directory,p.name))
                if update_ds_phases:
                    with open('%s/%s_refined.cif'%(self.gsasii_run_directory,p.name), 'r') as ciffile:
                        ciffile_content = ciffile.read()
                        self.ds.attrs['PhaseInd_%d_cif'%(e)] = ciffile_content
                if update_phases:
                    st = Structure.from_file('%s/%s_refined.cif'%(self.gsasii_run_directory,p.name))
                    self.phases[p.name] = st


    #     # refined phases
    #     self.refined_phases = {}
    #     for p in self.phases:
    #             st = Structure.from_file('%s/%s_refined.cif'%(self.gsasii_run_directory,p))
    #             self.refined_phases[p] = st

        if plot:
            ds_plotter(ds=self.ds, ds_previous=self.ds_previous,  plot_hint = 'refine_cell_params') # type: ignore
            




    def refine_strain_broadening(self,
                           phase='all',
                           set_to_false_after_refinement=True,
                           ):
        """
        """

        self.gpx.set_refinement({"set":{'Mustrain': {'refine':True}}},phase=phase)
        rwp_new, rwp_previous = self.gpx_refiner(self)

        if (phase=='all') or (phase==None):
            print('Strain broadening of all phases are refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))
        else:
            print('Strain broadening of %s phase is refined. Rwp is now %.3f (was %.3f)'%(self.gpx.phases()[phase].name,rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            self.gpx.set_refinement({"set":{'Mustrain': {'refine':False}}},phase=phase)

        self.gpx_saver()


    def refine_size_broadening(self,
                           phase='all',
                           set_to_false_after_refinement=True,
                           ):
        """
        """

        self.gpx.set_refinement({"set":{'Size': {'refine':True}}},phase=phase)
        rwp_new, rwp_previous = self.gpx_refiner(self)

        if (phase=='all') or (phase==None):
            print('Size broadening of all phases are refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))
        else:
            print('Size broadening of %s phase is refined. Rwp is now %.3f (was %.3f)'%(self.gpx.phases()[phase].name,rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            self.gpx.set_refinement({"set":{'Size': {'refine':False}}},phase=phase)

        self.gpx_saver()





    def refine_phase_fractions(self,
                           set_to_false_after_refinement=True,
                           ):
        """
        """
        self.gpx['PWDR data.xy']['Sample Parameters']['Scale'][1]=False
        for e,p in enumerate(self.phases):
            self.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][1]=True

        rwp_new, rwp_previous = self.gpx_refiner(self)
        print('Phase fractions of all phases are refined. Rwp is now %.3f (was %.3f)'%(rwp_new,rwp_previous))

        if set_to_false_after_refinement:
            self.gpx['PWDR data.xy']['Sample Parameters']['Scale'][1]=False
            for e,p in enumerate(self.phases):
                self.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][1]=True

        self.gpx_saver()




    def refine_preferred_orientation(self,
                                    phase='all',
                                    harmonics_order=4,
                                    set_to_false_after_refinement=True,
                                    ):
        
        import GSASIIlattice as G2lat
        phase_ind = phase
        L=harmonics_order

        for e,st in enumerate(self.phases):
            if e==phase_ind:
                coef_dict = {}
                sytsym=self.gpx['Phases'][st]['General']['SGData']['SGLaue']
                for l in range(2,L+1):
                    coeffs = G2lat.GenShCoeff(sytsym=sytsym,L=l)
                    try:
                        cst = coeffs[0][0][:6]
                        coef_dict[cst]=0.0
                    except:
                        pass

                self.gpx['Phases'][st]['Histograms']['PWDR data.xy']['Pref.Ori.'] = ['SH', 1.0, True, [0, 0, 1], L, coef_dict, [''], 0.1]
                
                rwp_new, rwp_previous = self.gpx_refiner(self)
                print('Preferred orientation for %s phase is refined. Rwp is now %.3f (was %.3f)'%(self.gpx.phases()[phase].name,rwp_new,rwp_previous)) 

                if set_to_false_after_refinement:     
                    self.gpx['Phases'][st]['Histograms']['PWDR data.xy']['Pref.Ori.'][2] = False





    def export_ds(self,
                  save_dir='.',
                  save_name='ds.nc'
                  ):
        """
        """
        self.ds.to_netcdf('%s/%s'%(save_dir,save_name),
                     engine="h5netcdf",
                     encoding={'i2d': {'zlib': True, 'complevel': 9}}) # pip install h5netcdf



    def fine_tune_gpx(self):
        """
        """
        subprocess.check_call(['%s/../../RunGSASII.sh'%self.gsasii_lib_directory, '%s/gsas.gpx'%self.gsasii_run_directory])
        import GSASIIscriptable as G2sc
        self.gpx = G2sc.G2Project(gpxfile='%s/gsas.gpx'%self.gsasii_run_directory)
        self.gpx.refine()


    def replace_gpx_with(self,
                         newgpx_to_replace
                         ):
        """
        """
        shutil.copy(newgpx_to_replace,'%s/gsas.gpx'%self.gsasii_lib_directory)
        import GSASIIscriptable as G2sc
        self.gpx = G2sc.G2Project(gpxfile='%s/gsas.gpx'%self.gsasii_run_directory)
        self.gpx.refine()


    def export_gpx_to(self,
                      export_to='gsas.gpx'
                         
                         ):
        """
        """
        shutil.copy('%s/gsas.gpx'%self.gsasii_run_directory,export_to)












    # def plot_refinement(self,
    #                 label_x = 0.85,
    #                 label_y = 0.8,
    #                 label_y_shift = -0.2,
    #                 ylogscale=True
    #                 ):


    #     wtSum = 0.0
    #     for e,p in enumerate(self.phases):
    #         mass = self.gpx['Phases'][p]['General']['Mass']
    #         phFr = self.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][0]
    #         wtSum += mass*phFr
    #     # for e,p in enumerate(sample.phases):
    #         # weightFr = sample.gpx['Phases'][p]['Histograms']['PWDR data.xy']['Scale'][0]*sample.gpx['Phases'][p]['General']['Mass']/wtSum



        
    #     for p in self.gpx.phases():
    #         p.export_CIF(outputname='%s/%s_refined.cif'%(self.gsasii_run_directory,p.name))

    #     if 'i2d' in self.ds.keys():
    #         fig = plt.figure(figsize=(8,6),dpi=128)
    #         mosaic = """
    #                     A
    #                     A
    #                     B
    #                     B
    #                     B
    #                     B
    #                     C
    #                     """
    #     else:
    #         fig = plt.figure(figsize=(8,4),dpi=128)
    #         mosaic = """
    #                     B
    #                     B
    #                     B
    #                     B
    #                     C
    #                     """

    #     ax_dict = fig.subplot_mosaic(mosaic, sharex=True)

    #     if 'i2d' in self.ds.keys():
    #         ax = ax = ax_dict["A"]
    #         if ylogscale:
    #             # np.log(self.ds.i2d).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
    #             np.log(self.ds.i2d-self.ds.i2d_baseline+self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys')
    #         else:
    #             (self.ds.i2d-self.ds.i2d_baseline+self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys')
    #         # self.ds.i2d.plot.imshow(ax=ax,robust=False,add_colorbar=False,cmap='viridis',vmin=0)
    #         ax.set_xlabel(None)
    #         ax.set_ylabel('Azimuthal')

    #         try:
    #             roi_xy = [self.ds.i1d.radial.values[0],self.ds.i1d.attrs['roi_azimuthal_range'][0]]
    #             roi_width = self.ds.i1d.radial.values[-1] - self.ds.i1d.radial.values[0]
    #             roi_height = self.ds.i1d.attrs['roi_azimuthal_range'][1] - self.ds.i1d.attrs['roi_azimuthal_range'][0]
    #             rect = matplotlib.patches.Rectangle(xy = roi_xy, width=roi_width, height=roi_height,color ='r',alpha=0.1)
    #             ax.add_patch(rect)
    #         except:
    #             pass

    #     ax = ax_dict["B"]
    #     if ylogscale:
    #         np.log(self.ds.i1d-self.ds.i1d_baseline+self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']).plot(ax=ax,color='k',label='Yobs.')
    #         np.log(self.ds.i1d_refined+self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']).plot(ax=ax, alpha=0.9, linewidth=1, color='y',label='Ycalc. (Rwp=%.3f)'%self.gpx['Covariance']['data']['Rvals']['Rwp'])   
    #         np.log(self.ds.i1d_gsas_background+self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']).plot(ax=ax, alpha=0.9, linewidth=1, color='r',label='Ybkg.')  
    #         # ax.fill_between(self.ds.i1d.radial.values,
    #         #                 self.ds.i1d.radial.values*0+np.log(self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']),
    #         #                 alpha=0.2,
    #         #                 color='C7',
    #         #                 )
            
    #         # ax.fill_between(self.ds.i1d.radial.values, 
    #         #                 y1=np.log((self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']+self.ds.i1d_gsas_background).values),
    #         #                 y2=np.log((self.yshift_multiplier*self.ds.i2d.attrs['normalized_to'])),
    #         #                 alpha=0.2,
    #         #                 color='C9',
    #         #                 label='Ybkg.'
    #         #                 )
    #         ax.set_ylabel('Log$_{10}$(data+10) (a.u.)')

    #     else:
    #         (self.ds.i1d-self.ds.i1d_baseline+self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']).plot(ax=ax,color='k',label='Yobs.')
    #         (self.ds.i1d_refined).plot(ax=ax, alpha=0.9, linewidth=1, color='y',label='Ycalc. (Rwp=%.3f)'%self.gpx['Covariance']['data']['Rvals']['Rwp'])   
    #         ax.fill_between(self.ds.i1d.radial.values,
    #                         self.ds.i1d.radial.values*0+self.yshift_multiplier*self.ds.i2d.attrs['normalized_to'],
    #                         alpha=0.2,
    #                         color='C7',
    #                         )
            
    #         ax.fill_between(self.ds.i1d.radial.values, 
    #                         y1=(self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']+self.ds.i1d_gsas_background).values,
    #                         y2=self.yshift_multiplier*self.ds.i2d.attrs['normalized_to'],
    #                         alpha=0.2,
    #                         color='C9',
    #                         label='Ybkg.'
    #                         )
    #         ax.set_ylabel('data+10 (a.u.)')





    #     ax.set_xlabel(None)

    #     # ax.set_ylim(bottom=-np.log(self.yshift_multiplier*self.ds.i2d.attrs['normalized_to']))
    #     ax.legend(loc='upper right')
    #     ax.set_xlim([self.ds.i1d.radial[0],self.ds.i1d.radial[-1]])









    #     xrdc = XRDCalculator(wavelength=self.ds.i1d.attrs['wavelength_in_angst'])








    #     # supplied phases
    #     for e,st in enumerate(self.phases):
    #         ps = xrdc.get_pattern(self.phases[st],
    #                             scaled=True,
    #                             two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([self.ds.i1d.radial.values[0],self.ds.i1d.radial.values[-1]]) * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) )
    #                             )
    #         refl_X, refl_Y = ((4 * np.pi) / (self.ds.i1d.attrs['wavelength_in_angst'])) * np.sin(np.deg2rad(ps.x) / 2), ps.y

    #         for i in refl_X:
    #             if 'i2d' in self.ds.keys():
    #                 ax_dict["A"].axvline(x=i,lw=0.3,color='C%d'%e)
    #             ax_dict["B"].axvline(x=i,lw=0.3,color='C%d'%e)
    #             ax_dict["C"].axvline(x=i,lw=0.3,color='C%d'%e)

    #         markerline, stemlines, baseline = ax_dict["C"].stem(refl_X,refl_Y,markerfmt=".")
    #         ax_dict["C"].set_ylim(bottom=0)

    #         plt.setp(stemlines, linewidth=0.5, color='C%d'%e)
    #         plt.setp(markerline, color='C%d'%e)

    #         weightFr = self.gpx['Phases'][st]['Histograms']['PWDR data.xy']['Scale'][0]*self.gpx['Phases'][st]['General']['Mass']/wtSum
    #         ax_dict["C"].text(label_x,label_y+e*label_y_shift,'%s (%.3f)'%(st,weightFr),color='C%d'%e,transform=ax_dict["C"].transAxes)






    #     # refined phases
    #     self.refined_phases = {}
    #     for p in self.phases:
    #             st = Structure.from_file('%s/%s_refined.cif'%(self.gsasii_run_directory,p))
    #             self.refined_phases[p] = st

    #     for e,st in enumerate(self.refined_phases):
    #         ps = xrdc.get_pattern(self.refined_phases[st],
    #                             scaled=True,
    #                             two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([self.ds.i1d.radial.values[0],self.ds.i1d.radial.values[-1]]) * ( (self.ds.i1d.attrs['wavelength_in_angst']) / (4 * np.pi))   ) )
    #                             )
    #         refl_X, refl_Y = ((4 * np.pi) / (self.ds.i1d.attrs['wavelength_in_angst'])) * np.sin(np.deg2rad(ps.x) / 2), ps.y

    #         for i in refl_X:
    #             if 'i2d' in self.ds.keys():
    #                 ax_dict["A"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
    #             ax_dict["B"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)
    #             ax_dict["C"].axvline(x=i,lw=0.3,linestyle='--',color='C%d'%e)

    #         markerline, stemlines, baseline = ax_dict["C"].stem(refl_X,refl_Y,markerfmt="+")
    #         ax_dict["C"].set_ylim(bottom=0)

    #         plt.setp(stemlines, linewidth=0.5, linestyle='--', color='C%d'%e)
    #         plt.setp(markerline, color='C%d'%e)

    #     ax_dict["C"].set_xlabel(self.ds.i1d.attrs['xlabel'])





































