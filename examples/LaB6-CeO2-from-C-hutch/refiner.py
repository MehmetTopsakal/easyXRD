# -*- coding: utf-8 -*-
"""refiner.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S2_G_5hcUVqiGcBNCAXiRskcIlLy-caI

The purpose of this notebook is to demonstrate how to refine with locally defined GSAS-II functions.

# Imports
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

import sys
sys.path += ['../../../easyXRD']

import easyxrd

from easyxrd.core import exrd

import xarray as xr
import numpy as np
import pybaselines
from copy import deepcopy

# Commented out IPython magic to ensure Python compatibility.
# importing matplotlib for plots.
# %matplotlib widget
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.constrained_layout.use'] = True

# we use pyFAI for integrations: https://pyfai.readthedocs.io/en/v2023.1/
# It there exists a poni file and mask, we can load them like this:
import pyFAI,fabio

ai_file = 'tiff_files/_calibration.poni'
mask_file = 'tiff_files/_mask.edf'

ai = pyFAI.load(ai_file)
mask = fabio.open(mask_file).data




sample_Kapton = exrd()
# sample_Kapton.load_xrd_data(from_tiff_file='tiff_files/Kapton.tiff',
#                             ai = ai,
#                             mask = mask,
#                             radial_range=(0.2,10.2),
#                             plot=True
#                             )
# sample_Kapton.export_ds(save_dir='nc_files',save_name='Kapton.nc')
sample_Kapton.load_xrd_data(from_nc_file='nc_files/Kapton.nc',plot=False)

sample_Air = exrd()
# sample_Air.load_xrd_data(from_tiff_file='tiff_files/Air_scattering.tiff',
#                             ai = ai,
#                             mask = mask,
#                             radial_range=(0.2,10.2),
#                             plot=True
#                             )
# sample_Air.export_ds(save_dir='nc_files',save_name='Air.nc')
sample_Air.load_xrd_data(from_nc_file='nc_files/Air.nc',plot=False)

sample_LaB6 = exrd()
# sample_LaB6.load_xrd_data(from_tiff_file='tiff_files/NIST-LaB6.tiff',
#                             ai = ai,
#                             mask = mask,
#                             radial_range=(0.2,10.2),
#                             plot=True
#                             )
# sample_LaB6.export_ds(save_dir='nc_files',save_name='NIST-LaB6.nc')
sample_LaB6.load_xrd_data(from_nc_file='nc_files/NIST-LaB6.nc',plot=False)

sample_CeO2 = exrd()
# sample_CeO2.load_xrd_data(from_tiff_file='tiff_files/NIST-CeO2.tiff',
#                             ai = ai,
#                             mask = mask,
#                             radial_range=(0.2,10.2),
#                             plot=True
#                             )
# sample_CeO2.export_ds(save_dir='nc_files',save_name='NIST-CeO2.nc')
sample_CeO2.load_xrd_data(from_nc_file='nc_files/NIST-CeO2.nc',plot=False)

sample_mix = exrd()
# sample_mix.load_xrd_data(from_tiff_file='tiff_files/NIST-LaB6-CeO2-mix.tiff',
#                             ai = ai,
#                             mask = mask,
#                             radial_range=(0.2,10.2),
#                             plot=True
#                             )
# sample_mix.export_ds(save_dir='nc_files',save_name='NIST-LaB6-CeO2-mix.nc')
sample_mix.load_xrd_data(from_nc_file='nc_files/NIST-LaB6-CeO2-mix.nc',plot=False)

phases_LaB6 = [
        {"mp_id":'none', "cif":'_cifs/LaB6_a=4.1568_NIST_value.cif', "label":"LaB6", "scale":1, "scale_a":1, "scale_b":1, "scale_c":1},
        ]

phases_CeO2 = [
        {"mp_id":'none', "cif":'_cifs/CeO2_a=5.4113_NIST_value.cif', "label":"CeO2", "scale":1, "scale_a":1, "scale_b":1, "scale_c":1},
        ]

phases_mix = [
        {"mp_id":'none', "cif":'_cifs/LaB6_a=4.1568_NIST_value.cif', "label":"LaB6", "scale":1, "scale_a":1, "scale_b":1, "scale_c":1},
        {"mp_id":'none', "cif":'_cifs/CeO2_a=5.4113_NIST_value.cif', "label":"CeO2", "scale":1, "scale_a":1, "scale_b":1, "scale_c":1},
         ]







#
#
# sample = sample_mix
# sample.get_baseline(i1d_bkg=sample_Kapton.ds.i1d,
#                     arpls_lam=1e5,
#                     use_arpls=False,
#                     roi_radial_range=[1.25,8.5],
#                     plot=True)
#
# sample.load_phases(phases=phases_mix,plot=True)
# # INITIAL REFINEMENT
# sample.setup_gsas2_calculator(instprm_from_gpx='gsas_LaB6.gpx')
# sample.set_LeBail()
# import time
# time.sleep(10)
# sample.refine_background(num_coeffs=10)
# import time
# time.sleep(10)
# sample.refine_cell_params()
# import time
# time.sleep(10)
# sample.plot_refinement()
# import time
# time.sleep(10)
# # SET TO RIETVELD
# sample.set_LeBail(set_to=False,refine=True)
# sample.plot_refinement()






sample = sample_mix
sample.get_baseline(i1d_bkg=sample_Kapton.ds.i1d,
                    arpls_lam=1e5,
                    use_arpls=False,
                    roi_radial_range=[1.25,8.5],
                    plot=True)

sample.load_phases(phases=phases_mix,plot=True)

sample.setup_gsas2_calculator(instprm_from_gpx='gsas_LaB6.gpx')

sample.refine_size_broadening(set_to_false_after_refinement=False)


# sample.set_LeBail(set_to=True,phase_ind=1,refine=False)
# sample.refine_background(num_coeffs=10)


# sample.set_LeBail()
# import time
# time.sleep(10)
# sample.refine_background(num_coeffs=10)
# import time
# time.sleep(10)
# sample.refine_cell_params()
# import time
# time.sleep(10)
# sample.plot_refinement()
# import time
# time.sleep(10)
# # SET TO RIETVELD
# sample.set_LeBail(set_to=False,refine=True)
# sample.plot_refinement()



