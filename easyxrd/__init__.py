
import sys
import os
import subprocess

import importlib
from importlib.metadata import version




print('\n\nChecking required packages:\n')
# These are big python libraries that we will need in pySULI.
# If the required library doesn't exist, we install it via pip

required_big_packages   = {'numpy','scipy','xarray','ipympl','pymatgen','pyFAI'}

for rp in required_big_packages:
    try:
        globals()[rp] = importlib.import_module(rp)
        print('---%s package with version %s is available and imported '%(rp,version(rp)))
    except:
        print('\n\nInstalling %s'%rp)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', rp])
        globals()[rp] = importlib.import_module(rp)

# these are other packages that are usually installed by big packages above. 
# Otherwise, we pip-install them

required_other_packages   = {'fabio','pandas','mp_api'}

for rp in required_other_packages:
    try:
        globals()[rp] = importlib.import_module(rp)
        print('---%s package with version %s is available and imported '%(rp,version(rp)))
    except:
        print('\n\nInstalling %s'%rp)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', rp])
        globals()[rp] = importlib.import_module(rp)






# # Setting up gsas2_scratch folder
# user_home = os.path.expanduser('~')
# if not os.path.isdir(os.path.join(user_home,'.gsas2_scratch')):
#     os.mkdir(os.path.join(user_home,'.gsas2_scratch'))





# # Setting up gsas2_lib folder

# gsasii_loc = input("\n\nEnter location of GSASII folder on your GSAS-II installation\n\n")
# sys.path += [gsasii_loc]
# try:
#     import GSASIIscriptable as G2sc
#     import pybaselines # this comes with gsas2_package
# except:
#     gsasii_loc = input("\n\nUnable to import GSASIIscriptable. Please re-enter GSASII folder on your GSAS-II installation\n\n")
#     try:
#         import GSASIIscriptable as G2sc
#         import pybaselines # this comes with gsas2_package
#     except:
#         print('Still unable to import GSASIIscriptable')
#         print('Please check GSAS-II installations instructions here: https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html ')





def set_defaults(name, val):
    ''' set a global variable.'''
    global pysuli_defaults
    pysuli_defaults[name] = val

def print_defaults():
    for key, val in pysuli_defaults.items():
        print("- {} : {}".format(key, val))



# defaults
pysuli_defaults  = dict()
# pysuli_defaults['gsas2_scratch'] = os.path.join(user_home,'.gsas2_scratch')
# pysuli_defaults['gsas2_lib'] = gsasii_loc

print("\n\nImported pysuli with following configuration:\n")
print_defaults()








# other imports and aliases
import time

import numpy as np
import xarray as xr
from IPython.display import clear_output


import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})



