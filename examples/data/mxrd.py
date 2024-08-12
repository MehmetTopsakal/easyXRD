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




    def gpx_refiner(self):
            if self.verbose:
                self.gpx.refine()
            else:
                with HiddenPrints():
                    self.gpx.refine()


    def gpx_saver(self):
            if self.verbose:
                self.gpx.save()
            else:
                with HiddenPrints():
                    self.gpx.save()




    def read_data(self,
                nc_file=None,
                txt_file=None,
                txt_file_wavelength_in_nm=1.814e-11,
                txt_file_comments='#',
                txt_file_skiprows=0,
                txt_file_usecols=(0,1),
                txt_file_radial_unit='tth',
                radial_range=None,
                plot=True,
                ax_inp=None

                ):


        if (nc_file is None) and (txt_file is None):
            print('Please enter a valid nc_file or txt file path to read data')
            return
        elif (nc_file is not None) and (txt_file is None)  :
            if os.path.isfile(nc_file):
                try:
                    with xr.open_dataset(nc_file) as self.ds:
                        pass
                except Exception as exc:
                    print('Unable to read %s \nPlease check %s is a valid xarray nc file\n\n'%(nc_file,nc_file))
                    print('Error msg from xarray:\n%s'%exc)
                    return
            else:
                print('%s does not exist. Please check the file path. '%nc_file)
                return
        elif (nc_file is None) and (txt_file is not None)  :
            if os.path.isfile(txt_file):
                try:
                    X,Y = np.loadtxt(txt_file,comments=txt_file_comments,skiprows=txt_file_skiprows,usecols=txt_file_usecols,unpack=True)
                    if txt_file_radial_unit.lower()[0] == 't':
                        X = ((4 * np.pi) / (txt_file_wavelength_in_nm*10e9)) * np.sin(np.deg2rad(X) / 2)
                    elif txt_file_radial_unit.lower()[0] == 'q':
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
                                                           'wavelength_in_nm':txt_file_wavelength_in_nm,
                                                           'i1d_from':txt_file,
                                                            })
                except Exception as exc:
                    print('Unable to read %s \nPlease check %s is a valid plain text file\n\n'%(txt_file,txt_file))
                    print('Error msg from np.loadtxt:\n%s'%exc)
                    return
            else:
                print('%s does not exist. Please check the file path.'%txt_file)
                return


        if radial_range is not None:
            self.ds = self.ds.sel(radial=slice(radial_range[0],radial_range[1]))



        if plot:
            if ax_inp == None:
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                self.ds.i1d.plot(ax=ax)
                ax.set_xlabel(self.ds.i1d.attrs['xlabel'])
                ax.set_ylabel(self.ds.i1d.attrs['ylabel'])
                # ax.set_xlim([1,11])
            else:
                self.ds.i1d.plot(ax=ax_inp)
                ax_inp.set_xlabel(self.ds.i1d.attrs['xlabel'])
                ax_inp.set_ylabel(self.ds.i1d.attrs['ylabel'])





    def get_baseline(self,
                     i1d_bkg=None,
                     use_arpls=True,
                     arpls_lam=1e5,
                     plot=True
                     ):

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
                print('here')

            if i1d_bkg_useable:
                bkg_scale = self.ds['i1d'].values[0]/i1d_bkg.values[0]
                while (min((self.ds['i1d'].values-bkg_scale*i1d_bkg.values)) < 0):
                    bkg_scale = bkg_scale*0.99
                if use_arpls:
                    baseline = pybaselines.Baseline(x_data=self.ds['i1d'].radial.values).arpls((self.ds['i1d']-bkg_scale*i1d_bkg).values, lam=arpls_lam)[0]
                    self.ds['i1d_baseline'] = xr.DataArray(data=(baseline+bkg_scale*i1d_bkg.values),dims=['radial'],coords={'radial':self.ds.i1d.radial},attrs={'arpls_lam':arpls_lam})
                else:
                    baseline = bkg_scale*i1d_bkg
                    self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),dims=['radial'],coords={'radial':self.ds.i1d.radial},attrs={'arpls_lam':arpls_lam})
                
            else:
                baseline = pybaselines.Baseline(x_data=self.ds['i1d'].radial.values).arpls((self.ds['i1d']).values, lam=arpls_lam)[0]
                self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),dims=['radial'],coords={'radial':self.ds.i1d.radial},attrs={'arpls_lam':arpls_lam})
        else:
            print('i1d_bkg is not provided. Using arpls to find the baseline\n\n')
            baseline = pybaselines.Baseline(x_data=self.ds['i1d'].radial.values).arpls((self.ds['i1d']).values, lam=arpls_lam)[0]
            self.ds['i1d_baseline'] = xr.DataArray(data=(baseline),dims=['radial'],coords={'radial':self.ds.i1d.radial},attrs={'arpls_lam':arpls_lam})

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



            if 'i2d' in self.ds.keys():
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
            (self.ds.i1d).plot(ax=ax,label='data')
            (self.ds.i1d_baseline).plot(ax=ax,label='baseline')
            ax.set_yscale('log')
            ax.set_xlabel(self.ds.i1d.attrs['xlabel'])
            ax.set_ylabel(self.ds.i1d.attrs['ylabel'])
            ax.legend()

            if 'i2d' in self.ds.keys():
                ax = ax = ax_dict["B"]
                np.log(self.ds.i2d).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
                ax.set_xlabel(None)
                ax.set_ylabel('Azimuthal')

            ax = ax = ax_dict["C"]
            np.log(self.ds.i1d-self.ds.i1d_baseline+10).plot(ax=ax,color='k')
            ax.fill_between(self.ds.i1d.radial.values, self.ds.i1d.radial.values*0+np.log(10),alpha=0.2)
            ax.set_xlabel(None)
            ax.set_ylabel('Log$_{10}$(data-baseline+10) (a.u.)')
            ax.set_ylim(bottom=np.log(8))
            
            ax = ax = ax_dict["D"]
            (self.ds.i1d-self.ds.i1d_baseline).plot(ax=ax,color='k')
            ax.axhline(y=0,alpha=0.5,color='y')
            ax.set_xlabel(self.ds.i1d.attrs['xlabel'])
            ax.set_ylim([-1,1])











    def load_phases(self,
                    phases,
                    mp_rester_api_key='dHgNQRNYSpuizBPZYYab75iJNMJYCklB',
                    plot = True,
                    label_x = 0.9,
                    label_y = 0.8,
                    label_y_shift = -0.2
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
                ax.set_xlabel(None)
                ax.set_ylabel('Azimuthal')

            ax = ax_dict["B"]
            np.log(self.ds.i1d-self.ds.i1d_baseline+10).plot(ax=ax,color='k')
            ax.fill_between(self.ds.i1d.radial.values, self.ds.i1d.radial.values*0+np.log(10),alpha=0.2)
            ax.set_xlabel(None)
            ax.set_ylabel('Log$_{10}$(data-baseline+10) (a.u.)')
            ax.set_ylim(bottom=np.log(8))

            xrdc = XRDCalculator(wavelength=self.ds.i1d.attrs['wavelength_in_nm']*10e9)

            for e,st in enumerate(self.phases):
                ps = xrdc.get_pattern(self.phases[st],
                                    scaled=True,
                                    two_theta_range=np.rad2deg( 2 * np.arcsin( np.array([self.ds.i1d.radial.values[0],self.ds.i1d.radial.values[-1]]) * ( (self.ds.i1d.attrs['wavelength_in_nm']*10e9) / (4 * np.pi))   ) )
                                    )
                refl_X, refl_Y = ((4 * np.pi) / (self.ds.i1d.attrs['wavelength_in_nm']*10e9)) * np.sin(np.deg2rad(ps.x) / 2), ps.y

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

            ax_dict["C"].set_xlabel(self.ds.i1d.attrs['xlabel'])



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
                               instprm_Zero=0,                                                                   
                               instprm_U=173,                                  
                               instprm_V=-1,                                 
                               instprm_W=1,   
                               instprm_X=0, 
                               instprm_Y=-7, 
                               instprm_Z=0, 
                               instprm_SHL=0.002,   
                               do_1st_refinement=True,
                        ):
        
        if gsasii_lib_directory is None:
            user_loc = input("Enter location of GSASII directory on your GSAS-II installation.")
            sys.path += [gsasii_lib_directory]
            try:
                import GSASIIscriptable as G2sc
            except:
                try:
                    gsasii_lib_directory = input("\nUnable to import GSASIIscriptable. Please re-enter GSASII directory on your GSAS-II installation\n")
                    sys.path += [gsasii_lib_directory]
                    import GSASIIscriptable as G2sc
                except:
                    gsasii_lib_directory = input("\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html")
                    return
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
                    except:
                        gsasii_lib_directory = input("\n Still unable to import GSASIIscriptable. Please check GSAS-II installation notes here: \n\n https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html")
                        return
                    else:
                        #clear_output()
                        self.gsasii_lib_directory = gsasii_lib_directory
                else:
                    #clear_output()
                    self.gsasii_lib_directory = gsasii_lib_directory
            else:
                print('%s does NOT exist. Please check!'%gsasii_lib_directory)
                return



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




        # np.savetxt('%s/data.qye'%self.gsasii_run_directory,
        #            fmt='%.7e',
        #            X=np.column_stack( (self.ds.i1d.radial.values, (self.ds.i1d-self.ds.i1d_baseline).values) ))
        np.savetxt('%s/data.xy'%self.gsasii_run_directory,
                   fmt='%.7e',
                   X=np.column_stack( (np.rad2deg( 2 * np.arcsin( self.ds.i1d.radial.values * ( (self.ds.i1d.attrs['wavelength_in_nm']*10e9) / (4 * np.pi))   ) ), (self.ds.i1d-self.ds.i1d_baseline).values+10 ) ))        


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
                    f.write('Lam:%s\n'%(self.ds.i1d.attrs['wavelength_in_nm']*10e9))
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
                print('gpx file for reading instrument parameters do net exist. Plewase check the path')
                return

        else:
            with open('%s/gsas.instprm'%self.gsasii_run_directory, 'w') as f:
                f.write('#GSAS-II instrument parameter file; do not add/delete items!\n')
                f.write('Type:PXC\n')
                f.write('Bank:1.0\n')
                f.write('Lam:%s\n'%(self.ds.i1d.attrs['wavelength_in_nm']*10e9))
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
            self.gpx_previous = copy.deepcopy(self.gpx)
            self.gpx_refiner()
            rwp_1st = self.gpx['Covariance']['data']['Rvals']['Rwp']
            print('\nRwp from 1st refinement is = %.3f \n '%(rwp_1st))


        self.gpx_saver()








    def set_LeBail(self, set_to=True, phase_ind=None, refine=True):
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
            rwp_now = self.gpx['Covariance']['data']['Rvals']['Rwp']
            print('\nset_LeBail output:\nRwp is now %.3f (was %.3f)'%(rwp_now,rwp_previous))
        self.gpx_saver()


    def refine_background(self, num_coeffs=10, background_type='chebyschev-1', set_to_false_after_refinement=True):
        ParDict = {'set': {'Background': {'refine': True,
                                        'type': background_type,
                                        'no. coeffs': num_coeffs
                                        }}}
        self.gpx.set_refinement(ParDict)

        self.gpx_previous = copy.deepcopy(self.gpx)
        rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
        self.gpx_refiner()
        rwp_now = self.gpx['Covariance']['data']['Rvals']['Rwp']

        print('\nrefine_background output:\nRwp is now %.3f (was %.3f)'%(rwp_now,rwp_previous))
        if set_to_false_after_refinement:
            self.gpx.set_refinement({'set': {'Background': {'refine': False}}})
        self.gpx_saver()
                

    def refine_cell_params(self, phase_ind=None, set_to_false_after_refinement=True):

        for e,p in enumerate(self.gpx.phases()):
            if phase_ind is None:
                self.gpx['Phases'][p.name]['General']['Cell'][0]= True
            else:
                if e == phase_ind:
                    self.gpx['Phases'][p.name]['General']['Cell'][0]= True

        self.gpx_previous = copy.deepcopy(self.gpx)
        rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
        self.gpx_refiner()
        rwp_now = self.gpx['Covariance']['data']['Rvals']['Rwp']

        print('\nrefine_cell_params output:\nRwp is now %.3f (was %.3f)'%(rwp_now,rwp_previous))
        if set_to_false_after_refinement:
            phases = self.gpx.phases()
            for p in phases:
                self.gpx['Phases'][p.name]['General']['Cell'][0]= False
        self.gpx_saver()


    def refine_strain_broadening(self,set_to_false_after_refinement=True):

        ParDict = {'set': {'Mustrain': {'type': 'isotropic',
                                        'refine': True
                                        }}}
        self.gpx.set_refinement(ParDict)


        self.gpx_previous = copy.deepcopy(self.gpx)
        rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
        self.gpx_refiner()
        rwp_now = self.gpx['Covariance']['data']['Rvals']['Rwp']

        print('\nrefine_strain_broadening output:\nRwp is now %.3f (was %.3f)'%(rwp_now,rwp_previous))

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

        self.gpx_previous = copy.deepcopy(self.gpx)
        rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
        self.gpx_refiner()
        rwp_now = self.gpx['Covariance']['data']['Rvals']['Rwp']

        print('\nrefine_size_broadening output:\nRwp is now %.3f (was %.3f)'%(rwp_now,rwp_previous))


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

        self.gpx_previous = copy.deepcopy(self.gpx)
        rwp_previous = self.gpx_previous['Covariance']['data']['Rvals']['Rwp']
        self.gpx_refiner()
        rwp_now = self.gpx['Covariance']['data']['Rvals']['Rwp']

        print('\nrefine_inst_parameters output:')
        print('%s parameters are refined'%inst_pars_to_refine)
        print('Rwp is now %.3f (was %.3f)'%(rwp_now,rwp_previous))

        if set_to_false_after_refinement:
            ParDict = {"clear": {'Instrument Parameters': ['X', 'Y', 'Z', 'Zero', 'SH/L', 'U', 'V', 'W']}}
            self.gpx.set_refinement(ParDict)
        self.gpx_saver()
            







