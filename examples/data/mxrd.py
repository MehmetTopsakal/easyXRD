from scipy.signal import savgol_filter
import pybaselines

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifWriter


class exrd():
    def __init__(self):
        pass



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
                            export_extension='exported.cif'
                            ):
        if phase_ind is None:
            for e,st in enumerate(self.phases):
                CifWriter(self.phases[st],symprec=0.01).write_file("%s/%s_%s"%(export_to,st,export_extension))
        else:
            for e,st in enumerate(self.phases):
                if e == phase_ind:
                    CifWriter(self.phases[st],symprec=0.01).write_file("%s/%s_%s"%(export_to,st,export_extension))
            

                
