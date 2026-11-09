"""Numerical and state regressions from both November 2026 package reviews."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from easyxrd import exrd, HiddenPrints
from easyxrd.baseline import _optimize_bkg_scale, _compute_i2d_baseline
from easyxrd.plotters import _calc_reflection_pattern, _PATTERN_CACHE, i1d_plotter
from easyxrd.utils import load_gsas_module
from tests.test_refinement import sample, phase, configurable_sample


def profile(q=None, y=None):
    q = np.linspace(1, 5, 40) if q is None else np.asarray(q)
    y = 10 + np.sin(q) if y is None else np.asarray(y)
    s = exrd()
    s.load_xrd_data(from_i1d_array=np.column_stack([q, y]), plot=False)
    return s


@pytest.mark.parametrize('obs,expected', [([100,1,1], 1.), ([10,5,1,5],1.), ([1,1,1],1.)])
def test_background_scale_does_not_depend_on_first_point(obs, expected):
    assert _optimize_bkg_scale(obs, np.ones(len(obs))) == expected
    assert _optimize_bkg_scale(obs[::-1], np.ones(len(obs))) == expected


def test_background_negative_noise_never_produces_negative_scale():
    with pytest.warns(UserWarning, match='No nonnegative'):
        assert _optimize_bkg_scale([-1, 3], [1,1]) == 0


def test_misaligned_background_preserves_everything():
    s, b = profile([1,2,3]), profile([1.1,2.1,3.1])
    s.ds['i1d_refined'] = s.ds.i1d.copy()
    before = s.ds.copy(deep=True)
    with pytest.raises(ValueError, match='grids must match'):
        s.get_baseline(input_bkg=b, use_iarpls=False, plot=False)
    xr.testing.assert_identical(s.ds, before)


def test_background_noop_is_really_noop():
    s = profile(); s.ds['i1d_refined'] = s.ds.i1d.copy()
    before = s.ds.copy(deep=True)
    s.get_baseline(use_iarpls=False, plot=False)
    xr.testing.assert_identical(s.ds, before)


def test_independent_background_variance():
    s, b = profile(y=np.ones(40)*10), profile(y=np.ones(40)*2)
    s.ds['i1d_variance'] = xr.full_like(s.ds.i1d, 4)
    b.ds['i1d_variance'] = xr.full_like(b.ds.i1d, 3)
    s.get_baseline(input_bkg=b, use_iarpls=False, plot=False)
    np.testing.assert_allclose(s.ds.i1d_baseline, 10)
    np.testing.assert_allclose(s.ds.i1d_baseline_variance, 75)
    np.testing.assert_allclose(s.ds.i1d_variance, 4)


def test_row_fit_failure_is_explicit(monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError('numeric failure')
    monkeypatch.setattr('pybaselines.Baseline.iarpls', fail)
    da = xr.DataArray([[1.,3.,1.]], dims=['azimuthal_i2d','radial_i2d'], coords={'radial_i2d':[1.,2.,3.]})
    with pytest.raises(RuntimeError, match='sector baseline'):
        _compute_i2d_baseline(da)


@pytest.mark.parametrize('kwargs', [{}, {'from_i1d_array':np.ones((3,2)), 'from_nc_file':'missing'},
    {'from_txt_file':'not-found.xy'}, {'from_i1d_array':[[2,1],[1,2]]},
    {'from_i1d_array':[[1,1],[2,2]],'i1d_array_radial_unit':'wrong'},
    {'from_img_array':np.ones((3,3))}])
def test_bad_load_does_not_clear_existing_data(kwargs):
    s=profile(); before=s.ds.copy(deep=True)
    with pytest.raises((ValueError,FileNotFoundError)):
        s.load_xrd_data(**kwargs, plot=False)
    xr.testing.assert_identical(s.ds,before)


def test_sigma_column_preserved_through_netcdf(tmp_path):
    s=exrd(); s.load_xrd_data(from_i1d_array=[[1.,-2.,3.],[2.,4.,2.]],plot=False)
    np.testing.assert_equal(s.ds.i1d_variance.values,[9,4])
    s.export_ds_to(tmp_path/'variance.nc')
    t=exrd(); t.load_xrd_data(from_nc_file=tmp_path/'variance.nc',plot=False)
    xr.testing.assert_identical(s.ds,t.ds)


def test_invalid_export_coordinates_and_modes(tmp_path):
    s=profile()
    with pytest.raises(ValueError,match='mode'):
        s.export_i1d_to(tmp_path/'bad.xy',mode='typo',header=True)
    s.ds.i1d.attrs['wavelength_in_angst']=100.
    with pytest.raises(ValueError,match='q must'):
        s.export_i1d_to(tmp_path/'bad.xy')
    with pytest.raises(ValueError,match='Provide to'):
        s.export_ds_to()
    assert not (tmp_path/'bad.xy').exists()


def test_cache_distinguishes_atomic_positions_and_bounds_memory():
    calc=XRDCalculator(wavelength=1.5406)
    a=Structure(Lattice.cubic(4),['Si','Si'],[[0,0,0],[.5,.5,.5]])
    b=Structure(Lattice.cubic(4),['Si','Si'],[[0,0,0],[.25,.25,.25]])
    _PATTERN_CACHE.clear()
    x1,_,_=_calc_reflection_pattern(calc,a,1.5406,1,6)
    x2,y2,_=_calc_reflection_pattern(calc,b,1.5406,1,6)
    assert len(x1)==7 and len(x2)==11
    y2[:]=0
    assert np.any(_calc_reflection_pattern(calc,b,1.5406,1,6)[1]>0)


@pytest.mark.parametrize('symmetry', ['m3m','mmm','2/m','-1'])
@pytest.mark.parametrize('order', [4,10])
def test_po_coefficients_match_gsas_and_honor_flags(tmp_path,symmetry,order):
    lat=pytest.importorskip('GSASII.GSASIIlattice')
    s=configurable_sample(tmp_path)
    s.gpx.phases()[0].data['General']['SGData']['SGLaue']=symmetry
    s.set_preferred_orientation_refinement(False,harmonics_order_to=order,save_gpx=False)
    po=s.gpx.phases()[0].data['Histograms'][s._histogram_name]['Pref.Ori.']
    assert list(po[5])==lat.GenSHCoeffT(symmetry,'0',order,False)
    assert po[2] is False
    s.set_preferred_orientation_refinement(True,phase_ind='all',harmonics_order_to=order,save_gpx=False)
    assert s.gpx.phases()[0].data['Histograms'][s._histogram_name]['Pref.Ori.'][2] is True


def test_exception_after_mutation_restores_file_and_flags(tmp_path,monkeypatch):
    s=configurable_sample(tmp_path);s.last_refinement_accepted=True
    shared={'scale':.001}; s.ds.attrs['lastShifts']=shared
    s.gpx.data['Covariance']['data']['Rvals']['lastShifts']=shared
    s.gpx.save(); before_file=s.gpx.filename.read_bytes(); before_ds=s.ds.copy(deep=True)
    def fail():
        shared['scale']=999.
        s.gpx.data['Covariance']['data']['Rvals']['Rwp']=999.
        s.gpx.save()
        raise RuntimeError('failed after write')
    monkeypatch.setattr(s.gpx,'refine',fail)
    with pytest.raises(RuntimeError,match='after write'):
        s.refine_background()
    assert not s.last_refinement_accepted
    assert s.ds.refinement_status.item()=='error'
    xr.testing.assert_identical(s.ds.drop_dims('refinement_step'),before_ds)
    assert s.gpx.filename.read_bytes()==before_file


def test_default_plot_and_unfitted_previous_do_not_crash():
    s=profile(); s.plot(close=True)
    old=s.ds.copy(deep=True)
    s.ds['i1d_refined']=s.ds.i1d.copy();s.ds['i1d_gsas_background']=xr.zeros_like(s.ds.i1d)
    fig,ax=plt.subplots()
    i1d_plotter(s.ds,ax,ds_previous=old)
    plt.close(fig)
    assert not plt.get_fignums()


def test_error_only_history_plot():
    s=exrd();s.ds=xr.Dataset({'rwp_history':('refinement_step',[np.nan]),'rwp_attempted_history':('refinement_step',[np.nan])},coords={'refinement_step':[1],'refinement_status':('refinement_step',['error']),'refinement_label':('refinement_step',['refine'])})
    fig,_=s.plot_refinement_history(return_fig=True)
    plt.close(fig)


def test_nested_hidden_prints(capsys):
    h=HiddenPrints()
    with h:
        print('outer')
        with h: print('inner')
        print('outer again')
    print('visible')
    assert capsys.readouterr().out=='visible\n'


def test_package_import_does_not_load_gsas_or_write_scratch(tmp_path):
    scratch=tmp_path/'absent'
    env=os.environ.copy();env['EASYXRD_SCRATCH']=str(scratch)
    subprocess.run([sys.executable,'-c',"import easyxrd, sys; assert 'GSASII.GSASIIscriptable' not in sys.modules"],env=env,check=True)
    assert not scratch.exists()


def test_real_pyfai_1d_matches_with_and_without_cake():
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import Detector
    ai=AzimuthalIntegrator(dist=.1,poni1=.003,poni2=.004,detector=Detector(pixel1=1e-4,pixel2=1e-4),wavelength=1e-10)
    y,x=np.indices((100,100)); img=(100+2*x+y).astype(float)
    kwargs=dict(from_img_array=img,ai=ai,mask=(x>60)&(y<50),radial_range=(.05,.35),delta_q=.01,npt_azimuthal=20,plot=False,error_model='poisson')
    a,b=exrd(),exrd();a.load_xrd_data(integrate2d=False,**kwargs);b.load_xrd_data(integrate2d=True,**kwargs)
    xr.testing.assert_identical(a.ds.i1d,b.ds.i1d)
    xr.testing.assert_identical(a.ds.i1d_variance,b.ds.i1d_variance)
    assert np.all(a.ds.i1d_variance>0)
    assert 'i2d_normalization' in b.ds
    observed=b.ds.i1d.copy(deep=True); raw=b.ds.i2d.copy(deep=True)
    b.get_baseline(plot=False)
    xr.testing.assert_identical(b.ds.i1d,observed)
    xr.testing.assert_identical(b.ds.i2d,raw)


@pytest.mark.parametrize("integrate2d", [False, True])
@pytest.mark.parametrize("polarization_factor", [None, 0.95])
def test_detector_data_direct_netcdf_roundtrip(tmp_path, integrate2d, polarization_factor):
    """TIFF/PONI/mask loading must support ordinary xarray serialization."""
    from fabio.tifimage import TifImage
    from fabio.edfimage import EdfImage
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.detectors import Detector

    ai = AzimuthalIntegrator(
        dist=0.1, poni1=0.003, poni2=0.004,
        detector=Detector(pixel1=1e-4, pixel2=1e-4), wavelength=1e-10,
    )
    y, x = np.indices((100, 100))
    image_path, mask_path, poni_path = (tmp_path / name for name in
                                       ("image.tiff", "mask.edf", "geometry.poni"))
    TifImage(data=(100 + 2*x + y).astype(np.float32)).write(str(image_path))
    EdfImage(data=((x > 60) & (y < 50)).astype(np.int8)).write(str(mask_path))
    ai.save(str(poni_path))
    s = exrd()
    kwargs = {} if polarization_factor is None else {"polarization_factor": polarization_factor}
    s.load_xrd_data(
        from_tiff_file=image_path, poni_file=str(poni_path), mask_file=mask_path,
        integrate2d=integrate2d, radial_range=(0.05, 0.35), delta_q=0.01,
        npt_azimuthal=20, plot=False, **kwargs,
    )
    before = s.ds.copy(deep=True)
    path = tmp_path / "data.nc"
    s.ds.to_netcdf(path)
    restored = xr.load_dataset(path)
    xr.testing.assert_identical(restored, before)
    xr.testing.assert_identical(s.ds, before)
    provenance = json.loads(restored.attrs["integration_provenance"])
    assert provenance["poni_file"]["sha256"] == hashlib.sha256(poni_path.read_bytes()).hexdigest()
    assert provenance["mask_file"]["sha256"] == hashlib.sha256(mask_path.read_bytes()).hexdigest()
    assert provenance["geometry"]["wavelength"] == 1e-10
    assert provenance["polarization_factor"] == polarization_factor
    expected = "None" if polarization_factor is None else polarization_factor
    for name in (["i1d", "i2d"] if integrate2d else ["i1d"]):
        assert restored[name].attrs["polarization_factor"] == expected


def real_sample(tmp_path):
    pytest.importorskip('GSASII.GSASIIscriptable')
    from easyxrd import easyxrd_defaults
    easyxrd_defaults['easyxrd_scratch_path']=str(tmp_path)
    s=exrd();q=np.linspace(1,6,300)
    y=100+1000*np.exp(-((q-3.09)/.08)**2)
    s.load_xrd_data(from_i1d_array=np.column_stack([q,y,np.sqrt(y)]),i1d_array_wavelength_in_angstrom=1.5406,plot=False)
    s.ds['i1d_baseline']=xr.full_like(s.ds.i1d,101.)
    structure=Structure.from_spacegroup('Fm-3m',Lattice.cubic(3.52),['Ni'],[[0,0,0]])
    cif=tmp_path/'Ni.cif';structure.to(filename=cif,fmt='cif')
    s.load_phases(from_phases_dict=[{'cif':str(cif),'label':'Ni'}],plot=False)
    return s


def test_real_gsas_preserves_signed_data_weights_and_resets_normalization(tmp_path):
    s=real_sample(tmp_path)
    y=s.ds.i1d.values-s.ds.i1d_baseline.values
    var=s.ds.i1d_variance.values.copy()
    s.setup_gsas2_refiner(do_1st_refinement=False,normalize=True,plot=False)
    factor=s.ds.i1d.attrs['normalization_multiplier']
    np.testing.assert_allclose(s.gpx.histograms()[0].getdata('yweight'),1/(var*factor**2))
    s.setup_gsas2_refiner(do_1st_refinement=False,normalize=False,plot=False)
    h=s.gpx.histograms()[0]
    np.testing.assert_allclose(h.getdata('yobs'),y)
    np.testing.assert_allclose(h.getdata('yweight'),1/var)
    assert np.any(h.getdata('yobs')<0) and np.all(h.getdata('yweight')>0)
    assert 'normalized_to' not in s.ds.i1d.attrs
    s.gpx_saver()
    module=load_gsas_module('GSASIIscriptable')
    reread=module.G2Project(gpxfile=str(Path(s.gsasii_run_directory)/'gsas.gpx'))
    np.testing.assert_allclose(reread.histograms()[0].getdata('yweight'),1/var)


def test_real_gsas_result_extraction_and_lebail_fraction_guard(tmp_path):
    s=real_sample(tmp_path)
    s.setup_gsas2_refiner(plot=False)
    assert s.ds.attrs['refinement_mode']=='LeBail'
    assert np.isnan(s.ds.attrs['PhaseInd_0_wt_fraction'])
    assert np.all(np.isfinite(s.ds.i1d_refined))
    with pytest.raises(ValueError,match='LeBail=False'):
        s.refine_phase_fractions()
    s.set_LeBail(False,plot=False)
    assert s.ds.attrs['refinement_mode']=='Rietveld'
    assert s.ds.attrs['PhaseInd_0_wt_fraction']==pytest.approx(100.)
    assert s.ds.refinement_label.values[-1]=='set_LeBail:False'


def test_real_project_replacement_with_custom_histogram_name(tmp_path):
    s=real_sample(tmp_path)
    s.setup_gsas2_refiner(plot=False)
    s.set_LeBail(False,plot=False)
    module=load_gsas_module('GSASIIscriptable')
    candidate=module.G2Project(gpxfile=str(Path(s.gsasii_run_directory)/'gsas.gpx'))
    old=candidate.histograms()[0].name; new='PWDR renamed sample'
    candidate.data[new]=candidate.data.pop(old)
    for names in candidate.names:
        if names[0]==old: names[0]=new
    for ph in candidate.phases():
        ph.data['Histograms'][new]=ph.data['Histograms'].pop(old)
    candidate.save(str(tmp_path/'renamed.gpx'))
    steps=s.ds.sizes['refinement_step']
    s.replace_gpx_with(tmp_path/'renamed.gpx',min_gof_improvement=None)
    assert s._histogram_name==new
    assert s.ds.sizes['refinement_step']==steps+1
    assert set(s.phases)=={'Ni'}
    assert np.all(np.isfinite(s.ds.i1d_refined))
    np.testing.assert_allclose(s.ds.i1d_refined-s.ds.i1d_baseline,s.gpx.histograms()[0].getdata('ycalc'))
    s.refine_background(min_gof_improvement=None)
    assert s.last_refinement_accepted


def test_spot_products_are_separate_and_repeatable():
    s=profile();q=s.ds.radial.values
    cake=np.tile(s.ds.i1d.values,(3,1));cake[0,15]+=100
    s.ds['i2d']=xr.DataArray(cake,dims=['azimuthal_i2d','radial_i2d'],coords={'azimuthal_i2d':[-90,0,90],'radial_i2d':q},attrs={'wavelength_in_meter':1e-10})
    original=s.ds.copy(deep=True)
    with pytest.raises(ValueError,match='get_i2d_baseline'):
        s.get_baseline(spotty_data_correction=True,plot=False)
    with pytest.warns(UserWarning):
        s.get_baseline(get_i2d_baseline=True,spotty_data_correction=True,plot=False)
    first=s.ds.i1d_spot_corrected.copy(deep=True)
    with pytest.warns(UserWarning):
        s.get_baseline(get_i2d_baseline=True,spotty_data_correction=True,plot=False)
    xr.testing.assert_identical(s.ds.i2d,original.i2d)
    xr.testing.assert_identical(s.ds.i1d,original.i1d)
    xr.testing.assert_identical(s.ds.i1d_spot_corrected,first)


def test_phase_reload_cleans_metadata_without_temp_files(tmp_path,monkeypatch):
    crystal=Structure(Lattice.cubic(4),['Ni'],[[0,0,0]])
    path=tmp_path/'input.cif';crystal.to(filename=path,fmt='cif')
    s=profile();monkeypatch.chdir(tmp_path)
    s.load_phases(from_phases_dict=[{'label':'a','cif':str(path)},{'label':'b','cif':str(path)}],plot=False)
    s.ds.attrs['PhaseInd_1_cell_a']=4.
    s.load_phases(from_phases_dict=[{'label':'a','cif':str(path)}],plot=False)
    assert not any(k.startswith('PhaseInd_1_') for k in s.ds.attrs)
    assert list(tmp_path.glob('*.cif'))==[path]
    with pytest.raises(ValueError,match='exactly one'):
        s.load_phases(plot=False)
