# easyXRD


We have developed a versatile X-ray diffraction (XRD) analysis tool that utilizes modern and open-source Python packages such as pyFAI, xarray, pymatgen, pybaselines,... for data processing/storage and interfaced to Jupyter notebooks powered with actively developed visualization packages such as ipywidgets, and matplotlib. It provides easy access to the Materials Project database which hosts thousands of crystal structures that can be used for phase identification - a critical part of XRD analysis - and utilizes [GSAS-II suite](https://github.com/AdvancedPhotonSource/GSAS-II) for XRD refinements in a user-friendly and intuitive manner. Ultimate goal of this tool is to make X-ray diffraction analysis easy for users and help them to process, refine, store, and share their XRD data conveniently.

You can try the tutorial notebooks below on Google Colab:



<a target="_blank"  href="https://colab.research.google.com/github/MehmetTopsakal/easyXRD_examples/blob/main/01_basic.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> 01_basic.ipynb</a> <br>
<a target="_blank"  href="https://colab.research.google.com/github/MehmetTopsakal/easyXRD_examples/blob/main/02_intermediate.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> 02_intermediate.ipynb</a> <br>
<a target="_blank"  href="https://colab.research.google.com/github/MehmetTopsakal/easyXRD_examples/blob/main/03_advanced-part-1.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> 03_advanced-part-1.ipynb</a> <br>
<a target="_blank"  href="https://colab.research.google.com/github/MehmetTopsakal/easyXRD_examples/blob/main/03_advanced-part-2.ipynb">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" /> 03_advanced-part-2.ipynb</a> <br>


.... more to come






In order to use `easyXRD` Python package, you need to have a Python environment that can be easily installed through Conda.

You can follow the instructions on this link: https://www.anaconda.com/docs/getting-started/miniconda/install



Examples of environment setup is shown below:

* Linux Bash:

```bash
mkdir -p ~/.miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/.miniconda3/miniconda.sh
bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
rm ~/.miniconda3/miniconda.sh

source ~/.miniconda3/bin/activate
conda init --all
```

* Windows PowerShell (After installing, open the “Anaconda Powershell Prompt (miniconda3)”):

```bash
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\.miniconda.exe"
Start-Process -FilePath ".\.miniconda.exe" -ArgumentList "/S" -Wait
del .\.miniconda.exe
```


* macOS Bash (Apple Silicon):

```bash
mkdir -p ~/.miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/.miniconda3/miniconda.sh
bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
rm ~/.miniconda3/miniconda.sh

source ~/.miniconda3/bin/activate
conda init --all
```


* macOS Bash (Intel):

```bash
mkdir -p ~/.miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/.miniconda3/miniconda.sh
bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
rm ~/.miniconda3/miniconda.sh

source ~/.miniconda3/bin/activate
conda init --all
```



Then we need to create a virtual environment as shown below:



```bash
conda create --name env_py3.14_np_2.4 -c conda-forge -y  python=3.14 numpy=2.4
```
Once the new virtual environment is created, we need to activate and install easyXRD and GSAS-II packages directly from GitHub. 

```bash
conda activate env_py3.14_np_2.4

# for easyXRD
python -m pip install "easyXRD[notebook] @ git+https://github.com/MehmetTopsakal/easyXRD.git"



# for GSAS-II (or use the Python helper below)

# These worked nicely on my Ubuntu 26.04.1 LTS
python -m pip install meson-python ninja wheel Cython pyproject-metadata tomli
NUMPY_PC_DIR="$(numpy-config --pkgconfigdir)"
export PKG_CONFIG_PATH="$NUMPY_PC_DIR${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
python -m pip install --no-build-isolation "GSAS-II[useful] @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git"

# It was a bit painful to install GSAS-II inside conda environment. These worked for me on my virtual Windows 11.
python -m pip install meson-python ninja wheel Cython pyproject-metadata tomli
conda install -c conda-forge fortran-compiler c-compiler ninja meson-python cython numpy scipy wxpython lld llvm-tools m2w64-toolchain
python -m pip install --no-build-isolation "GSAS-II[useful] @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git"
# if it fails in the previous line by complaining "Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/", do that and repeat previous line.

# Unfortunately I don't have and Arm Mac. 
# Try `python -m pip install --no-build-isolation "GSAS-II[useful] @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git" `
# and ask AI on what to do if it fails :)

# 
```


Once easyXRD is installed, you can install GSAS-II from Python in the same
environment (also works in a notebook):

```python
import easyxrd

gsas_path = easyxrd.install_gsas2()
# Later, to update:
# easyxrd.install_gsas2(upgrade=True)
```

This installs build dependencies and the official GSAS-II GitHub package using
the running Python interpreter, verifies its scripting import in a fresh process,
and updates `easyxrd.easyxrd_defaults["gsasii_lib_path"]`. Restart Python or your
notebook kernel afterward. Git and C/Fortran compilers must already be installed;
the helper does not install operating-system packages. Use a dedicated conda or
virtual environment with internet access.

GUI dependencies (including wxPython) are included by default because easyXRD's
refinement setup requires wxPython. If you manage wxPython separately or only
need GSAS-II scripting, use `easyxrd.install_gsas2(gui=False)`. Building wxPython
on Linux can take considerable time and require system development libraries.
See the [official pip installation instructions](https://advancedphotonsource.github.io/GSAS-II-tutorials/install-pip.html)
for platform prerequisites; the manual commands above are also available.

If all is successful, you should be able to import easyXRD inside python:

```python
from easyxrd import exrd
```

To display package versions, the Python executable, the GSAS-II module location,
and current configuration (including a masked Materials Project API key), run:

```python
import easyxrd

easyxrd.show_config()
```

After this step, you can contine with `exrd` as we explained in the Google Colab notebooks that are listed above.



## X-axis units

The default plot x-axis is scattering vector `q` in inverse Angstrom. Select
2θ in degrees when creating an analysis object:

```python
from easyxrd import exrd

sample = exrd(radial_unit="two-theta")
sample.load_xrd_data(...)  # Automatic plots and sample.plot() use 2θ.
```

This applies to 1D profiles, 2D cakes, baselines, phase reflections, and refinement
plots. Conversion uses each profile's `wavelength_in_angst` metadata. Analysis
coordinates in `sample.ds` remain in `q`; input units are still specified with
`i1d_array_radial_unit` or `txt_file_radial_unit`. The plotting argument
`i1d_plot_radial_range` uses the selected display unit. Export units remain
controlled by `export_i1d_to(mode=...)` (`"xy"` for 2θ, `"qxy"` for q).

For detector integration and text-file loading, `radial_range` also uses the
object's selected unit, including bounds from `easyxrd_defaults`. For example:

```python
sample = exrd(radial_unit="two-theta")
sample.load_xrd_data(from_tiff_file="image.tif", poni_file="geometry.poni",
                     radial_range=(10, 60))  # 10° to 60° in 2θ
```

Degree bounds must be between 0 and 180 and are converted using the source's
wavelength. `radial_range=None` disables the bounds. `delta_q` continues to
specify bin spacing in inverse Angstrom; for degree bounds the upper limit is
kept exact and the q spacing is at most `delta_q`. Set `delta_q=None` to select
the number of bins with `radial_npts` instead.

Use `exrd(radial_unit="q")` for q plots, or set
`easyxrd.easyxrd_defaults["radial_unit"] = "two-theta"` for new objects.


## Rwp history in the dataset

`sample.setup_gsas2_refiner(...)` starts a new refinement history in
`sample.ds`. Each optimizer attempt, including setup's initial passes and calls
through helpers such as `refine_background()`, appends one `refinement_step`:

```python
sample.ds.rwp_history                    # retained Rwp (%) after each attempt
sample.ds.rwp_attempted_history          # optimizer Rwp before acceptance/rollback
sample.ds.refinement_status              # "accepted", "rejected", or "error"
sample.ds.refinement_label               # helper name or a custom log_label

sample.refine(log_label="Final joint refinement")
sample.ds.rwp_history.plot(marker="o")
sample.export_ds_to("refined.nc")         # includes the history
```

The history updates even when `refine(update_ds=False, ...)` disables profile
updates. Rejected attempts retain the previous fit's Rwp in `rwp_history` and
keep the trial value in `rwp_attempted_history`; `refinement_reason` records why
the result was rejected. Optimizer errors are logged with NaN Rwp values and
then re-raised. Each entry has a UTC `refinement_timestamp`. History survives
NetCDF export/reload; setting up a new GSAS-II engine resets it. With
`do_1st_refinement=False`, the history starts empty.

## Trying initial lattice scales

If cell refinement gets stuck because the starting lattice is too far from the
measured peaks, try several initial scales:

```python
sample.refine_cell_parameters(
    scale_trials=np.linspace(0.98, 1.02, 9),
    phase_ind="all",
    plot=True,
)
sample.cell_scale_best    # winning starting factor, or None if none improved Rwp
sample.cell_scale_trials  # scale, attempted Rwp, status and reason for each trial
```

Each factor multiplies `a`, `b`, and `c` relative to the cells at the start of the
call, preserving angles. Use `phase_ind=0` to scale one phase; `"all"` scales all
phases by the same factor. Trials start from independent copies of the same
project. The method retains the lowest finite Rwp that improves the original
fit and passes the existing Uiso and `min_gof_improvement` checks. Failed trials
are logged and skipped. If none improves the fit, the original fit is restored.
Only the retained fit is plotted. The existing Rwp history includes every trial
and its scale in `refinement_label`. Omitting `scale_trials` keeps the ordinary
single-refinement behavior.

## Automatic date versions

Enable the repository's version hook once after cloning:

```bash
git config --local core.hooksPath .githooks
```

Before each normal `git commit`, including `git commit --amend`, the hook sets
`[project].version` in `pyproject.toml` to the committer date as `YYYY.MM.DD`
(for example, `2026.10.19`) and includes that version in the commit. It uses Git's
committer timestamp and timezone, including an explicit `GIT_COMMITTER_DATE`.
Commits on the same calendar day share the same version.

The hook updates only the version in the staged manifest and preserves unrelated
unstaged edits. It requires Python 3 and must be enabled separately in each clone.


## Acknowledgements

Development of `easyXRD` was supported by FY24 Scientific Technique and Expertise Development – Instrument Scientist Support Project managed by Nuclear Science User Facilities (NSUF), https://nsuf.inl.gov/. If you find `easyXRD` useful for your research, please acknowldege using the sentence below:

"This work utilized `easyXRD` an open-source tool used for processing and analyzing X-ray diffrection data and is supported by the DOE, Office of Nuclear Energy, under DOE Idaho Operations Office Contract DE-AC07-05ID14517, as part of a Nuclear Science User Facilities project."

In addition, you need to acknowledge GSAS-II if you use the refinement components of `easyXRD`. You can check the original GSAS-II repo, https://github.com/AdvancedPhotonSource/GSAS-II, for further details.


Feel free to contact me (metokal@gmail[-remove-this].com) if you have any questions about `easyXRD`

## Quantitative data and review corrections

`load_xrd_data` accepts exactly one input source and validates it before replacing
an existing dataset. Radial coordinates must increase strictly. Array input may
contain `(q_or_2theta, intensity, sigma)`; the third column is a **standard
deviation**, stored as `ds.i1d_variance = sigma**2`. Alternatively pass
`i1d_variance=`. For text data, select the uncertainty column explicitly with
`txt_file_usecols=(0, 1, 2)`.

For detector images, pass a calibrated detector `variance=` array or a suitable
pyFAI `error_model=`. Use `error_model="poisson"` only when that model is justified
by the detector data. Integration accepts `polarization_factor`, `dark`, `flat`,
and `normalization_factor`. The quantitative 1D profile always comes from
`integrate1d`; `integrate2d=True` additionally stores a cake and its normalization
accumulators. With `delta_q=None`, `radial_npts` controls the bin count.

Refinement uses explicit inverse-variance weights and preserves signed
background-subtracted observations. Scaling intensities also scales their
variance. If no variance is supplied, easyXRD warns and uses unit weights; GoF
and parameter uncertainties then have no counting-statistics interpretation.
Independent measured-background variance is propagated conditional on the chosen
scale. Uncertainty in an estimated background, its scale, and covariance between
pixel-split bins are not estimated.

Background grids must match explicitly. To use a different grid, first resample
it deliberately and account for the resulting uncertainty/correlation. Invalid
coordinates or input files raise exceptions without discarding the current data.
Changing data, phases, or a baseline invalidates an existing refinement engine;
call `setup_gsas2_refiner` again. Reloading an identical NetCDF dataset preserves
an attached engine and refinement history.

Median filtering now creates `i2d_filtered` or `image_filtered` diagnostic data.
Spot selection creates `i2d_spot_mask` and `i1d_spot_corrected`, without changing
raw intensities or subtracting a selection threshold. Neither diagnostic replaces
the quantitative observations used for refinement.

Phase weight fractions are unavailable (`NaN`, `weight_fractions_valid=False`)
during Le Bail extraction. Switch all phases with `set_LeBail(False)` before
quantitative phase-fraction refinement. All reported fractions are normalized to
the modeled crystalline phases; they do not establish an amorphous fraction.

Use `sample.plot(close=True, export_fig_as="pattern.png")` in batch processing.
`install_gsas2()` keeps GUI dependencies optional; use `gui=True` for the GUI.
`fine_tune_gpx` launches the installed package using the current interpreter.
Replacement GPX files must contain one powder histogram with compatible
coordinates and observations; its histogram name may differ.

The combined review, implementation decisions, and remaining scientific
validation work are recorded in [code_review.md](code_review.md).

### Parameter uncertainties, covariance, and provenance

Accepted refinements with `update_ds=True` store the latest parameter results
directly in `sample.ds`:

| Variable | Meaning |
|---|---|
| `gsas_parameter_value` | Directly varied GSAS-II parameters, indexed by `gsas_parameter` |
| `gsas_parameter_su` | Reported standard uncertainties in the same order |
| `gsas_covariance` | Native covariance, with separate named row/column dimensions |
| `gsas_correlation` | Correlation computed from covariance; undefined entries remain `NaN` |
| `gsas_fit_statistic` | Named Rwp, GoF, observation/variable counts, chi-square, singularities, and maximum shift/s.u. |
| `gsas_converged` | `1` converged, `0` not converged, `-1` unavailable |

The parameter names and native units follow the
[GSAS-II covariance schema](https://gsas-ii-scripting.readthedocs.io/en/latest/objvarorg.html#covariance-tree-item).
They may represent reciprocal-cell or constrained variables; these are not
automatically transformed into uncertainties on lattice constants or phase
fractions. The stored covariance is copied without additional GoF scaling and
does not include uncertainty from preprocessing. Missing covariance is recorded
as unavailable, never replaced by a fabricated diagonal matrix. These diagnostics
do not by themselves certify a physically valid or identifiable model.

```python
sample.ds.gsas_parameter_su
sample.ds.gsas_fit_statistic.sel(gsas_statistic="Max shft/sig")
sample.export_ds_to("refined.nc")
```

Changing the varied parameter list replaces the previous result dimensions.
Rejected fits retain the accepted parameter results; `gsas_result_step` identifies
their attempt number. Changing phases/baseline or setting up a new engine clears
these results. With `update_ds=False`, results remain those of the last dataset
update. The attempt history remains a separate record.

`integration_provenance` and `refinement_provenance` dataset attributes are JSON
strings. Read them with `json.loads(sample.ds.attrs["integration_provenance"])`.
They record software versions, actual detector geometry, integration settings,
instrument parameters, normalization, and weighting. PONI, file-based mask, and
instrument-source files include SHA-256 identifiers. Supplied in-memory masks,
dark/flat/variance arrays and raw images are not archived or hashed; retain those
inputs separately. Calibration records describe the supplied geometry, not its
experimental accuracy or calibration uncertainty.

### Configurable defaults

`easyxrd_defaults` in `easyxrd/__init__.py` holds plotting, data loading,
baseline, refinement, and export-format defaults. List them with
`easyxrd.print_defaults()` and change session settings with `set_defaults()`:

```python
import easyxrd

easyxrd.set_defaults("plot_figsize", (10, 7))
easyxrd.set_defaults("plot_dpi", 160)
easyxrd.set_defaults("load_xrd_data_plot", False)
easyxrd.set_defaults("get_baseline_iarpls_lam", 1e6)
easyxrd.set_defaults("refine_background_num_coeffs", 12)

sample = easyxrd.exrd()
# After loading data:
sample.plot(figsize=(12, 8), dpi=200)  # explicit values override defaults
```

Shared plotting keys start with `plot_`; other keys generally use
`<method>_<parameter>`. The refinement-history wrapper and standalone
`refinement_plotter` retain their original separate DPI defaults (128 and 150).
Omitted arguments read the current dictionary at call time, including after
modules have already been imported. Signatures display `<easyxrd default>` for
these arguments; use `print_defaults()` for their current values. Explicit
arguments, including `False` and `None`, keep their existing meanings.

An `exrd` object copies `verbose`, `figsize`, `i2d_robust`, `i2d_logscale`, and
`i1d_ylogscale` at construction. Changing package defaults later affects new
objects; change an existing object's attributes or pass a plot argument to
override its settings. Mutable configuration values are copied for each call.
Input datasets, arrays, file destinations, and internal helper defaults remain
local to their calls. Settings are session-only; put `set_defaults()` calls in
your script or notebook to reuse them.

### Logging and development checks

Automatic easyXRD progress and rejection messages use Python logging. Enable
notebook progress explicitly:

```python
from easyxrd import configure_logging
configure_logging("INFO")       # progress and rejection messages
configure_logging("WARNING")    # rejection/warning messages only
```

Applications may instead configure the `easyxrd` logger themselves. The helper
does not change root handlers and repeated calls do not duplicate its handler.
`verbose=True` still controls GSAS-II's own stdout. Explicit `show_config()` and
`print_defaults()` reports continue to print.

Run `python -m pytest -q` for the suite. Tests requiring GSAS-II skip when it is
absent. The CI workflow includes Linux, macOS and Windows core tests and a Linux
job that installs GSAS-II and runs the real integration tests. Adding the workflow
does not establish that those remote jobs or interactive GUI checks have passed.

