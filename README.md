# easyXRD

[![image](https://img.shields.io/pypi/v/easyxrd.svg)](https://pypi.python.org/pypi/easyxrd)
[![image](https://img.shields.io/pypi/l/easyxrd.svg)](https://pypi.python.org/pypi/easyxrd)
[![image](https://img.shields.io/pypi/pyversions/easyxrd.svg)](https://pypi.python.org/pypi/easyxrd)



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






## Installation from this source checkout

Use a dedicated environment with Python 3.11 or newer. Compatibility of the
complete scientific stack depends on available dependency and GSAS-II binaries;
the declared Python minimum is not a tested compatibility matrix.

From the extracted project directory:

```bash
python -m pip install ".[notebook,materials]"
```

For development:

```bash
python -m pip install -e ".[notebook,materials,dev]"
python -m unittest discover -s tests -v
```

Inside Jupyter, install into the running kernel's environment (replace the path):

```python
import subprocess
import sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "/absolute/path/to/easyXRD-main[notebook,materials]",
])
```

Restart the kernel after installation. The `notebook` extra provides Jupyter and
interactive plotting; `materials` provides the Materials Project client. GSAS-II
is installed separately following its [official installation instructions](https://advancedphotonsource.github.io/GSAS-II-tutorials/install-pip.html).
It is required for refinement and GPX operations, not basic data processing.

```python
from easyxrd.core import exrd
analysis = exrd()
```

For a legacy GSAS-II installation, supply the directory containing
`GSASIIscriptable.py`:

```python
# After loading data and phases:
analysis.setup_gsas2_refiner(gsasii_lib_path="/path/to/GSASII")
```

Package import is quiet and does not create directories, install packages, or
import GSAS-II. Scratch directories are created when an operation needs them.
The existing configuration API remains available:

```python
from easyxrd import print_defaults, set_defaults
print_defaults()  # API keys are redacted
set_defaults("mp_api_key", "your-api-key")
```

Existing `~/.easyxrd_scratch/mp_api_key.dat` files remain supported. Missing,
empty, or unreadable key files no longer prevent importing the package.

## Troubleshooting

```bash
python -m easyxrd.diagnostics
```

This reports the active Python executable, installed distribution versions, and
NumPy's runtime version, module location, and header directory. It does not
verify the ABI compatibility of GSAS-II or other compiled extensions. Use the
same interpreter when installing or rebuilding packages. No API keys are included.

See `CHANGES.md` for the scope and validation limits of this revision.

It should be noted that, you need to acknowledge GSAS-II if you use the refinement components of `easyXRD`. You can check original GSAS-II repo, https://github.com/AdvancedPhotonSource/GSAS-II, for further details.


Feel free to contact me (metokal@gmail[-remove-this].com) if you have any questions about `easyXRD`



