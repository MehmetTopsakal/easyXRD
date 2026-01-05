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
rm ~/miniconda3/miniconda.sh

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



Once you have a working conda environment through these steps above, now we need to create a virtual environment as shown below:



```bash
conda create --name env_py3.13_np_2.2 -c conda-forge -y  python=3.13 numpy=2.2 jupyterlab
```
Once the new virtual environment is created, ne we need to activate it and then call jupyter lab interface

```bash
conda activate env_py3.13_np_2.2
cd
jupyter lab
```

Once you have the jupyter lab interface, we can continue from there. A browser window with a Jupyter session should open after you execute `jupyter lab` prompt. 

In a new cell, we need to install additional python packages for `easyXRD`.\
You need to Copy-Paste the contents of code below into the Jupyter lab cell and execute it (Ctrl+Enter).

```python
# Here we install the necessary packages via pip
# It may take a while for this cell to complete. This will be quicker in next runs...
required_packages = {
    "mp-api",
    "scipy",
    "xarray",
    "h5netcdf",
    "ipympl",
    "pymatgen",
    "pyFAI",
    "fabio",
    "pybaselines",
    "ipympl",
    "easyxrd"
}
import subprocess,sys
for p in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", p])
from IPython.display import clear_output
clear_output()


# Finally, we nedd to reset kernel for the package installations to take effect
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

If you successfully executed these in a Jupyter notebook cell, this means you have a working Python environment. As next, we need to install GSAS-II into our python environment based on the instructions shown here: https://advancedphotonsource.github.io/GSAS-II-tutorials/install-pip.html. Please note, we are using `pip` based installation.

```python
try:
    import GSASII.GSASIIscriptable as G2sc
except:
    !! git clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II.git ./G2
    !! pip install ./G2[useful]
```

We can now import easyXRD.
If you are running this cell for the first time, it will need to download GSAS-II libraries and binaries from GitHub. For the settings to take effect, the jupyter kernel needs to be restarted.

```python
from easyxrd.core import exrd
```

After this step, you can contine with `exrd` as we explained in the Google Colab notebooks that are listed above.


It should be noted that, you need to acknowledge GSAS-II if you use the refinement components of `easyXRD`. You can check original GSAS-II repo, https://github.com/AdvancedPhotonSource/GSAS-II, for further details.


Feel free to contact me (metokal@gmail[-remove-this].com) if you have any questions about `easyXRD`



