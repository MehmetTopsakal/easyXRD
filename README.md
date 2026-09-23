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



Then we need to create a virtual environment as shown below:



```bash
conda create --name env_py3.14_np_2.4 -c conda-forge -y  python=3.14 numpy=2.4
```
Once the new virtual environment is created, we need to activate and install easyXRD and GSAS-II packages directly from GitHub. 

```bash
conda activate env_py3.14_np_2.4

# for easyXRD
python -m pip install "easyXRD[notebook] @ git+https://github.com/MehmetTopsakal/easyXRD.git"



# for GSAS-II

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


If all is successful, you should be able to import easyXRD inside python:

```python
from easyxrd import exrd
```

After this step, you can contine with `exrd` as we explained in the Google Colab notebooks that are listed above.



## Acknowledgements

Development of `easyXRD` was supported by FY24 Scientific Technique and Expertise Development – Instrument Scientist Support Project managed by Nuclear Science User Facilities (NSUF), https://nsuf.inl.gov/. If you find `easyXRD` useful for your research, please acknowldege using the sentence below:

"This work utilized `easyXRD` an open-source tool used for processing and analyzing X-ray diffrection data and is supported by the DOE, Office of Nuclear Energy, under DOE Idaho Operations Office Contract DE-AC07-05ID14517, as part of a Nuclear Science User Facilities project."

In addition, you need to acknowledge GSAS-II if you use the refinement components of `easyXRD`. You can check the original GSAS-II repo, https://github.com/AdvancedPhotonSource/GSAS-II, for further details.


Feel free to contact me (metokal@gmail[-remove-this].com) if you have any questions about `easyXRD`
