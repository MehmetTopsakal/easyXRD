import sys
import os
import subprocess

import importlib
from importlib.metadata import version


class HiddenPrints:
    """
    This class hides print outputs from functions. It is useful for processes like refinement which produce a lot of text prints.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


print("\n\nChecking required packages:\n")
# These are big python libraries that we will need in pySULI.
# If the required library doesn't exist, we install it via pip

required_big_packages = {
    "numpy",
    "scipy",
    "xarray",
    "ipympl",
    "pymatgen",
    "pyFAI",
    "pybaselines",
}

for rp in required_big_packages:
    try:
        globals()[rp] = importlib.import_module(rp)
        print(
            "---%s package with version %s is available and can be imported "
            % (rp, version(rp))
        )
    except:
        print("\n\nInstalling %s" % rp)
        subprocess.check_call([sys.executable, "-m", "pip", "install", rp])
        globals()[rp] = importlib.import_module(rp)

# these are other packages that are usually installed by big packages above.
# Otherwise, we pip-install them

required_other_packages = {"fabio", "pandas", "mp_api"}

for rp in required_other_packages:
    try:
        globals()[rp] = importlib.import_module(rp)
        print(
            "---%s package with version %s is available and can be imported "
            % (rp, version(rp))
        )
    except:
        print("\n\nInstalling %s" % rp)
        subprocess.check_call([sys.executable, "-m", "pip", "install", rp])
        globals()[rp] = importlib.import_module(rp)


# defaults
easyxrd_defaults = dict()
user_home = os.path.expanduser("~")


# Setting up easyxrd_scratch folder
if not os.path.isdir(os.path.join(user_home, ".easyxrd_scratch")):
    os.mkdir(os.path.join(user_home, ".easyxrd_scratch"))
easyxrd_defaults["easyxrd_scratch_path"] = os.path.join(user_home, ".easyxrd_scratch")


# check g2full lib path
if os.name == "nt":
    # assuming you followed these instructions for installing GSASS-II: https://advancedphotonsource.github.io/GSAS-II-tutorials/install.html#gsas2pkg-conda-package (section 1.2)
    try:
        default_gsasii_lib_path = os.path.join(
            user_home,
            "Appdata",
            "Local",
            "miniforge3",
            "envs",
            "GSASII",
            "GSAS-II",
            "GSASII",
        )
        print(default_gsasii_lib_path)
        sys.path += [default_gsasii_lib_path]
        with HiddenPrints():
            import GSASIIscriptable as G2s
        default_gsasii_lib_path = default_gsasii_lib_path
    except:
        print(
            "\nUnable to import GSASS-II libraries. See the link below for GSASS-II installation \nhttps://advancedphotonsource.github.io/GSAS-II-tutorials/install.html "
        )
        default_gsasii_lib_path = "not found"
elif os.name == "posix":
    # assuming you followed these instructions for installing GSASS-II: https://advancedphotonsource.github.io/GSAS-II-tutorials/install-g2f-linux.html
    default_gsasii_install_path = os.path.join(user_home, "g2full/GSAS-II/GSASII")
    sys.path += [default_gsasii_install_path]
    try:
        with HiddenPrints():
            import GSASIIscriptable as G2sc
        default_gsasii_lib_path = default_gsasii_install_path
    except:
        print(
            "\nUnable to import GSASS-II libraries. See the link below for GSASS-II installation \nhttps://advancedphotonsource.github.io/GSAS-II-tutorials/install.html "
        )
        default_gsasii_lib_path = "not found"
easyxrd_defaults["gsasii_lib_path"] = default_gsasii_lib_path


# check Materials Project API key in easyxrd_scratch folder
if os.path.isfile(os.path.join(user_home, ".easyxrd_scratch", "mp_api_key.dat")):
    with open(
        os.path.join(user_home, ".easyxrd_scratch", "mp_api_key.dat"), "r"
    ) as api_key_file:
        api_key_file_content = api_key_file.read().split()[-1]
        if len(api_key_file_content) == 32:
            mp_api_key = api_key_file_content
        else:
            mp_api_key = "not found"
else:
    mp_api_key = "none"
easyxrd_defaults["mp_api_key"] = mp_api_key


def set_defaults(name, val):
    """set a global variable."""
    global easyxrd_defaults
    easyxrd_defaults[name] = val


def print_defaults():
    for key, val in easyxrd_defaults.items():

        if key != "mp_api_key":
            print("%s : %s" % (key, val))
        else:
            print("%s : %s......" % (key, val[:5]))


print("\n\nImported easyxrd with the following configuration:\n")
print_defaults()
