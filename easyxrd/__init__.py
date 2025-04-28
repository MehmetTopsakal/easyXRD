import sys
import os
import subprocess

import importlib
from importlib.metadata import version

import shutil
import numpy as np
import urllib.request

import tarfile
import time


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


# Get GSAS-II and binaries from GitHub

gsas2_path_in_easyxrd_scratch = os.path.join(user_home, ".easyxrd_scratch", "GSAS-II")
sys.path += [os.path.join(gsas2_path_in_easyxrd_scratch, "GSASII")]

try:
    with HiddenPrints():
        import GSASIIscriptable as G2sc
    print(
        "\nFound usable GSAS-II lib path @ %s"
        % os.path.join(gsas2_path_in_easyxrd_scratch, "GSASII")
    )
    easyxrd_defaults["gsasii_lib_path"] = os.path.join(
        gsas2_path_in_easyxrd_scratch, "GSASII"
    )
except Exception as exc:
    # print(exc)
    if os.path.isdir(gsas2_path_in_easyxrd_scratch):
        shutil.rmtree(gsas2_path_in_easyxrd_scratch)

    # print("\nClonning GSAS-II package from GitHub")
    # import git
    # git.Repo.clone_from(
    #     "https://github.com/AdvancedPhotonSource/GSAS-II",
    #     to_path=gsas2_path_in_easyxrd_scratch,
    #     multi_options=["--depth 1"],
    # )

    print("\nDownloading GSAS-II (version 5805) from GitHub")
    urllib.request.urlretrieve(
        "https://github.com/AdvancedPhotonSource/GSAS-II/archive/refs/tags/5805.tar.gz",
        os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-5805.tar.gz"),
    ),

    with tarfile.open(
        os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-5805.tar.gz"),
        "r:gz",
    ) as tar:
        tar.extractall(os.path.join(easyxrd_defaults["easyxrd_scratch_path"]))
    os.rename(
        os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II-5805"),
        os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II"),
    )

    os.makedirs(
        os.path.join(gsas2_path_in_easyxrd_scratch, "GSASII-bin"), exist_ok=True
    )

    if os.name == "posix":
        os_str = "linux_64"
    elif os.name == "nt":
        os_str = "win_64"
    py_str = "p%d.%d" % (sys.version_info.major, sys.version_info.minor)
    np_str = "n%s.%s" % (
        np.version.version.split(".")[0],
        np.version.version.split(".")[1],
    )
    gsas2_bin_tgz = (
        "https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/%s_%s_%s.tgz"
        % (os_str, py_str, np_str)
    )
    print(gsas2_bin_tgz)

    os.makedirs(
        os.path.join(
            gsas2_path_in_easyxrd_scratch,
            "GSASII-bin",
            gsas2_bin_tgz.split("/")[-1][:-4],
        ),
        exist_ok=True,
    )

    try:
        print("Downloading %s from GitHub" % gsas2_bin_tgz)
        urllib.request.urlretrieve(
            gsas2_bin_tgz,
            os.path.join(
                gsas2_path_in_easyxrd_scratch,
                "GSASII-bin",
                gsas2_bin_tgz.split("/")[-1][:-4],
                "bin.tgz",
            ),
        )
        with tarfile.open(
            os.path.join(
                gsas2_path_in_easyxrd_scratch,
                "GSASII-bin",
                gsas2_bin_tgz.split("/")[-1][:-4],
                "bin.tgz",
            ),
            "r:gz",
        ) as tar:
            tar.extractall(
                path=os.path.join(
                    gsas2_path_in_easyxrd_scratch,
                    "GSASII-bin",
                    gsas2_bin_tgz.split("/")[-1][:-4],
                )
            )
        easyxrd_defaults["gsasii_lib_path"] = os.path.join(
            gsas2_path_in_easyxrd_scratch, "GSASII"
        )

        print(
            "\n!!!! Please re-run this cell (after kernel restart) for the GSAS-II installation to take effect !!!!!"
        )
        time.sleep(2)
        os._exit(1)
    except Exception as exc:
        print(exc)
        easyxrd_defaults["gsasii_lib_path"] = "not found"


# check Materials Project API key in easyxrd_scratch folder
if os.path.isfile(os.path.join(user_home, ".easyxrd_scratch", "mp_api_key.dat")):
    with open(
        os.path.join(user_home, ".easyxrd_scratch", "mp_api_key.dat"), "r"
    ) as api_key_file:
        api_key_file_content = api_key_file.read().split()[-1]
        if len(api_key_file_content) == 32:
            mp_api_key = api_key_file_content
        else:
            mp_api_key = "invalid"
else:
    mp_api_key = "not found"
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
            print("%s : %s.........." % (key, val[:9]))


print("\n\nImported easyxrd with the following configuration:\n")
print_defaults()
