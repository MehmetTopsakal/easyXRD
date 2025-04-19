import sys
import os
import subprocess


import importlib
from importlib.metadata import version

import shutil


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

    gsasii_lib_path = "not found"
    possible_gsas_lib_locations = [
        os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II", "GSASII"),
    ]
    for p in possible_gsas_lib_locations:
        try:
            sys.path += [p]
            with HiddenPrints():
                import GSASIIscriptable as G2sc
            print("\n\nfound useable GSAS-II lib in %s" % p)
            gsasii_lib_path = p
            break
        except:
            sys.path.remove(p)

    if gsasii_lib_path == "not found":
        print("\nTrying to get GSAS-II lib and binaries from GitHub")
        here = os.getcwd()
        os.chdir(os.path.join(easyxrd_defaults["easyxrd_scratch_path"]))
        if os.path.isdir(
            os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II")
        ):
            shutil.rmtree(
                os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II")
            )

        import numpy, sys

        if (sys.version_info.minor == 12) and (
            numpy.version.version[:3] == "2.2"
        ):
            import git
            os.makedirs(os.path.join(easyxrd_defaults["easyxrd_scratch_path"],'GSAS-II'),exist_ok=True)

            git.Repo.clone_from("https://github.com/AdvancedPhotonSource/GSAS-II",
                                os.path.join(easyxrd_defaults["easyxrd_scratch_path"],
                                             'GSAS-II'
                                             ))
            os.makedirs(os.path.join(easyxrd_defaults["easyxrd_scratch_path"],'GSAS-II','GSASII-bin'),exist_ok=True)
            os.makedirs(os.path.join(easyxrd_defaults["easyxrd_scratch_path"],'GSAS-II','GSASII-bin','win_64_p3.12_n2.2'),exist_ok=True)

            import urllib.request
            urllib.request.urlretrieve("https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/win_64_p3.12_n2.2.tgz",
                                        os.path.join(easyxrd_defaults["easyxrd_scratch_path"],'GSAS-II','GSASII-bin','win_64_p3.12_n2.2','win_64_p3.12_n2.2.tgz'))


            import tarfile
            with tarfile.open(os.path.join(easyxrd_defaults["easyxrd_scratch_path"],'GSAS-II','GSASII-bin','win_64_p3.12_n2.2','win_64_p3.12_n2.2.tgz'), 'r:gz') as tar:
                tar.extractall(path=os.path.join(easyxrd_defaults["easyxrd_scratch_path"],'GSAS-II','GSASII-bin','win_64_p3.12_n2.2'))



            print('\n!!!! Please re-run this cell (after kernel restart) for the GSAS-II installation to take effect !!!!!')

            import time
            time.sleep(5)
            os._exit(1)


        os.chdir(here)


    gsasii_lib_path = os.path.join(user_home, ".easyxrd_scratch", 'GSAS-II', 'GSASII')

elif os.name == "posix":

    gsasii_lib_path = "not found"
    possible_gsas_lib_locations = [
        os.path.join(user_home, "g2full", "GSAS-II", "GSASII"),
        os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II", "GSASII"),
    ]
    for p in possible_gsas_lib_locations:
        try:
            sys.path += [p]
            with HiddenPrints():
                import GSASIIscriptable as G2sc
                print("\n\nfound useable GSAS-II lib in %s" % p)
                gsasii_lib_path = p
                break
        except:
            sys.path.remove(p)
    if gsasii_lib_path == "not found":
        print("\nTrying to get GSAS-II lib and binaries from GitHub")
        here = os.getcwd()
        os.chdir(os.path.join(easyxrd_defaults["easyxrd_scratch_path"]))
        if os.path.isdir(
            os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II")
        ):
            shutil.rmtree(
                os.path.join(easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II")
            )

        import numpy, sys

        if (sys.version_info.minor == 11) and (
            numpy.version.version.split(".")[1] == "26"
        ):
            os.system(
                "git clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II"
            )
            os.system(
                "mkdir GSAS-II/GSASII-bin GSAS-II/GSASII-bin/linux_64_p3.11_n1.26"
            )
            os.system(
                "curl -s -L https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/linux_64_p3.11_n1.26.tgz | tar zxvf - -C GSAS-II/GSASII-bin/linux_64_p3.11_n1.26"
            )

            print('\n!!!! Please re-run this cell (after kernel restart) for the GSAS-II installation to take effect !!!!!')

            import time
            time.sleep(5)
            os._exit(1)
 
            gsasii_lib_path = os.path.join(
                easyxrd_defaults["easyxrd_scratch_path"], "GSAS-II", "GSASII"
            )

        os.chdir(here)


easyxrd_defaults["gsasii_lib_path"] = gsasii_lib_path


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
