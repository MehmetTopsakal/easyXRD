import os
import sys
import logging
from pathlib import Path

# Preload Pixi environment's libfontconfig to prevent symbol collision (FcConfigSetDefaultSubstitute)
# between PyQt6 (which loads the system fontconfig) and wxPython/Pango (which requires Pixi's fontconfig).
if sys.platform.startswith("linux"):
    try:
        import ctypes
        fc_lib = Path(sys.executable).resolve().parents[1] / "lib" / "libfontconfig.so.1"
        if fc_lib.is_file():
            ctypes.CDLL(str(fc_lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

from .utils import HiddenPrints, install_gsas2

logger = logging.getLogger("easyxrd")




# defaults
easyxrd_defaults = dict()
user_home = os.path.expanduser("~")

# Setting up easyxrd_scratch folder
scratch_dir = os.environ.get(
    "EASYXRD_SCRATCH", os.path.join(user_home, ".easyxrd_scratch")
)
os.makedirs(scratch_dir, exist_ok=True)
easyxrd_defaults["easyxrd_scratch_path"] = scratch_dir

try:
    import sys
    sys.modules.pop("wx", None)
    sys.modules.pop("wx.__version__", None)
except:
    pass
# Check GSAS-II library path
try:
    with HiddenPrints():
        import GSASII.GSASIIscriptable as G2sc
        gsas_path = str(Path(G2sc.__file__).resolve().parent)
    easyxrd_defaults["gsasii_lib_path"] = gsas_path
except Exception:
    easyxrd_defaults["gsasii_lib_path"] = "not found"

# Check Materials Project API key from env or scratch folder
mp_api_key = os.environ.get("MP_API_KEY")
if not mp_api_key:
    key_file = os.path.join(scratch_dir, "mp_api_key.dat")
    if os.path.isfile(key_file):
        try:
            with open(key_file, "r") as api_key_file:
                api_key_file_content = api_key_file.read().strip().split()[-1]
                if len(api_key_file_content) == 32:
                    mp_api_key = api_key_file_content
                else:
                    mp_api_key = "invalid"
        except Exception:
            mp_api_key = "invalid"
    else:
        mp_api_key = "not found"

easyxrd_defaults["mp_api_key"] = mp_api_key or "not found"


def set_defaults(name, val):
    """Set a global configuration variable."""
    global easyxrd_defaults
    easyxrd_defaults[name] = val


def print_defaults():
    """Print current configuration defaults."""
    for key, val in easyxrd_defaults.items():
        if key != "mp_api_key":
            print("%s : %s" % (key, val))
        else:
            masked = "%s.........." % val[:9] if val and len(val) >= 9 else val
            print("%s : %s" % (key, masked))


def show_config():
    """Print environment versions and configuration, with the API key masked."""
    import platform
    from importlib import metadata

    versions = {
        "Platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    for package in [
        "easyxrd", "pyFAI", "fabio", "numpy", "scipy", "pandas",
        "pymatgen", "GSAS-II",
    ]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed or local"

    try:
        with HiddenPrints():
            from GSASII import GSASIIscriptable
        versions["GSASII_module"] = GSASIIscriptable.__file__
    except Exception:
        versions["GSASII_module"] = "not found"

    print("=" * 40)
    print("easyXRD environment and configurations")
    print("=" * 40)
    for key, value in versions.items():
        print(f"{key:<20}: {value}")
    for key, value in easyxrd_defaults.items():
        if key == "mp_api_key" and value not in (None, "", "not found", "invalid"):
            value = str(value)
            value = value[:5] + "......" if len(value) > 5 else "......"
        print(f"{key:<20}: {value}")
    print("-" * 40)
    print()


# Re-export exrd for convenient package imports
from .core import exrd

__all__ = [
    "exrd",
    "install_gsas2",
    "easyxrd_defaults",
    "set_defaults",
    "print_defaults",
    "show_config",
    "HiddenPrints",
]
