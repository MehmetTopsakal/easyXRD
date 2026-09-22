import os
import sys
import logging
from pathlib import Path

from .utils import HiddenPrints

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


# Re-export exrd for convenient package imports
from .core import exrd

__all__ = [
    "exrd",
    "easyxrd_defaults",
    "set_defaults",
    "print_defaults",
    "HiddenPrints",
]
