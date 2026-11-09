import os
import sys
import logging
from pathlib import Path

from .utils import HiddenPrints, install_gsas2, read_tiff_as_DataArray

logger = logging.getLogger("easyxrd")
logger.addHandler(logging.NullHandler())


def configure_logging(level=logging.INFO, stream=None):
    """Enable easyXRD progress messages without configuring the root logger.

    Repeated calls replace only the handler installed by this function.
    Use level="WARNING" for rejections only or "ERROR" for quiet batch runs.
    Third-party GSAS-II stdout remains controlled by sample.verbose.
    """
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler._easyxrd_owned = True
    logger.setLevel(level)
    for existing in list(logger.handlers):
        if getattr(existing, "_easyxrd_owned", False):
            logger.removeHandler(existing)
            existing.close()
    logger.addHandler(handler)
    logger.propagate = False
    return logger




# defaults
easyxrd_defaults = dict()
user_home = os.path.expanduser("~")

# Setting up easyxrd_scratch folder
scratch_dir = os.environ.get(
    "EASYXRD_SCRATCH", os.path.join(user_home, ".easyxrd_scratch")
)
easyxrd_defaults["easyxrd_scratch_path"] = scratch_dir

# GSAS-II is imported on first refinement use, not during package import.
easyxrd_defaults["gsasii_lib_path"] = os.environ.get("GSASII_PATH", "not found")

# Check Materials Project API key from env or scratch folder
mp_api_key = os.environ.get("MP_API_KEY")
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
            masked = "......" if val not in (None, "", "not found", "invalid") else val
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
        "easyxrd", "pyFAI", "fabio", "numpy", "scipy",
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
            value = "......"
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
    "configure_logging",
    "HiddenPrints",
    "read_tiff_as_DataArray",
]
