"""Run ``python -m easyxrd.diagnostics`` to inspect the active environment."""

import json
import platform
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def collect_diagnostics():
    """Report distribution versions and NumPy headers without printing secrets.

    Distribution metadata does not prove binary compatibility; NumPy is the only
    scientific package imported by this diagnostic.
    """
    packages = {}
    for name in ("easyxrd", "numpy", "scipy", "xarray", "pyFAI", "fabio",
                 "pymatgen", "pybaselines", "h5netcdf", "h5py", "mp-api", "GSAS-II"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed (distribution metadata absent)"
    result = {"python_executable": sys.executable, "python_version": sys.version,
              "platform": platform.platform(), "packages": packages}
    try:
        numpy = import_module("numpy")
        result["numpy_runtime"] = {"version": numpy.__version__, "file": numpy.__file__,
                                   "headers": numpy.get_include()}
    except Exception as exc:
        result["numpy_import_error"] = f"{type(exc).__name__}: {exc}"
    return result


if __name__ == "__main__":
    print(json.dumps(collect_diagnostics(), indent=2))
