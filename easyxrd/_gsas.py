"""Load GSAS-II only when refinement is requested."""

import importlib
import sys
from pathlib import Path

from . import easyxrd_defaults


def load_gsasii(path=None):
    """Return the scriptable module from an installed package or legacy directory.

    An explicit directory takes priority. Failed legacy imports restore sys.path.
    A successful legacy path remains available for GSAS-II's later imports.
    """
    configured = path if path is not None else easyxrd_defaults["gsasii_lib_path"]
    if configured not in (None, "none", "not found", "invalid"):
        directory = Path(configured).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"GSAS-II directory does not exist: {directory}")
        previous_path = sys.path[:]
        sys.path.insert(0, str(directory))
        try:
            module = importlib.import_module("GSASIIscriptable")
        except Exception as exc:
            sys.path[:] = previous_path
            raise ImportError(f"Cannot load GSAS-II from {directory}: {exc}") from exc
    else:
        try:
            module = importlib.import_module("GSASII.GSASIIscriptable")
        except ModuleNotFoundError as exc:
            if exc.name not in ("GSASII", "GSASII.GSASIIscriptable"):
                raise ImportError(f"GSAS-II dependency is missing: {exc}") from exc
            try:
                module = importlib.import_module("GSASIIscriptable")
            except ImportError as legacy_exc:
                raise ImportError(
                    "GSAS-II is unavailable. Install GSAS-II in this Python environment "
                    "or pass gsasii_lib_path='/path/to/GSASII' to setup_gsas2_refiner(). "
                    f"Original error: {legacy_exc}"
                ) from legacy_exc
    return module
