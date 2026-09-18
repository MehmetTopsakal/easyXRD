"""Configuration for easyXRD; scientific dependencies load with ``easyxrd.core``."""

import os
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("easyxrd")
except PackageNotFoundError:
    __version__ = "0.1.6"


class HiddenPrints:
    """Temporarily suppress stdout (process-wide; not thread-safe)."""

    def __enter__(self):
        self._sink = open(os.devnull, "w")
        self._original_stdout = sys.stdout
        sys.stdout = self._sink
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._sink.close()


def _read_api_key(path):
    """Read the historical key format without failing on empty/unreadable files."""
    try:
        fields = Path(path).read_text().split()
    except FileNotFoundError:
        return "not found"
    except (OSError, UnicodeError):
        return "invalid"
    return fields[-1] if fields and len(fields[-1]) == 32 else "invalid"


user_home = str(Path.home())
_scratch = Path(user_home) / ".easyxrd_scratch"
easyxrd_defaults = {
    "easyxrd_scratch_path": str(_scratch),
    "gsasii_lib_path": "not found",
    "mp_api_key": _read_api_key(_scratch / "mp_api_key.dat"),
}


def set_defaults(name, val):
    """Set a configuration value used by subsequent operations."""
    easyxrd_defaults[name] = val


def print_defaults():
    """Display configuration without exposing any part of the API key."""
    for key, val in easyxrd_defaults.items():
        if key == "mp_api_key" and val not in ("invalid", "not found", "none", None):
            val = "<configured>"
        print(f"{key} : {val}")
