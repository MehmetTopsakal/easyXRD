import importlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys


def ensure_wx_importable():
    """Import wxPython, repairing only its known stale-import state.

    GSAS-II imports wxPython lazily while loading export routines. In long-lived
    processes such as Jupyter kernels, a previous interrupted wx import can
    leave ``wx`` or one of its submodules in ``sys.modules``. That state causes
    wxPython's ``__init__`` to fail with a misleading circular-import error.
    """
    try:
        import wx  # noqa: F401
    except AttributeError as exc:
        if not (
            "partially initialized module 'wx'" in str(exc)
            or "has no attribute '__version__'" in str(exc)
        ):
            raise

        # wx has not completed initialization, so no caller can safely rely on
        # its modules. Remove the whole partially loaded package and retry once.
        for module_name in list(sys.modules):
            if module_name == "wx" or module_name.startswith("wx."):
                sys.modules.pop(module_name, None)
        import wx  # noqa: F401


class HiddenPrints:
    """
    Context manager that hides stdout prints from functions.
    Useful for processes like refinement which produce voluminous terminal output.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def install_gsas2(*, upgrade=False, gui=False):
    """Install GSAS-II from its official GitHub repository using pip.

    Installation targets the running Python interpreter (including a notebook's
    kernel environment). Git and working C/Fortran compilers must already be
    installed. Build dependencies are installed first, then GSAS-II is built
    against this environment's NumPy with build isolation disabled.

    Parameters
    ----------
    upgrade : bool, optional
        Ask pip to upgrade an existing GSAS-II installation.
    gui : bool, optional
        Include GUI dependencies, including wxPython (default True). easyXRD's
        refinement setup currently requires wxPython even in notebooks. Set
        False for GSAS-II scripting alone or when wxPython is managed separately.

    Returns
    -------
    str
        Installed GSASII library directory, also saved in easyxrd_defaults.

    Raises
    ------
    RuntimeError
        If Git is missing, pip fails, or the installed scripting module cannot
        be imported in a fresh Python process. Pip output is streamed directly.

    Notes
    -----
    Internet access is required. This changes the current Python environment;
    use a dedicated virtual/conda environment. Restart Python or the notebook
    kernel afterward so already-loaded dependencies use the installed versions.
    No installation is performed automatically by importing this module.
    """
    if shutil.which("git") is None:
        raise RuntimeError("Git is required to install GSAS-II. Install Git and retry.")

    pip = [sys.executable, "-m", "pip", "install"]
    try:
        subprocess.run(
            pip + ["meson-python", "ninja", "wheel", "Cython", "numpy",
                   "pyproject-metadata>=0.5.0", "tomli>=1.0.0"],
            check=True,
        )
        # NumPy 2 provides pkg-config metadata for the GSAS-II Meson build.
        # Derive the location from this interpreter, never a different numpy-config
        # executable that happens to be on PATH (common in notebook sessions).
        import numpy as np

        env = os.environ.copy()
        numpy_root = Path(np.__file__).resolve().parent
        for core in ("_core", "core"):
            pkgconfig = numpy_root / core / "lib" / "pkgconfig"
            if pkgconfig.is_dir():
                env["PKG_CONFIG_PATH"] = os.pathsep.join(
                    filter(None, [str(pkgconfig), env.get("PKG_CONFIG_PATH")])
                )
                break

        extras = "gui,useful" if gui else "useful"
        command = pip + ["--no-build-isolation"]
        if upgrade:
            command.append("--upgrade")
        command.append(
            f"GSAS-II[{extras}] @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git"
        )
        subprocess.run(command, check=True, env=env)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(
            "GSAS-II installation failed; see pip output above. Check that Git, "
            "C/Fortran compilers and platform build libraries are available. "
            "See https://advancedphotonsource.github.io/GSAS-II-tutorials/install-pip.html"
        ) from exc

    try:
        # A subprocess avoids validating stale modules already loaded in Jupyter.
        result = subprocess.run(
            [sys.executable, "-c",
             "from pathlib import Path; "
             "import GSASII.GSASIIscriptable as G2sc; "
             "print(Path(G2sc.__file__).resolve().parent)"],
            check=True, capture_output=True, text=True,
        )
        library_path = result.stdout.strip().splitlines()[-1]
    except (subprocess.CalledProcessError, OSError, IndexError) as exc:
        details = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            "Pip completed, but GSAS-II scripting verification failed. "
            f"Restart Python after resolving the following error:\n{details}"
        ) from exc

    from . import easyxrd_defaults

    importlib.invalidate_caches()
    easyxrd_defaults["gsasii_lib_path"] = library_path
    logging.getLogger("easyxrd").warning(
        "GSAS-II installed at %s. Restart Python or your notebook kernel before use.",
        library_path,
    )
    return library_path
