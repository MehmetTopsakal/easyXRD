import importlib
import logging
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack, redirect_stdout
from pathlib import Path

import fabio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class HiddenPrints:
    """
    Context manager that hides stdout prints from functions.
    Useful for processes like refinement which produce voluminous terminal output.
    """

    def __init__(self):
        self._contexts = []

    def __enter__(self):
        stack = ExitStack()
        output = stack.enter_context(open(os.devnull, "w"))
        stack.enter_context(redirect_stdout(output))
        self._contexts.append(stack)
        return self

    def __exit__(self, *args):
        return self._contexts.pop().__exit__(*args)


def load_gsas_module(name, paths=()):
    """Import current or legacy GSAS-II modules without leaving sys.path changes."""
    # Preserve the Linux Qt/wx fontconfig workaround, but only when GSAS-II
    # is actually requested. Never remove another application's loaded wx.
    if sys.platform.startswith("linux"):
        library = (
            Path(sys.executable).resolve().parents[1] / "lib" / "libfontconfig.so.1"
        )
        if library.is_file():
            import ctypes

            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                logging.getLogger("easyxrd").debug(
                    "Could not preload fontconfig", exc_info=True
                )
    candidates = [None] + list(paths)
    last_error = None
    for path in candidates:
        original_path = sys.path[:]
        try:
            if path and os.path.isdir(path):
                sys.path[:0] = [str(path), str(Path(path).parent)]
            try:
                return importlib.import_module("GSASII." + name)
            except ModuleNotFoundError as exc:
                # Missing transitive dependencies must retain their real error.
                if exc.name not in ("GSASII", "GSASII." + name):
                    raise
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name not in (name, "GSASII", "GSASII." + name):
                raise
            last_error = exc
        finally:
            sys.path[:] = original_path
    raise ImportError(
        "GSAS-II is unavailable; install it or provide gsasii_lib_path"
    ) from last_error


def refinement_transaction(function):
    """Restore engine and analysis state on exceptions, retaining attempt history."""
    import copy
    from functools import wraps

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_transaction_active", False):
            return function(self, *args, **kwargs)
        names = (
            "gpx",
            "ds",
            "phases",
            "gpx_previous",
            "ds_previous",
            "phases_previous",
            "gsasii_run_directory",
            "gsasii_lib_path",
            "yshift_multiplier",
        )
        original_gpx = getattr(self, "gpx", None)
        old_steps = getattr(self, "ds", xr.Dataset()).sizes.get("refinement_step", 0)
        state = {
            name: copy.deepcopy(getattr(self, name))
            for name in names
            if hasattr(self, name)
        }
        self._transaction_active = True
        # The first pre-mutation checkpoint can borrow this protected copy.
        # Nested checkpoints after flags/cell changes still need their own copy.
        self._transaction_checkpoint = (original_gpx, state.get("gpx"))
        self._transaction_checkpoint_available = original_gpx is not None
        try:
            return function(self, *args, **kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            history = None
            if hasattr(self, "ds") and "rwp_history" in self.ds:
                history = self.ds[["rwp_history", "rwp_attempted_history"]].copy(
                    deep=True
                )
                if (
                    history.sizes["refinement_step"] > old_steps
                    and history.refinement_status.values[-1] == "accepted"
                ):
                    statuses = history.refinement_status.values.tolist()
                    reasons = history.refinement_reason.values.tolist()
                    statuses[-1], reasons[-1] = "error", f"{type(exc).__name__}: {exc}"
                    history = history.assign_coords(
                        refinement_status=("refinement_step", statuses),
                        refinement_reason=("refinement_step", reasons),
                    )
                    history.rwp_history.values[-1] = np.nan
            for name in names:
                if name in state:
                    setattr(self, name, state[name])
                elif hasattr(self, name):
                    delattr(self, name)
            if history is not None and hasattr(self, "ds"):
                self.ds = xr.merge(
                    [self.ds.drop_dims("refinement_step", errors="ignore"), history],
                    join="exact",
                )
            self.last_refinement_accepted = False
            self.last_refinement_rejection_reason = "error"
            if original_gpx is not None and hasattr(original_gpx, "__dict__"):
                original_gpx.__dict__.clear()
                original_gpx.__dict__.update(self.gpx.__dict__)
                self.gpx = original_gpx
            if hasattr(self, "gpx"):
                self.gpx_saver()
            raise
        finally:
            self._transaction_active = False
            del self._transaction_checkpoint
            del self._transaction_checkpoint_available

    return wrapped


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
        Include GUI dependencies, including wxPython (default False). Enable
        this when using the interactive GSAS-II editor.

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
            pip
            + [
                "meson-python",
                "ninja",
                "wheel",
                "Cython",
                "numpy",
                "pyproject-metadata>=0.5.0",
                "tomli>=1.0.0",
            ],
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
            [
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "import GSASII.GSASIIscriptable as G2sc; "
                "print(Path(G2sc.__file__).resolve().parent)",
            ],
            check=True,
            capture_output=True,
            text=True,
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


def read_tiff_as_DataArray(tiff, plot=False):
    """Read a TIFF image into an xarray DataArray with pixel coordinates.

    Parameters
    ----------
    tiff : str or Path
        Path to the TIFF image file.
    plot : bool, optional
        If True, displays a 2D imshow plot of the image with robust color scaling.
        Default is False.

    Returns
    -------
    xr.DataArray
        2D DataArray containing image intensity, with dimensions ('pixel_y', 'pixel_x')
        and header metadata in attrs.
    """
    with fabio.open(str(tiff)) as fabio_img:
        tiff_data = fabio_img.data.copy()
        header_dict = dict(fabio_img.header)

    da = xr.DataArray(
        tiff_data,
        dims=("pixel_y", "pixel_x"),
        coords={
            "pixel_y": np.arange(tiff_data.shape[0]),
            "pixel_x": np.arange(tiff_data.shape[1]),
        },
        attrs={"tiff_source": str(tiff), **header_dict},
    )

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        da.plot.imshow(robust=True, ax=ax, cbar_kwargs={"shrink": 0.5}, origin="upper")
        ax.set_aspect("equal")

    return da
