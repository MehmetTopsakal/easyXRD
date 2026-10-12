import os
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
