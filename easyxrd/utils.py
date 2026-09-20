import os
import sys


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
