import sys
import io
import pytest

def test_top_level_import():
    import easyxrd
    from easyxrd import exrd
    from easyxrd.core import exrd as core_exrd

    assert exrd is core_exrd
    assert hasattr(easyxrd, "easyxrd_defaults")
    assert hasattr(easyxrd, "set_defaults")
    assert hasattr(easyxrd, "print_defaults")


def test_defaults_configuration():
    from easyxrd import easyxrd_defaults, set_defaults, print_defaults

    original = easyxrd_defaults.get("custom_key")
    try:
        set_defaults("custom_key", "test_value")
        assert easyxrd_defaults["custom_key"] == "test_value"

        # Ensure print_defaults runs without error
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            print_defaults()
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "custom_key : test_value" in output
    finally:
        if original is None:
            easyxrd_defaults.pop("custom_key", None)
        else:
            easyxrd_defaults["custom_key"] = original


def test_hidden_prints():
    from easyxrd import HiddenPrints

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        with HiddenPrints():
            print("This should not be in captured")
    finally:
        sys.stdout = old_stdout

    assert "This should not be in captured" not in captured.getvalue()
    assert sys.stdout is old_stdout
