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


@pytest.mark.parametrize("api_key", ["abcde-secret-api-key", "short", "not found", "invalid", None])
def test_show_config_masks_key_and_reports_environment(monkeypatch, capsys, api_key):
    import easyxrd
    from types import SimpleNamespace

    monkeypatch.setitem(easyxrd.easyxrd_defaults, "mp_api_key", api_key)
    monkeypatch.setitem(
        sys.modules, "GSASII",
        SimpleNamespace(GSASIIscriptable=SimpleNamespace(__file__="/test/GSASIIscriptable.py")),
    )

    assert easyxrd.show_config() is None
    output = capsys.readouterr().out
    assert "easyXRD environment and configurations" in output
    assert sys.executable in output
    assert "/test/GSASIIscriptable.py" in output
    assert easyxrd.easyxrd_defaults["easyxrd_scratch_path"] in output
    if api_key and api_key not in ("not found", "invalid"):
        assert api_key not in output
        assert "......" in output
    else:
        assert str(api_key) in output


def test_show_config_with_missing_packages(monkeypatch, capsys):
    import easyxrd
    from importlib import metadata

    def missing_version(package):
        raise metadata.PackageNotFoundError(package)

    monkeypatch.setattr(metadata, "version", missing_version)
    monkeypatch.setitem(sys.modules, "GSASII", None)
    easyxrd.show_config()
    output = capsys.readouterr().out
    assert "not installed or local" in output
    assert "GSASII_module       : not found" in output
