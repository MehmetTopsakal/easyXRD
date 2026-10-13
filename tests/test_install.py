"""Exercise installer boundaries without downloading or modifying environments."""

import os
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

import easyxrd
from easyxrd import utils


@pytest.fixture
def installer(monkeypatch, tmp_path):
    monkeypatch.setattr(utils.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(easyxrd, "easyxrd_defaults", {"gsasii_lib_path": "old"})
    pkgconfig = tmp_path / "numpy" / "_core" / "lib" / "pkgconfig"
    pkgconfig.mkdir(parents=True)
    monkeypatch.setattr(np, "__file__", str(tmp_path / "numpy" / "__init__.py"))
    monkeypatch.setenv("PKG_CONFIG_PATH", "existing")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "startup message\n/installed/GSASII\n", ""))
    monkeypatch.setattr(utils.subprocess, "run", run)
    return run, pkgconfig


def test_install_current_interpreter_and_defaults(installer):
    run, pkgconfig = installer
    assert easyxrd.install_gsas2() == "/installed/GSASII"
    assert easyxrd.easyxrd_defaults["gsasii_lib_path"] == "/installed/GSASII"
    build, package, verify = run.call_args_list
    for call in (build, package, verify):
        assert call.args[0][0] == sys.executable
        assert call.kwargs["check"] is True
    assert "meson-python" in build.args[0]
    assert "--no-build-isolation" in package.args[0]
    assert "GSAS-II[gui,useful] @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git" in package.args[0]
    assert "--upgrade" not in package.args[0]
    assert package.kwargs["env"]["PKG_CONFIG_PATH"] == str(pkgconfig) + os.pathsep + "existing"
    assert os.environ["PKG_CONFIG_PATH"] == "existing"
    assert "GSASII.GSASIIscriptable" in verify.args[0][-1]


def test_upgrade_without_gui(installer):
    run, _ = installer
    easyxrd.install_gsas2(upgrade=True, gui=False)
    command = run.call_args_list[1].args[0]
    assert "--upgrade" in command
    assert command[-1].startswith("GSAS-II[useful] @ ")


def test_missing_git_does_not_install(installer, monkeypatch):
    run, _ = installer
    monkeypatch.setattr(utils.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Git is required"):
        easyxrd.install_gsas2()
    run.assert_not_called()


@pytest.mark.parametrize("failed_step", [0, 1, 2])
def test_failures_preserve_defaults(installer, failed_step):
    run, _ = installer
    failure = subprocess.CalledProcessError(1, "command", stderr="compiler or import error")
    run.side_effect = [subprocess.CompletedProcess([], 0)] * failed_step + [failure]
    with pytest.raises(RuntimeError) as error:
        easyxrd.install_gsas2()
    assert error.value.__cause__ is failure
    assert run.call_count == failed_step + 1
    assert easyxrd.easyxrd_defaults["gsasii_lib_path"] == "old"
    if failed_step == 2:
        assert "compiler or import error" in str(error.value)
