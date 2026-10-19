"""Exercise version stamping through actual commits in disposable repositories."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


def git(repo, *args, date="2026-10-19T12:00:00-04:00", check=True):
    env = os.environ.copy()
    env.update(
        GIT_AUTHOR_NAME="Version hook test",
        GIT_AUTHOR_EMAIL="version-test@example.invalid",
        GIT_COMMITTER_NAME="Version hook test",
        GIT_COMMITTER_EMAIL="version-test@example.invalid",
        GIT_AUTHOR_DATE="2025-01-01T12:00:00+00:00",
        GIT_COMMITTER_DATE=date,
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
    )
    return subprocess.run(
        ["git", "-C", str(repo), *args], env=env, text=True,
        capture_output=True, check=check,
    )


@pytest.fixture
def version_repo(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "core.autocrlf", "false")
    git(tmp_path, "config", "core.hooksPath", ".githooks")
    hook = tmp_path / ".githooks/pre-commit"
    hook.parent.mkdir()
    source = Path(__file__).resolve().parents[1] / ".githooks/pre-commit"
    shutil.copyfile(source, hook)
    hook.chmod(0o755)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "easyxrd"\nversion = "2026.09.20"\n'
        'description = "original"\n\n[tool.example]\nversion = "unchanged"\n'
    )
    (tmp_path / "README.md").write_text("Initial content\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "Initial commit", date="2026-10-18T12:00:00-04:00")
    return tmp_path


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2026-10-19T12:00:00-04:00", "2026.10.19"),
        ("2026-01-01T00:30:00+14:00", "2026.01.01"),
        ("2026-12-31T23:30:00-12:00", "2026.12.31"),
    ],
)
def test_version_uses_committer_date_and_timezone(version_repo, date, expected):
    repo = version_repo
    (repo / "README.md").write_text("Next commit\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "Next commit", date=date)

    committed = tomllib.loads(git(repo, "show", "HEAD:pyproject.toml").stdout)
    assert committed["project"]["version"] == expected
    assert committed["tool"]["example"]["version"] == "unchanged"
    assert tomllib.loads((repo / "pyproject.toml").read_text()) == committed
    assert git(repo, "status", "--porcelain").stdout == ""


@pytest.mark.parametrize("stage_manifest", [False, True])
def test_version_preserves_unstaged_manifest_edits(version_repo, stage_manifest):
    repo = version_repo
    path = repo / "pyproject.toml"
    if stage_manifest:
        path.write_text(path.read_text().replace('"original"', '"staged"'))
        git(repo, "add", "pyproject.toml")
    staged = git(repo, "show", ":pyproject.toml").stdout
    path.write_text(path.read_text().replace(
        '"staged"' if stage_manifest else '"original"', '"unstaged"'
    ))
    (repo / "README.md").write_text("Next commit\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "Preserve unstaged changes")

    committed = git(repo, "show", "HEAD:pyproject.toml").stdout
    assert committed == staged.replace("2026.10.18", "2026.10.19")
    working = tomllib.loads(path.read_text())["project"]
    assert working["version"] == "2026.10.19"
    assert working["description"] == "unstaged"
    assert git(repo, "diff", "--cached").stdout == ""


def test_same_day_commits_keep_same_version(version_repo):
    repo = version_repo
    for hour in (12, 13):
        (repo / "README.md").write_text(f"Commit at {hour}\n")
        git(repo, "add", "README.md")
        git(repo, "commit", "-m", f"Commit at {hour}",
            date=f"2026-10-19T{hour}:00:00-04:00")
        committed = tomllib.loads(git(repo, "show", "HEAD:pyproject.toml").stdout)
        assert committed["project"]["version"] == "2026.10.19"
        assert git(repo, "status", "--porcelain").stdout == ""


def test_amend_uses_new_committer_date(version_repo):
    repo = version_repo
    git(repo, "commit", "--amend", "--no-edit")
    committed = tomllib.loads(git(repo, "show", "HEAD:pyproject.toml").stdout)
    assert committed["project"]["version"] == "2026.10.19"


def test_missing_project_version_blocks_commit(version_repo):
    repo = version_repo
    previous_head = git(repo, "rev-parse", "HEAD").stdout
    path = repo / "pyproject.toml"
    path.write_text(path.read_text().replace('version = "2026.10.18"\n', ""))
    git(repo, "add", "pyproject.toml")
    result = git(repo, "commit", "-m", "Invalid package manifest", check=False)
    assert result.returncode != 0
    assert "expected one version field" in result.stderr
    assert git(repo, "rev-parse", "HEAD").stdout == previous_head


def test_staged_manifest_deletion_is_preserved(version_repo):
    repo = version_repo
    git(repo, "rm", "pyproject.toml")
    git(repo, "commit", "-m", "Remove manifest")
    assert not (repo / "pyproject.toml").exists()
    assert git(repo, "status", "--porcelain").stdout == ""
