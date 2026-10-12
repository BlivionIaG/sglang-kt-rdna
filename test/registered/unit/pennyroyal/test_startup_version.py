"""CPU-only checks for the Penny Royal release identity printed at startup.

The startup banner has to name the Penny Royal release and the exact source it
runs, in every launch route. Nothing here can come from Git tags -- the published
container is a shallow single-commit checkout with no tags at all -- and nothing
may borrow SGLang's own package version, whose 0.5.x number (and 0.0.0.dev0
setuptools-scm fallback) is not a Penny Royal release. A locally modified
checkout says so instead of claiming a clean release.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CONFIGS = REPO / "configs" / "pennyroyal"
SUMMARY = CONFIGS / "startup-summary.sh"
LAUNCH_PATHS = (
    CONFIGS / "serve-flash-next-frspec.sh",
    CONFIGS / "serve-flash-next.sh",
    CONFIGS / "serve-qwen38-27b-dflash2.sh",
    REPO / "docker" / "pennyroyal" / "launch" / "config" / "start-flash-next-frspec.sh",
    REPO / "docker" / "pennyroyal" / "launch" / "config" / "start-flash-next.sh",
    REPO / "docker" / "pennyroyal" / "launch" / "config" / "start-27b-dflash2.sh",
)
VERSION = (CONFIGS / "pennyroyal.version").read_text().strip()


def checkout(tmp_path: Path, tags: tuple[str, ...] = ()) -> Path:
    """A throwaway tagless single-commit source tree using the real configs."""
    root = tmp_path / "opt pennyroyal"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "pennyroyal").symlink_to(CONFIGS)
    (root / "VERSION").write_text("stand-in source\n")
    for command in (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.email", "t@example.com"],
        ["git", "-C", str(root), "config", "user.name", "T"],
        ["git", "-C", str(root), "add", "VERSION"],
        ["git", "-C", str(root), "commit", "-qm", "one commit, no tags"],
    ):
        subprocess.run(command, check=True, capture_output=True, text=True)
    for tag in tags:
        subprocess.run(
            ["git", "-C", str(root), "tag", tag],
            check=True,
            capture_output=True,
            text=True,
        )
    return root


def banner(root: Path | None) -> str:
    env = {**os.environ, "REPO_ROOT": str(root) if root else "/nonexistent-root"}
    return subprocess.check_output(
        ["bash", "-c", 'source "$1"; pennyroyal_version_banner', "bash", str(SUMMARY)],
        env=env,
        text=True,
    )


def test_the_version_asset_is_a_penny_royal_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION), VERSION
    # A three-part release line, and specifically not the SGLang package
    # version (0.5.x upstream, 0.0.0.dev0 when setuptools-scm sees no tags).
    assert not VERSION.startswith("0.")
    assert (CONFIGS / "pennyroyal.version").read_text().endswith("\n")


def test_a_tagless_shaped_checkout_names_release_and_commit(tmp_path):
    root = checkout(tmp_path)
    assert (
        subprocess.run(  # no tag can be the source of the number
            ["git", "-C", str(root), "tag", "-l"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short=10", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert banner(root) == f"Penny Royal {VERSION} | source {revision}\n"


def test_a_stray_tag_does_not_become_the_release(tmp_path):
    root = checkout(tmp_path, ("v9.9.9",))
    assert banner(root).split()[2] == VERSION


def test_local_modifications_are_named_rather_than_hidden(tmp_path):
    root = checkout(tmp_path)
    (root / "VERSION").write_text("someone edited the image source\n")
    assert "locally modified" in banner(root)
    assert "clean" not in banner(root)


def test_a_source_without_git_still_names_the_release(tmp_path):
    # No repository at all (a copied tree, or git refusing the path): the
    # release line survives and the revision is admitted to be unknown.
    assert banner(root=None) == f"Penny Royal {VERSION} | source revision unavailable\n"


def test_every_launch_path_prints_the_identity_before_its_preflight():
    for path in LAUNCH_PATHS:
        text = path.read_text()
        assert "pennyroyal_version_banner" in text, path
        configs = (
            "$IMAGE_CONFIGS" if "IMAGE_CONFIGS/tp-devices.sh" in text else "$SCRIPT_DIR"
        )
        assert f'source "{configs}/startup-summary.sh"\npennyroyal_version_banner' in (
            text
        ), path
        # Once: the identity is stated ahead of the settings summary, not twice.
        assert text.count("pennyroyal_version_banner") == 1, path
        assert text.index("pennyroyal_version_banner") < text.index(
            'mkdir -p "$CACHE_BASE"'
        ), path


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
