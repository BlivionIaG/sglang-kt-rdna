"""CPU-only checks for the TileLang kernel cache route (issue #65).

TileLang derives its cache directory from ~ alone, so a container started with
`--user 1001:1001` (no passwd entry, so runc falls back to HOME=/) tried to
create /.tilelang at import time and died with EACCES, while the documented
default UID 1000 kept its compiled kernels in the disposable image layer and
recompiled them on every restart. Every launch path -- the native recipes, the
image's own profiles, the shipped and configurator-generated mounted startup
files, both model profiles -- now names the cache under the durable cache root,
creates it before the launch runs its interpreter, and still honours an explicit
operator TILELANG_CACHE_DIR: nothing bakes a value into the image, because ENV
cannot be told apart from the operator's own identical choice. No GPU and no
container runtime: the real image entry point runs against a read-only HOME and a
recording stand-in recipe.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
CONFIGS = REPO / "configs" / "pennyroyal"
CONTAINER = REPO / "docker" / "pennyroyal"
ENTRYPOINT = CONTAINER / "entrypoint.sh"
# Every launch path that ends up importing SGLang, so none of them may leave a
# compiler cache to ~.
LAUNCH_PATHS = (
    CONFIGS / "serve-flash-next-frspec.sh",
    CONFIGS / "serve-flash-next.sh",
    CONFIGS / "serve-qwen38-27b-dflash2.sh",
    CONTAINER / "launch" / "config" / "start-flash-next-frspec.sh",
    CONTAINER / "launch" / "config" / "start-flash-next.sh",
    CONTAINER / "launch" / "config" / "start-27b-dflash2.sh",
)
EXPORT = 'export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$CACHE_BASE/tilelang}"'
CREATED = 'mkdir -p "$TILELANG_CACHE_DIR"'
# The image's own profile routing, so the stand-in lands where the real one is.
RECIPE_BY_PROFILE = {
    "next": "serve-flash-next-frspec.sh",
    "next-plain": "serve-flash-next.sh",
    "27b": "serve-qwen38-27b-dflash2.sh",
}
# Where a launch path first hands work to an interpreter that imports SGLang
# (the optional NVMe preflight, the identity probe, the server itself). The
# container files' earlier torch-only device-count probe is not one of these.
IMPORT_MARKERS = (
    "ple-backend.sh",
    "PENNY_IDENTITY_PROBE=",
    'TORCH_VERSION="$("$PYTHON"',
    'exec "$SGLANG_EXE"',
)


def test_every_launch_path_routes_tilelang_under_the_cache_root():
    for path in LAUNCH_PATHS:
        text = path.read_text()
        assert EXPORT in text, path
        assert CREATED in text, path
        imports = [
            index
            for marker in IMPORT_MARKERS
            if marker in text and (index := text.index(marker)) >= 0
        ]
        assert imports, f"no interpreter entry point in {path}"
        # Set up, and the directory created, before that first import.
        assert text.index(CREATED) < min(imports), path
        assert text.index(EXPORT) < min(imports), path


def test_the_generated_startup_file_carries_the_route(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "pennyroyal"))
    import penny_config as pc  # noqa: E402

    for name in ("models", "cache", "nixl"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "models" / "some-model").mkdir()
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / "models" / "some-model" / name).write_text('{"marker": "x"}')
    saved = {
        "HOST_MODELS_ROOT": str(tmp_path / "models"),
        "HOST_CACHE_BASE": str(tmp_path / "cache"),
        "HOST_NIXL_STORAGE_BASE": str(tmp_path / "nixl"),
        "LAUNCH_DIR": str(tmp_path / "generated"),
        "TARGET_MODEL": "/models/some-model",
    }
    env_file = tmp_path / "container.env"
    env_file.write_text(pc.serialize_env([("", sorted(saved.items()))]))
    plan = pc.build_plan(
        "container",
        pc.load_config("container", env_file, {}, REPO),
        {},
        repo_root=REPO,
    )
    assert [issue.message for issue in plan.errors] == []
    files = pc.container_launch_files(plan)
    assert any(EXPORT in text and CREATED in text for _, text in files), files


def _entrypoint(
    tmp_path: Path,
    profile: str = "next",
    run: tuple[str, ...] | None = None,
    **updates: str,
) -> subprocess.CompletedProcess[str]:
    """Run the real image entry point over a stand-in /opt/pennyroyal.

    The stand-in recipe records the cache path it was handed, which is what the
    image's own recipe (and so TileLang) would see. HOME is read-only: a cache
    left to ~ cannot be created, anywhere.
    """
    root = tmp_path / "opt pennyroyal"
    # The recipe this profile's built-in entry route execs.
    recipe = root / "configs" / "pennyroyal" / RECIPE_BY_PROFILE[profile]
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text('printf "%s\\n" "${TILELANG_CACHE_DIR:-unset}"\n')
    cache, nixl, home = tmp_path / "cache", tmp_path / "nixl", tmp_path / "home"
    for directory in (cache, nixl, home):
        directory.mkdir(exist_ok=True)
    home.chmod(0o500)
    env = {
        name: value
        for name, value in os.environ.items()
        if name != "TILELANG_CACHE_DIR"
    }
    env.update(
        REPO_ROOT=str(root),
        CACHE_BASE=str(cache),
        NIXL_STORAGE_BASE=str(nixl),
        # The entry point asks the interpreter how many GPUs Docker granted;
        # this route is about the cache path, so the device view is given.
        CUDA_VISIBLE_DEVICES="0",
        HOME=str(home),
        **updates,
    )
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *(run or (profile,))],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_the_container_entry_point_ignores_home_entirely(tmp_path):
    # The reported crash case: no passwd entry means HOME=/, so nothing may be
    # created outside the mounts. The default follows the durable root instead,
    # for every built-in profile.
    for profile in RECIPE_BY_PROFILE:
        result = _entrypoint(tmp_path, profile)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(tmp_path / "cache" / "tilelang")
        assert (tmp_path / "cache" / "tilelang").is_dir()


def test_the_image_bakes_no_tilelang_path(tmp_path):
    # Nothing may set TILELANG_CACHE_DIR in the image: an ENV value and an
    # explicit operator choice of the same string cannot be told apart, so a
    # baked default would silently outrank the operator's path. The entry point
    # supplies the default instead, under a CACHE_BASE that itself defaults to
    # /cache -- so the untouched image still keeps kernels on that mount.
    dockerfile = (CONTAINER / "Dockerfile").read_text()
    assert "TILELANG_CACHE_DIR=" not in dockerfile  # the guide may name the variable
    entrypoint = ENTRYPOINT.read_text()
    assert 'export CACHE_BASE="${CACHE_BASE:-/cache}"' in entrypoint
    assert EXPORT in entrypoint


def test_a_moved_cache_root_takes_the_kernels_with_it(tmp_path):
    # No TILELANG_CACHE_DIR at all, CACHE_BASE elsewhere: the default has to
    # follow the mount the operator actually prepared, for the built-in profile
    # route and for a mounted startup script handed to 'pennyroyal exec'.
    script = tmp_path / "launch config" / "start.sh"
    script.parent.mkdir(exist_ok=True)
    script.write_text('printf "%s\\n" "${TILELANG_CACHE_DIR:-unset}"\n')
    for run, prepared in ((None, True), (("exec", "bash", str(script)), False)):
        shutil.rmtree(tmp_path / "cache" / "tilelang", ignore_errors=True)
        result = _entrypoint(tmp_path, run=run)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(tmp_path / "cache" / "tilelang"), run
        # Only the profile route reaches the entry point's own mkdir; the exec'd
        # script prepares the directories it uses, as its shipped copy does.
        assert (tmp_path / "cache" / "tilelang").is_dir() == prepared, run


def test_an_explicit_path_is_never_rewritten(tmp_path):
    # '/cache/tilelang' is exactly the value the image used to bake. Named by the
    # operator while CACHE_BASE points elsewhere, it must survive unchanged --
    # that identity is the one thing a derived default cannot guess. The exec
    # route is checked here because it hands over before the entry point's own
    # mkdir, so no test host needs a real /cache.
    script = tmp_path / "launch config" / "start.sh"
    script.parent.mkdir(exist_ok=True)
    script.write_text('printf "%s\\n" "${TILELANG_CACHE_DIR:-unset}"\n')
    for path in ("/cache/tilelang", str(tmp_path / "operator kernels")):
        result = _entrypoint(
            tmp_path, run=("exec", "bash", str(script)), TILELANG_CACHE_DIR=path
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == path
    # On the profile route the same value is also what the recipe gets, and the
    # directory the entry point exported is the one it prepares.
    chosen = tmp_path / "operator kernels"
    result = _entrypoint(tmp_path, TILELANG_CACHE_DIR=str(chosen))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(chosen)
    assert chosen.is_dir()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
