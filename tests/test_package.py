"""domainia 2.0 identity: version, public names, no JAX on import, no release leftovers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import domainia
from domainia import wilson

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

PUBLIC = {"Grid", "MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError",
          "__version__", "frequency_covariance", "transform", "wilson"}


def test_the_version_is_2_0_0_everywhere():
    assert domainia.__version__ == "2.0.0"
    assert tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"] == "2.0.0"


def test_the_sdist_lists_what_it_includes():
    """The sdist takes ``src/`` and ``tests/`` only (hatchling adds README.md, LICENSE,
    pyproject.toml, .gitignore and PKG-INFO itself), so the specs and plans under docs/ and
    any untracked folder at the root stay out.  hatchling is not installed in the test env,
    so this reads the build config rather than building."""
    sdist = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert sdist["only-include"] == ["src", "tests"]
    assert "exclude" not in sdist
    assert not [path for path in sdist["only-include"] if path.split("/")[0] == "docs"]


def test_the_project_metadata():
    """The package ships ``py.typed`` and says so; ``test`` is the only extra."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert "Typing :: Typed" in project["classifiers"]
    assert (SRC / "domainia" / "py.typed").is_file()
    assert project["optional-dependencies"] == {"test": ["pytest>=8"]}


def test_the_public_names():
    assert set(domainia.__all__) == PUBLIC
    assert all(hasattr(domainia, name) for name in domainia.__all__)
    assert all(hasattr(wilson, name) for name in wilson.__all__)
    assert {"MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError", "windows",
            "require_window", "sample_window", "pack", "unpack"} <= set(wilson.__all__)
    assert wilson.pack is wilson.grid.pack and wilson.unpack is wilson.grid.unpack
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid


def test_no_release_leftovers():
    for name in ("paper", "__release__", "__wdm_api__"):
        assert not hasattr(wilson, name) and name not in wilson.__all__


def test_importing_every_module_resolves_to_src_and_does_not_import_jax():
    """Every module of the package (``pkgutil.walk_packages``), imported in a fresh
    interpreter, leaves JAX unimported.  The JAX backend and its Bessel shim are gone."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC)
    code = ("import importlib, json, pkgutil, sys, domainia\n"
            "mods = [m.name for m in pkgutil.walk_packages(domainia.__path__, 'domainia.')]\n"
            "for name in mods:\n"
            "    importlib.import_module(name)\n"
            "jax = sorted(k for k in sys.modules if k.split('.')[0] in ('jax', 'jaxlib'))\n"
            "print(json.dumps({'file': domainia.__file__, 'modules': mods, 'jax': jax}))")
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(REPO),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert Path(info["file"]).resolve().is_relative_to(SRC.resolve())
    assert {"domainia._transform", "domainia.wilson", "domainia.testing"} <= set(info["modules"])
    assert not {"domainia.wilson.backend", "domainia.wilson.special"} & set(info["modules"])
    assert info["jax"] == []


def test_importing_domainia_leaves_the_unlisted_modules_unimported():
    """The package docstring's promise: a fresh ``import domainia`` does not import
    domainia.testing."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(SRC)
    code = ("import json, sys, domainia\n"
            "print(json.dumps(sorted(k for k in sys.modules if k.startswith('domainia.'))))")
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(REPO),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert "domainia._transform" in loaded and "domainia.testing" not in loaded
