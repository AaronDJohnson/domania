"""domainia 2.0 identity: version, public names, no JAX on import, no release leftovers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import domainia
from domainia import translate, wilson

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

PUBLIC = {
    "CellSet", "ChannelAxis", "FourierBins", "FourierDomain", "Grid", "MEYER", "Meyer", "Operator",
    "RampWindow", "SampledWindow", "TimeDomain", "TimeSeries", "TranslatedOperator", "WDMCells",
    "WDMDomain", "Window", "WindowError", "__version__", "data", "domains", "linalg",
    "log_coordinate_scale", "transform", "translate", "translate_operator", "wilson",
}


def test_the_version_is_2_0_0_everywhere():
    assert domainia.__version__ == "2.0.0"
    assert tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"] == "2.0.0"


def test_the_public_names():
    assert set(domainia.__all__) == PUBLIC
    assert all(hasattr(domainia, name) for name in domainia.__all__)
    assert all(hasattr(wilson, name) for name in wilson.__all__)
    assert all(hasattr(translate, name) for name in translate.__all__)
    assert {"MEYER", "Meyer", "RampWindow", "SampledWindow", "Window", "WindowError", "windows",
            "require_window", "sample_window", "pack", "unpack"} <= set(wilson.__all__)
    assert wilson.pack is wilson.grid.pack and wilson.unpack is wilson.grid.unpack
    assert domainia.MEYER is wilson.MEYER and domainia.Grid is wilson.Grid


def test_no_release_leftovers():
    for name in ("paper", "__release__", "__wdm_api__"):
        assert not hasattr(wilson, name) and name not in wilson.__all__


def test_importing_every_module_resolves_to_src_and_does_not_import_jax():
    """Every module of the package (``pkgutil.walk_packages``), imported in a fresh
    interpreter, leaves JAX unimported."""
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
    assert {"domainia.translate", "domainia.wilson", "domainia.wilson.backend", "domainia.linalg",
            "domainia.covariance.base", "domainia.maps.time", "domainia.testing"} <= set(info["modules"])
    assert info["jax"] == []
