"""The frozen oracle: its bytes are pinned and it imports as its own package."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parents[1] / "src"
ORACLE = TESTS / "oracle" / "vendored_2f7136c"
#: sha256 of the oracle copies (relative imports rewritten to vendored_2f7136c.*).
ORACLE_SHA256 = {
    "filter.py": "ea52a6e394a5d8783f0dfe61bb038b706f737ed1beffee618083f883bfd7754b",
    "grid.py": "5d9f10a23af55fcba8e869944e541dfe178666d8a683463460e3f1b087fe2594",
    "noise.py": "5419255939429dc6e95277efc58eac9c1b96b06033005b548dd88c045501e368",
    "stationary.py": "bd7d2fbf60cb989ded1f7f5b4ff74f3edb30574aa6825bcdc5b9381b489bbd62",
    "transform.py": "b7d4922a5d7c47f7beef4e20e267fffc3a66d3ed2ce32bd69f4f258c6bcb9091",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(code: str, *paths: Path) -> str:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(str(p) for p in (SRC, *paths)))
    out = subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(TESTS),
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_oracle_imports_as_its_own_package_without_wdm():
    out = _run(
        "import sys\n"
        "import vendored_2f7136c.transform, vendored_2f7136c.noise\n"
        "import vendored_2f7136c.stationary, vendored_2f7136c.filter, vendored_2f7136c.grid\n"
        "loaded = any(k.split('.')[0] in ('wdm', 'domania') for k in sys.modules)\n"
        "print(loaded, vendored_2f7136c.transform.__file__)",
        ORACLE.parent,
    )
    loaded, path = out.split(" ", 1)
    assert loaded == "False"
    assert Path(path).resolve().is_relative_to(ORACLE.resolve())


@pytest.mark.parametrize("name", sorted(ORACLE_SHA256))
def test_oracle_bytes_are_pinned(name):
    assert _sha(ORACLE / name) == ORACLE_SHA256[name]
