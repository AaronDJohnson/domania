"""C3 / G5: forward_time(d) reproduces the stored w_d of every sinusoid case.

Bit for bit up to v0.11.0.  The stored w_d came from the omega-normalised window; the Hz
window of v0.12.0 (Paper I eq:Meyer-def) is the same product rounded differently, so the
gate is W_D_RTOL of max|w_d| -- the loader's own check (wdm_models.sinusoid.data.W_D_RTOL,
kept as a literal so the wdm tests do not import wdm_models).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import wdm

DATA = Path(__file__).resolve().parents[3] / "data" / "sinusoid"
CASES = ("manuscript_low", "manuscript_mid", "manuscript_high")
NF, NT, DT = 64, 256, 2.0**-10
#: forward_time(d) vs the stored w_d, relative to max|w_d|.  Hz window (v0.12.0): measured
#: 3.1e-16 (max over the three cases).
W_D_RTOL = 1e-15


def _load(case):
    path = DATA / case / "toy_data.npz"
    if not path.is_file():
        pytest.skip(f"{path} is missing: run section A, Task A4 (it ships "
                    "data/sinusoid/manuscript_*/toy_data.npz) first")
    return np.load(path)


@pytest.mark.parametrize("case", CASES)
def test_wd_to_rounding(case):
    z = _load(case)
    w = wdm.forward_time(z["d"], NF, NT, DT)
    assert w.dtype == z["w_d"].dtype
    assert np.max(np.abs(w - z["w_d"])) <= W_D_RTOL * np.max(np.abs(z["w_d"]))


@pytest.mark.parametrize("case", CASES)
def test_wd_jax(case, jax_be):
    import jax

    z = _load(case)
    out = wdm.forward_time(jax_be.asarray(z["d"]), NF, NT, DT)
    assert isinstance(out, jax.Array), type(out)      # no eager NumPy fallback
    w = np.asarray(out)
    assert np.max(np.abs(w - z["w_d"])) <= 1e-13 * np.max(np.abs(z["w_d"]))
