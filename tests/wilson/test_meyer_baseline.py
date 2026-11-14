"""The default window reproduces the Meyer baseline recorded at d991ac4, to 1e-12 of each
array's scale (domainia 3.0 spec, section 6.1).

``meyer_cases.mismatches`` is the one comparator, shared with ``meyer_cases.py --check``.  The
``meyer_cases.DROPPED`` families, of code that 3.0 deletes, are set aside; every other recorded
key is recomputed under its old name.
"""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson

from .meyer_cases import DROPPED, RTOL, THREADED, compute, load_baseline, mismatches


@pytest.fixture(scope="module")
def baseline() -> dict[str, np.ndarray]:
    return load_baseline()


def test_the_dropped_families_are_96_of_the_1504_keys(baseline):
    """maps 54, inner_product 24 and the band-limited entry points 18."""
    assert len(baseline) == 1504
    assert sum(k.startswith(DROPPED) for k in baseline) == 96


def test_the_comparator():
    """dtype and shape exact, integers exact, floats to ``RTOL`` of the array's largest value,
    an all-zero array to ``RTOL`` of its function's; a NaN fails."""
    a, z, i = ("wilson::noise.band_averaged_variance::8x16::dt=1.0",
               "wilson::noise.band_averaged_variance::8x16::dt=0.25",
               "wilson::evolutionary.wrap_cut_columns::8x16::dt=1.0")
    w = np.random.default_rng(0).standard_normal((3, 4))
    want = {a: w, z: np.zeros(3), i: np.arange(3)}
    peak = float(np.max(np.abs(w)))
    nan = w.copy()
    nan[1, 2] = np.nan
    assert mismatches({a: w + 0.5 * RTOL * peak, z: np.full(3, 0.5 * RTOL * peak), i: np.arange(3)},
                      want) == []
    for k, got in ((a, w + 2 * RTOL * peak), (a, nan), (a, w.astype(np.float32)), (a, w[:2]),
                   (z, np.full(3, 2 * RTOL * peak)), (i, np.arange(3) + 1), (i, np.arange(3.0))):
        assert mismatches({**want, k: got}, want) == [k]
    s = a.replace("8x16", "16x32")                # a's family, an array 1e-6 as large as a
    small = {**want, s: 1e-6 * w}
    assert mismatches({**small, s: 1e-6 * w + 0.5 * RTOL * peak}, small) == [s]
    with pytest.raises(AssertionError, match="baseline keys differ"):
        mismatches({a: w}, want)
    assert mismatches({}, {k: w for k in ("domainia::TimeToWDM.apply::8x16::dt=1.0",
                                          "wilson::inner_product.gaussian::8x16::dt=1.0")}) == []


def test_default_window_reproduces_the_meyer_baseline(baseline):
    assert not mismatches(compute(wilson, {}, {}), baseline)


def test_explicit_meyer_reproduces_the_baseline(baseline):
    got = compute(wilson, {"window": wilson.MEYER}, {}, only=THREADED)
    want = {k: v for k, v in baseline.items()
            if k.startswith("wilson::") and k.split("::")[1].startswith(THREADED)}
    assert want and not mismatches(got, want)


def test_explicit_meyer_on_both_layers_reproduces_the_whole_baseline(baseline):
    got = compute(wilson, {"window": wilson.MEYER}, {"window": wilson.MEYER})
    assert not mismatches(got, baseline)
