"""The default window reproduces the Meyer baseline recorded at d991ac4 bit for bit.

Except the ``meyer_cases.RESCALED`` families: units (follow-up spec, section 1).  Their
values are now Wilson coefficients, ``got = f(N, dt) * recorded`` to 1e-13, with the same
dtype; every other array is still ``np.array_equal``.  ``meyer_cases.mismatches`` is the one
comparator, shared with ``meyer_cases.py --check``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import domainia.covariance
import domainia.domains
import domainia.maps
import domainia.spectra
import domainia.testing
import domainia.variance
from domainia import wilson

from .meyer_cases import BASELINE, RESCALED, THREADED, compute, mismatches, rescale_factor

DM = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                     variance=domainia.variance, covariance=domainia.covariance,
                     testing=domainia.testing)


@pytest.fixture(scope="module")
def baseline() -> dict[str, np.ndarray]:
    with np.load(BASELINE) as z:
        return {k: z[k] for k in z.files}


def test_the_rescaled_families_are_exactly_the_spec_table(baseline):
    """Five families (spec section 1.7), each present on every grid and dt of the baseline."""
    assert set(RESCALED) == {("domainia", "FourierToWDM.dense"), ("domainia", "FourierToWDM.fft"),
                             ("domainia", "FourierToWDM.adjoint"),
                             ("wilson", "transform.forward_bands"),
                             ("wilson", "transform.inverse_bands")}
    rescaled = [k for k in baseline if rescale_factor(k) is not None]
    assert len(rescaled) == 5 * 6                       # 3 grids x 2 dt each
    assert rescale_factor("domainia::FourierToWDM.fft::16x32::dt=0.25") == np.sqrt(2 * 0.25 / 512)
    assert rescale_factor("wilson::transform.inverse_bands::32x8::dt=1.0") == 1.0
    assert rescale_factor("wilson::transform.forward::8x16::dt=1.0") is None


def test_default_window_reproduces_the_meyer_baseline_bit_for_bit(baseline):
    assert not mismatches(compute(wilson, DM, {}, {}), baseline)


def test_explicit_meyer_reproduces_the_baseline_bit_for_bit(baseline):
    got = compute(wilson, DM, {"window": wilson.MEYER}, {}, only=THREADED)
    want = {k: v for k, v in baseline.items()
            if k.startswith("wilson::") and k.split("::")[1].startswith(THREADED)}
    assert want and not mismatches(got, want)


def test_explicit_meyer_on_both_layers_reproduces_the_whole_baseline(baseline):
    got = compute(wilson, DM, {"window": wilson.MEYER}, {"window": wilson.MEYER})
    assert not mismatches(got, baseline)
