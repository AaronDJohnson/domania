"""The default window reproduces the Meyer baseline recorded at d991ac4 bit for bit."""

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

from .meyer_cases import BASELINE, compute

DM = SimpleNamespace(domains=domainia.domains, maps=domainia.maps, spectra=domainia.spectra,
                     variance=domainia.variance, covariance=domainia.covariance,
                     testing=domainia.testing)


@pytest.fixture(scope="module")
def baseline() -> dict[str, np.ndarray]:
    with np.load(BASELINE) as z:
        return {k: z[k] for k in z.files}


def _mismatches(got: dict, want: dict) -> list[str]:
    assert sorted(got) == sorted(want), sorted(set(got) ^ set(want))[:10]
    return [k for k in want if got[k].dtype != want[k].dtype or not np.array_equal(got[k], want[k])]


def test_default_window_reproduces_the_meyer_baseline_bit_for_bit(baseline):
    assert not _mismatches(compute(wilson, DM, {}, {}), baseline)


#: Case-name prefixes whose kernels take ``window=`` (extended task by task).
THREADED = ("transform.", "matrix.")


def test_explicit_meyer_reproduces_the_baseline_bit_for_bit(baseline):
    got = compute(wilson, DM, {"window": wilson.MEYER}, {}, only=THREADED)
    want = {k: v for k, v in baseline.items()
            if k.startswith("wilson::") and k.split("::")[1].startswith(THREADED)}
    assert want and not _mismatches(got, want)
