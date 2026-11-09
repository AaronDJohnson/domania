"""TabulatedSpectrum and GenericPSD."""

from __future__ import annotations

import numpy as np
import pytest

from domainia.domains import FourierDomain
from domainia.spectra import GenericPSD, TabulatedSpectrum


def test_tabulated_spectrum_slices_by_absolute_bin_and_refuses_outside():
    dom = FourierDomain(4096, 5.0, 100, 50, "sigma0", sigma0=3.0)
    v = np.arange(100.0).reshape(2, 50)
    s = TabulatedSpectrum(dom, v)
    assert s.channels == 2
    assert np.array_equal(s.on(FourierDomain(4096, 5.0, 110, 5, "sigma0", sigma0=3.0)), v[:, 10:15])
    assert np.array_equal(s.on(domain=FourierDomain(4096, 5.0, 110, 5, "sigma0", sigma0=3.0)),
                          v[:, 10:15])
    # unit-blind: a PSD does not depend on the data's Fourier normalisation
    for units, kw in (("rfft", {}), ("dt_rfft", {}), ("sigma0", {"sigma0": 7.0})):
        assert np.array_equal(s.on(FourierDomain(4096, 5.0, 110, 5, units, **kw)), v[:, 10:15])
    assert np.array_equal(s.at([100, 149]), v[:, [0, 49]])
    assert np.array_equal(s.at([100.0]), v[:, [0]]) and s.at([]).shape == (2, 0)
    assert not np.shares_memory(s.values, v) and not s.values.flags.writeable   # a frozen copy
    with pytest.raises(ValueError, match="bins must hold integer indices, got 100.5"):
        s.at([100.5])                                                            # never truncated
    with pytest.raises(ValueError, match="outside"):
        s.on(FourierDomain(4096, 5.0, 140, 20, "sigma0", sigma0=3.0))
    with pytest.raises(ValueError, match="outside"):
        s.at([99])
    with pytest.raises(ValueError, match="record"):
        s.on(FourierDomain(8192, 5.0, 110, 5, "sigma0", sigma0=3.0))
    with pytest.raises(ValueError, match="channels"):
        TabulatedSpectrum(dom, np.zeros((2, 49)))


def test_generic_psd_binds():
    g = GenericPSD(lambda f, t, nu: nu[0] * f + t)
    assert g(2.0, 1.0, (3.0,)) == 7.0 and g.bind((3.0,))(2.0, 1.0) == 7.0
