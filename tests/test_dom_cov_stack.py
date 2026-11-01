"""ChannelStack: one covariance per channel, factors split and summed."""

from __future__ import annotations

import numpy as np
import pytest

from domainia.covariance import ChannelStack, DiagonalCovariance, StackFactor


def test_channel_stack_splits_and_sums(backend):
    covs = [DiagonalCovariance(lambda nu, be, s=s: be.xp.exp(nu[0]) * be.asarray(np.full(3, s)), 3)
            for s in (1.0, 2.0)]
    st = ChannelStack(covs, 2)
    nu = np.array([0.1, 9.0, 0.2, 9.0])
    F = st.factor(backend.asarray(nu), backend, jit=False)
    assert isinstance(F, StackFactor) and len(F.channels) == 2 and st.n == 6 and F.n_u == 0
    assert st.n_u == 0 and st.state_key(nu) is None
    want = 3 * 0.1 + 3 * (np.log(2.0) + 0.2)
    assert float(F.logdet) == pytest.approx(want, rel=1e-14)
    b = np.ones((2, 3))
    assert float(F.quad(backend.asarray(b))) == pytest.approx(3 / np.exp(0.1) + 3 / (2 * np.exp(0.2)),
                                                             rel=1e-14)
    with pytest.raises(ValueError, match="entries"):
        st.factor(np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        ChannelStack(covs, -1)
