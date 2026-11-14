# domainia

domainia does two things, in float64 NumPy. `transform` translates data arrays between the
time, frequency and WDM (Wilson-Daubechies-Meyer) domains. `frequency_covariance` and
`wdm_covariance` evaluate a K × K noise model as its covariance at each frequency bin or WDM
cell. It computes no likelihood: the caller does.

## Install and test

```bash
pip install -e ".[test]"
PYTHONDONTWRITEBYTECODE=1 python -P -m pytest -q
```

Three oracle tests compare against source trees outside this repository; they skip unless
`DOMAINIA_ORACLE_ROOT` names the directory that holds them (the skip message lists the
subdirectories each one expects).

## Data

```python
import numpy as np
from domainia import transform

Nf, Nt, dt = 6, 12, 0.3
N = Nf * Nt
x = np.random.default_rng(6).standard_normal((2, N))            # any leading axes

w = transform(x, "time", "wdm", N=N, dt=dt, target_wdm=(Nf, Nt))       # (2, 6, 12) float64
X = transform(w, "wdm", "frequency", N=N, dt=dt, source_wdm=(Nf, Nt))  # (2, 37): dt * rfft(x)
v = transform(X, "frequency", "wdm", N=N, dt=dt, target_wdm=(12, 6))   # another WDM grid
assert np.allclose(transform(v, "wdm", "time", N=N, dt=dt, source_wdm=(12, 6)), x)

# A full-FFT slice (~1e-16 imaginary at Nyquist) is taken as it is; a residue above 1e-10
# of the peak is refused.
assert np.allclose(transform(dt * np.fft.fft(x)[:, : N // 2 + 1], "frequency", "time",
                             N=N, dt=dt), x)
```

## Covariance

```python
import numpy as np
from domainia import frequency_covariance, wdm_covariance

dt, N, Nf, Nt = 5.0, 1024, 16, 64

def S(f):                                       # one-sided 2 x 2 cross-spectral density
    p = 1e-2 / (1.0 + (f / 0.02) ** 2)
    c = 0.8 * np.exp(-2j * np.pi * f * 36.5)    # coherence 0.8, a 36.5 s delay: complex
    return p * np.array([[np.ones_like(c), c], [np.conj(c), np.ones_like(c)]])

C = frequency_covariance(S, N=N, dt=dt)                                  # (2, 2, 513) complex
D = wdm_covariance(lambda f, t: S(f), Nf=Nf, Nt=Nt, dt=dt)               # (2, 2, 17, 64) real
E = wdm_covariance(lambda f, t: S(f) * (1.3 + 0.3 * np.cos(2 * np.pi * t / (N * dt))),
                   Nf=Nf, Nt=Nt, dt=dt)                                  # evolutionary
assert np.array_equal(C, C.conj().transpose(1, 0, 2))
assert np.array_equal(D, D.transpose(1, 0, 2, 3)) and not D[:, :, 0, 1].any()
```

## Conventions

- Trailing-axis layouts: time is real `(N,)`; frequency is complex `(N // 2 + 1,)`,
  `dt * rfft(x)`; WDM is real `(Nf + 1, Nt)` with the edge bands, `Nf` and `Nt` even and
  `Nf * Nt == N`. The atoms have unit norm (`sum(w**2) == dt * sum(x**2)`), and the inactive
  WDM cells (bands 0 and `Nf` at odd `n`) are 0 in every output of `transform` and
  `wdm_covariance`.
- A model `S(f)` or `S(f, t)` (`t` in seconds from the first sample) returns `(K, K, ...)`,
  or `(K, K, 1)` if it is constant, with `S[b, a] = conj(S[a, b])`, oriented so that
  `E[X_a conj(X_b)] = (N dt / 2) S_ab`.
- With `H = (S + S^H) / 2`, `frequency_covariance` is `(N dt / 2) H(f_k)`, real at DC and
  even-`N` Nyquist, and `wdm_covariance` is `½ BA_m[Re H(·, n Nf dt)]`, the band average of
  WDM Paper II, Eq. 36. Both are linear in `S`, exactly Hermitian (or symmetric), and fp64.
- For an evolutionary model Eq. 36 is a local-stationarity approximation. If the model is not
  periodic over the record, drop the edge columns `wilson.evolutionary.wrap_cut_columns` names.
- `transform`, `frequency_covariance` and `wdm_covariance` raise `TypeError` for a wrong kind
  of argument and `ValueError` for a wrong value (`WindowError` for an inadmissible window).

## Kernels

`domainia.wilson` holds the kernels: `forward`, `inverse`, `forward_time` and `inverse_time`
(any leading axes), the windows that `transform` and `wdm_covariance` take as `window=`
(`MEYER`, the default, `Meyer(d)`, `RampWindow`, `SampledWindow`), and the dense references.
