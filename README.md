# domainia

Move data and covariance between the time, Fourier and time-frequency domains.

* `domainia.wilson` holds the Wilson-basis kernels: forward and inverse transforms, the
  stationary and evolutionary noise models, inner products and dense oracles. With the
  default Meyer window (`MEYER`) the Wilson basis is the WDM transform. Every kernel that
  samples the window takes `window=`, so the same code runs any *admissible* window: real,
  even, zero for |f| ≥ ΔF, and a partition of unity on the lattice. Such a window gives an
  orthonormal basis and block-tridiagonal stationary covariances.
  The built-in windows are `Meyer(d)` (integer `d ≥ 1`), `RampWindow(ramp, name)` (Meyer's
  geometry with a user ramp `nu` on [0, 1]) and `SampledWindow(fn, name)` (a dimensionless
  profile `psi(u)`, `u = f / ΔF`, with `psi(0) = 1`). `RampWindow` and `SampledWindow` are
  screened at construction, and every window is validated on each new grid when it is
  sampled; an inadmissible one raises `WindowError`, at construction or when sampled.
  `wilson.sample_window` is the kernels' validated entry point, also for a `Window` of your
  own.
* `domainia.translate` is the translation layer. `transform` moves `{channel: array}`
  dicts between the full-record domains.
* `domainia.domains`, `domainia.data`, `domainia.maps`, `domainia.variance` and
  `domainia.spectra` cover partial domains (WDM cell sets, Fourier bands): maps between them
  and cell variances.

## Quick start

```python
import numpy as np
from domainia import transform, wilson

Nf, Nt, dt = 16, 64, 0.5
N = Nf * Nt
rec = {"N": N, "dt": dt}
meyer8 = wilson.Meyer(8)

x = np.random.default_rng(0).standard_normal((2, N))         # two channels: any leading axes
w = transform(x, "time", "wdm", **rec, target_wdm=(Nf, Nt))       # (2, Nf + 1, Nt) float64, Meyer
X = transform(w, "wdm", "frequency", **rec, source_wdm=(Nf, Nt))  # (2, N // 2 + 1): dt * rfft(x)
w8 = transform(X, "frequency", "wdm", **rec, target_wdm=(Nf, Nt), window=meyer8)  # another window
assert np.allclose(transform(w8, "wdm", "time", **rec, source_wdm=(Nf, Nt), window=meyer8), x)
```

The WDM layout holds the `N = Nf * Nt` active cells; the inactive edge cells are 0.
`transform` uses this dense `(Nf + 1, Nt)` layout, while `domainia.data.WDMCells` and the
maps and covariances on `WDMDomain.full(grid)` use the packed `(N,)` layout (the active cells
in band order); convert with `domainia.wilson.pack(w, Nf, Nt)` and
`domainia.wilson.unpack(y, Nf, Nt)`.

`transform` is NumPy only: a JAX array is copied to host NumPy, and `transform` cannot be
traced under `jax.jit` or `jax.grad` (JAX raises its `TracerArrayConversionError`, a
`TypeError`).

### Canonical inputs

`transform`'s outputs are exactly canonical: inactive WDM cells are 0, and the Fourier DC
bin (and the Nyquist bin for even `N`) is real. Its inputs may carry a residue there of up
to 1e-10 of their peak, so raw `wilson.forward_time` output and a slice of a full FFT
(`dt * numpy.fft.fft(x)[:N // 2 + 1]`), which carry ~1e-16 round-off there on many grids,
are taken as they are. A larger residue is refused, and the error names the cell or bin:

```python
import numpy as np
from domainia import transform, wilson

Nf, Nt, dt = 6, 12, 0.3
N = Nf * Nt
x = np.random.default_rng(6).standard_normal(N)

w = wilson.forward_time(x, Nf, Nt, dt)        # raw kernel output: ~1e-17 in the inactive cells
assert np.allclose(transform(w, "wdm", "time", N=N, dt=dt, source_wdm=(Nf, Nt)), x)
X = dt * np.fft.fft(x)[: N // 2 + 1]          # a full-FFT slice: ~1e-17 imaginary at Nyquist
assert np.allclose(transform(X, "frequency", "time", N=N, dt=dt), x)
```

## Units

A WDM value is always a Wilson coefficient `w = W x` of the time record (`W W^T = dt I`),
whichever path produced it: `transform`, the maps `TimeToWDM` and `FourierToWDM` (both
implementations, and `after()`) and the `domainia.wilson` band kernels; the WDM variances
and covariances describe these coefficients. A Fourier value `z` carries the units its
`FourierDomain` declares, and `FourierDomain.rfft_scale` is the `u` with
`numpy.fft.rfft(x) = u * z`:

| `units`     | `z`                 | `u`      |
|-------------|---------------------|----------|
| `"rfft"`    | `numpy.fft.rfft(x)` | 1        |
| `"dt_rfft"` | `dt * rfft(x)`      | `1 / dt` |
| `"sigma0"`  | `rfft(x) / sigma0`  | `sigma0` |

`FourierDomain(N, dt, k0, n, "sigma0", sigma0=s)` stores the value `s` (positive and
finite; it is part of the domain's equality and hash), and the other units take no
`sigma0`. Every map honours the label: on a `WDMDomain` `dst` of interior bands (which
`FourierToWDM` requires), `FourierToWDM(src, dst).apply(z)` equals
`TimeToWDM(time, dst).apply(x)` for every `units` value, and
`FourierZeroPad` and `FourierCrop` return the rfft of the zero-padded or cropped record in
the source's units (`sigma0` included).

## Errors

Everything fails closed. `TypeError` means the wrong kind of argument: a type that can never
be valid in that place (a float or a `bool` where an integer is taken, a string where a flag
is, a `Grid` where a domain is), or an array whose dtype can never be (a complex or bool
array where real numbers are taken). `ValueError` means the right kind with a wrong value:
shape, range, finiteness, integrality, an unknown option string, inactive cells, endpoints,
or a domain or window mismatch. `WindowError` (a `ValueError`) means an inadmissible window,
complex window samples included. In `domainia.translate`, complex values on a time or WDM
domain raise `ValueError`, because they are valid on a `FourierDomain`.

## Install and test

```bash
pip install -e ".[test]"
PYTHONDONTWRITEBYTECODE=1 python -P -m pytest -q
```

Three oracle tests compare against source trees outside this repository; they skip unless
`DOMAINIA_ORACLE_ROOT` names the directory that holds them (the skip message lists the
subdirectories each one expects).
