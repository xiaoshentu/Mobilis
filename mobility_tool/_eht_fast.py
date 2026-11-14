# -*- coding: utf-8 -*-
"""Compiled intermolecular overlap block for the EHT baseline (s and p shells), used when numba is installed.

The numpy version in eht.overlap_matrix builds arrays for every pair of shell types and every Cartesian component;
for a pair of rubrene molecules that is about 20 ms per dimer, most of it in temporary arrays. Here the same
Obara–Saika expressions run as plain loops compiled by numba, and primitive pairs whose Gaussian prefactor
exp(−μR²) is below e⁻⁴⁰ (≈ 4 × 10⁻¹⁸) are skipped, which leaves S unchanged to machine precision. Without numba the
loops run as ordinary Python (only used by the tests, far too slow for production) and the numpy version is used.

Switch off with the environment variable MOBILIS_NUMBA=0."""
import math
import os

import numpy as np

SCREEN = 40.0

try:                                                      # numba is optional
    if os.environ.get("MOBILIS_NUMBA", "1") == "0":
        raise ImportError("disabled by MOBILIS_NUMBA=0")
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


def _overlap_sp(ca, la, oa, npa, aa, wa, cb, lb, ob, npb, ab, wb, out, cutoff2, screen):
    """out[oa + i, ob + j] += overlap of normalized contracted s / p shells (Cartesian p: x, y, z)."""
    pi = math.pi
    for i in range(ca.shape[0]):
        for j in range(cb.shape[0]):
            ABx = ca[i, 0] - cb[j, 0]
            ABy = ca[i, 1] - cb[j, 1]
            ABz = ca[i, 2] - cb[j, 2]
            r2 = ABx * ABx + ABy * ABy + ABz * ABz
            if r2 >= cutoff2:
                continue
            li = la[i]
            lj = lb[j]
            o1 = oa[i]
            o2 = ob[j]
            for p in range(npa[i]):
                a = aa[i, p]
                for q in range(npb[j]):
                    b = ab[j, q]
                    pp = a + b
                    e = a * b / pp * r2
                    if e > screen:
                        continue
                    base = (pi / pp) ** 1.5 * math.exp(-e) * wa[i, p] * wb[j, q]
                    if li == 0 and lj == 0:
                        out[o1, o2] += base
                        continue
                    fa = -b / pp                       # P − A = −b/(a+b) (A − B)
                    fb = a / pp                        # P − B =  a/(a+b) (A − B)
                    pa0, pa1, pa2 = fa * ABx, fa * ABy, fa * ABz
                    pb0, pb1, pb2 = fb * ABx, fb * ABy, fb * ABz
                    if li == 1 and lj == 0:
                        out[o1, o2] += base * pa0
                        out[o1 + 1, o2] += base * pa1
                        out[o1 + 2, o2] += base * pa2
                    elif li == 0 and lj == 1:
                        out[o1, o2] += base * pb0
                        out[o1, o2 + 1] += base * pb1
                        out[o1, o2 + 2] += base * pb2
                    else:
                        h = 0.5 / pp
                        out[o1, o2] += base * (pa0 * pb0 + h)
                        out[o1, o2 + 1] += base * pa0 * pb1
                        out[o1, o2 + 2] += base * pa0 * pb2
                        out[o1 + 1, o2] += base * pa1 * pb0
                        out[o1 + 1, o2 + 1] += base * (pa1 * pb1 + h)
                        out[o1 + 1, o2 + 2] += base * pa1 * pb2
                        out[o1 + 2, o2] += base * pa2 * pb0
                        out[o1 + 2, o2 + 1] += base * pa2 * pb1
                        out[o1 + 2, o2 + 2] += base * (pa2 * pb2 + h)


_overlap_sp_py = _overlap_sp
_overlap_sp_jit = njit(cache=True, fastmath=False)(_overlap_sp) if HAVE_NUMBA else None
_WEIGHTS = {}


def _pack(shells):
    """Arrays for the kernel: centres, l, offsets, number of primitives, exponents and normalized weights."""
    from .eht import _self_norm
    n = len(shells)
    mp = max(len(sh["a"]) for sh in shells)
    c = np.empty((n, 3))
    l = np.empty(n, np.int64)
    o = np.empty(n, np.int64)
    k = np.empty(n, np.int64)
    a = np.zeros((n, mp))
    w = np.zeros((n, mp))
    for i, sh in enumerate(shells):
        key = (sh["key"], sh["l"])
        if key not in _WEIGHTS:
            ex, cc, li = sh["a"], sh["c"], sh["l"]
            norm = (2 * ex / np.pi) ** 0.75 * (4 * ex) ** (li / 2)
            _WEIGHTS[key] = (np.asarray(ex, float), cc * norm / _self_norm(sh))
        ex, ww = _WEIGHTS[key]
        c[i] = sh["center"]
        l[i] = sh["l"]
        o[i] = sh["offset"]
        k[i] = len(ex)
        a[i, :len(ex)] = ex
        w[i, :len(ex)] = ww
    return c, l, o, k, a, w


def usable(shells_a, shells_b):
    return all(sh["l"] <= 1 for sh in shells_a) and all(sh["l"] <= 1 for sh in shells_b)


def overlap_block(shells_a, nbf_a, shells_b, nbf_b, cutoff_bohr=18.0, python=False):
    """Intermolecular overlap block S_AB (nbf_a × nbf_b), equal to eht.overlap_matrix(..., shells_b=...)."""
    out = np.zeros((nbf_a, nbf_b))
    A = _pack(shells_a)
    B = _pack(shells_b)
    f = _overlap_sp_py if (python or not HAVE_NUMBA) else _overlap_sp_jit
    f(A[0], A[1], A[2], A[3], A[4], A[5], B[0], B[1], B[2], B[3], B[4], B[5], out, cutoff_bohr ** 2, SCREEN)
    return out
