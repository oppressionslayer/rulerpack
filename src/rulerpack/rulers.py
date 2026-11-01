"""The rulers. Every one is a Mersenne ladder m ^ (m+1) scaled by a power of two.

All functions accept an int or a numpy int64 array for i.
"""

import numpy as np


def v2(n):
    """2-adic valuation: trailing zero bits of n (int or array)."""
    n = np.asarray(n, dtype=np.int64)
    out = np.log2((n & -n).astype(np.float64)).astype(np.int64)
    return int(out) if out.ndim == 0 else out


def ladder(m):
    """m ^ (m+1) == 2^(v2(m+1)+1) - 1: 1, 3, 1, 7, 1, 3, 1, 15, ..."""
    return m ^ (m + 1)


def RP(i):
    """The original ruler (4i) ^ (4i+7) == 2^(v2(i+1)+3) - 1: 7, 15, 7, 31, 7, 15, 7, 63, ..."""
    return (4 * i) ^ (4 * i + 7)


def rp_xor1(i):
    """RP ^ 1: 6, 14, 6, 30, ... (clears the low bit)."""
    return RP(i) ^ 1


def ruler(i, L, c):
    """General ruler ((Li) ^ (Li + cL + 1)) ^ 1 == L * (i ^ (i + c)).
    With L a power of two and c a power of two, this is L*c * ladder(i // c)."""
    return ((L * i) ^ (L * i + c * L + 1)) ^ 1


def R257(i):
    """((4i) ^ (4i+257)) ^ 1 == 256 * ladder(i >> 6): blocks of 64."""
    return ruler(i, 4, 64)


def R513(i):
    """((256i) ^ (256i+513)) ^ 1 == 512 * ladder(i >> 1): blocks of 2."""
    return ruler(i, 256, 2)


def key(n):
    """KEY[i] = 2*primes[i] ^ (RP[i] ^ 1): 2, 8, 12, 16, 16, 20, 36, ...
    Inverse: primes2 == key ^ (RP ^ 1)."""
    from .primes import primes2
    return primes2(n) ^ rp_xor1(np.arange(n, dtype=np.int64))


def topo(n):
    """Dict of the session's sequences, each length n."""
    from .primes import primes2
    i = np.arange(n, dtype=np.int64)
    p2 = primes2(n)
    r = RP(i)
    return {"primes2": p2, "rp": r, "rp_xor1": r ^ 1, "key": p2 ^ (r ^ 1),
            "primes2_mod6": p2 % 6, "primes2_mod8": p2 % 8}


# ── the 512 slide (Lars Rocha, session 6696-6703) ─────────────────────────────

def R896(i):
    """The two-arange ruler: (896 + 512i) ^ (512i) == 384 + 512 * ladder(i).

    896, 1920, 896, 3968, 896, 1920, 896, 8064, ... — 896 half the time, the spike at every
    power of two. It is the ladder carried up to bit 9 and offset by the constant 384."""
    return (896 + 512 * i) ^ (512 * i)


def carry(i):
    """The ruler's carry at i: ladder(i) ^ ladder(i+1). Only ever 2, 6, 14, 30, ..., in pairs:
    2, 6, 6, 2, 2, 14, 14, 2, 2, 6, 6, 2, 2, 30, 30, 2, ..."""
    return ladder(i) ^ ladder(i + 1)


def slide512(p, i=None):
    """**The 512 slide**: ((p * 512) ^ 512) ^ R896(i), the array of In [6697].

    `i` defaults to 0, 1, 2, ... along p. Every entry is 384 + 512 * ((p ^ 1) ^ ladder(i)), so
    it is 384 mod 512, 128 mod 256 and 0 mod 128 for any p whatever; the primes live from bit 9
    up, and `unslide512` reads them straight back."""
    p = np.asarray(p, dtype=np.int64)
    if i is None:
        i = np.arange(p.size, dtype=np.int64).reshape(p.shape)
    return ((p * 512) ^ 512) ^ R896(i)


def unslide512(b, i):
    """p back from a slide entry and its index: (((b - 384) // 512) ^ ladder(i)) ^ 1."""
    b = np.asarray(b, dtype=np.int64)
    return (((b - 384) // 512) ^ ladder(np.asarray(i, dtype=np.int64))) ^ 1


def slide_repeats(p):
    """Indices i where the slide repeats, slide[i] == slide[i+1].

    That happens exactly when p[i] ^ p[i+1] == carry(i): the gap's XOR agrees with the ruler's
    carry. Two thirds of them are twin primes, most of the rest gaps of six. The carry is
    independent of the gap, so this is a lamp on the sequence, not a sieve of it."""
    p = np.asarray(p, dtype=np.int64)
    i = np.arange(p.size - 1, dtype=np.int64)
    return np.flatnonzero((p[:-1] ^ p[1:]) == carry(i))
