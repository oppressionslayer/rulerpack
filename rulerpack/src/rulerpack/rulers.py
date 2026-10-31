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
