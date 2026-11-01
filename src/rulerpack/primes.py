"""Primes, the seed sieve, and the sieve's survivors."""

from math import isqrt

import numpy as np

SEEDS = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
         67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131]


def seeds():
    """The 31 odd seed primes 3..131 (the first row of the topo table, halved)."""
    return list(SEEDS)


def primes_first(n):
    """First n primes as an int64 numpy array, by an odd-only sieve. No sympy."""
    if n <= 0:
        return np.array([], dtype=np.int64)
    limit = int(n * (np.log(n) + np.log(np.log(n)))) + 100 if n > 10 else 100
    sz = (limit + 1) // 2
    s = np.ones(sz, dtype=bool)
    s[0] = False
    for q in range(3, isqrt(limit) + 1, 2):
        if s[q // 2]:
            s[q * q // 2::q] = False
    return np.concatenate(([2], 2 * np.flatnonzero(s) + 1))[:n].astype(np.int64)


def primes2(n):
    """The topo array: 2 * each of the first n primes. 4, 6, 10, 14, 22, ..."""
    return primes_first(n) * 2


def allowed(h, use_sieve=True):
    """Seed rule for an odd candidate h: a seed, or no seed divides it, and not a square."""
    if not use_sieve:
        return True
    if h in SEEDS:
        return True
    if any(h % q == 0 for q in SEEDS):
        return False
    return isqrt(h) ** 2 != h


def leftovers(n):
    """First n survivors of the seed sieve, doubled, starting with 4.
    Equals primes2(n) through index 1975; first miss is 2*137*137 at index 2141."""
    out = [4]
    c = 1
    while len(out) < n:
        c += 2
        if allowed(c):
            out.append(2 * c)
    return np.array(out[:n], dtype=np.int64)


_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _isprime_py(n):
    """Deterministic Miller-Rabin: exact for every n < 3.3e24, strong probable prime above."""
    n = int(n)
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


try:
    import gmpy2 as _g

    def isprime(n):
        """gmpy2.is_prime (fast, GMP Miller-Rabin, 25 rounds). Falls back to pure Python without gmpy2."""
        return bool(_g.is_prime(int(n)))
except ImportError:          # pragma: no cover
    isprime = _isprime_py


def wheel_pass(a, count=100, seeds=None):
    """Run the seed wheel over ``a, a+1, ..., a+count-1``.

    Returns a list of ``(d, divisor)``: the offset and the first of ``[2, *seeds]`` that divides
    ``a + d``, or ``None`` when no seed does and the number survives the wheel. ``seeds`` defaults
    to ``SEEDS``; pass a shorter or longer list to see what a smaller or bigger wheel removes.
    ``a`` may be any Python int, so a Mersenne number works as well as a small anchor.
    """
    qs = [2, *(SEEDS if seeds is None else seeds)]
    return [(d, next((q for q in qs if (a + d) % q == 0), None)) for d in range(count)]


def wheel_report(a, count=100, seeds=None, show=print):
    """Print ``wheel_pass`` one line per offset: ``survives`` or ``removed — divisible by q``.

    Returns the number of survivors.
    """
    survivors = 0
    for d, q in wheel_pass(a, count, seeds):
        if q is None:
            survivors += 1
            show(f"{d:2}: survives")
        else:
            show(f"{d:2}: removed — divisible by {q}")
    return survivors
