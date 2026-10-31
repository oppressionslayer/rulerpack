"""Prime counting, the forward map: prime in, index out.

Each method is its own function so they can be compared.

    primepi_sieve(x)      count by sieving to x                 O(x)
    phi(x, a)             Legendre: how many n <= x survive the first a primes
    primepi_legendre(x)   pi(x) = phi(x, a) + a - 1,  a = pi(sqrt x)
    primepi_meissel(x)    Legendre with a = pi(cbrt x) and a correction sum
    primepi(x)            the fastest exact one here (meissel)
    index_of(p)           zero-based index of prime p, = primepi(p) - 1

All exact. Pure integers plus one numpy sieve table.
"""

import sys
from math import isqrt

import numpy as np

from .primes import isprime


def _icbrt(x):
    r = int(round(x ** (1 / 3)))
    while r ** 3 > x:
        r -= 1
    while (r + 1) ** 3 <= x:
        r += 1
    return r


def _sieve_table(limit):
    """(primes, pi) with pi[n] = number of primes <= n, for n <= limit."""
    s = np.ones(limit + 1, dtype=bool)
    s[:2] = False
    for q in range(2, isqrt(limit) + 1):
        if s[q]:
            s[q * q::q] = False
    pi = np.cumsum(s, dtype=np.int64)
    return np.flatnonzero(s).astype(np.int64), pi


def primepi_sieve(x):
    """Count primes <= x by sieving. The slow, obviously-right reference."""
    x = int(x)
    if x < 2:
        return 0
    return int(_sieve_table(x)[1][x])


# Wheel for small a: phi(x, a) is periodic mod the product of the first a primes.
_WHEEL_A = 7                      # 2*3*5*7*11*13*17 = 510510
_WPRIMES = [2, 3, 5, 7, 11, 13, 17]
_WMOD = 510510
_wheel_mask = np.ones(_WMOD, dtype=bool)
for _q in _WPRIMES:
    _wheel_mask[::_q] = False
_wheel_mask[0] = False
_WHEEL = np.concatenate(([0], np.cumsum(_wheel_mask, dtype=np.int64)))
_WHEEL_COUNT = int(_WHEEL[_WMOD])       # survivors per period (96 * 1430... = 92160)


def _phi(x, a, primes):
    if a <= _WHEEL_A:
        # count n in [1, x] coprime to the first a primes
        if a == _WHEEL_A:
            return (x // _WMOD) * _WHEEL_COUNT + int(_WHEEL[x % _WMOD + 1] - _WHEEL[1]) + int(_wheel_mask[0])
        if a == 0:
            return x
        if a == 1:
            return (x + 1) // 2
        if a == 2:
            return (x // 6) * 2 + (1 if x % 6 >= 1 else 0) + (1 if x % 6 >= 5 else 0)
    raise ValueError("_phi handles a in {0, 1, 2, 7} only")


def _make_phi(primes, pi=None):
    """Legendre recursion phi(x,a) = phi(x,a-1) - phi(x//p_a, a-1) with a memo.

    With a pi table: when p_a^2 > x every survivor of the first a primes is 1 or a
    prime, so phi(x, a) = pi(x) - a + 1 is read off instead of recursed."""
    P = [int(p) for p in primes]
    memo = {}
    need = len(P) + 1000                      # recursion is one frame per prime
    if sys.getrecursionlimit() < need:
        sys.setrecursionlimit(need)
    limit = len(pi) - 1 if pi is not None else -1
    pil = pi.tolist() if pi is not None else None

    def phi(x, a):
        if a <= 2 or (a == _WHEEL_A):
            return _phi(x, a, P)
        pa = P[a - 1]
        if x < pa:
            return 1
        if x <= limit and pa * pa > x:
            return pil[x] - a + 1
        key = (x, a)
        v = memo.get(key)
        if v is None:
            v = phi(x, a - 1) - phi(x // pa, a - 1)
            memo[key] = v
        return v
    return phi


def phi(x, a):
    """Legendre's phi: how many n in [1, x] have no prime factor among the first a primes."""
    x, a = int(x), int(a)
    primes, _ = _sieve_table(max(a * (int(np.log(a + 2)) + 3), 20))
    while len(primes) < a:
        primes, _ = _sieve_table(len(primes) * 4 + 100)
    return _make_phi(primes)(x, a)


def primepi_legendre(x):
    """pi(x) = phi(x, a) + a - 1 with a = pi(sqrt x). Table only to sqrt x."""
    x = int(x)
    if x < 2:
        return 0
    r = isqrt(x)
    primes, pi = _sieve_table(r)
    a = int(pi[r])
    return _make_phi(primes, pi)(x, a) + a - 1


def primepi_meissel(x):
    """Meissel: a = pi(cbrt x), b = pi(sqrt x).
    pi(x) = phi(x, a) + a - 1 - sum_{a < i <= b} (pi(x / p_i) - i + 1).
    Needs pi() up to x^(2/3), from one sieve table."""
    x = int(x)
    if x < 2:
        return 0
    c = _icbrt(x)
    r = isqrt(x)
    primes, pi = _sieve_table(max(x // max(c, 1), r) + 1)
    a = int(pi[c])
    b = int(pi[r])
    ph = _make_phi(primes, pi)
    total = ph(x, a) + a - 1
    idx = np.arange(a + 1, b + 1, dtype=np.int64)          # 1-based i, prime p_i = primes[i-1]
    total -= int(np.sum(pi[x // primes[a:b]] - idx + 1))
    return total


primepi = primepi_meissel


def index_of(p):
    """Zero-based index of prime p in the prime list; raises if p is not prime."""
    p = int(p)
    if not isprime(p):
        raise ValueError(f"{p} is not prime")
    return primepi(p) - 1


# ── the inverse: index in, prime out ─────────────────────────────────────────

def li(x):
    """Logarithmic integral li(x), the classic estimate of pi(x). Series by Ramanujan."""
    from math import log, sqrt, exp, factorial
    x = float(x)
    if x < 2:
        return 0.0
    g = 0.5772156649015329
    lx = log(x)
    total = 0.0
    inner = 0.0
    for n in range(1, 200):
        inner += 1 / (2 * ((n - 1) // 2) + 1) if n % 2 == 1 else 0
        k = (n - 1) // 2
        s = sum(1 / (2 * j + 1) for j in range(0, k + 1))
        term = ((-1) ** (n - 1)) * lx ** n / (2 ** (n - 1) * factorial(n)) * s
        total += term
        if abs(term) < 1e-17 * abs(total) and n > 10:
            break
    return g + log(lx) + sqrt(x) * total


def li_inverse(y):
    """x with li(x) = y, by Newton on li; derivative is 1/ln x."""
    from math import log
    y = float(y)
    if y < 2:
        return 2.0
    x = y * log(y) if y > 3 else 3.0
    for _ in range(60):
        step = (li(x) - y) * log(x)
        x -= step
        if abs(step) < 0.5:
            break
    return x


_MOBIUS = [0, 1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1, 0, -1, 0, 1, 1, -1, 0,
           0, 1, 0, 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, 1, 1, 0, -1, -1, -1, 0, 0, 1, -1, 0, 0,
           0, 1, 0, -1, 0, 1, 0, 1, 1, -1, 0]


def riemann_r(x):
    """Riemann's R(x) = sum mu(n)/n * li(x^(1/n)). li minus its sqrt(x) bias; much closer to pi(x)."""
    x = float(x)
    s = 0.0
    for n in range(1, len(_MOBIUS)):
        m = _MOBIUS[n]
        if m == 0:
            continue
        v = x ** (1.0 / n)
        if v < 2:
            break
        s += m / n * li(v)
    return s


def riemann_r_inverse(y):
    """x with R(x) = y, Newton from the li inverse."""
    from math import log
    y = float(y)
    if y < 2:
        return 2.0
    x = li_inverse(y)
    for _ in range(50):
        step = (riemann_r(x) - y) * log(x)
        x -= step
        if abs(step) < 0.5:
            break
    return x


def _walk(x, k):
    """From integer x, move k primes up (k > 0) or down (k < 0). Returns the prime landed on.
    k == 0 means the largest prime <= x."""
    from .primes import allowed
    n = int(x)
    if k >= 0:
        if n % 2 == 0:
            n -= 1
        # position on the largest prime <= x first
        while not (n >= 2 and isprime(n)):
            n -= 1 if n > 3 else 0
            if n < 3:
                n = 2
                break
        for _ in range(k):
            n += 1 if n == 2 else 2
            while not isprime(n):
                n += 2
        return n
    n = _walk(x, 0)
    for _ in range(-k):
        n -= 1 if n == 3 else 2
        while not isprime(n):
            n -= 2
    return n


def nth_prime(i, report=False, guess=None):
    """The prime at zero-based index i, exact.

    guess(y) proposes x (default riemann_r_inverse; li_inverse for comparison),
    primepi pins pi(x) exactly, the walk closes the gap with isprime, up or down.
    report=True also returns (guess, pi(guess), gap walked)."""
    guess = riemann_r_inverse if guess is None else guess
    i = int(i)
    if i < 0:
        raise ValueError("index must be >= 0")
    if i < 6:
        return [2, 3, 5, 7, 11, 13][i] if not report else ([2, 3, 5, 7, 11, 13][i], None, None, 0)
    x = int(guess(i + 1))
    px = primepi(x)                   # number of primes <= x
    gap = (i + 1) - px                # how many primes past x the target is
    p = _walk(x, gap)
    return (p, x, px, gap) if report else p


def primepi_primecount(x):
    """pi(x) by Kim Walisch's primecount through its shared library, if installed
    (Ubuntu: `sudo apt install primecount`). Raises OSError when it is not."""
    import ctypes, ctypes.util
    name = ctypes.util.find_library("primecount") or "libprimecount.so.8"
    lib = ctypes.CDLL(name)
    lib.primecount_pi.restype = ctypes.c_int64
    lib.primecount_pi.argtypes = [ctypes.c_int64]
    return int(lib.primecount_pi(int(x)))


def primepi_fastest(x):
    """primecount when present, else our Meissel."""
    try:
        return primepi_primecount(x)
    except OSError:
        return primepi_meissel(x)
