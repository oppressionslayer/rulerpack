"""Sequential builders and the window sweep."""

import numpy as np

from .primes import allowed
from .rulers import RP


def build_kx(n):
    """Build (L, K, X) step by step from the seed rule, with RP and 4i.
    K = L ^ RP ^ 1, X = L ^ 4i. Asserts K ^ X == RP ^ 1 ^ 4i at every step."""
    L, K, X = [], [], []
    h = 1
    for i in range(n):
        if i == 0:
            c = 4
        else:
            while True:
                h += 2
                if allowed(h):
                    break
            c = 2 * h
        r = RP(i)
        ar = 4 * i
        k = c ^ r ^ 1
        x = c ^ ar
        assert k ^ x == r ^ 1 ^ ar
        L.append(c); K.append(k); X.append(x)
    return (np.array(L, dtype=np.int64), np.array(K, dtype=np.int64),
            np.array(X, dtype=np.int64))


def sweep(P, step=None, use_sieve=True, prime_test=None):
    """Walk odd candidates past the seed rule. With step, also require
    h % step == P[i] % step (lookup mode). With prime_test (e.g. rp.isprime),
    a candidate that fails it is rejected ("nah, next one") and counted.

    Returns (index, built, true, rejected) at the first wrong-index hit,
    or (None, rejected) if every prime in P is reproduced."""
    h = 1
    rejected = 0
    for i in range(1, len(P)):
        true = int(P[i])
        while True:
            h += 2
            if not allowed(h, use_sieve):
                continue
            if step and h % step != true % step:
                continue
            if prime_test is not None and not prime_test(h):
                rejected += 1
                continue
            break
        if h != true:
            return i, h, true, rejected
    return None, rejected
