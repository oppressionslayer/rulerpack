"""The (L, c, M) pack family.

    K = 2p ^ ruler(i, L, c)
    X = 2p ^ L*i
    K ^ X = L*(i + c)          for every i and p
    N = K*M + X                M = 2^shift, fixed per family; needs X < M

Decode: X = N mod M, K = N div M, i = (K^X)/L - c, p = (X ^ Li)/2.
M is a parameter like L and c. It is never stored in N and never guessed back;
guessing it back is a search, and that search collides.
"""

from dataclasses import dataclass

import numpy as np

from .rulers import ruler


@dataclass(frozen=True)
class Family:
    L: int = 256
    c: int = 2
    shift: int = 32

    def __post_init__(self):
        if self.L < 4 or self.L & (self.L - 1):
            raise ValueError("L must be a power of two and at least 4: X = 2p ^ L*i needs the "
                             "index above bit 1 so it never touches the bits of 2p that decode p")
        if self.shift < 1:
            raise ValueError("shift must be at least 1")

    @property
    def M(self):
        return 1 << self.shift

    @property
    def lbits(self):
        return self.L.bit_length() - 1

    def ruler(self, i):
        return ruler(i, self.L, self.c)

    def encode(self, i, p):
        """(K, X) for prime p at index i."""
        return (2 * p) ^ self.ruler(i), (2 * p) ^ (self.L * i)

    def decode(self, K, X):
        """(i, p) from the pair. Pure integer ops."""
        i = ((K ^ X) >> self.lbits) - self.c
        p = (X ^ (self.L * i)) >> 1
        return i, p

    def pack(self, i, p):
        K, X = self.encode(i, p)
        if X >> self.shift:
            raise ValueError(f"X={X} needs {X.bit_length()} bits; M=2^{self.shift} too small")
        return (K << self.shift) | X

    def unpack(self, N):
        return self.decode(N >> self.shift, N & (self.M - 1))

    def encode_array(self, P, i=None):
        """Vectorised (K, X) for a prime array P (indices default to 0..n-1)."""
        P = np.asarray(P, dtype=np.int64)
        i = np.arange(len(P), dtype=np.int64) if i is None else np.asarray(i, dtype=np.int64)
        return (2 * P) ^ self.ruler(i), (2 * P) ^ (self.L * i)

    def decode_array(self, K, X):
        K = np.asarray(K, dtype=np.int64)
        X = np.asarray(X, dtype=np.int64)
        i = ((K ^ X) >> self.lbits) - self.c
        return i, (X ^ (self.L * i)) >> 1

    def check(self, n):
        """Round-trip the first n primes through encode/decode. Returns (ok, last_prime)."""
        from .primes import primes_first
        P = primes_first(n)
        K, X = self.encode_array(P)
        i, p = self.decode_array(K, X)
        ok = bool(np.array_equal(i, np.arange(len(P)))) and bool(np.array_equal(p, P))
        return ok, int(P[-1])


F513 = Family(256, 2, 32)      # pack256.py's ruler
F257 = Family(4, 64, 32)       # pack.py's ruler
DEFAULT = F513


def encode(i, p, family=DEFAULT):
    return family.encode(i, p)


def decode(K, X, family=DEFAULT):
    return family.decode(K, X)


def pack(i, p, family=DEFAULT):
    return family.pack(i, p)


def unpack(N, family=DEFAULT):
    return family.unpack(N)


def encode_array(P, i=None, family=DEFAULT):
    return family.encode_array(P, i)


def decode_array(K, X, family=DEFAULT):
    return family.decode_array(K, X)
