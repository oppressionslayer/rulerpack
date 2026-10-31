"""Ruler marks: our own way to count, with no li, no Riemann, no Meissel.

Sieve once. At every 2^k-th prime (one mark per ruler block) keep the packed
number N = pack(i, p). The table is self-describing: each entry carries its
own index and its own prime, and unpack() reads them back with mod and XOR.

From then on:
    nth_prime(i)  : take the mark at or below i, walk up (i - mark) primes
    index_of(p)   : take the mark at or below p, walk up to p counting
Walking is the seed rule plus isprime. That is the whole method.
"""

import os

import numpy as np

from .pack import Family

MARKS_FAMILY = Family(256, 2, 64)      # X = 2p ^ 256i; 64 bits covers any index that fits a word
from .primes import primes_first, isprime, allowed

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "marks.npy")


class Marks:
    def __init__(self, N, every, family=MARKS_FAMILY):
        self.N = np.asarray(N, dtype=object)
        self.every = every
        self.family = family
        pairs = [family.unpack(int(n)) for n in self.N]
        self.idx = np.array([i for i, _ in pairs], dtype=np.int64)
        self.p = np.array([p for _, p in pairs], dtype=np.int64)

    @classmethod
    def build(cls, n_primes, every=1 << 10, family=MARKS_FAMILY):
        """Sieve the first n_primes primes once and keep every `every`-th as a packed N."""
        P = primes_first(n_primes)
        i = np.arange(0, len(P), every, dtype=np.int64)
        N = [family.pack(int(k), int(P[k])) for k in i]
        return cls(N, every, family)

    def save(self, path=_DEFAULT_PATH):
        """File = header string 'every:<n>:contiguous' then one decimal N per mark."""
        np.save(path, np.array([f"every:{self.every}:contiguous"] + [str(n) for n in self.N]))
        return path

    @staticmethod
    def _read(path, family):
        """(every, sparse, N list). Accepts the headerless v0 format when it has 2+ marks."""
        s = [str(x) for x in np.load(path)]
        if s and s[0].startswith("every:"):
            _, every, kind = s[0].split(":")
            return int(every), kind == "sparse", [int(x) for x in s[1:]]
        N = [int(x) for x in s]
        if len(N) < 2:
            raise ValueError("headerless marks file with one mark: cannot infer spacing; pass every=")
        return family.unpack(N[1])[0], False, N

    @classmethod
    def load(cls, path=_DEFAULT_PATH, every=None, family=MARKS_FAMILY):
        try:
            ev, sparse, N = cls._read(path, family)
        except ValueError:
            if every is None:
                raise
            N = [int(x) for x in np.load(path)]
            ev, sparse = every, False
        if sparse:
            raise ValueError("this is a sparse LiveMarks file; load it with LiveMarks.load")
        if not N:
            raise ValueError("empty marks file")
        every = ev if every is None else every
        for k, n in enumerate(N):
            if family.unpack(n)[0] != k * every:
                raise ValueError(f"mark {k} is not at index {k * every}: not a contiguous table with every={every}")
        return cls(N, every, family)

    def __len__(self):
        return len(self.N)

    @property
    def last_index(self):
        return int(self.idx[-1]) + self.every - 1

    # ── walking ──────────────────────────────────────────────────────────
    @staticmethod
    def _next_prime(n):
        n += 1 if n == 2 else 2
        while not (allowed(n) and isprime(n)):
            n += 2
        return n

    def nth_prime(self, i):
        """Prime at zero-based index i, by walking from the mark below it."""
        i = int(i)
        if i < 0 or i > self.last_index + self.every:
            raise ValueError(f"index {i} is outside the marks (0..{self.last_index})")
        k = i // self.every
        if k >= len(self.N):
            k = len(self.N) - 1
        mi, p = self.family.unpack(int(self.N[k]))
        for _ in range(i - mi):
            p = self._next_prime(p)
        return p

    def index_of(self, p):
        """Zero-based index of prime p, by walking up from the mark below it."""
        p = int(p)
        if not isprime(p):
            raise ValueError(f"{p} is not prime")
        k = int(np.searchsorted(self.p, p, side="right")) - 1
        if k < 0:
            raise ValueError("below the first mark")
        mi, q = self.family.unpack(int(self.N[k]))
        i = mi
        while q < p:
            q = self._next_prime(q)
            i += 1
        if q != p:
            raise ValueError("walked past p; p is not prime")
        return i


class LiveMarks:
    """Marks that compute themselves. Sparse: only touched blocks exist.

    A miss costs one counting call (Riemann guess, Meissel pin, walk) and is
    stored as a packed N; every later lookup in that block is a walk. Same
    walk and same pack as Marks, with a dict instead of a vector so gaps are
    allowed. Seed it from a built Marks with `seed()` to avoid computing those."""

    def __init__(self, every=64, family=MARKS_FAMILY):
        self.every = every
        self.family = family
        self.marks = {}

    def seed(self, m):
        """Take the marks of a built table that fall on this table's grid.
        A finer source (every=16 into every=64) contributes every 4th mark; a
        coarser one (every=256 into every=64) contributes all of its marks.
        Two source marks can never land on one block."""
        used = 0
        for n in m.N:
            i, _ = self.family.unpack(int(n))
            if i % self.every == 0:
                self.marks[i // self.every] = int(n)
                used += 1
        if used == 0:
            raise ValueError(f"no mark of a table with every={m.every} lies on a grid of every={self.every}")
        return self

    def save(self, path):
        """Sparse file: header 'every:<n>:sparse' then the stored marks in index order."""
        np.save(path, np.array([f"every:{self.every}:sparse"] + [str(self.marks[k]) for k in sorted(self.marks)]))
        return path

    @classmethod
    def load(cls, path, family=MARKS_FAMILY):
        """Load a sparse or contiguous marks file into a LiveMarks."""
        every, _sparse, N = Marks._read(path, family)
        lm = cls(every, family)
        for n in N:
            i, _ = family.unpack(n)
            if i % every:
                raise ValueError(f"mark at index {i} is not on the every={every} grid")
            lm.marks[i // every] = n
        return lm

    def __len__(self):
        return len(self.marks)

    def mark(self, k):
        """(index, prime) for block k, computing and storing it if absent."""
        n = self.marks.get(k)
        if n is not None:
            return self.family.unpack(n)
        from .count import nth_prime
        i = k * self.every
        p = nth_prime(i)
        self.marks[k] = self.family.pack(i, p)
        return i, p

    def nth_prime(self, i):
        mi, p = self.mark(int(i) // self.every)
        for _ in range(int(i) - mi):
            p = Marks._next_prime(p)
        return p

    def index_of(self, p):
        if not isprime(p):
            raise ValueError(f"{p} is not prime")
        return self.pi(p) - 1

    def pi(self, x):
        """pi(x) for any x: settle on the block holding x with a few marks, then walk."""
        from math import log
        from .count import riemann_r
        x = int(x)
        if x < 2:
            return 0
        lnx = max(log(x), 1.0)
        k = int(max(riemann_r(x), 1)) // self.every
        while True:
            _, p = self.mark(k)
            if p > x:
                k = max(k - max(int((p - x) / lnx / self.every), 1), 0)
                continue
            _, pn = self.mark(k + 1)
            if pn <= x:
                k += max(int((x - pn) / lnx / self.every), 1)
                continue
            i, q = self.mark(k)
            count = i + 1
            while True:
                q = Marks._next_prime(q)
                if q > x:
                    return count
                count += 1



def _marks_pi(self, x):
    """pi(x) for any x the table covers: walk from the mark below x, counting."""
    x = int(x)
    if x < 2:
        return 0
    k = int(np.searchsorted(self.p, x, side="right")) - 1
    i, q = self.family.unpack(int(self.N[k]))
    count = i + 1
    while True:
        q = Marks._next_prime(q)
        if q > x:
            return count
        count += 1


Marks.pi = _marks_pi
