"""rulerpack.experiments: things that were fun in the session and might be more.

Nothing here is used by the rest of the library. Each function reproduces an IPython moment
exactly and names what it saw, so it can be picked up again later.

    import rulerpack.experiments as rx

Experiment 1, the step-10 index (In [475]-[478]):

    idx = rx.stride_index(primes)            # np.arange(2, len(primes), 10)
    rx.stripe(primes)                        # (idx ^ 2p) % 4 : 2, 0, 2, 0, ... after the first
    rx.jumps(primes)                         # where idx ^ 2p leaps by more than 300
    rx.block_crossings(primes)               # where 2p crosses a multiple of 256
    rx.last_before_blocks(primes)            # 127, 251, 383, 509, ... the last prime under each 128k
"""

import numpy as np


def stride_index(primes, start=2, step=10):
    """`np.arange(start, len(primes), step)`: every tenth index from the third prime."""
    return np.arange(start, len(primes), step, dtype=np.int64)


def picked(primes, start=2, step=10):
    """The primes the stride index picks, `primes[stride_index(primes)]`."""
    primes = np.asarray(primes, dtype=np.int64)
    return primes[stride_index(primes, start, step)]


def xor_index(primes, start=2, step=10):
    """`idx ^ (2p)` over the first len(idx) primes: the array of In [477].

    XOR cancels the top bits the two sides share, so the values sit small and leap only
    when one side crosses a power of two or a multiple of 256."""
    primes = np.asarray(primes, dtype=np.int64)
    idx = stride_index(primes, start, step)
    return idx ^ (primes[: len(idx)] * 2)


def stripe(primes, start=2, step=10):
    """`(idx ^ 2p) % 4`: 2, then 0, 2, 0, 2, ... forever.

    Bit 1 of idx alternates because the step is 2 mod 4; 2p is 2 mod 4 for every odd p; the
    one exception is the first entry, where p = 2 and 2p = 4. That is the whole stripe."""
    return xor_index(primes, start, step) % 4


def jumps(primes, start=2, step=10, more_than=300):
    """Indices of the entries where `idx ^ 2p` has just leapt by more than `more_than`.

    Two kinds: 2p crossing a power of two (1024, 2048, 4096, 8192 at indices 97, 172, 309,
    564 on the first 1000 primes) and idx crossing a multiple of 256, every 25.6 entries."""
    x = xor_index(primes, start, step)
    return np.flatnonzero(np.abs(np.diff(x)) > more_than) + 1


def block_crossings(primes, block=256):
    """`np.flatnonzero(np.diff((2p) >> 8))`: the indices where 2p steps into the next block.

    With block 256 that is p crossing a multiple of 128. The spacing is the count of primes
    in each 128-wide block, which thins as the prime number theorem says."""
    primes = np.asarray(primes, dtype=np.int64)
    return np.flatnonzero(np.diff((primes * 2) // block))


def last_before_blocks(primes, width=128):
    """The last prime under each multiple of `width`: 127, 251, 383, 509, 631, ...

    The largest primes under powers of two show up here on the dot: 127, 1021, 2039, 4093,
    and 8191 when the list reaches it."""
    primes = np.asarray(primes, dtype=np.int64)
    return primes[block_crossings(primes, 2 * width)]


def gaps_to_blocks(primes, width=128):
    """How far each `last_before_blocks` prime sits under its boundary. Always odd."""
    last = last_before_blocks(primes, width)
    return width * (np.arange(len(last)) + 1) - last


# ── experiment 1, continued: the layers under the stripe (In [481]) ───────────

def layers(primes, moduli=(4, 8, 16, 32), start=2, step=10, n=40):
    """`z % m` for each modulus, z = idx ^ 2p: one more bit of the primes per row.

    (a ^ b) % m == (a % m) ^ (b % m) for a power of two, so each row is the stripe with one
    more layer on top, and the layers underneath never change. At % 8 the new bit is whether
    the prime is 1 or 3 mod 4; at % 16 its residue mod 8; and so on."""
    z = xor_index(primes, start, step)[:n]
    return {m: z % m for m in moduli}


def bit_layers(primes, bits=(1, 2, 3, 4), start=2, step=10, n=40):
    """`(z >> bit) & 1` for each bit: the layers on their own."""
    z = xor_index(primes, start, step)[:n]
    return {b: (z >> b) & 1 for b in bits}


def recover(primes, m=32, start=2, step=10):
    """Peel the index back off a layer: `((z % m) ^ (idx % m)) >> 1 == p % (m // 2)`.

    XOR is its own inverse, so no layer hides anything: the `% 32` row carries the primes'
    residues mod 16, reversibly mixed with the index. The same fact as `rp.unslide512`."""
    primes = np.asarray(primes, dtype=np.int64)
    idx = stride_index(primes, start, step)
    z = idx ^ (primes[: len(idx)] * 2)
    return ((z % m) ^ (idx % m)) >> 1


def side_by_side(primes, m=32, start=2, step=10, n=40):
    """Columns `idx % m`, `2p % m`, `z % m`: where each repeated value in a layer comes from.

    The runs of 28, 14, 28, 14 are the index stepping through 2, 12, 22, 32 mod 32 against
    primes that happen to sit in the same residue class; read the rows and the run is traced."""
    primes = np.asarray(primes, dtype=np.int64)
    idx = stride_index(primes, start, step)[:n]
    p2 = primes[:n] * 2
    return np.column_stack((idx % m, p2 % m, (idx ^ p2) % m))


def stripe_control(n=1000, start=2, step=10):
    """The stripe over odd composites instead of primes: 0, 2, 0, 2, ... with no exception.

    The primes' version opens with an extra 2 because the first prime is 2; the composites
    have no even member, so the alternation starts clean.

    The stripe is a fact about bit 1 of an index stepping by 10 and of twice an odd number.
    Any odd sequence gives it, which is the control that says it is not about primes."""
    odds = np.arange(9, 9 + 2 * n, 2, dtype=np.int64)
    comps = odds[np.array([any(o % q == 0 for q in (3, 5, 7)) for o in odds])][:n]
    idx = np.arange(start, len(comps) * step, step, dtype=np.int64)[: len(comps)]
    return (idx ^ (comps[: len(idx)] * 2)) % 4
