"""rulerpack: XOR rulers over the primes, and the pack that rides on them.

    >>> import rulerpack as rp
    >>> N = rp.pack(78501, 1000039)
    >>> rp.unpack(N)
    (78501, 1000039)

Families of ruler, each a Mersenne ladder scaled by a power of two:

    rp.RP(i)           (4i) ^ (4i+7)            7, 15, 7, 31, ...   the original
    rp.ruler(i, L, c)  ((Li) ^ (Li+cL+1)) ^ 1   the general form
    rp.R257(i)         L=4,   c=64              256 * ladder(i >> 6)
    rp.R513(i)         L=256, c=2               512 * ladder(i >> 1)

Pack family (L, c, M): K = 2p ^ ruler, X = 2p ^ Li, N = K*M + X.
Decode is mod, div, two XORs. No search, no primality, no table.
"""

from .primes import primes_first, primes2, seeds, allowed, leftovers, isprime
from .rulers import v2, ladder, RP, rp_xor1, ruler, R257, R513, key, topo
from .pack import (Family, F257, F513, DEFAULT, encode, decode, pack, unpack,
                   encode_array, decode_array)
from .build import build_kx, sweep
from .marks import Marks, LiveMarks
from .window import window, window_deep, residues_mod, any_prime_after, worker_pool, close_pools, ranked_window, RANK_KEYS, xor_additive, xor_additive_bits, gap_profile, squeeze_order, seeds_for, seed_depth, window_width, next_prime_after, next_prime_after_parallel, jfilter, primes_between, indexed_after, hinted_next, hint_of, hinted_family, hint_from_x, primes_from_x
from .count import primepi_sieve, phi, primepi_legendre, primepi_meissel, primepi, index_of, li, li_inverse, riemann_r, riemann_r_inverse, nth_prime, primepi_primecount, primepi_fastest

__version__ = "0.1.0"
__all__ = [
    "primes_first", "primes2", "seeds", "allowed", "leftovers", "isprime",
    "v2", "ladder", "RP", "rp_xor1", "ruler", "R257", "R513", "key", "topo",
    "Family", "F257", "F513", "DEFAULT", "encode", "decode", "pack", "unpack",
    "encode_array", "decode_array",
    "build_kx", "sweep", "Marks", "LiveMarks", "window", "window_deep", "residues_mod", "any_prime_after", "worker_pool", "close_pools", "ranked_window", "RANK_KEYS", "xor_additive", "xor_additive_bits", "gap_profile", "squeeze_order", "seeds_for", "seed_depth", "window_width", "next_prime_after", "next_prime_after_parallel", "jfilter", "primes_between", "indexed_after", "hinted_next", "hint_of", "hinted_family", "hint_from_x", "primes_from_x",
    "primepi_sieve", "phi", "primepi_legendre", "primepi_meissel", "primepi", "index_of", "li", "li_inverse", "riemann_r", "riemann_r_inverse", "nth_prime", "primepi_primecount", "primepi_fastest",
]
