"""The anchored window: sieve at ott, not one candidate at a time.

Pick an anchor (10^12, 2^44, anything). Every candidate is anchor + d, so its
residue mod a seed q is (anchor mod q) + d. Compute anchor mod q once per seed,
then clear the window one pass per seed: the first d that q divides is
q - (anchor mod q), and every q after. No divisions per candidate. Every prime
survives; near 10^12 the 3..541 seeds clear about 82% of the odd numbers.

    survivors = window(10**12, 4096)         # odd offsets d with no seed factor
    next_prime_after(10**12)                 # the closest survivor that certifies

The j-filter from the session (residues of the stepping counter j^1 rather than
of the candidate) is kept as `jfilter=` so it can be compared; it is a choice,
not a sieve: it removes primes too.
"""

import numpy as np

from .primes import SEEDS, isprime


def window(anchor, width, seeds=None, odd_only=True):
    """Offsets d in [0, width) such that anchor + d has no factor among seeds.
    Returns a numpy int64 array of offsets, ascending (closest to anchor first)."""
    anchor = int(anchor)
    seeds = SEEDS if seeds is None else list(seeds)
    alive = np.ones(width, dtype=bool)
    if odd_only:
        alive[(anchor % 2)::2] = False          # even candidates out
    for q in seeds:
        first = (-anchor) % q                    # first d with q | anchor + d
        alive[first::q] = False
        if anchor <= q < anchor + width:         # the seed itself is prime, keep it
            alive[q - anchor] = True         # (at anchor 0 it is the second stride, not the first)
    return np.flatnonzero(alive).astype(np.int64)


_SEED_CACHE = {}

EXACT_LIMIT = 1 << 50          # seed to sqrt(anchor): every survivor is prime, no test needed
MAX_GAP_BELOW_2_64 = 1550      # largest prime gap below 2^64 (exhaustively verified)


def seed_depth(anchor, width=0, exact=False):
    """How deep to seed for an anchor of this size. Measured on next_prime_after
    (single call, gmpy2 certification): deeper seeds only pay once one
    certification costs more than the extra strike-outs.

    * through 512 bits: None (the 31 default seeds; certification is microseconds)
    * through 1536 bits: 8191
    * above: min(1_000_000, 125 * bits); at 8191 bits a certification is ~100 ms
      and a million seeds still removes 15% of what 131071 leaves.

    exact=True seeds to sqrt(anchor + width) instead, so every survivor is prime
    and no test is needed. Right for wide windows (primes_between does this);
    for one next prime below 2^50 it costs more than the test it saves."""
    hi = int(anchor) + int(width)
    if exact:
        return None if hi < 131 * 131 else int(hi ** 0.5) + 1
    bits = hi.bit_length()
    if bits <= 512:
        return None
    if bits <= 1536:
        return 8191
    return min(1_000_000, 125 * bits)


def seeds_for(anchor, width=0, exact=False):
    """Seed list for an anchor (odd primes up to seed_depth). None means use SEEDS."""
    depth = seed_depth(anchor, width, exact)
    if depth is None:
        return None
    if depth not in _SEED_CACHE:
        if len(_SEED_CACHE) > 8:
            _SEED_CACHE.clear()
        _SEED_CACHE[depth] = _primes_upto(depth)[1:].tolist()
    return _SEED_CACHE[depth]


def _seeds_for(bits):
    """Old entry point (by bit length); kept for callers that used it."""
    return seeds_for((1 << bits) - 1)


def window_width(anchor):
    """First-window width: largest known gap plus slack below 2^64 (the next
    prime is then always inside), 7 * ln(anchor) above (misses about one
    anchor in a thousand; the caller slides on a miss)."""
    anchor = int(anchor)
    if anchor < 1 << 64:
        return MAX_GAP_BELOW_2_64 + 50
    return 7 * anchor.bit_length() * 693 // 1000 + 1


def _certify(c):
    """Cheap base-2 strong probable prime test first, then the full one."""
    try:
        import gmpy2
        return gmpy2.is_strong_prp(c, 2) and gmpy2.is_prime(c)
    except ImportError:
        return isprime(c)


def next_prime_after(anchor, width=None, seeds=None, exact=False, key=None):
    """Closest prime above anchor: sieve the window, certify survivors in order.
    Any size. The window is the largest known gap plus slack below 2^64, so the
    answer is in the first window; above that 7 ln(anchor), sliding on a miss.
    exact=True (anchor below 2^50) seeds to sqrt and returns the first survivor
    with no primality test.
    key=None tests survivors lowest offset first. key="blocks" (or any RANK_KEYS
    name / callable) tests them in that order and stops as soon as the lowest
    prime seen has every survivor below it cleared."""
    anchor = int(anchor)
    if anchor < 2:
        return 2
    if width is None:
        width = window_width(anchor)
    if exact and anchor + 2 * width >= EXACT_LIMIT:
        raise ValueError("exact windows need anchor + width below 2^50")
    if seeds is None:
        seeds = seeds_for(anchor, width, exact)
    f = None if key is None else (RANK_KEYS[key] if isinstance(key, str) else key)
    base = anchor + 1
    while True:
        w = window(base, width, seeds)
        if exact and len(w):
            return base + int(w[0])
        if f is None:
            for d in w:
                c = base + int(d)
                if _certify(c):
                    return c
        else:
            offs = [int(d) for d in w]
            order = sorted(range(len(offs)), key=lambda k: (f(offs[k]), offs[k]))
            status = [None] * len(offs)
            best = None
            for k in order:
                status[k] = _certify(base + offs[k])
                if status[k] and (best is None or k < best):
                    best = k
                if best is not None and all(status[i] is False for i in range(best)):
                    return base + offs[best]
        base += width


def jfilter(ot, anchor, steps, seeds=None):
    """The session's loop: j = (ot ^ anchor) - 2, step -4, candidate = j ^ ot ^ 1.
    Returns rows (j, candidate, j_clean, candidate_clean, prime) so the two
    sieves can be compared side by side."""
    seeds = SEEDS if seeds is None else list(seeds)
    j = (ot ^ anchor) - 2
    out = []
    for _ in range(steps):
        cand = j ^ ot ^ 1
        jc = not any((j ^ 1) % q == 0 for q in seeds)
        cc = not any(cand % q == 0 for q in seeds)
        out.append((j, cand, jc, cc, isprime(cand)))
        j -= 4
    return out


# ── the full segmented sieve: start anywhere, bring the primes to sqrt(hi) ──

_UPTO_CACHE = {}


def _primes_upto(n):
    """Primes <= n by a plain sieve (used for the seed list). Cached by n."""
    n = int(n)
    if n < 2:
        return np.array([], dtype=np.int64)
    if n in _UPTO_CACHE:
        return _UPTO_CACHE[n]
    r = _primes_upto_uncached(n)
    if len(_UPTO_CACHE) > 8:
        _UPTO_CACHE.clear()
    _UPTO_CACHE[n] = r
    return r


def _primes_upto_uncached(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for q in range(2, int(n ** 0.5) + 1):
        if s[q]:
            s[q * q::q] = False
    return np.flatnonzero(s).astype(np.int64)


def primes_between(lo, hi, segment=1 << 22):
    """Every prime p with lo <= p < hi, as an int64 array. No primality test:
    the seeds are all primes up to sqrt(hi), so every survivor is prime.

    Sieve of Eratosthenes started at lo. Works in segments so a window of a
    billion at 10^18 costs memory of one segment, not of the window."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return np.array([], dtype=np.int64)
    seeds = _primes_upto(int(hi ** 0.5) + 1)
    out = []
    base = lo
    while base < hi:
        width = min(segment, hi - base)
        alive = np.ones(width, dtype=bool)
        # a seed q strikes from max(q*q, first multiple >= base); q itself survives
        first = (-base) % seeds
        first = np.where(seeds * seeds >= base, seeds * seeds - base, first)
        small = seeds <= width          # these hit many times: loop
        for q, f in zip(seeds[small].tolist(), first[small].tolist()):
            if f < width:
                alive[f::q] = False
        big = ~small                    # these hit at most once per segment: one vector op
        hits = first[big]
        alive[hits[hits < width]] = False
        found = np.flatnonzero(alive).astype(np.int64) + base
        out.append(found[found >= 2])
        base += width
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def indexed_after(anchor, n, count=None, family=None):
    """The next n primes above anchor, each with its exact zero-based index and
    its packed N. One count at the anchor (or pass `count` = pi(anchor) from
    anywhere, e.g. primecount), then the window sieve supplies the primes in
    order and the index increments.

    Returns a list of (index, prime, N)."""
    from .marks import MARKS_FAMILY
    from .count import primepi_fastest
    anchor = int(anchor)
    family = MARKS_FAMILY if family is None else family
    if count is None:
        count = primepi_fastest(anchor)      # primecount if installed, else Meissel
    out = []
    from math import log
    lo = anchor + 1
    width = max(4096, int(1.5 * n * log(anchor + 2)))   # about n primes per window
    idx = int(count)                         # zero-based index of the next prime
    while len(out) < n:
        for p in primes_between(lo, lo + width):
            out.append((idx, int(p), family.pack(idx, int(p))))
            idx += 1
            if len(out) == n:
                break
        lo += width
    return out


def _prp(c):
    import gmpy2
    return gmpy2.is_strong_prp(c, 2) and gmpy2.is_prime(c)


def _prp_only(c):
    """The sort as the answer (Lars Rocha): strong probable-prime test on bases
    2 and 3 and nothing more. Every prime passes. For a random candidate that
    has survived the sieve the chance a composite passes both is far below
    2^-100 at these sizes (Damgard-Landrock-Pomerance bounds), so this is the
    'turn the full test off' option. Not for adversarial inputs: Carmichael-
    style numbers built to fool fixed bases exist, and is_prime catches them."""
    import gmpy2
    return gmpy2.is_strong_prp(c, 2) and gmpy2.is_strong_prp(c, 3)


def _first_prp_only(cands):
    for c in cands:
        if _prp_only(c):
            return c
    return None


_POOL = {}


def worker_pool(workers=None):
    """A process pool kept alive between calls (spinning up 16 processes that
    import numpy costs about a second at every call otherwise). One pool per
    worker count; shut down with close_pools()."""
    from concurrent.futures import ProcessPoolExecutor
    import os
    workers = workers or os.cpu_count() or 1
    if workers not in _POOL:
        _POOL[workers] = ProcessPoolExecutor(workers)
    return _POOL[workers]


def close_pools():
    for pool in _POOL.values():
        pool.shutdown(wait=False, cancel_futures=True)
    _POOL.clear()


def squeeze_order(n):
    """Index order that works both ends of a window toward the centre:
    0, n-1, 1, n-2, 2, n-3, ... (Lars Rocha's squeeze)."""
    out = []
    lo, hi = 0, n - 1
    while lo <= hi:
        out.append(lo)
        if hi != lo:
            out.append(hi)
        lo += 1
        hi -= 1
    return out


def next_prime_after_parallel(anchor, workers=None, width=None, seeds=None, order="front"):
    """next_prime_after with the survivors certified across all cores.
    The sieve gives the whole candidate list at once, and the tests are
    independent, so they run in parallel and the smallest pass is returned.
    Worth it above roughly 1024 bits; below that the single-core version wins.

    order="front"   hand out the lowest untested survivors first (default).
    order="squeeze" alternate front and back toward the centre. A prime found
                    from the back is an upper bound; the search stops as soon as
                    every survivor below the lowest prime seen is cleared."""
    from concurrent.futures import ProcessPoolExecutor
    import os
    anchor = int(anchor)
    if anchor < 2:
        return 2
    if width is None:
        width = window_width(anchor)
    if seeds is None:
        seeds = seeds_for(anchor, width)
    workers = workers or os.cpu_count() or 1
    base = anchor + 1
    pool = worker_pool(workers)
    if True:
        while True:
            cands = [base + int(d) for d in window(base, width, seeds)]
            n = len(cands)
            idx = squeeze_order(n) if order == "squeeze" else list(range(n))
            status = [None] * n                       # None untested, True/False
            best = None                               # lowest prime seen so far
            for s in range(0, n, 4 * workers):
                chunk = idx[s:s + 4 * workers]
                for i, ok in zip(chunk, pool.map(_prp, [cands[i] for i in chunk])):
                    status[i] = ok
                    if ok and (best is None or i < best):
                        best = i
                if best is not None and all(status[i] is False for i in range(best)):
                    return cands[best]
            base += width


def hinted_next(anchor, hint, bits=17):
    """The next prime above anchor given only its low `bits` bits, with no
    primality test: the first number above anchor that matches the hint.

    Correct whenever the prime gap at that size is below 2^bits. Proven for
    bits >= 11 below 2^64 (largest gap 1,550); heuristic above: 13 bits to
    about 1e39, 17 bits to about 2^256 (Cramer, with headroom).

        p = next_prime_after(2**256)           # certified once
        hinted_next(2**256, p % 2**17)          # p again, zero tests"""
    anchor, hint = int(anchor), int(hint)
    m = 1 << bits
    c = anchor + 1
    c += (hint - c) % m                        # first c > anchor with c ≡ hint (mod 2^bits)
    return c


def hint_of(p, bits=17):
    """The low `bits` bits of p: what hinted_next needs."""
    return int(p) & ((1 << bits) - 1)


# ── hinted packs: the hint lives inside X ───────────────────────────────────

def hinted_family(bits=17, c=2, shift=None):
    """A pack family whose X field exposes `bits` clean low bits of p.

    X = 2p ^ L*i with L = 2^(bits+1), so the bottom bits+1 bits of X are the
    bottom bits+1 bits of 2p, untouched by the index. K ^ X = L(i+c) still
    holds, so the pack opens exactly as before. shift defaults to 64; raise it
    for primes past 2^63 (X needs about bits(p)+1 bits, or bits(L*i) if larger)."""
    from .pack import Family
    L = 1 << (bits + 1)
    return Family(L, c, 64 if shift is None else shift)


def hint_from_x(X, bits=17):
    """p mod 2^bits read straight off a hinted pack's X."""
    return (int(X) & ((1 << (bits + 1)) - 1)) >> 1


def primes_from_x(start, Xs, bits=17):
    """Walk the primes from `start` using only a hinted pack's X values.
    No K, no index, no count, no primality test: one modulo per prime."""
    p = int(start)
    out = []
    for x in Xs:
        p = hinted_next(p, hint_from_x(x, bits), bits)
        out.append(p)
    return out


# ── gap accounting per range (Lars Rocha): the ruler of a realm is ln x ──

def gap_profile(anchor, n=100_000):
    """Account for the gaps among the first n primes above anchor.

    Returns a dict:
        gaps      int64 array of the n-1 consecutive gaps
        hist      int64 array, hist[g] = how many gaps of size g (index = gap)
        twins     count of gap-2 pairs, and twin_share = twins / (n-1)
        mean, max the mean and largest gap seen
        ln        ln(anchor): the predicted mean gap in this realm
        twin_pred 1.32 / ln(anchor): the Hardy-Littlewood twin share
        over_ln   fraction of gaps longer than ln(anchor) (about 0.37 everywhere)
    Exact below 2^50 or so (sieve to sqrt); above, the primes come from
    next_prime_after so n should be small."""
    import math
    anchor = int(anchor)
    if anchor < 1 << 50:
        est = int(n * math.log(anchor + n) * 1.2) + 1000
        P = primes_between(anchor, anchor + est)
        while len(P) < n:
            est *= 2
            P = primes_between(anchor, anchor + est)
        P = P[:n].astype(object) if anchor + est >= 1 << 63 else P[:n]
        gaps = np.diff(P).astype(np.int64)
    else:
        ps = [next_prime_after(anchor)]
        for _ in range(n - 1):
            ps.append(next_prime_after(ps[-1]))
        gaps = np.array([b - a for a, b in zip(ps, ps[1:])], dtype=np.int64)
    L = math.log(anchor)
    hist = np.bincount(gaps)
    twins = int(hist[2]) if len(hist) > 2 else 0
    return dict(gaps=gaps, hist=hist, twins=twins, twin_share=twins / len(gaps),
                mean=float(gaps.mean()), max=int(gaps.max()), ln=L,
                twin_pred=1.32 / L, over_ln=float((gaps > L).mean()))


# ── sorting a window (Lars Rocha): put the next prime at the top ──

def _fermat2(c):
    """Two modular powers, bases 2 and 3. True for every prime, false for almost
    every composite. Base 2 alone lets the Fermat numbers 2^(2^k)+1 through,
    which is exactly what sits at offset 0 of a 2^256 window; base 3 catches them."""
    try:
        import gmpy2
        return gmpy2.powmod(2, c - 1, c) == 1 and gmpy2.powmod(3, c - 1, c) == 1
    except ImportError:
        return pow(2, c - 1, c) == 1 and pow(3, c - 1, c) == 1


def _farkey(d):
    """Lars Rocha's 'farthest prime key': the key value (d >> 3) ^ 16, candidates
    whose key is itself a prime number first, largest key first, then the rest."""
    k = (d >> 3) ^ 16
    return (0 if isprime(k) else 1, -k)


RANK_KEYS = {
    "offset": lambda d: d,                 # lowest offset first (the gap distribution favours it)
    "blocks": lambda d: (d >> 4) ^ 8,      # Lars Rocha's key: 16-wide blocks, pairs of 8 swapped
    "blocks16": lambda d: d >> 4,          # plain 16-wide blocks
    "farkey": _farkey,                     # prime key values first, farthest out first
}


def ranked_window(anchor, width=None, seeds=None, key="offset", power=True):
    """Survivors of the window sorted so the next prime comes first.

    Three stages, each one cheap for what it removes:
      1. sieve: window() strikes every candidate with a small factor
      2. order: sort what is left by `key` on the offset ("offset", "blocks",
         "blocks16", or a callable d -> number). Measured on 150 windows at 512
         bits the true next prime lands in the top 10% two thirds of the time
         with any of these; residue and bit statistics of the candidates do not
         sort (they match a random key).
      3. power (power=True): candidates that pass two modular powers (bases 2
         and 3) go ahead of those that fail, in offset order (they are primes,
         the lowest is the answer); the key orders the ones that failed.
         Every prime passes; almost no composite does.
    Returns (offsets, passed); with power=False, passed is all True."""
    anchor = int(anchor)
    if width is None:
        width = window_width(anchor)
    if seeds is None:
        seeds = seeds_for(anchor, width)
    f = RANK_KEYS[key] if isinstance(key, str) else key
    base = anchor + 1
    offs = [int(d) for d in window(base, width, seeds)]
    passed = [_fermat2(base + d) for d in offs] if power else [True] * len(offs)
    # passed candidates are primes: lowest offset first, the key orders the rest
    order = sorted(range(len(offs)), key=lambda k: (not passed[k], offs[k] if (power and passed[k]) else f(offs[k]), offs[k]))
    return np.array([offs[k] for k in order], dtype=np.int64), np.array([passed[k] for k in order])


# ── which offsets add the same as they XOR ──

def xor_additive(anchor, offsets):
    """Mask: anchor + d == anchor ^ d exactly when anchor & d == 0, i.e. when
    every set bit of d falls on a zero bit of the anchor. No carries, so XOR
    is addition there; where bits collide XOR subtracts instead."""
    anchor = int(anchor)
    return np.array([(anchor & int(d)) == 0 for d in offsets])


def xor_additive_bits(anchor, width):
    """The zero bits of the anchor below width: offsets built only from these
    bits are the additive ones. Returns the bit positions."""
    anchor = int(anchor)
    return [b for b in range(int(width).bit_length()) if not (anchor >> b) & 1]


def any_prime_after(anchor, workers=None, width=None, seeds=None, certify="full", key=None, deep=None):
    """A prime above anchor, the first one any core certifies. Not necessarily
    the next prime: there is no stop rule to wait on, so every core's pass is an
    answer and nothing is tested twice or past the finish. For "give me a prime
    of this size" this is the fast path; for the next prime use
    next_prime_after_parallel.
    certify="full"  strong test base 2, then gmpy2.is_prime (25 rounds)
    certify="prp"   strong tests on bases 2 and 3 only; the full test is off
    key=None        candidates go out lowest offset first; a RANK_KEYS name or
                    callable on the offset orders them instead. key="primeidx"
                    (Lars Rocha) sends survivors whose index in the list is a
                    prime number first, highest index first.
    deep=None       seed depth for window_deep (e.g. 3_000_000): residues for
                    every seed are computed in one vectorised pass, so millions
                    of seeds cost well under a second"""
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    import os
    anchor = int(anchor)
    if anchor < 2:
        return 2
    if width is None:
        width = window_width(anchor)
    if seeds is None:
        seeds = seeds_for(anchor, width)
    workers = workers or os.cpu_count() or 1
    base = anchor + 1
    pool = worker_pool(workers)
    f = None if key in (None, "primeidx") else (RANK_KEYS[key] if isinstance(key, str) else key)
    while True:
        w = window_deep(base, width, deep) if deep else window(base, width, seeds)
        offs = [int(d) for d in w]
        if key == "primeidx":
            order = sorted(range(len(offs)), key=lambda i: (0 if isprime(i) else 1, -i))
            offs = [offs[i] for i in order]
        elif f is not None:
            offs.sort(key=lambda d: (f(d), d))
        cands = [base + d for d in offs]
        # short chunks in offset order: the pool picks them up lowest first,
        # every core is busy, and no core holds a long slice when another wins
        chunk = 4
        fn = _first_prp_only if certify == "prp" else _first_prp
        futs = {pool.submit(fn, cands[k:k + chunk]): k for k in range(0, len(cands), chunk)}
        while futs:
            done, _ = wait(futs, return_when=FIRST_COMPLETED)
            for f in done:
                futs.pop(f)
                r = f.result()
                if r is not None:
                    for g in futs:
                        g.cancel()
                    return r
        base += width


def _first_prp(cands):
    for c in cands:
        if _prp(c):
            return c
    return None


# ── deep windows: residues for millions of seeds at once ──

def residues_mod(anchor, seeds):
    """anchor mod q for every q in `seeds` (int64 array), vectorised: the anchor
    is fed in 32-bit limbs through Horner's rule, so there is no big-integer
    division per seed. Seeds must be below 2^31."""
    anchor = int(anchor)
    q = np.asarray(seeds, dtype=np.int64)
    r = np.zeros_like(q)
    limbs = []
    while anchor:
        limbs.append(anchor & 0xFFFFFFFF)
        anchor >>= 32
    for limb in reversed(limbs):
        r = ((r << 32) + limb) % q
    return r


_DEEP_CACHE = {}


def deep_seeds(depth):
    """Odd primes up to depth as an int64 array, cached."""
    depth = int(depth)
    if depth not in _DEEP_CACHE:
        if len(_DEEP_CACHE) > 4:
            _DEEP_CACHE.clear()
        _DEEP_CACHE[depth] = _primes_upto(depth)[1:]
    return _DEEP_CACHE[depth]


def window_deep(anchor, width, depth):
    """window() with seeds up to `depth` (millions is fine). Small seeds strike
    in a loop; seeds wider than the window hit at most once, so they are struck
    in one vectorised pass. Returns ascending offsets like window()."""
    anchor = int(anchor)
    seeds = deep_seeds(depth)
    first = (-residues_mod(anchor, seeds)) % seeds           # first d with q | anchor + d
    alive = np.ones(width, dtype=bool)
    alive[(anchor % 2)::2] = False
    small = seeds < width
    for q, f in zip(seeds[small].tolist(), first[small].tolist()):
        alive[f::q] = False
        if anchor + f == q:
            alive[f] = True
    big_first = first[~small]
    alive[big_first[big_first < width]] = False
    return np.flatnonzero(alive).astype(np.int64)
