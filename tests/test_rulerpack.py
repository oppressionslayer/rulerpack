import numpy as np
import pytest

import rulerpack as rp

P2 = [4, 6, 10, 14, 22, 26, 34, 38, 46, 58, 62, 74, 82, 86, 94, 106, 118, 122,
      134, 142, 146, 158, 166, 178, 194, 202, 206, 214, 218, 226, 254]
RPS = [7, 15, 7, 31, 7, 15, 7, 63, 7, 15, 7, 31, 7, 15, 7, 127, 7, 15, 7, 31,
       7, 15, 7, 63, 7, 15, 7, 31, 7, 15, 7]
KEY_HEAD = [2, 8, 12, 16, 16, 20, 36, 24, 40, 52, 56, 84, 84, 88, 88, 20, 112,
            116, 128, 144, 148, 144, 160, 140, 196, 196, 200, 200, 220, 236, 248, 504]
KEY_TAIL = [3952, 3992, 3988, 3988, 3992, 4056, 4016, 4044, 4048, 4036, 4072]


def test_session_arrays():
    assert rp.primes2(31).tolist() == P2
    assert rp.RP(np.arange(31)).tolist() == RPS
    k = rp.key(309)
    assert k[:32].tolist() == KEY_HEAD and k[-11:].tolist() == KEY_TAIL
    assert np.array_equal(k ^ rp.rp_xor1(np.arange(309)), rp.primes2(309))


def test_rulers_are_ladders():
    i = np.arange(20000, dtype=np.int64)
    assert np.array_equal(rp.RP(i), (1 << (rp.v2(i + 1) + 3)) - 1)
    assert np.array_equal(rp.R257(i), 256 * rp.ladder(i >> 6))
    assert np.array_equal(rp.R513(i), 512 * rp.ladder(i >> 1))
    assert np.array_equal(rp.ruler(i, 4, 64), 4 * (i ^ (i + 64)))


def test_kx_identity():
    for fam in (rp.F513, rp.F257):
        P = rp.primes_first(5000)
        K, X = fam.encode_array(P)
        assert np.array_equal(K ^ X, fam.L * (np.arange(5000) + fam.c))


@pytest.mark.parametrize("fam", [rp.F513, rp.F257])
def test_roundtrip_100k(fam):
    ok, last = fam.check(100_000)
    assert ok and last == 1_299_709


def test_known_pair():
    assert rp.encode(78501, 1000039) == (2000590, 19669454)
    N = rp.pack(78501, 1000039)
    assert N == 8592468642374094 and rp.unpack(N) == (78501, 1000039)


def test_bignum():
    fam = rp.Family(256, 2, 1100)
    p = (1 << 1000) + 1245   # nextprime(2^1000)
    assert fam.unpack(fam.pack(78501, p)) == (78501, p)


def test_overflow_refused():
    with pytest.raises(ValueError):
        rp.Family(256, 2, 8).pack(5, 1000003)


def test_leftovers_and_build():
    # the seed rule drops squares, so 137*137 passes; first miss is 137*139
    assert np.array_equal(rp.leftovers(2163), rp.primes2(2163))
    assert rp.leftovers(2164)[2163] == 2 * 137 * 139
    L, K, X = rp.build_kx(309)
    assert np.array_equal(K, rp.key(309))
    assert np.array_equal(K ^ X ^ rp.RP(np.arange(309)) ^ 1, np.arange(0, 309 * 4, 4))


def test_sweep():
    P = rp.primes_first(3000)
    assert rp.sweep(P) == (2163, 137 * 139, 19051, 0)
    assert rp.sweep(P, prime_test=rp.isprime) == (None, 41)      # 137*139 and 40 more rejected; never a wrong index
    assert rp.sweep(P, step=256, prime_test=rp.isprime) == (None, 0)


def test_isprime():
    P = rp.primes_first(20000)
    assert all(rp.isprime(p) for p in P)
    from rulerpack.primes import _isprime_py
    assert all(_isprime_py(p) for p in P[:5000])
    ps = set(int(p) for p in P)
    assert not any(rp.isprime(n) for n in range(int(P[-1])) if n not in ps)
    assert rp.isprime((1 << 61) - 1) and not rp.isprime((1 << 67) - 1)
    assert rp.isprime(3825123056546413051) is False      # strong pseudoprime to bases 2..31
    assert rp.sweep(P[:2000], step=256) == (None, 0)


def test_counting_methods_agree():
    for x in (1, 2, 3, 10, 100, 1000, 12345, 1000039, 10**6):
        s = rp.primepi_sieve(x)
        assert rp.primepi_legendre(x) == s and rp.primepi_meissel(x) == s
    from math import gcd
    prod = [1, 2, 6, 30, 210, 2310, 30030, 510510, 9699690]
    for a in range(9):
        for x in (1, 7, 100, 1000, 12345):
            assert rp.phi(x, a) == sum(1 for n in range(1, x + 1) if gcd(n, prod[a]) == 1)
    assert rp.primepi_meissel(10**8) == 5761455
    assert rp.primepi_meissel(10**9) == 50847534
    assert rp.index_of(1000039) == 78501
    with pytest.raises(ValueError):
        rp.index_of(1000035)


def test_forward_then_pack():
    p = 15485863                                 # the millionth prime
    i = rp.index_of(p)
    assert i == 999999
    assert rp.unpack(rp.pack(i, p)) == (i, p)


def test_nth_prime():
    P = rp.primes_first(20000)
    for i in (0, 1, 5, 6, 7, 100, 1000, 19999):
        assert rp.nth_prime(i) == int(P[i])
    assert rp.nth_prime(78501) == 1000039
    assert rp.nth_prime(999999) == 15485863
    assert abs(rp.li(10**6) - 78627.5) < 1
    assert abs(rp.riemann_r(10**6) - 78527.4) < 1
    for i in (999, 99999, 999999):
        assert rp.nth_prime(i, guess=rp.li_inverse) == rp.nth_prime(i)


def test_four_directions():
    p = rp.nth_prime(78501)
    i = rp.index_of(p)
    N = rp.pack(i, p)
    assert (i, p) == (78501, 1000039) and rp.unpack(N) == (i, p)


def test_marks_walk_matches_sieve_and_meissel():
    M = rp.Marks.build(30000, every=1 << 10)
    P = rp.primes_first(30000)
    for i in (0, 1, 1023, 1024, 5000, 29999):
        assert M.nth_prime(i) == int(P[i])
        assert M.index_of(int(P[i])) == i
    assert M.nth_prime(20000) == rp.nth_prime(20000)
    with pytest.raises(ValueError):
        M.index_of(1000035)


def test_live_marks_fill_on_demand():
    lm = rp.LiveMarks(every=64)
    assert lm.nth_prime(78501) == 1000039 and len(lm) == 1
    assert lm.nth_prime(78502) == 1000081 and len(lm) == 1
    assert lm.index_of(1000039) == 78501
    assert lm.pi(10**6) == 78498 and lm.pi(10) == 4 and lm.pi(2) == 1
    m = rp.Marks.build(300_000, every=64)
    assert m.pi(10**6) == 78498 and m.pi(2) == 1
    assert len(rp.LiveMarks(every=64).seed(m)) == len(m)


def test_window_keeps_every_prime():
    P = rp.primes_first(200_000)
    for anchor in (10**6, 2**20, 1299700):
        d = rp.window(anchor, 50_000)
        got = set(anchor + int(x) for x in d)
        for p in P:
            if anchor <= p < anchor + 50_000:
                assert p in got
    assert rp.next_prime_after(10**12) == 1000000000039
    assert rp.next_prime_after(2**44) == 2**44 + 7
    assert rp.next_prime_after(2) == 3 and rp.next_prime_after(1) == 2
    d = rp.window(10**12, 100_000)
    assert len(d) < 20_000                      # about 18% survive


def test_primes_between_matches_sieve():
    P = rp.primes_first(200_000)
    assert rp.primes_between(0, 10).tolist() == [2, 3, 5, 7]
    assert rp.primes_between(2, 3).tolist() == [2]
    assert rp.primes_between(10, 10).size == 0
    for lo, hi in ((0, 1000), (1000, 50000), (2_000_000, 2_600_000), (131, 132), (1_000_000, 1_000_000 + 1)):
        want = [int(p) for p in P if lo <= p < hi]
        assert rp.primes_between(lo, hi).tolist() == want, (lo, hi)
    # small segment forces several segment boundaries
    assert rp.primes_between(1000, 60000, segment=777).tolist() == [int(p) for p in P if 1000 <= p < 60000]
    assert rp.primes_between(10**12, 10**12 + 100).tolist() == [1000000000039, 1000000000061, 1000000000063, 1000000000091]


def test_indexed_after():
    rows = rp.indexed_after(10**6, 3)
    assert [(i, p) for i, p, _ in rows] == [(78498, 1000003), (78499, 1000033), (78500, 1000037)]
    fam = rp.Family(256, 2, 64)
    for i, p, N in rows:
        assert fam.unpack(N) == (i, p)
    rows = rp.indexed_after(10**12, 2, count=37607912018)     # count from primecount
    assert [(i, p) for i, p, _ in rows] == [(37607912018, 1000000000039), (37607912019, 1000000000061)]


def test_big_next_prime_matches_gmpy2():
    gmpy2 = pytest.importorskip("gmpy2")
    for k in (127, 521, 1024):
        want = int(gmpy2.next_prime(2**k))
        assert rp.next_prime_after(2**k) == want
    assert rp.next_prime_after_parallel(2**1024, workers=4) == int(gmpy2.next_prime(2**1024))


def test_hinted_next_recovers_the_prime():
    P = rp.primes_first(300_000)
    bad = sum(1 for k in range(len(P) - 1) if rp.hinted_next(int(P[k]), rp.hint_of(P[k + 1], 11), 11) != int(P[k + 1]))
    assert bad == 0                                     # 11 bits: no gap below 2^64 reaches 2048
    for k in (10, 64, 128, 256):
        p = rp.next_prime_after(2**k)
        assert rp.hinted_next(2**k, rp.hint_of(p, 17), 17) == p


def test_hinted_pack_walks_primes_from_x_alone():
    for bits in (11, 13, 17):
        fam = rp.hinted_family(bits)
        P = rp.primes_first(50_000)
        K, X = fam.encode_array(P)
        i, p = fam.decode_array(K, X)
        assert (p == P).all() and (i == range(len(P))).all()        # the pack still opens
        assert rp.primes_from_x(2, X[1:], bits) == [int(q) for q in P[1:]]
    fam = rp.hinted_family(17, shift=300)
    a = 2**256
    ps = [rp.next_prime_after(a)]
    for _ in range(4):
        ps.append(rp.next_prime_after(ps[-1]))
    Xs = [fam.encode(k, q)[1] for k, q in enumerate(ps)]
    assert rp.primes_from_x(a, Xs, 17) == ps
    assert fam.unpack(fam.pack(3, ps[3])) == (3, ps[3])


def test_review_findings():
    sp = str(__import__("pathlib").Path(__file__).parent / "_tmp_marks.npy")
    # 4. L below 4 is refused, with a reason
    for L in (1, 2):
        with pytest.raises(ValueError):
            rp.Family(L, 2, 32)
    # 2. seeding from a finer or coarser table never puts two marks in one block
    P = rp.primes_first(4000)
    lm = rp.LiveMarks(every=64).seed(rp.Marks.build(4000, every=16))
    assert lm.nth_prime(0) == 2 and lm.nth_prime(63) == int(P[63]) and lm.nth_prime(3999) == int(P[3999])
    lm2 = rp.LiveMarks(every=64).seed(rp.Marks.build(4000, every=256))
    assert len(lm2) == 16 and lm2.nth_prime(300) == int(P[300])
    lm48 = rp.LiveMarks(every=64).seed(rp.Marks.build(4000, every=48))   # only indices 0,192,384,... land
    assert len(lm48) == len([i for i in range(0, 4000, 48) if i % 64 == 0])
    assert lm48.nth_prime(200) == int(P[200])
    # 3. one-mark file round-trips; sparse file goes to LiveMarks.load and is refused by Marks.load
    m1 = rp.Marks.build(10, every=1024)
    m1.save(sp)
    assert len(rp.Marks.load(sp)) == 1 and rp.Marks.load(sp).nth_prime(9) == 29
    lm.save(sp)
    with pytest.raises(ValueError):
        rp.Marks.load(sp)
    lm3 = rp.LiveMarks.load(sp)
    assert lm3.every == 64 and len(lm3) == len(lm) and lm3.nth_prime(3999) == int(P[3999])
    __import__("os").remove(sp)


def test_seed_bands_and_exact_windows():
    import gmpy2
    from rulerpack.window import EXACT_LIMIT
    # default: 31 seeds through 512 bits (they are already exact below 131^2)
    assert rp.seed_depth(8191, 100) is None and rp.seed_depth(2**512 - 1) is None
    assert all(gmpy2.is_prime(8191 + int(d)) for d in rp.window(8191, 100))
    assert rp.seed_depth(2**1024) == 8191
    assert rp.seed_depth(2**2048) == 125 * 2049 and rp.seed_depth(2**8191) == 1_000_000
    # exact: seed to sqrt -> every survivor is prime, no certification
    for a in (131071, 2**24, 10**9, 2**40):
        w = rp.window(a, 4096, seeds=rp.seeds_for(a, 4096, exact=True))
        assert len(w) and all(gmpy2.is_prime(a + int(d)) for d in w)
    assert rp.next_prime_after(131071, exact=True) == 131071 + 30
    assert rp.next_prime_after(10**12, exact=True) == 1000000000039
    with pytest.raises(ValueError):
        rp.next_prime_after(EXACT_LIMIT, exact=True)
    # width: largest gap plus slack below 2^64, 7 ln(anchor) above
    assert rp.window_width(10**12) == 1600 and rp.window_width(2**64 - 1) == 1600
    assert 39_000 < rp.window_width(2**8191) < 41_000
    for a in (10**12, 2**44, 2**60, 2**64 + 1, 2**600, 2**1100):
        assert rp.next_prime_after(a) == int(gmpy2.next_prime(a))
    # the record gap below 2^64 fits the first window
    assert rp.next_prime_after(18361375334787046697) == 18361375334787046697 + 1550


def test_squeeze_order_and_parallel_option():
    assert rp.squeeze_order(5) == [0, 4, 1, 3, 2]
    assert rp.squeeze_order(4) == [0, 3, 1, 2]
    assert sorted(rp.squeeze_order(37)) == list(range(37))
    import gmpy2
    a = 2**1024
    want = int(gmpy2.next_prime(a))
    assert rp.next_prime_after_parallel(a, workers=4, order="squeeze") == want
    # tight window where the prime sits near the top: both orders agree
    assert rp.next_prime_after_parallel(2**8191 + 1, workers=4, width=2000, order="squeeze") == 2**8191 + 1911


def test_gap_profile_follows_ln_x():
    g = rp.gap_profile(10**12, 50_000)
    assert len(g["gaps"]) == 49_999 and g["hist"].sum() == 49_999
    assert g["hist"][2] == g["twins"] and g["hist"][1] == 0 and g["hist"][3] == 0
    assert abs(g["twin_share"] - g["twin_pred"]) < 0.006          # 4.8% vs 4.78%
    assert abs(g["mean"] - g["ln"]) / g["ln"] < 0.05
    assert 0.3 < g["over_ln"] < 0.45
    big = rp.gap_profile(2**256, 20)
    assert len(big["gaps"]) == 19 and big["gaps"].min() >= 2


def test_ranked_window_puts_next_prime_first():
    import gmpy2
    for a in (10**12, 2**64 + 1, 2**256, 2**1024):
        offs, passed = rp.ranked_window(a)
        want = int(gmpy2.next_prime(a))
        assert a + 1 + int(offs[0]) == want and passed[0]
    # your window: 2^8191+1, width 2000, prime at +1910 goes to the top
    offs, passed = rp.ranked_window(2**8191, width=2000)
    assert int(offs[0]) == 1910 and passed[0] and not passed[1:].any()


def test_xor_additive_rule():
    a = 2**8191 + 1
    l = rp.window(a, 2000)
    assert rp.xor_additive(a, l).all()                     # all even offsets, anchor has bit 0 and bit 8191
    assert not rp.xor_additive(2**8191 - 1, l).any()       # all-ones anchor: every offset collides
    assert rp.xor_additive_bits(a, 2000) == list(range(1, 11))
    assert rp.xor_additive_bits(0b1010, 16) == [0, 2, 4]
    for d in range(0, 2000, 7):
        assert ((a + d) == (a ^ d)) == bool(rp.xor_additive(a, [d])[0])


def test_ranked_window_keys():
    import gmpy2
    a = 10**12
    want = int(gmpy2.next_prime(a)) - a - 1
    for key in ("offset", "blocks", "blocks16", lambda d: d // 32):
        offs, passed = rp.ranked_window(a, key=key)
        assert int(offs[0]) == want and passed[0]
    # without the power stage the order is the key alone
    offs, passed = rp.ranked_window(a, key="blocks", power=False)
    assert passed.all() and list(offs) == sorted(offs, key=lambda d: ((d >> 4) ^ 8, d))
    offs2, _ = rp.ranked_window(a, key="offset", power=False)
    assert list(offs2) == sorted(offs2)


def test_next_prime_after_key_orders_agree():
    import gmpy2
    for a in (10**12, 2**64 + 1, 2**300 + 12345):
        want = int(gmpy2.next_prime(a))
        for key in (None, "offset", "blocks", "blocks16", lambda d: -d):
            assert rp.next_prime_after(a, key=key) == want


def test_any_prime_after_returns_a_prime_in_the_window():
    import gmpy2
    for a in (10**12, 2**300 + 7):
        p = rp.any_prime_after(a, workers=4)
        assert p > a and gmpy2.is_prime(p) and p - a - 1 < rp.window_width(a)


def test_any_prime_after_prp_only_agrees_with_full_test():
    import gmpy2
    for a in (10**12, 2**300 + 7, 2**1024):
        p = rp.any_prime_after(a, workers=4, certify="prp")
        assert p > a and gmpy2.is_prime(p)
    rp.close_pools()


def test_any_prime_after_key_orders():
    import gmpy2
    for key in ("farkey", "blocks", lambda d: -d):
        p = rp.any_prime_after(2**300 + 7, workers=4, key=key)
        assert gmpy2.is_prime(p) and p > 2**300 + 7
    rp.close_pools()


def test_window_deep_matches_window_and_residues():
    import gmpy2
    from rulerpack.window import deep_seeds
    a = 2**2000 + 12345
    s = deep_seeds(200_000)
    assert (rp.residues_mod(a, s) == np.array([a % int(q) for q in s])).all()
    w = rp.window(a, 8192, seeds=s.tolist())
    assert list(rp.window_deep(a, 8192, 200_000)) == list(w)
    p = rp.any_prime_after(2**1500 + 1, workers=4, deep=500_000)
    assert gmpy2.is_prime(p)
    rp.close_pools()


def test_window_at_anchor_zero_keeps_the_seeds():
    # a seed at offset q of the window anchored at 0 was struck by its own stride
    assert list(rp.window(0, 20)) == [1, 3, 5, 7, 11, 13, 17, 19]   # offsets; at anchor 0, the numbers
    assert list(rp.window(3, 10)) == [0, 2, 4, 8]                   # 3, 5, 7, 11


def test_wheel_pass_defaults_to_the_seeds_and_takes_its_own():
    a = 2**8191 - 1
    full = rp.wheel_pass(a, 100)
    assert len(full) == 100 and full[0] == (0, None), "the Mersenne prime itself survives"
    assert full[1] == (1, 2), "2^8191 is even"
    # every survivor really has no seed factor; every removal names a real divisor
    for d, q in full:
        if q is None:
            assert all((a + d) % s for s in [2, *rp.seeds()])
        else:
            assert (a + d) % q == 0 and q in [2, *rp.seeds()]
    # a smaller wheel removes less, and the two agree wherever the small one removes
    small = rp.wheel_pass(a, 100, seeds=[3, 5, 7])
    assert sum(q is None for _, q in small) > sum(q is None for _, q in full)
    assert all(fq is not None for (_, sq), (_, fq) in zip(small, full) if sq is not None)
    lines = []
    n = rp.wheel_report(a, 10, show=lines.append)
    assert n == sum(q is None for _, q in full[:10]) and lines[0] == " 0: survives"


def test_the_hint_walk_rebuilds_the_primes_from_seventeen_bits_each():
    primes = rp.primes_first(100_000)
    fam = rp.hinted_family(17)
    K, X = fam.encode_array(primes)
    hints = rp.hints_from_x(X[1:], 17)
    assert hints.max() < 1 << 17
    assert np.array_equal(hints, (X[1:] >> 1) & ((1 << 17) - 1)), "Lars's one-liner"
    out = rp.hint_walk(2, hints, 17)
    assert out == primes[1:].tolist()
    assert out == rp.primes_from_x(2, X[1:], 17)
    # the walk does not care that they are primes: any climb under 2^bits per step
    seq = [10, 25, 26, 1000, 1001 + 2**17 - 1]
    hs = [x & (2**17 - 1) for x in seq[1:]]
    assert rp.hint_walk(seq[0], hs, 17) == seq[1:]
    # one step too far and the hint wraps to the wrong number: that is the one condition
    assert rp.hint_walk(0, [0], 17) == [2**17]


def test_the_512_slide_is_the_session_array_and_opens_back_to_the_primes():
    primes = rp.primes_first(2262)
    i = np.arange(2262)
    # In [6696]: the two aranges
    assert np.array_equal(rp.R896(np.arange(2047)), np.arange(7 * 128, 2048 * 512, 512) ^ np.arange(0, 2047 * 512, 512))
    assert np.array_equal(rp.R896(i), 384 + 512 * rp.ladder(i))
    # In [6697]: the slide over the first 31 primes
    want = ((primes[:31] * 512) ^ 512) ^ np.arange(7 * 128, 32 * 512, 512) ^ np.arange(0, 31 * 512, 512)
    B = rp.slide512(primes)
    assert np.array_equal(B[:31], want)
    assert B[:12].tolist()[:3] == [1408, 896, 2944]
    # the residues belong to the constant, not the primes
    assert (B % 512 == 384).all() and (B % 256 == 128).all() and (B % 128 == 0).all()
    assert (rp.slide512(np.arange(31)) % 256 == 128).all()
    # and the primes come straight back
    assert np.array_equal(rp.unslide512(B, i), primes)
    # repeats: 165 among 2261 steps, every one where p ^ p' is the carry, two thirds twins
    rep = rp.slide_repeats(primes)
    assert len(rep) == 165 and rep[:4].tolist() == [9, 12, 25, 27]
    assert (B[rep] == B[rep + 1]).all()
    assert np.array_equal(rep, np.flatnonzero(B[:-1] == B[1:]))
    gaps = primes[rep + 1] - primes[rep]
    assert (gaps == 2).sum() == 112 and (gaps == 6).sum() == 46
    assert rp.carry(np.arange(1, 17)).tolist() == [2, 6, 6, 2, 2, 14, 14, 2, 2, 6, 6, 2, 2, 30, 30, 2]
