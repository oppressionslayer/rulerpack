"""rulerpack command line.

    rulerpack 78501 1000039            pack a pair
    rulerpack 8592468642374094         unpack N
    rulerpack --check 1000000          round-trip the first n primes
    rulerpack --ladder 32              show the ladder and both rulers
    rulerpack --between lo hi          every prime in [lo, hi), sieve started at lo
    rulerpack --next N                 the closest prime above N
    options: --ruler 513|257  --shift S
"""

import sys

from .pack import Family, F513, F257
from .rulers import ladder, R257, R513, RP


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    fam = F513
    if "--ruler" in argv:
        k = argv.index("--ruler")
        fam = {"513": F513, "257": F257}[argv[k + 1]]
        del argv[k:k + 2]
    if "--shift" in argv:
        k = argv.index("--shift")
        fam = Family(fam.L, fam.c, int(argv[k + 1]))
        del argv[k:k + 2]
    if not argv:
        print(__doc__)
        return 0
    if argv[0] == "--check":
        n = int(argv[1]) if len(argv) > 1 else 100_000
        ok, last = fam.check(n)
        print(f"{n:,} primes, last {last:,}, L={fam.L} c={fam.c} M=2^{fam.shift}: "
              f"round-trip {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    if argv[0] == "--between":
        from .window import primes_between
        P = primes_between(int(argv[1]), int(argv[2]))
        print("\n".join(str(int(p)) for p in P))
        return 0
    if argv[0] == "--next":
        from .window import next_prime_after
        print(next_prime_after(int(argv[1])))
        return 0
    if argv[0] == "--ladder":
        n = int(argv[1]) if len(argv) > 1 else 32
        print(f"{'i':>4} {'ladder':>7} {'RP':>5} {'R257':>7} {'R513':>7}")
        for i in range(n):
            print(f"{i:>4} {ladder(i):>7} {RP(i):>5} {R257(i):>7} {R513(i):>7}")
        return 0
    if len(argv) == 2:
        i, p = int(argv[0]), int(argv[1])
        K, X = fam.encode(i, p)
        N = fam.pack(i, p)
        print(f"i={i} p={p}   (L={fam.L} c={fam.c} M=2^{fam.shift})")
        print(f"K = {K}\nX = {X}\nN = {N}   ({N.bit_length()} bits)")
        print(f"unpack(N) = {fam.unpack(N)}")
        return 0
    N = int(argv[0])
    i, p = fam.unpack(N)
    print(f"N -> i={i} p={p}   (L={fam.L} c={fam.c} M=2^{fam.shift})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
