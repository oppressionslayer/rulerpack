# rulerpack

XOR rulers over the primes, and the pack that rides on them.
Created by Lars Rocha. Python implementation with Claude assisting.

```python
import rulerpack as rp

N = rp.pack(78501, 1000039)      # 8592468642374094, one integer
rp.unpack(N)                     # (78501, 1000039), the index and the prime back
```

Decoding is mod, div and two XORs. No search, no primality test, no table.

## Credit

The ideas in this library are Lars Rocha's. Claude assisted with the
implementation and the measurements; that does not imply ownership.

| idea | what it became here |
|---|---|
| XOR the doubled prime against a ruler and against the index lattice | the pack: `K ^ X = L(i+c)`, one integer opens to the pair in four ops, no collisions |
| sieve the leftovers with the 3..131 seeds | `allowed`, `window`; taken to the square root it is `primes_between`, no primality test |
| anchor at a number and calculate the residues there | Eratosthenes started anywhere: one division per seed, then strike-outs |
| the prime closest to the anchor | survivors in distance order; `next_prime_after` is the first |
| the survivors are a list, not a stream | `next_prime_after_parallel`: certify on every core, 4x gmpy2's `next_prime` at 12,000 bits |
| count once, then increment | `indexed_after`: next n primes with exact indices and packs; thousands of times faster than calling primecount's `nth_prime` per index, about 5x slower than primesieve's list mode, which returns no indices |
| remember the counts | `Marks` and `LiveMarks`: each checkpoint a packed N |
| the seed rule at the cube root | its rejections are Meissel's correction term, measured equal |

The segmented sieve, Legendre's phi, Meissel's formula, Miller-Rabin and the
li / R estimates are classical and credited to their authors. primecount is
Kim Walisch's and is used as an optional counter.

## The ruler

Every ruler here is a Mersenne ladder, `m ^ (m+1)`, scaled by a power of two.
Drawn as tick marks it is a tape measure: every block gets a tick, and the
tick is taller at every power of two.

```
  0 |        1
  1 ||       3
  2 |        1
  3 |||      7
  4 |        1
  5 ||       3
  6 |        1
  7 ||||     15
```

| name | formula | equals |
|---|---|---|
| `rp.RP(i)` | `(4i) ^ (4i+7)` | `2^(v2(i+1)+3) - 1`, the original 7, 15, 7, 31, ... |
| `rp.ruler(i, L, c)` | `((Li) ^ (Li+cL+1)) ^ 1` | `L * (i ^ (i+c))` |
| `rp.R257(i)` | L=4, c=64 | `256 * ladder(i >> 6)` |
| `rp.R513(i)` | L=256, c=2 | `512 * ladder(i >> 1)` |
| `rp.R896(i)` | `(896 + 512i) ^ (512i)` | `384 + 512 * ladder(i)`: 896, 1920, 896, 3968, ... |
| `rp.carry(i)` | `ladder(i) ^ ladder(i+1)` | 2, 6, 6, 2, 2, 14, 14, 2, ... the ruler's carry |

**The 512 slide.** `rp.slide512(primes)` is the session's `((p*512) ^ 512) ^ R896(i)`;
every entry is `384 + 512 * ((p ^ 1) ^ ladder(i))`, so the array is 384 mod 512 and
128 mod 256 for any input at all, and `rp.unslide512(B, i)` reads the primes straight
back. `rp.slide_repeats(primes)` finds where it repeats, `B[i] == B[i+1]`, which is
exactly where `p ^ p' == carry(i)`: 165 of the first 2,261 steps, two thirds of them
twin primes. The carry is independent of the gap, so the repeats are a lamp on the
sequence rather than a sieve of it; the measurements are in `ruler_stack` next door.

## The pack

For a prime `p` at zero-based index `i`, in the family `(L, c, M)`:

```
K = 2p ^ ruler(i, L, c)
X = 2p ^ L*i
N = K*M + X                 M = 2^shift, fixed for the family; X < M is checked on encode
```

Then `K ^ X = L*(i + c)` for every `i` and `p`, so

```
X = N mod M,  K = N div M,  i = (K ^ X)/L - c,  p = (X ^ L*i)/2
```

`M` is a parameter of the family like `L` and `c`. It is never stored in `N`
and never guessed back from it; guessing it is a search and the search collides.

```python
rp.F513                          # Family(L=256, c=2, shift=32), the default
rp.F257                          # Family(L=4, c=64, shift=32)
rp.Family(256, 2, 1100)          # for 1000-bit primes
rp.encode(i, p), rp.decode(K, X)
rp.encode_array(P)               # numpy, a million primes at once
```

Verified round trip on the first 1,000,000 primes for both families.

## Counting, the forward map

Each method is its own function so they can be compared.

| function | what | π(10^9) |
|---|---|---|
| `rp.primepi_sieve(x)` | sieve to x | 28.9 s |
| `rp.phi(x, a)` | Legendre: numbers up to x with no factor among the first a primes | |
| `rp.primepi_legendre(x)` | `phi(x, a) + a - 1`, a = π(√x) | 4.3 s |
| `rp.primepi_meissel(x)` | a = π(∛x) plus the correction sum; `rp.primepi` | 0.07 s |
| `rp.index_of(p)` | `primepi(p) - 1`, refuses a composite | |
| `rp.nth_prime(i)` | Riemann R guesses, Meissel pins, the walk closes | billionth in 0.67 s |

All exact, checked against the published values of π up to 10^12.

The seed rule (odd, no factor in 3..131, not a square) is Legendre's `phi`
applied one candidate at a time. With seeds up to ∛x, the composites that
survive the rule are exactly the terms of Meissel's correction sum. Checked:
at x = 10^6 both are 42,286.

If you need fast counting, use [primecount](https://github.com/kimwalisch/primecount).
It is about 100x faster than this at 10^12. This library counts so the pack
can be tested from nothing; counting is not what it is for.

## Marks, our own way

Sieve once, keep every k-th prime as a packed `N`, and from then on walk from
the nearest mark with the seed rule and `isprime`. No li, no Riemann, no
Meissel. Each mark carries its own index and prime, so the table is self-describing.

```python
M = rp.Marks.build(50_000_000, every=1024)   # 4.4 s, 48,829 marks
M.nth_prime(49_999_999)                       # 982451653 in about 1 ms
M.index_of(982451653)                         # 49999999
```

Flat a few milliseconds at any index the table covers. The Rust crate
(`rulerpack-rs`) does the same with a spacing of 64 in about 100 µs.
`M.pi(x)` counts to any x the table covers by the same walk.

## Live marks, the table that fills itself

Same walk, same pack, no table up front. A block you have never touched costs
one counting call (`nth_prime`: Riemann guess, Meissel pin, walk) and is stored
as a packed N; every later lookup in that block is a walk.

```python
lm = rp.LiveMarks(every=64)
lm.nth_prime(999_999_999)        # 22801763489 in 0.67 s, one mark computed
lm.nth_prime(999_999_998)        # 22801763477 in 1 ms, same mark
lm.pi(10**9)                     # 50847534 in 0.19 s, three marks
lm.seed(rp.Marks.load())         # start from a built table
```

## The window: a sieve of Eratosthenes that starts anywhere

Every candidate is `anchor + d`, so its residue mod a seed q is `(anchor mod q) + d`.
One division per seed finds where the multiples of q fall in the window; after
that it is strike-outs, no arithmetic per candidate.

```python
rp.window(10**12, 1_000_000)        # survivor offsets, closest first; 2 ms
rp.next_prime_after(10**12)         # 1000000000039 in about 100 us
rp.next_prime_after(2**60)          # 1152921504606847009
rp.primes_between(10**15, 10**15 + 10**7)   # every prime there, certified, 0.5 s
```

Any size. The first window is the largest known prime gap plus slack below
2^64 (the next prime is always inside) and 7 ln(anchor) above (one miss in a
thousand, then it slides). Seed depth follows the cost of a test: the 31 small
seeds through 512 bits, 8191 through 1536, up to a million above, where a
single certification costs 100 ms and the deeper sieve still pays. A base-2
probable-prime test runs before the full one.

The survivors are independent, so the parallel paths certify them on every
core. `next_prime_after_parallel` returns the smallest that passes, same
answers as gmpy2's `next_prime`. `any_prime_after` returns the first pass from
any core: a prime above the anchor, not necessarily the next one, with no stop
rule to wait on. For "a random prime of this size" that is the fast path.
The worker pool stays alive between calls (`rp.close_pools()` to release it).

| anchor | gmpy2.next_prime | any_prime_after | next_prime_after_parallel |
|---|---|---|---|
| 2^2048 | 0.15 s | 0.07 s | 0.08 s |
| 2^4096 | 1.29 s | 0.43 s | 0.68 s |
| 2^8191 | 7.84 s | 2.91 s | 3.31 s |
| 2^12191 | 61.6 s | 14.7 s | 21.2 s |

(16 cores, warm pool, every result a verified prime.) The ordering of the
survivors is the one knob that cannot be improved: the next prime needs every
survivor below it cleared, and offset order tests exactly those. We measured
residue statistics, bit patterns, Jacobi symbols and offset keys as sort keys
against a random key; the only things that sort are the offset itself (the
gap distribution) and a modular power of the candidate. `ranked_window` shows
that order; `order="squeeze"` (both ends toward the centre) and `key=` are
there to try your own. `gap_profile` accounts for a range: gap histogram,
twin share against 1.32/ln x, mean against ln x.

`window` strikes with the 31 seeds up to 131 and leaves primes plus a few
composites for `isprime` to settle. `primes_between` brings every prime up to
the square root of the top, so what survives is prime and nothing is tested.
Counts agree with primecount at 10^12, 10^15 and 10^18.

| primes in | count | Python | Rust |
|---|---|---|---|
| [10^12, +10^7) | 361,726 | 0.12 s | 38 ms |
| [10^15, +10^7) | 289,394 | 0.52 s | 161 ms |
| [10^18, +10^6) | 24,280 | 7.7 s | 3.7 s |

(The 10^18 case is mostly building the 50 million seeds up to 10^9.)

```
rulerpack --between 1000000000000 1000000000100
rulerpack --next 1000000000000
```

## Hints: the next prime from its low bits, no test

A number between two consecutive primes is less than one gap from the next
prime, so it cannot share that prime's low `b` bits once 2^b exceeds the gap.
`hinted_next(anchor, hint, bits)` is therefore one modulo and one add, and it
is right whenever the gap is under 2^bits: proven for 11 bits below 2^64
(largest gap 1,550), heuristic above (13 bits to about 1e39, 17 bits to about
2^256 with headroom). Choose the bits.

```python
p = rp.next_prime_after(2**256)             # certified once: 2^256 + 297
rp.hinted_next(2**256, rp.hint_of(p, 17))   # p again, about 1 us, no test
```

The hint can live inside the pack. `hinted_family(bits)` uses L = 2^(bits+1),
so the bottom bits of X are the bottom bits of 2p, untouched by the index, and
`K ^ X = L(i+c)` still opens the pair. A bare stream of X values then walks
the primes from any start, no K, no index, no count, no test:

```python
fam = rp.hinted_family(17, shift=300)
K, X = fam.encode_array(rp.primes_first(1_000_000))
rp.primes_from_x(2, X[1:], 17)              # every prime back, 0.5 s, zero tests
```

**Seventeen bits per prime is the whole thing.** Strip X down to the hints and
throw the rest away; the walk still runs, and so does the comparison Lars ran
in the session:

```python
hints = (X >> 1) & ((1 << 17) - 1)          # rp.hints_from_x(X, 17)

p = 2
out = []
for h in hints[1:31]:
    p = rp.hinted_next(p, int(h), 17)
    out.append(p)

out == rp.primes_from_x(2, X[1:31], 17)     # True
rp.hint_walk(2, hints[1:], 17)              # the same walk, as one call
```

So one encoding reads two ways: K and X together open any pair directly, by
index or by prime, and the low bits of X alone rebuild the whole sequence in
order. The walk is not about primes in particular; any increasing integers
rebuild from their low bits as long as each step stays under the hint modulus,
and the prime gaps do so to about 2^256 at 17 bits.

## Indexed primes above an anchor

Count once at the anchor, then every prime found above it takes the next index
and packs. With primecount installed the count is milliseconds at any size.

Baseline, stated exactly. The only route in Kim Walisch's tools that returns an
index is primecount's `nth_prime`, called once per prime; against that this is
thousands of times faster (10,000 indexed primes above 10^12: 39 ms here, 190 s
there). His list mode, primesieve, sieves a range without indices and is faster
than this at listing: a million primes above 10^14 in 0.04 s against 1.1 s for
`primes_between` in Python and 0.22 s for the Rust `between`. A reader with both
tools gets an indexed list in about 0.3 s. What they do not get is the pack,
the marks, or random access by index; that is what this library is for.
The Rust crate's `indexed_after` runs the same race in compiled code: given the
count, primesieve lists a million primes above 10^14 in 49 ms and the Rust
sieve takes 597 ms while also attaching indices and packs.

```python
rp.indexed_after(10**12, 3)            # [(37607912018, 1000000000039, N), ...]
rp.indexed_after(2**44, 3)             # 53 ms with primecount installed
rp.indexed_after(2**60, 3)             # 20 s, all of it primecount counting to 1.15e18
rp.primepi_fastest(x)                  # primecount if present, else Meissel
```

## Which to use

| tool | holds | cost per call | reaches |
|---|---|---|---|
| `primepi` (Meissel) | nothing | grows with x, 0.07 s at 10^9 | about 10^12 here |
| `Marks` | a table built once | flat, the walk length | the end of the table |
| `LiveMarks` | only blocks you have touched | first touch counts, after that the walk | wherever the counter reaches |
| [primecount](https://github.com/kimwalisch/primecount) | nothing | 20 ms at 10^12, one thread | 10^20 and beyond |
| `any_prime_after` | a warm pool | 2.9 s at 8191 bits, 16 cores | any size; a prime, not the next |
| `next_prime_after_parallel` | a warm pool | 3.3 s at 8191 bits | any size; the next prime |

primecount computes; marks remember. Use the calculator for a one-off count of
any size, the ruler for many lookups in a range you have drawn.

## Also

- `rp.isprime(n)`: gmpy2's Miller-Rabin when gmpy2 is present, else a pure
  Python deterministic one exact below 3.3·10^24.
- `rp.sweep(P, step, prime_test)`: the window sweep. With a residue hint of
  p mod 256 the seed rule finds the next prime with no primality test for the
  first 831,000 primes.
- `rp.li`, `rp.li_inverse`, `rp.riemann_r`, `rp.riemann_r_inverse`.
- `rp.key(n)`, `rp.topo(n)`, `rp.build_kx(n)`: the original session arrays.
- `rp.wheel_pass(a, count=100, seeds=None)`: the seed wheel run over `a, a+1, ...`; each
  offset's first divisor among `[2, *seeds]`, or `None` when it survives. `seeds` defaults
  to `SEEDS`; `a` is any int, so `rp.wheel_report(2**8191 - 1)` prints which of the hundred
  numbers past the Mersenne prime the wheel removes and by what (twelve survive).

## Experiments

`rulerpack.experiments` is the drawer for things that were fun in a session and might be
more. Nothing else imports it. Each function reproduces an IPython moment exactly and says
what it saw.

```python
import rulerpack.experiments as rx
rx.stripe(primes)              # (idx ^ 2p) % 4 with idx = arange(2, n, 10): 2, 0, 2, 0, ...
rx.jumps(primes)               # 97, 172, 309, 564 among them: 2p crossing 1024, 2048, 4096, 8192
rx.last_before_blocks(primes)  # 127, 251, 383, 509, ... 1021, 2039, 4093, 8191 among them
rx.layers(primes)              # z % 4, 8, 16, 32: one more bit of the primes per row, the stripe under all of them
rx.recover(primes, 32)         # ((z % 32) ^ (idx % 32)) >> 1 == p % 16: XOR hides nothing
rx.stripe_control()            # odd composites stripe the same way: the stripe is not about primes
```

## Install

```
pip install -e .            # numpy only; gmpy2 optional (pip install -e ".[fast]")
pytest
rulerpack --check 1000000
rulerpack 78501 1000039
rulerpack --ladder 32
```

MIT license.

## Changes

**0.2.1** (2026-11-01)

- `rulerpack.experiments`: the drawer, with the step-10 index, its layers, the recovery and the
  odd-composite control. Ported to the Rust crate's `experiments` module in the same release.

**0.2.0** (2026-11-01)

- `wheel_pass`, `wheel_report`: the seed wheel over any anchor, `seeds` as a parameter with
  `SEEDS` the default; `2**8191 - 1` works as an anchor.
- `hints_from_x`, `hint_walk`: the primes back in order from seventeen bits each, no test.
- `R896`, `carry`, `slide512`, `unslide512`, `slide_repeats`: the 512 slide and where it repeats.

**0.1.0** — the rulers, the pack, counting, marks, the window, hints, indexed primes.
