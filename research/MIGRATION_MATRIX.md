# Rust/CUDA migration matrix

Tracks, per CKKS primitive operation, exactly how far the "CKKS RUST +
CUDA EVOLUTION" effort (`rust/` workspace, Path B: Rust host +
EXISTING, unmodified CUDA C++ kernels via FFI) has gotten. Updated each
round a new operation is touched. See `EXPERIMENT_LEDGER.md` entries
#32 (NTT), #35 (rescale), and #36 (key switching) for full narrative
detail behind each row.

Columns, left to right, are cumulative prerequisites (a later column
implies the earlier ones hold):

- **C++-only**: the operation exists in the main `gpufhe` C++/CUDA
  engine (`src/`, `kernels/`) and nowhere else. This is the default/
  starting state for everything in the library.
- **FFI-exposed**: a narrow `cpp_bridge/*.cpp` file forwards an
  existing C++ function through `extern "C"` so Rust can call it
  without reimplementing its arithmetic.
- **Rust-abstracted**: a Rust struct/function in `ckks-cuda` wraps the
  FFI calls + existing CUDA kernels behind a safe API, following the
  setup/hot-path split (`NttContext`/`RescaleContext` pattern).
- **Independently tested**: the Rust wrapper is checked against a
  reference computed FRESH in Rust (not imported from the C++ test),
  not merely "two implementations agree."
- **Benchmarked fairly**: a same-setup-scope 4-tier (cold-start /
  context-init / steady-state / end-to-end) comparison exists in BOTH
  languages, with results reported as medians across multiple trials,
  run in both orders to rule out a GPU-state confound.
- **Production-integrated**: something in the actual engine or a real
  application calls the Rust path instead of (or alongside) the C++
  path. **Nothing in this table has reached this column yet.**

| Operation | C++-only | FFI-exposed | Rust-abstracted | Independently tested | Benchmarked fairly | Production-integrated |
|---|---|---|---|---|---|---|
| Primitive-root search (`native_root`) | done | done (`native_root_bridge.cpp`) | done (`ntt::native_root`) | done (order-exactly-2n check) | n/a (sub-microsecond host call, folded into NTT's cold-start tier) | no |
| Coefficient-domain NTT/INTT round trip | done | n/a (kernel called directly, no bridge needed) | done (`NttContext`) | done (56/56 exact cases, `test_ntt_roundtrip_n1024.cpp`-equivalent, both one-shot and reused-context) | **done** (entry #32: C++ 119.6us vs Rust 108.7us median, equivalent within noise) | no |
| Fused-rescale constants (`native_rescale_consts`) | done | done (`rescale_consts_bridge.cpp`, this round) | done (`rescale::native_rescale_consts`) | done (folded into rescale's own differential test) | n/a (sub-microsecond host call, folded into rescale's cold-start tier) | no |
| Single-modulus RNS rescale (`rescale_resident_raw`'s algorithm) | done | n/a (kernels called directly; constants via the bridge above) | done (`RescaleContext`, this round) | **done** (entry #35: 4 new tests, 0 mismatches vs. a fresh independent RNS reference, including a real domain-mismatch bug CAUGHT in the first draft of the C++ comparison and fixed before any timing was trusted) | **done** (entry #35: C++ 535-573us vs Rust 534-542us steady-state median, equivalent within noise, reproduced in both run orders) | no |
| Full Hybrid key switching (`keyswitch_resident`, `DeviceKSContext`/`DeviceKSWork`) | done | done (`keyswitch_bridge.cpp`, this round -- a single fixture-lifecycle bridge, not one function per primitive, since the setup state is a genuine eval key, not bookkeeping) | done (`KeySwitchContext`, this round) | **partial, explicitly flagged** (entry #36: 6 new tests, bit-exact vs. `keyswitch_core_resident` -- the established oracle for this exact operation -- across random/boundary/repeated-call cases; this is NOT a from-scratch independent reference the way rescale's RNS identity was, since both paths implement the same high-level algorithm; the one attempted from-scratch algebraic check for this operation (`ba0+ba1*s =?= a*s_old`) was already proven an invalid identity in entry #30 and is not reattempted) | **done** (entry #36: fixture creation ~6.5-6.8ms, steady-state ~1.06-1.09ms, end-to-end ~8.4-9.2ms median, both languages, equivalent within noise, reproduced in both run orders) | no |
| EvalMult / tensor product + relin | done | not started | not started | not started | not started | no |
| Bootstrap pipeline (C2S/S2C/EvalMod) | done | not started | not started | not started | not started | no |

**Reminder for future rounds**: do not infer from this table that the
whole engine is "being ported" or that any security property has been
re-certified for the Rust path -- three operations out of a large engine
are abstracted and benchmarked, nothing is production-integrated, and
each new row requires its own independent correctness reference, not
an assumption that the established pattern transfers automatically. In
particular, key switching's "independently tested" status is weaker
than rescale's or NTT's -- see the detail column above and
`EXPERIMENT_LEDGER.md` entry #36 before citing it as equivalent
evidence.

## Memory: what this round measured (entry #36)

Peak GPU device memory was measured (sampled every 50ms via
`nvidia-smi` across an extended run of each operation's own Rust
benchmark) for all three Rust-abstracted operations at their respective
n=1024-scale configurations: NTT, rescale, and key switching ALL peak
at **85 MiB** -- identical across all three, with no growth across
500,000 (NTT/rescale) or 200,000 (key switching) repeated steady-state
calls on one reused context (confirmed leak-free, not merely
"eventually stable"). This 85 MiB is the fixed CUDA runtime/driver
context floor for this GPU/driver combination, not operation-specific
data: analytically, NTT's own tables are ~40KB, rescale's ~200KB, and
even the FULL key-switch eval key + device context + scratch is on the
order of a few hundred KB at this n/tower count -- all negligible next
to the fixed floor. **Conclusion, per the governing directive's own
"measure before changing the architecture" instruction: there is
currently no measurable benefit to sharing root tables or constants
across Rust contexts at this scale** -- the C++ engine's OWN
process-lifetime static caches for key-switching's root tables and
per-part constants (`g_rtab`/`upc` in `src/keyswitch_resident.cpp`,
confirmed via source reading this round) already solve this problem at
the ONE place it has ever mattered in this codebase's history (dozens
of `DeviceKSContext`s rebuilt across many rotations in the bootstrap
hot loop) -- and Rust's `KeySwitchContext` inherits that sharing for
free simply by calling into the existing `ks_context_create`, without
needing its own cache. Introducing a similar shared, mutable cache
PURELY WITHIN the Rust crate (across separate `NttContext`/
`RescaleContext` instances at the same modulus) is deliberately NOT
done this round: no real multi-context-same-modulus Rust workload
exists yet to justify the added complexity (thread-safety, invalidation,
shared-ownership risk) against a savings of tens of KB. Revisit this
decision if/when a genuine multi-tower or bootstrap-scale Rust workload
emerges -- at n=131072 the per-tower table sizes (and tower counts)
could make the trade-off real, unlike at n=1024.
