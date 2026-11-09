//! GOD PROMPT "CKKS PHASE 25": the registered-memory budget-exhaustion test,
//! moved OUT of `codec.rs`'s unit-test module and into its own integration-test
//! file so that Cargo gives it its own PROCESS.
//!
//! # Why it had to move
//!
//! The C++ registered-host-memory pool enforces a self-imposed 16 MiB cumulative
//! cap (`kRegisteredBudgetBytes` in
//! `src/encrypt_host_shoup_pooled_barrett_batched_registered.cpp`). Two
//! properties of that pool make it process-global and one-way:
//!
//!   * `g_registeredBytesUsed` only ever **increases** -- there is no release,
//!     eviction or deregistration path, by design: a registered buffer is meant
//!     to stay registered for the process lifetime so it can be reused.
//!   * `registered_cache()` never evicts an `(n, size_q)` entry.
//!
//! That is a perfectly sensible production contract. But it means a test that
//! deliberately exhausts the budget permanently consumes it for *every other
//! test sharing the process*. `cargo test` runs all unit tests of a crate in
//! ONE process, so this test and its neighbours were coupled through 16 MiB of
//! global state.
//!
//! The observable consequence, measured this round:
//!
//! ```text
//! cargo test --release -- --test-threads=1
//!   codec::registered_tests::prewarm_is_idempotent_and_makes_first_encrypt_registered  FAILED
//!   codec::registered_tests::repeated_encryption_pool_reuse_stays_registered_and_correct FAILED
//!   55 passed; 2 failed
//! ```
//!
//! Both failures are the same shape: a test asks for a *small, fresh* key to
//! register successfully, and gets `budget_skipped` instead, because this test
//! had already taken the whole budget. Nothing is wrong with the production code
//! and nothing is wrong with those assertions -- the test harness was simply
//! sharing a one-way global resource.
//!
//! Phase 24 diagnosed this correctly as a deterministic test-order artifact.
//! Phase 25 adds the fix, and also corrects one part of Phase 24's conclusion:
//! "skipping that one test makes everything else pass unconditionally" is true
//! only in SERIAL mode. In the default parallel mode a completely separate
//! defect -- a shared device scratch buffer in `src/rotate.cpp`, now fixed with
//! `thread_local` -- was corrupting the rotation tests, and skipping this test
//! did nothing for those. See `tests/test_rotate_concurrent_cache.cpp`.
//!
//! # Why a separate process is the right fix
//!
//! The directive's constraints were: do not skip the test, do not relax it, and
//! do not change production semantics. A separate process satisfies all three:
//!
//!   * **Not skipped.** Plain `cargo test` still builds and runs this file; it
//!     is simply a different binary. Nothing is `#[ignore]`d.
//!   * **Not relaxed.** Every assertion below is byte-for-byte the original,
//!     including the two that make the test meaningful (`saw_registered > 0`
//!     and `saw_budget_skipped > 0`). In a fresh process the budget starts at 0
//!     and the ~23 MiB of requests below still exceed the 16 MiB cap, so the
//!     budget still genuinely binds -- the test proves exactly what it did
//!     before.
//!   * **Production untouched.** No C++ file changed for this. In particular no
//!     test-only reset/deregistration hook was added: that would have put a
//!     budget-release path into production code purely to serve a test, which is
//!     the tail wagging the dog, and would have weakened the very "registered
//!     buffers live for the process lifetime" contract the pool exists to
//!     provide.
//!
//! The alternative fixes were considered and rejected: ordering constraints
//! (Rust's test harness offers none, and tests run in parallel by default);
//! a shared mutex plus state restoration (needs the production release path
//! above); and loosening the neighbours' assertions to tolerate a skip (that is
//! relaxing a test to hide an infrastructure problem).
//!
//! # Duplication, stated honestly
//!
//! `make_fixture` and `test_message` are duplicated from `codec.rs`'s test
//! module because they are private test helpers there. They are a handful of
//! lines built entirely from public API (`KeySwitchContext`, `native_root`), and
//! duplicating them was preferable to widening the crate's public surface with a
//! test-support module just to share them.

use ckks_core::{Modulus, RingDimension};
use ckks_cuda::codec::{encrypt, encrypt_registered};
use ckks_cuda::{native_root, KeySwitchContext};

struct Fixture {
    n: u32,
    mods: Vec<u64>,
    roots: Vec<u64>,
    pk_a: Vec<u64>,
    pk_b: Vec<u64>,
}

fn make_fixture(n: u32, size_q: u32, seed: u64) -> Fixture {
    // num_part must stay <= size_q (KeySwitchContext::new itself rejects
    // num_part > size_q) -- clamp rather than hardcode 4, since this helper is
    // exercised at size_q values as small as 1.
    let num_part = 4u32.min(size_q);
    let ctx = KeySwitchContext::new(RingDimension::new(n).unwrap(), size_q, 2, num_part, seed)
        .expect("valid P>=Q_part_max fixture");
    let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
    let roots: Vec<u64> = mods
        .iter()
        .map(|&q| native_root(ctx.ring_dimension(), Modulus::new(q).unwrap()))
        .collect();
    Fixture { n, mods, roots, pk_a: ctx.public_key_a(), pk_b: ctx.public_key_b() }
}

fn test_message(n: u32, seed: i64) -> Vec<i64> {
    (0..n as i64).map(|k| ((k + seed) % 97) - 48).collect()
}

/// Items 6/7: budget exhaustion and the resulting safe fallback, observed from
/// Rust. Mirrors the C++ concurrency audit's own scenario 6 (distinct large
/// keys deliberately summing over the 16MiB C++-side budget) but exercised
/// through the FFI boundary, confirming the diagnostic out-params propagate
/// correctly across it.
///
/// This is the ONLY test in this binary, deliberately: it is here precisely
/// because it must not share a process with tests that need a non-exhausted
/// budget. Do not add more tests to this file -- add them to
/// `codec.rs`'s unit-test module instead.
#[test]
fn budget_exhaustion_falls_back_safely_and_stays_correct() {
    // NOTE (NOT a registered-memory finding -- a separate, narrow, pre-existing
    // defect this test's own fixture sweep surfaced): `KeySwitchContext::new`/
    // `gpufhe_ks_fixture_create` crashes with `std::bad_alloc` (NOT a clean
    // `Err`) at `size_q=5, size_p=2, num_part=4`, reproducibly, regardless of
    // `n`, confirmed via a standalone diagnostic sweeping `size_q` 1..=11 with
    // `num_part=4.min(size_q)` at a fixed `n=8192` (size_q=1-4,6-8 -> Ok;
    // size_q=5 -> `std::bad_alloc`; size_q=9-11 -> a CLEAN, EXPECTED
    // `P < Q_part_max` rejection, not a crash -- `num_part=4` with `size_p=2`
    // simply does not scale past `size_q=8`, the same well-known boundary from
    // entries #42/#43). `size_q=5` is EXCLUDED below for exactly this reason --
    // a real, separate, narrow bug worth recording, still out of scope to fix
    // here (the governing directive's "do not modify keyswitch_resident.cpp
    // internals without a standalone test proving the proposed path first"
    // applies).
    //
    // Packed bytes = 4*size_q*n*8. These 7 SAFE size_q values at BOTH n=8192
    // and n=16384 (14 fixtures total) sum to ~23.25MiB, comfortably over the
    // C++ pool's 16MiB budget -- enough to genuinely bind without size_q=5 or
    // size_q>=9.
    let sizes_q: [u32; 7] = [1, 2, 3, 4, 6, 7, 8];
    let mut saw_registered = 0usize;
    let mut saw_budget_skipped = 0usize;
    for &size_q in &sizes_q {
        for &n in &[8192u32, 16384u32] {
            let f = make_fixture(n, size_q, 8000 + n as u64 + size_q as u64);
            let m = test_message(f.n, 3);
            let (c0_ref, c1_ref) =
                encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 3).unwrap();
            let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 3)
                .unwrap();
            assert_eq!(
                r.c0, c0_ref,
                "n={n} size_q={size_q}: c0 must be bit-exact regardless of which path ran"
            );
            assert_eq!(
                r.c1, c1_ref,
                "n={n} size_q={size_q}: c1 must be bit-exact regardless of which path ran"
            );
            if r.used_registered {
                saw_registered += 1;
                assert!(!r.budget_skipped && r.last_error_code == 0);
            } else if r.budget_skipped {
                saw_budget_skipped += 1;
                assert_eq!(
                    r.last_error_code, 0,
                    "a budget skip is not a registration failure -- error code must stay 0"
                );
            }
        }
    }
    assert!(
        saw_registered > 0,
        "expected at least some keys to register before the budget bound. If this fails, \
         something ELSE in this process consumed the 16MiB budget first -- this file is \
         supposed to be the only test in its binary, so check that no test was added here."
    );
    assert!(
        saw_budget_skipped > 0,
        "expected the budget to genuinely bind for at least one of these 14 keys -- \
         otherwise this test is not exercising the fallback path at all"
    );
}
