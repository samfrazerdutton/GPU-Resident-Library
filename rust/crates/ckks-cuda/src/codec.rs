//! CKKS encode/decode/encrypt/decrypt primitives (GOD PROMPT "CKKS PHASE
//! 5": the real encrypted EvalMult round trip). Thin, stateless FFI
//! wrappers over the EXISTING, unmodified `gpufhe::encode_host`/
//! `decode_host`/`encrypt_host`/`decrypt_host` (`src/encode.cpp`,
//! `src/encdec.cpp`), via `cpp_bridge/encdec_bridge.cpp`. No cryptographic
//! or encoding arithmetic is reimplemented here -- this module exists only
//! to adapt the calling convention (complex slot vectors cross as separate
//! real/imaginary `f64` arrays; everything else as plain pointer+length).
//!
//! ## Why this module exists now, and not in an earlier round
//!
//! `ntt`/`rescale`/`keyswitch`/`evalmult` all operate at the raw RNS
//! level: their own tests generate synthetic per-tower residues directly
//! and never touch a real plaintext, a real scale, or real encryption
//! noise (see `evalmult`'s own module doc comment for why that was a
//! deliberate scope boundary, not an oversight). This module -- paired
//! with [`crate::keyswitch::KeySwitchContext::secret_key`]/
//! `public_key_a`/`public_key_b` (new this round) -- is what finally lets
//! a test perform encode -> encrypt -> `EvalMultContext::
//! multiply_relin_rescale` -> decrypt -> decode, and compare against the
//! mathematically expected product within a REAL, noise-inclusive
//! tolerance, instead of a bit-exact RNS identity. See
//! `evalmult::tests`'s `real_encrypted_evalmult_round_trip` tests for the
//! actual round trip built on top of this module.
//!
//! ## `decrypt`'s tower-0-only recovery, and what it means for scale choice
//!
//! `decrypt_host` recovers the message by centering the phase
//! `c0+c1*s` AT TOWER 0 ONLY (the largest modulus in the chain), not a
//! full CRT reconstruction across all towers -- confirmed by reading
//! `src/encdec.cpp` directly, not assumed. This is a pre-existing,
//! documented limitation (the same one `tests/test_keyswitch.cpp`'s own
//! comments flag for a different function), not something this round
//! introduces or can relax. It means correct decryption REQUIRES the
//! message's true integer magnitude (`scale * |slot value|`, roughly) to
//! stay below `tower_0_modulus / 2` at every stage that gets decrypted --
//! which is exactly why `evalmult::tests`'s round-trip tests pick a scale
//! comparable to the rescale-dropped tower's modulus (so the POST-rescale
//! scale, `delta^2 / q_last`, stays close to `delta` instead of collapsing
//! toward 1) and keep slot magnitudes small.
use ckks_core::CkksError;

extern "C" {
    fn gpufhe_encode(z_re: *const f64, z_im: *const f64, num_slots: u32, n: u32, delta: f64, m_out: *mut i64);
    fn gpufhe_decode(m: *const i64, n: u32, delta: f64, z_re_out: *mut f64, z_im_out: *mut f64);
    fn gpufhe_encrypt(
        m: *const i64,
        n: u32,
        pk_a: *const u64,
        pk_b: *const u64,
        mod_: *const u64,
        root: *const u64,
        size_q: u32,
        ns: u64,
        sigma: f64,
        seed: u64,
        c0_out: *mut u64,
        c1_out: *mut u64,
    );
    fn gpufhe_decrypt(
        c0: *const u64,
        c1: *const u64,
        s: *const u64,
        n: u32,
        mod_: *const u64,
        root: *const u64,
        size_q: u32,
        m_out: *mut i64,
    );
    // GOD PROMPT "CKKS PHASE 22": EXPERIMENTAL -- CUDA-registered host
    // memory (Phase 20's `encrypt_host_shoup_pooled_barrett_batched_
    // registered`), reachable from Rust for the first time. Same
    // ownership shape as `gpufhe_encrypt` above: every argument is
    // copied into/out of a `std::vector` on the C++ side, nothing is
    // retained across the call, and the underlying C++ function calls
    // `cudaDeviceSynchronize()` before returning -- so by the time this
    // call returns, no asynchronous GPU work still touches `c0_out`/
    // `c1_out`'s backing memory, and no pointer this function received
    // is held past the call. `out_used_registered`/`out_budget_skipped`/
    // `out_last_error_code` are ALWAYS written by the C++ side (see
    // `encrypt_host_shoup_pooled_barrett_batched_registered_diag`'s own
    // comment) -- never read uninitialized here.
    fn gpufhe_encrypt_registered(
        m: *const i64,
        n: u32,
        pk_a: *const u64,
        pk_b: *const u64,
        mod_: *const u64,
        root: *const u64,
        size_q: u32,
        ns: u64,
        sigma: f64,
        seed: u64,
        c0_out: *mut u64,
        c1_out: *mut u64,
        out_used_registered: *mut u8,
        out_budget_skipped: *mut u8,
        out_last_error_code: *mut i32,
    );
    fn gpufhe_prewarm_registered_buffer(n: u32, size_q: u32, out_budget_skipped: *mut u8, out_last_error_code: *mut i32) -> u8;
    fn gpufhe_registered_buffer_is_ready(n: u32, size_q: u32) -> u8;
}

/// Result of [`encrypt_registered`] -- the ciphertext PLUS diagnostics
/// about which allocation path actually executed. The C++ side's own
/// invariant (see `EncryptBufPoolRegistered`'s comment in `src/
/// encrypt_host_shoup_pooled_barrett_batched_registered.cpp`)
/// guarantees a key is budget-skipped XOR attempted-and-failed XOR
/// ready, never more than one of those three.
#[derive(Debug, Clone, PartialEq)]
pub struct EncryptRegisteredResult {
    pub c0: Vec<u64>,
    pub c1: Vec<u64>,
    /// true iff the CUDA-registered path was actually used for this
    /// call; false means the unmodified production pageable path
    /// (`encrypt_host_shoup_pooled_barrett_batched`) computed `c0`/`c1`
    /// instead -- bit-exact either way (see this crate's own
    /// correctness tests), but the allocation-cost profile differs.
    pub used_registered: bool,
    /// true iff `!used_registered` because this `(n, size_q)` key's
    /// packed buffer would have exceeded the C++ pool's self-imposed
    /// 16MiB cumulative registered-bytes budget -- registration was
    /// never even ATTEMPTED in that case. A deliberate, documented
    /// resource limit, not a failure.
    pub budget_skipped: bool,
    /// The `cudaError_t` value from the registration attempt, as a
    /// plain integer (0 == `cudaSuccess`). Meaningful ONLY when
    /// `!used_registered && !budget_skipped` -- i.e. registration was
    /// genuinely attempted and failed. Always 0 when `used_registered`
    /// or `budget_skipped` is true.
    pub last_error_code: i32,
}

/// Encodes up to `n/2` REAL values (imaginary part zero) into `n`
/// integer coefficients at scale `delta`. `values.len()` must be `<=
/// n/2`; unused slots are implicitly zero. This is the real-only
/// convention the directive's own worked example (`[1.0, 2.0, 3.0,
/// 4.0]`) uses -- `gpufhe::encode_host` itself supports full complex
/// slots, but nothing in this round's tests needs them.
pub fn encode_real(values: &[f64], n: u32, delta: f64) -> Result<Vec<i64>, CkksError> {
    if values.len() as u32 > n / 2 {
        return Err(CkksError::InvalidRingDimension { n: values.len() as u64 });
    }
    let zeros = vec![0.0f64; values.len()];
    let mut m = vec![0i64; n as usize];
    unsafe {
        gpufhe_encode(values.as_ptr(), zeros.as_ptr(), values.len() as u32, n, delta, m.as_mut_ptr());
    }
    Ok(m)
}

/// Decodes `n` integer coefficients at scale `delta` into `n/2` complex
/// slots, returned as `(real, imaginary)` parts.
pub fn decode(m: &[i64], n: u32, delta: f64) -> Result<(Vec<f64>, Vec<f64>), CkksError> {
    if m.len() != n as usize {
        return Err(CkksError::InvalidRingDimension { n: m.len() as u64 });
    }
    let slots = (n / 2) as usize;
    let mut re = vec![0.0f64; slots];
    let mut im = vec![0.0f64; slots];
    unsafe {
        gpufhe_decode(m.as_ptr(), n, delta, re.as_mut_ptr(), im.as_mut_ptr());
    }
    Ok((re, im))
}

/// Encrypts `m` (`n` integer coefficients) under public key
/// `(pk_a, pk_b)` (each `size_q*n` elements) over the modulus chain
/// `mod_`/`root` (`size_q` entries each, eval-domain NTT roots matching
/// `crate::ntt::native_root`'s own convention -- the SAME convention
/// `KeySwitchContext`'s fixture uses internally). Returns `(c0, c1)`,
/// each `size_q*n` elements, eval domain.
pub fn encrypt(
    m: &[i64],
    n: u32,
    pk_a: &[u64],
    pk_b: &[u64],
    mods: &[u64],
    roots: &[u64],
    ns: u64,
    sigma: f64,
    seed: u64,
) -> Result<(Vec<u64>, Vec<u64>), CkksError> {
    let size_q = mods.len() as u32;
    if m.len() != n as usize {
        return Err(CkksError::InvalidRingDimension { n: m.len() as u64 });
    }
    if pk_a.len() != size_q as usize * n as usize || pk_b.len() != size_q as usize * n as usize {
        return Err(CkksError::InvalidRingDimension { n: pk_a.len() as u64 });
    }
    if roots.len() != mods.len() {
        return Err(CkksError::InvalidRingDimension { n: roots.len() as u64 });
    }
    let mut c0 = vec![0u64; size_q as usize * n as usize];
    let mut c1 = vec![0u64; size_q as usize * n as usize];
    unsafe {
        gpufhe_encrypt(
            m.as_ptr(), n, pk_a.as_ptr(), pk_b.as_ptr(), mods.as_ptr(), roots.as_ptr(), size_q,
            ns, sigma, seed, c0.as_mut_ptr(), c1.as_mut_ptr(),
        );
    }
    Ok((c0, c1))
}

/// GOD PROMPT "CKKS PHASE 22": EXPERIMENTAL sibling of [`encrypt`],
/// identical arguments and identical arithmetic, computed via Phase
/// 20's CUDA-registered (NOT cudaMallocHost-pinned) host staging
/// buffer when that pool is ready for this `(n, size_q)` key, or via
/// the SAME unmodified production pageable path [`encrypt`] itself
/// uses otherwise. Bit-exact with [`encrypt`] either way -- see this
/// module's `tests::registered_matches_production_and_pinned` and the
/// underlying C++ correctness tests
/// (`tests/test_encrypt_host_shoup_pooled_barrett_batched_registered_
/// matches_reference.cpp`) this wraps. `encrypt`'s own default
/// production behavior is UNCHANGED by this function's existence --
/// nothing calls this unless a caller explicitly does.
///
/// Returns [`EncryptRegisteredResult`], which reports WHICH path
/// actually executed and why, rather than hiding that behind a silent
/// fallback -- see that struct's own field docs.
pub fn encrypt_registered(
    m: &[i64],
    n: u32,
    pk_a: &[u64],
    pk_b: &[u64],
    mods: &[u64],
    roots: &[u64],
    ns: u64,
    sigma: f64,
    seed: u64,
) -> Result<EncryptRegisteredResult, CkksError> {
    let size_q = mods.len() as u32;
    if m.len() != n as usize {
        return Err(CkksError::InvalidRingDimension { n: m.len() as u64 });
    }
    if pk_a.len() != size_q as usize * n as usize || pk_b.len() != size_q as usize * n as usize {
        return Err(CkksError::InvalidRingDimension { n: pk_a.len() as u64 });
    }
    if roots.len() != mods.len() {
        return Err(CkksError::InvalidRingDimension { n: roots.len() as u64 });
    }
    let mut c0 = vec![0u64; size_q as usize * n as usize];
    let mut c1 = vec![0u64; size_q as usize * n as usize];
    let mut used_registered: u8 = 0;
    let mut budget_skipped: u8 = 0;
    let mut last_error_code: i32 = 0;
    unsafe {
        gpufhe_encrypt_registered(
            m.as_ptr(), n, pk_a.as_ptr(), pk_b.as_ptr(), mods.as_ptr(), roots.as_ptr(), size_q,
            ns, sigma, seed, c0.as_mut_ptr(), c1.as_mut_ptr(),
            &mut used_registered, &mut budget_skipped, &mut last_error_code,
        );
    }
    Ok(EncryptRegisteredResult {
        c0,
        c1,
        used_registered: used_registered != 0,
        budget_skipped: budget_skipped != 0,
        last_error_code,
    })
}

/// Explicit, idempotent prewarm for [`encrypt_registered`]'s pool --
/// mirrors Phase 16's own `encrypt_prewarm_pinned_buffer` C++ API
/// shape, now exposed to Rust for the registered pool specifically.
/// Returns `(ready, budget_skipped, last_error_code)`; see
/// [`EncryptRegisteredResult`]'s field docs for how to interpret them.
/// Calling this before the first [`encrypt_registered`] call at a
/// given `(n, size_q)` moves that key's registration cost OUT of the
/// first real encryption call -- calling it again at the SAME key is
/// a no-op query (idempotent, never re-registers).
pub fn prewarm_registered_buffer(n: u32, size_q: u32) -> (bool, bool, i32) {
    let mut budget_skipped: u8 = 0;
    let mut last_error_code: i32 = 0;
    let ready = unsafe { gpufhe_prewarm_registered_buffer(n, size_q, &mut budget_skipped, &mut last_error_code) };
    (ready != 0, budget_skipped != 0, last_error_code)
}

/// Query-only: is the registered pool already ready for `(n, size_q)`?
/// Never allocates, never registers, never takes the pool's mutex for
/// longer than a lookup.
pub fn registered_buffer_is_ready(n: u32, size_q: u32) -> bool {
    unsafe { gpufhe_registered_buffer_is_ready(n, size_q) != 0 }
}

/// Decrypts `(c0, c1)` (each `size_q*n` elements) under secret key `s`
/// (`size_q*n` elements) over `mod_`/`root` (`size_q` entries). Returns
/// `n` integer coefficients. See this module's doc comment for why
/// `size_q` and the scale must be chosen so the true magnitude stays
/// under `mods[0]/2`.
pub fn decrypt(
    c0: &[u64],
    c1: &[u64],
    s: &[u64],
    n: u32,
    mods: &[u64],
    roots: &[u64],
) -> Result<Vec<i64>, CkksError> {
    let size_q = mods.len() as u32;
    let expected = size_q as usize * n as usize;
    if c0.len() != expected || c1.len() != expected || s.len() != expected {
        return Err(CkksError::InvalidRingDimension { n: c0.len() as u64 });
    }
    if roots.len() != mods.len() {
        return Err(CkksError::InvalidRingDimension { n: roots.len() as u64 });
    }
    let mut m = vec![0i64; n as usize];
    unsafe {
        gpufhe_decrypt(c0.as_ptr(), c1.as_ptr(), s.as_ptr(), n, mods.as_ptr(), roots.as_ptr(), size_q, m.as_mut_ptr());
    }
    Ok(m)
}

// ============================================================================
// GOD PROMPT "CKKS PHASE 22": Rust/FFI tests for `encrypt_registered`/
// `prewarm_registered_buffer`/`registered_buffer_is_ready` -- the FIRST
// time Phase 20's CUDA-registered host-memory path has been reachable
// from Rust at all. Scope, stated explicitly: these tests compare the
// new `encrypt_registered` against the EXISTING production `encrypt`
// (both reachable from Rust); Phase 15/16's PINNED variant is NOT
// FFI-exposed (deliberately -- it was never promoted, and wiring it in
// was not this phase's mission), so the three-way production/pinned/
// registered comparison stays at the C++ level, where Phase 20's own
// `tests/test_encrypt_host_shoup_pooled_barrett_batched_registered_
// matches_reference.cpp` already performs it and is re-run as part of
// this phase's regression, not duplicated here.
//
// Also scoped out, and why: items from the governing directive's
// Section 5 list that do not apply at THIS boundary, because of how
// the FFI call is shaped (see `gpufhe_encrypt_registered`'s own doc
// comment) --
//   - "registration failure" from Rust: there is no way to pass an
//     invalid `cudaHostRegister` flag or similar through this FFI
//     surface (by design -- Rust callers cannot misconfigure it that
//     way). Registration-failure handling is tested at the C++ level
//     (`tests/test_registered_host_buffer.cpp`'s scenarios 4/5) and is
//     re-run as part of this phase's regression.
//   - "shutdown/resource destruction," "repeated construction and
//     destruction," "CUDA operation completion before resource
//     release": the registered buffer is a C++-pool-owned,
//     process-lifetime resource that NEVER crosses the FFI boundary as
//     an object -- `gpufhe_encrypt_registered` copies every argument
//     in, calls `cudaDeviceSynchronize()` (confirmed by reading `src/
//     encrypt_host_shoup_pooled_barrett_batched_registered.cpp`) before
//     copying the result out, and returns. There is no Rust-side
//     handle whose `Drop` could race anything, and no asynchronous GPU
//     work is ever still in flight when the call returns -- this is a
//     structural property of the call shape, not something a
//     dedicated test could meaningfully exercise beyond what every
//     correctness test below already implicitly relies on (a test that
//     immediately decrypts and compares the returned ciphertext would
//     fail if the GPU work were not actually complete).
#[cfg(test)]
mod registered_tests {
    use super::*;
    use crate::keyswitch::KeySwitchContext;
    use ckks_core::RingDimension;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    struct Fixture {
        n: u32,
        mods: Vec<u64>,
        roots: Vec<u64>,
        pk_a: Vec<u64>,
        pk_b: Vec<u64>,
        #[allow(dead_code)] // kept for fixture completeness/future decrypt-based tests; not read by every test
        s: Vec<u64>,
    }

    fn make_fixture(n: u32, size_q: u32, seed: u64) -> Fixture {
        // num_part must stay <= size_q (KeySwitchContext::new itself
        // rejects num_part > size_q) -- clamp rather than hardcode 4,
        // since this helper is deliberately exercised at size_q values
        // as small as 1 (the budget-exhaustion/concurrency tests sweep
        // size_q in 1..=8 to match the C++ concurrency audit's own
        // scenario 6 range).
        let num_part = 4u32.min(size_q);
        let ctx = KeySwitchContext::new(RingDimension::new(n).unwrap(), size_q, 2, num_part, seed).expect("valid P>=Q_part_max fixture");
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        Fixture { n, mods, roots, pk_a: ctx.public_key_a(), pk_b: ctx.public_key_b(), s: ctx.secret_key() }
    }

    fn test_message(n: u32, seed: i64) -> Vec<i64> {
        (0..n as i64).map(|k| ((k + seed) % 97) - 48).collect()
    }

    /// Item 1: successful registered encryption, bit-exact against the
    /// existing production `encrypt`.
    #[test]
    fn registered_matches_production() {
        let f = make_fixture(1024, 4, 7001);
        let m = test_message(f.n, 11);
        let (c0_ref, c1_ref) = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 42).unwrap();
        let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 42).unwrap();
        assert_eq!(r.c0, c0_ref, "c0 mismatch between encrypt and encrypt_registered");
        assert_eq!(r.c1, c1_ref, "c1 mismatch between encrypt and encrypt_registered");
        assert!(!r.budget_skipped, "a single small fresh key should never be budget-skipped");
        assert_eq!(r.last_error_code, 0, "no registration failure expected for a small fresh key");
    }

    /// Item 2: repeated encryption and pool reuse -- the SAME key
    /// called many times must stay bit-exact against production every
    /// time, and once registered, must STAY registered (never silently
    /// fall back on a later call).
    #[test]
    fn repeated_encryption_pool_reuse_stays_registered_and_correct() {
        let f = make_fixture(2048, 4, 7002);
        let mut saw_registered_after_first = false;
        for seed in 1u64..=25 {
            let m = test_message(f.n, seed as i64);
            let (c0_ref, c1_ref) = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
            let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
            assert_eq!(r.c0, c0_ref, "seed {seed}: c0 mismatch");
            assert_eq!(r.c1, c1_ref, "seed {seed}: c1 mismatch");
            if seed >= 2 {
                // Call 1 may be the key's first-ever use (allocates);
                // every call from the 2nd onward must find the SAME
                // key already registered and ready.
                assert!(r.used_registered, "seed {seed}: expected the pool to stay registered after its first use");
                saw_registered_after_first = true;
            }
        }
        assert!(saw_registered_after_first, "test did not actually exercise reuse");
    }

    /// Items 3/4: explicit prewarming, and idempotency (calling prewarm
    /// twice at the same fresh key must not re-register or change the
    /// outcome).
    #[test]
    fn prewarm_is_idempotent_and_makes_first_encrypt_registered() {
        let f = make_fixture(4096, 4, 7003);
        assert!(!registered_buffer_is_ready(f.n, f.mods.len() as u32), "fresh key must not already be ready");

        let (ready1, skipped1, err1) = prewarm_registered_buffer(f.n, f.mods.len() as u32);
        let (ready2, skipped2, err2) = prewarm_registered_buffer(f.n, f.mods.len() as u32);
        assert_eq!((ready1, skipped1, err1), (ready2, skipped2, err2), "prewarm must be idempotent: identical result on a second call");
        assert!(ready1, "a small fresh key should register successfully");

        let m = test_message(f.n, 5);
        let (c0_ref, c1_ref) = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 5).unwrap();
        let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 5).unwrap();
        assert!(r.used_registered, "the FIRST encrypt call after an explicit prewarm must already be registered");
        assert_eq!(r.c0, c0_ref);
        assert_eq!(r.c1, c1_ref);
    }

    /// Items 6/7: budget exhaustion and the resulting safe fallback,
    /// observed from Rust. Mirrors the C++ concurrency audit's own
    /// scenario 6 (16 distinct large keys, ~27MiB total, deliberately
    /// over the 16MiB C++-side budget) but exercised through the FFI
    /// boundary this time, confirming the diagnostic out-params
    /// propagate correctly across it.
    #[test]
    fn budget_exhaustion_falls_back_safely_and_stays_correct() {
        // GOD PROMPT "CKKS PHASE 22" NOTE (NOT a registered-memory
        // finding -- a separate, narrow, pre-existing defect this
        // test's own fixture sweep surfaced): `KeySwitchContext::new`/
        // `gpufhe_ks_fixture_create` crashes with `std::bad_alloc`
        // (NOT a clean `Err`) at `size_q=5, size_p=2, num_part=4`,
        // reproducibly, regardless of `n`, confirmed via a standalone
        // diagnostic sweeping `size_q` 1..=11 with `num_part=4.min(size_q)`
        // at a fixed `n=8192` (size_q=1-4,6-8 -> Ok; size_q=5 ->
        // `std::bad_alloc`; size_q=9-11 -> a CLEAN, EXPECTED
        // `P < Q_part_max` rejection, not a crash -- `num_part=4` with
        // `size_p=2` simply does not scale past `size_q=8`, the same
        // well-known boundary from entries #42/#43). `size_q=5` is
        // EXCLUDED below for exactly this reason -- this is a real,
        // separate, narrow bug worth recording (see research/
        // EXPERIMENT_LEDGER.md), but out of THIS phase's scope to fix
        // (the governing directive's own "do not modify
        // keyswitch_resident.cpp internals without a standalone test
        // proving the proposed path first" applies, and this phase's
        // mission is registered-memory validation, not a general
        // eval-keygen bug hunt).
        //
        // Packed bytes = 4*size_q*n*8. These 7 SAFE size_q values at
        // BOTH n=8192 and n=16384 (14 fixtures total) sum to ~23.25MiB,
        // comfortably over the C++ pool's 16MiB budget -- enough to
        // genuinely bind without size_q=5 or size_q>=9.
        let sizes_q: [u32; 7] = [1, 2, 3, 4, 6, 7, 8];
        let mut saw_registered = 0usize;
        let mut saw_budget_skipped = 0usize;
        for &size_q in &sizes_q {
            for &n in &[8192u32, 16384u32] {
                let f = make_fixture(n, size_q, 8000 + n as u64 + size_q as u64);
                let m = test_message(f.n, 3);
                let (c0_ref, c1_ref) = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 3).unwrap();
                let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 3).unwrap();
                assert_eq!(r.c0, c0_ref, "n={n} size_q={size_q}: c0 must be bit-exact regardless of which path ran");
                assert_eq!(r.c1, c1_ref, "n={n} size_q={size_q}: c1 must be bit-exact regardless of which path ran");
                if r.used_registered {
                    saw_registered += 1;
                    assert!(!r.budget_skipped && r.last_error_code == 0);
                } else if r.budget_skipped {
                    saw_budget_skipped += 1;
                    assert_eq!(r.last_error_code, 0, "a budget skip is not a registration failure -- error code must stay 0");
                }
            }
        }
        assert!(saw_registered > 0, "expected at least some keys to register before the budget bound");
        assert!(saw_budget_skipped > 0, "expected the budget to genuinely bind for at least one of these 14 keys -- otherwise this test is not exercising the fallback path at all");
    }

    /// Item 8: FFI error propagation parity -- the SAME Rust-side
    /// pre-FFI validation `encrypt` already performs must reject the
    /// same malformed inputs for `encrypt_registered`, before ever
    /// reaching the FFI call.
    #[test]
    fn validation_errors_match_production_encrypt() {
        let f = make_fixture(1024, 4, 7004);
        let bad_m = vec![0i64; f.n as usize + 1]; // wrong length
        let err_ref = encrypt(&bad_m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 1).unwrap_err();
        let err_new = encrypt_registered(&bad_m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 1).unwrap_err();
        assert_eq!(format!("{err_ref:?}"), format!("{err_new:?}"), "encrypt_registered must reject malformed input the same way encrypt does");
    }

    /// Item 9: concurrent Rust callers -- many threads calling
    /// `encrypt_registered` simultaneously, both at the SAME key (pool
    /// contention) and at DISTINCT fresh keys (pool growth under
    /// concurrency), must never produce a mismatch against production.
    #[test]
    fn concurrent_rust_callers_same_and_distinct_keys() {
        let shared = make_fixture(1024, 4, 7005);
        let mismatches = AtomicUsize::new(0);
        let total = AtomicUsize::new(0);

        thread::scope(|scope| {
            // 6 threads hammering the SAME key.
            for t in 0..6u64 {
                let f = &shared;
                let mismatches = &mismatches;
                let total = &total;
                scope.spawn(move || {
                    for it in 0..10u64 {
                        let seed = 100_000 + t * 1000 + it;
                        let m = test_message(f.n, seed as i64);
                        let (c0_ref, c1_ref) = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
                        let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
                        total.fetch_add(1, Ordering::Relaxed);
                        if r.c0 != c0_ref || r.c1 != c1_ref {
                            mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
            // 4 threads, each growing the pool with its OWN distinct
            // fresh key concurrently with the above. size_q=6, not 5
            // -- size_q=5 at num_part=4/size_p=2 is a separate,
            // narrow, pre-existing `gpufhe_ks_fixture_create` defect
            // (see `budget_exhaustion_falls_back_safely_and_stays_correct`'s
            // own comment), unrelated to what THIS test exercises.
            for (idx, &size_q) in [1u32, 2, 3, 6].iter().enumerate() {
                let mismatches = &mismatches;
                let total = &total;
                scope.spawn(move || {
                    let f = make_fixture(512, size_q, 200_000 + idx as u64);
                    for it in 0..6u64 {
                        let seed = 300_000 + idx as u64 * 1000 + it;
                        let m = test_message(f.n, seed as i64);
                        let (c0_ref, c1_ref) = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
                        let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
                        total.fetch_add(1, Ordering::Relaxed);
                        if r.c0 != c0_ref || r.c1 != c1_ref {
                            mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });

        let total = total.load(Ordering::Relaxed);
        let mismatches = mismatches.load(Ordering::Relaxed);
        eprintln!("concurrent_rust_callers: total={total} mismatches={mismatches}");
        assert_eq!(mismatches, 0, "concurrent Rust callers must never see a mismatch against production encrypt");
        assert_eq!(total, 6 * 10 + 4 * 6);
    }

    /// Item 10: explicit prewarm racing with real encryption at a
    /// NEVER-before-seen key, from Rust -- mirrors the C++ concurrency
    /// audit's scenario 5.
    #[test]
    fn prewarm_racing_with_encryption_stays_correct() {
        let f = make_fixture(8192, 2, 7006); // fresh key, distinct from every other test's (n,size_q)
        let mismatches = AtomicUsize::new(0);
        thread::scope(|scope| {
            for t in 0..8u64 {
                let f = &f;
                let mismatches = &mismatches;
                scope.spawn(move || {
                    if t % 2 == 0 {
                        let _ = prewarm_registered_buffer(f.n, f.mods.len() as u32);
                    } else {
                        for it in 0..5u64 {
                            let seed = 400_000 + t * 1000 + it;
                            let m = test_message(f.n, seed as i64);
                            let (c0_ref, c1_ref) = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
                            let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, seed).unwrap();
                            if r.c0 != c0_ref || r.c1 != c1_ref {
                                mismatches.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });
        assert_eq!(mismatches.load(Ordering::Relaxed), 0);
        assert!(registered_buffer_is_ready(f.n, f.mods.len() as u32), "after the race, this small key should end up ready one way or another");
    }

    /// The critical end-to-end test the directive names explicitly:
    /// encode -> encrypt_registered -> EvalMult -> decrypt -> decode,
    /// compared against the SAME independent correctness oracle
    /// `evalmult.rs`'s own real-encrypted tests use (the expected
    /// elementwise product), NOT an RNS-only identity.
    #[test]
    fn real_encrypted_evalmult_via_registered_path_matches_expected_product() {
        use crate::evalmult::EvalMultContext;
        let ctx = EvalMultContext::new(RingDimension::new(8192).unwrap(), 8, 2, 4, 1234).unwrap();
        let n = ctx.ring_dimension().get();
        let size_q = 8u32;
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let delta = ctx.modulus_q(size_q - 1) as f64;
        let pk_a = ctx.public_key_a();
        let pk_b = ctx.public_key_b();

        let slots_a = [1.0, 2.0, 3.0, 4.0];
        let slots_b = [5.0, 6.0, 7.0, 8.0];
        let expected: Vec<f64> = slots_a.iter().zip(slots_b.iter()).map(|(a, b)| a * b).collect();

        let m_a = encode_real(&slots_a, n, delta).unwrap();
        let m_b = encode_real(&slots_b, n, delta).unwrap();
        // The registered path encrypts BOTH ciphertexts here -- this is
        // the actual substitution being evaluated for promotion, not a
        // synthetic RNS-level stand-in.
        let ra = encrypt_registered(&m_a, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, 9101).unwrap();
        let rb = encrypt_registered(&m_b, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, 9102).unwrap();

        let (r0, r1) = ctx.multiply_relin_rescale(&ra.c0, &ra.c1, &rb.c0, &rb.c1).expect("multiply failed");
        let survivors = size_q - 1;
        let s_full = ctx.secret_key();
        let s_survivors: Vec<u64> = s_full[..survivors as usize * n as usize].to_vec();
        let mods_survivors = &mods[..survivors as usize];
        let roots_survivors = &roots[..survivors as usize];
        let m_result = decrypt(&r0, &r1, &s_survivors, n, mods_survivors, roots_survivors).expect("decrypt failed");
        let (re, im) = decode(&m_result, n, delta * delta / mods[size_q as usize - 1] as f64).expect("decode failed");

        for (i, (&got, &exp)) in re[..expected.len()].iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "slot {i}: got={got} expected={exp}");
        }
        for (i, &v) in im[..expected.len()].iter().enumerate() {
            assert!(v.abs() < 1e-6, "slot {i}: unexpected imaginary component {v}");
        }
        eprintln!(
            "real_encrypted_evalmult_via_registered_path: used_registered(a)={} used_registered(b)={}",
            ra.used_registered, rb.used_registered
        );
    }

    /// GOD PROMPT "CKKS PHASE 22" Section 7, item D: steady-state
    /// `encrypt` vs `encrypt_registered` latency THROUGH THE ACTUAL
    /// RUST/FFI API (not the raw C++ call Phase 20 measured) -- tests
    /// whether the FFI boundary itself (the `Vec` allocations/
    /// `memcpy`s on both sides of `gpufhe_encrypt`/
    /// `gpufhe_encrypt_registered`) adds meaningful overhead on top
    /// of the C++-measured numbers. Not a pass/fail correctness test
    /// (that is every OTHER test in this module) -- a timing
    /// diagnostic, run with `--nocapture` to see the numbers; use
    /// `cargo test ... -- --ignored` is NOT needed since this stays
    /// fast enough (n=8192, 20 steady-state reps) to run by default.
    #[test]
    fn rust_ffi_steady_state_timing_production_vs_registered() {
        let f = make_fixture(8192, 8, 7777);
        let m = test_message(f.n, 99);
        // Warm-up: pays CUDA context init + first-key registration for
        // BOTH paths before any timed sample, so neither path's first
        // measured call is confounded by a one-time cost the other
        // path already paid (the same "first-use vs steady-state"
        // separation Phase 19/20 established).
        let _ = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 1).unwrap();
        let _ = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 1).unwrap();

        let reps = 20u64;
        let mut prod_us = Vec::with_capacity(reps as usize);
        let mut reg_us = Vec::with_capacity(reps as usize);
        for i in 0..reps {
            let t0 = std::time::Instant::now();
            let _ = encrypt(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 2000 + i).unwrap();
            prod_us.push(t0.elapsed().as_secs_f64() * 1e6);

            let t0 = std::time::Instant::now();
            let r = encrypt_registered(&m, f.n, &f.pk_a, &f.pk_b, &f.mods, &f.roots, 1, 3.19, 3000 + i).unwrap();
            reg_us.push(t0.elapsed().as_secs_f64() * 1e6);
            assert!(r.used_registered, "key should still be registered at this point");
        }
        let median = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let prod_median = median(&mut prod_us.clone());
        let reg_median = median(&mut reg_us.clone());
        eprintln!(
            "rust_ffi_steady_state: production_median_us={prod_median:.2} registered_median_us={reg_median:.2} ratio={:.3}",
            reg_median / prod_median
        );
        eprintln!("  production samples: {prod_us:.2?}");
        eprintln!("  registered samples: {reg_us:.2?}");
    }
}

