//! Third ported operation (GOD PROMPT "CKKS RUST + CUDA PHASE 3"): full
//! Hybrid key switching, `gpufhe::keyswitch_resident` in
//! `src/keyswitch_resident.cpp`.
//!
//! ## Why this operation is NOT abstracted the way `ntt`/`rescale` were
//!
//! `NttContext` and `RescaleContext` rebuild their OWN input-independent
//! state (root tables, fused-rescale constants) in Rust/via a one-function
//! FFI call, because that state is pure bookkeeping or a single genuine
//! number-theoretic computation. Key switching is different in kind: its
//! input-independent state IS an evaluation key -- `av`/`bv` in
//! `KeySwitchConstants`, produced by `gpufhe::evalkeygen_host`, which in
//! turn needs a secret/public key pair, a full modulus chain (Q and the
//! auxiliary P), and several precomputed constant FAMILIES (decompose
//! `QHatInv`/`QHatModp`, ModDown `PHatInv`/`PHatModq`, Barrett constants).
//! None of that is bookkeeping a Rust wrapper should re-derive -- it is the
//! actual cryptographic key-generation pipeline. So this module does NOT
//! reconstruct any of it: [`KeySwitchContext::new`] calls into ONE opaque
//! C++ bridge function (`gpufhe_ks_fixture_create`, `cpp_bridge/
//! keyswitch_bridge.cpp`) that runs the EXISTING, unmodified native
//! (OpenFHE-free) pipeline end to end --
//! `native_primes`/`native_root` (`src/params.cpp`) for the modulus chain,
//! `keygen_host` (`src/keygen.cpp`) for the secret/public key,
//! `compute_keyswitch_constants` (`src/ks_constants.cpp`) for the constant
//! families, `evalkeygen_host` (`src/evalkeygen.cpp`, a RELIN-style eval
//! key: "sOld = s^2" per its own source comment) for `av`/`bv`, then
//! `ks_context_create`/`ks_work_create` (`src/keyswitch_resident.cpp`) for
//! the device-resident context and scratch -- and returns a single opaque
//! handle. This is the SAME exact recipe already used by
//! `tests/test_keyswitch_core_native_diff.cpp` and (with OpenFHE-sourced
//! moduli instead) `tests/test_ks_resident.cpp`; re-run this round to
//! confirm it still executes before this module was written.
//!
//! ## Ownership
//!
//! A [`KeySwitchContext`] owns exactly two things: the opaque C++ handle
//! (the eval key, device-resident context, AND per-call scratch
//! `DeviceKSWork` all live inside it, freed together by
//! `gpufhe_ks_fixture_destroy` in `Drop`) and three Rust-owned
//! [`DeviceBuf`]s for the operation's own input/output (`d_a`, `d_ba0`,
//! `d_ba1`) -- allocated ONCE in `new`, reused by every [`KeySwitchContext::keyswitch`]
//! call, the same setup/hot-path split `NttContext`/`RescaleContext`
//! established. The C++-owned `DeviceKSWork` scratch inside the opaque
//! handle is mutable, per-call state, exactly like the hot-path buffers --
//! calling `keyswitch` concurrently from two threads on the SAME context
//! would race on it, which is why `KeySwitchContext` is not `Send`/`Sync`
//! (enforced implicitly: it holds a raw `*mut c_void`) and this module
//! makes no concurrency claim to test.
//!
//! ## Correctness evidence, and its honest limits
//!
//! [`KeySwitchContext::keyswitch`] (the device-resident path,
//! `keyswitch_resident`) is checked against
//! [`KeySwitchContext::keyswitch_reference`] (the host-orchestrated
//! `keyswitch_core_resident`, exposed via the SAME opaque fixture so both
//! calls see the IDENTICAL `KeySwitchConstants`) for bit-exact agreement.
//! This is the established methodology for this exact operation
//! (`tests/test_ks_resident.cpp` does the same comparison, at a different,
//! OpenFHE-sourced configuration) -- but it is NOT a from-scratch
//! independent mathematical reference the way `rescale`'s RNS-identity
//! check was: both functions implement the same high-level Hybrid
//! key-switch algorithm, so a bug shared by both would not be caught here.
//! The genuinely independent link in the chain is `keyswitch_core_resident`
//! itself, proven bit-exact against OpenFHE's own `KeySwitchCore` in
//! `tests/test_keyswitch.cpp` -- but ONLY at an OpenFHE-sourced
//! configuration (n=32768), never at this module's native, OpenFHE-free one
//! (n=1024, matching `test_keyswitch_core_native_diff.cpp`'s parameters).
//! A prior round (`research/EXPERIMENT_LEDGER.md` entry #30) found that the
//! obvious algebraic shortcut for an independent native-config check
//! (`ba0+ba1*s =?= a*s_old` mod q) is NOT a valid identity for this
//! eval-key convention -- it fails even on OpenFHE-proven-correct output --
//! so this module does not attempt it either. **What this module's tests
//! actually prove**: `keyswitch_resident`'s GPU-resident path agrees
//! bit-exact with `keyswitch_core_resident`'s host path for a fully native
//! key, at this specific small configuration, across random/boundary
//! inputs and repeated calls on a reused context. What they do NOT prove:
//! that either implementation is correct relative to an independent
//! cryptographic reference at THIS configuration -- that remains open,
//! exactly like `test_polyeval`'s multi-level noise question (unrelated,
//! also still open) is not resolved by anything in this module.
use crate::util::cuda_err;
use ckks_core::{CkksError, RingDimension};
use cudarc::runtime::result as rt;
use std::ffi::{c_char, c_void, CStr};

extern "C" {
    fn gpufhe_ks_fixture_create(
        n: u32,
        size_q: u32,
        size_p: u32,
        num_part: u32,
        seed: u64,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> *mut c_void;
    fn gpufhe_ks_fixture_destroy(handle: *mut c_void);
    fn gpufhe_ks_fixture_size_ql(handle: *mut c_void) -> u32;
    fn gpufhe_ks_fixture_size_p(handle: *mut c_void) -> u32;
    fn gpufhe_ks_fixture_num_part(handle: *mut c_void) -> u32;
    fn gpufhe_ks_fixture_modulus_q(handle: *mut c_void, tower: u32) -> u64;
    fn gpufhe_ks_fixture_secret_key(handle: *mut c_void, out: *mut u64, out_len: usize) -> bool;
    fn gpufhe_ks_fixture_public_key_a(handle: *mut c_void, out: *mut u64, out_len: usize) -> bool;
    fn gpufhe_ks_fixture_public_key_b(handle: *mut c_void, out: *mut u64, out_len: usize) -> bool;
    fn gpufhe_ks_fixture_run_resident(
        handle: *mut c_void,
        d_a: *const u64,
        d_ba0: *mut u64,
        d_ba1: *mut u64,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;
    fn gpufhe_ks_fixture_run_reference(
        handle: *mut c_void,
        a: *const u64,
        a_len: usize,
        ba0_out: *mut u64,
        ba1_out: *mut u64,
        out_len: usize,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> i32;
    fn gpufhe_ks_fixture_eval_key_part(
        handle: *mut c_void,
        part: u32,
        out_av: *mut u64,
        out_bv: *mut u64,
        out_len: usize,
    ) -> bool;
    fn gpufhe_ks_fixture_eval_key_towers(handle: *mut c_void) -> u32;
    fn gpufhe_ks_fixture_alpha(handle: *mut c_void) -> u32;
}

const ERR_BUF_LEN: usize = 512;

fn err_from_buf(buf: &[u8]) -> String {
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    unsafe { CStr::from_ptr(buf[..=nul.min(buf.len() - 1)].as_ptr() as *const c_char) }
        .to_string_lossy()
        .into_owned()
}

use crate::util::DeviceBuf;

/// Reusable key-switch execution context for one fixed, fully native
/// (OpenFHE-free) key: an `n`-dimension ring, `size_q` Q-towers, `size_p`
/// auxiliary P-towers, decomposed into `num_part` digit parts. See this
/// module's top-level doc comment for exactly what gets built once, in
/// `new`, versus what the hot path (`keyswitch`) does per call.
pub struct KeySwitchContext {
    handle: *mut c_void,
    n: RingDimension,
    size_ql: u32,
    size_p: u32,
    d_a: DeviceBuf,
    d_ba0: DeviceBuf,
    d_ba1: DeviceBuf,
}

impl KeySwitchContext {
    /// Builds a fresh native key-switch fixture (modulus chain, secret/eval
    /// key, device-resident context + scratch) and the per-call
    /// input/output device buffers. `num_part` must be in `1..=size_q`
    /// (validated here, BEFORE the FFI call -- `compute_keyswitch_constants`
    /// divides by `num_part` and does unchecked unsigned arithmetic on
    /// `size_q - alpha*part` internally, so an out-of-range `num_part`
    /// passed through unchecked would be a real latent bug, not merely a
    /// defensive-sounding check; found while tracing this round's call
    /// graph, not fixed in `src/ks_constants.cpp` itself -- out of this
    /// round's scope, guarded here instead).
    pub fn new(
        n: RingDimension,
        size_q: u32,
        size_p: u32,
        num_part: u32,
        seed: u64,
    ) -> Result<Self, CkksError> {
        if size_q == 0 {
            return Err(CkksError::InvalidRingDimension { n: size_q as u64 });
        }
        if size_p == 0 {
            return Err(CkksError::InvalidRingDimension { n: size_p as u64 });
        }
        if num_part == 0 || num_part > size_q {
            return Err(CkksError::InvalidRingDimension { n: num_part as u64 });
        }

        rt::device::set(0).map_err(cuda_err)?;

        let mut err_buf = [0u8; ERR_BUF_LEN];
        let handle = unsafe {
            gpufhe_ks_fixture_create(
                n.get(),
                size_q,
                size_p,
                num_part,
                seed,
                err_buf.as_mut_ptr() as *mut c_char,
                ERR_BUF_LEN,
            )
        };
        if handle.is_null() {
            return Err(CkksError::Cuda(format!(
                "gpufhe_ks_fixture_create failed: {}",
                err_from_buf(&err_buf)
            )));
        }

        let size_ql = unsafe { gpufhe_ks_fixture_size_ql(handle) };
        let size_p_actual = unsafe { gpufhe_ks_fixture_size_p(handle) };
        let bytes = n.get() as usize * 8 * size_ql as usize;

        let d_a = DeviceBuf::alloc(bytes).inspect_err(|_| unsafe {
            gpufhe_ks_fixture_destroy(handle);
        })?;
        let d_ba0 = DeviceBuf::alloc(bytes).inspect_err(|_| unsafe {
            gpufhe_ks_fixture_destroy(handle);
        })?;
        let d_ba1 = DeviceBuf::alloc(bytes).inspect_err(|_| unsafe {
            gpufhe_ks_fixture_destroy(handle);
        })?;

        Ok(KeySwitchContext {
            handle,
            n,
            size_ql,
            size_p: size_p_actual,
            d_a,
            d_ba0,
            d_ba1,
        })
    }

    pub fn ring_dimension(&self) -> RingDimension {
        self.n
    }

    pub fn size_ql(&self) -> u32 {
        self.size_ql
    }

    pub fn size_p(&self) -> u32 {
        self.size_p
    }

    pub fn num_part(&self) -> u32 {
        unsafe { gpufhe_ks_fixture_num_part(self.handle) }
    }

    /// The actual Q modulus for one tower, so a caller can generate valid
    /// in-range residues without re-deriving this fixture's own
    /// `native_primes`/`native_root` choices.
    pub fn modulus_q(&self, tower: u32) -> u64 {
        assert!(tower < self.size_ql, "tower out of range");
        unsafe { gpufhe_ks_fixture_modulus_q(self.handle, tower) }
    }

    /// The Q-restricted secret key (`size_ql * n` elements, eval domain)
    /// this fixture's eval key was built under. GOD PROMPT "CKKS PHASE
    /// 5", new -- TEST-ONLY, explicitly not a production API: exists so
    /// a caller can perform a REAL `codec::encrypt`/`codec::decrypt`
    /// round trip under the EXACT SAME key, to prove `EvalMultContext`
    /// correct as actual CKKS ciphertext arithmetic rather than only at
    /// the raw RNS level. See `cpp_bridge/keyswitch_bridge.cpp`'s
    /// `gpufhe_ks_fixture_secret_key` for the full rationale (same
    /// doc comment, not duplicated here) and its security caveats.
    pub fn secret_key(&self) -> Vec<u64> {
        let len = self.size_ql as usize * self.n.get() as usize;
        let mut out = vec![0u64; len];
        let ok = unsafe { gpufhe_ks_fixture_secret_key(self.handle, out.as_mut_ptr(), len) };
        assert!(ok, "gpufhe_ks_fixture_secret_key: length mismatch (internal bug)");
        out
    }

    /// The Q-restricted public key component `a` (`size_ql * n`
    /// elements, eval domain). See [`Self::secret_key`]'s doc comment.
    pub fn public_key_a(&self) -> Vec<u64> {
        let len = self.size_ql as usize * self.n.get() as usize;
        let mut out = vec![0u64; len];
        let ok = unsafe { gpufhe_ks_fixture_public_key_a(self.handle, out.as_mut_ptr(), len) };
        assert!(ok, "gpufhe_ks_fixture_public_key_a: length mismatch (internal bug)");
        out
    }

    /// The Q-restricted public key component `b`. See
    /// [`Self::secret_key`]'s doc comment.
    pub fn public_key_b(&self) -> Vec<u64> {
        let len = self.size_ql as usize * self.n.get() as usize;
        let mut out = vec![0u64; len];
        let ok = unsafe { gpufhe_ks_fixture_public_key_b(self.handle, out.as_mut_ptr(), len) };
        assert!(ok, "gpufhe_ks_fixture_public_key_b: length mismatch (internal bug)");
        out
    }

    /// GOD PROMPT "CKKS PHASE 7": the RAW eval key component
    /// `(av[part], bv[part])`, full QP range (`eval_key_towers() * n`
    /// elements each, eval domain) -- TEST-ONLY. Bypasses
    /// `keyswitch`/`keyswitch_reference` entirely, exposing exactly
    /// what `evalkeygen_host` generated, to check its OWN claimed
    /// decryption identity (`bv[part]-av[part]*s == P*s2` in part's
    /// Q-range + small noise, per `src/evalkeygen.cpp`'s doc comment)
    /// directly, isolated from the accumulate/decompose/moddown logic.
    pub fn eval_key_part(&self, part: u32) -> (Vec<u64>, Vec<u64>) {
        let len = self.eval_key_towers() as usize * self.n.get() as usize;
        let mut av = vec![0u64; len];
        let mut bv = vec![0u64; len];
        let ok = unsafe { gpufhe_ks_fixture_eval_key_part(self.handle, part, av.as_mut_ptr(), bv.as_mut_ptr(), len) };
        assert!(ok, "gpufhe_ks_fixture_eval_key_part: length mismatch or invalid part (internal bug)");
        (av, bv)
    }

    /// Tower count of the eval key's own QP range (`size_ql + size_p`,
    /// confirmed via the accessor rather than assumed).
    pub fn eval_key_towers(&self) -> u32 {
        unsafe { gpufhe_ks_fixture_eval_key_towers(self.handle) }
    }

    /// Per-part tower span (`alpha`) `compute_keyswitch_constants`
    /// derived -- so a caller can locate each part's own Q-range
    /// without re-deriving `ceil(size_ql/alpha)` itself.
    pub fn alpha(&self) -> u32 {
        unsafe { gpufhe_ks_fixture_alpha(self.handle) }
    }

    /// Steady-state per-call path: upload `a` (eval domain, `size_ql * n`
    /// elements), run `keyswitch_resident` on the reused device context +
    /// scratch, download `ba0`/`ba1` (each `size_ql * n` elements).
    pub fn keyswitch(&self, a: &[u64], ba0: &mut [u64], ba1: &mut [u64]) -> Result<(), CkksError> {
        let expected = self.size_ql as usize * self.n.get() as usize;
        if a.len() != expected {
            return Err(CkksError::InvalidRingDimension { n: a.len() as u64 });
        }
        if ba0.len() != expected || ba1.len() != expected {
            return Err(CkksError::InvalidRingDimension {
                n: ba0.len().max(ba1.len()) as u64,
            });
        }

        self.d_a.upload_u64(a)?;

        let mut err_buf = [0u8; ERR_BUF_LEN];
        let rc = unsafe {
            gpufhe_ks_fixture_run_resident(
                self.handle,
                self.d_a.as_const_u64(),
                self.d_ba0.as_mut_u64(),
                self.d_ba1.as_mut_u64(),
                err_buf.as_mut_ptr() as *mut c_char,
                ERR_BUF_LEN,
            )
        };
        if rc != 0 {
            return Err(CkksError::Cuda(format!(
                "keyswitch_resident failed (code {rc}): {}",
                err_from_buf(&err_buf)
            )));
        }

        self.d_ba0.download_u64(ba0)?;
        self.d_ba1.download_u64(ba1)?;
        Ok(())
    }

    /// Device-resident hot path for composition into a LARGER fused
    /// operation (GOD PROMPT "CKKS RUST + CUDA PHASE 4": `evalmult`'s
    /// relinearization step, which key-switches the tensor product's
    /// third component without ever bringing it back to the host). Unlike
    /// [`Self::keyswitch`], this does NOT upload/download through this
    /// context's own `d_a`/`d_ba0`/`d_ba1` buffers -- it forwards the
    /// caller's OWN device pointers straight to `keyswitch_resident` via
    /// the SAME bridge call `keyswitch` uses. This is the "adaptation"
    /// the governing directive asks for instead of assuming `keyswitch`'s
    /// host-slice-oriented signature is sufficient as-is: a caller already
    /// holding device memory (as `evalmult::EvalMultContext` does) would
    /// otherwise pay two pointless host round trips per relinearization.
    ///
    /// # Safety
    /// `d_a` must be a valid device pointer with at least `size_ql() *
    /// ring_dimension()` readable `u64` elements; `d_ba0`/`d_ba1` the same,
    /// writable, and non-overlapping with `d_a` or each other (the
    /// underlying kernels assume distinct input/output buffers, same as
    /// `keyswitch_resident`'s own contract in `src/keyswitch_resident.cpp`).
    /// This call synchronizes the default stream itself (see
    /// `gpufhe_ks_fixture_run_resident`'s own doc comment in
    /// `cpp_bridge/keyswitch_bridge.cpp`) before returning, so the caller
    /// may safely queue further default-stream work on `d_ba0`/`d_ba1`
    /// immediately after.
    pub(crate) unsafe fn keyswitch_device_resident(
        &self,
        d_a: *const u64,
        d_ba0: *mut u64,
        d_ba1: *mut u64,
    ) -> Result<(), CkksError> {
        let mut err_buf = [0u8; ERR_BUF_LEN];
        let rc = gpufhe_ks_fixture_run_resident(
            self.handle,
            d_a,
            d_ba0,
            d_ba1,
            err_buf.as_mut_ptr() as *mut c_char,
            ERR_BUF_LEN,
        );
        if rc != 0 {
            return Err(CkksError::Cuda(format!(
                "keyswitch_resident (device-resident) failed (code {rc}): {}",
                err_from_buf(&err_buf)
            )));
        }
        Ok(())
    }

    /// Host-orchestrated reference path (`keyswitch_core_resident`) on the
    /// SAME `KeySwitchConstants` this context holds. NOT the fast path --
    /// exists only as this module's correctness oracle (see the module doc
    /// comment for the honest limits of what agreement with this proves).
    pub fn keyswitch_reference(&self, a: &[u64]) -> Result<(Vec<u64>, Vec<u64>), CkksError> {
        let expected = self.size_ql as usize * self.n.get() as usize;
        if a.len() != expected {
            return Err(CkksError::InvalidRingDimension { n: a.len() as u64 });
        }
        let mut ba0 = vec![0u64; expected];
        let mut ba1 = vec![0u64; expected];
        let mut err_buf = [0u8; ERR_BUF_LEN];
        let rc = unsafe {
            gpufhe_ks_fixture_run_reference(
                self.handle,
                a.as_ptr(),
                a.len(),
                ba0.as_mut_ptr(),
                ba1.as_mut_ptr(),
                expected,
                err_buf.as_mut_ptr() as *mut c_char,
                ERR_BUF_LEN,
            )
        };
        if rc != 0 {
            return Err(CkksError::Cuda(format!(
                "keyswitch_core_resident reference failed (code {rc}): {}",
                err_from_buf(&err_buf)
            )));
        }
        Ok((ba0, ba1))
    }
}

impl Drop for KeySwitchContext {
    fn drop(&mut self) {
        unsafe {
            gpufhe_ks_fixture_destroy(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    // Same configuration as tests/test_keyswitch_core_native_diff.cpp
    // ("mirror test_polyeval's exact first-level parameters"): n=1024,
    // sizeQ=8 (1x60-bit + 7x50-bit), sizeP=2, numPart=2 -- a known-working
    // native parameter set, re-confirmed this round by re-running that
    // existing test before writing any Rust code against it.
    const N: u32 = 1024;
    const SIZE_Q: u32 = 8;
    const SIZE_P: u32 = 2;
    const NUM_PART: u32 = 2;

    fn new_ctx(seed: u64) -> KeySwitchContext {
        let n = RingDimension::new(N).unwrap();
        KeySwitchContext::new(n, SIZE_Q, SIZE_P, NUM_PART, seed).expect("context init failed")
    }

    fn random_input(ctx: &KeySwitchContext, rng: &mut impl Rng) -> Vec<u64> {
        let nn = ctx.ring_dimension().get() as usize;
        let mut a = vec![0u64; ctx.size_ql() as usize * nn];
        for t in 0..ctx.size_ql() {
            let q = ctx.modulus_q(t);
            for k in 0..nn {
                a[t as usize * nn + k] = rng.gen_range(0..q);
            }
        }
        a
    }

    #[test]
    fn keyswitch_matches_host_reference_random_data() {
        let ctx = new_ctx(4242);
        let expected_len = ctx.size_ql() as usize * ctx.ring_dimension().get() as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123456);

        for trial in 0..3 {
            let a = random_input(&ctx, &mut rng);
            let mut ba0 = vec![0u64; expected_len];
            let mut ba1 = vec![0u64; expected_len];
            ctx.keyswitch(&a, &mut ba0, &mut ba1).expect("keyswitch failed");
            let (ref0, ref1) = ctx.keyswitch_reference(&a).expect("reference failed");
            assert_eq!(ba0, ref0, "trial {trial}: ba0 mismatch vs host reference");
            assert_eq!(ba1, ref1, "trial {trial}: ba1 mismatch vs host reference");
        }
    }

    #[test]
    fn keyswitch_matches_reference_reused_context_repeated_calls() {
        // Calls keyswitch() SEVERAL times on the SAME context with
        // DIFFERENT random inputs, proving the reused DeviceKSWork scratch
        // (owned by the opaque C++ fixture) does not leak state between
        // calls.
        let ctx = new_ctx(777);
        let expected_len = ctx.size_ql() as usize * ctx.ring_dimension().get() as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        for call in 0..5 {
            let a = random_input(&ctx, &mut rng);
            let mut ba0 = vec![0u64; expected_len];
            let mut ba1 = vec![0u64; expected_len];
            ctx.keyswitch(&a, &mut ba0, &mut ba1).expect("keyswitch failed");
            let (ref0, ref1) = ctx.keyswitch_reference(&a).expect("reference failed");
            assert_eq!(ba0, ref0, "call {call}: ba0 mismatch on reused context");
            assert_eq!(ba1, ref1, "call {call}: ba1 mismatch on reused context");
        }
    }

    #[test]
    fn keyswitch_matches_reference_on_boundary_inputs() {
        // Zero input and max-residue (q-1 everywhere) input -- deterministic
        // boundary cases, not just random ones.
        let ctx = new_ctx(31415);
        let nn = ctx.ring_dimension().get() as usize;
        let expected_len = ctx.size_ql() as usize * nn;

        for &boundary_max in &[false, true] {
            let mut a = vec![0u64; expected_len];
            if boundary_max {
                for t in 0..ctx.size_ql() {
                    let q = ctx.modulus_q(t);
                    for k in 0..nn {
                        a[t as usize * nn + k] = q - 1;
                    }
                }
            }
            let mut ba0 = vec![0u64; expected_len];
            let mut ba1 = vec![0u64; expected_len];
            ctx.keyswitch(&a, &mut ba0, &mut ba1).expect("keyswitch failed");
            let (ref0, ref1) = ctx.keyswitch_reference(&a).expect("reference failed");
            assert_eq!(ba0, ref0, "boundary_max={boundary_max}: ba0 mismatch");
            assert_eq!(ba1, ref1, "boundary_max={boundary_max}: ba1 mismatch");
        }
    }

    #[test]
    fn invalid_input_length_is_rejected() {
        let ctx = new_ctx(1);
        let bad_a = vec![0u64; 10]; // way too short
        let mut ba0 = vec![0u64; ctx.size_ql() as usize * ctx.ring_dimension().get() as usize];
        let mut ba1 = ba0.clone();
        let err = ctx.keyswitch(&bad_a, &mut ba0, &mut ba1);
        assert!(err.is_err(), "undersized input must be rejected, not silently truncated/overrun");

        let valid_a = vec![0u64; ctx.size_ql() as usize * ctx.ring_dimension().get() as usize];
        let mut bad_ba0 = vec![0u64; 3];
        let err2 = ctx.keyswitch(&valid_a, &mut bad_ba0, &mut ba1);
        assert!(err2.is_err(), "undersized output buffer must be rejected");
    }

    #[test]
    fn invalid_num_part_is_rejected_at_construction() {
        let n = RingDimension::new(N).unwrap();
        assert!(
            KeySwitchContext::new(n, SIZE_Q, SIZE_P, 0, 1).is_err(),
            "num_part=0 must be rejected (compute_keyswitch_constants divides by it)"
        );
        assert!(
            KeySwitchContext::new(n, SIZE_Q, SIZE_P, SIZE_Q + 1, 1).is_err(),
            "num_part > size_q must be rejected (unsigned underflow inside compute_keyswitch_constants otherwise)"
        );
    }

    #[test]
    fn invalid_dimensions_are_rejected_at_construction() {
        let n = RingDimension::new(N).unwrap();
        assert!(KeySwitchContext::new(n, 0, SIZE_P, 1, 1).is_err(), "size_q=0 must be rejected");
        assert!(KeySwitchContext::new(n, SIZE_Q, 0, 1, 1).is_err(), "size_p=0 must be rejected");
    }
}
