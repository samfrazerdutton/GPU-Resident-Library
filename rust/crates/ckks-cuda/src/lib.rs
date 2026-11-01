//! Rust host orchestration over this project's EXISTING, proven CUDA C++
//! NTT/INTT kernels (`kernels/ntt_ct.cu`/`kernels/intt_ct.cu`, reused
//! unmodified via FFI -- see `build.rs`). This is Path B of the GOD
//! PROMPT "CKKS RUST + CUDA EVOLUTION" three-way comparison: Rust host +
//! the SAME CUDA C++ kernel, isolating host-orchestration cost/benefit
//! from the GPU arithmetic itself (which stays byte-for-byte the kernel
//! already proven correct by `tests/test_ntt_roundtrip_n1024.cpp`,
//! 56/56 cases, in the main C++ project).
//!
//! No cryptographic algorithm is reimplemented here: root-table
//! construction below is plain modular-exponentiation/bit-reversal
//! bookkeeping (the same bookkeeping `tests/test_ntt_roundtrip_n1024.cpp`
//! itself performs on the host side, in C++, to drive the SAME kernels --
//! this is NOT a parallel algorithm implementation, it's the identical
//! preprocessing step reimplemented in a second language so it can be
//! independently cross-checked). Primitive-root search (a nontrivial
//! number-theoretic search, not just bookkeeping) is NOT reimplemented --
//! it calls the existing, unmodified `gpufhe::native_root` via the tiny
//! FFI bridge in `cpp_bridge/native_root_bridge.cpp`.
//!
//! ## Setup vs. hot path (read this before benchmarking or extending)
//!
//! An earlier draft of this crate did `cudaSetDevice`, a primitive-root
//! FFI call, AND full host-side table rebuilding on EVERY call to a
//! single `ntt_roundtrip(coeffs, n, q)` function. That made any A/B
//! comparison against a C++ path that reuses its tables an apples-to-
//! oranges measurement of two DIFFERENT workloads, not a host-
//! orchestration comparison -- a real methodological bug, caught before
//! any performance number was trusted (see
//! `research/EXPERIMENT_LEDGER.md` for the exact round this happened in).
//!
//! The fix is [`NttContext`]: it does every reusable, input-independent
//! step exactly ONCE in [`NttContext::new`] --
//!   - `cudaSetDevice`
//!   - the primitive-root search (FFI)
//!   - host-side forward/inverse root-and-Shoup-precon table construction
//!   - `cudaMalloc` for every buffer the roundtrip needs, table buffers
//!     included
//!   - uploading the (immutable, input-independent) tables to the device
//!     ONCE
//! -- so that [`NttContext::roundtrip`] does ONLY input-dependent work:
//! upload this call's coefficients into the already-allocated scratch
//! buffer, launch the two kernels, synchronize, download. That is the
//! steady-state, per-operation cost a real caller doing many round trips
//! actually pays.
//!
//! [`ntt_roundtrip`] still exists as a one-shot convenience wrapper
//! (build a context, run one round trip, tear it down) -- it is
//! correctness-equivalent but NOT the function to benchmark for steady-
//! state throughput; its doc comment says so.

use ckks_core::{CkksError, Modulus, RingDimension};
use cudarc::runtime::result as rt;
use std::ffi::c_void;

extern "C" {
    // Exact signatures from include/ntt.h / include/intt.h, unmodified.
    fn LaunchNTT_CT(
        d_x: *mut u64,
        d_roots: *const u64,
        d_precon: *const u64,
        n: u32,
        q: u64,
        s: *mut c_void,
    );
    fn LaunchINTT_GS(
        d_x: *mut u64,
        d_invroots: *const u64,
        d_invprecon: *const u64,
        n: u32,
        q: u64,
        n_inv: u64,
        n_inv_precon: u64,
        s: *mut c_void,
    );
    // From cpp_bridge/native_root_bridge.cpp.
    fn gpufhe_native_root(n: u32, q: u64) -> u64;
    // Standard CUDA runtime symbol, called directly rather than relying on
    // cudarc::runtime::result::stream::synchronize's implicit-null-stream
    // typing, to make the explicit "wait for both kernels before reading
    // results back" step unambiguous to a future reader.
    fn cudaDeviceSynchronize() -> i32;
}

fn cuda_err(e: impl std::fmt::Debug) -> CkksError {
    CkksError::Cuda(format!("{e:?}"))
}

/// Finds a primitive 2n-th root of unity mod q by calling the existing,
/// unmodified `gpufhe::native_root` C++ implementation. Deliberately NOT
/// reimplemented in Rust -- see this module's own doc comment. This is a
/// genuine search (not O(1) bookkeeping); callers doing many round trips
/// at the same (n, q) should call it once via [`NttContext::new`], not
/// per operation.
pub fn native_root(n: RingDimension, q: Modulus) -> u64 {
    unsafe { gpufhe_native_root(n.get(), q.get()) }
}

/// Performs exactly the host-only work [`NttContext::new`] does before it
/// ever touches the GPU (primitive-root search + both forward/inverse
/// table builds), and returns a cheap checksum so the optimizer cannot
/// discard the work. Exists so a benchmark can measure this "cold-start"
/// cost on its own, symmetrically with the equivalent C++ measurement in
/// `bench/bench_ntt_roundtrip_cpp_vs_rust.cpp`'s cold-start tier -- without
/// this, a benchmark calling only [`native_root`] would be timing LESS
/// work than the C++ side's cold-start tier and the comparison would be
/// unfair again, the same mistake this module's own top doc comment
/// describes fixing once already.
pub fn cold_start_cost_probe(n: RingDimension, q: Modulus) -> u64 {
    let nn = n.get();
    let qq = q.get();
    let root = native_root(n, q);
    let fwd = build_tables(nn, qq, root, false);
    let inv = build_tables(nn, qq, root, true);
    fwd.roots[0]
        .wrapping_add(fwd.precon[0])
        .wrapping_add(inv.roots[0])
        .wrapping_add(inv.precon[0])
}

fn powmod(b: u64, e: u64, q: u64) -> u64 {
    let qq = q as u128;
    let mut r: u128 = 1 % qq;
    let mut bb: u128 = (b as u128) % qq;
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            r = (r * bb) % qq;
        }
        bb = (bb * bb) % qq;
        e >>= 1;
    }
    r as u64
}

fn invmod(a: u64, q: u64) -> u64 {
    powmod(a % q, q - 2, q)
}

fn bitrev(mut v: u32, bits: u32) -> u32 {
    let mut r = 0u32;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

struct Tables {
    roots: Vec<u64>,
    precon: Vec<u64>,
}

/// Builds the (root power, Shoup-precon) table for one NTT direction.
/// Matches `tests/test_ntt_roundtrip_n1024.cpp`'s `build_tables` and
/// `kernels/ntt_ct.cu`/`intt_ct.cu`'s own documented convention exactly:
/// standard order in, bit-reversed out for forward; the mirror for
/// inverse (via `invmod(root)` as the base). O(n) host-side work --
/// input-independent, so [`NttContext::new`] does this once, not per call.
fn build_tables(n: u32, q: u64, root: u64, inverse: bool) -> Tables {
    let logn = n.trailing_zeros();
    let base = if inverse { invmod(root, q) } else { root };
    let nu = n as usize;
    let mut pw = vec![0u64; nu];
    pw[0] = 1 % q;
    for i in 1..nu {
        pw[i] = ((pw[i - 1] as u128 * base as u128) % q as u128) as u64;
    }
    let mut roots = vec![0u64; nu];
    let mut precon = vec![0u64; nu];
    for i in 0..nu {
        let r = pw[bitrev(i as u32, logn) as usize];
        roots[i] = r;
        precon[i] = (((r as u128) << 64) / q as u128) as u64;
    }
    Tables { roots, precon }
}

/// RAII guard for one `cudaMalloc`'d buffer: frees exactly once, on every
/// exit path (including an early `?` return or a panic unwind, since
/// `Drop` still runs), never double-frees. This is the full extent of
/// this crate's safety claim for device memory -- synchronization
/// correctness (not reading/writing a buffer while a kernel that touches
/// it is still in flight) is the CALLER's responsibility within this
/// module, enforced here by always calling `cudaDeviceSynchronize()`
/// before any host-visible use of a result.
struct DeviceBuf {
    ptr: *mut c_void,
}

impl DeviceBuf {
    fn alloc(bytes: usize) -> Result<Self, CkksError> {
        let ptr = unsafe { rt::malloc_sync(bytes).map_err(cuda_err)? };
        Ok(DeviceBuf { ptr })
    }

    fn upload_u64(&self, host: &[u64]) -> Result<(), CkksError> {
        unsafe { rt::memcpy_htod_sync(self.ptr, host).map_err(cuda_err) }
    }

    fn download_u64(&self, host: &mut [u64]) -> Result<(), CkksError> {
        unsafe { rt::memcpy_dtoh_sync(host, self.ptr).map_err(cuda_err) }
    }

    fn as_mut_u64(&self) -> *mut u64 {
        self.ptr as *mut u64
    }

    fn as_const_u64(&self) -> *const u64 {
        self.ptr as *const u64
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        // Best-effort free; a failure here (e.g. context already torn
        // down) cannot be reported from Drop, so it is deliberately
        // swallowed rather than panicking during unwind.
        unsafe {
            let _ = rt::free_sync(self.ptr);
        }
    }
}

/// Reusable NTT execution context for one fixed `(n, q)`. Owns everything
/// that round trip after round trip stays the same: the device-selection
/// call, the forward/inverse root-and-precon tables (host-built once,
/// uploaded to the device once), and a scratch device buffer sized for
/// `n` coefficients. See this module's top-level doc comment for why this
/// split exists.
///
/// Not `Sync`/`Send` on purpose for now: the scratch buffer is reused
/// in-place by every [`NttContext::roundtrip`] call, so concurrent use
/// from multiple threads without additional synchronization would be a
/// data race. A thread-safe variant (e.g. one scratch buffer per thread,
/// or an internal mutex) is a reasonable future extension once a real
/// concurrent workload needs it -- not built speculatively here.
pub struct NttContext {
    n: RingDimension,
    q: Modulus,
    n_inv: u64,
    n_inv_precon: u64,
    d_fr: DeviceBuf,
    d_fp: DeviceBuf,
    d_ir: DeviceBuf,
    d_ip: DeviceBuf,
    d_x: DeviceBuf,
}

impl NttContext {
    /// Performs every step that does NOT depend on the per-call input:
    /// device selection, primitive-root search, host-side table
    /// construction, device allocation, and table upload. This is the
    /// "cold start" / "initialization" cost a caller pays once per
    /// `(n, q)`, not per operation -- see the benchmark in
    /// `examples/bench_roundtrip.rs` for how this is measured separately
    /// from steady-state per-call cost.
    pub fn new(n: RingDimension, q: Modulus) -> Result<Self, CkksError> {
        rt::device::set(0).map_err(cuda_err)?;

        let nn = n.get();
        let qq = q.get();
        let root = native_root(n, q);

        let fwd = build_tables(nn, qq, root, false);
        let inv = build_tables(nn, qq, root, true);
        let n_inv = invmod(nn as u64, qq);
        let n_inv_precon = (((n_inv as u128) << 64) / qq as u128) as u64;

        let bytes = nn as usize * 8;
        let d_fr = DeviceBuf::alloc(bytes)?;
        d_fr.upload_u64(&fwd.roots)?;
        let d_fp = DeviceBuf::alloc(bytes)?;
        d_fp.upload_u64(&fwd.precon)?;
        let d_ir = DeviceBuf::alloc(bytes)?;
        d_ir.upload_u64(&inv.roots)?;
        let d_ip = DeviceBuf::alloc(bytes)?;
        d_ip.upload_u64(&inv.precon)?;
        let d_x = DeviceBuf::alloc(bytes)?;

        Ok(NttContext {
            n,
            q,
            n_inv,
            n_inv_precon,
            d_fr,
            d_fp,
            d_ir,
            d_ip,
            d_x,
        })
    }

    pub fn ring_dimension(&self) -> RingDimension {
        self.n
    }

    pub fn modulus(&self) -> Modulus {
        self.q
    }

    /// Steady-state per-call path: upload `coeffs` into the context's
    /// already-allocated scratch buffer, run forward NTT then inverse
    /// NTT (the EXISTING, unmodified kernels), synchronize, and write the
    /// result into `out`. No allocation, no table rebuild, no device
    /// re-selection -- everything reusable was already done in
    /// [`NttContext::new`]. For a correct NTT/INTT pair `out` must equal
    /// `coeffs` exactly (modular equality, not a numerical
    /// approximation) -- see this crate's tests.
    pub fn roundtrip(&self, coeffs: &[u64], out: &mut [u64]) -> Result<(), CkksError> {
        let nn = self.n.get();
        if coeffs.len() != nn as usize || out.len() != nn as usize {
            return Err(CkksError::InvalidRingDimension {
                n: coeffs.len() as u64,
            });
        }

        self.d_x.upload_u64(coeffs)?;

        unsafe {
            LaunchNTT_CT(
                self.d_x.as_mut_u64(),
                self.d_fr.as_const_u64(),
                self.d_fp.as_const_u64(),
                nn,
                self.q.get(),
                std::ptr::null_mut(),
            );
            LaunchINTT_GS(
                self.d_x.as_mut_u64(),
                self.d_ir.as_const_u64(),
                self.d_ip.as_const_u64(),
                nn,
                self.q.get(),
                self.n_inv,
                self.n_inv_precon,
                std::ptr::null_mut(),
            );
            let rc = cudaDeviceSynchronize();
            if rc != 0 {
                return Err(CkksError::Cuda(format!(
                    "cudaDeviceSynchronize failed, cudaError_t={rc}"
                )));
            }
        }

        self.d_x.download_u64(out)
    }
}

/// One-shot convenience wrapper: builds a fresh [`NttContext`], runs ONE
/// round trip, tears the context down. Correctness-equivalent to
/// `NttContext::new(...)?.roundtrip(...)`, but pays the FULL cold-start
/// cost (device select + primitive-root search + table build + 5
/// `cudaMalloc`s + 4 table uploads) on every call. Do NOT use this in a
/// steady-state throughput benchmark or a hot loop -- build one
/// [`NttContext`] and call [`NttContext::roundtrip`] repeatedly instead.
pub fn ntt_roundtrip(coeffs: &[u64], n: RingDimension, q: Modulus) -> Result<Vec<u64>, CkksError> {
    let ctx = NttContext::new(n, q)?;
    let mut out = vec![0u64; n.get() as usize];
    ctx.roundtrip(coeffs, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same 8-tower modulus chain as this round's C++ work
    // (test_rescale_n1024.cpp / test_ntt_roundtrip_n1024.cpp), so results
    // are directly comparable across languages.
    const MODULI: [u64; 8] = [
        1152921504606830593,
        36028797018820609,
        36028797018802177,
        36028797018789889,
        36028797018746881,
        36028797018667009,
        36028797018652673,
        36028797018646529,
    ];

    fn cases(n: u32, q: u64, seed: u64) -> Vec<(&'static str, Vec<u64>)> {
        use rand::Rng;
        let mut rng = rand_chacha_free_rng(seed);
        let mut v = vec![
            ("zero", vec![0u64; n as usize]),
            ("sparse_dc", {
                let mut v = vec![0u64; n as usize];
                v[0] = 1;
                v
            }),
            ("sparse_mid", {
                let mut v = vec![0u64; n as usize];
                v[(n / 2) as usize] = 1;
                v
            }),
            ("dense_boundary", vec![q - 1; n as usize]),
        ];
        let dense_random: Vec<u64> = (0..n).map(|_| rng.gen_range(0..q)).collect();
        v.push(("dense_random", dense_random));
        let alternating: Vec<u64> = (0..n)
            .map(|i| if i % 2 == 0 { 0 } else { q - 1 })
            .collect();
        v.push(("alternating_boundary", alternating));
        v
    }

    // Tiny local PRNG shim so this crate's dev-dependency on `rand` stays
    // minimal and deterministic across platforms (StdRng's exact stream
    // is not guaranteed stable across rand versions, which does not
    // matter for a test that only needs SOME dense random input, not a
    // specific one).
    fn rand_chacha_free_rng(seed: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    #[test]
    fn roundtrip_exact_all_towers_one_shot() {
        let n = RingDimension::new(1024).unwrap();
        let mut total = 0;
        let mut failed = 0;
        for (ti, &q) in MODULI.iter().enumerate() {
            let modulus = Modulus::new(q).unwrap();
            for (name, input) in cases(1024, q, 777 + ti as u64) {
                total += 1;
                let out = ntt_roundtrip(&input, n, modulus)
                    .unwrap_or_else(|e| panic!("tower {ti} case {name}: {e}"));
                if out != input {
                    failed += 1;
                    let first_bad = out
                        .iter()
                        .zip(input.iter())
                        .position(|(a, b)| a != b)
                        .unwrap();
                    eprintln!(
                        "tower={ti} q={q} case={name} FAILED at idx={first_bad} got={} want={}",
                        out[first_bad], input[first_bad]
                    );
                }
            }
        }
        assert_eq!(failed, 0, "{failed}/{total} NTT round-trip cases failed");
    }

    /// Same test vectors as above, but through a SINGLE reused
    /// `NttContext` per tower (the steady-state API) instead of a fresh
    /// context per case -- proves the context's scratch buffer is
    /// correctly re-uploaded and re-read every call, with no stale state
    /// leaking between calls.
    #[test]
    fn roundtrip_exact_all_towers_reused_context() {
        let n = RingDimension::new(1024).unwrap();
        let mut total = 0;
        let mut failed = 0;
        for (ti, &q) in MODULI.iter().enumerate() {
            let modulus = Modulus::new(q).unwrap();
            let ctx = NttContext::new(n, modulus).unwrap();
            for (name, input) in cases(1024, q, 1777 + ti as u64) {
                total += 1;
                let mut out = vec![0u64; 1024];
                ctx.roundtrip(&input, &mut out)
                    .unwrap_or_else(|e| panic!("tower {ti} case {name}: {e}"));
                if out != input {
                    failed += 1;
                    eprintln!("tower={ti} q={q} case={name} FAILED (reused context)");
                }
            }
        }
        assert_eq!(
            failed, 0,
            "{failed}/{total} NTT round-trip cases failed (reused context)"
        );
    }

    #[test]
    fn native_root_matches_known_value() {
        // Cross-check against the exact value this repo's own C++ tests
        // have been using and printing all session (see
        // tests/test_rescale_n1024.cpp's diagnostics): tower 0's root is
        // not independently recorded in this crate, so this test only
        // asserts native_root returns a VALID primitive 2n-th root (its
        // order is exactly 2n), not a specific hardcoded value -- a
        // weaker but still real, self-contained check.
        let n = RingDimension::new(1024).unwrap();
        let q = Modulus::new(MODULI[0]).unwrap();
        let root = native_root(n, q);
        assert_ne!(root, 0);
        // order must be exactly 2n: root^(2n) == 1, root^n == -1 (i.e. q-1)
        let pow2n = powmod(root, 2 * n.get() as u64, q.get());
        let pow_n = powmod(root, n.get() as u64, q.get());
        assert_eq!(pow2n, 1, "root^(2n) should be 1 mod q");
        assert_eq!(
            pow_n,
            q.get() - 1,
            "root^n should be -1 mod q (order must be exactly 2n, not a divisor)"
        );
    }
}
