//! Second ported operation (GOD PROMPT "CKKS RUST + CUDA PHASE 2"):
//! single-modulus RNS rescale, reusing the EXISTING, unmodified
//! `LaunchSwitchModulus`/`LaunchNTT_CT`/`LaunchINTT_GS`/`LaunchRescaleFuse`
//! kernels via FFI, and the existing `gpufhe::native_rescale_consts` for
//! the per-tower fuse constants (via the `cpp_bridge/rescale_consts_
//! bridge.cpp` C ABI wrapper -- that function itself still does the real
//! math, unmodified).
//!
//! ## Why this operation, and why it is safe to call independently
//!
//! The production C++ implementation is `rescale_resident_raw` in
//! `src/keyswitch_resident.cpp`:
//! ```text
//! void rescale_resident_raw(uint64_t* d_c, uint32_t towers,
//!                           const DeviceKSContext& C,
//!                           const std::vector<uint64_t>& s1, const std::vector<uint64_t>& s2,
//!                           uint64_t* d_scratch, uint64_t* d_drop, cudaStream_t s)
//! {
//!     const uint32_t n=C.n; if(towers<2) return;
//!     const uint32_t last=towers-1;
//!     uint64_t qLast=C.qModHost[last]; int rl=row_of(C,qLast);
//!     cudaMemcpyAsync(d_drop, d_c+(size_t)last*n, (size_t)n*8, cudaMemcpyDeviceToDevice, s);
//!     LaunchINTT_GS(d_drop, C.d_ir[rl], C.d_ip[rl], n, qLast, C.ninv[rl], C.ninv_p[rl], s);
//!     for(uint32_t t=0;t<last;++t){
//!         uint64_t qi=C.qModHost[t]; int r=row_of(C,qi);
//!         cudaMemcpyAsync(d_scratch, d_drop, (size_t)n*8, cudaMemcpyDeviceToDevice, s);
//!         LaunchSwitchModulus(d_scratch, qLast, qi, n, s);
//!         LaunchNTT_CT(d_scratch, C.d_fr[r], C.d_fp[r], n, qi, s);
//!         LaunchRescaleFuse(d_c+(size_t)t*n, d_scratch, s1[t], s2[t], qi, n, s);
//!     }
//! }
//! ```
//! It takes a `DeviceKSContext` only for three things: the per-tower
//! modulus list (`C.qModHost`), and CACHED per-modulus forward/inverse
//! NTT root tables (`C.d_fr`/`d_fp`/`d_ir`/`d_ip`, `C.ninv`/`ninv_p`,
//! looked up by `row_of`). None of that is specific to key-switching --
//! it is exactly the same kind of root-table bookkeeping
//! `crate::ntt::NttContext` already builds independently, and this
//! round's `tests/test_rescale_vector_diff_n1024.cpp` (an earlier round,
//! 0/35840 mismatches) already proved the four kernels above are
//! correct when driven this way, without needing the eval-key fields
//! (`K.av`/`K.bv`) `DeviceKSContext` also carries but this algorithm
//! never reads. [`RescaleContext`] is that same algorithm, rebuilding
//! its OWN root tables (reusing `crate::util::build_tables`, proven in
//! the `ntt` module) instead of depending on `DeviceKSContext` at all --
//! narrower, independently constructible, and verified below against an
//! independent RNS-only reference, not merely "it compiles and launches
//! a kernel".

use crate::util::{build_tables, cuda_err, cudaDeviceSynchronize, invmod, DeviceBuf};
use ckks_core::{CkksError, Modulus, RingDimension};
use cudarc::runtime::result as rt;
use std::ffi::c_void;

extern "C" {
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
    fn LaunchSwitchModulus(d_v: *mut u64, q_old: u64, q_new: u64, n: u32, s: *mut c_void);
    fn LaunchRescaleFuse(
        d_a: *mut u64,
        d_b: *const u64,
        s1: u64,
        s2: u64,
        q: u64,
        n: u32,
        s: *mut c_void,
    );
    // From cpp_bridge/rescale_consts_bridge.cpp.
    fn gpufhe_native_rescale_consts(
        mod_: *const u64,
        mod_len: u32,
        last: u32,
        s1_out: *mut u64,
        s2_out: *mut u64,
    );
}

/// Calls the existing, unmodified `gpufhe::native_rescale_consts` (via
/// the `cpp_bridge` wrapper) for the fused-rescale constants `(s1, s2)`
/// for dropping the LAST tower of `mods`. Deliberately NOT reimplemented
/// in Rust -- the exact-division trick it uses (`D=(Ql*u-1)/ql` tracked
/// mod `ql*qi` via 128-bit arithmetic) is genuine, non-trivial modular
/// arithmetic, not bookkeeping. Returns `(s1, s2)`, each of length
/// `mods.len() - 1`.
pub fn native_rescale_consts(mods: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let last = (mods.len() - 1) as u32;
    let mut s1 = vec![0u64; last as usize];
    let mut s2 = vec![0u64; last as usize];
    unsafe {
        gpufhe_native_rescale_consts(
            mods.as_ptr(),
            mods.len() as u32,
            last,
            s1.as_mut_ptr(),
            s2.as_mut_ptr(),
        );
    }
    (s1, s2)
}

/// Performs exactly the host-only work [`RescaleContext::new`] does
/// before it ever touches the GPU (the fused-rescale constants via FFI,
/// plus a primitive-root search + table build for EVERY tower in the
/// chain), and returns a cheap checksum so the optimizer cannot discard
/// the work. Mirrors `crate::ntt::cold_start_cost_probe`, so a benchmark
/// can measure this "cold-start" tier symmetrically with the equivalent
/// C++ measurement in `bench/bench_rescale_cpp_vs_rust.cpp`'s Tier 1.
pub fn cold_start_cost_probe(n: RingDimension, mods: &[u64]) -> u64 {
    let nn = n.get();
    let last = mods.len() - 1;
    let (s1, s2) = native_rescale_consts(mods);
    let mut sink = s1[0].wrapping_add(s2[0]);
    for (t, &qi) in mods.iter().enumerate() {
        let root = crate::ntt::native_root(n, Modulus::new(qi).unwrap());
        let tbl = build_tables(nn, qi, root, t == last);
        sink = sink.wrapping_add(tbl.roots[0]).wrapping_add(tbl.precon[0]);
    }
    sink
}

/// Reusable rescale execution context for one fixed modulus chain
/// `mods` (the LAST entry is the tower that gets dropped). Owns
/// everything that stays the same across repeated rescale calls: device
/// selection, the per-tower fused-rescale constants (`s1`/`s2`, via FFI
/// to the existing `native_rescale_consts`), per-survivor-tower forward
/// NTT tables, the dropped tower's inverse NTT table, and reusable
/// device scratch buffers. See this module's top-level doc comment for
/// why reconstructing these independently of `DeviceKSContext` is a
/// faithful, narrower port of the same algorithm, not a different one.
///
/// Not `Sync`/`Send`, same reasoning as `crate::ntt::NttContext`: the
/// scratch buffers are reused in place by every [`RescaleContext::rescale`]
/// call.
pub struct RescaleContext {
    n: RingDimension,
    towers: u32,
    mods: Vec<u64>,
    s1: Vec<u64>,
    s2: Vec<u64>,
    /// Per-survivor-tower forward (root, precon) table pairs, index
    /// matches `mods[0..towers-1]`.
    fwd_tables: Vec<(DeviceBuf, DeviceBuf)>,
    /// The dropped (last) tower's inverse (root, precon) table.
    inv_table: (DeviceBuf, DeviceBuf),
    n_inv_last: u64,
    n_inv_precon_last: u64,
    /// `towers * n` u64 elements: the ciphertext component being
    /// rescaled, uploaded fresh each call, survivor towers downloaded
    /// after.
    d_c: DeviceBuf,
    d_scratch: DeviceBuf,
    d_drop: DeviceBuf,
}

impl RescaleContext {
    /// Performs every step that does NOT depend on the per-call input:
    /// device selection, the fused-rescale constants, per-tower NTT
    /// table construction (reusing `crate::ntt::native_root` +
    /// `crate::util::build_tables`, both already proven), and device
    /// allocation. `mods` must have at least 2 entries (nothing to drop
    /// otherwise, matching `rescale_resident_raw`'s own `if(towers<2)
    /// return;` guard -- here surfaced as a constructor error instead of
    /// a silent no-op, since a context that can never do useful work is
    /// a caller mistake worth reporting immediately).
    pub fn new(n: RingDimension, mods: &[u64]) -> Result<Self, CkksError> {
        if mods.len() < 2 {
            return Err(CkksError::InvalidRingDimension {
                n: mods.len() as u64,
            });
        }
        for &q in mods {
            Modulus::new(q)?;
        }

        rt::device::set(0).map_err(cuda_err)?;

        let nn = n.get();
        let towers = mods.len() as u32;
        let last = towers - 1;
        let q_last = mods[last as usize];
        let bytes = nn as usize * 8;

        let (s1, s2) = native_rescale_consts(mods);

        let mut fwd_tables = Vec::with_capacity(last as usize);
        for &qi in &mods[..last as usize] {
            let root = crate::ntt::native_root(n, Modulus::new(qi)?);
            let t = build_tables(nn, qi, root, false);
            let d_fr = DeviceBuf::alloc(bytes)?;
            d_fr.upload_u64(&t.roots)?;
            let d_fp = DeviceBuf::alloc(bytes)?;
            d_fp.upload_u64(&t.precon)?;
            fwd_tables.push((d_fr, d_fp));
        }

        let root_last = crate::ntt::native_root(n, Modulus::new(q_last)?);
        let inv_t = build_tables(nn, q_last, root_last, true);
        let d_ir = DeviceBuf::alloc(bytes)?;
        d_ir.upload_u64(&inv_t.roots)?;
        let d_ip = DeviceBuf::alloc(bytes)?;
        d_ip.upload_u64(&inv_t.precon)?;
        let n_inv_last = invmod(nn as u64, q_last);
        let n_inv_precon_last = (((n_inv_last as u128) << 64) / q_last as u128) as u64;

        let d_c = DeviceBuf::alloc(bytes * towers as usize)?;
        let d_scratch = DeviceBuf::alloc(bytes)?;
        let d_drop = DeviceBuf::alloc(bytes)?;

        Ok(RescaleContext {
            n,
            towers,
            mods: mods.to_vec(),
            s1,
            s2,
            fwd_tables,
            inv_table: (d_ir, d_ip),
            n_inv_last,
            n_inv_precon_last,
            d_c,
            d_scratch,
            d_drop,
        })
    }

    pub fn ring_dimension(&self) -> RingDimension {
        self.n
    }

    /// Number of SURVIVOR towers after one rescale (`towers - 1`).
    pub fn survivor_towers(&self) -> u32 {
        self.towers - 1
    }

    /// Steady-state per-call path, the EXACT same sequence as
    /// `rescale_resident_raw`: upload the full `towers`-wide eval-domain
    /// ciphertext component into the reusable scratch buffer, copy+INTT
    /// the last tower, then for each survivor tower switch-modulus +
    /// NTT the dropped value and fuse it into that tower via
    /// `LaunchRescaleFuse`. `coeffs_in` must have `towers * n` elements
    /// (eval domain, all towers); `out` must have `(towers-1) * n`
    /// elements and receives only the survivor towers (eval domain) --
    /// matching `rescale_resident_raw`'s own contract of leaving the
    /// dropped tower's slot in `d_c` stale/unused.
    pub fn rescale(&self, coeffs_in: &[u64], out: &mut [u64]) -> Result<(), CkksError> {
        let nn = self.n.get() as usize;
        let towers = self.towers as usize;
        let last = towers - 1;
        if coeffs_in.len() != towers * nn {
            return Err(CkksError::InvalidRingDimension {
                n: coeffs_in.len() as u64,
            });
        }
        if out.len() != last * nn {
            return Err(CkksError::InvalidRingDimension { n: out.len() as u64 });
        }

        self.d_c.upload_u64(coeffs_in)?;

        let q_last = self.mods[last];
        let (d_ir, d_ip) = &self.inv_table;

        unsafe {
            // Drop the last tower into its own scratch buffer, INTT it
            // (eval -> coefficient domain) -- exactly rescale_resident_
            // raw's first two lines.
            self.d_drop
                .copy_from_ptr(self.d_c.tower_const_u64(last, nn) as *const c_void, nn)?;
            LaunchINTT_GS(
                self.d_drop.as_mut_u64(),
                d_ir.as_const_u64(),
                d_ip.as_const_u64(),
                nn as u32,
                q_last,
                self.n_inv_last,
                self.n_inv_precon_last,
                std::ptr::null_mut(),
            );

            for t in 0..last {
                let qi = self.mods[t];
                let (d_fr, d_fp) = &self.fwd_tables[t];
                self.d_scratch
                    .copy_from_ptr(self.d_drop.as_const_u64() as *const c_void, nn)?;
                LaunchSwitchModulus(
                    self.d_scratch.as_mut_u64(),
                    q_last,
                    qi,
                    nn as u32,
                    std::ptr::null_mut(),
                );
                LaunchNTT_CT(
                    self.d_scratch.as_mut_u64(),
                    d_fr.as_const_u64(),
                    d_fp.as_const_u64(),
                    nn as u32,
                    qi,
                    std::ptr::null_mut(),
                );
                LaunchRescaleFuse(
                    self.d_c.tower_mut_u64(t, nn),
                    self.d_scratch.as_const_u64(),
                    self.s1[t],
                    self.s2[t],
                    qi,
                    nn as u32,
                    std::ptr::null_mut(),
                );
            }

            let rc = cudaDeviceSynchronize();
            if rc != 0 {
                return Err(CkksError::Cuda(format!(
                    "cudaDeviceSynchronize failed, cudaError_t={rc}"
                )));
            }
        }

        // Download only the survivor towers (first `last*nn` elements of
        // d_c, since `out.len() == last*nn` and `download_u64` copies
        // exactly `out.len()` elements from the start of the buffer) --
        // the dropped tower's slot is left stale, matching
        // rescale_resident_raw's own contract.
        self.d_c.download_u64(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::native_root;

    // Same 8-tower modulus chain as this round's C++ work
    // (test_rescale_n1024.cpp / test_rescale_vector_diff_n1024.cpp), so
    // results are directly comparable across languages.
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

    fn forward_ntt(coeff: &[u64], n: u32, q: u64, root: u64) -> Vec<u64> {
        let t = build_tables(n, q, root, false);
        let bytes = n as usize * 8;
        let d_x = DeviceBuf::alloc(bytes).unwrap();
        d_x.upload_u64(coeff).unwrap();
        let d_fr = DeviceBuf::alloc(bytes).unwrap();
        d_fr.upload_u64(&t.roots).unwrap();
        let d_fp = DeviceBuf::alloc(bytes).unwrap();
        d_fp.upload_u64(&t.precon).unwrap();
        unsafe {
            LaunchNTT_CT(
                d_x.as_mut_u64(),
                d_fr.as_const_u64(),
                d_fp.as_const_u64(),
                n,
                q,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        let mut out = vec![0u64; n as usize];
        d_x.download_u64(&mut out).unwrap();
        out
    }

    /// Independent RNS-only reference, position-wise over residues (no
    /// big-integer CRT reconstruction needed) -- IDENTICAL methodology
    /// to the earlier round's `tests/test_rescale_vector_diff_n1024.cpp`
    /// (C++), which proved this exact formula correct against
    /// `rescale_resident_raw` with 0/35840 mismatches:
    ///   v[k]   = center(c_last[k], qLast)
    ///   y_i[k] = (c_i[k] - v[k]) * s1[i] mod qi,  i = 0..towers-2
    /// Computed here fresh in Rust (not imported from the C++ test) so
    /// the two implementations cannot share a transcription bug.
    fn rns_rescale_reference(
        c: &[Vec<u64>],
        mods: &[u64],
        s1: &[u64],
        n: usize,
    ) -> Vec<Vec<u64>> {
        let last = mods.len() - 1;
        let q_last = mods[last];
        let half = q_last >> 1;
        let mut yref = vec![vec![0u64; n]; last];
        for k in 0..n {
            let x_last = c[last][k];
            let neg = x_last > half;
            let v_abs = if neg { q_last - x_last } else { x_last };
            for i in 0..last {
                let qi = mods[i];
                let v_mod_qi = if neg {
                    (qi - (v_abs % qi)) % qi
                } else {
                    v_abs % qi
                };
                let diff = (c[i][k] + qi - v_mod_qi) % qi;
                yref[i][k] = ((diff as u128 * s1[i] as u128) % qi as u128) as u64;
            }
        }
        yref
    }

    #[test]
    fn rescale_matches_independent_rns_reference_random_data() {
        use rand::{Rng, SeedableRng};
        let n_dim = RingDimension::new(1024).unwrap();
        let n = 1024usize;
        let ctx = RescaleContext::new(n_dim, &MODULI).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(424242);
        let mut total_mismatches = 0usize;
        for trial in 0..3 {
            // Independent per-coefficient random residues per tower --
            // NOT a constant (which would trivially satisfy NTT(constant)
            // = constant and hide position-mixing bugs).
            let c: Vec<Vec<u64>> = MODULI
                .iter()
                .map(|&q| (0..n).map(|_| rng.gen_range(0..q)).collect())
                .collect();

            let yref = rns_rescale_reference(&c, &MODULI, &ctx.s1, n);

            // Real eval-domain input: forward-NTT every tower's
            // coefficient-domain residues.
            let mut input = vec![0u64; MODULI.len() * n];
            for (t, &q) in MODULI.iter().enumerate() {
                let root = native_root(n_dim, Modulus::new(q).unwrap());
                let eval = forward_ntt(&c[t], n as u32, q, root);
                input[t * n..(t + 1) * n].copy_from_slice(&eval);
            }

            let mut actual = vec![0u64; ctx.survivor_towers() as usize * n];
            ctx.rescale(&input, &mut actual).unwrap();

            for i in 0..MODULI.len() - 1 {
                let q = MODULI[i];
                let root = native_root(n_dim, Modulus::new(q).unwrap());
                let expected_eval = forward_ntt(&yref[i], n as u32, q, root);
                for k in 0..n {
                    if actual[i * n + k] != expected_eval[k] {
                        total_mismatches += 1;
                        if total_mismatches <= 3 {
                            eprintln!(
                                "trial={trial} tower={i} k={k} got={} want={}",
                                actual[i * n + k],
                                expected_eval[k]
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            total_mismatches, 0,
            "{total_mismatches} mismatches vs independent RNS reference"
        );
    }

    #[test]
    fn rescale_matches_reference_reused_context_repeated_calls() {
        // Calls rescale() SEVERAL times on the SAME context with
        // DIFFERENT random inputs each time, proving no stale state
        // leaks between calls (the scratch buffers are fully overwritten
        // every call, not partially stale from a previous one).
        use rand::{Rng, SeedableRng};
        let n_dim = RingDimension::new(1024).unwrap();
        let n = 1024usize;
        let ctx = RescaleContext::new(n_dim, &MODULI).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        for call in 0..5 {
            let c: Vec<Vec<u64>> = MODULI
                .iter()
                .map(|&q| (0..n).map(|_| rng.gen_range(0..q)).collect())
                .collect();
            let yref = rns_rescale_reference(&c, &MODULI, &ctx.s1, n);
            let mut input = vec![0u64; MODULI.len() * n];
            for (t, &q) in MODULI.iter().enumerate() {
                let root = native_root(n_dim, Modulus::new(q).unwrap());
                let eval = forward_ntt(&c[t], n as u32, q, root);
                input[t * n..(t + 1) * n].copy_from_slice(&eval);
            }
            let mut actual = vec![0u64; ctx.survivor_towers() as usize * n];
            ctx.rescale(&input, &mut actual).unwrap();
            let q0 = MODULI[0];
            let root0 = native_root(n_dim, Modulus::new(q0).unwrap());
            let expected0 = forward_ntt(&yref[0], n as u32, q0, root0);
            assert_eq!(
                actual[0..n], expected0[..],
                "call={call}: tower 0 mismatch on reused context"
            );
        }
    }

    #[test]
    fn invalid_input_length_is_rejected() {
        let n_dim = RingDimension::new(1024).unwrap();
        let ctx = RescaleContext::new(n_dim, &MODULI).unwrap();
        let bad_input = vec![0u64; 10]; // way too short
        let mut out = vec![0u64; ctx.survivor_towers() as usize * 1024];
        let err = ctx.rescale(&bad_input, &mut out);
        assert!(err.is_err(), "undersized input must be rejected, not silently truncated/overrun");
    }

    #[test]
    fn single_tower_chain_is_rejected_at_construction() {
        let n_dim = RingDimension::new(1024).unwrap();
        let err = RescaleContext::new(n_dim, &MODULI[..1]);
        assert!(
            err.is_err(),
            "a 1-tower chain has nothing to drop -- must fail at construction, not silently no-op"
        );
    }
}
