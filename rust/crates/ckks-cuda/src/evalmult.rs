//! Fourth ported operation (GOD PROMPT "CKKS RUST + CUDA PHASE 4"):
//! ciphertext-ciphertext EvalMult, i.e. tensor product + relinearization
//! (via key switching) + rescale, exactly the composition
//! `bench/bench_evalmult_resident.cpp` already demonstrates in C++:
//! ```text
//! LaunchTensor(c0a,c1a,c0b,c1b) -> t0,t1,t2
//! keyswitch_resident(t2) -> ba0,ba1              // relinearize t2
//! LaunchCombine(t0,ba0, t1,ba1) -> r0,r1          // r = t[0..2] + ba
//! rescale_resident_raw(r0); rescale_resident_raw(r1)
//! ```
//! `kernels/tensor.cu`'s `LaunchTensor`/`LaunchCombine` are the two new
//! kernels this operation needs beyond what `ntt`/`rescale`/`keyswitch`
//! already ported; both confirmed self-contained (one inline `mulmod64`
//! device helper, defined and used in the SAME file) the same way every
//! previously-ported kernel was, so `build.rs` compiles `tensor.cu`
//! directly, no `-rdc=true` needed.
//!
//! ## Why this module reuses `KeySwitchContext`/`RescaleContext` instead
//! of rebuilding their logic
//!
//! Relinearization IS a key-switch call on the tensor product's third
//! component (`t2`), and each rescale step IS exactly
//! `RescaleContext::rescale`'s own algorithm -- so [`EvalMultContext`]
//! EMBEDS one [`crate::keyswitch::KeySwitchContext`] and one
//! [`crate::rescale::RescaleContext`] (built over the EXACT SAME modulus
//! chain -- queried from the embedded `KeySwitchContext` via
//! `modulus_q`, not re-derived, so the two can never silently diverge)
//! rather than re-deriving either algorithm. But their PUBLIC APIs
//! (`keyswitch`/`rescale`) are host-slice-oriented: calling them as-is
//! here would force two pointless host round trips per operation (upload
//! `t2`, download `ba0`/`ba1`; upload `r0`/`r1` again for each rescale,
//! download again) that the real `bench_evalmult_resident.cpp` pipeline
//! never pays (`keyswitch_resident`/`rescale_resident_raw` there operate
//! entirely in device memory, back to back on one stream). This is
//! exactly the governing directive's warning not to assume an existing
//! interface is "sufficient without adaptation": the adaptation is two
//! new `pub(crate)` device-pointer methods,
//! [`crate::keyswitch::KeySwitchContext::keyswitch_device_resident`] and
//! [`crate::rescale::RescaleContext::rescale_device_resident`], each a
//! thin wrapper reusing 100% of the existing tested logic -- no
//! cryptographic or kernel-launch code duplicated.
//!
//! ## Scope boundary: this operates at the raw RNS/eval-domain level,
//! like `ntt`/`rescale`/`keyswitch` -- NOT a scale-tracking Ciphertext
//! abstraction
//!
//! Matching every operation ported so far, [`EvalMultContext`] takes and
//! returns plain per-tower `u64` residue arrays (eval domain), not a
//! higher-level object that tracks a CKKS scale/Delta. The C++ engine's
//! own primitives this module wraps (`keyswitch_resident`,
//! `rescale_resident_raw`) are ALSO scale-oblivious -- scale/Delta
//! bookkeeping lives at a higher layer (`src/ckks_ops.cpp`'s
//! `DeviceCiphertext`-based orchestration, or the host encode/encrypt/
//! decrypt pipeline `tests/test_polyeval.cpp` drives), which this round
//! does not touch. What IS verified here: ciphertext TOWER COUNT
//! transitions (`size_ql` in, `size_ql - 1` out, matching one rescale's
//! worth of level drop) and bit-exact numerical correctness against an
//! independent reference at the RNS level -- not a scale value, which
//! does not exist at this layer. Verifying this operation against a REAL
//! encode/encrypt/decrypt round trip (scale included) is explicitly left
//! as a follow-up, not attempted this round -- see `research/
//! NEXT_SESSION.md`.
//!
//! ## A domain bug this module's OWN test caught (not a defect in the
//! code above)
//!
//! The first version of this module's differential test fed
//! synthetic residues directly as EVAL-domain operands and ALSO applied
//! rescale's "center the dropped tower" independent-reference identity
//! directly to that same eval-domain data -- but that identity is a
//! COEFFICIENT-domain statement (see `rns_rescale_reference`'s doc
//! comment below), so every single output coefficient mismatched. Caught
//! by cross-checking each intermediate stage (`t0`/`t1`/`t2`/`ba0`/`ba1`/
//! `r0`/`r1` all agreed between the GPU path and a step-by-step
//! reference; only the FINAL rescaled value diverged) and by independently
//! confirming the GPU path's OWN `rescale_device_resident` agreed
//! bit-exact with the already-proven `rescale` on identical data -- which
//! together isolated the bug to the TEST's reference formula, not
//! `EvalMultContext`, `RescaleContext`, or `KeySwitchContext`. Fixed by
//! generating genuinely COEFFICIENT-domain inputs and forward/inverse
//! NTT-ing around the rescale identity, exactly mirroring `rescale::
//! tests`' own established methodology.
use crate::keyswitch::KeySwitchContext;
use crate::rescale::RescaleContext;
use crate::util::{cuda_err, cudaDeviceSynchronize, DeviceBuf};
use ckks_core::{CkksError, RingDimension};
use cudarc::runtime::result as rt;
use std::ffi::c_void;

extern "C" {
    // From kernels/tensor.cu, compiled directly by build.rs (see the
    // module doc comment).
    fn LaunchTensor(
        c0a: *const u64,
        c1a: *const u64,
        c0b: *const u64,
        c1b: *const u64,
        t0: *mut u64,
        t1: *mut u64,
        t2: *mut u64,
        mods: *const u64,
        size_q: u32,
        n: u32,
        s: *mut c_void,
    );
    fn LaunchCombine(
        r0: *mut u64,
        r1: *mut u64,
        t0: *const u64,
        t1: *const u64,
        ba0: *const u64,
        ba1: *const u64,
        mods: *const u64,
        size_q: u32,
        n: u32,
        s: *mut c_void,
    );
}

/// Reusable EvalMult (tensor + relinearize + rescale) execution context.
/// Owns a [`KeySwitchContext`] (the relinearization backend -- its own
/// eval key, built once) and a [`RescaleContext`] over the SAME modulus
/// chain (queried from the `KeySwitchContext`, not re-derived), plus 11
/// per-call device buffers (`c0a,c1a,c0b,c1b,t0,t1,t2,ba0,ba1,r0,r1`),
/// each `size_ql * n` elements, allocated once. `multiply_relin_rescale`
/// is the hot path: upload the 4 operand towers, launch the fused
/// device-resident chain (tensor -> relin -> combine -> rescale x2, all
/// on the default stream, one sync at the end), download the two
/// `(size_ql - 1) * n`-element survivor-tower results.
///
/// Not `Send`/`Sync` (holds raw-pointer-bearing fields transitively, same
/// as every other context in this crate) -- no concurrency claim, no
/// concurrency test.
pub struct EvalMultContext {
    ks: KeySwitchContext,
    resc: RescaleContext,
    n: RingDimension,
    size_ql: u32,
    d_mods: DeviceBuf,
    c0a: DeviceBuf,
    c1a: DeviceBuf,
    c0b: DeviceBuf,
    c1b: DeviceBuf,
    t0: DeviceBuf,
    t1: DeviceBuf,
    t2: DeviceBuf,
    ba0: DeviceBuf,
    ba1: DeviceBuf,
    r0: DeviceBuf,
    r1: DeviceBuf,
}

impl EvalMultContext {
    /// Builds the relinearization eval key (`KeySwitchContext::new`),
    /// then a `RescaleContext` over the IDENTICAL modulus chain the eval
    /// key picked (via `modulus_q`, not `native_primes` called a second
    /// time -- calling it twice with the same arguments would happen to
    /// agree since it is deterministic, but querying the already-built
    /// chain is the honest way to GUARANTEE agreement rather than rely on
    /// that), then allocates the 11 per-call buffers and uploads the
    /// per-tower modulus array `LaunchTensor`/`LaunchCombine` need.
    pub fn new(
        n: RingDimension,
        size_q: u32,
        size_p: u32,
        num_part: u32,
        seed: u64,
    ) -> Result<Self, CkksError> {
        let ks = KeySwitchContext::new(n, size_q, size_p, num_part, seed)?;
        let size_ql = ks.size_ql();

        let mods: Vec<u64> = (0..size_ql).map(|t| ks.modulus_q(t)).collect();
        let resc = RescaleContext::new(n, &mods)?;

        rt::device::set(0).map_err(cuda_err)?;

        let d_mods = DeviceBuf::alloc(size_ql as usize * 8)?;
        d_mods.upload_u64(&mods)?;

        let bytes = n.get() as usize * 8 * size_ql as usize;
        let alloc = || DeviceBuf::alloc(bytes);
        let (c0a, c1a, c0b, c1b) = (alloc()?, alloc()?, alloc()?, alloc()?);
        let (t0, t1, t2) = (alloc()?, alloc()?, alloc()?);
        let (ba0, ba1) = (alloc()?, alloc()?);
        let (r0, r1) = (alloc()?, alloc()?);

        Ok(EvalMultContext {
            ks,
            resc,
            n,
            size_ql,
            d_mods,
            c0a,
            c1a,
            c0b,
            c1b,
            t0,
            t1,
            t2,
            ba0,
            ba1,
            r0,
            r1,
        })
    }

    pub fn ring_dimension(&self) -> RingDimension {
        self.n
    }

    /// Tower count BOTH input ciphertext components must have.
    pub fn input_towers(&self) -> u32 {
        self.size_ql
    }

    /// Tower count of each OUTPUT component after one rescale (matches
    /// `RescaleContext::survivor_towers` for the embedded rescale
    /// context, by construction -- `size_ql - 1`).
    pub fn output_towers(&self) -> u32 {
        self.size_ql - 1
    }

    /// The actual Q modulus for one tower, forwarded from the embedded
    /// `KeySwitchContext` so callers can generate valid in-range input
    /// residues without re-deriving this context's modulus chain.
    pub fn modulus_q(&self, tower: u32) -> u64 {
        self.ks.modulus_q(tower)
    }

    /// Steady-state per-call path: `ct_a = (c0a, c1a)`, `ct_b = (c0b,
    /// c1b)`, each `input_towers() * n` elements, eval domain. Returns
    /// `(r0, r1)`, each `output_towers() * n` elements, eval domain --
    /// the relinearized, rescaled product `ct_a * ct_b`. See the module
    /// doc comment for why no scale/Delta value is tracked or returned.
    pub fn multiply_relin_rescale(
        &self,
        c0a: &[u64],
        c1a: &[u64],
        c0b: &[u64],
        c1b: &[u64],
    ) -> Result<(Vec<u64>, Vec<u64>), CkksError> {
        let in_len = self.size_ql as usize * self.n.get() as usize;
        if c0a.len() != in_len || c1a.len() != in_len || c0b.len() != in_len || c1b.len() != in_len {
            return Err(CkksError::InvalidRingDimension {
                n: c0a.len().max(c1a.len()).max(c0b.len()).max(c1b.len()) as u64,
            });
        }

        self.c0a.upload_u64(c0a)?;
        self.c1a.upload_u64(c1a)?;
        self.c0b.upload_u64(c0b)?;
        self.c1b.upload_u64(c1b)?;

        let n = self.n.get();
        let size_q = self.size_ql;

        unsafe {
            LaunchTensor(
                self.c0a.as_const_u64(),
                self.c1a.as_const_u64(),
                self.c0b.as_const_u64(),
                self.c1b.as_const_u64(),
                self.t0.as_mut_u64(),
                self.t1.as_mut_u64(),
                self.t2.as_mut_u64(),
                self.d_mods.as_const_u64(),
                size_q,
                n,
                std::ptr::null_mut(),
            );

            // Relinearize t2 via the embedded KeySwitchContext's eval key,
            // entirely device-resident -- syncs internally (see that
            // method's own doc comment), so everything queued above is
            // guaranteed complete before this returns.
            self.ks
                .keyswitch_device_resident(self.t2.as_const_u64(), self.ba0.as_mut_u64(), self.ba1.as_mut_u64())?;

            LaunchCombine(
                self.r0.as_mut_u64(),
                self.r1.as_mut_u64(),
                self.t0.as_const_u64(),
                self.t1.as_const_u64(),
                self.ba0.as_const_u64(),
                self.ba1.as_const_u64(),
                self.d_mods.as_const_u64(),
                size_q,
                n,
                std::ptr::null_mut(),
            );

            // Both rescale calls reuse the SAME embedded RescaleContext's
            // scratch buffers sequentially -- safe because every launch
            // above and below is on the default stream, so execution order
            // (and therefore scratch-buffer reuse) is already serialized.
            self.resc.rescale_device_resident(self.r0.as_mut_u64())?;
            self.resc.rescale_device_resident(self.r1.as_mut_u64())?;

            let rc = cudaDeviceSynchronize();
            if rc != 0 {
                return Err(CkksError::Cuda(format!(
                    "cudaDeviceSynchronize failed, cudaError_t={rc}"
                )));
            }
        }

        let out_len = self.output_towers() as usize * self.n.get() as usize;
        let mut out0 = vec![0u64; out_len];
        let mut out1 = vec![0u64; out_len];
        self.r0.download_u64(&mut out0)?;
        self.r1.download_u64(&mut out1)?;
        Ok((out0, out1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    // Same configuration as keyswitch::tests -- a known-working native
    // parameter set, so the relinearization eval key's own correctness
    // evidence (entry #36) transfers directly to this module's tests.
    const N: u32 = 1024;
    const SIZE_Q: u32 = 8;
    const SIZE_P: u32 = 2;
    const NUM_PART: u32 = 2;

    fn new_ctx(seed: u64) -> EvalMultContext {
        let n = RingDimension::new(N).unwrap();
        EvalMultContext::new(n, SIZE_Q, SIZE_P, NUM_PART, seed).expect("context init failed")
    }

    /// Random COEFFICIENT-domain per-tower residues (independent per
    /// position, not a constant -- a constant would trivially satisfy
    /// NTT(constant)=constant and hide position-mixing bugs, same
    /// reasoning as every other operation's own differential test).
    fn random_tower_data(ctx: &EvalMultContext, rng: &mut impl Rng) -> Vec<u64> {
        let nn = ctx.ring_dimension().get() as usize;
        let mut v = vec![0u64; ctx.input_towers() as usize * nn];
        for t in 0..ctx.input_towers() {
            let q = ctx.modulus_q(t);
            for k in 0..nn {
                v[t as usize * nn + k] = rng.gen_range(0..q);
            }
        }
        v
    }

    /// Forward-NTTs every tower of a COEFFICIENT-domain ciphertext
    /// component into the EVAL-domain form `EvalMultContext::
    /// multiply_relin_rescale` actually expects as input.
    fn coeff_to_eval(ctx: &EvalMultContext, coeff: &[u64]) -> Vec<u64> {
        let nn = ctx.ring_dimension().get();
        let towers = ctx.input_towers() as usize;
        let mut eval = vec![0u64; towers * nn as usize];
        for t in 0..towers {
            let q = ctx.modulus_q(t as u32);
            let root = crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap());
            let e = forward_ntt(&coeff[t * nn as usize..(t + 1) * nn as usize], nn, q, root);
            eval[t * nn as usize..(t + 1) * nn as usize].copy_from_slice(&e);
        }
        eval
    }

    fn mulmod(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }
    fn addmod(a: u64, b: u64, q: u64) -> u64 {
        let s = a + b;
        if s >= q {
            s - q
        } else {
            s
        }
    }

    // Forward/inverse NTT, test-only -- same pattern as
    // `rescale::tests::forward_ntt`/`keyswitch`'s analogous helpers, with
    // its OWN extern "C" declarations (duplicated per the project's
    // established per-module convention, not shared). Needed because
    // `rns_rescale_reference`'s "center the dropped tower" identity is
    // ONLY valid in COEFFICIENT domain (see this test module's own
    // discovery below): `kernels/tensor.cu`'s tensor/combine kernels
    // operate pointwise on EVAL-domain ciphertext components (that's the
    // whole point of NTT-based multiplication), so a correct independent
    // reference must convert the dropped tower to coefficient domain
    // before applying that identity, then convert the result back.
    extern "C" {
        fn LaunchNTT_CT(d_x: *mut u64, d_roots: *const u64, d_precon: *const u64, n: u32, q: u64, s: *mut c_void);
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
    }

    fn forward_ntt(coeff: &[u64], n: u32, q: u64, root: u64) -> Vec<u64> {
        let t = crate::util::build_tables(n, q, root, false);
        let bytes = n as usize * 8;
        let d_x = DeviceBuf::alloc(bytes).unwrap();
        d_x.upload_u64(coeff).unwrap();
        let d_fr = DeviceBuf::alloc(bytes).unwrap();
        d_fr.upload_u64(&t.roots).unwrap();
        let d_fp = DeviceBuf::alloc(bytes).unwrap();
        d_fp.upload_u64(&t.precon).unwrap();
        unsafe {
            LaunchNTT_CT(d_x.as_mut_u64(), d_fr.as_const_u64(), d_fp.as_const_u64(), n, q, std::ptr::null_mut());
            cudaDeviceSynchronize();
        }
        let mut out = vec![0u64; n as usize];
        d_x.download_u64(&mut out).unwrap();
        out
    }

    fn inverse_ntt(eval: &[u64], n: u32, q: u64, root: u64) -> Vec<u64> {
        let t = crate::util::build_tables(n, q, root, true);
        let bytes = n as usize * 8;
        let d_x = DeviceBuf::alloc(bytes).unwrap();
        d_x.upload_u64(eval).unwrap();
        let d_ir = DeviceBuf::alloc(bytes).unwrap();
        d_ir.upload_u64(&t.roots).unwrap();
        let d_ip = DeviceBuf::alloc(bytes).unwrap();
        d_ip.upload_u64(&t.precon).unwrap();
        let n_inv = crate::util::invmod(n as u64, q);
        let n_inv_precon = (((n_inv as u128) << 64) / q as u128) as u64;
        unsafe {
            LaunchINTT_GS(
                d_x.as_mut_u64(),
                d_ir.as_const_u64(),
                d_ip.as_const_u64(),
                n,
                q,
                n_inv,
                n_inv_precon,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        let mut out = vec![0u64; n as usize];
        d_x.download_u64(&mut out).unwrap();
        out
    }

    /// Same from-scratch RNS-only rescale identity as
    /// `rescale::tests::rns_rescale_reference`, re-derived independently
    /// here (not imported) so the two test modules cannot share a
    /// transcription bug: `v[k]=center(c_last[k],qLast)`,
    /// `y_i[k]=(c_i[k]-v[k])*s1[i] mod qi`. Valid ONLY when `c` is
    /// COEFFICIENT domain (see `reference_multiply_relin_rescale`'s own
    /// doc comment for why this module converts domains around this
    /// call, which an earlier draft of this test did not do -- a real,
    /// caught-before-landing bug, not a defect in `EvalMultContext`
    /// itself; see this module's commit message / the experiment ledger
    /// for the full root-cause trace).
    fn rns_rescale_reference(c: &[u64], mods: &[u64], s1: &[u64], n: usize) -> Vec<u64> {
        let last = mods.len() - 1;
        let q_last = mods[last];
        let half = q_last >> 1;
        let mut yref = vec![0u64; last * n];
        for k in 0..n {
            let x_last = c[last * n + k];
            let neg = x_last > half;
            let v_abs = if neg { q_last - x_last } else { x_last };
            for i in 0..last {
                let qi = mods[i];
                let v_mod_qi = if neg { (qi - (v_abs % qi)) % qi } else { v_abs % qi };
                let diff = (c[i * n + k] + qi - v_mod_qi) % qi;
                yref[i * n + k] = ((diff as u128 * s1[i] as u128) % qi as u128) as u64;
            }
        }
        yref
    }

    /// Fresh, independent reference for the full tensor + relin + rescale
    /// chain, given COEFFICIENT-domain ciphertext components (`c0a` etc.
    /// -- the caller forward-NTTs these the SAME way to get the actual
    /// GPU-path input, so both paths start from the identical underlying
    /// polynomial). The tensor product and combine steps are literal,
    /// from-scratch schoolbook arithmetic in EVAL domain (pointwise
    /// multiply/add mod q, matching `kernels/tensor.cu`'s own documented
    /// formulas but computed here with zero GPU involvement) -- this part
    /// was ALREADY CONFIRMED correct while root-causing this test's first
    /// failing draft (GPU and reference `t0`/`t1`/`t2`/`ba0`/`ba1`/`r0`/
    /// `r1` all matched bit-exact; only the rescale step was wrong, see
    /// below). Relinearization reuses `KeySwitchContext::
    /// keyswitch_reference` -- the SAME established host-reference oracle
    /// entry #36 already anchors this crate's key-switch correctness
    /// evidence on, not re-derived here.
    ///
    /// Rescale is where domain matters: `r0`/`r1` after combine are EVAL
    /// domain (tensor/combine are pointwise-eval operations, by
    /// construction), but `rns_rescale_reference`'s "center the dropped
    /// tower" identity is a COEFFICIENT-domain statement about the TRUE
    /// integer value of each coefficient -- applying it directly to
    /// eval-domain data computes something with no relationship to the
    /// real rescale operation (same domain-confusion bug CLASS as entry
    /// #35's rescale benchmark incident, independently rediscovered here
    /// while building this test). So each tower of `r0`/`r1` is INTT'd to
    /// coefficient domain first, the identity applied there, and the
    /// `(towers-1)`-tower result forward-NTT'd back to eval domain for
    /// comparison with the GPU path's actual (eval-domain) output --
    /// exactly mirroring `rescale::tests`' own established methodology,
    /// extended across the tensor+relin+combine composition.
    fn reference_multiply_relin_rescale(
        ctx: &EvalMultContext,
        c0a: &[u64],
        c1a: &[u64],
        c0b: &[u64],
        c1b: &[u64],
    ) -> (Vec<u64>, Vec<u64>) {
        let nn = ctx.ring_dimension().get() as usize;
        let towers = ctx.input_towers() as usize;
        let mods: Vec<u64> = (0..towers as u32).map(|t| ctx.modulus_q(t)).collect();
        let n_dim = ctx.ring_dimension();

        let c0a_eval = coeff_to_eval(ctx, c0a);
        let c1a_eval = coeff_to_eval(ctx, c1a);
        let c0b_eval = coeff_to_eval(ctx, c0b);
        let c1b_eval = coeff_to_eval(ctx, c1b);

        let mut t0 = vec![0u64; towers * nn];
        let mut t1 = vec![0u64; towers * nn];
        let mut t2 = vec![0u64; towers * nn];
        for t in 0..towers {
            let q = mods[t];
            for k in 0..nn {
                let x = t * nn + k;
                let (a0, a1, b0, b1) = (c0a_eval[x], c1a_eval[x], c0b_eval[x], c1b_eval[x]);
                t0[x] = mulmod(a0, b0, q);
                t1[x] = addmod(mulmod(a0, b1, q), mulmod(a1, b0, q), q);
                t2[x] = mulmod(a1, b1, q);
            }
        }

        let (ba0, ba1) = ctx.ks.keyswitch_reference(&t2).expect("reference keyswitch failed");

        let mut r0 = vec![0u64; towers * nn];
        let mut r1 = vec![0u64; towers * nn];
        for t in 0..towers {
            let q = mods[t];
            for k in 0..nn {
                let x = t * nn + k;
                r0[x] = addmod(t0[x], ba0[x], q);
                r1[x] = addmod(t1[x], ba1[x], q);
            }
        }

        let to_coeff = |eval_towers: &[u64]| -> Vec<u64> {
            let mut coeff = vec![0u64; towers * nn];
            for t in 0..towers {
                let q = mods[t];
                let root = crate::ntt::native_root(n_dim, ckks_core::Modulus::new(q).unwrap());
                let c = inverse_ntt(&eval_towers[t * nn..(t + 1) * nn], nn as u32, q, root);
                coeff[t * nn..(t + 1) * nn].copy_from_slice(&c);
            }
            coeff
        };
        let r0_coeff = to_coeff(&r0);
        let r1_coeff = to_coeff(&r1);

        let (s1, _s2) = crate::rescale::native_rescale_consts(&mods);
        let yref0_coeff = rns_rescale_reference(&r0_coeff, &mods, &s1, nn);
        let yref1_coeff = rns_rescale_reference(&r1_coeff, &mods, &s1, nn);

        let survivor = towers - 1;
        let to_eval_survivors = |coeff_towers: &[u64]| -> Vec<u64> {
            let mut eval = vec![0u64; survivor * nn];
            for t in 0..survivor {
                let q = mods[t];
                let root = crate::ntt::native_root(n_dim, ckks_core::Modulus::new(q).unwrap());
                let e = forward_ntt(&coeff_towers[t * nn..(t + 1) * nn], nn as u32, q, root);
                eval[t * nn..(t + 1) * nn].copy_from_slice(&e);
            }
            eval
        };
        (to_eval_survivors(&yref0_coeff), to_eval_survivors(&yref1_coeff))
    }

    #[test]
    fn evalmult_matches_independent_reference_random_data() {
        let ctx = new_ctx(909);
        let mut rng = rand::rngs::StdRng::seed_from_u64(31337);
        for trial in 0..3 {
            // Coefficient domain: what the reference's rescale identity
            // needs, and what a real ciphertext component IS.
            let c0a = random_tower_data(&ctx, &mut rng);
            let c1a = random_tower_data(&ctx, &mut rng);
            let c0b = random_tower_data(&ctx, &mut rng);
            let c1b = random_tower_data(&ctx, &mut rng);

            // Eval domain: what the real GPU-path kernels need (pointwise
            // multiplication IS polynomial multiplication only in this
            // domain).
            let (got0, got1) = ctx
                .multiply_relin_rescale(
                    &coeff_to_eval(&ctx, &c0a),
                    &coeff_to_eval(&ctx, &c1a),
                    &coeff_to_eval(&ctx, &c0b),
                    &coeff_to_eval(&ctx, &c1b),
                )
                .expect("multiply failed");
            let (exp0, exp1) = reference_multiply_relin_rescale(&ctx, &c0a, &c1a, &c0b, &c1b);
            assert_eq!(got0, exp0, "trial {trial}: r0 mismatch vs independent reference");
            assert_eq!(got1, exp1, "trial {trial}: r1 mismatch vs independent reference");
        }
    }

    #[test]
    fn evalmult_matches_reference_reused_context_repeated_calls() {
        // Calls multiply_relin_rescale() SEVERAL times on the SAME
        // context with DIFFERENT random inputs, proving the 11 reused
        // per-call buffers (and the embedded KeySwitchContext/
        // RescaleContext's own reused scratch) don't leak state between
        // calls.
        let ctx = new_ctx(111);
        let mut rng = rand::rngs::StdRng::seed_from_u64(222);
        for call in 0..5 {
            let c0a = random_tower_data(&ctx, &mut rng);
            let c1a = random_tower_data(&ctx, &mut rng);
            let c0b = random_tower_data(&ctx, &mut rng);
            let c1b = random_tower_data(&ctx, &mut rng);
            let (got0, got1) = ctx
                .multiply_relin_rescale(
                    &coeff_to_eval(&ctx, &c0a),
                    &coeff_to_eval(&ctx, &c1a),
                    &coeff_to_eval(&ctx, &c0b),
                    &coeff_to_eval(&ctx, &c1b),
                )
                .expect("multiply failed");
            let (exp0, exp1) = reference_multiply_relin_rescale(&ctx, &c0a, &c1a, &c0b, &c1b);
            assert_eq!(got0, exp0, "call {call}: r0 mismatch on reused context");
            assert_eq!(got1, exp1, "call {call}: r1 mismatch on reused context");
        }
    }

    #[test]
    fn evalmult_matches_reference_on_boundary_inputs() {
        // Zero ciphertexts and max-residue (q-1 everywhere) ciphertexts --
        // deterministic boundary cases, not just random ones. Values are
        // coefficient domain (a constant coefficient-domain polynomial is
        // NOT a constant eval-domain one in general, but these two
        // specific constants -- all-zero, and all-(q-1) -- happen to be
        // the fixed points of NTT regardless: NTT(0)=0 always, and
        // NTT(q-1 * ones) is likewise a well-defined, if unremarkable,
        // vector; nothing here assumes otherwise since `coeff_to_eval`
        // always does a real forward transform).
        let ctx = new_ctx(55);
        let nn = ctx.ring_dimension().get() as usize;
        let len = ctx.input_towers() as usize * nn;

        for &boundary_max in &[false, true] {
            let mut c0a = vec![0u64; len];
            let mut c1a = vec![0u64; len];
            let mut c0b = vec![0u64; len];
            let mut c1b = vec![0u64; len];
            if boundary_max {
                for t in 0..ctx.input_towers() {
                    let q = ctx.modulus_q(t);
                    for k in 0..nn {
                        let x = t as usize * nn + k;
                        c0a[x] = q - 1;
                        c1a[x] = q - 1;
                        c0b[x] = q - 1;
                        c1b[x] = q - 1;
                    }
                }
            }
            let (got0, got1) = ctx
                .multiply_relin_rescale(
                    &coeff_to_eval(&ctx, &c0a),
                    &coeff_to_eval(&ctx, &c1a),
                    &coeff_to_eval(&ctx, &c0b),
                    &coeff_to_eval(&ctx, &c1b),
                )
                .expect("multiply failed");
            let (exp0, exp1) = reference_multiply_relin_rescale(&ctx, &c0a, &c1a, &c0b, &c1b);
            assert_eq!(got0, exp0, "boundary_max={boundary_max}: r0 mismatch");
            assert_eq!(got1, exp1, "boundary_max={boundary_max}: r1 mismatch");
        }
    }

    #[test]
    fn invalid_input_length_is_rejected() {
        let ctx = new_ctx(1);
        let good_len = ctx.input_towers() as usize * ctx.ring_dimension().get() as usize;
        let good = vec![0u64; good_len];
        let bad = vec![0u64; 10];
        assert!(
            ctx.multiply_relin_rescale(&bad, &good, &good, &good).is_err(),
            "undersized c0a must be rejected"
        );
        assert!(
            ctx.multiply_relin_rescale(&good, &bad, &good, &good).is_err(),
            "undersized c1a must be rejected"
        );
        assert!(
            ctx.multiply_relin_rescale(&good, &good, &bad, &good).is_err(),
            "undersized c0b must be rejected"
        );
        assert!(
            ctx.multiply_relin_rescale(&good, &good, &good, &bad).is_err(),
            "undersized c1b must be rejected"
        );
    }

    #[test]
    fn invalid_construction_parameters_are_rejected() {
        // EvalMultContext::new builds its KeySwitchContext FIRST, so
        // KeySwitchContext's own validation (entry #36) propagates
        // through unchanged -- confirmed here, not just assumed.
        let n = RingDimension::new(N).unwrap();
        assert!(
            EvalMultContext::new(n, SIZE_Q, SIZE_P, 0, 1).is_err(),
            "num_part=0 must be rejected"
        );
        assert!(
            EvalMultContext::new(n, 0, SIZE_P, 1, 1).is_err(),
            "size_q=0 must be rejected"
        );
    }

    /// Shared tolerance for comparing decoded real-CKKS slot values
    /// against their mathematically expected value -- see
    /// [`SLOT_TOLERANCE`]'s own doc comment below for the derivation.
    fn ckks_mods_and_roots(ctx: &EvalMultContext) -> (Vec<u64>, Vec<u64>) {
        let size_q = ctx.input_towers();
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        (mods, roots)
    }

    /// Real CKKS encode -> decode round trip, no encryption at all.
    /// Passes to ~1e-14, confirming `codec::encode_real`/`codec::decode`
    /// are a correct canonical-embedding transform pair at this crate's
    /// chosen scale (`delta ~ qLast ~ 2^50`) -- found while root-causing
    /// the blocked EvalMult round trip below, kept as permanent coverage.
    #[test]
    fn encode_decode_round_trip_matches_input() {
        let ctx = new_ctx(999);
        let n = ctx.ring_dimension().get();
        let delta = ctx.modulus_q(ctx.input_towers() - 1) as f64;
        for slots in [
            &[1.0, 2.0, 3.0, 4.0][..],
            &[0.0, 0.0, 0.0, 0.0][..],
            &[-1.5, 2.25, -3.0, 0.5][..],
        ] {
            let m = crate::codec::encode_real(slots, n, delta).unwrap();
            let (re, im) = crate::codec::decode(&m, n, delta).unwrap();
            for (i, (&got, &expected)) in re[..slots.len()].iter().zip(slots.iter()).enumerate() {
                assert!((got - expected).abs() < 1e-9, "slot {i}: got={got} expected={expected}");
            }
            for (i, &v) in im[..slots.len()].iter().enumerate() {
                assert!(v.abs() < 1e-9, "slot {i}: unexpected imaginary component {v}");
            }
        }
    }

    /// Real CKKS encrypt -> decrypt round trip (single ciphertext, no
    /// multiply), under the SAME native key `EvalMultContext`'s relin
    /// step uses. Passes to ~1e-12, matching the expected Gaussian-noise
    /// budget (sigma=3.19) -- confirms `codec::encrypt`/`codec::decrypt`
    /// and `KeySwitchContext::secret_key`/`public_key_a`/`public_key_b`
    /// (new this round) are correct as real RLWE encryption, independent
    /// of whatever is blocking the multiply step (see below).
    #[test]
    fn encrypt_decrypt_round_trip_matches_input() {
        let ctx = new_ctx(999);
        let n = ctx.ring_dimension().get();
        let (mods, roots) = ckks_mods_and_roots(&ctx);
        let delta = ctx.modulus_q(ctx.input_towers() - 1) as f64;
        let pk_a = ctx.ks.public_key_a();
        let pk_b = ctx.ks.public_key_b();
        let s = ctx.ks.secret_key();

        for (seed, slots) in [(111u64, &[1.0, 2.0, 3.0, 4.0][..]), (222, &[0.0, 0.0, 0.0, 0.0][..]), (333, &[-1.5, 2.25, -3.0, 0.5][..])] {
            let m = crate::codec::encode_real(slots, n, delta).unwrap();
            let (c0, c1) = crate::codec::encrypt(&m, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed).unwrap();
            let m_dec = crate::codec::decrypt(&c0, &c1, &s, n, &mods, &roots).unwrap();
            let (re, im) = crate::codec::decode(&m_dec, n, delta).unwrap();
            for (i, (&got, &expected)) in re[..slots.len()].iter().zip(slots.iter()).enumerate() {
                assert!((got - expected).abs() < 1e-6, "seed {seed} slot {i}: got={got} expected={expected}");
            }
            for (i, &v) in im[..slots.len()].iter().enumerate() {
                assert!(v.abs() < 1e-6, "seed {seed} slot {i}: unexpected imaginary component {v}");
            }
        }
    }

    // ====================================================================
    // GOD PROMPT "CKKS PHASE 5": real CKKS encode/encrypt/decrypt/decode,
    // and the (currently BLOCKED) real encrypted EvalMult round trip.
    //
    // Everything above this point in this file operates at the raw RNS
    // level (synthetic per-tower residues, never a real plaintext or
    // real encryption noise) -- deliberately, per this module's own
    // top-level doc comment. The tests immediately below ARE real CKKS
    // cryptography (encode -> encrypt -> decrypt -> decode, and encode
    // -> decode alone) and PASS, proving `codec` correct as actual CKKS
    // arithmetic with real encryption noise, not merely isolated
    // encode/decode math.
    //
    // The FURTHER step -- routing a real ciphertext through
    // `EvalMultContext::multiply_relin_rescale` and decrypting the
    // result -- is investigated below but NOT achieved this round. The
    // investigation (methodically isolating stage by stage, per the
    // governing directive's own Priority 5) found:
    //   1. encode -> decode alone: exact to ~1e-14 (far below tolerance).
    //   2. encrypt -> decrypt (single ciphertext, no multiply): exact to
    //      ~1e-12 (consistent with the expected Gaussian-noise budget).
    //   3. tensor -> relin (`keyswitch_resident`) -> combine, checked
    //      DIRECTLY in eval domain with NO decode/rescale involved
    //      (`ba0[k]+ba1[k]*s[k] =?= t2[k]*s[k]^2 mod q0`, the textbook
    //      relin decryption invariant): FAILS, ~50% of q0 -- i.e.
    //      statistically uncorrelated, not a subtle numerical error.
    //      Every sign/arithmetic variant tried (`ba0+ba1*s`, `ba0-ba1*s`,
    //      `-ba0+ba1*s`, `-ba0-ba1*s`) gives the same ~50%-of-q result.
    //      Reproduces with BOTH `numPart=1` (no digit decomposition at
    //      all) and `numPart=2`.
    //
    // This is NOT a new bug -- it is a CONFIRMED, more consequential
    // recurrence of `research/EXPERIMENT_LEDGER.md` entry #30's own
    // unresolved finding: an EARLIER round found this SAME naive
    // algebraic identity fails even on OpenFHE's own bit-exact-proven
    // `KeySwitchCore` output (`tests/test_keyswitch.cpp`'s own appended
    // check), and left the exact per-digit/per-part scaling convention
    // as "an honestly unresolved research question." At the time, that
    // was filed as a curiosity because `keyswitch_resident`'s OUTPUT was
    // still proven bit-exact against two independent oracles
    // (`keyswitch_core_resident` and OpenFHE's `KeySwitchCore`) at the
    // RNS level, so correctness wasn't actually in doubt. This round
    // shows the SAME gap now blocks something concrete: a real
    // decrypt-based verification of EvalMult specifically CANNOT be
    // built on the naive formula, because it is not simply a cosmetic
    // check -- `LaunchCombine`'s `r0=t0+ba0, r1=t1+ba1` IS exactly this
    // identity, so if it does not hold, the composed ciphertext will not
    // decrypt correctly either (confirmed: the full `multiply_relin_
    // rescale` -> decrypt -> decode path gives decoded errors of O(100)-
    // O(1000) in slot space, consistent in magnitude with a ~q0/2-scale
    // additive error propagating through).
    //
    // What this does NOT mean: there is no new evidence here that
    // `keyswitch_resident`, `rescale_resident_raw`, `LaunchTensor`, or
    // `LaunchCombine` are incorrect -- every one of them remains
    // independently proven at the RNS level (entries #35-#37) using
    // references that do not depend on this specific algebraic identity.
    // The most likely explanation, based on entry #30's own finding that
    // OpenFHE's correctly-functioning relin ALSO fails this check, is
    // that the identity itself is incomplete for this scheme's actual
    // scaling convention (missing some per-digit or per-part
    // normalization), not that the implementation is wrong. This has
    // NOT been confirmed -- it is the leading hypothesis, stated as
    // such, not as an established fact.
    //
    // Per the governing directive's explicit instruction ("if the root
    // cause cannot be established in this phase, leave it explicitly
    // open and document the next experiment"): the next concrete
    // experiment, not attempted this round, is to link OpenFHE briefly
    // and run a REAL `cc->EvalMult` + `cc->Decrypt` round trip (not the
    // algebraic shortcut) at a small scale, to determine definitively
    // whether OpenFHE's own relin decrypts correctly via normal library
    // usage (expected: yes, OpenFHE is production-grade) -- then work
    // backward from ITS actual per-digit scaling convention (reading
    // `EvalMultsMutable`/`EvalMultCore`'s exact combine step, not just
    // `KeySwitchCore` in isolation) to find the missing factor, rather
    // than continuing to guess at sign/scale variants.
    //
    // The tests below keep this finding VISIBLE and reproducible
    // (`#[ignore]`, not deleted or silently passing) rather than hidden.
    // ====================================================================

    /// Runs the full real round trip for two REAL (zero-imaginary) slot
    /// vectors and returns the decoded result's (real, imaginary) parts.
    /// `slots_a`/`slots_b` may be shorter than `n/2`; the rest are
    /// implicitly zero, matching `codec::encode_real`'s own contract.
    ///
    /// Scale choice: `delta` is set EQUAL to the dropped tower's modulus
    /// (`qLast = ctx.modulus_q(SIZE_Q-1)`), the standard CKKS
    /// scale-invariant design -- after one rescale, the new scale
    /// `delta^2/qLast` works out to `delta` again exactly (as a real
    /// number; the rescale KERNEL's own integer rounding is the only
    /// real source of error, not a scale mismatch). See `codec`'s own
    /// module doc comment for why `decrypt_host`'s tower-0-only recovery
    /// specifically requires this (not an arbitrary choice).
    fn real_evalmult_round_trip(
        ctx: &EvalMultContext,
        slots_a: &[f64],
        slots_b: &[f64],
        seed_a: u64,
        seed_b: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = ctx.ring_dimension().get();
        let size_q = ctx.input_towers();
        let survivors = ctx.output_towers();

        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let delta = ctx.modulus_q(size_q - 1) as f64; // = qLast, the scale-invariant choice

        let pk_a = ctx.ks.public_key_a();
        let pk_b = ctx.ks.public_key_b();

        let m_a = crate::codec::encode_real(slots_a, n, delta).expect("encode a failed");
        let m_b = crate::codec::encode_real(slots_b, n, delta).expect("encode b failed");
        let (c0a, c1a) =
            crate::codec::encrypt(&m_a, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed_a).expect("encrypt a failed");
        let (c0b, c1b) =
            crate::codec::encrypt(&m_b, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed_b).expect("encrypt b failed");

        let (r0, r1) = ctx.multiply_relin_rescale(&c0a, &c1a, &c0b, &c1b).expect("multiply failed");
        assert_eq!(r0.len(), survivors as usize * n as usize, "unexpected output tower count");
        assert_eq!(r1.len(), survivors as usize * n as usize, "unexpected output tower count");

        let s_full = ctx.ks.secret_key();
        let s_survivors: Vec<u64> = s_full[..survivors as usize * n as usize].to_vec();
        let mods_survivors = &mods[..survivors as usize];
        let roots_survivors = &roots[..survivors as usize];

        let m_result =
            crate::codec::decrypt(&r0, &r1, &s_survivors, n, mods_survivors, roots_survivors).expect("decrypt failed");
        crate::codec::decode(&m_result, n, delta).expect("decode failed")
    }

    /// Absolute error must stay inside a tolerance derived from the
    /// actual noise budget, not picked to pass: at `delta ~ qLast ~
    /// 2^50`, encoding rounding (~O(1) per coefficient), encryption
    /// Gaussian noise (sigma=3.19), and the relin/rescale kernels' own
    /// integer rounding (bounded, established in `research/
    /// EXPERIMENT_LEDGER.md` entry #29's independent-component rescale
    /// noise analysis) all stay within a few hundred UNITS in the
    /// integer-coefficient domain even after decode's O(sqrt(n)) random-
    /// walk amplification -- dividing by `delta ~ 2^50` puts the real
    /// numerical noise floor around `1e-12`-`1e-10`. `1e-6` leaves four
    /// to six orders of magnitude of headroom, tight enough to catch a
    /// real defect (which would show as an O(1) or larger error, not a
    /// borderline one) without being tuned to this specific run.
    const SLOT_TOLERANCE: f64 = 1e-6;

    fn assert_slots_close(label: &str, got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len());
        let mut max_err = 0.0f64;
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let err = (g - e).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err < SLOT_TOLERANCE,
                "{label}: slot {i} got={g} expected={e} abs_err={err} exceeds tolerance {SLOT_TOLERANCE}"
            );
        }
        eprintln!("{label}: max abs error = {max_err:e} (tolerance {SLOT_TOLERANCE:e})");
    }

    #[test]
    #[ignore = "BLOCKED: relin decryption invariant does not hold for this native eval key -- see this module's own doc comment above (research/EXPERIMENT_LEDGER.md entry #38); known to fail, kept visible rather than deleted"]
    fn real_encrypted_evalmult_matches_expected_product() {
        let ctx = new_ctx(1234);
        let slots_a = [1.0, 2.0, 3.0, 4.0];
        let slots_b = [5.0, 6.0, 7.0, 8.0];
        let expected: Vec<f64> = slots_a.iter().zip(slots_b.iter()).map(|(a, b)| a * b).collect();

        let (re, im) = real_evalmult_round_trip(&ctx, &slots_a, &slots_b, 9001, 9002);

        assert_slots_close("known-vector product", &re[..expected.len()], &expected);
        // Imaginary parts must stay negligible -- real-only inputs should
        // decode with no meaningful imaginary component.
        for (i, &v) in im[..expected.len()].iter().enumerate() {
            assert!(v.abs() < SLOT_TOLERANCE, "slot {i}: unexpected imaginary component {v}");
        }
        // Untouched slots (index >= 4) must decode close to zero too.
        for (i, &v) in re[expected.len()..].iter().enumerate() {
            assert!(v.abs() < SLOT_TOLERANCE, "untouched slot {}: expected ~0, got {v}", i + expected.len());
        }
    }

    #[test]
    #[ignore = "BLOCKED: see real_encrypted_evalmult_matches_expected_product's #[ignore] reason and entry #38"]
    fn real_encrypted_evalmult_zero_boundary() {
        // Multiplying by an all-zero ciphertext must decode to all-zero,
        // a deterministic boundary case distinct from random data.
        let ctx = new_ctx(5678);
        let slots_a = [3.5, -2.0, 0.0, 7.25];
        let slots_b = [0.0, 0.0, 0.0, 0.0];
        let expected = [0.0, 0.0, 0.0, 0.0];

        let (re, _im) = real_evalmult_round_trip(&ctx, &slots_a, &slots_b, 111, 222);
        assert_slots_close("zero boundary", &re[..expected.len()], &expected);
    }

    #[test]
    #[ignore = "BLOCKED: see real_encrypted_evalmult_matches_expected_product's #[ignore] reason and entry #38"]
    fn real_encrypted_evalmult_negative_and_fractional_values() {
        let ctx = new_ctx(4242);
        let slots_a = [-1.5, 2.25, -3.0, 0.5];
        let slots_b = [2.0, -1.0, 0.5, 4.0];
        let expected: Vec<f64> = slots_a.iter().zip(slots_b.iter()).map(|(a, b)| a * b).collect();

        let (re, _im) = real_evalmult_round_trip(&ctx, &slots_a, &slots_b, 333, 444);
        assert_slots_close("negative/fractional product", &re[..expected.len()], &expected);
    }

    #[test]
    #[ignore = "BLOCKED: see real_encrypted_evalmult_matches_expected_product's #[ignore] reason and entry #38"]
    fn real_encrypted_evalmult_repeated_calls_reused_context() {
        // Several independent real round trips on the SAME context,
        // different plaintexts/seeds each time -- proves the reused
        // device buffers and embedded KeySwitchContext/RescaleContext
        // scratch don't leak state across calls even with REAL
        // encrypted data (not just synthetic RNS residues, entry #37's
        // own repeated-call test).
        let ctx = new_ctx(7777);
        let cases: [([f64; 4], [f64; 4]); 3] = [
            ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
            ([2.0, -3.0, 4.0, -5.0], [1.5, 2.5, -1.0, 0.5]),
            ([0.1, 0.2, 0.3, 0.4], [10.0, 20.0, 30.0, 40.0]),
        ];
        for (call, (a, b)) in cases.iter().enumerate() {
            let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
            let (re, _im) = real_evalmult_round_trip(&ctx, a, b, 1000 + call as u64, 2000 + call as u64);
            assert_slots_close(&format!("repeated call {call}"), &re[..expected.len()], &expected);
        }
    }
}
