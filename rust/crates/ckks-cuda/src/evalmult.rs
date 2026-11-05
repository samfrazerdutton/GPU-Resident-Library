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

    /// GOD PROMPT "CKKS PHASE 22": the public key's `a`/`b` components
    /// and the secret key, forwarded from the embedded
    /// `KeySwitchContext` -- added because a caller OUTSIDE this
    /// module (e.g. `codec`'s own tests, composing `codec::encrypt`/
    /// `codec::encrypt_registered` with `multiply_relin_rescale` for a
    /// real encode->encrypt->EvalMult->decrypt->decode round trip) has
    /// no other supported way to reach these keys: `ks` itself is, and
    /// stays, private (this context owns it, nothing outside should
    /// mutate or outlive it independently), so these are plain,
    /// read-only pass-throughs, not a relaxation of that ownership.
    /// `this module's own tests` used `ctx.ks.public_key_a()` etc.
    /// directly, which only works from inside this module -- these
    /// three methods are the minimal, justified fix for every OTHER
    /// module that needs the same real keys this context's own relin
    /// step already uses internally.
    pub fn public_key_a(&self) -> Vec<u64> {
        self.ks.public_key_a()
    }
    pub fn public_key_b(&self) -> Vec<u64> {
        self.ks.public_key_b()
    }
    pub fn secret_key(&self) -> Vec<u64> {
        self.ks.secret_key()
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

    /// GOD PROMPT "CKKS PHASE 6": TEST-ONLY diagnostic. Identical to
    /// `multiply_relin_rescale` up through `LaunchCombine`, but STOPS
    /// there -- returns the combined, UN-RESCALED (t0+ba0, t1+ba1), each
    /// still `input_towers() * n` elements (full `size_ql`, not
    /// `size_ql-1`). Exists to isolate whether a real decode-level
    /// failure in the full pipeline originates in [tensor+keyswitch+
    /// combine] or in [rescale]/the composition around it -- same
    /// diagnostic technique that confirmed (via a parallel OpenFHE-based
    /// investigation, see `tests/test_keyswitch.cpp`'s "real-data
    /// control" block) that `keyswitch_resident`/`LaunchCombine` satisfy
    /// the textbook relin identity when checked correctly.
    #[cfg(test)]
    pub(crate) fn multiply_relin_no_rescale_for_test(
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

            let rc = cudaDeviceSynchronize();
            if rc != 0 {
                return Err(CkksError::Cuda(format!(
                    "cudaDeviceSynchronize failed, cudaError_t={rc}"
                )));
            }
        }

        let out_len = self.input_towers() as usize * self.n.get() as usize;
        let mut out0 = vec![0u64; out_len];
        let mut out1 = vec![0u64; out_len];
        self.r0.download_u64(&mut out0)?;
        self.r1.download_u64(&mut out1)?;
        Ok((out0, out1))
    }

    /// GOD PROMPT "CKKS PHASE 7": TEST-ONLY diagnostic, narrower than
    /// `multiply_relin_no_rescale_for_test`. Stops after `LaunchTensor`
    /// alone -- no keyswitch, no combine. Returns the RAW tensor
    /// components `(t0, t1, t2)`, each full `size_ql` towers. Exists to
    /// isolate [tensor product itself] from [keyswitch/relin], since
    /// Phase 7's Control A found the combined, relinearized result does
    /// NOT match the expected product for REAL ciphertext data under
    /// this crate's own native eval key -- narrowing further requires
    /// checking the raw tensor identity (`t0+t1*s+t2*s^2 ~= m_a*m_b`)
    /// BEFORE relin even touches the data.
    #[cfg(test)]
    pub(crate) fn tensor_only_for_test(
        &self,
        c0a: &[u64],
        c1a: &[u64],
        c0b: &[u64],
        c1b: &[u64],
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>), CkksError> {
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
            let rc = cudaDeviceSynchronize();
            if rc != 0 {
                return Err(CkksError::Cuda(format!(
                    "cudaDeviceSynchronize failed, cudaError_t={rc}"
                )));
            }
        }

        let out_len = self.input_towers() as usize * self.n.get() as usize;
        let mut out0 = vec![0u64; out_len];
        let mut out1 = vec![0u64; out_len];
        let mut out2 = vec![0u64; out_len];
        self.t0.download_u64(&mut out0)?;
        self.t1.download_u64(&mut out1)?;
        self.t2.download_u64(&mut out2)?;
        Ok((out0, out1, out2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    // GOD PROMPT "CKKS PHASE 9": NUM_PART was 2 (alpha=4) until this round.
    // Entry #42 root-caused the real-encrypted-EvalMult blocker to THIS
    // choice, not to SIZE_P or any key-switch code: with NUM_PART=2, each
    // digit covers alpha=4 Q towers (Q_part_max ~210 bits: one 60-bit +
    // three 50-bit towers), while P = 2*60-bit = 120 bits -- violating the
    // Hybrid key-switch's REQUIRED P >= Q_part_max correctness condition
    // (P must be at least as large as the largest digit's own modulus
    // product, a standard RNS-CKKS requirement, independent of this
    // project's own code). That deficit (-90 bits) makes the decompose
    // step's bounded approximation error, once multiplied through the
    // gadget/eval-key structure and divided by P in ModDown, land near
    // q/2 instead of negligible -- the "~50% of q, looks random" signature
    // every earlier round saw. NUM_PART=4 (alpha=2, Q_part_max ~110 bits)
    // restores a +10-bit margin over P -- confirmed via
    // tests/test_ks_pq_ratio_sweep.cpp to drop the relin-identity residual
    // from ~50% of q to ~6e-12% of q, with SIZE_Q, SIZE_P, n, and the
    // secret distribution (h=64) all UNCHANGED -- this is a digit-grouping
    // choice with no security-parameter impact (log2(QP) and n are
    // unaffected; see research/EXPERIMENT_LEDGER.md entry #42).
    const N: u32 = 1024;
    const SIZE_Q: u32 = 8;
    const SIZE_P: u32 = 2;
    const NUM_PART: u32 = 4;

    fn new_ctx(seed: u64) -> EvalMultContext {
        let n = RingDimension::new(N).unwrap();
        EvalMultContext::new(n, SIZE_Q, SIZE_P, NUM_PART, seed).expect("context init failed")
    }

    /// GOD PROMPT "CKKS PHASE 7": same fixture, but `num_part=1`
    /// (eliminates digit decomposition -- a single "part" covering all
    /// `size_ql` towers at once). Entry #38 tried this already, but
    /// with the WRONG (eval-domain-pointwise) comparison methodology,
    /// so its "numPart=1 also fails ~50%" result was itself an
    /// artifact, not real evidence -- revisiting here, with the
    /// corrected (coefficient-domain) comparison and a REAL encrypted
    /// t2, is newly justified, not a blind re-run.
    fn new_ctx_num_part(seed: u64, num_part: u32) -> EvalMultContext {
        let n = RingDimension::new(N).unwrap();
        EvalMultContext::new(n, SIZE_Q, SIZE_P, num_part, seed).expect("context init failed")
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

    // GOD PROMPT "CKKS PHASE 10": the FORMER default fixture (NUM_PART=2)
    // was invalid per the Hybrid key-switch's P>=Q_part_max requirement
    // (research/EXPERIMENT_LEDGER.md entry #42) -- this pins down that it
    // is now REJECTED at construction time (entry #43's `ks_context_
    // create` validation, propagated through `KeySwitchContext::new`
    // which `EvalMultContext::new` builds first), independent of this
    // module's own `NUM_PART` constant having already moved to 4.
    #[test]
    fn former_invalid_num_part_2_fixture_is_now_rejected() {
        let n = RingDimension::new(N).unwrap();
        let result = EvalMultContext::new(n, SIZE_Q, SIZE_P, 2, 1);
        let err = match result {
            Ok(_) => panic!("the old NUM_PART=2 fixture (SIZE_Q=8, SIZE_P=2) violates P>=Q_part_max and must be rejected"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("Q_part_max") || msg.contains("Hybrid key-switch"),
            "rejection error should name the actual violated condition, got: {msg}"
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
    // GOD PROMPT "CKKS PHASE 5"/"PHASE 6": real CKKS encode/encrypt/
    // decrypt/decode, and the (currently BLOCKED, but for a DIFFERENT
    // and narrower reason than originally suspected) real encrypted
    // EvalMult round trip.
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
    // result -- is investigated below and still NOT achieved, but the
    // ORIGINAL suspected cause is now RESOLVED (PHASE 6,
    // `research/EXPERIMENT_LEDGER.md` entry #39):
    //   1. encode -> decode alone: exact to ~1e-14 (far below tolerance).
    //   2. encrypt -> decrypt (single ciphertext, no multiply): exact to
    //      ~1e-12 (consistent with the expected Gaussian-noise budget).
    //   3. tensor -> relin (`keyswitch_resident`) -> combine: Phase 5
    //      checked `ba0[k]+ba1[k]*s[k] =?= t2[k]*s[k]^2 mod q0` DIRECTLY
    //      by comparing raw per-coefficient values in NTT (eval) domain,
    //      and found it FAILS at ~50% of q0. Phase 6 proved this was a
    //      TEST METHODOLOGY BUG, not a real gap: ring multiplication is
    //      pointwise ONLY in NTT/evaluation-domain representation; NTT
    //      is bijective but highly MIXING, so a tiny coefficient-domain
    //      difference (genuine, small key-switch noise) spreads into
    //      what looks like a large, pseudo-random difference at every
    //      individual NTT point. Comparing raw values pointwise in eval
    //      domain is therefore not a valid "are these nearly the same
    //      polynomial" test. Checked properly (convert to coefficient
    //      domain, i.e. an actual INTT, before comparing, exactly
    //      mirroring how `cc->Decrypt` itself always converts first):
    //      the identity holds to ~1e-10% of q for real ciphertext data
    //      -- see `tests/test_evalmult_relin_openfhe_reference.cpp`
    //      (new, OpenFHE-only cross-validation) and the "real-data
    //      control" experiment added to `tests/test_keyswitch.cpp`.
    //
    // **`keyswitch_resident`'s `ba0`/`ba1` output and `LaunchCombine`'s
    // `r0=t0+ba0, r1=t1+ba1` formula were correct all along.** This
    // closes `research/EXPERIMENT_LEDGER.md` entry #30's long-open
    // "unresolved research question" and this module's own entry #38
    // finding -- both were the SAME test-methodology artifact, not a
    // missing sign, scale factor, per-digit normalization, or basis-
    // conversion term, and not a gap in understanding this scheme's
    // algebraic convention. Do not re-open this question or re-run the
    // sign/numPart/P-factor sweeps entry #38 tried -- they were chasing
    // an artifact, not a real gap.
    //
    // **What remains BLOCKED, and is a DIFFERENT, narrower bug**: the
    // full `multiply_relin_rescale` -> `codec::decrypt` -> `codec::
    // decode` round trip below still produces a genuine, large decode-
    // level error (not explained by the methodology bug above, since
    // `codec::decrypt`/`decode` already convert domains correctly).
    // `LaunchTensor`, `keyswitch_device_resident`/`LaunchCombine`, and
    // `rescale_device_resident` all remain independently proven correct
    // in isolation (entries #35-37, RNS level; entry #39, real-data,
    // decrypt-level for keyswitch+combine) -- and this exact fused
    // composition is ALSO already proven bit-exact against a step-by-
    // step reference for SYNTHETIC RNS data
    // (`evalmult_matches_independent_reference_random_data`, Phase 4).
    // This narrows the remaining bug to something specific to REAL,
    // CKKS-scaled ciphertext data (large dynamic range, Δ²-scaled
    // before rescale), most plausibly in `rescale_device_resident`'s
    // behavior at that magnitude or in the test's own scale/tower-count
    // bookkeeping -- not a generic buffer-reuse/sequencing hazard, which
    // would also break the synthetic-data differential test.
    //
    // **Next concrete experiment, not yet attempted** (see
    // `research/NEXT_SESSION.md`'s "READ THIS FIRST" section for full
    // detail): use the new TEST-ONLY
    // `EvalMultContext::multiply_relin_no_rescale_for_test` method
    // (returns the combined, UN-rescaled result at full tower count) to
    // isolate [tensor+keyswitch+combine] from [rescale] directly on
    // real ciphertext data -- either by building a decode capable of
    // handling its `delta^2`-scaled, full-tower-count value, or by
    // reproducing Phase 4's own step-by-step cross-check methodology
    // with real (not synthetic) inputs feeding `rescale_device_resident`
    // specifically.
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

    // Sanity check on THIS TEST's OWN methodology (not the multiply
    // pipeline): for a SINGLE, un-multiplied, fresh ciphertext, does
    // c0+c1*s (eval domain, pointwise, computed by hand here -- NOT via
    // codec::decrypt) match the known encoded plaintext once converted
    // to coefficient domain? If this ALSO shows ~50% of q, the bug is in
    // this test's own eval-domain/root/layout handling, not in anything
    // upstream of rescale -- must be checked BEFORE trusting Control A's
    // result above.
    fn sanity_check_single_ciphertext_eval_domain_methodology(ctx: &EvalMultContext, slots: &[f64], seed: u64) -> f64 {
        let n = ctx.ring_dimension().get();
        let size_q = ctx.input_towers();
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let delta = ctx.modulus_q(size_q - 1) as f64;

        let pk_a = ctx.ks.public_key_a();
        let pk_b = ctx.ks.public_key_b();
        let s_full = ctx.ks.secret_key();

        let m = crate::codec::encode_real(slots, n, delta).expect("encode failed");
        let (c0, c1) = crate::codec::encrypt(&m, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed).expect("encrypt failed");

        let nn = n as usize;
        let mut worst_frac_of_q = 0.0f64;
        for t in 0..size_q as usize {
            let q = mods[t];
            let root = roots[t];
            let reduce_mod_q = |x: i64| -> u64 { (((x % q as i64) + q as i64) % q as i64) as u64 };
            let m_mod_q: Vec<u64> = m[..nn].iter().map(|&x| reduce_mod_q(x)).collect();

            let mut actual_eval = vec![0u64; nn];
            for k in 0..nn {
                let idx = t * nn + k;
                actual_eval[k] = addmod(c0[idx], mulmod(c1[idx], s_full[idx], q), q);
            }
            let actual_coeff = inverse_ntt(&actual_eval, n, q, root);

            let mut worst = 0u64;
            for k in 0..nn {
                let diff = if actual_coeff[k] >= m_mod_q[k] { actual_coeff[k] - m_mod_q[k] } else { m_mod_q[k] - actual_coeff[k] };
                let centered = diff.min(q - diff);
                if centered > worst {
                    worst = centered;
                }
            }
            let frac = worst as f64 / q as f64;
            eprintln!("[sanity] tower={t} q={q} worst|c0+c1*s - m|={worst} ({:.6e}% of q)", frac * 100.0);
            if frac > worst_frac_of_q {
                worst_frac_of_q = frac;
            }
        }
        worst_frac_of_q
    }

    #[test]
    fn sanity_check_single_ciphertext_eval_domain_methodology_test() {
        let ctx = new_ctx(1234);
        let slots = [1.0, 2.0, 3.0, 4.0];
        let worst_frac = sanity_check_single_ciphertext_eval_domain_methodology(&ctx, &slots, 9001);
        eprintln!("[sanity] worst fraction of q across all towers: {:.6e}%", worst_frac * 100.0);
        assert!(
            worst_frac < 0.01,
            "THIS TEST's own eval-domain methodology is broken (worst_frac={:.6}% of q) -- \
             fix this before trusting any Control A/B result built on it.",
            worst_frac * 100.0
        );
    }

    #[test]
    fn sanity_check_both_control_a2_ciphertexts_decrypt_individually() {
        let ctx = new_ctx(1234);
        let fa = sanity_check_single_ciphertext_eval_domain_methodology(&ctx, &[1.0, 2.0, 3.0, 4.0], 9001);
        eprintln!("[sanity a] worst fraction of q: {:.6e}%", fa * 100.0);
        let fb = sanity_check_single_ciphertext_eval_domain_methodology(&ctx, &[5.0, 6.0, 7.0, 8.0], 9002);
        eprintln!("[sanity b] worst fraction of q: {:.6e}%", fb * 100.0);
        assert!(fa < 0.01, "ciphertext a (seed 9001) does not decrypt correctly on its own: {:.6}% of q", fa * 100.0);
        assert!(fb < 0.01, "ciphertext b (seed 9002) does not decrypt correctly on its own: {:.6}% of q", fb * 100.0);
    }

    // Control A, CORRECTED: an earlier version compared the pre-rescale
    // combined result against the NOISE-FREE expected product
    // (m_a_eval*m_b_eval) -- WRONG for the same reason Control A2's
    // first draft was wrong (see its own comment): the raw tensor's
    // cross-noise terms are genuinely large pre-rescale, so comparing
    // against a noise-free baseline fails for ANY correct
    // implementation. The MEANINGFUL comparison is: does relin's own
    // combine output match the EXACT raw tensor identity it is supposed
    // to approximate? I.e. (t0+ba0)+(t1+ba1)*s should be CLOSE to
    // t0+t1*s+t2*s^2 (both computed directly from ciphertext data, no
    // plaintext involved) -- the ONLY expected discrepancy here is the
    // keyswitch's OWN noise (entry #39 proved this is small, ~1e-10% of
    // q, for OpenFHE's eval key; this checks it for THIS crate's own
    // native eval key with a REAL encrypted t2, which was never
    // independently exercised before -- entry #36's own stated gap).
    // Compared in COEFFICIENT domain (an actual INTT), per entry #39's
    // methodology lesson.
    fn control_a_relin_matches_raw_tensor_within_keyswitch_noise(ctx: &EvalMultContext, slots_a: &[f64], slots_b: &[f64], seed_a: u64, seed_b: u64) -> f64 {
        let n = ctx.ring_dimension().get();
        let size_q = ctx.input_towers();
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let delta = ctx.modulus_q(size_q - 1) as f64;

        let pk_a = ctx.ks.public_key_a();
        let pk_b = ctx.ks.public_key_b();
        let s_full = ctx.ks.secret_key();

        let m_a = crate::codec::encode_real(slots_a, n, delta).expect("encode a failed");
        let m_b = crate::codec::encode_real(slots_b, n, delta).expect("encode b failed");
        let (c0a, c1a) =
            crate::codec::encrypt(&m_a, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed_a).expect("encrypt a failed");
        let (c0b, c1b) =
            crate::codec::encrypt(&m_b, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed_b).expect("encrypt b failed");

        let (t0, t1, t2) = ctx.tensor_only_for_test(&c0a, &c1a, &c0b, &c1b).expect("tensor-only failed");
        let (pre_r0, pre_r1) = ctx
            .multiply_relin_no_rescale_for_test(&c0a, &c1a, &c0b, &c1b)
            .expect("pre-rescale multiply failed");
        assert_eq!(pre_r0.len(), size_q as usize * n as usize, "pre-rescale output must keep full tower count");
        assert_eq!(pre_r1.len(), size_q as usize * n as usize, "pre-rescale output must keep full tower count");

        let nn = n as usize;
        let mut worst_frac_of_q = 0.0f64;
        for t in 0..size_q as usize {
            let q = mods[t];
            let root = roots[t];

            // Expected side: the EXACT raw tensor identity, computed
            // directly from the ciphertext data (already proven exact
            // in control_a2) -- NOT the noise-free plaintext product.
            let mut expected_eval = vec![0u64; nn];
            for k in 0..nn {
                let idx = t * nn + k;
                let s = s_full[idx];
                let s2 = mulmod(s, s, q);
                expected_eval[k] = addmod(addmod(t0[idx], mulmod(t1[idx], s, q), q), mulmod(t2[idx], s2, q), q);
            }

            // Actual side: the GPU pipeline's own pre-rescale combined
            // (relinearized) result, decrypted (eval domain, pointwise).
            let mut actual_eval = vec![0u64; nn];
            for k in 0..nn {
                let idx = t * nn + k;
                actual_eval[k] = addmod(pre_r0[idx], mulmod(pre_r1[idx], s_full[idx], q), q);
            }

            // Compare in COEFFICIENT domain ONLY -- an eval-domain
            // pointwise comparison here would reproduce entry #39's
            // original artifact regardless of whether this is correct.
            let expected_coeff = inverse_ntt(&expected_eval, n, q, root);
            let actual_coeff = inverse_ntt(&actual_eval, n, q, root);

            let mut worst = 0u64;
            for k in 0..nn {
                let diff = if actual_coeff[k] >= expected_coeff[k] {
                    actual_coeff[k] - expected_coeff[k]
                } else {
                    expected_coeff[k] - actual_coeff[k]
                };
                let centered = diff.min(q - diff);
                if centered > worst {
                    worst = centered;
                }
            }
            let frac = worst as f64 / q as f64;
            eprintln!(
                "[control A] tower={t} q={q} worst|(t0+ba0)+(t1+ba1)*s - (t0+t1*s+t2*s^2)|={worst} ({:.6e}% of q)",
                frac * 100.0
            );
            if frac > worst_frac_of_q {
                worst_frac_of_q = frac;
            }
        }
        worst_frac_of_q
    }

    #[test]
    // GOD PROMPT "CKKS PHASE 9": un-ignored. Entry #41's "sizeP=2-specific
    // key-switch reconstruction defect" was itself superseded by entry #42:
    // the real cause was NUM_PART=2 (alpha=4) giving each digit a modulus
    // product (~210 bits) LARGER than P (2*60-bit=120 bits), violating the
    // Hybrid key-switch's P>=Q_part_max requirement -- not a code defect in
    // ks_constants.cpp/keyswitch.cpp/basis_convert.cu/moddown.cu, all of
    // which were independently proven correct in isolation this round
    // (tests/test_moddown_crt_basis_isolated.cpp, tests/test_decompose_crt_
    // basis_isolated.cpp, tests/test_moddown_combine_isolated.cpp). Raising
    // NUM_PART to 4 (alpha=2, Q_part_max ~110 bits, comfortably under P)
    // fixes it with SIZE_Q/SIZE_P/n/h unchanged. See entry #42.
    fn control_a_relin_combine_matches_raw_tensor_within_keyswitch_noise_test() {
        let ctx = new_ctx(1234);
        let slots_a = [1.0, 2.0, 3.0, 4.0];
        let slots_b = [5.0, 6.0, 7.0, 8.0];
        let worst_frac = control_a_relin_matches_raw_tensor_within_keyswitch_noise(&ctx, &slots_a, &slots_b, 9001, 9002);
        eprintln!("[control A] worst fraction of q across all towers: {:.6e}%", worst_frac * 100.0);
        // 1% of q is many orders of magnitude above any plausible
        // keyswitch noise (entry #39 found ~5.5e-11% of q for the
        // analogous OpenFHE-sourced check) and many orders below "looks
        // like uncorrelated randomness" (~50% of q) -- decisively
        // distinguishes "small real noise" from "a genuine defect in
        // keyswitch/relin for this crate's own native eval key."
        assert!(
            worst_frac < 0.01,
            "relin's combine output does NOT match the raw tensor identity within keyswitch noise -- \
             worst_frac={:.6}% of q. This means keyswitch_device_resident/LaunchCombine has a genuine defect \
             for real encrypted data under this crate's own native eval key.",
            worst_frac * 100.0
        );
    }

    #[test]
    #[ignore = "EXPECTED TO FAIL, kept as a documented negative case (see entry #42): num_part=1 means alpha=size_q=8, so the single part's own modulus product (Q_part ~410 bits: one 60-bit + seven 50-bit towers) vastly exceeds SIZE_P=2's P (~120 bits), badly violating the Hybrid key-switch's P>=Q_part_max requirement. This is NOT evidence that digit decomposition is irrelevant -- it is the opposite: entry #42 found splitting Q into MORE, SMALLER parts (not fewer) is what keeps each part's own modulus under P. Do not un-ignore this test without also increasing SIZE_P enough to cover Q_part_max at num_part=1 (~410 bits), which would change security-relevant parameters and is out of scope here."]
    fn control_a_relin_combine_matches_raw_tensor_num_part_1_test() {
        let ctx = new_ctx_num_part(1234, 1);
        let slots_a = [1.0, 2.0, 3.0, 4.0];
        let slots_b = [5.0, 6.0, 7.0, 8.0];
        let worst_frac = control_a_relin_matches_raw_tensor_within_keyswitch_noise(&ctx, &slots_a, &slots_b, 9001, 9002);
        eprintln!("[control A, num_part=1] worst fraction of q across all towers: {:.6e}%", worst_frac * 100.0);
        assert!(
            worst_frac < 0.01,
            "with num_part=1 (no digit decomposition), relin's combine output STILL does NOT match the raw \
             tensor identity -- worst_frac={:.6}% of q. This rules out digit decomposition as the cause.",
            worst_frac * 100.0
        );
    }

    // Control B/C hybrid: does the FAST device-resident keyswitch path
    // (`keyswitch_device_resident`, used by `EvalMultContext`) agree
    // BIT-EXACT with the HOST-ORCHESTRATED reference oracle
    // (`keyswitch_reference`/`keyswitch_core_resident`) on this EXACT
    // REAL t2? Entry #36 proved these two agree for SYNTHETIC data; this
    // is the first time they are compared on a REAL encrypted t2. If
    // they agree, the bug is shared by both (most likely in
    // `evalkeygen_host`'s eval key generation itself, not a
    // device-resident-specific defect); if they disagree, the bug is
    // specific to the fast path.
    #[test]
    fn fast_keyswitch_matches_reference_keyswitch_on_real_t2() {
        let ctx = new_ctx(1234);
        let n = ctx.ring_dimension().get();
        let size_q = ctx.input_towers();
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let delta = ctx.modulus_q(size_q - 1) as f64;
        let pk_a = ctx.ks.public_key_a();
        let pk_b = ctx.ks.public_key_b();

        let m_a = crate::codec::encode_real(&[1.0, 2.0, 3.0, 4.0], n, delta).expect("encode a failed");
        let m_b = crate::codec::encode_real(&[5.0, 6.0, 7.0, 8.0], n, delta).expect("encode b failed");
        let (c0a, c1a) = crate::codec::encrypt(&m_a, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, 9001).expect("encrypt a failed");
        let (c0b, c1b) = crate::codec::encrypt(&m_b, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, 9002).expect("encrypt b failed");

        let (t0, t1, t2) = ctx.tensor_only_for_test(&c0a, &c1a, &c0b, &c1b).expect("tensor-only failed");
        let (pre_r0, pre_r1) = ctx
            .multiply_relin_no_rescale_for_test(&c0a, &c1a, &c0b, &c1b)
            .expect("pre-rescale multiply failed");

        // Fast path's ba0/ba1, recovered from the combine output:
        // pre_r0 = t0+ba0, pre_r1 = t1+ba1 (mod q, per tower).
        let nn = n as usize;
        let mut fast_ba0 = vec![0u64; t0.len()];
        let mut fast_ba1 = vec![0u64; t0.len()];
        for t in 0..size_q as usize {
            let q = mods[t];
            for k in 0..nn {
                let idx = t * nn + k;
                fast_ba0[idx] = if pre_r0[idx] >= t0[idx] { pre_r0[idx] - t0[idx] } else { pre_r0[idx] + q - t0[idx] };
                fast_ba1[idx] = if pre_r1[idx] >= t1[idx] { pre_r1[idx] - t1[idx] } else { pre_r1[idx] + q - t1[idx] };
            }
        }

        let (ref_ba0, ref_ba1) = ctx.ks.keyswitch_reference(&t2).expect("reference keyswitch failed");

        let mismatches0 = fast_ba0.iter().zip(ref_ba0.iter()).filter(|(a, b)| a != b).count();
        let mismatches1 = fast_ba1.iter().zip(ref_ba1.iter()).filter(|(a, b)| a != b).count();
        eprintln!("[fast vs reference, real t2] ba0 mismatches={mismatches0}/{} ba1 mismatches={mismatches1}/{}", fast_ba0.len(), fast_ba1.len());
        assert_eq!(mismatches0, 0, "fast keyswitch_device_resident disagrees with keyswitch_reference on ba0 for a REAL t2 -- bug is specific to the fast path");
        assert_eq!(mismatches1, 0, "fast keyswitch_device_resident disagrees with keyswitch_reference on ba1 for a REAL t2 -- bug is specific to the fast path");
    }

    // Simplest possible boundary case: key-switch a literal ZERO input
    // (not derived from any ciphertext at all). digit decomposition of
    // zero is trivially zero in every digit, removing any real-vs-
    // synthetic-structure confound entirely. Expected: ba0+ba1*s ~= 0
    // (small noise only). This isolates the eval key's OWN decryption
    // behavior from everything else in the pipeline.
    #[test]
    fn keyswitch_reference_on_zero_input_gives_near_zero() {
        let ctx = new_ctx(1234);
        let n = ctx.ring_dimension().get() as usize;
        let size_q = ctx.input_towers() as usize;
        let mods: Vec<u64> = (0..size_q as u32).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let s_full = ctx.ks.secret_key();

        let zero_a = vec![0u64; size_q * n];
        let (ba0, ba1) = ctx.ks.keyswitch_reference(&zero_a).expect("reference keyswitch on zero failed");

        let mut worst_frac_of_q = 0.0f64;
        for t in 0..size_q {
            let q = mods[t];
            let root = roots[t];
            let mut actual_eval = vec![0u64; n];
            for k in 0..n {
                let idx = t * n + k;
                actual_eval[k] = addmod(ba0[idx], mulmod(ba1[idx], s_full[idx], q), q);
            }
            let actual_coeff = inverse_ntt(&actual_eval, n as u32, q, root);
            let mut worst = 0u64;
            for k in 0..n {
                let centered = actual_coeff[k].min(q - actual_coeff[k]);
                if centered > worst {
                    worst = centered;
                }
            }
            let frac = worst as f64 / q as f64;
            eprintln!("[zero input] tower={t} q={q} worst|ba0+ba1*s - 0|={worst} ({:.6e}% of q)", frac * 100.0);
            if frac > worst_frac_of_q {
                worst_frac_of_q = frac;
            }
        }
        eprintln!("[zero input] worst fraction of q across all towers: {:.6e}%", worst_frac_of_q * 100.0);
        assert!(
            worst_frac_of_q < 0.01,
            "key-switching a literal ZERO input does NOT give ba0+ba1*s ~= 0 -- worst_frac={:.6}% of q. \
             This isolates a defect in the eval key / keyswitch machinery itself, independent of any input structure.",
            worst_frac_of_q * 100.0
        );
    }

    // The zero-input test above is TRIVIALLY uninformative: digit
    // decomposition of 0 is 0 in every digit, so ba0=ba1=0 EXACTLY
    // regardless of whether the eval key itself is correct (0 times
    // anything is 0). Use a CONSTANT, non-zero EVAL-DOMAIN value instead
    // -- same residue at every position in every tower. By CRT
    // uniqueness, a residue smaller than every q_i, repeated identically
    // across all towers, corresponds to exactly that one small integer
    // -- i.e. this IS a CRT-consistent value (same true value under every
    // prime), unlike entry #30/#38's per-tower-independently-random test
    // data, without needing a real ciphertext's encryption noise at all.
    // This directly and meaningfully exercises digit decomposition and
    // the eval key's own P*s^2 term.
    #[test]
    // GOD PROMPT "CKKS PHASE 9": un-ignored, see entry #42 -- NUM_PART=4
    // fixes the P>=Q_part_max violation this test (at the old NUM_PART=2)
    // was reproducing.
    fn keyswitch_reference_on_constant_crt_consistent_input() {
        let ctx = new_ctx(1234);
        let n = ctx.ring_dimension().get() as usize;
        let size_q = ctx.input_towers() as usize;
        let mods: Vec<u64> = (0..size_q as u32).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let s_full = ctx.ks.secret_key();

        const CONST_VAL: u64 = 5;
        let a_eval: Vec<u64> = vec![CONST_VAL; size_q * n]; // same residue, every tower, every position
        let (ba0, ba1) = ctx.ks.keyswitch_reference(&a_eval).expect("reference keyswitch on constant failed");

        let mut worst_frac_of_q = 0.0f64;
        for t in 0..size_q {
            let q = mods[t];
            let root = roots[t];

            // expected = a_eval * s^2 (eval domain, pointwise -- valid
            // ring multiplication), converted to coefficient domain.
            let mut expected_eval = vec![0u64; n];
            for k in 0..n {
                let idx = t * n + k;
                let s = s_full[idx];
                let s2 = mulmod(s, s, q);
                expected_eval[k] = mulmod(a_eval[idx], s2, q);
            }
            let mut actual_eval = vec![0u64; n];
            for k in 0..n {
                let idx = t * n + k;
                actual_eval[k] = addmod(ba0[idx], mulmod(ba1[idx], s_full[idx], q), q);
            }
            let expected_coeff = inverse_ntt(&expected_eval, n as u32, q, root);
            let actual_coeff = inverse_ntt(&actual_eval, n as u32, q, root);

            let mut worst = 0u64;
            for k in 0..n {
                let diff = if actual_coeff[k] >= expected_coeff[k] { actual_coeff[k] - expected_coeff[k] } else { expected_coeff[k] - actual_coeff[k] };
                let centered = diff.min(q - diff);
                if centered > worst {
                    worst = centered;
                }
            }
            let frac = worst as f64 / q as f64;
            eprintln!("[const CRT input] tower={t} q={q} worst|ba0+ba1*s - a*s^2|={worst} ({:.6e}% of q)", frac * 100.0);
            if frac > worst_frac_of_q {
                worst_frac_of_q = frac;
            }
        }
        eprintln!("[const CRT input] worst fraction of q across all towers: {:.6e}%", worst_frac_of_q * 100.0);
        assert!(
            worst_frac_of_q < 0.01,
            "key-switching a CONSTANT, CRT-consistent input does NOT satisfy ba0+ba1*s ~= a*s^2 -- \
             worst_frac={:.6}% of q. This isolates a defect specific to CRT-consistent (real-data-like) inputs.",
            worst_frac_of_q * 100.0
        );
    }

    // GOD PROMPT "CKKS PHASE 7": direct check of evalkeygen_host's OWN
    // raw output (K.av[0], K.bv[0]), bypassing keyswitch entirely.
    // Checks TOWERS OUTSIDE part 0's own Q-range (no P*s2 term added at
    // all there, per the `if(t>=startIdx && t<endIdx)` guard in
    // evalkeygen_host_sold), so the result isolates whether the basic
    // "encryption-of-zero" machinery (pkA*u+ns*e1, pkB*u+ns*e0, and
    // evalkeygen_host_sold's OWN private NTT table cache) is sound.
    //
    // GOD PROMPT "CKKS PHASE 8": re-examined with the CORRECT sign. This
    // test originally computed `bv0-av0*s` ("the sign the source comment
    // claims" -- src/evalkeygen.cpp's own doc comment literally says
    // "bv[p] - av[p]*s == (P*s2 in part's range) + small noise"), taken
    // at face value without independently re-deriving it. A fresh,
    // isolated C++ test (tests/test_evalkeygen_isolated.cpp, new) proved
    // the TRUE relation is `bv+av*s ~= (P*s2 or 0)` (PLUS, not minus) --
    // confirmed via a hand-constructed nonzero secret satisfying
    // pkB+pkA*s_hat=0 exactly, AND via the full real keygen_host+
    // evalkeygen_host chain with the exact same configuration this test
    // uses. Entry #40's "confirmed defect" was this test's OWN sign
    // bug, not a defect in evalkeygen_host -- see research/
    // EXPERIMENT_LEDGER.md entry #41.
    #[test]
    fn eval_key_outside_own_part_range_is_near_zero() {
        let ctx = new_ctx(1234); // NUM_PART=4, alpha=2, SIZE_Q=8 (see entry #42)
        let n = ctx.ring_dimension().get() as usize;
        let alpha = ctx.ks.alpha();
        let eval_key_towers = ctx.ks.eval_key_towers();
        eprintln!("[eval key raw] alpha={alpha} eval_key_towers={eval_key_towers}");
        let (av0, bv0) = ctx.ks.eval_key_part(0);
        assert_eq!(av0.len(), eval_key_towers as usize * n);

        let s_full = ctx.ks.secret_key(); // Q-only, size_ql*n
        let size_q = ctx.input_towers();
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();

        let start_idx = 0u32; // part 0: startIdx = alpha*0 = 0
        let end_idx = alpha.min(size_q);
        eprintln!("[eval key raw] part 0 own Q-range: [{start_idx}, {end_idx})");

        let mut worst_frac_of_q = 0.0f64;
        let mut checked_any = false;
        for t in end_idx..size_q {
            // OUTSIDE part 0's own range, but still a Q-tower (not a P-tower)
            // -- s_full[t] is valid here since s_full covers all size_q towers.
            checked_any = true;
            let q = mods[t as usize];
            let root = roots[t as usize];
            let mut actual_eval = vec![0u64; n];
            for k in 0..n {
                let idx = t as usize * n + k;
                // bv + av*s (CORRECTED sign -- see entry #41; the
                // literal source comment's "bv - av*s" is wrong)
                let av_s = mulmod(av0[idx], s_full[idx], q);
                actual_eval[k] = addmod(bv0[idx], av_s, q);
            }
            let actual_coeff = inverse_ntt(&actual_eval, n as u32, q, root);
            let mut worst = 0u64;
            for k in 0..n {
                let centered = actual_coeff[k].min(q - actual_coeff[k]);
                if centered > worst {
                    worst = centered;
                }
            }
            let frac = worst as f64 / q as f64;
            eprintln!("[eval key raw] tower={t} (outside part 0's range) q={q} worst|bv+av*s - 0|={worst} ({:.6e}% of q)", frac * 100.0);
            if frac > worst_frac_of_q {
                worst_frac_of_q = frac;
            }
        }
        assert!(checked_any, "test fixture has no towers outside part 0's range -- adjust NUM_PART/alpha");
        eprintln!("[eval key raw] worst fraction of q (outside part 0's range): {:.6e}%", worst_frac_of_q * 100.0);
        assert!(
            worst_frac_of_q < 0.01,
            "OUTSIDE part 0's own Q-range (no P*s2 term at all), bv[0]+av[0]*s is NOT ~0 -- worst_frac={:.6}% of q. \
             This isolates a defect in evalkeygen_host_sold's basic encryption-of-zero machinery itself \
             (pkA*u+ns*e1 / pkB*u+ns*e0, or its own private NTT table cache).",
            worst_frac_of_q * 100.0
        );
    }

    // Control A2, CORRECTED: an earlier version of this check compared
    // the raw tensor product against the NOISE-FREE expected product
    // (m_a_eval*m_b_eval, using the encoded plaintexts alone) with a
    // tight (1% of q) tolerance -- WRONG, and the resulting ~50%-of-q
    // "failure" was a methodology artifact, not a defect: the RAW,
    // UN-RESCALED tensor product's cross-noise terms (m_a (*) e_b,
    // e_a (*) m_b -- RING CONVOLUTIONS, not simple sums) are genuinely
    // LARGE by design. Convolving an n=1024-coefficient, Delta~2^50-scale
    // polynomial with even a tiny (sigma=3.19) noise polynomial sums
    // n terms each up to ~Delta*noise, giving a result up to roughly
    // n*Delta*sigma ~ 2^60 -- comparable to or exceeding several of this
    // fixture's own moduli. This is exactly WHY CKKS rescales after each
    // multiplication (to divide this amplified noise back down), not a
    // bug. The CORRECT check is a PURE ALGEBRAIC IDENTITY with ZERO
    // tolerance needed: t0+t1*s+t2*s^2 must EXACTLY equal
    // (c0a+c1a*s)*(c0b+c1b*s), both computed directly from the
    // ciphertext components themselves (no plaintext/noise-free value
    // involved at all) -- this is checkable in EITHER domain since it
    // is an EXACT equality claim (NTT is a bijection: two values are
    // equal iff their NTT transforms are equal), not an approximate
    // "is this small" claim that requires coefficient domain.
    fn control_a2_raw_tensor_matches_direct_ciphertext_product(ctx: &EvalMultContext, slots_a: &[f64], slots_b: &[f64], seed_a: u64, seed_b: u64) -> u64 {
        let n = ctx.ring_dimension().get();
        let size_q = ctx.input_towers();
        let mods: Vec<u64> = (0..size_q).map(|t| ctx.modulus_q(t)).collect();
        let roots: Vec<u64> = mods
            .iter()
            .map(|&q| crate::ntt::native_root(ctx.ring_dimension(), ckks_core::Modulus::new(q).unwrap()))
            .collect();
        let delta = ctx.modulus_q(size_q - 1) as f64;

        let pk_a = ctx.ks.public_key_a();
        let pk_b = ctx.ks.public_key_b();
        let s_full = ctx.ks.secret_key();

        let m_a = crate::codec::encode_real(slots_a, n, delta).expect("encode a failed");
        let m_b = crate::codec::encode_real(slots_b, n, delta).expect("encode b failed");
        let (c0a, c1a) =
            crate::codec::encrypt(&m_a, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed_a).expect("encrypt a failed");
        let (c0b, c1b) =
            crate::codec::encrypt(&m_b, n, &pk_a, &pk_b, &mods, &roots, 1, 3.19, seed_b).expect("encrypt b failed");

        let (t0, t1, t2) = ctx.tensor_only_for_test(&c0a, &c1a, &c0b, &c1b).expect("tensor-only failed");

        let nn = n as usize;
        let mut mismatches = 0u64;
        for t in 0..size_q as usize {
            let q = mods[t];
            let mut bad_this_tower = 0u64;
            for k in 0..nn {
                let idx = t * nn + k;
                let s = s_full[idx];
                let s2 = mulmod(s, s, q);

                let phase_a = addmod(c0a[idx], mulmod(c1a[idx], s, q), q);
                let phase_b = addmod(c0b[idx], mulmod(c1b[idx], s, q), q);
                let expected = mulmod(phase_a, phase_b, q);

                let actual = addmod(addmod(t0[idx], mulmod(t1[idx], s, q), q), mulmod(t2[idx], s2, q), q);

                if actual != expected {
                    bad_this_tower += 1;
                }
            }
            eprintln!("[control A2] tower={t} q={q} mismatches={bad_this_tower}/{nn} (exact algebraic identity, no tolerance)");
            mismatches += bad_this_tower;
        }
        mismatches
    }

    #[test]
    fn control_a2_raw_tensor_matches_direct_ciphertext_product_exactly() {
        let ctx = new_ctx(1234);
        let slots_a = [1.0, 2.0, 3.0, 4.0];
        let slots_b = [5.0, 6.0, 7.0, 8.0];
        let mismatches = control_a2_raw_tensor_matches_direct_ciphertext_product(&ctx, &slots_a, &slots_b, 9001, 9002);
        eprintln!("[control A2] total mismatches across all towers: {mismatches}");
        assert_eq!(
            mismatches, 0,
            "t0+t1*s+t2*s^2 does NOT exactly equal (c0a+c1a*s)*(c0b+c1b*s) -- this is a pure algebraic identity \
             with zero tolerance, so any mismatch means LaunchTensor itself computes the wrong value for real \
             encrypted ciphertext data."
        );
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
    // GOD PROMPT "CKKS PHASE 9": un-ignored. The full round trip was
    // blocked by NUM_PART=2 violating the Hybrid key-switch's
    // P>=Q_part_max requirement (entry #42), not by any defect in
    // evalkeygen_host (disproven, entry #41) or in the key-switch
    // reconstruction code itself (independently proven correct in
    // isolation this round). NUM_PART=4 fixes it.
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
    // GOD PROMPT "CKKS PHASE 9": un-ignored, see entry #42.
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
    // GOD PROMPT "CKKS PHASE 9": un-ignored, see entry #42.
    fn real_encrypted_evalmult_negative_and_fractional_values() {
        let ctx = new_ctx(4242);
        let slots_a = [-1.5, 2.25, -3.0, 0.5];
        let slots_b = [2.0, -1.0, 0.5, 4.0];
        let expected: Vec<f64> = slots_a.iter().zip(slots_b.iter()).map(|(a, b)| a * b).collect();

        let (re, _im) = real_evalmult_round_trip(&ctx, &slots_a, &slots_b, 333, 444);
        assert_slots_close("negative/fractional product", &re[..expected.len()], &expected);
    }

    #[test]
    // GOD PROMPT "CKKS PHASE 9": un-ignored, see entry #42.
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
