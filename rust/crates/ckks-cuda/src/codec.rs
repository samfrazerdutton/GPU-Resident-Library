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
