//! Host-side CKKS parameter and error types shared by the Rust evolution
//! crates (`ckks-cuda`, and later `ckks-api`/`ckks-ffi` if those become
//! justified). This crate owns validated parameter *metadata* only --
//! no NTT, no rescale, no key switching. Those remain either in the
//! existing, proven C++/CUDA engine (reused via FFI) or in `ckks-cuda`'s
//! own thin orchestration layer, per the governing directive's own
//! instruction not to duplicate complicated cryptographic algorithms
//! just to make the code look Rust-native.

use std::fmt;

/// Errors a caller can hit before any GPU work starts, plus a catch-all
/// for forwarded CUDA/driver failures (kept as a `String` here so this
/// crate does not need to depend on the CUDA binding crate at all).
#[derive(Debug)]
pub enum CkksError {
    /// `n` was not a power of two, or was outside the range this project's
    /// existing NTT kernels (`kernels/ntt_ct.cu`/`intt_ct.cu`) are proven
    /// correct for. See `research/EXPERIMENT_LEDGER.md` entry #29 for the
    /// exact proven range (n=1024, this round's vertical slice) -- this
    /// type does not claim correctness outside what has actually been
    /// tested.
    InvalidRingDimension { n: u64 },
    /// `q` was zero or even. This is a NECESSARY but NOT SUFFICIENT check
    /// for "q is a valid NTT-friendly prime modulus" -- it does not run a
    /// primality test or verify q ≡ 1 (mod 2n), the actual requirement for
    /// an NTT-friendly prime. Callers must still supply a modulus they
    /// independently know is correct (e.g. from the existing `native_primes`
    /// C++ helper); this type only catches the cheapest, unambiguous
    /// mistakes before they reach the GPU.
    InvalidModulus { q: u64 },
    /// A CUDA driver/runtime call failed. The message is whatever the
    /// underlying binding reported; this variant exists so `ckks-core`
    /// can define `CkksError` without depending on `cudarc`.
    Cuda(String),
}

impl fmt::Display for CkksError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CkksError::InvalidRingDimension { n } => {
                write!(f, "invalid ring dimension n={n}: must be a power of two")
            }
            CkksError::InvalidModulus { q } => {
                write!(f, "invalid modulus q={q}: must be a nonzero odd integer (necessary, not sufficient, condition)")
            }
            CkksError::Cuda(msg) => write!(f, "CUDA error: {msg}"),
        }
    }
}

impl std::error::Error for CkksError {}

/// A validated ring dimension: a power of two. Does not claim that `n` is
/// a size the GPU kernels have been *proven* correct for -- only that it
/// is structurally valid input to an NTT. Check the experiment ledger for
/// which `n` actually have passing correctness evidence before trusting
/// output at a new size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingDimension(u32);

impl RingDimension {
    pub fn new(n: u32) -> Result<Self, CkksError> {
        if n == 0 || !n.is_power_of_two() {
            return Err(CkksError::InvalidRingDimension { n: n as u64 });
        }
        Ok(RingDimension(n))
    }

    pub fn get(self) -> u32 {
        self.0
    }

    pub fn log2(self) -> u32 {
        self.0.trailing_zeros()
    }
}

/// A modulus that has passed the cheap, necessary structural checks
/// (nonzero, odd). See [`CkksError::InvalidModulus`] for exactly what is
/// and is not verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Modulus(u64);

impl Modulus {
    pub fn new(q: u64) -> Result<Self, CkksError> {
        if q == 0 || q % 2 == 0 {
            return Err(CkksError::InvalidModulus { q });
        }
        Ok(Modulus(q))
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_dimension_accepts_powers_of_two() {
        assert_eq!(RingDimension::new(1024).unwrap().get(), 1024);
        assert_eq!(RingDimension::new(1024).unwrap().log2(), 10);
        assert_eq!(RingDimension::new(1).unwrap().get(), 1);
    }

    #[test]
    fn ring_dimension_rejects_non_powers_of_two() {
        assert!(RingDimension::new(0).is_err());
        assert!(RingDimension::new(1000).is_err());
        assert!(RingDimension::new(1023).is_err());
    }

    #[test]
    fn modulus_accepts_nonzero_odd() {
        assert_eq!(Modulus::new(1152921504606830593).unwrap().get(), 1152921504606830593);
    }

    #[test]
    fn modulus_rejects_zero_and_even() {
        assert!(Modulus::new(0).is_err());
        assert!(Modulus::new(2).is_err());
        assert!(Modulus::new(36028797018646530).is_err());
    }
}
