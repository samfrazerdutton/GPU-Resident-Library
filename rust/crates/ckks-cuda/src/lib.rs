//! Rust host orchestration over this project's EXISTING, proven CUDA C++
//! kernels, reused unmodified via FFI (see `build.rs`). This is Path B of
//! the GOD PROMPT "CKKS RUST + CUDA EVOLUTION" architecture: Rust host +
//! the SAME CUDA C++ kernels, isolating host-orchestration cost/benefit
//! from the GPU arithmetic itself.
//!
//! ## Module map
//!
//! - [`util`]: shared host-side bookkeeping (modular arithmetic, NTT
//!   root-table construction) and the `DeviceBuf` RAII device-memory
//!   guard, used by every operation below.
//! - [`ntt`]: the first ported operation -- coefficient-domain NTT/INTT
//!   round trip. See [`ntt::NttContext`].
//! - [`rescale`]: the second ported operation -- single-modulus RNS
//!   rescale, reusing `LaunchSwitchModulus`/`LaunchRescaleFuse` and the
//!   existing `native_rescale_consts`. See [`rescale::RescaleContext`].
//! - [`keyswitch`]: the third ported operation -- full Hybrid key
//!   switching. Unlike `ntt`/`rescale`, its setup-time state (the
//!   evaluation key) is genuine cryptographic key material, not
//!   bookkeeping, so it is built end to end by ONE opaque C++ bridge call
//!   rather than reconstructed field-by-field in Rust. See
//!   [`keyswitch::KeySwitchContext`] and its module doc comment for why,
//!   and for the honest limits of this module's correctness evidence.
//! - [`evalmult`]: the fourth ported operation -- ciphertext-ciphertext
//!   EvalMult (tensor product + relinearization + rescale). Composes
//!   [`keyswitch::KeySwitchContext`] (as the relinearization backend) and
//!   [`rescale::RescaleContext`] (for the two post-relin rescale steps)
//!   rather than re-deriving either algorithm -- see
//!   [`evalmult::EvalMultContext`]'s module doc comment for exactly how,
//!   and why their existing host-slice-oriented public APIs needed a
//!   small, explicit device-pointer adaptation first.
//!
//! `ntt` and `rescale` follow an IDENTICAL pattern: a context struct built
//! once via `::new` that owns every reusable, input-independent resource
//! (device selection, host-computed tables/constants, device allocations,
//! table uploads), and a hot-path method that does ONLY per-call,
//! input-dependent work. `keyswitch` follows the SAME setup/hot-path split
//! at the API level (`KeySwitchContext::new` vs. `::keyswitch`), but its
//! setup step is a single opaque FFI call into the existing native
//! key-generation pipeline instead of Rust-side reconstruction -- see
//! `ntt`'s module doc comment for why the split exists at all, and
//! `keyswitch`'s for why ITS setup step could not be narrowed the same way
//! `ntt`'s and `rescale`'s were. `evalmult` follows the same split again,
//! one level up: its setup step builds one `KeySwitchContext` and one
//! `RescaleContext`, its hot path chains their (new, device-resident)
//! internals with two new kernels (`LaunchTensor`/`LaunchCombine`) on one
//! stream.
//!
//! No cryptographic algorithm is reimplemented in Rust here. Where a
//! computation is genuine, non-trivial number theory (primitive-root
//! search, the rescale fuse constants' exact-division trick, the entire
//! key-switch key-generation pipeline), this crate calls the EXISTING,
//! unmodified C++ implementation via a narrow FFI bridge (`cpp_bridge/`)
//! instead of re-deriving it. Where a computation is plain bookkeeping
//! that the main C++ engine ALSO performs on the host side to drive these
//! same kernels (NTT root-table construction), it is reimplemented
//! independently in Rust so the two implementations cannot share a
//! transcription bug and can be cross-checked.

pub mod evalmult;
pub mod keyswitch;
pub mod ntt;
pub mod rescale;
mod util;

pub use evalmult::EvalMultContext;
pub use keyswitch::KeySwitchContext;
pub use ntt::{cold_start_cost_probe, native_root, ntt_roundtrip, NttContext};
pub use rescale::{cold_start_cost_probe as rescale_cold_start_cost_probe, native_rescale_consts, RescaleContext};
