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
//!
//! Both operation modules follow the SAME pattern established by `ntt`
//! and extended (not redesigned) by `rescale`: a context struct built
//! once via `::new` that owns every reusable, input-independent resource
//! (device selection, host-computed tables/constants, device allocations,
//! table uploads), and a hot-path method that does ONLY per-call,
//! input-dependent work. See `ntt`'s module doc comment for why this
//! split exists and what went wrong before it did.
//!
//! No cryptographic algorithm is reimplemented in Rust here. Where a
//! computation is genuine, non-trivial number theory (primitive-root
//! search, the rescale fuse constants' exact-division trick), this crate
//! calls the EXISTING, unmodified C++ implementation via a narrow FFI
//! bridge (`cpp_bridge/`) instead of re-deriving it. Where a computation
//! is plain bookkeeping that the main C++ engine ALSO performs on the
//! host side to drive these same kernels (NTT root-table construction),
//! it is reimplemented independently in Rust so the two implementations
//! cannot share a transcription bug and can be cross-checked.

pub mod ntt;
pub mod rescale;
mod util;

pub use ntt::{cold_start_cost_probe, native_root, ntt_roundtrip, NttContext};
pub use rescale::{cold_start_cost_probe as rescale_cold_start_cost_probe, native_rescale_consts, RescaleContext};
