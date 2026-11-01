//! Shared host-side bookkeeping and device-buffer RAII used by every
//! operation in this crate (`ntt`, `rescale`, and any future one). Moved
//! here out of a single `lib.rs` when the second operation (rescale) was
//! added, so the two operation modules don't duplicate `DeviceBuf`/
//! `build_tables`/modular-arithmetic helpers -- a justified, minimal
//! refactor, not speculative abstraction (see the crate root doc comment
//! for why duplicating these per-operation would be the wrong call).
use ckks_core::CkksError;
use cudarc::runtime::result as rt;
use std::ffi::c_void;

extern "C" {
    // Standard CUDA runtime symbol, called directly rather than through
    // cudarc::runtime::result::stream::synchronize's null-stream typing,
    // to keep "wait for the kernels before reading results back" visible
    // and unambiguous at every call site that needs it.
    pub fn cudaDeviceSynchronize() -> i32;
}

pub fn cuda_err(e: impl std::fmt::Debug) -> CkksError {
    CkksError::Cuda(format!("{e:?}"))
}

pub fn powmod(b: u64, e: u64, q: u64) -> u64 {
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

pub fn invmod(a: u64, q: u64) -> u64 {
    powmod(a % q, q - 2, q)
}

pub fn bitrev(mut v: u32, bits: u32) -> u32 {
    let mut r = 0u32;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

pub struct Tables {
    pub roots: Vec<u64>,
    pub precon: Vec<u64>,
}

/// Builds the (root power, Shoup-precon) table for one NTT direction.
/// Matches `tests/test_ntt_roundtrip_n1024.cpp`'s `build_tables` and
/// `kernels/ntt_ct.cu`/`intt_ct.cu`'s own documented convention exactly:
/// standard order in, bit-reversed out for forward; the mirror for
/// inverse (via `invmod(root)` as the base). O(n) host-side work --
/// input-independent, so every context in this crate builds these once
/// at setup, never per call.
pub fn build_tables(n: u32, q: u64, root: u64, inverse: bool) -> Tables {
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
/// it is still in flight) is each OPERATION module's responsibility,
/// enforced by always calling `cudaDeviceSynchronize()` before any
/// host-visible use of a result.
pub struct DeviceBuf {
    ptr: *mut c_void,
}

impl DeviceBuf {
    pub fn alloc(bytes: usize) -> Result<Self, CkksError> {
        let ptr = unsafe { rt::malloc_sync(bytes).map_err(cuda_err)? };
        Ok(DeviceBuf { ptr })
    }

    pub fn upload_u64(&self, host: &[u64]) -> Result<(), CkksError> {
        unsafe { rt::memcpy_htod_sync(self.ptr, host).map_err(cuda_err) }
    }

    pub fn download_u64(&self, host: &mut [u64]) -> Result<(), CkksError> {
        unsafe { rt::memcpy_dtoh_sync(host, self.ptr).map_err(cuda_err) }
    }

    /// Device-to-device copy of `len_u64` elements from a raw device
    /// pointer (which may be an offset into ANOTHER buffer, e.g. one
    /// tower of a multi-tower allocation via
    /// [`DeviceBuf::tower_const_u64`]) into `self`. Mirrors the main C++
    /// engine's own `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, s)`
    /// pattern in `src/keyswitch_resident.cpp`'s `rescale_resident_raw` --
    /// used by the `rescale` module to copy the dropped tower into
    /// scratch buffers without a host round-trip.
    ///
    /// # Safety
    /// `src` must be a valid device pointer with at least `len_u64 * 8`
    /// readable bytes, and `self` must have at least that many bytes
    /// allocated. Callers within this crate only ever pass pointers
    /// derived from a live `DeviceBuf` of the right size.
    pub unsafe fn copy_from_ptr(&self, src: *const c_void, len_u64: usize) -> Result<(), CkksError> {
        rt::memcpy_dtod_sync(self.ptr, src, len_u64 * 8).map_err(cuda_err)
    }

    pub fn as_mut_u64(&self) -> *mut u64 {
        self.ptr as *mut u64
    }

    pub fn as_const_u64(&self) -> *const u64 {
        self.ptr as *const u64
    }

    /// Pointer to this buffer offset by `tower_idx * n` u64 elements --
    /// for indexing into one of several towers packed contiguously in a
    /// single multi-tower allocation (`rescale::RescaleContext`'s `d_c`).
    pub fn tower_mut_u64(&self, tower_idx: usize, n: usize) -> *mut u64 {
        unsafe { self.as_mut_u64().add(tower_idx * n) }
    }

    pub fn tower_const_u64(&self, tower_idx: usize, n: usize) -> *const u64 {
        unsafe { self.as_const_u64().add(tower_idx * n) }
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
