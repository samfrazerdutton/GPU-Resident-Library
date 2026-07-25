# GPU-Resident FHE Library

A from-scratch GPU-resident FHE library. Ciphertexts live in VRAM and are
owned by the library end-to-end; data crosses PCIe once on upload and once on
download, never per operation.

## Why

Accelerating OpenFHE through a hardware abstraction layer cannot beat CPU on
this class of hardware: a paired-difference sweep (RTX 2060 Max-Q, WSL2) found
no CPU/GPU crossover at any parameter, because per-operation PCIe transfer
scales with data volume. Worse, operation-level residency is structurally
impossible in OpenFHE — PolyImpl::SetValues reallocates the coefficient buffer
on every operation, so no device cache keyed on a buffer pointer survives
across ops.

This library removes the boundary entirely by owning the representation. The
open question it exists to answer: does a fully VRAM-resident multiply chain
beat CPU once the PCIe boundary is gone?

## Design

- `DeviceCiphertext` owns RNS tower buffers in VRAM. No cache, no host-pointer
  keying; the object is the owner. Representation is general (scheme tag + RNS
  towers + format) to hold CKKS/BFV/BGV; CKKS operations are implemented first.
- OpenFHE is a test-only oracle: encrypt/decrypt there, compare bit-exact.
  Zero runtime dependency.
- The NTT is bit-exact against OpenFHE's Cooley-Tukey Shoup transform,
  validated across real CKKS tower moduli.

## Status

Foundation: device-resident ciphertext + in-place NTT, bit-exact vs OpenFHE.
Next: on-device multiply-rescale chain + head-to-head benchmark vs CPU.
