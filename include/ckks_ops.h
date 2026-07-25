#pragma once
#include "device_ciphertext.h"
#include <cstdint>
#include <vector>

namespace gpufhe {

// Per-modulus root tables, uploaded to device ONCE and reused across the chain.
// Holding these resident is what keeps rescale from smuggling host transfers
// mid-chain -- the tables are constant per modulus for the context's lifetime.
struct DeviceTables {
    // indexed [tower]: device pointers to fwd/inv roots + precons for that modulus
    std::vector<uint64_t*> d_froots, d_fprecon, d_iroots, d_iprecon;
    std::vector<uint64_t>  q;          // modulus per tower
    std::vector<uint64_t>  ninv, ninv_precon;
    uint32_t n = 0;
    ~DeviceTables();
};

// Build device-resident root tables for every modulus in the chain.
// moduli[i] = q for tower i (full tower set, before any drops).
void build_device_tables(DeviceTables& T, uint32_t n,
                         const std::vector<uint64_t>& moduli);

// Resident coefficient-wise multiply: ct_a *= ct_b, per tower, in place on
// device. Both must be EVALUATION form, same tower count. No host transfer.
void mul_resident(DeviceCiphertext& a, const DeviceCiphertext& b,
                  const DeviceTables& T);

// Resident rescale: drops the last tower and applies the correction entirely on
// device, using pre-uploaded tables. s1[t]=qlInvModq, s2[t]=QlQlInvModqlDivqlModq
// for the current level. Operates on both components. Scratch is device-side.
void rescale_resident(DeviceCiphertext& ct, const DeviceTables& T,
                      const std::vector<uint64_t>& s1,
                      const std::vector<uint64_t>& s2);

} // namespace gpufhe
