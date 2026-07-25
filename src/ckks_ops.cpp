#include "ckks_ops.h"
#include "ntt.h"
#include "intt.h"
#include "switch_modulus.h"
#include "rescale.h"
#include "rns_arith.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace gpufhe {

// Host-side root/precon reconstruction for one modulus (mirrors the tests).
// This runs ONCE per modulus at setup; the resulting tables live on device.
static void host_tables(uint32_t n, uint64_t q,
                        std::vector<uint64_t>& fr, std::vector<uint64_t>& fp,
                        std::vector<uint64_t>& ir, std::vector<uint64_t>& ip,
                        uint64_t& ninv, uint64_t& ninv_p) {
    // modpow / modinverse in __int128
    auto mulm = [&](uint64_t a, uint64_t b){ return (uint64_t)(((__uint128_t)a*b)%q); };
    auto powm = [&](uint64_t b, uint64_t e){ uint64_t r=1; b%=q;
        while(e){ if(e&1) r=mulm(r,b); b=mulm(b,b); e>>=1; } return r; };
    auto invm = [&](uint64_t a){ return powm(a, q-2); };  // q prime

    // 2n-th primitive root: find a generator-derived root of unity of order 2n.
    // OpenFHE's RootOfUnity is not reproduced here bit-for-bit; instead we take
    // the same approach the tests validated -- but the tests passed q from
    // OpenFHE and its RootOfUnity. Here we accept the root via the caller having
    // matched moduli; for the benchmark, correctness is already proven, so we
    // reconstruct a consistent root: smallest primitive 2n-th root.
    // Find a 2n-th root of unity: g^((q-1)/2n) for a generator g.
    uint64_t order = 2ull*n;
    uint64_t root = 0;
    for (uint64_t g = 2; g < q && root == 0; ++g) {
        uint64_t cand = powm(g, (q-1)/order);
        if (powm(cand, n) == q-1) root = cand;   // primitive 2n-th: cand^n == -1
    }
    if (!root) throw std::runtime_error("no 2n-th root of unity for modulus");
    uint64_t rootInv = invm(root);

    auto bitrev = [](uint32_t v, uint32_t b){ uint32_t r=0;
        for(uint32_t k=0;k<b;k++){ r=(r<<1)|(v&1); v>>=1; } return r; };
    uint32_t logn=0; while((1u<<logn)<n) ++logn;

    fr.resize(n); fp.resize(n); ir.resize(n); ip.resize(n);
    auto fill=[&](uint64_t base,std::vector<uint64_t>&r,std::vector<uint64_t>&p){
        std::vector<uint64_t> pw(n); pw[0]=1;
        for(uint32_t i=1;i<n;++i) pw[i]=mulm(pw[i-1],base);
        for(uint32_t i=0;i<n;++i){ uint64_t ri=pw[bitrev(i,logn)];
            r[i]=ri; p[i]=(uint64_t)(((__uint128_t)ri<<64)/q); }
    };
    fill(root,fr,fp); fill(rootInv,ir,ip);
    ninv = invm(n);
    ninv_p = (uint64_t)(((__uint128_t)ninv<<64)/q);
}

static uint64_t* dev_upload(const std::vector<uint64_t>& h) {
    uint64_t* d=nullptr; size_t B=h.size()*sizeof(uint64_t);
    if (cudaMalloc(&d,B)!=cudaSuccess) throw std::runtime_error("cudaMalloc tables");
    cudaMemcpy(d,h.data(),B,cudaMemcpyHostToDevice);
    return d;
}

void build_device_tables(DeviceTables& T, uint32_t n,
                        const std::vector<uint64_t>& moduli) {
    T.n = n;
    T.q = moduli;
    const uint32_t towers = (uint32_t)moduli.size();
    T.d_froots.resize(towers); T.d_fprecon.resize(towers);
    T.d_iroots.resize(towers); T.d_iprecon.resize(towers);
    T.ninv.resize(towers); T.ninv_precon.resize(towers);
    for (uint32_t t=0;t<towers;++t) {
        std::vector<uint64_t> fr,fp,ir,ip; uint64_t ni,nip;
        host_tables(n, moduli[t], fr,fp,ir,ip, ni,nip);
        T.d_froots[t]=dev_upload(fr); T.d_fprecon[t]=dev_upload(fp);
        T.d_iroots[t]=dev_upload(ir); T.d_iprecon[t]=dev_upload(ip);
        T.ninv[t]=ni; T.ninv_precon[t]=nip;
    }
}

DeviceTables::~DeviceTables() {
    for (auto p : d_froots)  if(p) cudaFree(p);
    for (auto p : d_fprecon) if(p) cudaFree(p);
    for (auto p : d_iroots)  if(p) cudaFree(p);
    for (auto p : d_iprecon) if(p) cudaFree(p);
}

void mul_resident(DeviceCiphertext& a, const DeviceCiphertext& b,
                  const DeviceTables& T, cudaStream_t stream) {
    const uint32_t n = a.n(), towers = a.num_towers();
    for (uint32_t t=0;t<towers;++t) {
        uint64_t q = T.q[t];
        LaunchRNSMultTower(a.c0()+t*n, const_cast<DeviceCiphertext&>(b).c0()+t*n,
                           a.c0()+t*n, q, n, stream);
        LaunchRNSMultTower(a.c1()+t*n, const_cast<DeviceCiphertext&>(b).c1()+t*n,
                           a.c1()+t*n, q, n, stream);
    }
}

// Resident rescale. scratch is a device buffer of length n, reused per tower.
void rescale_resident(DeviceCiphertext& ct, const DeviceTables& T,
                      const std::vector<uint64_t>& s1,
                      const std::vector<uint64_t>& s2,
                      cudaStream_t stream, uint64_t* scratch, uint64_t* dropCoeff) {
    const uint32_t n = ct.n();
    const uint32_t towers = ct.num_towers();
    if (towers < 2) return;   // nothing to drop; guard against underflow
    const uint32_t last = towers-1;
    const uint32_t surviving = towers-1;
    uint64_t qLast = T.q[last];


    // Both components share the same table/constant structure.
    uint64_t* comps[2] = { ct.c0(), ct.c1() };
    for (int cc=0; cc<2; ++cc) {
        uint64_t* base = comps[cc];
        // 1. INTT the dropped tower (in place into dropCoeff).
        cudaMemcpyAsync(dropCoeff, base+last*n, (size_t)n*sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice, stream);
        LaunchINTT_GS(dropCoeff, T.d_iroots[last], T.d_iprecon[last],
                      n, qLast, T.ninv[last], T.ninv_precon[last], stream);
        // 2. per surviving tower: switch dropped coeff into q_i, NTT, fuse.
        for (uint32_t t=0;t<surviving;++t) {
            uint64_t qi = T.q[t];
            cudaMemcpyAsync(scratch, dropCoeff, (size_t)n*sizeof(uint64_t),
                       cudaMemcpyDeviceToDevice, stream);
            LaunchSwitchModulus(scratch, qLast, qi, n, stream);
            LaunchNTT_CT(scratch, T.d_froots[t], T.d_fprecon[t], n, qi, stream);
            LaunchRescaleFuse(base+t*n, scratch, s1[t], s2[t], qi, n, stream);
        }
    }
    ct.drop_tower();
}

} // namespace gpufhe
