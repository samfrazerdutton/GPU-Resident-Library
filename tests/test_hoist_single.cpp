// Standalone correctness test for the CORRECT-ORDER rotation hoisting
// design (see research/NEXT_SESSION.md for the full derivation): decompose
// the UNPERMUTED c1 (ks_accumulate_qp), fold c0*P into the QP-basis
// accumulator (LaunchScaledAddInto), apply the automorphism to the QP-basis
// accumulator (automorphism_eval_fast_device, already proven), moddown
// (ks_moddown). Rotation key generated with the INVERSE automorphism
// convention (psi_{k^-1}(s), not psi_k(s)).
//
// This tests ONLY correctness of the reordering + scaling + key convention,
// for a SINGLE rotation -- NOT yet the multi-rotation decompose-sharing
// benefit (that is a separate, subsequent step once this is proven).
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include "stage_acc.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void automorphism_eval_fast_device(uint64_t*, uint32_t, uint32_t, uint32_t, uint64_t*, cudaStream_t);
void rotate_ct_resident(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                        const KeySwitchConstants&, uint32_t, uint32_t,
                        const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&,
                  const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                  uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using cd=std::complex<double>;

// extended Euclid: modular inverse of a mod m (a, m coprime)
static uint64_t modinv(uint64_t a, uint64_t m){
    int64_t old_r=(int64_t)a, r=(int64_t)m, old_s=1, s=0;
    while(r!=0){ int64_t q=old_r/r;
        int64_t tr=old_r-q*r; old_r=r; r=tr;
        int64_t ts=old_s-q*s; old_s=s; s=ts; }
    int64_t res=old_s % (int64_t)m; if(res<0) res+=(int64_t)m;
    return (uint64_t)res;
}

static bool run_case(uint32_t n, uint32_t tw, uint32_t k,
                     const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                     const std::vector<uint64_t>& modP, const std::vector<uint64_t>& mq,
                     const std::vector<uint64_t>& rq, gpufhe::KeyPairHost& KPqp, gpufhe::KeyPairHost& KPq,
                     const std::vector<uint64_t>& PM, uint32_t sizeP, uint32_t numPart,
                     const std::vector<uint64_t>& c0, const std::vector<uint64_t>& c1)
{
    const uint32_t M=2*n, sizeQlP=tw+sizeP;
    uint64_t kinv=modinv(k,M);

    // Ground truth key: textbook convention (psi_k(s)).
    std::vector<uint64_t> sA_truth=KPqp.s;
    gpufhe::automorphism_eval_host(sA_truth,sizeQlP,n,k,mq,rq);
    gpufhe::KeySwitchConstants Ktruth; Ktruth.n=n;
    gpufhe::compute_keyswitch_constants(Ktruth,mod,modP,numPart);
    for(uint32_t i=0;i<tw;++i){Ktruth.rootModList.push_back(mod[i]);Ktruth.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){Ktruth.rootModList.push_back(modP[j]);Ktruth.rootValList.push_back(rq[tw+j]);}
    gpufhe::evalkeygen_host_sold(Ktruth,KPqp.s,sA_truth,KPqp.pkA,KPqp.pkB,PM,mq,rq,1,3.19,900+(uint64_t)k);

    // Hoisted key: this codebase's evalkeygen_host_sold ALWAYS encrypts its
    // payload (s2) UNDER the secret implied by (pkA_QP,pkB_QP) -- it never
    // reads sQP for the arithmetic. To get a key encrypting s UNDER
    // sigma_{k^-1}(s) (the convention the hoisted design actually needs,
    // confirmed by re-reading evalkeygen_host_sold's body: pkA/pkB alone
    // determine the encrypting secret), permute the PUBLIC KEY by k^-1
    // (automorphism is equivariant on RLWE public keys) and keep the
    // payload as the UNPERMUTED base secret s. (First attempt permuted s2
    // instead and kept pkA/pkB as-is -- that encrypts P*sigma_kinv(s) UNDER
    // s, the OPPOSITE direction, and failed 100%.)
    std::vector<uint64_t> pkA_hoist=KPqp.pkA, pkB_hoist=KPqp.pkB;
    gpufhe::automorphism_eval_host(pkA_hoist,sizeQlP,n,(uint32_t)kinv,mq,rq);
    gpufhe::automorphism_eval_host(pkB_hoist,sizeQlP,n,(uint32_t)kinv,mq,rq);
    gpufhe::KeySwitchConstants Khoist; Khoist.n=n;
    gpufhe::compute_keyswitch_constants(Khoist,mod,modP,numPart);
    for(uint32_t i=0;i<tw;++i){Khoist.rootModList.push_back(mod[i]);Khoist.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){Khoist.rootModList.push_back(modP[j]);Khoist.rootValList.push_back(rq[tw+j]);}
    gpufhe::evalkeygen_host_sold(Khoist,KPqp.s,KPqp.s,pkA_hoist,pkB_hoist,PM,mq,rq,1,3.19,900+(uint64_t)k);

    // ---- Ground truth: existing, proven rotate_ct_resident ----
    std::vector<uint64_t> gc0=c0, gc1=c1;
    gpufhe::rotate_ct_resident(gc0,gc1,k,Ktruth,tw,n,mod,root);

    // ---- Hoisted path ----
    auto C=gpufhe::ks_context_create(Khoist);
    auto W=gpufhe::ks_work_create(C);
    size_t T=(size_t)tw*n, B=T*8;
    size_t TQP=(size_t)sizeQlP*n, BQP=TQP*8;
    uint64_t *d_c0,*d_c1,*d_scratch_qp;
    cudaMalloc(&d_c0,B); cudaMalloc(&d_c1,B); cudaMalloc(&d_scratch_qp,BQP);
    cudaMemcpy(d_c0,c0.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1,c1.data(),B,cudaMemcpyHostToDevice);

    gpufhe::ks_accumulate_qp(d_c1,C,W,0);          // decompose the UNPERMUTED c1
    for(uint32_t t=0;t<tw;++t)                      // fold c0*P into Q-tower range of res0
        LaunchScaledAddInto(W.d_res0+(size_t)t*n, d_c0+(size_t)t*n, PM[t], mod[t], n, 0);
    gpufhe::automorphism_eval_fast_device(W.d_res0,sizeQlP,n,k,d_scratch_qp,0);
    gpufhe::automorphism_eval_fast_device(W.d_res1,sizeQlP,n,k,d_scratch_qp,0);
    uint64_t *d_ba0,*d_ba1; cudaMalloc(&d_ba0,B); cudaMalloc(&d_ba1,B);
    gpufhe::ks_moddown(C,W,W.d_res0,d_ba0,0);
    gpufhe::ks_moddown(C,W,W.d_res1,d_ba1,0);
    cudaDeviceSynchronize();

    std::vector<uint64_t> hc0(T),hc1(T);
    cudaMemcpy(hc0.data(),d_ba0,B,cudaMemcpyDeviceToHost);
    cudaMemcpy(hc1.data(),d_ba1,B,cudaMemcpyDeviceToHost);
    cudaFree(d_c0);cudaFree(d_c1);cudaFree(d_scratch_qp);cudaFree(d_ba0);cudaFree(d_ba1);
    gpufhe::ks_work_destroy(W); gpufhe::ks_context_destroy(C);

    // Khoist and Ktruth use the IDENTICAL RNG seed (900+k) for their fresh
    // per-part u/e0/e1 draws (evalkeygen_host_sold's RNG depends only on
    // seed/ns/sigma/numPart, never on the supplied pkA/pkB/s2 values -- see
    // src/evalkeygen.cpp:74-100), so raw noise alone is NOT why a bit-exact
    // ciphertext comparison could legitimately fail here. But the hoisted
    // path routes c1 through decompose BEFORE automorphism while the ground
    // truth decomposes sigma_k(c1) directly -- those are only equal up to
    // ApproxSwitchCRTBasis's own rounding, which need not land on the exact
    // same residue. The invariant that must hold is "decrypts to the same
    // message", not "bit-identical ciphertext". Decrypt both under the same
    // secret (KPq.s, Q-range) and compare plaintexts with a noise-scale
    // tolerance instead.
    std::vector<int64_t> mg, mh;
    gpufhe::decrypt_host(mg, gc0, gc1, KPq.s, n, mod, root);
    gpufhe::decrypt_host(mh, hc0, hc1, KPq.s, n, mod, root);
    int64_t maxAbs=0, maxDiff=0;
    for(size_t i=0;i<mg.size();++i){
        int64_t a=(mg[i]<0?-mg[i]:mg[i]);
        if(a>maxAbs) maxAbs=a;
        int64_t d=mg[i]-mh[i]; if(d<0) d=-d;
        if(d>maxDiff) maxDiff=d;
    }
    // Noise-scale tolerance: full modulus chain product is astronomically
    // larger than the signal (scale 2^55); CKKS keyswitch noise after
    // moddown is bounded by roughly P-range rounding * secret norm, many
    // orders of magnitude below the signal itself. 2^-20 relative (~1e-6)
    // is generous for correctness (vs. the ~1e-7..1e-9 relative noise CKKS
    // keyswitching actually exhibits at this scale) while still catching a
    // genuine convention/algebra bug (which produces ~100% garbage, O(q)).
    bool ok = maxDiff < (maxAbs>>20) + (1LL<<30);
    std::cout<<"  k="<<k<<" kinv="<<kinv<<"  decrypted maxAbs="<<maxAbs
              <<" maxDiff="<<maxDiff<<(ok?" [close enough]":" [FAIL]")<<"\n";
    if(!ok){
        std::cout<<"    mg[0..3]="<<mg[0]<<","<<mg[1]<<","<<mg[2]<<","<<mg[3]<<"\n";
        std::cout<<"    mh[0..3]="<<mh[0]<<","<<mh[1]<<","<<mh[2]<<","<<mh[3]<<"\n";
    }
    return ok;
}

int main(){
    const uint32_t n=8192, tw=30, S=n/2, sizeP=2, numPart=(tw+1)/2; const uint64_t ns=1;
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> mq(mod),rq;
    std::vector<uint64_t> root(tw); for(uint32_t i=0;i<tw;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rq=root; for(auto p:modP){ mq.push_back(p); rq.push_back(gpufhe::native_root(n,p)); }
    const uint32_t sizeQlP=tw+sizeP;

    auto KPqp=gpufhe::keygen_host(n,mq,rq,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}

    // Rotation indices: small typical ones, plus edge cases -- M-1 (=2n-1,
    // the "most-wrapped" automorphism, analogous to a maximal negative
    // rotation), a large random odd index, and k=1 (identity, retained as a
    // regression anchor for the plumbing proof already established).
    const uint32_t M=2*n;
    std::vector<uint32_t> ks = {1u,5u,7u,11u,13u,99u, M-1u, 8191u, 12289u%M};
    for(auto &kk:ks) if((kk&1u)==0) kk+=1u; // keep odd (coprime to M=2^k)

    // Multiple ciphertexts / random plaintexts: 3 independent encryptions
    // with different seeds and different random slot content (not just the
    // single deterministic sin() vector from before).
    std::mt19937_64 prng(2026);
    std::uniform_real_distribution<double> ud(-0.5,0.5);
    bool allOk=true;
    size_t ncases=0, nfail=0;
    for(int trial=0; trial<3; ++trial){
        std::vector<cd> z(S);
        for(uint32_t i=0;i<S;++i) z[i]={ud(prng), (trial==0?0.0:ud(prng))}; // trial 0: real-only edge case
        std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,std::pow(2.0,55));
        std::vector<uint64_t> c0,c1;
        gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303+(uint64_t)trial*97);
        std::cout<<"trial "<<trial<<":\n";
        for(uint32_t k : ks){
            bool ok=run_case(n,tw,k,mod,root,modP,mq,rq,KPqp,KPq,PM,sizeP,numPart,c0,c1);
            ++ncases; if(!ok) ++nfail;
            allOk = allOk && ok;
        }
    }
    std::cout<<(allOk?"[PASS] ":"[FAIL] ")<<"correct-order hoisted single rotation vs proven rotate_ct_resident -- "
              <<"decrypted-plaintext comparison (noise-tolerant), "<<(ncases-nfail)<<"/"<<ncases<<" cases\n";
    return allOk?0:1;
}
