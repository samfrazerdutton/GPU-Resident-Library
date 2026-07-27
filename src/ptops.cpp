// Plaintext-ciphertext ops (host reference for gates; pointwise, no keyswitch).
//   ct_add_pt:   c0 += encode(pt)          (scale must match)
//   ct_mul_pt:   c0 *= pt_eval, c1 *= pt_eval   (scale multiplies -> rescale after)
//   ct_add_ct:   c0+=d0, c1+=d1
// pt_eval = NTT of the encoded plaintext per tower.
#include <cstdint>
#include <vector>
namespace gpufhe {
namespace {
uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
}
void ct_add_ct_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                    const std::vector<uint64_t>& d0, const std::vector<uint64_t>& d1,
                    uint32_t towers, uint32_t n, const std::vector<uint64_t>& mod){
    for(uint32_t t=0;t<towers;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
            c0[x]=am(c0[x],d0[x],q); c1[x]=am(c1[x],d1[x],q);}}
}
void ct_mul_pt_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                    const std::vector<uint64_t>& ptEval,
                    uint32_t towers, uint32_t n, const std::vector<uint64_t>& mod){
    for(uint32_t t=0;t<towers;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
            c0[x]=mm(c0[x],ptEval[x],q); c1[x]=mm(c1[x],ptEval[x],q);}}
}
// embed an encoded integer poly into towers*n eval form
void pt_to_eval_host(std::vector<uint64_t>& out, const std::vector<int64_t>& m,
                     uint32_t towers, uint32_t n, const std::vector<uint64_t>& mod,
                     const std::vector<uint64_t>& root);
} // namespace gpufhe
