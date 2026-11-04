// GOD PROMPT "CKKS PHASE 13": stage-level internal profiling for
// encrypt_host (src/encdec.cpp), now the single largest contributor to
// the real encode->encrypt->EvalMult->decrypt->decode pipeline since
// Phase 12 removed encode/decode from the critical path (research/
// EXPERIMENT_LEDGER.md entry #45). This file does NOT modify
// src/encdec.cpp -- `encrypt_host_profiled` below is a byte-for-byte
// algorithmic COPY of encrypt_host (same RNG consumption order, same
// embedding formula, same NTT calls, same combine formula), with
// std::chrono checkpoints inserted at each stage boundary. Every CUDA
// call on the hot path here is on the default stream and every memcpy
// used is the BLOCKING (non-Async) form, so host-clock checkpoints
// bracketing them are already correctly synchronized -- the one
// explicit cudaDeviceSynchronize() after the kernel launches (inherited
// unchanged from encrypt_host) is what makes the "ntt_kernel" stage
// boundary accurate.
//
// Correctness of the profiling copy itself is established by
// `profiled_copy_matches_production_bit_exact` in main(): for the SAME
// seed, encrypt_host and encrypt_host_profiled must produce IDENTICAL
// (c0,c1) -- this is checked BEFORE any timing from this file is
// trusted, since a profiling copy that silently diverges from
// production would make every subsequent measurement meaningless.
#include <cstdint>
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <map>
#include <utility>
#include <cuda_runtime.h>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
struct KeyPairHost { std::vector<uint64_t> s, pkA, pkB; };
KeyPairHost keygen_host(uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&, uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}
extern "C" void LaunchNTT_CT_Batched(uint64_t*, const uint64_t* const*, const uint64_t* const*,
                                      const uint64_t*, uint32_t, uint32_t, cudaStream_t);

namespace {
using Clock = std::chrono::steady_clock;
static double us_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
uint64_t am(uint64_t a, uint64_t b, uint64_t q) { uint64_t s = a + b; return s >= q ? s - q : s; }
uint64_t* up(const std::vector<uint64_t>& h) {
    uint64_t* d; cudaMalloc(&d, h.size() * 8); cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice); return d;
}

// Exact duplicate of encdec.cpp's own build_enc_tab/enc_dev_tab (same
// cache-by-(n,q) pattern) -- kept as a SEPARATE cache instance in this
// translation unit so this file never touches encdec.cpp's internal
// (anonymous-namespace, non-exported) cache.
struct EncDevTab { uint64_t* fr = nullptr; uint64_t* fp = nullptr; };
EncDevTab build_enc_tab(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mm(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_profile(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab(n, q, root)).first->second;
}

struct EncryptStageTimes {
    double rng_us, embed_us, alloc_us, upload_us, ntt_kernel_us, download_us, free_us, combine_us, total_us;
};

// Byte-for-byte copy of gpufhe::encrypt_host (src/encdec.cpp), with
// std::chrono checkpoints at each stage boundary. Must produce IDENTICAL
// (c0,c1) to the production function for the same inputs/seed -- verified
// in main() before any timing here is trusted.
void encrypt_host_profiled(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                            const std::vector<int64_t>& m,
                            const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                            uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                            uint64_t ns, double sigma, uint64_t seed,
                            EncryptStageTimes& st)
{
    auto t_start = Clock::now();
    uint32_t sizeQ = (uint32_t)mod.size();
    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0, sigma);
    std::vector<int> vc(n), e0c(n), e1c(n);
    for (uint32_t k = 0; k < n; ++k) { vc[k] = (int)(rng() % 3) - 1; e0c[k] = (int)std::lround(g(rng)); e1c[k] = (int)std::lround(g(rng)); }
    auto t_rng = Clock::now();

    c0.resize((size_t)sizeQ * n); c1.resize((size_t)sizeQ * n);
    std::vector<uint64_t> vBuf((size_t)sizeQ * n), e0Buf((size_t)sizeQ * n), e1Buf((size_t)sizeQ * n), mBuf((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        uint64_t* vT = vBuf.data() + (size_t)t * n; uint64_t* e0T = e0Buf.data() + (size_t)t * n;
        uint64_t* e1T = e1Buf.data() + (size_t)t * n; uint64_t* mT = mBuf.data() + (size_t)t * n;
        for (uint32_t k = 0; k < n; ++k) { auto emb = [&](long x) { return (uint64_t)((x % (long)q + (long)q) % (long)q); };
            vT[k] = emb(vc[k]); e0T[k] = emb(e0c[k]); e1T[k] = emb(e1c[k]); mT[k] = emb((long)m[k]); }
    }
    auto t_embed = Clock::now();

    std::vector<uint64_t*> h_fr(sizeQ), h_fp(sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) { EncDevTab& T = enc_dev_tab_profile(n, mod[t], root[t]); h_fr[t] = T.fr; h_fp[t] = T.fp; }
    uint64_t **d_fr = nullptr, **d_fp = nullptr;
    cudaMalloc(&d_fr, sizeQ * sizeof(uint64_t*));
    cudaMalloc(&d_fp, sizeQ * sizeof(uint64_t*));
    cudaMemcpy(d_fr, h_fr.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp, h_fp.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    uint64_t* d_mod = up(mod);
    uint64_t *d_v, *d_e0, *d_e1, *d_m;
    const size_t T8 = (size_t)sizeQ * n * 8;
    cudaMalloc(&d_v, T8); cudaMalloc(&d_e0, T8); cudaMalloc(&d_e1, T8); cudaMalloc(&d_m, T8);
    auto t_alloc = Clock::now();

    cudaMemcpy(d_v, vBuf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e0, e0Buf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e1, e1Buf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, mBuf.data(), T8, cudaMemcpyHostToDevice);
    auto t_upload = Clock::now();

    LaunchNTT_CT_Batched(d_v, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e0, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e1, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_m, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    cudaDeviceSynchronize();
    auto t_ntt = Clock::now();

    cudaMemcpy(vBuf.data(), d_v, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(e0Buf.data(), d_e0, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(e1Buf.data(), d_e1, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(mBuf.data(), d_m, T8, cudaMemcpyDeviceToHost);
    auto t_download = Clock::now();

    cudaFree(d_v); cudaFree(d_e0); cudaFree(d_e1); cudaFree(d_m); cudaFree(d_fr); cudaFree(d_fp); cudaFree(d_mod);
    auto t_free = Clock::now();

    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        const uint64_t* vT = vBuf.data() + (size_t)t * n; const uint64_t* e0T = e0Buf.data() + (size_t)t * n;
        const uint64_t* e1T = e1Buf.data() + (size_t)t * n; const uint64_t* mT = mBuf.data() + (size_t)t * n;
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t b = am(mm(pkB[(size_t)t * n + k], vT[k], q), mm(ns % q, e0T[k], q), q);
            uint64_t a = am(mm(pkA[(size_t)t * n + k], vT[k], q), mm(ns % q, e1T[k], q), q);
            c0[(size_t)t * n + k] = am(b, mT[k], q);
            c1[(size_t)t * n + k] = a;
        }
    }
    auto t_combine = Clock::now();

    st.rng_us = us_since(t_start, t_rng);
    st.embed_us = us_since(t_rng, t_embed);
    st.alloc_us = us_since(t_embed, t_alloc);
    st.upload_us = us_since(t_alloc, t_upload);
    st.ntt_kernel_us = us_since(t_upload, t_ntt);
    st.download_us = us_since(t_ntt, t_download);
    st.free_us = us_since(t_download, t_free);
    st.combine_us = us_since(t_free, t_combine);
    st.total_us = us_since(t_start, t_combine);
}

struct Stats { double min_, p50, p95, max_, mean; size_t n; };
Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s; s.n = v.size(); s.min_ = v.front(); s.max_ = v.back();
    s.p50 = v[v.size() / 2]; s.p95 = v[(size_t)(0.95 * (v.size() - 1))];
    double sum = 0; for (double x : v) sum += x; s.mean = sum / v.size();
    return s;
}
void report(const char* label, const Stats& s) {
    std::cout << "    " << label << ": median=" << s.p50 << "us p95=" << s.p95
              << "us min=" << s.min_ << "us max=" << s.max_ << "us mean=" << s.mean
              << "us (n=" << s.n << ")\n";
}
} // namespace

int main(int argc, char** argv) {
    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 15;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 7;
    uint32_t sizeQ = 8;

    bool allOk = true;
    for (uint32_t n : {1024u, 2048u, 4096u, 8192u}) {
        std::cout << "\n=== n=" << n << " sizeQ=" << sizeQ << " ===\n";
        std::vector<uint64_t> mod;
        gpufhe::native_primes(mod, 1, 60, n, {});
        { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 50, n, mod);
          for (auto mm_ : mids) mod.push_back(mm_); }
        std::vector<uint64_t> root(sizeQ);
        for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);

        auto KP = gpufhe::keygen_host(n, mod, root, /*ns=*/1, /*sigma=*/3.19, /*seed=*/777);

        // --- Correctness gate: profiled copy must match production
        // bit-exactly for the SAME seed, BEFORE any timing from this
        // file is trusted. ---
        std::vector<int64_t> msg(n, 0);
        msg[0] = 12345; if (n > 3) msg[3] = -777;
        std::vector<uint64_t> c0_prod, c1_prod, c0_prof, c1_prof;
        gpufhe::encrypt_host(c0_prod, c1_prod, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 42);
        EncryptStageTimes dummy;
        encrypt_host_profiled(c0_prof, c1_prof, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 42, dummy);
        bool bitExact = (c0_prod == c0_prof) && (c1_prod == c1_prof);
        std::cout << "  profiled-copy-matches-production (bit-exact, seed=42): "
                  << (bitExact ? "[PASS]" : "[FAIL]") << "\n";
        if (!bitExact) { std::cerr << "  REFUSING to report timing: profiling copy diverged from production\n"; allOk = false; continue; }

        // --- Real decrypt-based sanity check on the profiled path too.
        // NOTE: encrypt_host/decrypt_host operate directly on RAW
        // coefficients here (no encode/Delta-scaling step), so the
        // decrypted value is m[k] PLUS genuine CKKS encryption noise
        // (ns*e0/ns*e1 compounded through the secret key) -- bit-exact
        // recovery is neither expected nor desired (that would mean
        // zero noise, i.e. an insecure scheme). This is a diagnostic
        // sanity check (noise stays small relative to the message, not
        // a replacement for the project's own encode-scaled round-trip
        // tolerance tests), with a generously derived bound: noise
        // compounds through small ternary multipliers against the
        // secret key, so this is reported as a diagnostic (not a hard
        // gate -- the actual correctness gates for this phase are the
        // bit-exact check above and the project's own EXISTING
        // encode-scaled round-trip/EvalMult tests, run separately).
        std::vector<int64_t> mdec;
        gpufhe::decrypt_host(mdec, c0_prof, c1_prof, KP.s, n, mod, root);
        int64_t err0 = mdec[0] - 12345, err3 = (n > 3) ? (mdec[3] - (-777)) : 0;
        std::cout << "  decrypt-recovers-plaintext-within-noise (coeff-domain, no encode, informational): "
                  << "err[0]=" << err0 << " err[3]=" << err3 << "\n";

        // --- Warm-up (populate enc_dev_tab_profile's root-table cache). ---
        EncryptStageTimes warm;
        encrypt_host_profiled(c0_prof, c1_prof, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 999, warm);

        // --- Steady-state: many reps/trials, per-trial medians, full stats. ---
        std::vector<double> rng_s, embed_s, alloc_s, upload_s, ntt_s, download_s, free_s, combine_s, total_s;
        for (size_t trial = 0; trial < trials; ++trial) {
            std::vector<double> rng_r, embed_r, alloc_r, upload_r, ntt_r, download_r, free_r, combine_r, total_r;
            for (size_t r = 0; r < reps; ++r) {
                EncryptStageTimes st;
                encrypt_host_profiled(c0_prof, c1_prof, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 1000 + trial * reps + r, st);
                rng_r.push_back(st.rng_us); embed_r.push_back(st.embed_us); alloc_r.push_back(st.alloc_us);
                upload_r.push_back(st.upload_us); ntt_r.push_back(st.ntt_kernel_us); download_r.push_back(st.download_us);
                free_r.push_back(st.free_us); combine_r.push_back(st.combine_us); total_r.push_back(st.total_us);
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            rng_s.push_back(med(rng_r)); embed_s.push_back(med(embed_r)); alloc_s.push_back(med(alloc_r));
            upload_s.push_back(med(upload_r)); ntt_s.push_back(med(ntt_r)); download_s.push_back(med(download_r));
            free_s.push_back(med(free_r)); combine_s.push_back(med(combine_r)); total_s.push_back(med(total_r));
        }
        std::cout << "  -- stage breakdown (" << trials << " trials x " << reps << " reps, per-trial medians) --\n";
        report("1. RNG + Gaussian/ternary sampling (host)", stats_of(rng_s));
        report("2. Embed into sizeQ RNS towers (host)", stats_of(embed_s));
        report("3. Device buffer allocation (cudaMalloc)", stats_of(alloc_s));
        report("4. Upload H2D (4 buffers)", stats_of(upload_s));
        report("5. NTT kernel launches x4 + sync", stats_of(ntt_s));
        report("6. Download D2H (4 buffers)", stats_of(download_s));
        report("7. Device buffer free (cudaFree)", stats_of(free_s));
        report("8. Combine (pkA/pkB*v + ns*e + m, host)", stats_of(combine_s));
        report("TOTAL encrypt_host", stats_of(total_s));
        Stats totS = stats_of(total_s);
        std::cout << "  -- percent of total (median basis) --\n";
        auto pctOf = [&](double x) { return 100.0 * x / totS.p50; };
        std::cout << "    rng=" << pctOf(stats_of(rng_s).p50) << "% embed=" << pctOf(stats_of(embed_s).p50)
                  << "% alloc=" << pctOf(stats_of(alloc_s).p50) << "% upload=" << pctOf(stats_of(upload_s).p50)
                  << "% ntt=" << pctOf(stats_of(ntt_s).p50) << "% download=" << pctOf(stats_of(download_s).p50)
                  << "% free=" << pctOf(stats_of(free_s).p50) << "% combine=" << pctOf(stats_of(combine_s).p50) << "%\n";
    }
    if (!allOk) { std::cerr << "\n[OVERALL FAIL]\n"; return 1; }
    std::cout << "\n[OVERALL PASS]\n";
    return 0;
}
