// GOD PROMPT "CKKS PHASE 27", Mission 2: extract the REAL rotation-key
// access sequence tests/test_boot.cpp's bootstrap generates, WITHOUT running
// the bootstrap itself. The diagonal-finding and tower-bookkeeping logic
// (stageDiags/stageDiagsFwd/applyMerged/applyMergedFwd/rotAmt, the C2S/S2C
// stage loops' tower decrement) is PURE complex-arithmetic bookkeeping over
// vectors of size S=n/2 -- it touches no ciphertext, no device memory, no
// CUDA at all. This file replicates that logic VERBATIM (same n, LST, RST,
// tower-decrement pattern, EvalMod depth arithmetic) from test_boot.cpp's
// own source (read in full for this extraction) to produce the actual
// ordered sequence of (rotation_key, tower_count) pairs the real bootstrap
// would request from KROT/KSC/ks_context_create, in the EXACT order it
// would request them -- replacing Phase 26's "~77" estimate and its
// synthetic round-robin toy sequence with a measured, faithful one.
//
// This is NOT the bootstrap. No encryption, no CUDA, no device memory. Cheap
// enough to run at n=8192 (and even n=131072) on this machine regardless of
// the host-memory gate, because it is O(S^2) complex arithmetic over S=4096
// doubles-pairs, not an 8GB cryptographic pipeline.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>
#include <map>
#include <list>
#include <set>
#include <algorithm>

using cd = std::complex<double>;

struct Access { uint32_t k, tw; };  // one (rotation key, tower count) request

int main(int argc, char** argv) {
    const uint32_t n = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 8192;
    const uint32_t sizeQ = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 30;
    const uint32_t HW = 64;  // h=64, UNCHANGED -- used only for the EvalMod depth arithmetic below

    const uint32_t S = n / 2, M = 2 * n;
    uint32_t lgS = 0; while ((1u << lgS) < S) ++lgS;
    uint32_t LST = 3;
    // Mirror test_boot.cpp's own guard: LST must divide lgS.
    if (lgS % LST) {
        // fall back to a divisor, matching the project's own documented
        // usable-L table (n=8192 -> lgS=12 -> L in {2,3,4,6})
        for (uint32_t cand : {4u, 2u, 6u, 1u}) if (lgS % cand == 0) { LST = cand; break; }
    }
    const uint32_t RST = lgS / LST;
    std::printf("=== real key-access sequence extraction (n=%u, sizeQ=%u, LST=%u, RST=%u) ===\n",
                n, sizeQ, LST, RST);

    std::vector<uint64_t> rk(S);
    { uint64_t r = 1; for (uint32_t k = 0; k < S; ++k) { rk[k] = r; r = (r * 5) % M; } }
    auto rotAmt = [&](uint32_t r) -> uint32_t {
        uint64_t k = 1; for (uint32_t t = 0; t < r; ++t) k = (k * 5) % M; return (uint32_t)k;
    };
    auto twd = [&](uint32_t k, uint32_t st) -> cd {
        uint64_t e = ((uint64_t)rk[k] << st) % M;
        return std::polar(1.0, 2 * M_PI * (double)e / (double)M);
    };
    auto applyMerged = [&](std::vector<cd>& v, uint32_t g) {
        for (uint32_t st = g * RST; st < (g + 1) * RST; ++st) {
            uint32_t h = S >> (st + 1); std::vector<cd> o(S);
            for (uint32_t base = 0; base < S; base += 2 * h)
                for (uint32_t k = 0; k < h; ++k) {
                    cd a = v[base + k], b = v[base + k + h], t = twd(k, st);
                    o[base + k] = (a + b) * 0.5; o[base + k + h] = (a - b) / (2.0 * t);
                }
            v.swap(o);
        }
    };
    auto applyMergedFwd = [&](std::vector<cd>& v, uint32_t g) {
        for (int st = (int)((g + 1) * RST) - 1; st >= (int)(g * RST); --st) {
            uint32_t h = S >> (st + 1); std::vector<cd> o(S);
            for (uint32_t base = 0; base < S; base += 2 * h)
                for (uint32_t k = 0; k < h; ++k) {
                    cd a = v[base + k], b = v[base + k + h], t = twd(k, (uint32_t)st);
                    o[base + k] = a + t * b; o[base + k + h] = a - t * b;
                }
            v.swap(o);
        }
    };
    auto stageDiags = [&](uint32_t g, std::vector<uint32_t>& offs) {
        std::vector<uint8_t> used(S, 0);
        for (uint32_t jj = 0; jj < S; ++jj) {
            std::vector<cd> u(S, cd{0, 0}); u[jj] = 1; applyMerged(u, g);
            for (uint32_t ii = 0; ii < S; ++ii) if (std::abs(u[ii]) > 1e-12) {
                uint32_t d = (jj + S - ii) % S; used[d] = 1;
            }
        }
        offs.clear();
        for (uint32_t d = 0; d < S; ++d) if (used[d]) offs.push_back(d);
    };
    auto stageDiagsFwd = [&](uint32_t g, std::vector<uint32_t>& offs) {
        std::vector<uint8_t> used(S, 0);
        for (uint32_t jj = 0; jj < S; ++jj) {
            std::vector<cd> u(S, cd{0, 0}); u[jj] = 1; applyMergedFwd(u, g);
            for (uint32_t ii = 0; ii < S; ++ii) if (std::abs(u[ii]) > 1e-12) {
                uint32_t d = (jj + S - ii) % S; used[d] = 1;
            }
        }
        offs.clear();
        for (uint32_t d = 0; d < S; ++d) if (used[d]) offs.push_back(d);
    };

    // EvalMod depth arithmetic, copied verbatim from test_boot.cpp line 773-776:
    const double Kbud = (HW ? 2.5 * std::sqrt((double)HW) : 40.0 * std::sqrt((double)n / 1024.0));
    const uint32_t rDbl = (uint32_t)std::ceil(std::log2(2 * M_PI * (Kbud + 0.25)));
    const uint32_t degC = 10;
    const uint32_t evalModOutTw = sizeQ - (4 + degC + rDbl);
    std::printf("EvalMod: Kbud=%.3f rDbl=%u degC=%u -> output tw=%u\n", Kbud, rDbl, degC, evalModOutTw);

    std::vector<Access> seq;  // the REAL ordered access sequence

    // ---- C2S: ctw starts at sizeQ, decrements by 1 after each of LST stages ----
    uint32_t ctw = sizeQ;
    for (uint32_t g = 0; g < LST; ++g) {
        std::vector<uint32_t> offs;
        stageDiags(g, offs);
        for (uint32_t off : offs) if (off) seq.push_back({rotAmt(off), ctw});  // off==0 needs no rotation (identity)
        --ctw;
    }
    const uint32_t c2sFinalTw = ctw;  // tw right after C2S's 3 rescales (== ModRaise_tw - 3)

    // ---- S2C: stw starts at EvalMod's output tw, decrements after each of LST stages,
    // processed g = LST-1 .. 0 (reverse order). The FIRST stage processed
    // (g==LST-1) runs TWO passes (real branch oR, imaginary branch oI) over
    // the SAME offs set -- doubling that stage's access count, not its
    // DISTINCT key set. ----
    uint32_t stw = evalModOutTw;
    for (int g = (int)LST - 1; g >= 0; --g) {
        std::vector<uint32_t> offs;
        stageDiagsFwd((uint32_t)g, offs);
        const uint32_t passes = ((uint32_t)g == LST - 1) ? 2u : 1u;
        for (uint32_t pass = 0; pass < passes; ++pass)
            for (uint32_t off : offs) if (off) seq.push_back({rotAmt(off), stw});
        --stw;
    }

    // Also account for the two fixed conjugation rotations (M-1) used once
    // each for the real/imaginary split before EvalMod (test_boot.cpp lines
    // around 633, 637-ish: "auto Kl=atLevel(rotFull(M-1,7400),nt)") -- these
    // use the modraise-stage tower count, which is c2sFinalTw (post-C2S,
    // pre-EvalMod). Included for completeness; they are a tiny fraction of
    // the total and use a DISTINCT key (M-1, the conjugation automorphism).
    seq.push_back({(uint32_t)(M - 1), c2sFinalTw});

    std::printf("\nTotal diagonal rotation accesses (excluding off==0 identity diagonals): %zu\n", seq.size());

    // ---- distinct (k, tw) pairs: the TRUE device-context working set ----
    std::set<std::pair<uint32_t, uint32_t>> distinctKTw;
    std::set<uint32_t> distinctK;
    for (auto& a : seq) { distinctKTw.insert({a.k, a.tw}); distinctK.insert(a.k); }
    std::printf("Distinct (k, tw) pairs [KSC's real working set]  : %zu\n", distinctKTw.size());
    std::printf("Distinct k alone (ignoring tw) [KROT's real set]  : %zu\n", distinctK.size());

    // breakdown by tw
    std::map<uint32_t, size_t> perTw, perTwDistinctK;
    std::map<uint32_t, std::set<uint32_t>> keysAtTw;
    for (auto& a : seq) { perTw[a.tw]++; keysAtTw[a.tw].insert(a.k); }
    std::printf("\nPer-tower-count breakdown:\n");
    for (auto& kv : perTw)
        std::printf("  tw=%2u: %4zu accesses, %4zu distinct keys\n", kv.first, kv.second, keysAtTw[kv.first].size());

    // ---- LRU cache hit-rate simulation against the REAL sequence ----
    std::printf("\n=== LRU hit-rate simulation (key on (k,tw), REAL access order) ===\n");
    std::printf("%-8s %-8s %-8s %-10s\n", "cap", "hits", "misses", "hit_rate");
    for (uint32_t cap : {1u, 2u, 4u, 8u, 16u, 24u, 32u, 40u, 48u, 64u, (uint32_t)distinctKTw.size(), 96u}) {
        std::map<std::pair<uint32_t,uint32_t>, std::list<std::pair<uint32_t,uint32_t>>::iterator> pos;
        std::list<std::pair<uint32_t,uint32_t>> lru;
        std::set<std::pair<uint32_t,uint32_t>> present;
        size_t hits = 0, misses = 0;
        for (auto& a : seq) {
            auto key = std::make_pair(a.k, a.tw);
            if (present.count(key)) {
                ++hits;
                lru.erase(pos[key]); lru.push_front(key); pos[key] = lru.begin();
            } else {
                ++misses;
                if (present.size() >= cap) {
                    auto victim = lru.back(); lru.pop_back(); pos.erase(victim); present.erase(victim);
                }
                lru.push_front(key); pos[key] = lru.begin(); present.insert(key);
            }
        }
        std::printf("%-8u %-8zu %-8zu %.1f%%\n", cap, hits, misses, 100.0 * hits / (hits + misses));
    }

    return 0;
}
