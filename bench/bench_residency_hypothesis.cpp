// GOD PROMPT "CKKS PHASE 28", Objective C: standalone, additive-only probe
// for the residency hypothesis "pt_to_eval_host's internal D2H, plus the
// subsequent H2D re-upload of that same plaintext for LaunchPtMulAcc, are a
// real but small fraction of the fused-kernel stage's time -- worth
// quantifying before touching src/rotate.cpp at all."
//
// This does NOT call pt_to_eval_host or modify any production file. It
// measures the raw D2H/H2D transfer cost for a buffer of the EXACT size
// pt_to_eval_host moves (towers*n*8 bytes) via cudaEvent-timed cudaMemcpy,
// isolating "how much of pt_to_eval_host's ~16% of stage time COULD be
// transfer, at most" from "how much is unavoidable NTT kernel compute" --
// without touching src/rotate.cpp's thread_local scratch() buffer or any
// other production state. Per the governing directive: hypothesis-only,
// standalone, before any production change.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

static float timeCudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, int reps) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < reps; ++i) cudaMemcpy(dst, src, bytes, kind);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / reps;
}

int main() {
    const uint32_t towers = 30, n = 8192;
    const size_t bytes = (size_t)towers * n * 8;
    const int REPS = 200;
    std::printf("=== Phase 28, Objective C: residency hypothesis -- transfer-only ceiling ===\n");
    std::printf("buffer size matching pt_to_eval_host's xf_multi (towers=%u, n=%u): %.3f MiB, %d reps\n",
                towers, n, bytes / 1048576.0, REPS);

    std::vector<uint64_t> h(towers * (size_t)n, 0xABCD);
    uint64_t* d = nullptr;
    cudaMalloc(&d, bytes);

    float h2d = timeCudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice, REPS);
    float d2h = timeCudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost, REPS);
    std::printf("  H2D (this buffer size): %.4f ms\n", h2d);
    std::printf("  D2H (this buffer size): %.4f ms\n", d2h);
    std::printf("  one plaintext's share of the stage-loop's combined H2D (the stage benchmark\n");
    std::printf("  uploads 3 equal-size buffers per diagonal: b0, b1, d -- d is 1 of 3): %.4f ms\n", h2d);

    // Context from bench_stage_timeline.cpp (Phase 28, this round, n=8192 tw=30
    // NDIAG=16): pt_to_eval_host (opaque, sums H2D+NTT+D2H across 16 diagonals)
    // averaged 0.1036s/trial = 6.475ms/diagonal; total Arm B stage mean was
    // 0.6326s/trial = 39.5ms/diagonal.
    const double ptEvalPerDiagMs = 103.6 / 16.0;
    const double stagePerDiagMs = 632.6 / 16.0;
    double d2hShareOfPtEval = 100.0 * d2h / ptEvalPerDiagMs;
    double h2dShareOfStage = 100.0 * h2d / stagePerDiagMs;
    std::printf("\n=== Interpretation ===\n");
    std::printf("  pt_to_eval_host per diagonal (measured, Arm B): %.3f ms\n", ptEvalPerDiagMs);
    std::printf("  of which this buffer's D2H alone is: %.3f ms (%.1f%% of pt_to_eval_host's own time)\n",
                d2h, d2hShareOfPtEval);
    std::printf("  => the REST (%.1f%%) of pt_to_eval_host's time is H2D-in + the batched NTT kernel\n",
                100.0 - d2hShareOfPtEval);
    std::printf("  itself, which a residency change CANNOT remove (the transform still has to run).\n");
    std::printf("  skipping the plaintext's share of the stage's separate H2D upload would save\n");
    std::printf("  ~%.3f ms/diagonal, %.2f%% of the stage's own per-diagonal total.\n", h2d, h2dShareOfStage);
    std::printf("  UPPER-BOUND estimate if BOTH this D2H and that H2D were fully eliminated by a\n");
    std::printf("  device-resident pt_to_eval variant: ~%.2f%% of stage time -- real, but an order\n",
                100.0 * (d2h + h2d) / stagePerDiagMs);
    std::printf("  of magnitude smaller than the encode_host_fft swap's measured 2.5x stage-level\n");
    std::printf("  speedup (research/ENCODE_FFT_PROMOTION.md). NOT recommended as this session's\n");
    std::printf("  next investment given that gap -- see research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md\n");
    std::printf("  addendum for the full writeup and why this is left for a future round.\n");

    cudaFree(d);
    return 0;
}
