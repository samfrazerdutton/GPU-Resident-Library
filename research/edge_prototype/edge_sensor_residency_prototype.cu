// Phase 32, Work Package C: standalone, self-contained prototype demonstrating
// GPU-resident intermediate state across a chained sensor-fusion-style pipeline.
// Deliberately written against raw CUDA (no project library dependency) because
// the Phase 32 architecture audit (Work Package A) runs in parallel and had not
// landed a stable buffer/context primitive at the time this was written -- see
// research/EDGE_SENSOR_RESIDENCY_AUDIT.md section "Limitations" for the
// migration note. Not part of the main CMake build; compiled and run standalone
// so it cannot collide with or affect the project's build/test configuration.
//
// Synthetic pipeline (numerical/memory-residency demonstration only -- NOT a
// robotics, autonomy, or safety system):
//   - A resident point cloud (N points, xyz) uploaded ONCE.
//   - A resident 3x4 pose transform matrix, updated in-place on the GPU each
//     step from a small (6-float) per-step delta -- a simplified small-angle
//     incremental rotation, not a rigorous SO(3)/quaternion integrator (stated
//     explicitly: this is a numerical demo, not a validated pose estimator).
//   - Each step, chained ON THE GPU with no host round-trip in between:
//       pose_update -> transform all points by the resident pose ->
//       exponential-moving-average filter into a resident filtered buffer.
//   - Every K steps, the filtered cloud's centroid is reduced on the GPU and
//     the resulting 3 floats are read back to the host (periodic retrieval).
//   - A single-threaded CPU reference repeats the identical sequence
//     (same seed, same deltas) for correctness comparison.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1);} }while(0)

static const uint32_t N_POINTS = 1'000'000;   // 12 MB resident cloud (xyz, float)
static const uint32_t N_STEPS  = 500;
static const uint32_t READBACK_EVERY = 50;    // 10 periodic readbacks total
static const float ALPHA = 0.1f;              // EMA filter coefficient

__device__ float g_pose[12]; // resident 3x4 row-major [R|t]

__global__ void pose_update_kernel(const float* delta /*6: dx,dy,dz,drx,dry,drz*/){
    // Single-thread kernel: small-angle incremental update of the resident
    // pose. Deliberately simplified (first-order small-angle approx, not a
    // proper rotation-matrix renormalization) -- adequate for a numerical
    // residency demo at the small per-step angles used here, NOT a
    // general-purpose pose integrator.
    if (threadIdx.x==0 && blockIdx.x==0){
        float dx=delta[0], dy=delta[1], dz=delta[2];
        float rx=delta[3], ry=delta[4], rz=delta[5];
        float R[9]; for(int i=0;i<3;i++) for(int j=0;j<3;j++) R[i*3+j]=g_pose[i*3+j];
        float t[3]={g_pose[9],g_pose[10],g_pose[11]};
        // incremental rotation (first-order): R' = (I + skew(rz,ry,rx)) * R
        float dR[9] = {1,-rz,ry,  rz,1,-rx,  -ry,rx,1};
        float Rn[9];
        for(int i=0;i<3;i++) for(int j=0;j<3;j++){
            float s=0; for(int k=0;k<3;k++) s+=dR[i*3+k]*R[k*3+j];
            Rn[i*3+j]=s;
        }
        for(int i=0;i<9;i++) g_pose[i]=Rn[i];
        g_pose[9]=t[0]+dx; g_pose[10]=t[1]+dy; g_pose[11]=t[2]+dz;
    }
}

__global__ void transform_kernel(const float* pts, float* out, uint32_t n){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=n) return;
    float x=pts[i*3+0], y=pts[i*3+1], z=pts[i*3+2];
    float ox = g_pose[0]*x+g_pose[1]*y+g_pose[2]*z+g_pose[9];
    float oy = g_pose[3]*x+g_pose[4]*y+g_pose[5]*z+g_pose[10];
    float oz = g_pose[6]*x+g_pose[7]*y+g_pose[8]*z+g_pose[11];
    out[i*3+0]=ox; out[i*3+1]=oy; out[i*3+2]=oz;
}

__global__ void ema_filter_kernel(const float* transformed, float* filtered, uint32_t n, float alpha){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=n) return;
    for(int c=0;c<3;c++){
        float prev = filtered[i*3+c];
        filtered[i*3+c] = alpha*transformed[i*3+c] + (1.0f-alpha)*prev;
    }
}

__global__ void centroid_reduce_kernel(const float* filtered, uint32_t n, float* out_sum /*3*/){
    __shared__ float sx[256], sy[256], sz[256];
    uint32_t tid=threadIdx.x;
    float ax=0,ay=0,az=0;
    for(uint32_t i=blockIdx.x*blockDim.x+tid; i<n; i+=gridDim.x*blockDim.x){
        ax+=filtered[i*3+0]; ay+=filtered[i*3+1]; az+=filtered[i*3+2];
    }
    sx[tid]=ax; sy[tid]=ay; sz[tid]=az;
    __syncthreads();
    for(uint32_t s=blockDim.x/2; s>0; s>>=1){
        if (tid<s){ sx[tid]+=sx[tid+s]; sy[tid]+=sy[tid+s]; sz[tid]+=sz[tid+s]; }
        __syncthreads();
    }
    if (tid==0){ atomicAdd(&out_sum[0],sx[0]); atomicAdd(&out_sum[1],sy[0]); atomicAdd(&out_sum[2],sz[0]); }
}

struct Delta { float v[6]; };

static std::vector<Delta> genDeltas(uint32_t steps, uint64_t seed){
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dT(-0.01f,0.01f), dR(-0.002f,0.002f);
    std::vector<Delta> d(steps);
    for(auto& x: d){ x.v[0]=dT(rng); x.v[1]=dT(rng); x.v[2]=dT(rng); x.v[3]=dR(rng); x.v[4]=dR(rng); x.v[5]=dR(rng); }
    return d;
}

static void cpuReference(const std::vector<float>& pts, const std::vector<Delta>& deltas,
                          std::vector<float>& centroidsOut /*3 per readback*/){
    uint32_t n = pts.size()/3;
    std::vector<float> transformed(n*3), filtered(n*3,0.0f);
    float pose[12] = {1,0,0, 0,1,0, 0,0,1, 0,0,0};
    for (size_t step=0; step<deltas.size(); ++step){
        const Delta& d = deltas[step];
        float R[9]; for(int i=0;i<9;i++) R[i]=pose[i];
        float t[3]={pose[9],pose[10],pose[11]};
        float dR[9] = {1,-d.v[5],d.v[4],  d.v[5],1,-d.v[3],  -d.v[4],d.v[3],1};
        float Rn[9];
        for(int i=0;i<3;i++) for(int j=0;j<3;j++){
            float s=0; for(int k=0;k<3;k++) s+=dR[i*3+k]*R[k*3+j];
            Rn[i*3+j]=s;
        }
        for(int i=0;i<9;i++) pose[i]=Rn[i];
        pose[9]=t[0]+d.v[0]; pose[10]=t[1]+d.v[1]; pose[11]=t[2]+d.v[2];
        for(uint32_t i=0;i<n;i++){
            float x=pts[i*3+0], y=pts[i*3+1], z=pts[i*3+2];
            transformed[i*3+0] = pose[0]*x+pose[1]*y+pose[2]*z+pose[9];
            transformed[i*3+1] = pose[3]*x+pose[4]*y+pose[5]*z+pose[10];
            transformed[i*3+2] = pose[6]*x+pose[7]*y+pose[8]*z+pose[11];
        }
        for(uint32_t i=0;i<n*3;i++) filtered[i] = ALPHA*transformed[i] + (1.0f-ALPHA)*filtered[i];
        if ((step+1)%READBACK_EVERY==0){
            double cx=0,cy=0,cz=0;
            for(uint32_t i=0;i<n;i++){ cx+=filtered[i*3+0]; cy+=filtered[i*3+1]; cz+=filtered[i*3+2]; }
            centroidsOut.push_back((float)(cx/n)); centroidsOut.push_back((float)(cy/n)); centroidsOut.push_back((float)(cz/n));
        }
    }
}

int main(){
    std::printf("=== Phase 32 edge/sensor-fusion GPU-residency prototype ===\n");
    std::printf("N_POINTS=%u N_STEPS=%u READBACK_EVERY=%u ALPHA=%.2f\n", N_POINTS, N_STEPS, READBACK_EVERY, ALPHA);

    size_t freeB0, totalB; CK(cudaMemGetInfo(&freeB0,&totalB));

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> U(-10.f,10.f);
    std::vector<float> pts(N_POINTS*3);
    for(auto& v: pts) v = U(rng);
    auto deltas = genDeltas(N_STEPS, 1234);

    float *d_pts, *d_transformed, *d_filtered, *d_delta, *d_centroid;
    CK(cudaMalloc(&d_pts, N_POINTS*3*sizeof(float)));
    CK(cudaMalloc(&d_transformed, N_POINTS*3*sizeof(float)));
    CK(cudaMalloc(&d_filtered, N_POINTS*3*sizeof(float)));
    CK(cudaMalloc(&d_delta, 6*sizeof(float)));
    CK(cudaMalloc(&d_centroid, 3*sizeof(float)));
    CK(cudaMemset(d_filtered, 0, N_POINTS*3*sizeof(float)));

    float identityPose[12] = {1,0,0, 0,1,0, 0,0,1, 0,0,0};
    CK(cudaMemcpyToSymbol(g_pose, identityPose, 12*sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    uint32_t block=256, grid=(N_POINTS+block-1)/block;

    // --- initial upload latency ---
    auto t0 = std::chrono::high_resolution_clock::now();
    CK(cudaMemcpyAsync(d_pts, pts.data(), N_POINTS*3*sizeof(float), cudaMemcpyHostToDevice, s));
    CK(cudaStreamSynchronize(s));
    auto t1 = std::chrono::high_resolution_clock::now();
    double uploadMs = std::chrono::duration<double,std::milli>(t1-t0).count();

    size_t h2dBytes = N_POINTS*3*sizeof(float);
    size_t d2hBytes = 0;
    uint32_t syncCount = 1; // the upload sync above

    std::vector<double> stepMsAll;
    std::vector<float> gpuCentroids;

    for (uint32_t step=0; step<N_STEPS; ++step){
        auto a0 = std::chrono::high_resolution_clock::now();
        CK(cudaMemcpyAsync(d_delta, deltas[step].v, 6*sizeof(float), cudaMemcpyHostToDevice, s));
        h2dBytes += 6*sizeof(float);
        pose_update_kernel<<<1,1,0,s>>>(d_delta);
        transform_kernel<<<grid,block,0,s>>>(d_pts, d_transformed, N_POINTS);
        ema_filter_kernel<<<grid,block,0,s>>>(d_transformed, d_filtered, N_POINTS, ALPHA);
        if ((step+1)%READBACK_EVERY==0){
            float zero3[3]={0,0,0};
            CK(cudaMemcpyAsync(d_centroid, zero3, 3*sizeof(float), cudaMemcpyHostToDevice, s));
            centroid_reduce_kernel<<<64,256,0,s>>>(d_filtered, N_POINTS, d_centroid);
            float c[3];
            CK(cudaMemcpyAsync(c, d_centroid, 3*sizeof(float), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            syncCount++;
            d2hBytes += 3*sizeof(float);
            gpuCentroids.push_back(c[0]/(float)N_POINTS); gpuCentroids.push_back(c[1]/(float)N_POINTS); gpuCentroids.push_back(c[2]/(float)N_POINTS);
        } else {
            CK(cudaStreamSynchronize(s)); // measuring per-step latency honestly; see note below
            syncCount++;
        }
        auto a1 = std::chrono::high_resolution_clock::now();
        stepMsAll.push_back(std::chrono::duration<double,std::milli>(a1-a0).count());
    }

    size_t freeB1; CK(cudaMemGetInfo(&freeB1,&totalB));
    size_t peakResidentBytes = freeB0 - freeB1; // approx: this process's net device allocation

    // cold (first step) vs warm (median of steps 50..499, past warmup) latency
    double coldMs = stepMsAll[0];
    std::vector<double> warm(stepMsAll.begin()+50, stepMsAll.end());
    std::sort(warm.begin(), warm.end());
    double warmMedianMs = warm[warm.size()/2];

    // --- CPU reference, timed ---
    auto c0 = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuCentroids;
    cpuReference(pts, deltas, cpuCentroids);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpuTotalMs = std::chrono::duration<double,std::milli>(c1-c0).count();
    double cpuPerStepMs = cpuTotalMs / N_STEPS;

    double maxAbsErr = 0;
    for (size_t i=0;i<gpuCentroids.size() && i<cpuCentroids.size(); ++i)
        maxAbsErr = std::max(maxAbsErr, (double)std::fabs(gpuCentroids[i]-cpuCentroids[i]));

    std::printf("\n-- Results --\n");
    std::printf("initial upload: %.3f ms (%.3f MB)\n", uploadMs, h2dBytes/1e6);
    std::printf("per-step latency: cold(step0)=%.4f ms, warm-median(steps 50-499)=%.4f ms\n", coldMs, warmMedianMs);
    std::printf("total GPU-side wall time for %u steps: %.3f ms (%.4f ms/step avg)\n",
                N_STEPS, [&]{double s=0; for(auto v:stepMsAll)s+=v; return s;}(), [&]{double s=0; for(auto v:stepMsAll)s+=v; return s/N_STEPS;}());
    std::printf("CPU reference total: %.3f ms (%.4f ms/step)\n", cpuTotalMs, cpuPerStepMs);
    std::printf("transfer volume: H2D=%.3f MB, D2H=%.3f KB, sync calls=%u\n", h2dBytes/1e6, d2hBytes/1e3, syncCount);
    std::printf("peak net device allocation (approx, cudaMemGetInfo delta): %.3f MB\n", peakResidentBytes/1e6);
    std::printf("correctness: max abs centroid error GPU vs CPU reference across %zu readbacks: %.6g\n", gpuCentroids.size()/3, maxAbsErr);
    std::printf("GPU/CPU per-step speedup (warm): %.2fx\n", cpuPerStepMs / warmMedianMs);

    cudaFree(d_pts); cudaFree(d_transformed); cudaFree(d_filtered); cudaFree(d_delta); cudaFree(d_centroid);
    cudaStreamDestroy(s);
    return (maxAbsErr < 1e-3) ? 0 : 1; // tolerance on MEAN centroid coords (O(1-10) magnitude), accounting for float32 summation-order variance (GPU tree+atomic reduction vs CPU sequential sum)
}
