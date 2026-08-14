// Probe P2P capabilities and read/write asymmetry between GPU pairs.
// Build: nvcc -O3 -arch=sm_100 -o pcie_probe scripts/pcie_probe.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// Pure remote-read kernel: pull from peer memory into local.
__global__ void read_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n; i += stride) {
        uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xdeadbeef) dst[0] = acc;  // never taken, defeats DCE
}

// Pure remote-write kernel: push from local into peer memory.
__global__ void write_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        dst[i] = src[i];
}

// Remote atomic throughput/latency: red.add to peer memory.
__global__ void atomic_kernel(int* remote, int iters) {
    for (int i = 0; i < iters; ++i)
        atomicAdd(remote, 1);
}

// Round-trip latency: store a flag to peer, poll a local flag.
__global__ void rtt_kernel(volatile int* remote_flag, volatile int* local_flag,
                           int rounds, int is_leader, long long* out_cycles) {
    if (threadIdx.x != 0) return;
    long long t0 = clock64();
    for (int r = 1; r <= rounds; ++r) {
        if (is_leader) {
            *remote_flag = r;
            while (*local_flag < r) {}
        } else {
            while (*local_flag < r) {}
            *remote_flag = r;
        }
    }
    *out_cycles = clock64() - t0;
}

static double bench(void (*launch)(const uint4*, uint4*, size_t, cudaStream_t),
                    const uint4* src, uint4* dst, size_t n, cudaStream_t s) {
    for (int i = 0; i < 3; ++i) launch(src, dst, n, s);
    CHK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
    CHK(cudaEventRecord(a, s));
    const int reps = 20;
    for (int i = 0; i < reps; ++i) launch(src, dst, n, s);
    CHK(cudaEventRecord(b, s));
    CHK(cudaEventSynchronize(b));
    float ms = 0; CHK(cudaEventElapsedTime(&ms, a, b));
    CHK(cudaEventDestroy(a)); CHK(cudaEventDestroy(b));
    double bytes = (double)n * sizeof(uint4) * reps;
    return bytes / (ms / 1e3) / 1e9;  // GB/s
}

static void launch_read(const uint4* s, uint4* d, size_t n, cudaStream_t st) {
    read_kernel<<<1024, 256, 0, st>>>(s, d, n);
}
static void launch_write(const uint4* s, uint4* d, size_t n, cudaStream_t st) {
    write_kernel<<<1024, 256, 0, st>>>(s, d, n);
}

int main(int argc, char** argv) {
    int num_gpus = 0;
    CHK(cudaGetDeviceCount(&num_gpus));
    printf("Visible GPUs: %d\n\n", num_gpus);

    printf("=== P2P attributes (src -> dst) ===\n");
    printf("%-10s %-8s %-8s %-14s %-14s\n", "pair", "access", "perfRank", "nativeAtomic", "arrayAccess");
    for (int i = 0; i < num_gpus; ++i) {
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) continue;
            int access = 0, rank = 0, atomic = 0, array = 0;
            CHK(cudaDeviceGetP2PAttribute(&access, cudaDevP2PAttrAccessSupported, i, j));
            CHK(cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, i, j));
            CHK(cudaDeviceGetP2PAttribute(&atomic, cudaDevP2PAttrNativeAtomicSupported, i, j));
            CHK(cudaDeviceGetP2PAttribute(&array, cudaDevP2PAttrCudaArrayAccessSupported, i, j));
            char buf[16]; snprintf(buf, sizeof(buf), "%d->%d", i, j);
            printf("%-10s %-8d %-8d %-14d %-14d\n", buf, access, rank, atomic, array);
        }
    }

    if (num_gpus < 2) return 0;

    // Enable peer access everywhere it is supported.
    for (int i = 0; i < num_gpus; ++i) {
        CHK(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) continue;
            int access = 0;
            CHK(cudaDeviceGetP2PAttribute(&access, cudaDevP2PAttrAccessSupported, i, j));
            if (access) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                    printf("  enablePeerAccess %d->%d failed: %s\n", i, j, cudaGetErrorString(e));
                cudaGetLastError();
            }
        }
    }

    const size_t kBytes = 256ull << 20;  // 256 MiB
    const size_t n = kBytes / sizeof(uint4);

    printf("\n=== Unidirectional P2P bandwidth, GPU0 as the active side (GB/s) ===\n");
    printf("%-8s %-14s %-14s %-14s\n", "peer", "read(pull)", "write(push)", "local-copy");
    for (int j = 1; j < num_gpus; ++j) {
        CHK(cudaSetDevice(0));
        uint4 *local_a = nullptr, *local_b = nullptr;
        CHK(cudaMalloc(&local_a, kBytes));
        CHK(cudaMalloc(&local_b, kBytes));
        CHK(cudaMemset(local_a, 1, kBytes));
        CHK(cudaSetDevice(j));
        uint4* remote = nullptr;
        CHK(cudaMalloc(&remote, kBytes));
        CHK(cudaMemset(remote, 2, kBytes));
        CHK(cudaSetDevice(0));
        cudaStream_t st; CHK(cudaStreamCreate(&st));

        double rd = bench(launch_read, remote, local_a, n, st);    // GPU0 reads peer
        double wr = bench(launch_write, local_a, remote, n, st);   // GPU0 writes peer
        double lc = bench(launch_write, local_a, local_b, n, st);  // local baseline

        printf("0<->%-4d %-14.1f %-14.1f %-14.1f\n", j, rd, wr, lc);
        CHK(cudaStreamDestroy(st));
        CHK(cudaFree(local_a)); CHK(cudaFree(local_b));
        CHK(cudaSetDevice(j)); CHK(cudaFree(remote));
    }

    printf("\n=== Remote atomic (atomicAdd to peer) ===\n");
    for (int j = 1; j < num_gpus; ++j) {
        CHK(cudaSetDevice(j));
        int* remote = nullptr;
        CHK(cudaMalloc(&remote, sizeof(int)));
        CHK(cudaMemset(remote, 0, sizeof(int)));
        CHK(cudaSetDevice(0));
        const int iters = 10000;
        cudaEvent_t a, b; CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
        atomic_kernel<<<1, 1>>>(remote, 100);
        CHK(cudaDeviceSynchronize());
        CHK(cudaEventRecord(a));
        atomic_kernel<<<1, 1>>>(remote, iters);
        CHK(cudaEventRecord(b));
        cudaError_t e = cudaEventSynchronize(b);
        if (e != cudaSuccess) {
            printf("0->%d  atomicAdd FAILED: %s\n", j, cudaGetErrorString(e));
            continue;
        }
        float ms = 0; CHK(cudaEventElapsedTime(&ms, a, b));
        // Verify the value actually landed.
        int host_val = -1;
        CHK(cudaMemcpy(&host_val, remote, sizeof(int), cudaMemcpyDeviceToHost));
        printf("0->%d  %.3f us/atomic, final value %d (expected %d) %s\n",
               j, ms * 1e3 / iters, host_val, iters + 100,
               host_val == iters + 100 ? "OK" : "*** MISMATCH ***");
        CHK(cudaEventDestroy(a)); CHK(cudaEventDestroy(b));
        CHK(cudaSetDevice(j)); CHK(cudaFree(remote));
    }
    return 0;
}
