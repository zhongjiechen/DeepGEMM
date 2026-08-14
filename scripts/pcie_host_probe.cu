// Measure the PCIe-only transports that need no privileged config change:
// GPU <-> pinned host memory, which is how a P2P-less PCIe machine must stage data.
// Build: nvcc -O3 -arch=sm_100 -o pcie_host_probe scripts/pcie_host_probe.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void read_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n; i += stride) {
        uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xdeadbeef) dst[0] = acc;
}

__global__ void write_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        dst[i] = src[i];
}

// System-scope atomic to host memory -- the barrier primitive a PCIe port would rely on.
__global__ void sys_atomic_kernel(int* target, int iters) {
    for (int i = 0; i < iters; ++i)
        atomicAdd_system(target, 1);
}

static double bench(bool is_read, const uint4* src, uint4* dst, size_t n) {
    for (int i = 0; i < 3; ++i)
        is_read ? read_kernel<<<1024, 256>>>(src, dst, n) : write_kernel<<<1024, 256>>>(src, dst, n);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
    CHK(cudaEventRecord(a));
    const int reps = 10;
    for (int i = 0; i < reps; ++i)
        is_read ? read_kernel<<<1024, 256>>>(src, dst, n) : write_kernel<<<1024, 256>>>(src, dst, n);
    CHK(cudaEventRecord(b));
    CHK(cudaEventSynchronize(b));
    float ms = 0; CHK(cudaEventElapsedTime(&ms, a, b));
    CHK(cudaEventDestroy(a)); CHK(cudaEventDestroy(b));
    return (double)n * sizeof(uint4) * reps / (ms / 1e3) / 1e9;
}

int main() {
    int num_gpus = 0;
    CHK(cudaGetDeviceCount(&num_gpus));
    const size_t kBytes = 256ull << 20;
    const size_t n = kBytes / sizeof(uint4);

    // Pinned host buffer, mapped into every GPU's address space.
    uint4* host_buf = nullptr;
    CHK(cudaHostAlloc(&host_buf, kBytes, cudaHostAllocMapped | cudaHostAllocPortable));

    printf("=== GPU <-> pinned host memory over PCIe (GB/s) ===\n");
    printf("%-8s %-16s %-16s\n", "gpu", "read from host", "write to host");
    for (int g = 0; g < num_gpus; ++g) {
        CHK(cudaSetDevice(g));
        uint4* local = nullptr;
        CHK(cudaMalloc(&local, kBytes));
        CHK(cudaMemset(local, 1, kBytes));
        uint4* dev_host_ptr = nullptr;
        CHK(cudaHostGetDevicePointer(&dev_host_ptr, host_buf, 0));
        double rd = bench(true, dev_host_ptr, local, n);
        double wr = bench(false, local, dev_host_ptr, n);
        printf("%-8d %-16.1f %-16.1f\n", g, rd, wr);
        CHK(cudaFree(local));
    }

    printf("\n=== System-scope atomicAdd to pinned host memory ===\n");
    int* host_counter = nullptr;
    CHK(cudaHostAlloc(&host_counter, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
    *host_counter = 0;
    for (int g = 0; g < num_gpus; ++g) {
        CHK(cudaSetDevice(g));
        int* dev_ptr = nullptr;
        CHK(cudaHostGetDevicePointer(&dev_ptr, host_counter, 0));
        const int iters = 5000;
        sys_atomic_kernel<<<1, 1>>>(dev_ptr, 100);
        CHK(cudaDeviceSynchronize());
        cudaEvent_t a, b; CHK(cudaEventCreate(&a)); CHK(cudaEventCreate(&b));
        CHK(cudaEventRecord(a));
        sys_atomic_kernel<<<1, 1>>>(dev_ptr, iters);
        CHK(cudaEventRecord(b));
        cudaError_t e = cudaEventSynchronize(b);
        if (e != cudaSuccess) { printf("gpu%d: FAILED %s\n", g, cudaGetErrorString(e)); continue; }
        float ms = 0; CHK(cudaEventElapsedTime(&ms, a, b));
        CHK(cudaDeviceSynchronize());
        printf("gpu%d: %.3f us/atomic, counter=%d (expected %d) %s\n",
               g, ms * 1e3 / iters, *host_counter, (g + 1) * (iters + 100),
               *host_counter == (g + 1) * (iters + 100) ? "OK" : "*** MISMATCH ***");
        CHK(cudaEventDestroy(a)); CHK(cudaEventDestroy(b));
    }

    // Cross-GPU visibility: GPU0 writes a flag to host memory, GPU1 must observe it.
    printf("\n=== Cross-GPU coherence through host memory (PCIe, no NVLink) ===\n");
    if (num_gpus >= 2) {
        volatile int* flag = nullptr;
        CHK(cudaHostAlloc((void**)&flag, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
        *flag = 0;
        CHK(cudaSetDevice(0));
        int* dev0 = nullptr;
        CHK(cudaHostGetDevicePointer(&dev0, (void*)flag, 0));
        sys_atomic_kernel<<<1, 1>>>(dev0, 1);
        CHK(cudaDeviceSynchronize());
        CHK(cudaSetDevice(1));
        int* dev1 = nullptr;
        CHK(cudaHostGetDevicePointer(&dev1, (void*)flag, 0));
        sys_atomic_kernel<<<1, 1>>>(dev1, 1);
        CHK(cudaDeviceSynchronize());
        printf("flag after GPU0 and GPU1 each +1: %d %s\n", *flag, *flag == 2 ? "OK" : "*** BROKEN ***");
    }
    return 0;
}
