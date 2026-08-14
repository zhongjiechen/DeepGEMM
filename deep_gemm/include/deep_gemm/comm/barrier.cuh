#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::comm {

// 60s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        const auto start_clock = clock64();
        do {
            new_value = ptx::ld_acq(count_ptr);
            if (clock64() - start_clock >= kNumTimeoutCycles) {
                printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
                       sm_idx, thread_idx, kGridSyncIndex, old_value, new_value, old_value ^ kFinishSumTag);
                DG_DEVICE_ASSERT(false and "Grid sync timeout");
            }
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    DG_STATIC_ASSERT(kNumThreads >= 32, "Need at least one warp to poll arrivals");

    // Grid sync before cross-rank signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // Cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;

        // The target alternates between 1 and 0 on successive uses of the same phase, so a
        // stale value from two barriers ago can never be mistaken for an arrival.
        const int target = signal_sign ? 0 : 1;

        // Announce arrival in every peer's slot for *our* rank. Each (peer, source) slot has
        // exactly one writer, so a plain release store replaces the remote atomic the previous
        // implementation needed when all ranks shared one counter.
        if (thread_idx < kNumRanks) {
            ptx::st_rel_sys(
                sym_buffer.map(workspace.get_nvl_barrier_signal_ptr(signal_phase, sym_buffer.rank_idx),
                               thread_idx),
                target);
        }
        sync_scope();

        // Advance the (rank-local) phase counter
        if (thread_idx == 0)
            ptx::red_add(counter_ptr, 1);

        // Wait for arrivals. Every load here is *local* -- we only ever read our own slot
        // array -- which is what keeps this off the PCIe critical path.
        if (thread_idx < 32) {
            const auto lane_idx = thread_idx;
            const auto start_clock = clock64();
            while (true) {
                bool arrived = true;
                #pragma unroll
                for (uint32_t base = 0; base < kNumRanks; base += 32) {
                    const uint32_t src_rank_idx = base + lane_idx;
                    if (src_rank_idx < kNumRanks)
                        arrived &= ptx::ld_acq_sys(
                            workspace.get_nvl_barrier_signal_ptr(signal_phase, src_rank_idx)) == target;
                }
                if (__all_sync(0xffffffff, arrived))
                    break;
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    printf("DeepGEMM cross-rank barrier timeout: rank=%d, lane=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                           sym_buffer.rank_idx, lane_idx, target, signal_phase, signal_sign, kTag);
                    DG_DEVICE_ASSERT(false and "Cross-rank barrier timeout");
                }
            }
        }
    }

    // Grid sync after cross-rank completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
