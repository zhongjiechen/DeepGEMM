# AGENTS.md — DeepGEMM `pcie-port`

Working agreements for agents on this branch. Read this before doing anything else.

## Goal

Port DeepGEMM's **Mega MoE** (fused EP dispatch → L1 → SwiGLU → L2 → EP combine mega-kernel)
so it runs on machines that have **PCIe only, no NVLink**.

Target topologies, in priority order:

1. GPUs under the **same PCIe switch** (direct switch-routed P2P).
2. GPUs under **different switches but the same PCIe root complex** (root-complex-routed P2P).

Cross-root-complex / cross-socket is out of scope for now, but the transport layer should
degrade to a host-staged path rather than break.

Test with **4 GPUs**.

## Environment

- **Always use the venv at `~/zj_py/`.** Install with `~/zj_py/bin/pip`, run with `~/zj_py/bin/python`.
  Do not install into the system Python (`/usr/local/lib/python3.12/dist-packages` has an NGC
  torch 2.11 — leave it alone).
- CUDA toolkit: `/usr/local/cuda-13.1` (`nvcc` is not on `PATH`; call it by absolute path).
- Installed: `torch 2.13.0+cu130`. Mega MoE needs torch >= 2.9 for
  `torch.distributed._symmetric_memory`.

**The machine gets rescheduled and comes back with the venv empty, submodules uninitialised and
no build.** That is expected, not a broken checkout. Full recovery (~6 min, mostly the torch
download):

```bash
git submodule update --init --recursive
~/zj_py/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130
ln -sf $PWD/third-party/cutlass/include/cutlass deep_gemm/include
ln -sf $PWD/third-party/cutlass/include/cute deep_gemm/include
CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH ~/zj_py/bin/python setup.py build
ln -sf ../build/lib.linux-x86_64-cpython-312/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so deep_gemm/
```

## Commit cadence

**Commit and push at least every 30 minutes.** The machine can be rescheduled at any time and
unpushed work is lost. Prefer many small commits over one big one; a work-in-progress commit
that does not build is still better than losing the work — mark it `wip:` in the subject.

```bash
git add -A && git commit -m "wip: <what changed>" && git push origin pcie-port
```

## This machine's hardware (measured, not assumed)

8x B200, driver 580.95.05, 2x Intel Xeon Platinum 8570, 4 NUMA nodes (SNC2).

**NVLink is present and fully connected** (18 links/GPU via NVSwitch, `NV18` between every
pair). That is the thing we are porting *away* from, so tests must avoid it.

PCIe topology — **every GPU sits alone under its own switch and its own root complex**:

| GPU | BDF | root complex | switch | NUMA |
|-----|-----|--------------|--------|------|
| 0 | `1a:00.0` | `pci0000:15` | `16:00.0` | 0 |
| 1 | `3b:00.0` | `pci0000:37` | `38:00.0` | 0 |
| 2 | `4c:00.0` | `pci0000:48` | `49:00.0` | 1 |
| 3 | `5d:00.0` | `pci0000:59` | `5a:00.0` | 1 |
| 4 | `9b:00.0` | `pci0000:97` | `98:00.0` | 2 |
| 5 | `bb:00.0` | `pci0000:b7` | `b8:00.0` | 2 |
| 6 | `ca:00.0` | `pci0000:c7` | `c8:00.0` | 3 |
| 7 | `dc:00.0` | `pci0000:d7` | `d8:00.0` | 3 |

Each switch holds exactly one GPU + one Mellanox IB NIC + one NVMe. So **neither target
topology (shared switch, shared root complex) exists natively on this box** — GPU pairs are
always cross-root-complex, and pairs (0,1) (2,3) (4,5) (6,7) merely share a NUMA node.

Note: this contradicts the initial guess that GPU0-3 shared a root complex. They do not.

## Measured link performance

Baseline, `scripts/pcie_probe.cu` and `scripts/pcie_host_probe.cu` (256 MiB, 4 GPUs):

| transport | read | write | atomic latency |
|-----------|------|-------|----------------|
| NVLink P2P (what we must avoid) | 745 GB/s | 699 GB/s | 0.008 µs |
| GPU ↔ pinned host memory (PCIe Gen5 x16) | 51 GB/s | 52 GB/s | 0.96 µs |

So PCIe is **~14x lower bandwidth** and **~120x higher atomic latency**. Any design that
depends on cheap fine-grained remote atomics or on remote *reads* will not survive the port.

`atomicAdd_system` to pinned host memory works correctly and is cross-GPU coherent (verified).

## Forcing PCIe during tests

CUDA gives no per-process knob to prefer PCIe over NVLink — if peer access is enabled between
two GPUs and NVLink exists, the driver uses NVLink. Options:

- **Host-staged transport** (`DG_PCIE_TRANSPORT=host`): put the symmetric buffer in pinned host
  memory. Provably never touches NVLink, needs no privileged change, and is a legitimate
  production backend for PCIe machines where P2P does not work across root complexes.
  This is the default way to test on this box.
- **`nvidia-smi nvlink -sLWidth 0`**: really disables NVLink, giving true 1-hop PCIe P2P.
  Machine-wide and possibly not reversible without a reboot, and we have no root here.
  **Do not run this without explicit user approval.**

We do not have `sudo` or `systemctl` in this environment, so driver reload and fabric-manager
tricks are unavailable.

## Design rules for the port

Derived from the numbers above. Violating these is what makes a PCIe port slow.

1. **Never read remote memory in the steady state.** PCIe reads are non-posted round trips.
   Convert every pull to a push — the side that owns the data writes it to the consumer.
   (Upstream dispatch pulls tokens with `tma_load_1d` from a peer pointer; that must become a
   sender-side push.)
2. **No remote atomics on the critical path.** Replace `red_add_rel_sys` / `atomic_add_sys` to
   peers with per-rank slot stores plus *local* polling: rank `i` stores a monotonic sequence
   number into peer's `signal[i]`, and every rank polls only its own local array.
3. **Coalesce remote writes to >= 256 B.** A 4 B or 16 B PCIe write carries ~24 B of TLP header;
   scattered small stores waste most of the link.
4. **Order with `__threadfence_system()`, then flag.** PCIe preserves ordering for posted writes
   along a single path, so data → fence → flag to the *same* peer is safe. Do not rely on a
   flag written to a different peer than the data.
5. **Assume comm dominates.** On NVLink, communication hides under compute; at 51 GB/s the
   reverse is true. Chunk finer and start sending earlier so compute can begin on partial data.

## Where the comm lives

- `deep_gemm/include/deep_gemm/comm/barrier.cuh` — grid sync + cross-rank barrier.
- `deep_gemm/include/deep_gemm/layout/sym_buffer.cuh` — peer pointer mapping (`SymBuffer::map`).
- `deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh` — dispatch pull, combine push.
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` — same, FP8/FP4 variant.
- `deep_gemm/mega/__init__.py` — `SymmBuffer`, allocation and rendezvous.
- `csrc/apis/mega.hpp` — host-side entry points.

## The complete remote-access surface

`grep -n sym_buffer.map` over the impls and `barrier.cuh` is the authoritative list — it is the
*only* way the kernel reaches another rank. 7 sites in BF16, 8 in FP8/FP4:

| site (BF16 / FP8) | region touched | direction |
|---|---|---|
| `barrier.cuh:68` | workspace `nvl_barrier_signal` | remote atomic |
| `:336` / `:376` | workspace `src_token_topk_idx` | 4 B store |
| `:352` / `:392` | workspace `expert_recv_count` | 8 B store |
| `:356` / `:396` | workspace `expert_recv_count_sum` | remote atomic |
| `:494` / `:533` | `input_token_buffer` | **TMA read (the pull)** |
| — / `:562` | `input_sf_buffer` | read |
| `:523` / `:581` | `input_topk_weights_buffer` | 4 B read |
| `:1120` / `:1299` | `combine_token_buffer` | 16 B store |

Everything else is local-only and must stay in HBM: `l1_*`, `l2_*`, `shared_*`,
`input_topk_idx_buffer`, `expert_send_count`, grid-sync and task counters, the L1/L2 ring
full/empty counts, and `token_src_metadata`.

This split matters because the host-staged test transport would otherwise put the GEMM
activation ring buffers in host DRAM, which would both destroy performance and misrepresent
what PCIe costs. Only the 8 regions above belong in the peer-visible allocation.

## Why host-staging cannot measure performance on its own

A receive buffer is *both* written remotely and read locally by the GEMM. Host-staging forces
one location for both, so the local read gets charged PCIe latency it would never pay on a real
PCIe machine (where the receive buffer sits in the receiver's own HBM and only the sender's
write crosses the link).

Concretely: `input_token_buffer` is remotely read by the dispatch pull *and* TMA-read locally by
the shared-expert L1 GEMM. After the pull→push conversion the problem simply moves — the
receiver's `l1_token_buffer` becomes the remotely-written region while still being the GEMM's
main TMA input at ~1400 GB/s. Putting that in host DRAM at 51 GB/s would dominate the whole
kernel and tell us nothing about PCIe.

**Consequence:** host-staging is a *correctness* vehicle, not a performance vehicle, unless the
design gives remote traffic its own landing buffer.

### The landing-buffer design (which we want anyway)

Senders push into a small contiguous **landing buffer**, and the receiver locally copies and
reformats from the landing buffer into its HBM ring. This is the right shape for PCIe
regardless of emulation: it turns scattered small remote writes into coalesced large ones
(rule 3), and it decouples wire format from GEMM layout.

It also makes host-staging measurable: only the landing buffer and the small control regions go
in host memory. The emulated number is then **pessimistic by exactly one PCIe hop on the
receive side** (landing buffer read), because a real P2P machine would land in HBM. That is a
bounded, explainable error we can subtract, unlike the unbounded error above.

## Correctness is transport-independent — validate on NVLink first

Every protocol change below (slot barrier, count all-gather, push dispatch) is correct or
incorrect regardless of which link carries the bytes. So each one is developed and verified
against the **NVLink baseline** first, where iteration is fast and the reference numbers exist.
Host-staging and PCIe measurement come *after* the protocol is PCIe-shaped, not before.


## Progress

| step | status | evidence |
|---|---|---|
| Per-rank slot barrier (replaces remote-atomic barrier) | **validated** | `EP 0/4 \| 2619 TFLOPS \| 363 us` — exact parity with baseline, `torch.equal` checks pass |
| Count all-gather (drop `expert_recv_count_sum` atomic) | in progress | |
| Dispatch pull → push | not started | |
| Landing buffer | not started | |
| Host-staged transport | not started | |

The test asserts `torch.equal(fused_y, baseline_y)` and `torch.equal(fused_stats, baseline_stats)`,
so a passing run is a real correctness check, not just a smoke test.

## Baseline (NVLink, what we are measured against)

4 GPUs (0-3), `--num-experts 32 --num-max-tokens-per-rank 1024`, `mma_type=fp8xfp4`:

```
EP 0/4 | 2619 TFLOPS | overlap: 2739 TFLOPS, HBM 1430 GB/s, NVL 382 GB/s | 363 us, reduction 15.8 us
```

Reproduce with:

```bash
PYTHONPATH=/home/zhongjiechen/DeepGEMM CUDA_HOME=/usr/local/cuda-13.1 \
  PATH=/usr/local/cuda-13.1/bin:$PATH CUDA_VISIBLE_DEVICES=0,1,2,3 \
  ~/zj_py/bin/python tests/test_mega_moe.py --num-processes 4 \
  --num-experts 32 --num-max-tokens-per-rank 1024
```

Note the kernel sustains 382 GB/s of NVLink traffic. PCIe tops out at ~51 GB/s, so comm time
alone rises by roughly 7.5x and stops hiding under compute.

## Build

```bash
git submodule update --init --recursive
ln -sf $PWD/third-party/cutlass/include/cutlass deep_gemm/include
ln -sf $PWD/third-party/cutlass/include/cute deep_gemm/include
CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH ~/zj_py/bin/python setup.py build
ln -sf ../build/lib.linux-x86_64-cpython-312/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so deep_gemm/
```

## Testing

```bash
~/zj_py/bin/python -m torch.distributed.run --nproc_per_node=4 tests/test_mega_moe.py
```
