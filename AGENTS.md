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

Test with **8 GPUs**.

## What we are actually competing against

**The baseline is not the NVLink number.** It is *this same PCIe-only machine running the
unfused path* — DeepEP dispatch/combine plus separate GEMM kernels, which is what
`tests/test_mega_moe.py` already benchmarks as `legacy` (currently inactive because `deep_ep`
is not installed, hence the `0.00x legacy+shared` in every run).

Both sides move their bytes over the same 51 GB/s link, so the PCIe bandwidth cost is *not* a
regression — the baseline pays it too. Comparing a PCIe run against the 363 µs NVLink number is
meaningless and led an earlier version of this document astray.

What fusion actually buys on PCIe: the unfused path must finish all of dispatch before any GEMM
starts, so its comm and compute **add**. The mega-kernel overlaps them, so it pays
`max(comm, compute)`. With comm dominating that saves the whole compute time — about **12%**,
and it is capped there.

### The byte count decides the outcome, not the fusion

12% is small enough that it is swamped by how many bytes each side puts on the wire, and here
Mega MoE is currently at a **disadvantage**.

Mega MoE sends per **(token, expert)**. `get_num_max_pool_tokens()` reserves
`num_max_recv_tokens * min(num_topk, num_experts_per_rank)` slots, one per (token, local
expert), and the dispatch pull fetches a full payload for each. A token selecting three experts
on one destination rank crosses the link three times.

DeepEP sends per **(token, destination rank)**. Its dispatch returns `num_recv_tokens` and
`num_expanded_tokens` as *separate* quantities, and `do_expand` is documented as "one slot per
expert per token" — a **layout**, applied after the transfer. If the wire carried per-expert
copies the two counts would be identical. Combine takes `[num_tokens, hidden]` and has a
"reduce epilogue", pointing to the same design mirrored (weaker evidence than the dispatch
side).

Expected remote copies per token, uniform expert selection:

| config | Mega MoE | DeepEP | ratio |
|---|---|---|---|
| 4 ranks, 32 experts, topk 6 | 4.50 | 2.55 | **1.76x** |
| **8 ranks, 32 experts, topk 6** | 5.25 | 4.09 | **1.28x** |
| 16 ranks, topk 8 | 7.50 | 6.05 | 1.24x |
| 64 ranks, topk 8 | ~7.9 | ~7.5 | ~1.05x |

(`DeepEP = (R-1) * [1 - C(E-E/R, k)/C(E, k)]`, `Mega MoE = k * (R-1)/R`.)

On a link that is bandwidth-saturated, moving 1.28x the bytes *is* 1.28x slower, and no amount
of protocol cleanliness recovers it. Note the penalty is worst at small rank counts — upstream's
per-expert design is entirely reasonable at large EP, where collisions are rare.

**Consequence: dispatch dedup and combine pre-reduction are not optimisations, they are the
difference between winning and losing this comparison.** They rank ahead of the landing buffer
and push machinery. Conveniently the landing buffer *is* the dedup mechanism — see below.

Two things that could move these numbers and are not yet measured:

- DeepEP's own PCIe efficiency. Its intranode path is NVLink-tuned (fine-grained atomics, small
  writes), so it may fall well short of 51 GB/s, which would shift the comparison back in our
  favour. **This must be measured, not assumed.**
- Routing skew. The table assumes uniform expert selection; real models do not route uniformly.

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
| `barrier.cuh:74` | workspace barrier arrival slot | 4 B store (was a remote atomic) |
| `:336` / `:376` | workspace `src_token_topk_idx` | 4 B store |
| `:352` / `:392` | workspace `expert_recv_count` | 8 B store (was + a remote atomic) |
| `:494` / `:533` | `input_token_buffer` | **TMA read (the pull)** |
| — / `:562` | `input_sf_buffer` | read |
| `:523` / `:581` | `input_topk_weights_buffer` | 4 B read |
| `:1120` / `:1299` | `combine_token_buffer` | 16 B store |

**No remote atomics remain** — `ptx::atomic_add_sys` and `ptx::red_add_rel_sys` are still
defined but have no call sites. Design rule 2 is satisfied. What is left is rule 1: the
dispatch pull, which is the next and by far the largest piece of work.

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

## Next step: converting the dispatch pull to a push

This is the largest remaining piece. The choice that drives everything else is **who assigns the
destination pool slot**, and it is not obvious in the direction it first appears.

### The slot assignment, and why it should stay on the receiver

Today the receiver decides: for local expert `e` with per-rank counts `c[0..R-1]`, it walks pool
slot `s` through a round-robin min-peel (`sm100_bf16_mega_moe.cuh` ~line 421) to recover
`(src_rank, token_idx_in_rank)`. A push needs the *inverse* — given "rank `m`'s `k`-th token for
expert `e`", produce `s`:

```
remaining = c; offset = 0; s_base = 0
loop:
    A = #{r : remaining[r] > 0}                  // active ranks
    L = min{remaining[r] : remaining[r] > 0}     // this round's depth
    if k < offset + L:                           // my token lands in this round
        a = rank m's 0-based position among the active ranks
        s = s_base + (k - offset) * A + a
        break
    s_base += L * A; offset += L
    for r: remaining[r] -= min(remaining[r], L)
```

Substituting back into the forward peel recovers `token_idx_in_rank == k`, so the two agree.
`pool_token_idx = pool_block_offset(e) * BLOCK_M + s`.

The trap is assuming the *sender* must run this. It only must if the sender writes the ring
directly, and that costs two things:

- **A count all-gather.** `s` depends on `c[*]`, all ranks' counts for that expert, so every
  sender needs the whole `num_ranks x num_experts` matrix rather than the single row it pushes
  today.
- **An extra cross-rank barrier.** `c[*]` is not known until every rank has finished counting,
  so slot assignment cannot start until a barrier completes. Today slot allocation
  (`atomicAdd_block`, ~line 331) runs *before* the count exchange precisely because it does not
  need global counts. A direct push serialises count -> barrier -> assign -> send.

**A landing buffer removes both.** If senders push into `landing[src_rank]` in whatever order
they like and the receiver places tokens into ring slots, then the inverse mapping runs on the
receiver — which already has `c[*]` locally and already waits for it in
`fetch_expert_recv_count()`. No all-gather, no extra barrier, and the sender never needs to know
anything about the destination's layout.

### Ring flow control cannot survive a naive push either

The receiver's ring slot is only reusable once its L1 GEMM consumer has drained it, which the
puller checks today via a **local** read of `l1_empty_count`. A sender writing the ring directly
would have to check that counter on the *receiver* — a remote read, which rule 1 forbids and
which is a non-posted PCIe round trip on the critical path.

The landing buffer answers this too:

- Sender pushes into `landing[src_rank]` on the receiver, a region it owns exclusively, so no
  cross-rank arbitration and no remote read.
- Receiver copies `landing -> ring` locally, honouring its own `l1_empty_count` as it does now,
  and applying the inverse mapping above.
- Receiver pushes a **credit** (a monotonic count of tokens drained) into the sender's local
  `credit[dst_rank]` slot. The sender polls only its own array.

Both sides then read only local memory, and both directions are posted writes.

**The cost is one extra local copy.** The pull does remote -> smem -> ring; landing does
remote -> landing, then landing -> smem -> ring, so each token pays an extra HBM write and read.
At ~51 GB/s on the wire against ~1400 GB/s of HBM that is not the binding constraint, but it is
the reason to keep direct-to-ring in mind as a later optimisation if profiling says otherwise.

Sizing: at ~51 GB/s a 14 KiB token occupies the wire for ~274 ns, so covering a credit round
trip of a few µs needs on the order of tens of tokens in flight per (src, dst) pair. 64 tokens
per source is ~900 KiB per source — cheap. Start there and tune against measurement.

**Suggested staging** — each step validated on NVLink before the next:

1. Landing buffer + receiver-side placement, sender still writing metadata as it does now.
   Correctness of the inverse mapping is checked by the existing `torch.equal` assertions.
2. Credit flow control, replacing whatever interim synchronisation step 1 uses.
3. Flip the token payload to a real push and delete the pull.

## Correctness is transport-independent — validate on NVLink first

Every protocol change below (slot barrier, count all-gather, push dispatch) is correct or
incorrect regardless of which link carries the bytes. So each one is developed and verified
against the **NVLink baseline** first, where iteration is fast and the reference numbers exist.
Host-staging and PCIe measurement come *after* the protocol is PCIe-shaped, not before.


## Progress

| step | status | evidence |
|---|---|---|
| Per-rank slot barrier (replaces remote-atomic barrier) | **validated** | `EP 0/4 \| 2619 TFLOPS \| 363 us` — exact parity with baseline, `torch.equal` checks pass |
| Count all-gather (drop `expert_recv_count_sum` atomic) | **validated** | `2606 TFLOPS \| 365 us`, −0.5% vs baseline; no remote atomics left in the kernel |
| Dispatch pull → push | not started | |
| Landing buffer | not started | |
| Host-staged transport | not started | |

The test asserts `torch.equal(fused_y, baseline_y)` and `torch.equal(fused_stats, baseline_stats)`,
so a passing run is a real correctness check, not just a smoke test.

**`--mma-type` defaults to `fp8xfp4`, so a default run does not touch
`sm100_bf16_mega_moe.cuh` at all.** Both impls carry the same protocol code, so every change
must be run twice:

```bash
... tests/test_mega_moe.py --num-processes 8 --num-experts 32 --num-max-tokens-per-rank 1024
... tests/test_mega_moe.py --num-processes 8 --num-experts 32 --num-max-tokens-per-rank 1024 --mma-type bf16xbf16
```

bf16xbf16 reference: `EP 0/4 | 1171 TFLOPS | 812 us`.

## Workspace size silently sets activation-buffer alignment

Every data buffer is chained off the previous one's `get_end_ptr()` and **no `Buffer` re-aligns
its own base**, so the first one starts wherever `Workspace::get_num_bytes()` ends. That end was
only 16 B-aligned, which means *adding or removing a single counter in the workspace shifts the
L1/L2/combine token buffers* — the ~1400 GB/s TMA traffic.

This is not theoretical. Removing the 64 B `expert_recv_count_sum` region cost 2.3% end-to-end
with no protocol change involved, and it took four measured variants to find that the layout,
not the protocol, was responsible:

| variant | EP 0/4 TFLOPS |
|---|---|
| baseline (before any change) | 2620, 2619 |
| count all-gather + 256 B workspace end alignment | **2606, 2606** |
| count all-gather + 64 B pad restoring the old offsets | 2610, 2602, 2600 |
| count all-gather + each ring counter array on its own 128 B line | 2583, 2581 |
| count all-gather, 16 B end alignment (naive) | 2557, 2560, 2565 |

Note the third row: per-array line padding, the "obvious" false-sharing fix, made things *worse*.
The ring counters were never the problem.

`get_num_bytes()` now aligns to 256 B (`kNumBufferAlignBytes`). Per-rank buffer sizes are large
multiples of 128 B, so one aligned start keeps everything downstream aligned. Two runs produced
byte-identical throughput, where the unaligned variants jittered.

**Rule: when you change the workspace layout, re-measure.** If throughput moves by ~1-2% and the
protocol change does not explain it, suspect this before anything else.

## Baseline (NVLink, what we are measured against)

**8 GPUs (0-7)**, `--num-experts 32 --num-max-tokens-per-rank 1024`, `mma_type=fp8xfp4`
(4 experts per rank, topk 6):

```
EP 0/8 | 2522 TFLOPS | overlap: 2632 TFLOPS, HBM 1005 GB/s, NVL 368 GB/s | 379 us, reduction 15.8 us
```

Reproducible to 1 TFLOPS across runs. Reproduce with:

```bash
PYTHONPATH=/home/zhongjiechen/DeepGEMM CUDA_HOME=/usr/local/cuda-13.1 \
  PATH=/usr/local/cuda-13.1/bin:$PATH CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ~/zj_py/bin/python tests/test_mega_moe.py --num-processes 8 \
  --num-experts 32 --num-max-tokens-per-rank 1024
```

Earlier 4-GPU reference, for the commits that were measured against it:
`EP 0/4 | 2619 TFLOPS | 363 us | NVL 382 GB/s` (and 2606 after the count all-gather).

The kernel sustains 368 GB/s of cross-rank traffic over 379 µs — about **140 MB per rank per
call**. At 51 GB/s that is ~2.7 ms of wire time, so on PCIe this kernel is comm-bound by roughly
7x and compute hides entirely under communication rather than the reverse.

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
