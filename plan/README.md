# Multi-Core NPU Extension Plan

Branch: `feat/multi_core_npu`
Updated: 2026-06-23

## Scope

The current working baseline is one `npu_top` controlled by PicoRV32. This plan
extends that baseline to multiple NPU cores while keeping PicoRV32 as the
reference/control CPU.

ZCU102 is treated as the carrier FPGA platform only. Board bring-up and board
validation are out of scope for this plan. The plan focuses on RTL structure,
resource-aware implementation choices, memory layout, and generated firmware.

## Non-Goals

- Do not replace PicoRV32 as the reference/control CPU.
- Do not require board validation as a planning checkpoint.
- Do not change `npu_top` internals for the first multi-core version.

## Document Index

| Document | Contents |
|----------|----------|
| [architecture.md](architecture.md) | PicoRV32 plus multi-NPU architecture, partitioning, synchronization |
| [rtl_changes.md](rtl_changes.md) | RTL modules to add or adapt, with resource notes |
| [firmware_changes.md](firmware_changes.md) | Generated PicoRV32 firmware scheduling plan |
| [dram_layout.md](dram_layout.md) | Shared activation/result/static buffer layout |
| [implementation_order.md](implementation_order.md) | Step-by-step implementation and regression checkpoints |
| [test_plan.md](test_plan.md) | Multi-core RTL, firmware, model, and resource verification plan |

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Reference CPU | PicoRV32 | Existing single-NPU flow already runs with PicoRV32 |
| Platform | ZCU102 carrier | Resource target only; board runtime is outside this plan |
| Core architecture | Replicate `npu_top` | Keeps each NPU core as a known-good black box |
| First target | `NUM_CORES=2` | Smallest useful multi-core target, lower debug risk |
| Later target | `NUM_CORES=4` | Only after resource/timing and 2-core firmware are stable |
| Partition axis | Output-channel `N` tiles | Independent writes, no cross-core reduction |
| A workspace | Shared `A_WORK` | A depends on M/K only, not N; pack once per M tile |
| Result workspace | Per-core `R_WORK[i]` | Each NPU writes its own raw INT32 results |
| Weight layout first pass | Existing per-N-tile layout | Avoids changing static asset generation at the same time |
| Scheduling first pass | One N tile per core per round | Conservative and matches current weight stride |
| Optimized scheduling | Multi-N-tile range per core | Later only after contiguous weight repack is defined |
| Completion sync | PicoRV32 polls STATUS | Simpler and does not depend on IRQ behavior |

## Implementation Status (Updated 2026-06-23)

| Phase | Status | Key Evidence |
|-------|--------|-------------|
| 0: Plan Lock | DONE | PicoRV32+ZCU102, shared A_WORK, per-core R_WORK |
| 1: RTL Infrastructure | DONE | 4 modules lint+elab, committed e5f6955 |
| 2: Firmware Generator | DONE | --num-cores support, mc scheduler, committed 153c4fb+3d98152 |
| 3: Simulation Tests | IN PROGRESS | 5 unit tests PASS; hardware signal confirms both cores busy simultaneously (cyc=310K); full VGG blocked by 42x simulation slowdown |
| 4: ZCU102 Carrier Top | NOT STARTED | |
| 5: Optimization | NOT STARTED | |

**Open issues**: 42x simulation slowdown, zero K-split test coverage, no fair
1-core vs 2-core baseline, ZCU102 resources unknown. See
[implementation_order.md](implementation_order.md) for complete list.

## Workload Summary

For each Conv2D layer, the NPU executes GEMM:

```text
C[M, N] = A[M, K] x W[K, N] + bias[N]
```

When partitioning by `N`, all cores share the same packed `A[M,K]` for a given
spatial tile. Each core reads a different `W[K,N_tile]`, writes a different
output-channel slice, and uses different bias/Q24 multiplier entries.

## Resource Policy

- Keep `FP16_ENABLE=0` for the first multi-core implementation.
- Keep derived performance counter logic disabled unless explicitly needed.
- Start with 2 cores; do not assume 4 cores until synthesis resource/timing data exists.
- Ensure `pingpong_buf` infers BRAM or banked memory before scaling core count.
- Treat per-core `PPB_DEPTH` as a major BRAM knob.
- Keep PicoRV32 firmware and data memory sizing explicit; current closed-loop uses `MEM_WORDS=8192`.

## Expected Effort Split

| Area | Complexity | Notes |
|------|------------|-------|
| RTL wrapper and interconnect | Medium | Mostly replication, but bridge and shared memory routing need care |
| Firmware generator | High | Multi-core launch, polling, postprocess, and edge handling live here |
| Memory layout | Medium | Shared A buffer, per-core R buffers, moved static/descriptor regions |
| Resource cleanup | Medium | Buffer inference and timing become more important as cores scale |
