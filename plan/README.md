# Multi-Core NPU Extension Plan

Branch: `feat/multi_core_npu`
Created: 2026-06-22

## Overview

Extend the current single-core NPU to a multi-core architecture, targeting 2-4
cores (parameterizable), to accelerate RepOpt VGG-like CIFAR-10 inference.

## Document Index

| Document | Contents |
|----------|----------|
| [architecture.md](architecture.md) | Top-level architecture, data dependency analysis, work partitioning strategy |
| [rtl_changes.md](rtl_changes.md) | All RTL file changes with module-level specifications |
| [firmware_changes.md](firmware_changes.md) | CPU firmware modifications for multi-core scheduling |
| [dram_layout.md](dram_layout.md) | DRAM address map and per-core buffer layout |
| [implementation_order.md](implementation_order.md) | Step-by-step execution order with verification checkpoints |

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Core architecture | Full replication of `npu_top` | Each core is an independent black box; zero changes to `npu_top.v` |
| Core count | Parameter `NUM_CORES` (default 2) | Scalable via generate, start at 2 for validation |
| Work partitioning | By N (output channel) dimension | Zero inter-core data dependency within a layer |
| DRAM interface | Per-core dedicated AXI4 slave port | No memory contention between cores |
| Scheduling | Firmware (CPU) pre-partitions by N, launches all cores | Simple barrier sync between layers |
| MMIO addressing | Stride `0x100` per core | `Core N` at `0x02000000 + N * 0x100` |

## Workload Analysis: VGG on CIFAR-10

9 Conv layers, 1024 total GEMM tiles. Within each layer:

- **M (spatial) and N (channel) tiles are independent** — write to disjoint OFM regions
- **K accumulation is fully internal to the NPU** — firmware never sees partial K results
- Therefore: partition tiles by N across cores; zero intra-layer dependency
- Per-layer barrier: all cores must finish before next layer's IFM is ready

## Effort Distribution

| Area | Effort | Complexity |
|------|--------|------------|
| RTL hardware | ~30% | Low — mostly replication and wiring |
| CPU firmware | ~50% | High — new RISC-V multi-core scheduling logic |
| Python tools | ~20% | Medium — per-core buffer allocation and codegen |
