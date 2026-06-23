# Implementation Order

## Phase 0: Lock The Correct Plan

- [x] Treat PicoRV32 as the reference/control CPU.
- [x] Treat ZCU102 as the carrier/resource target only.
- [x] Remove previous board-runtime assumptions from the multi-core plan.
- [x] Use shared `A_WORK` and per-core `R_WORK`.
- [x] Start with conservative one-N-tile-per-core scheduling.

## Phase 1: RTL Infrastructure — DONE (commit e5f6955)

### Step 1.1: `npu_mc_top.v`

- [x] Add `NUM_CORES` parameter.
- [x] Replicate `npu_top` instances with generate.
- [x] Use flattened AXI-Lite and AXI master buses.
- [x] Expose `npu_irq[NUM_CORES-1:0]` for optional status/debug.
- Verified: lint/elaborate `NUM_CORES=1` and `NUM_CORES=2` (Icarus + Verilator).

### Step 1.2: `axi_lite_mc_bridge.v`

- [x] Decode `NPU_BASE + core*0x100 + local_offset`.
- [x] Reject or safely handle offsets outside `NUM_CORES * 0x100`.
- [x] Forward only one PicoRV32 transaction to the selected core.
- [x] Keep the existing single-core `axi_lite_bridge.v` unchanged.
- Verified: unit test passes 8 checks (core0/core1 write/read, invalid window, isolation).
- Bugs fixed: bready must be always-1 (BVALID arrives one cycle late); read rdata must be combinational (registered rdata is one cycle too late for PicoRV32 sampling).

### Step 1.3: Simulation Shared Memory

- [x] Add `dram_multi_port.v` with one CPU simple port and `NUM_CORES` NPU AXI ports.
- [x] Use one shared backing store.
- [x] Serialize simultaneous writes deterministically.
- Verified: unit test passes 9 checks (CPU write→NPU read, burst write/read, concurrent reads).
- Known issue: per-port AXI FSMs (wr_active / ar_active state machines replicated per port) cause significant Verilator evaluation overhead. This is a simulation performance concern, not a correctness concern.

### Step 1.4: `soc_mc_top.v`

- [x] Integrate PicoRV32, SRAM, `dram_multi_port`, `axi_lite_mc_bridge`, and `npu_mc_top`.
- [x] Build CPU IRQ vector with a single generate assignment.
- Verified: lint/elaborate passes. PicoRV32 firmware executes correctly through the new infrastructure.

## Phase 2: Firmware Generator — DONE (commits 153c4fb, 3d98152)

### Step 2.1: Layout Constants

- [x] Move `FEAT_BASE`, `SCORE_BASE`, `MARKER_ADDR`, `DESC_BASE`, and `STATIC_BASE` per `dram_layout.md`.
- [x] Emit `A_WORK_SHARED`.
- [x] Emit `R_WORK_BASE[core]` and `R_WORK_STRIDE`.
- [x] Emit `NUM_CORES` and `NPU_CORE_STRIDE`.
- [x] Add `n_tiles` field to conv descriptor (offset 80) for mc firmware.

### Step 2.2: Multi-Core Conv Scheduler

- [x] Pack `A_WORK_SHARED` once per M tile.
- [x] Launch up to `NUM_CORES` cores per N-tile round.
- [x] Save each launched core's global `n_tile` or `n_base` for postprocess.
- [x] Poll only launched cores.
- [x] Postprocess from per-core `R_WORK` into disjoint OFM channels.
- Bug fixed: Q24 multiplier base address used temp register `t6` instead of saved register `s6` (commit 3d98152).
- Bug fixed: multi-core postprocess restarted `n_tile` at 0 for every launch round, so second and later N-tile rounds scattered into the wrong output channels. The generator now derives the round-start global `n_tile` as `next_n_tile - launched_count` before postprocess.

### Step 2.3: Error Path

- [x] Check `STATUS.error` for every launched core.
- [x] Optionally read `ERR_STATUS` for debug marker/log support.
- [x] Write fail marker and halt on any core error.

## Phase 3: Multi-Core Simulation Tests — IN PROGRESS

Detailed test staging and pass criteria are in [test_plan.md](test_plan.md).

- [x] 2-core MMIO smoke: PicoRV32 firmware writes/reads core0 and core1 registers.
- [x] 2-core shared-A smoke: both cores use same A buffer, different W/R buffers, verified against golden.
- [x] Both cores launch simultaneously: hardware signal `busy0=busy1=1` observed at cyc=310K via Verilator heartbeat testbench.
- [x] Short `soc_mc_top NUM_CORES=1` heartbeat baseline: about 222K cycles/sec on the current host.
- [x] Short `soc_mc_top NUM_CORES=2` heartbeat sample: about 89K cycles/sec on the current host with low-output heartbeat, about 2.5x slower than `NUM_CORES=1` in Verilator.
- [x] Full `NUM_CORES=1` regression through `soc_mc_top`: image2 PASS, frog/class 6, 114,014,769 cycles.
- [x] 2-core full closed-loop VGG: image2 PASS, frog/class 6, 151,892,523 cycles.
- [ ] 2-core single Conv layer: compare dense OFM against single-core output (need K=27 scale test, not yet verified).

### Test Coverage Gaps

| Tested | Not Tested |
|--------|------------|
| Core independence (busy signals) | Real data scale layer-level golden comparisons |
| shared A_WORK correctness | Real data scale (K=27 only in trace, not end-to-end verified) |
| Bridge multi-core MMIO | Broader maxpool/avgpool/classifier image/shape coverage |
| Both cores start simultaneously | Edge: final N tile round with fewer than NUM_CORES cores |
| Error isolation (one core fault) | K-split: large K sliced into multiple k_tile segments |
| Short 1-core/2-core heartbeat cps samples | 8x32 shape with multi-core |
| Fixed global `n_tile` postprocess bug | Random/data-driven stress testing |
| | ZCU102 FPGA resource/timing estimation |

## Phase 4: Resource-Oriented Carrier Top — NOT STARTED

This phase prepares code for the ZCU102 carrier but does not require board validation.

- [ ] Add or define `pico_npu_mc_top.v` as the synthesizable boundary.
- [ ] Keep simulation-only memory models out of this top.
- [ ] Expose clean memory/interconnect ports for board integration.
- [ ] Keep `FP16_ENABLE=0` by default.
- [ ] Confirm source file lists separate simulation and synthesis modules.
- [ ] Inspect synthesis resource estimates when available, especially buffers.

## Phase 5: Optional Optimization — NOT STARTED

- [ ] Define contiguous multi-N-tile weight stream format.
- [ ] Update generator to repack weights for multi-tile ranges.
- [ ] Change firmware to launch each core on an N range instead of one N tile.
- [ ] Compare against conservative scheduler before accepting speedup numbers.
- [ ] Scale from 2 cores to 4 cores only after resource and timing data justify it.

## Open Issues

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | 2-core full closed-loop VGG is slower than `NUM_CORES=1` | High | Image2 PASS at 151,892,523 cycles for `NUM_CORES=2` vs 114,014,769 cycles for `NUM_CORES=1`. Current multi-core firmware is dominated by serial CPU scheduling/postprocess. |
| 2 | Broader fair performance numbers are still missing | High | Image2 now has full-cycle `NUM_CORES=1` and `NUM_CORES=2` results. Other images/shapes and per-core NPU performance counters are still needed. |
| 3 | `dram_multi_port` has replicated per-port AXI FSMs | Medium | Adds Verilator evaluation overhead when `NUM_CORES>1`, along with the duplicated `npu_top` instances. Consider simplifying the simulation model only after correctness tests pass. |
| 4 | All tests use minimal data (M=4,N=8,K=4) | Medium | K-split logic, large A_WORK packing, and multi-N-tile edge cases remain weakly tested at real VGG scale. |
| 5 | gen_mc_tests.py uses absolute byte branch offsets | Medium | Branch targets can silently break when firmware grows. Use Asm label-based branches. |
| 6 | OFM clear is CPU-bound (290K cycles per layer) | Medium | PicoRV32 serial zero-fill dominates first-layer startup. Multi-core cannot help. Consider DMA zero-fill or strided store. |
| 7 | ZCU102 resource budget unknown for 2+ cores | Low | Two full npu_top instances with PPB_DEPTH=8192 may not fit ZCU102 BRAM/DSP. Blocked until Phase 4. |

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Repacking A per core | CPU work scales with core count | Use shared `A_WORK` |
| Current weight layout used with multi-N launch | Wrong weights read | Start with one N tile per core; optimize later with new repack |
| Ping-pong buffers infer registers | Area/timing failure as cores scale | Confirm BRAM inference or redesign buffer storage |
| Firmware grows beyond SRAM | Boot/runtime failure | Keep looped helpers; increase `MEM_WORDS` if needed |
| Shared memory model overestimates bandwidth | Unrealistic speedup | Treat simulation speedup as functional only until carrier resource data exists |
| Slow 2-core Verilator simulation slows optimization loops | Functional PASS exists for image2, but profiling/tuning runs are expensive | Use targeted layer tests and per-core counters before broad sweeps |
