# Implementation Order

## Phase 0: Lock The Correct Plan

- [x] Treat PicoRV32 as the reference/control CPU.
- [x] Treat ZCU102 as the carrier/resource target only.
- [x] Remove previous board-runtime assumptions from the multi-core plan.
- [x] Use shared `A_WORK` and per-core `R_WORK`.
- [x] Start with conservative one-N-tile-per-core scheduling.

## Phase 1: RTL Infrastructure

### Step 1.1: `npu_mc_top.v`

- [ ] Add `NUM_CORES` parameter.
- [ ] Replicate `npu_top` instances with generate.
- [ ] Use flattened AXI-Lite and AXI master buses.
- [ ] Expose `npu_irq[NUM_CORES-1:0]` for optional status/debug.
- Verify: lint/elaborate `NUM_CORES=1` and `NUM_CORES=2`.

### Step 1.2: `axi_lite_mc_bridge.v`

- [ ] Decode `NPU_BASE + core*0x100 + local_offset`.
- [ ] Reject or safely handle offsets outside `NUM_CORES * 0x100`.
- [ ] Forward only one PicoRV32 transaction to the selected core.
- [ ] Keep the existing single-core `axi_lite_bridge.v` unchanged.
- Verify: unit test writes and reads different registers in core0/core1.

### Step 1.3: Simulation Shared Memory

- [ ] Add `dram_multi_port.v` with one CPU simple port and `NUM_CORES` NPU AXI ports.
- [ ] Use one shared backing store.
- [ ] Serialize simultaneous writes deterministically.
- Verify: CPU writes can be read by all NPU ports; NPU writes are visible to CPU.

### Step 1.4: `soc_mc_top.v`

- [ ] Integrate PicoRV32, SRAM, `dram_multi_port`, `axi_lite_mc_bridge`, and `npu_mc_top`.
- [ ] Build CPU IRQ vector with a single assignment or keep IRQ unused.
- Verify: `NUM_CORES=1` runs existing closed-loop firmware unchanged or with only base-name changes.

## Phase 2: Firmware Generator

### Step 2.1: Layout Constants

- [ ] Move `FEAT_BASE`, `SCORE_BASE`, `MARKER_ADDR`, `DESC_BASE`, and `STATIC_BASE` per `dram_layout.md`.
- [ ] Emit `A_WORK_SHARED`.
- [ ] Emit `R_WORK_BASE[core]` and `R_WORK_STRIDE`.
- [ ] Emit `NUM_CORES` and `NPU_CORE_STRIDE`.

### Step 2.2: Multi-Core Conv Scheduler

- [ ] Pack `A_WORK_SHARED` once per M tile.
- [ ] Launch up to `NUM_CORES` cores per N-tile round.
- [ ] Save each launched core's `n_tile` or `n_base` for postprocess.
- [ ] Poll only launched cores.
- [ ] Postprocess from per-core `R_WORK` into disjoint OFM channels.

### Step 2.3: Error Path

- [ ] Check `STATUS.error` for every launched core.
- [ ] Optionally read `ERR_STATUS` for debug marker/log support.
- [ ] Write fail marker and halt on any core error.

## Phase 3: Multi-Core Simulation Tests

- [ ] 2-core MMIO smoke: start core0 and core1 on tiny independent GEMMs.
- [ ] 2-core shared-A smoke: both cores use the same A buffer and different W/R buffers.
- [ ] 2-core single Conv layer: compare dense OFM against single-core output.
- [ ] 2-core full closed-loop VGG: compare final class and optional feature buffer.
- [ ] Run `NUM_CORES=1` regression through the new infrastructure.

## Phase 4: Resource-Oriented Carrier Top

This phase prepares code for the ZCU102 carrier but does not require board
validation in this plan.

- [ ] Add or define `pico_npu_mc_top.v` as the synthesizable boundary.
- [ ] Keep simulation-only memory models out of this top.
- [ ] Expose clean memory/interconnect ports for board integration.
- [ ] Keep `FP16_ENABLE=0` by default.
- [ ] Confirm source file lists separate simulation and synthesis modules.
- [ ] Inspect synthesis resource estimates when available, especially buffers.

## Phase 5: Optional Optimization

- [ ] Define contiguous multi-N-tile weight stream format.
- [ ] Update generator to repack weights for multi-tile ranges.
- [ ] Change firmware to launch each core on an N range instead of one N tile.
- [ ] Compare against conservative scheduler before accepting speedup numbers.
- [ ] Scale from 2 cores to 4 cores only after resource and timing data justify it.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Repacking A per core | CPU work scales with core count | Use shared `A_WORK` |
| Current weight layout used with multi-N launch | Wrong weights read | Start with one N tile per core; optimize later with new repack |
| Ping-pong buffers infer registers | Area/timing failure as cores scale | Confirm BRAM inference or redesign buffer storage |
| Firmware grows beyond SRAM | Boot/runtime failure | Keep looped helpers; increase `MEM_WORDS` if needed |
| Shared memory model overestimates bandwidth | Unrealistic speedup | Treat simulation speedup as functional only until carrier resource data exists |
