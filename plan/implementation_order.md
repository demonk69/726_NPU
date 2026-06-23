# Implementation Order

## Phase 1: RTL Infrastructure (est. 1-2 days)

### Step 1.1: `rtl/top/npu_mc_top.v`
- [ ] Create `npu_mc_top.v` with parameter `NUM_CORES`
- [ ] `generate for` loop instantiating `NUM_CORES` `npu_top` instances
- [ ] Connect all replicated ports (AXI4-Lite × N, AXI4 Master × N, IRQ × N)
- **Verify**: Elaboration with Verilator (lint-only, no testbench yet)

### Step 1.2: `rtl/soc/dram_multi_port.v`
- [ ] Create `dram_multi_port.v` with parameter `NUM_PORTS`
- [ ] Instantiate backing store array
- [ ] Implement round-robin write arbiter
- [ ] Concurrent read ports
- **Verify**: Standalone testbench with CPU read/write + NPU DMA read/write

### Step 1.3: Modify `rtl/soc/axi_lite_bridge.v`
- [ ] Add `NUM_CORES` and `NPU_CORE_STRIDE` parameters
- [ ] Add address decode: `core_sel = addr[11:8]`
- [ ] Replicate AXI4-Lite output ports to `NUM_CORES` channels
- [ ] Only drive the selected core's channel
- **Verify**: Existing single-core tests still pass (NUM_CORES=1)

### Step 1.4: `rtl/soc/soc_mc_top.v`
- [ ] Create `soc_mc_top.v` with parameter `NUM_CORES`
- [ ] Integrate: PicoRV32 + SRAM + multi-port DRAM + multi-core AXI-Lite bridge + `npu_mc_top`
- [ ] IRQ mapping: `cpu_irq[7+i] <= npu_irq[i]`
- **Verify**: Elaboration with Verilator (lint-only)

## Phase 2: Single-Core Regression Gate (est. 0.5 day)

Before proceeding to multi-core firmware, verify that the new infrastructure
works with the existing single-core VGG flow:

- [ ] Create `tb/tb_soc_mc_smoke.v` — instantiates `soc_mc_top` with `NUM_CORES=1`
- [ ] Run existing VGG E2E test through the new SoC (firmware only uses Core 0)
- [ ] Run existing VGG closed-loop test through the new SoC
- **Gate**: Both tests pass identically to baseline `soc_top` version

## Phase 3: Multi-Core Firmware (est. 2-3 days)

### Step 3.1: Python generator changes
- [ ] Add `NUM_CORES` config to `gen_vgg_closed_loop.py`
- [ ] Update DRAM layout constants (per-core A_WORK, R_WORK)
- [ ] Update `emit_conv_layer_loop()`:
  - [ ] Inner loop: per-core launch (config N_DIM = N_PER_CORE, W_ADDR offset, etc.)
  - [ ] Poll loop: check all cores' STATUS registers
  - [ ] Post-process: per-core result read + scatter
- [ ] Update `emit_pack_a_tile()` for per-core A_WORK buffers
- [ ] Update `emit_wait_npu_done()` for multi-STATUS polling

### Step 3.2: Firmware verification (offline)
- [ ] Generate firmware for NUM_CORES=2, image index 0
- [ ] Inspect generated assembly for correctness (register reuse, branch targets, DRAM address offsets)
- [ ] Compare generated per-core A_WORK contents with single-core A_WORK (should be identical for N ranges)

## Phase 4: End-to-End Multi-Core VGG Test (est. 1-2 days)

### Step 4.1: Testbench
- [ ] Create `tb/tb_soc_mc_vgg_closed_loop.v` or adapt existing testbench
  - Load multi-core firmware into SRAM
  - Load multi-core DRAM image into multi-port DRAM
  - Route per-core AXI4 master ports to corresponding DRAM ports
  - Monitor MARKER_ADDR for PASS/FAIL

### Step 4.2: Run and debug
- [ ] Run test with NUM_CORES=2, image index 0
- [ ] Compare output classification with single-core (must be identical)
- [ ] Verify per-core STATUS polling works (no missed IRQs, no deadlocks)
- [ ] Verify layer barrier synchronization
- [ ] Run full regression (all 10 CIFAR-10 test images)

### Step 4.3: Performance validation
- [ ] Measure cycle count for 2-core vs 1-core
- [ ] Verify speedup ratio (target: ~1.6-1.9x for 2 cores on VGG)
- [ ] Measure per-core utilization via op_counter performance registers

## Phase 5: Scaling to 4 Cores (est. 1 day)

- [ ] Update DRAM layout for 4-core buffers
- [ ] Bump DRAM_WORDS if needed
- [ ] Test with NUM_CORES=4
- [ ] Verify no new issues from wider IRQ vector or address decode

## Phase 6: Documentation and Cleanup (est. 0.5 day)

- [ ] Update `doc/architecture.md` with multi-core diagrams
- [ ] Update `doc/user_manual.md` with multi-core run commands
- [ ] Add `run_vgg_mc_closed_loop.sh` convenience script
- [ ] Ensure all existing single-core tests still pass

---

## Total Estimated Effort

| Phase | Description | Est. Time |
|-------|-------------|-----------|
| 1 | RTL infrastructure | 1-2 days |
| 2 | Single-core regression gate | 0.5 day |
| 3 | Multi-core firmware | 2-3 days |
| 4 | E2E multi-core VGG test | 1-2 days |
| 5 | Scaling to 4 cores | 1 day |
| 6 | Documentation | 0.5 day |
| **Total** | | **6-9 days** |

---

## Risk Items

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Firmware exceeds SRAM (4KB) with multi-core loops | Medium | Compress loops, share common prologue, increase SRAM |
| DRAM multi-port write conflicts cause data corruption | Low | Round-robin arbiter ensures serialized writes |
| N_PER_CORE not divisible by TC (16) for some layers | Low | Pad last tile with zeros (same as single-core edge handling) |
| IRQ coalescing: both cores finish same cycle, CPU misses one | Low | Firmware polls STATUS, not IRQ edge-sensitive; robust to coalescing |
| Weight DMA contention: both cores read same static data | Low | Multi-port DRAM allows concurrent reads |
