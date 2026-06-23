# Multi-Core NPU Test Plan

Updated: 2026-06-23

## Scope

This plan verifies the multi-core extension where PicoRV32 remains the
reference/control CPU and ZCU102 is only the carrier/resource target. Board
bring-up is not a required checkpoint here.

The test strategy is staged so each layer of risk is isolated before full VGG
firmware is attempted.

## Test Principles

- Keep the existing single-core flow as the golden baseline.
- `NUM_CORES=1` through the new infrastructure must match single-core behavior.
- Verify shared `A_WORK` explicitly before full model tests.
- Verify each new RTL block independently before `soc_mc_top` tests.
- Do not trust speedup numbers until correctness and resource checks are stable.
- Treat simulation multi-port memory as a functional model, not a bandwidth model.

## Pass Criteria Summary

| Level | Required pass condition |
|-------|-------------------------|
| RTL lint/elab | No new errors; known legacy warnings only |
| Bridge unit tests | Correct core selection, local offset, invalid-window behavior |
| DRAM unit tests | CPU/NPU visibility through one shared backing store |
| NPU wrapper smoke | Both cores can run independent jobs without cross-talk |
| `NUM_CORES=1` SoC | Same final result as existing single-core path |
| 2-core shared-A smoke | Same A buffer feeds both cores; independent R outputs match golden |
| 2-core Conv layer | Dense OFM matches single-core/Python golden |
| 2-core full VGG | Final class and optional feature buffer match exact Python target |
| Resource check | No unexpected register explosion in buffers; timing/resource risk recorded |

## Stage 0: Static RTL Checks

Goal: catch syntax, port, and generate mistakes before writing testbenches.

Commands:

```bash
iverilog -g2012 -s npu_mc_top -o /tmp/npu_mc_top_elab.vvp \
  rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v \
  rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/top/*.v

iverilog -g2012 -s soc_mc_top -o /tmp/soc_mc_top_elab.vvp \
  rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v \
  rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/top/*.v \
  rtl/soc/*.v sim/picorv32.v

verilator --lint-only -Wall -Wno-fatal --top-module soc_mc_top \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSED -Wno-UNDRIVEN \
  -Wno-UNOPTFLAT -Wno-PINMISSING -Wno-CASEINCOMPLETE \
  -Wno-COMBDLY -Wno-INITIALDLY -Wno-LITENDIAN \
  rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v \
  rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/top/*.v \
  rtl/soc/*.v sim/picorv32.v
```

Acceptance:

- Elaboration succeeds.
- Any warnings are classified as pre-existing or filed as new issues.

## Stage 1: Unit Tests For New RTL

### `tb/tb_axi_lite_mc_bridge.v`

Purpose: verify PicoRV32 simple-bus to selected-core AXI-Lite routing.

Cases:

| Case | Stimulus | Expected |
|------|----------|----------|
| Core0 write | Write `NPU_BASE + 0x10` | Only core0 AW/W valid, local addr `0x10` |
| Core1 write | Write `NPU_BASE + 0x100 + 0x10` | Only core1 AW/W valid, local addr `0x10` |
| Core0 read | Read `NPU_BASE + 0x04` | Only core0 AR valid, returned core0 RDATA |
| Core1 read | Read `NPU_BASE + 0x100 + 0x04` | Only core1 AR valid, returned core1 RDATA |
| Invalid window | Read beyond `NUM_CORES*0x100` | Return `32'hDEADBEEF`, no core valid asserted |
| Backpressure | Delay AWREADY/WREADY/RVALID | PicoRV32 `iomem_ready` waits correctly |

Acceptance:

- No selected-core cross-talk.
- No transaction completes before required AXI handshakes.
- Invalid windows do not touch any core.

### `tb/tb_dram_multi_port.v`

Purpose: verify the simulation shared-memory model.

Cases:

| Case | Stimulus | Expected |
|------|----------|----------|
| CPU write, NPU read | CPU writes word, core0 AXI reads same addr | Core0 sees CPU data |
| NPU write, CPU read | Core1 AXI writes word, CPU reads same addr | CPU sees core1 data |
| Two NPU reads | Core0/core1 read different addresses simultaneously | Both receive correct data |
| Disjoint NPU writes | Core0/core1 write different addresses | Both writes visible |
| Same-address write conflict | Core0/core1 write same cycle same addr | Deterministic priority behavior documented |
| Burst read/write | Multi-beat AXI write then read back | Correct data sequence and `RLAST` |

Acceptance:

- One shared backing store is observed by all ports.
- Burst counters and `LAST` behavior match the existing `dram_model` contract.

### `tb/tb_npu_mc_top_smoke.v`

Purpose: verify wrapper-level independence of multiple `npu_top` instances.

Cases:

| Case | Stimulus | Expected |
|------|----------|----------|
| Core0 only | Start only core0 | Core1 status remains idle |
| Core1 only | Start only core1 | Core0 status remains idle |
| Both cores | Start both cores with different W/R buffers | Both complete and write separate results |
| Error isolation | Force bad addr on one core | Other core status/result unaffected |

Acceptance:

- `npu_mc_top` bus slicing is correct.
- Per-core IRQ/status does not alias.

## Stage 2: SoC Integration Tests

### `tb/tb_soc_mc_mmio.v`

Purpose: verify PicoRV32 can address multiple NPU register windows.

Firmware behavior:

```text
write core0 M_DIM = 0x1111
write core1 M_DIM = 0x2222
read core0 M_DIM, expect 0x1111
read core1 M_DIM, expect 0x2222
read invalid core window, expect DEADBEEF
write PASS marker
```

Acceptance:

- PicoRV32 observes independent register windows through `soc_mc_top`.

### `tb/tb_soc_mc_core0_compat.v`

Purpose: verify `NUM_CORES=1` compatibility.

Options:

- Instantiate `soc_mc_top #(.NUM_CORES(1))` and run an existing tiny Conv or VGG smoke with only core0.
- If existing firmware assumes `soc_top` hierarchy, create a minimal core0 compatibility test first.

Acceptance:

- Final marker matches existing single-core expected result.
- No new address or memory layout behavior is required for `NUM_CORES=1`.

### `tb/tb_soc_mc_shared_a.v`

Purpose: prove the central design choice: shared `A_WORK`, per-core `R_WORK`.

Data pattern:

```text
A_WORK_SHARED = one packed A tile
W_CORE0       = tile for output channels 0..TC-1
W_CORE1       = tile for output channels TC..2*TC-1
R_CORE0       = private result buffer
R_CORE1       = private result buffer
```

Firmware behavior:

```text
program core0 A_ADDR = A_WORK_SHARED, W_ADDR = W_CORE0, R_ADDR = R_CORE0
program core1 A_ADDR = A_WORK_SHARED, W_ADDR = W_CORE1, R_ADDR = R_CORE1
start both cores
poll both STATUS registers
compare R_CORE0/R_CORE1 to golden raw INT32 results
write PASS marker
```

Acceptance:

- Both cores read the same A buffer without requiring duplicate A packing.
- Results match independent golden values.
- Neither core overwrites the other core's result buffer.

## Stage 3: Firmware Generator Tests

Goal: verify the generated multi-core firmware before long full-model simulation.

Checks:

| Check | Method | Expected |
|-------|--------|----------|
| Layout constants | Inspect generated params/metadata | Shared `A_WORK`, per-core `R_WORK` |
| Core base math | Inspect generated assembly or disassembly | `NPU_BASE + core*0x100` |
| A pack count | Instrument marker/counter or inspect code | One A pack per M tile |
| N tile assignment | Metadata or trace markers | Each round assigns at most one N tile per core |
| Partial final round | Small artificial N tile count not divisible by cores | Only launched cores are polled/postprocessed |
| Error path | Force one invalid core launch in a debug case | Fail marker written, no hang |

Acceptance:

- Generator can emit `--num-cores 1` and `--num-cores 2` cases.
- `--num-cores 1` still matches the existing single-core result.

## Stage 4: Model-Level Tests

### Single-Layer Conv

Purpose: catch channel scatter, Q24 multiplier, and N tile assignment bugs before
full VGG.

Recommended layer order:

| Layer | Why |
|-------|-----|
| `stage1_0_conv` | Small `K=27`, catches weight-stride/alignment mistakes |
| `stage2_0_conv` | More channels, moderate M/N |
| `stage4_1_conv` | Largest `K=4608`, stresses A_WORK size and K splitting |

Acceptance:

- Dense OFM matches single-core firmware or Python golden after requant/scatter.

### Full Closed-Loop VGG

Target command after generator/testbench support exists:

```bash
./run_vgg_mc_closed_loop.sh --num-cores 2 --shape 16x16 --flow os
```

Sweep after first pass:

```bash
./run_vgg_mc_closed_loop.sh --num-cores 2 --shape 4x4  --flow os
./run_vgg_mc_closed_loop.sh --num-cores 2 --shape 8x8  --flow os
./run_vgg_mc_closed_loop.sh --num-cores 2 --shape 16x16 --flow os
./run_vgg_mc_closed_loop.sh --num-cores 2 --shape 8x32 --flow os
```

Add WS only after OS passes:

```bash
./run_vgg_mc_closed_loop.sh --num-cores 2 --shape 16x16 --flow ws
```

Acceptance:

- Final class equals exact Python target.
- Optional final feature buffer equals expected fixed-runtime features.
- No timeout, no core error, no fail marker.

## Stage 5: Performance And Resource Checks

Correctness comes first. After correctness passes:

Performance metrics:

- Total cycles for single-core baseline through `soc_top`.
- Total cycles for `soc_mc_top NUM_CORES=1`.
- Total cycles for `soc_mc_top NUM_CORES=2`.
- Per-core raw counters: busy cycles, compute cycles, read/write beats, read/write bytes.

Resource checks for carrier preparation:

- Synthesis resource estimate for `NUM_CORES=1` and `NUM_CORES=2`.
- Confirm `pingpong_buf` maps to intended memory resources.
- Confirm no accidental FP16 datapath inclusion when `FP16_ENABLE=0`.
- Confirm `dram_multi_port` is not included in synthesis file lists.
- Record resource estimate before trying `NUM_CORES=4`.

Acceptance:

- `NUM_CORES=1` new infrastructure has acceptable overhead versus current baseline.
- `NUM_CORES=2` gives positive speedup on Conv-heavy sections.
- Any missing speedup is attributed to PicoRV32 serial work, memory contention model, or scheduling overhead before optimization.

## Debug Markers

Use marker values that make failures localizable:

| Marker range | Meaning |
|--------------|---------|
| `0x2000_0000 + stage` | Test stage entered |
| `0x2100_0000 + core` | Core launch |
| `0x2200_0000 + core` | Core done observed |
| `0xE000_0000 + code` | Firmware/test failure |
| `0x0000_00FF` | Existing generic fail marker |
| `0x0000_0100 + class` | Existing final class marker |

## Recommended Implementation Order For Tests

1. Add `tb/tb_axi_lite_mc_bridge.v`.
2. Add `tb/tb_dram_multi_port.v`.
3. Add `tb/tb_npu_mc_top_smoke.v`.
4. Add `tb/tb_soc_mc_mmio.v`.
5. Add `tb/tb_soc_mc_shared_a.v`.
6. Add multi-core generator mode and run `NUM_CORES=1` compatibility.
7. Run 2-core single-layer Conv cases.
8. Run 2-core full closed-loop VGG.
9. Only then collect speed/resource numbers.
