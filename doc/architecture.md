# Current NPU/SoC Architecture

Updated: 2026-06-24

This document describes the current implementation state. It is not a historical roadmap and not a target-only architecture plan.

For verification results, see `doc/verification_status.md`.

## System Boundary

The current simulation SoC contains:

```text
Verilator testbench
  |
  | $readmemh firmware into SRAM
  | $readmemh DRAM init image into DRAM model
  | monitor marker word for PASS/FAIL/TIMEOUT
  v
soc_top
  |
  +-- PicoRV32 CPU
  |     +-- SRAM: firmware instruction/data memory
  |     +-- DRAM model: runtime buffers, model data, markers
  |     +-- AXI-Lite bridge to NPU register file
  |
  +-- NPU
  |     +-- AXI-Lite slave register file
  |     +-- AXI4 master DMA for W/A/bias/result traffic
  |     +-- ping-pong buffers
  |     +-- reconfigurable 16x16 physical PE array
  |     +-- post-process path and result FIFO
  |
  +-- DRAM model
        +-- CPU simple memory port
        +-- NPU AXI4 slave port
```

There is currently no UART, SPI Flash controller, Boot ROM, GPIO, or board-level I/O peripheral in RTL.

## Board-Level NPU Boundary

`rtl/top/npu_top.v` is the board-independent NPU integration module and provides a compact AXI4-Lite slave port (register file) plus an AXI4 master port (DMA for W/A/bias/result traffic). For FPGA board integration it should be wrapped with the target platform's AXI infrastructure (interconnect, clock, reset). The simulation SoC files under `rtl/soc/` are not board-level design sources.

`scripts/vivado_npu_filelist.tcl` is the maintained source list for this boundary. Its default list is INT8-only and excludes the optional FP16 datapath files. Use `$npu_vivado_fp16_project_rtl_files` and set `FP16_ENABLE=1` only when a full FP16 build is required.

## NPU Internal Data Path

The direct tile GEMM path used by the maintained VGG firmware is:

```text
CPU firmware
  |
  | AXI-Lite writes: dimensions, W/A/R addresses, bias, quant, shape, start
  v
npu_axi_lite --> npu_ctrl
                  |
                  +-- issues W/A DMA reads
                  +-- swaps ping-pong banks
                  +-- controls PE enable/flush/load
                  +-- issues bias fetch and result writeback

DRAM/DDR --AXI4 read--> npu_dma --W words--> W pingpong_buf --vectors--> PE array
DRAM/DDR --AXI4 read--> npu_dma --A words--> A pingpong_buf --vectors--> PE array

PE array --INT32 accumulators--> serializer/postprocess --> result FIFO
result FIFO --AXI4 write via npu_dma--> DRAM/DDR
```

The W/A ping-pong buffers hold 32-bit DMA words. In default INT8 mode each word contains four signed INT8 elements. Tile mode reads vectors from both buffers, feeds one row/column vector per logical K step, and lets each PE accumulate the dot product.

`FP16_ENABLE=0` removes the PE FP16 multiply/add datapath at elaboration time. The controller also rejects FP16 work requests so software cannot silently run FP16 descriptors on INT8 hardware.

## Address Map

| Region | Meaning |
|---|---|
| `0x0000_0000 .. 4*MEM_WORDS-1` | SRAM, used by PicoRV32 firmware |
| `4*MEM_WORDS .. 0x01FF_FFFF` | DRAM model, shared by CPU and NPU DMA |
| `0x0200_0000 ..` | NPU MMIO register window |

VGG testbenches use large simulation DRAM images. Generated artifacts under `sim/vgg_e2e/` and `sim/vgg_closed_loop/` are not source files.

## Main RTL Components

| File | Role |
|---|---|
| `rtl/soc/soc_top.v` | Integrates CPU, SRAM, DRAM model, AXI-Lite bridge, and NPU |
| `rtl/soc/soc_mem.v` | Simple SRAM for CPU firmware |
| `rtl/soc/dram_model.v` | Simulation DRAM with CPU and NPU access ports |
| `rtl/soc/axi_lite_bridge.v` | PicoRV32 simple bus to AXI-Lite bridge |
| `rtl/top/npu_top.v` | NPU top-level integration |
| `rtl/ctrl/npu_ctrl.v` | NPU scheduling FSM and descriptor decode |
| `rtl/axi/npu_dma.v` | AXI4 DMA for W/A/bias/result movement |
| `rtl/array/reconfig_pe_array.v` | Reconfigurable PE array wrapper |
| `rtl/pe/*.v` | PE datapath blocks |

## NPU Register Interface

CPU firmware programs the NPU through the MMIO base `0x02000000`.

| Offset | Name | Use |
|---:|---|---|
| `0x00` | `CTRL` | start, data mode, OS/WS mode, descriptor mode, bias enable, activation |
| `0x04` | `STATUS` | busy/done/error |
| `0x10` | `M_DIM` | GEMM M dimension |
| `0x14` | `N_DIM` | GEMM N dimension |
| `0x18` | `K_DIM` | GEMM K dimension |
| `0x20` | `W_ADDR` | W tile address in DRAM |
| `0x24` | `A_ADDR` | A tile address in DRAM |
| `0x28` | `R_ADDR` | Result address in DRAM |
| `0x30` | `ARR_CFG` | tile mode control |
| `0x34` | `CLK_DIV` | CE divisor: 0=1x, 1=1/2, 2=1/4, 3=1/8 |
| `0x38` | `CG_EN` | shape-based row/column CE masking |
| `0x3C` | `CFG_SHAPE` | shape select: 4x4, 8x8, 16x16, 8x32 |
| `0x40` | `DESC_BASE` | RTL descriptor-v1 base address |
| `0x44` | `DESC_COUNT` | RTL descriptor-v1 count |
| `0x98` | `BIAS_ADDR` | Bias vector address |
| `0x9C` | `QUANT_CFG` | Scalar hardware post-quant config |

The runtime closed-loop VGG path disables hardware `QUANT_CFG` and performs exact per-channel Q24 requant in CPU firmware.

## DFS (Clock-Enable Throttling)

The NPU supports clock-enable based compute rate control via `CLK_DIV` register (`0x34`). This is not PLL/MMCM dynamic reconfiguration — it uses a single `sys_clk` with CE pulse gating.

| `CLK_DIV` | Rate | CE duty cycle |
|---:|---|---|
| `0` | 1x (full speed) | 100% |
| `1` | 1/2 | 50% |
| `2` | 1/4 | 25% |
| `3` | 1/8 | 12.5% |

**Gated path:** `npu_power` → `global_ce` → `compute_ce` gates PE array, controller `tile_k_cycle`/`ws_consume_cnt`, drain state, `tile_feed_step`, `tile_vec_fire`, scalar PE enable, and PPBuf consumption.

**Not gated:** DMA, AXI-Lite register access, result serializer, performance counters, and controller FSM state decode run at full `sys_clk`.

**Latch:** An idle-only latch in `npu_top` captures `clk_div_eff ← clk_div_r` only when `STATUS_BUSY=0`. Writes to `CLK_DIV` during active compute are safely ignored.

Firmware entry: `--clk-div 0|1|2|3` via runner scripts and generator. Default `0`.

## PPB Depth And K Tiling

The W/A ping-pong buffer depth is controlled by the `PPB_DEPTH` Verilog parameter. In VGG closed-loop simulations this is generated as `VGG_CLOSED_PPB_DEPTH` and passed by the testbench into `soc_top.NPU_PPB_DEPTH` or `soc_mc_top.NPU_PPB_DEPTH`.

`PPB_DEPTH` affects two things that must stay matched:

| Component | Effect |
|---|---|
| `pingpong_buf.DEPTH` | Physical W/A buffer capacity in 32-bit words per bank |
| `npu_ctrl.PPB_DEPTH` | `k_tile_elems = (PPB_DEPTH * 4) / bytes_per_k`, controlling K-split |
| `gen_vgg_closed_loop.py` | Computes matching `KT_ELEMS` and emits `VGG_CLOSED_PPB_DEPTH` |

Runner entry: `--ppb-depth <words>` for single/multi closed-loop, and `--ppb-depths <csv>` for sweep. This is not an MMIO register; it is a build/testbench-time parameter.

## Multi-Core

`rtl/top/npu_mc_top.v` and `rtl/soc/soc_mc_top.v` wrap multiple `npu_top` instances sharing a single DRAM model and PicoRV32 CPU.

| File | Role |
|---|---|
| `rtl/top/npu_mc_top.v` | Multi-core NPU wrapper with per-core AXI-Lite/DMA ports |
| `rtl/soc/soc_mc_top.v` | SoC integrator: CPU, SRAM, `dram_multi_port`, AXI-Lite multi-core bridge |
| `rtl/soc/dram_multi_port.v` | Simulation multi-port DRAM shared across CPU + N NPU cores |
| `rtl/soc/axi_lite_mc_bridge.v` | Single PicoRV32 bus → per-core AXI-Lite fanout |

Each core has its own `CLK_DIV`, CG, and PE array configuration; firmware can set different divisors per core.

Entry: `./run_vgg_mc_closed_loop.sh --num-cores 1|2|4`.

## Execution Paths

### Fast VGG E2E

Entry points:

```bash
./run_vgg_e2e.sh
./run_all.sh standard 0
```

Generator: `tools/pth/gen_vgg_e2e.py`.

Python pre-generates all Conv A tile streams for all 9 Conv layers. Firmware iterates a 1024-entry VGG tile table, starts the NPU tile by tile, then performs avgpool, classifier, and argmax.

This is a full classification baseline, but not a runtime layer-to-layer deployment flow.

### Runtime Closed Loop

Entry points:

```bash
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg
./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg
```

Generator: `tools/pth/gen_vgg_closed_loop.py`.

Python generates static model assets and the selected input image. Firmware packs A tiles from dense activation buffers at runtime, runs the NPU, applies CPU-side per-channel Q24 requant/scatter, then performs maxpool, avgpool, classifier, and argmax.

This is the deployment-oriented simulation path.

## Model Inference Data Flow

There are two maintained model-level flows. They share the same RTL NPU datapath but differ in where Conv A tiles are produced.

### Fast E2E Baseline Data Flow

```text
Python generator
  |
  +-- runs host reference
  +-- pre-packs every Conv A tile
  +-- emits DRAM image with input, A tiles, W tiles, bias, quant, classifier
  +-- emits PicoRV32 firmware and tile table

Verilator testbench
  |
  +-- loads firmware into SRAM
  +-- loads generated DRAM image into DRAM model

Firmware
  |
  +-- iterates generated tile table
  +-- programs NPU registers for each tile
  +-- waits for NPU completion
  +-- runs avgpool, classifier, argmax on CPU
  +-- writes result marker
```

This flow is fast because Python has already created all Conv tile streams. It is useful as a regression baseline but is not the deployment model.

### Runtime Closed-Loop Data Flow

```text
Python generator
  |
  +-- emits dense input activation buffer
  +-- emits static weights, bias, Q24 multipliers, classifier params
  +-- emits image-independent PicoRV32 firmware

Firmware for each Conv layer
  |
  +-- reads dense INT8 activation buffer
  +-- packs the next A tile into A_WORK
  +-- programs W_ADDR, A_ADDR, R_ADDR, BIAS_ADDR, dimensions, shape
  +-- starts the NPU
  +-- reads raw INT32+bias tile results from R_WORK
  +-- applies ReLU and per-output-channel Q24 requant on CPU
  +-- scatters INT8 output into the next dense activation buffer

Firmware after Conv layers
  |
  +-- maxpool between selected Conv groups
  +-- avgpool final feature map
  +-- classifier and argmax
  +-- result marker for the testbench
```

This flow is the deployment-oriented simulation path because the runtime CPU owns dense activation buffers and performs tile packing/scatter itself. On Zynq deployment the PicoRV32/testbench roles are replaced by PS software and DDR, while the PL NPU register/DMA contract stays the same.

### RTL Descriptor-v1 Path

The NPU controller still contains an RTL descriptor-v1 path. The current VGG firmware does not use it. VGG uses a firmware-managed tile table and direct register writes.

## VGG Closed-Loop DRAM Layout

| Symbol | Address | Role |
|---|---:|---|
| `ACT_A` | `0x00010000` | Dense activation buffer A |
| `ACT_B` | `0x00030000` | Dense activation buffer B |
| `A_WORK` | `0x00050000` | Runtime packed A tile workspace |
| `R_WORK` | `0x00068000` | Raw NPU result tile workspace |
| `FEAT_BASE` | `0x00069000` | Avgpool feature buffer |
| `SCORE_BASE` | `0x0006A000` | Classifier score buffer |
| `MARKER_ADDR` | `0x0006B000` | Progress/result marker |
| `DESC_BASE` | `0x0006C000` | Firmware layer descriptor table |
| `STATIC_BASE` | `0x00070000` | Weights, bias, Q24 multipliers, classifier params |

## Quantization Status

The hardware scalar/tile `QUANT_CFG` path supports one scalar quant config per NPU launch. RepOpt VGG needs per-output-channel requant.

Current policy:

- Fast e2e keeps its generated quant path for regression.
- Runtime closed-loop disables hardware quant and reads raw INT32+bias results.
- Firmware applies ReLU and per-channel Q24 requant on the CPU.
- Testbench validates against exact Python model output, not the old per-16-channel approximation.
