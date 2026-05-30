# Current NPU/SoC Architecture

Updated: 2026-05-26

This document describes the current implementation state. It is not a historical roadmap and not a target-only architecture plan.

For verification results, see `doc/verification_status.md`. For the PYNQ-Z2 deployment direction, see `doc/pynq_z2_deployment.md`.

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
| `0x30` | `ARR_CFG` | tile mode control, including `bit7=1` |
| `0x3C` | `CFG_SHAPE` | shape select: 4x4, 8x8, 16x16, 8x32 |
| `0x40` | `DESC_BASE` | RTL descriptor-v1 base address |
| `0x44` | `DESC_COUNT` | RTL descriptor-v1 count |
| `0x98` | `BIAS_ADDR` | Bias vector address |
| `0x9C` | `QUANT_CFG` | Scalar hardware post-quant config |

The runtime closed-loop VGG path disables hardware `QUANT_CFG` and performs exact per-channel Q24 requant in CPU firmware.

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

## FPGA Deployment Direction

The current board target is PYNQ-Z2. The primary route uses Zynq PS ARM as the runtime CPU and keeps the NPU in PL. PS/host software will receive images, manage DDR buffers, program NPU registers, read raw performance counters, and compute derived TOPS and bus utilization.

The previous pure-PL UART/SPI Flash/Boot ROM route is deferred for non-Zynq boards. The current flow is documented in `doc/pynq_z2_deployment.md`.
