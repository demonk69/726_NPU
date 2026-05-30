# User Manual

Updated: 2026-05-26

This document summarizes the firmware-facing ABI currently used by the maintained VGG simulation flows.

## Maintained Run Commands

```bash
./run_vgg_e2e.sh
./run_vgg_e2e.sh --image ./pic/test_cifar10_2.jpg
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg
./run_all.sh standard 0
./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg
```

Older PowerShell/Icarus commands are archived and are not the current workflow.

## NPU MMIO Base

`NPU_BASE = 0x02000000`.

Firmware writes NPU registers through PicoRV32 memory-mapped I/O.

## Common NPU Registers

| Offset | Name | Direction | Description |
|---:|---|---|---|
| `0x00` | `CTRL` | W/R | start/control bits |
| `0x04` | `STATUS` | R | busy/done/error bits |
| `0x10` | `M_DIM` | W | GEMM M dimension |
| `0x14` | `N_DIM` | W | GEMM N dimension |
| `0x18` | `K_DIM` | W | GEMM K dimension |
| `0x20` | `W_ADDR` | W | DRAM W tile address |
| `0x24` | `A_ADDR` | W | DRAM A tile address |
| `0x28` | `R_ADDR` | W | DRAM result address |
| `0x30` | `ARR_CFG` | W | array mode, including tile mode |
| `0x3C` | `CFG_SHAPE` | W | shape select |
| `0x40` | `DESC_BASE` | W | RTL descriptor-v1 base |
| `0x44` | `DESC_COUNT` | W | RTL descriptor-v1 count |
| `0x98` | `BIAS_ADDR` | W | bias vector base |
| `0x9C` | `QUANT_CFG` | W | scalar hardware quant config |

## `CTRL` Bits Used By Current Firmware

| Bit(s) | Meaning |
|---|---|
| `0` | start |
| `3:2` | data mode, `00=INT8`, `10=FP16` |
| `5:4` | dataflow/stat mode, VGG uses OS |
| `7` | descriptor mode |
| `8` | direct scalar Conv/im2col mode |
| `9` | bias enable |
| `11:10` | activation mode |

VGG tile mode uses direct register writes, not RTL descriptor-v1.

## Shape Select

| `CFG_SHAPE` | Shape |
|---:|---|
| `0` | 4x4 |
| `1` | 8x8 |
| `2` | 16x16 |
| `3` | 8x32 |

Current VGG flows use `CFG_SHAPE=2` for 16x16.

## VGG E2E Tile Table ABI

The fast e2e firmware uses a 10-word table per tile. This is a firmware ABI and is separate from RTL descriptor-v1.

| Word | Meaning |
|---:|---|
| 0 | `M_DIM` |
| 1 | `N_DIM` |
| 2 | `K_DIM` |
| 3 | W tile base |
| 4 | A tile base |
| 5 | R tile base |
| 6 | Bias base |
| 7 | `QUANT_CFG` |
| 8 | `ARR_CFG` |
| 9 | `CFG_SHAPE` |

The e2e generator pre-populates all A tile streams and all table entries before simulation.

## Runtime Closed-Loop Descriptor ABI

The closed-loop generator writes per-layer firmware descriptors at `DESC_BASE`. Firmware reads these descriptors to know input/output buffers, W/bias/multiplier bases, dimensions, packed widths, and spatial shapes.

The exact fields are emitted in `tools/pth/gen_vgg_closed_loop.py` near the `fields = [...]` list. Treat that generator as the ABI source of truth until a packed C header is introduced.

## Quantization Guidance

Hardware `QUANT_CFG` supports one scalar quant configuration per NPU launch. This is not enough for exact RepOpt VGG per-channel requant.

Current closed-loop rule:

- Write `QUANT_DISABLED` to `REG_QUANT_CFG`.
- Start NPU with bias enabled but no hardware quant.
- Read raw INT32+bias results from `R_WORK`.
- Firmware applies ReLU and per-output-channel Q24 multiplier.

## Marker ABI

Testbenches monitor `MARKER_ADDR` in DRAM.

| Marker | Meaning |
|---:|---|
| `0x200 + step` | firmware progress marker |
| `0x100 + class_id` | classification result marker |
| `0x000000FF` | firmware failure marker |

Closed-loop testbench compares the class marker against exact Python model output.

## FPGA I/O ABI Planned

PYNQ-Z2 deployment uses PS ARM plus PL NPU as the primary route. Each image returns one class plus raw performance counters; derived TOPS and bus utilization are computed on the host. The current ABI direction is documented in `doc/pynq_z2_deployment.md`.
