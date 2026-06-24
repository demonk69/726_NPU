# User Manual

Updated: 2026-06-24

This document summarizes the firmware-facing ABI currently used by the maintained VGG simulation flows.

## Maintained Run Commands

```bash
./run_vgg_e2e.sh
./run_vgg_e2e.sh --image ./pic/test_cifar10_2.jpg
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg --clk-div 1
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg --ppb-depth 4096
./run_vgg_mc_closed_loop.sh --num-cores 2 --image ./pic/test_cifar10_2.jpg
./run_vgg_closed_loop_sweep.sh --lanes 4 --num-cores 1,2 --shapes 16x16 --ppb-depths 1024,4096,8192
./run_all.sh standard 0
./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg
```

Older PowerShell/Icarus commands are archived and are not the current workflow.

## Vivado Source Lists And FP16 Build Options

The default FPGA build is now INT8-only to reduce LUT/DSP pressure on Zynq boards. FP16 is not deleted from the RTL, but it is controlled by the synthesizable parameter `FP16_ENABLE`.

Use the maintained Vivado filelist:

```tcl
source scripts/vivado_npu_filelist.tcl
add_files -fileset sources_1 $npu_vivado_project_rtl_files
set_property top npu_top [get_filesets sources_1]
update_compile_order -fileset sources_1
```

This default list excludes the FP16 datapath source files because `FP16_ENABLE=0` makes `pe_top` elaborate only the INT8 datapath.

To build with FP16 enabled:

```tcl
add_files -fileset sources_1 $npu_vivado_fp16_project_rtl_files
set_property top npu_top [get_filesets sources_1]
update_compile_order -fileset sources_1
```

If the NPU is instantiated as a Vivado block-design module cell, also set the module parameter on the BD cell:

```tcl
set_property -dict [list CONFIG.FP16_ENABLE {1}] [get_bd_cells npu_0]
```

If `npu_top` is used directly as the RTL top, set the Verilog generic/parameter `FP16_ENABLE=1` in the synthesis run or instantiate the top with `.FP16_ENABLE(1)`.

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
| `0x34` | `CLK_DIV` | W/R | clock-enable divisor (0=1x, 1=1/2, 2=1/4, 3=1/8); invalid values fall back to 1x |
| `0x38` | `CG_EN` | W/R | enables shape-based row/column CE into the PE array |
| `0x3C` | `CFG_SHAPE` | W | shape select |
| `0x40` | `DESC_BASE` | W | RTL descriptor-v1 base |
| `0x44` | `DESC_COUNT` | W | RTL descriptor-v1 count |
| `0x98` | `BIAS_ADDR` | W | bias vector base |
| `0x9C` | `QUANT_CFG` | W | scalar hardware quant config |

## `CTRL` Bits Used By Current Firmware

| Bit(s) | Meaning |
|---|---|
| `0` | start |
| `3:2` | data mode, `00=INT8`, `10=FP16` when `FP16_ENABLE=1` |
| `5:4` | dataflow/stat mode, VGG uses OS |
| `7` | descriptor mode |
| `8` | direct scalar Conv/im2col mode |
| `9` | bias enable |
| `11:10` | activation mode |

VGG tile mode uses direct register writes, not RTL descriptor-v1.

When `FP16_ENABLE=0`, firmware should program `CTRL[3:2]=00`. If firmware requests FP16, the controller rejects the launch and sets `ERR_STATUS[10]`, encoded as `ERR_FP16_DISABLED = 32'h0000_0400`.

## Shape Select

| `CFG_SHAPE` | Shape |
|---:|---|
| `0` | 4x4 |
| `1` | 8x8 |
| `2` | 16x16 |
| `3` | 8x32 |

Current VGG flows use `CFG_SHAPE=2` for 16x16.

## Power/CE Control

`CG_EN=1` enables row/column clock-enable masks derived from `CFG_SHAPE`. This uses normal register clock-enable logic inside the PE array; it does not create gated clocks.

`CLK_DIV` controls a clock-enable pulse that gates the PE array and compute pipeline: `0=1x, 1=1/2, 2=1/4, 3=1/8`. DMA, AXI-Lite, and result serializer continue at full `sys_clk`. Write `CLK_DIV` while the NPU is idle (before asserting START); the RTL latches the new value only when `STATUS_BUSY=0`. Writes during active compute are safely ignored and do not affect the in-flight computation. The default is `0` (full speed).

All runners accept `--clk-div`:
```bash
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg --clk-div 1
./run_vgg_mc_closed_loop.sh --num-cores 2 --clk-div 2
./run_vgg_closed_loop_sweep.sh --clk-divs 0,1,2,3 --shapes 16x16
```

## PPB Depth Sweeps

`PPB_DEPTH` is the W/A ping-pong buffer depth in 32-bit words per bank. It is a Verilog parameter passed through the VGG closed-loop testbench, not an MMIO register.

Use these options before generation/compile:

```bash
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg --ppb-depth 4096
./run_vgg_mc_closed_loop.sh --num-cores 2 --ppb-depth 4096
./run_vgg_closed_loop_sweep.sh --shapes 16x16 --flows os --lanes 4 --ppb-depths 1024,4096,8192
```

The generator writes `VGG_CLOSED_PPB_DEPTH` into `soc_vgg_closed_loop_params.vh` and recomputes `KT_ELEMS`; the testbench passes the same value into `soc_top.NPU_PPB_DEPTH` or `soc_mc_top.NPU_PPB_DEPTH`.

## Multi-Core Operation

```bash
./run_vgg_mc_closed_loop.sh --num-cores 2 --image ./pic/test_cifar10_2.jpg
./run_vgg_closed_loop_sweep.sh --num-cores 1,2 --lanes 4 --shapes 16x16 --flows os
```

Each core has independent `CLK_DIV`, CG, and PE config registers. The firmware writes per-core registers through the multi-core AXI-Lite bridge (`axi_lite_mc_bridge`).

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
