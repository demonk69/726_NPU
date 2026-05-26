# RTL Reference

Updated: 2026-05-26

This document is a current module-level map. It intentionally avoids historical phase language.

## SoC Layer

| Module | File | Role |
|---|---|---|
| `soc_top` | `rtl/soc/soc_top.v` | Top-level SoC integration: PicoRV32, SRAM, DRAM model, AXI-Lite bridge, NPU |
| `soc_mem` | `rtl/soc/soc_mem.v` | Simple SRAM backing firmware instruction/data memory |
| `dram_model` | `rtl/soc/dram_model.v` | Simulation DRAM with CPU and AXI4/NPU access paths |
| `axi_lite_bridge` | `rtl/soc/axi_lite_bridge.v` | Bridges PicoRV32 memory-mapped I/O to NPU AXI-Lite registers |

Current limitation: there are no UART, SPI Flash, Boot ROM, GPIO, or board-level peripherals yet.

## NPU Top Layer

| Module | File | Role |
|---|---|---|
| `npu_top` | `rtl/top/npu_top.v` | NPU integration, register interface, controller, DMA, buffers, PE array, post-process/result FIFO |
| `npu_axi_lite` | `rtl/axi/npu_axi_lite.v` | NPU register file and performance/status counters |
| `npu_ctrl` | `rtl/ctrl/npu_ctrl.v` | Main scheduling FSM, direct-register mode, descriptor-v1 decode, tile iteration |
| `npu_dma` | `rtl/axi/npu_dma.v` | AXI4 read/write engine for W/A/descriptor/bias/result movement |

## Compute Layer

| Module | File | Role |
|---|---|---|
| `reconfig_pe_array` | `rtl/array/reconfig_pe_array.v` | Physical 16x16 PE array wrapper with shape modes |
| `pe_top` and PE submodules | `rtl/pe/*.v` | INT8/FP16 MAC datapath elements |
| `pingpong_buf` | `rtl/buf/*.v` | W/A ping-pong storage and streaming support |
| `psum_out_buf` or result FIFO blocks | `rtl/buf/*.v` | Result buffering and writeback path support |

## Shape Modes

The physical array is 16x16. Current shape select values:

| `CFG_SHAPE` | Mode |
|---:|---|
| `0` | 4x4 |
| `1` | 8x8 |
| `2` | 16x16 |
| `3` | 8x32 |

The VGG e2e and closed-loop flows use 16x16 shape mode.

## Post-Process Status

The RTL contains a scalar/tile post-process path for bias, activation, and scalar quantization. That path is still useful for tests where one `QUANT_CFG` applies to a whole tile/layer.

RepOpt VGG needs per-output-channel requant. The runtime closed-loop flow therefore disables hardware quant and performs per-channel Q24 requant in CPU firmware after raw INT32+bias NPU output.

## Execution Paths

| Path | Current use |
|---|---|
| Direct register mode | Used by VGG e2e and closed-loop firmware |
| RTL descriptor-v1 mode | Present in controller; not used by current VGG firmware |
| Direct scalar Conv/im2col path | Legacy/smoke-test path |

## Planned Additions For FPGA Deployment

| Planned module | Purpose |
|---|---|
| `uart` | Receive 3x32x32 INT8 images and return class byte |
| `spi_flash_reader` | Read firmware/static assets from SPI Flash |
| `boot_rom` | Copy Flash contents into SRAM/DRAM after reset |

See `doc/uart_spi_fpga_plan.md` for the planned interface and bring-up order.
