# RTL Reference

Updated: 2026-06-15

This document is the current module-level map for the maintained simulation SoC and Vivado NPU integration. Historical worklogs under `doc/archive/` may describe superseded flows.

## Source Sets

Use `scripts/vivado_npu_filelist.tcl` for board/Vivado projects.

| Source list | Purpose |
|---|---|
| `$npu_vivado_project_rtl_files` | Default INT8-only NPU source set for `npu_top` |
| `$npu_vivado_fp16_project_rtl_files` | Full source set for `FP16_ENABLE=1` builds |
| `$npu_vivado_excluded_sim_rtl_files` | Simulation-only SoC files that should not be board design sources |

Default board builds set `FP16_ENABLE=0`. In that mode `pe_top` does not elaborate `fp16_mul` or `fp32_add`, so `rtl/pe/fp16_mul.v` and `rtl/pe/fp32_add.v` are not required design sources.

## SoC Layer

These files are used by Verilator/PicoRV32 simulation. They model a complete CPU plus NPU environment and are not the Vivado PL NPU boundary.

| Module | File | Function | Main Inputs | Main Outputs |
|---|---|---|---|---|
| `soc_top` | `rtl/soc/soc_top.v` | Simulation SoC integrating PicoRV32, SRAM, DRAM model, AXI-Lite bridge, and NPU | `clk`, `rst_n` | Internal CPU/NPU/DRAM wiring, no board I/O ports |
| `soc_mem` | `rtl/soc/soc_mem.v` | CPU SRAM for firmware instruction/data memory | `clk`, `wen`, `addr`, `wdata` | `rdata` |
| `dram_model` | `rtl/soc/dram_model.v` | Simulation DRAM shared by CPU simple port and NPU AXI4 master port | CPU valid/we/address/data, AXI4 AW/W/B/AR/R handshakes | CPU read data/ready, AXI read/write responses |
| `axi_lite_bridge` | `rtl/soc/axi_lite_bridge.v` | PicoRV32 simple memory transaction to AXI4-Lite transaction converter | simple-bus valid/address/wdata/wstrb, AXI-Lite ready/response | simple-bus ready/rdata, AXI-Lite AW/W/B/AR/R signals |

Simulation address decode in `soc_top`:

| Address range | Target |
|---|---|
| `addr < 4*MEM_WORDS` | `soc_mem` SRAM |
| `4*MEM_WORDS <= addr < 0x0200_0000` | `dram_model` CPU port |
| `addr >= 0x0200_0000` | NPU registers through `axi_lite_bridge` |

## NPU Top-Level

| Module | File | Function | Main Inputs | Main Outputs |
|---|---|---|---|---|
| `npu_top` | `rtl/top/npu_top.v` | Board-independent NPU integration: register file, controller, DMA, W/A buffers, PE datapaths, postprocess, counters, power enables | `sys_clk`, `sys_rst_n`, compact AXI4-Lite slave inputs, AXI4 master ready/response/read-data inputs | AXI4-Lite outputs, AXI4 master outputs, `npu_irq` |

Important top parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `PHY_ROWS` | 16 | Physical PE rows |
| `PHY_COLS` | 16 | Physical PE columns |
| `DATA_W` | 32 | PE input data width; default carries four INT8 SIMD lanes |
| `ACC_W` | 32 | Accumulator and AXI data width |
| `PPB_DEPTH` | 64 | Words per ping-pong buffer bank |
| `PPB_THRESH` | 16 | Buffer ready threshold |
| `INT8_SIMD_LANES` | 4 | INT8 lanes per PE MAC cycle |
| `FP16_ENABLE` | 0 | Enables optional FP16 datapath when set to 1 |
| `PERF_ENABLE_DERIVED` | 0 | Enables division-heavy derived performance metrics |

## Register And Control Layer

| Module | File | Function | Main Inputs | Main Outputs |
|---|---|---|---|---|
| `npu_axi_lite` | `rtl/axi/npu_axi_lite.v` | AXI4-Lite register file, status, IRQ, error clear, performance snapshots | AXI4-Lite AW/W/B/AR/R slave handshakes, controller status/error, performance counters | Config registers to `npu_ctrl`, AXI4-Lite responses, `npu_irq`, `err_clear`, `perf_clear` |
| `npu_ctrl` | `rtl/ctrl/npu_ctrl.v` | Main scheduler FSM for direct mode and descriptor-v1 mode; computes tile loops, DMA byte lengths, PE controls, writeback sequencing | Config registers, descriptor words, DMA done/error, PPBuf status, `err_clear` | DMA commands, PE control signals, PPBuf swap/clear, result FIFO clear, status/done/error/IRQ |

Key `npu_ctrl` outputs:

| Output group | Signals | Meaning |
|---|---|---|
| DMA W/A | `dma_w_start`, `dma_w_addr`, `dma_w_len`, `dma_a_start`, `dma_a_addr`, `dma_a_len` | Launch weight and activation reads |
| DMA transformed A | `dma_a_ofm_*`, `dma_a_im2col_*` | Generate activation stream from previous OFM or on-the-fly im2col path |
| Bias/result | `dma_bias_start`, `dma_bias_addr`, `dma_r_start`, `dma_r_addr`, `dma_r_len` | Fetch bias and write result rows |
| PE control | `pe_en`, `pe_flush`, `pe_mode`, `pe_stat`, `pe_load_w`, `pe_swap_w`, `pe_half_en` | Drive scalar PE and array PE execution |
| Buffer control | `w_ppb_swap`, `a_ppb_swap`, `w_ppb_clear`, `a_ppb_clear`, `r_fifo_clear` | Manage ping-pong and result buffers |

When `FP16_ENABLE=0`, `npu_ctrl` forces hardware execution to INT8 and rejects FP16 launches with `ERR_FP16_DISABLED = 32'h0000_0400`.

## DMA And Memory Movement Layer

| Module | File | Function | Main Inputs | Main Outputs |
|---|---|---|---|---|
| `npu_dma` | `rtl/axi/npu_dma.v` | Shared AXI4 master with independent read/load FSM and writeback FSM | W/A/descriptor/bias/result commands, PPBuf full/status, result FIFO writes, AXI4 ready/response/read data | PPBuf write enables/data, descriptor/bias data, done flags, result FIFO full, AXI4 AW/W/B/AR/R master signals, DMA error status |
| `sync_fifo` | `rtl/common/fifo.v` | Generic synchronous FIFO used by result writeback path | `clk`, `rst_n`, `clear`, `wr_en`, `wr_data`, `rd_en` | `rd_data`, `full`, `empty`, almost flags, fill count |

`npu_dma` traffic classes:

| Class | Source/Destination | Notes |
|---|---|---|
| Weight load | DRAM/DDR to W PPBuf | 32-bit words, byte count from controller |
| Activation load | DRAM/DDR to A PPBuf | Direct packed tiles or generated from OFM/im2col modes |
| Descriptor fetch | DRAM/DDR to `desc_words[511:0]` | RTL descriptor-v1 path, not used by current VGG firmware |
| Bias fetch | DRAM/DDR one 32-bit word to `bias_data` | Used by scalar and tile postprocess paths |
| Result writeback | result FIFO to DRAM/DDR | Writes 32-bit accumulators row by row in tile mode |

## Buffer And Compute Layer

| Module | File | Function | Main Inputs | Main Outputs |
|---|---|---|---|---|
| `pingpong_buf` | `rtl/buf/pingpong_buf.v` | Dual-bank W/A buffer with DMA write side, scalar read side, and vector read side | `wr_en`, `wr_data`, `rd_en`, `rd_vec_en`, `rd_vec_lanes`, `swap`, `clear`, `packed_int8` | `rd_data`, `rd_vec`, `rd_vec_valid`, `buf_empty`, `buf_full`, `buf_ready`, fill counts |
| `reconfig_pe_array` | `rtl/array/reconfig_pe_array.v` | Physical 16x16 PE array wrapper with 4x4, 8x8, 16x16, and folded 8x32 modes | `cfg_shape`, `mode`, `stat_mode`, `en`, `flush`, `load_w`, `swap_w`, `ws_direct`, `half_en`, W/A vectors, accumulator inputs | flattened accumulator results, valid mask, `ws_load_row_out`, `pe_active` |
| `pe_top` | `rtl/pe/pe_top.v` | One PE with dual weight registers, INT8 SIMD MAC, optional FP16 MAC, OS/WS accumulation | `mode`, `stat_mode`, `en`, `flush`, `load_w`, `swap_w`, `acc_init_en`, `w_in`, `a_in`, `acc_in`, `acc_init` | `acc_out`, `valid_out` |
| `fp16_mul` | `rtl/pe/fp16_mul.v` | Optional FP16 multiply block | FP16 operands, clock/reset/enable | FP16/FP32-formatted product path into PE |
| `fp32_add` | `rtl/pe/fp32_add.v` | Optional FP32 adder for FP16 accumulation | FP32 operands | FP32 sum |
| `psum_out_buf` | `rtl/buf/psum_out_buf.v` | Tile-local dual-port PSUM/OUT storage block kept as reusable RTL | clear, valid mask, port A/B read/write controls | port A/B read data/valid, write conflict flag |

`psum_out_buf` is not part of the current NPU dependency closure. It remains in the optional extra source list for compatibility and future K-split/partial-sum work.

## Performance And Power Support

| Module | File | Function | Main Inputs | Main Outputs |
|---|---|---|---|---|
| `axi_monitor` | `rtl/common/axi_monitor.v` | Counts AXI read/write bursts, beats, bytes, and optional derived utilization | AXI handshake signals, `clear` | raw and derived AXI performance counters |
| `op_counter` | `rtl/common/op_counter.v` | Counts useful MAC/OPS, busy/compute/DMA cycles, peak ops per cycle | controller/PE activity, dimensions, shape context, SIMD lanes | MAC/OPS counters, utilization counters, derived TOPS when enabled |
| `npu_power` | `rtl/power/npu_power.v` | Emits clock-enable style global/row/column CE signals and keeps `npu_clk=clk` | `clk`, `rst_n`, divider select, row/col gate requests | `global_ce`, row enables, column enables, legacy aliases |

`npu_power` does not generate or gate fabric clocks. Its historical `row_clk_gated` and `col_clk_gated` port names now carry enable semantics. `npu_top` connects these CE signals into `reconfig_pe_array`, where they gate PE compute/control updates and OS shift-register updates through normal clock-enable logic. `CLK_DIV` is applied to the compute CE path via an idle-only latch in `npu_top`; the controller and top-level compute signals are CE-aware (`compute_ce` gates `tile_k_cycle`, `tile_feed_step`, `tile_vec_fire`, etc.).

## Shape Modes

The physical array is always 16x16. `CFG_SHAPE` selects the active logical tile shape.

| `CFG_SHAPE` | Logical shape | Active rows | Active cols | Notes |
|---:|---|---:|---:|---|
| `0` | 4x4 | 4 | 4 | Small-tile regression and edge handling |
| `1` | 8x8 | 8 | 8 | Medium square tile |
| `2` | 16x16 | 16 | 16 | Default VGG closed-loop shape |
| `3` | 8x32 | 8 | 32 | Folded two-pass mode over the 16x16 physical array |

## Execution Paths

| Path | Current use |
|---|---|
| Direct register tile mode | Used by VGG e2e and closed-loop firmware |
| RTL descriptor-v1 mode | Present in `npu_ctrl`; not used by current VGG firmware |
| Direct scalar Conv/im2col path | Legacy/smoke-test path and DMA address-generator coverage |

## Post-Process Status

The RTL contains scalar/tile bias, activation, and scalar quantization support. This is valid when one `QUANT_CFG` applies to the whole launch.

RepOpt VGG needs per-output-channel requant. Runtime closed-loop therefore disables hardware quant, writes raw INT32+bias results, and performs ReLU plus per-channel Q24 requant in CPU firmware.
