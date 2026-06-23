# Verification Status

Updated: 2026-06-23

This document is the current verification record. Older status files and worklogs in `doc/archive/` are historical and may describe superseded Windows/Icarus flows.

## Maintained Entry Points

| Command | Purpose | Expected status |
|---|---|---|
| `./run_vgg_e2e.sh` | Fast RepOpt VGG baseline | PASS |
| `./run_all.sh standard [idx]` | Unified wrapper for fast baseline | PASS |
| `./run_all.sh image <file>` | Fast baseline with arbitrary image input | Supported |
| `./run_vgg_closed_loop.sh [idx|--image <file>] [--shape <shape>]` | Runtime closed-loop VGG | PASS on tested default shape images |
| `./run_vgg_closed_loop_sweep.sh [--shapes <csv>] [--flows <csv>]` | Serial closed-loop shape/dataflow sweep | PASS for `4x4,8x8,16x16,8x32` x `os,ws` on 2026-06-02 |
| `./run_all.sh closed_loop [args...]` | Unified wrapper for runtime closed-loop | PASS on tested images |

`./run_all.sh all` is a fast regression set and intentionally does not include the long runtime closed-loop test.

## Current Verified Results

| Flow | Command | Result | Prediction | Cycles |
|---|---|---|---|---:|
| Fast VGG e2e baseline | `./run_vgg_e2e.sh` | PASS on 2026-05-31 | cat/class 3 | 10,768,727 |
| Runtime closed-loop image2 | `./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg` | PASS | frog/class 6 | 114,014,769 |
| Runtime closed-loop image4 | `./run_all.sh closed_loop --image ./pic/test_cifar10_4.jpg` | PASS | dog/class 5 | 114,013,544 |
| Multi-core closed-loop image2, `NUM_CORES=1` | `./run_vgg_mc_closed_loop.sh --num-cores 1 --image pic/test_cifar10_2.jpg` | PASS | frog/class 6 | 114,014,769 |
| Multi-core closed-loop image2, `NUM_CORES=2` | `./run_vgg_mc_closed_loop.sh --num-cores 2 --image pic/test_cifar10_2.jpg` | PASS | frog/class 6 | 151,892,523 |
| Runtime closed-loop local image, 4x4 OS | `./run_vgg_closed_loop.sh --image <local-image> --shape 4x4 --flow os` | PASS on 2026-06-01 | automobile/class 1 | 160,809,527 |
| Runtime closed-loop local image, 4x4 WS | `./run_vgg_closed_loop.sh --image <local-image> --shape 4x4 --flow ws` | PASS on 2026-06-01 | automobile/class 1 | 160,809,527 |
| Runtime closed-loop full shape/dataflow sweep | `./run_vgg_closed_loop_sweep.sh --image <local-image>` | PASS on 2026-06-02, 8/8 cases | ship/class 8 | see sweep table below |
| Closed-loop generator sanity image3 | `python3 -B tools/pth/gen_vgg_closed_loop.py --out-dir /tmp/opencode/cl_test_3_fixed --image pic/test_cifar10_3.jpg` | PASS generation | fixed-runtime 6, exact-python 6 | N/A |

## Closed-Loop Shape/Dataflow Sweep

Result directory: `sim/vgg_closed_loop_sweep_20260602_103852/`.

| Shape | Flow | Result | Pred | Exact | Fixed | Cycles | Elapsed s |
|---|---|---|---:|---:|---:|---:|---:|
| `4x4` | `os` | PASS | 8 | 8 | 8 | 160,809,627 | 2,206 |
| `4x4` | `ws` | PASS | 8 | 8 | 8 | 160,809,627 | 2,195 |
| `8x8` | `os` | PASS | 8 | 8 | 8 | 128,708,571 | 1,775 |
| `8x8` | `ws` | PASS | 8 | 8 | 8 | 128,675,931 | 1,762 |
| `16x16` | `os` | PASS | 8 | 8 | 8 | 114,010,939 | 1,582 |
| `16x16` | `ws` | PASS | 8 | 8 | 8 | 113,998,459 | 1,573 |
| `8x32` | `os` | PASS | 8 | 8 | 8 | 130,858,011 | 1,829 |
| `8x32` | `ws` | PASS | 8 | 8 | 8 | 130,842,651 | 1,803 |

The prior `4x4` timeout was caused by the old 150M-cycle default limit, not by an NPU deadlock. The default closed-loop timeout is now 250M cycles.

## RTL Bring-Up Checks

| Command | Coverage | Result |
|---|---|---|
| `iverilog -g2012 -o /tmp/opencode/tb_npu_axi_lite_desc.vvp rtl/axi/npu_axi_lite.v tb/tb_npu_axi_lite_desc.v && vvp /tmp/opencode/tb_npu_axi_lite_desc.vvp` | AXI-Lite register file, delayed `BREADY`, IRQ, `ERR_STATUS` W1C | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_dma_read_burst.vvp rtl/common/fifo.v rtl/axi/npu_dma.v tb/tb_dma_read_burst.v && vvp /tmp/opencode/tb_dma_read_burst.vvp` | DMA read burst split, `RRESP`, read alignment errors | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_dma_write_burst.vvp rtl/common/fifo.v rtl/axi/npu_dma.v tb/tb_dma_write_burst.v && vvp /tmp/opencode/tb_dma_write_burst.vvp` | DMA write burst split, `BRESP`, write alignment errors | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_dma_burst.vvp rtl/common/fifo.v rtl/axi/npu_dma.v tb/tb_dma_burst.v && vvp /tmp/opencode/tb_dma_burst.vvp` | Mixed DMA read/write burst data | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_dma_perf.vvp rtl/common/fifo.v rtl/axi/npu_dma.v tb/tb_dma_perf.v && vvp /tmp/opencode/tb_dma_perf.vvp` | DMA read/write utilization smoke | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_npu_ctrl_error_status.vvp rtl/ctrl/npu_ctrl.v tb/tb_npu_ctrl_error_status.v && vvp /tmp/opencode/tb_npu_ctrl_error_status.vvp` | Descriptor errors, direct invalid dimension, DMA error latching | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_npu_ctrl_tile.vvp rtl/ctrl/npu_ctrl.v tb/tb_npu_ctrl_tile.v && vvp /tmp/opencode/tb_npu_ctrl_tile.vvp` | Tile controller edge writeback | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_npu_ctrl_ksplit.vvp rtl/ctrl/npu_ctrl.v tb/tb_npu_ctrl_ksplit.v && vvp /tmp/opencode/tb_npu_ctrl_ksplit.vvp` | Tile K-split sequencing | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_npu_ctrl_dataflow_modes.vvp rtl/ctrl/npu_ctrl.v tb/tb_npu_ctrl_dataflow_modes.v && vvp /tmp/opencode/tb_npu_ctrl_dataflow_modes.vvp` | Direct OS/WS controller branches | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_reconfig_pe_acc_init.vvp rtl/pe/pe_top.v rtl/array/reconfig_pe_array.v tb/tb_reconfig_pe_acc_init.v && vvp /tmp/opencode/tb_reconfig_pe_acc_init.vvp` | PE-array accumulator init, continued MAC, and row CE gating | PASS |
| `iverilog -g2012 -o /tmp/opencode/tb_reconfig_pe_8x32.vvp rtl/pe/pe_top.v rtl/array/reconfig_pe_array.v tb/tb_reconfig_pe_8x32.v && vvp /tmp/opencode/tb_reconfig_pe_8x32.vvp` | 8x32 folded PE-array mapping and WS row wrap with CE ports tied on | PASS |
| `iverilog -g2012 -s npu_top -o /tmp/opencode/npu_top_elab.vvp rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/top/npu_top.v` | `npu_top` elaboration | PASS |
| `iverilog -g2012 -s npu_pynq_wrapper -o /tmp/opencode/npu_pynq_wrapper_elab.vvp rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/top/*.v` | PYNQ wrapper elaboration | PASS |
| `tb/tile4/run_verilator.sh --shape 16x16 --M 16 --K 4 --N 16 --bias` | 16x16 tile Icarus + Verilator smoke | PASS |
| `tb/tile4/run_verilator.sh --all --icarus --lanes {1,2,4}` | Tile GEMM lanes 1/2/4 across 4x4, 8x8, 16x16, 8x32, bias, and K-split smokes | PASS |
| `tb/tile4/run_verilator.sh --verilator --shape 16x16 --M 16 --K 5 --N 16 --lanes {1,2,4}` | Verilator cross-check for lanes 1/2/4 with non-multiple K | PASS |
| `verilator --lint-only -Wall -Wno-fatal --top-module npu_pynq_wrapper rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/top/*.v` | PYNQ wrapper lint | Completes with existing width/unused warnings |

CE integration was smoke-tested on 2026-06-16 with `CG_EN=0` default compatibility, PE-array CE ports tied on in direct tests, top/PYNQ elaboration, AXI-Lite/DMA/controller tests, and the 16x16 tile Icarus + Verilator flow. The long closed-loop sweep was not rerun for this CE-only change; the latest full closed-loop sweep remains the 2026-06-02 result above.

## Multi-Core NPU Plan Checks

These checks apply to the multi-core planning work under `plan/`. They do not
replace the maintained single-core VGG entry points above.

| Command | Coverage | Result |
|---|---|---|
| `python3 -B -m py_compile tools/pth/gen_vgg_closed_loop.py` | Generator syntax after global `n_tile` postprocess fix | PASS |
| `python3 -B tools/pth/gen_vgg_closed_loop.py --out-dir /tmp/opencode/vgg_mc_fixed --image pic/test_cifar10_2.jpg --shape 16x16 --flow os --lanes 4 --timeout-cycles 500000000 --num-cores 2` | 2-core closed-loop asset generation | PASS generation; exact-python pred 6 |
| `iverilog -g2012 -o /tmp/opencode/tb_axi_lite_mc_bridge.vvp rtl/soc/axi_lite_mc_bridge.v tb/tb_axi_lite_mc_bridge.v && vvp /tmp/opencode/tb_axi_lite_mc_bridge.vvp` | Multi-core AXI-Lite decode and invalid window | PASS, 8 checks |
| `iverilog -g2012 -o /tmp/opencode/tb_dram_multi_port.vvp rtl/soc/dram_multi_port.v tb/tb_dram_multi_port.v && vvp /tmp/opencode/tb_dram_multi_port.vvp` | Shared multi-port DRAM model | PASS, 9 checks |
| `iverilog -g2012 -o /tmp/opencode/tb_npu_mc_top_smoke.vvp rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/top/*.v tb/tb_npu_mc_top_smoke.v && vvp /tmp/opencode/tb_npu_mc_top_smoke.vvp` | NPU wrapper independence and error isolation | PASS, 8 checks |
| `iverilog -g2012 -o /tmp/opencode/tb_soc_mc_mmio.vvp sim/picorv32.v rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/soc/*.v rtl/top/*.v tb/tb_soc_mc_mmio.v && vvp /tmp/opencode/tb_soc_mc_mmio.vvp` | PicoRV32 multi-core MMIO access | PASS, 52 cycles |
| `iverilog -g2012 -o /tmp/opencode/tb_soc_mc_shared_a.vvp sim/picorv32.v rtl/pe/*.v rtl/common/*.v rtl/buf/*.v rtl/array/*.v rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v rtl/soc/*.v rtl/top/*.v tb/tb_soc_mc_shared_a.v && vvp /tmp/opencode/tb_soc_mc_shared_a.vvp` | Shared `A_WORK`, per-core `R_WORK` smoke | PASS, 1109 cycles |
| `verilator --lint-only --timing -DMC_HEARTBEAT_INTERVAL=1000000 -I/tmp/opencode/vgg_mc_fixed --top-module tb_mc_heart ... tb/tb_mc_heart.v` | 2-core heartbeat testbench lint after `NUM_CORES=1` compatibility update | PASS with existing warning suppressions |
| `./run_vgg_mc_closed_loop.sh --num-cores 1 --image pic/test_cifar10_2.jpg` | Multi-core SoC wrapper, single-core VGG baseline | PASS, frog/class 6, 114,014,769 cycles |
| `./run_vgg_mc_closed_loop.sh --num-cores 2 --image pic/test_cifar10_2.jpg` | Full 2-core closed-loop VGG | PASS, frog/class 6, 151,892,523 cycles |

Short-run diagnostic throughput samples on this host:

| Case | Observation |
|---|---:|
| `soc_top` single-core diagnostic heartbeat | about 222K cycles/sec |
| `soc_mc_top NUM_CORES=1` low-output heartbeat | about 222K cycles/sec |
| `soc_mc_top NUM_CORES=2` low-output heartbeat | about 89K cycles/sec |

These samples are not full-run performance results. They show that the older 42x
slowdown claim is not reproduced by the current short-run setup. Full image2 VGG
classification now passes with `NUM_CORES=2`, but it is slower than the
`NUM_CORES=1` baseline because the current multi-core firmware is still dominated
by serial PicoRV32 scheduling, packing, requant, and scatter work.

## Syntax Checks

| Command | Result |
|---|---|
| `python3 -B -m py_compile tools/pth/gen_vgg_closed_loop.py` | PASS |
| `bash -n run_vgg_closed_loop.sh run_vgg_closed_loop_sweep.sh` | PASS |
| `bash -n run_all.sh` | PASS |
| `tclsh scripts/create_pynq_z2_npu_project.tcl` | PASS syntax-only load; Vivado execution not run in this shell |

## Python Environment

The current environment has CPU PyTorch installed for the VGG generators:

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Observed package: `torch-2.12.0+cpu` for Python `3.13.13`.

## What Is Being Verified

### Fast E2E Baseline

The e2e flow verifies that the NPU can execute a 1024-tile, 9-Conv RepOpt VGG chain using pre-generated Conv A tiles. The CPU firmware performs avgpool, classifier, and argmax after the NPU tile chain.

This is the fast regression baseline. It is not a complete runtime layer-to-layer deployment because Python pre-generates all Conv A tile streams.

### Runtime Closed Loop

The closed-loop flow verifies that firmware can perform the runtime steps needed for deployment:

- receive/use a dense INT8 input activation buffer
- pack Conv A tiles at runtime
- run NPU tile GEMM with raw INT32+bias result output
- apply ReLU and per-output-channel Q24 requant on the CPU
- scatter INT8 bytes back to dense activation buffers
- run maxpool, avgpool, classifier, and argmax

The current closed-loop validation target is exact Python model output. The older per-16-channel hardware `QUANT_CFG` approximation is no longer used as the PASS criterion.

The default runtime closed-loop shape is `16x16`. The generator and script accept `4x4`, `8x8`, `16x16`, and `8x32`. The `4x4` shape is slower in the current firmware-driven Verilator path and needs a timeout above 150M cycles; `run_vgg_closed_loop.sh` defaults to 250M cycles.

## Deployment Status

The RTL SoC is still a simulation SoC. It has PicoRV32, SRAM, DRAM model, and NPU. The current board target is PYNQ-Z2, with PS ARM as the runtime CPU and the NPU in PL.

The deployment path is documented in `doc/pynq_z2_deployment.md`:

- PS ARM programs the PL NPU and accesses DDR buffers.
- Host sends one preprocessed 3x32x32 INT8 image per inference.
- Each image returns one class plus raw performance counters.
- TOPS and bus utilization are computed from raw counters on the host.

## Regression Notes

- Generated artifacts under `sim/vgg_e2e/` and `sim/vgg_closed_loop/` are not source files.
- `RepOpt/` and image/generated data should not be committed as source changes.
- The untracked `session` file is not part of the project source.
