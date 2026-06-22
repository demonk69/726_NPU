# NPU OpenAI Lab Prototype

Current state: Linux + Verilator simulation of a PicoRV32-controlled NPU SoC for RepOpt VGG-like CIFAR-10 inference.

This README describes the maintained flow. Older Windows/Icarus worklogs are archived under `doc/archive/` and should not be treated as current guidance. Obsolete plan documents have been removed.

## What Works Now

| Flow | Entry point | Purpose | Current status |
|---|---|---|---|
| Fast VGG e2e | `./run_vgg_e2e.sh` or `./run_all.sh standard` | Python pre-generates all Conv A tiles; CPU firmware runs 1024 NPU tiles plus avgpool/classifier/argmax | PASS, cat/class 3, 10,768,727 cycles |
| Runtime closed-loop VGG | `./run_vgg_closed_loop.sh` or `./run_all.sh closed_loop` | CPU firmware packs A tiles at runtime, runs NPU, performs per-channel requant/scatter, maxpool, avgpool, classifier | PASS on tested images, about 114M cycles for default `16x16`; about 161M cycles for `4x4` |
| Arbitrary image e2e | `./run_all.sh image <file>` | Classify an image after host resize/normalize/quantize | Supported |

The closed-loop flow is the path closest to an FPGA deployment: model assets are static, while each inference only needs a new 3x32x32 INT8 input image.

## Prerequisites

| Tool | Version / note |
|---|---|
| Linux (bash) | Required. Windows WSL works if Verilator is installed. |
| Python 3 | 3.10+ recommended. |
| PyTorch (CPU) | 2.x. See install command below. |
| Pillow | For `--image` input. |
| Verilator | 5.x recommended. 4.x may work. |
| Icarus Verilog | 11.0+ for `iverilog` / `vvp`. |

Install Python dependencies:

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install Pillow
```

Ensure Verilator and Icarus are on `$PATH`:

```bash
which verilator iverilog vvp
```

## Repository Layout

```
726_NPU_openai/
├── rtl/                  # Verilog RTL (PE array, ctrl, DMA, PPB, SoC…)
├── tb/                   # Testbenches
├── tools/pth/            # PyTorch model converters and generators
├── scripts/              # Vivado Tcl and helper scripts
├── sim/                  # Generated simulation assets (gitignored except picorv32.v)
│   ├── picorv32.v        # PicoRV32 RISC‑V CPU core
│   └── pth_repopt_probe/ # model_plan.json (generated, see below)
├── run_vgg_e2e.sh        # Fast e2e flow
├── run_vgg_closed_loop.sh# Runtime closed‑loop flow
└── run_all.sh            # Unified runner
```

Files that are **not** tracked in git and must be obtained or generated:

| File | Why missing | How to get it |
|---|---|---|
| `RepOpt/06_RepOpt_VGG/runs/…/qat_int8_quantized.pth` | Large model checkpoint, gitignored | Download from shared storage or re‑train |
| `sim/pth_repopt_probe/model_plan.json` | Generated from the checkpoint | Run `pth_to_npu_assets.py` (see below) |

The repo does **not** ship the trained RepOpt VGG checkpoint because it is ~50–100 MB of binary weight data. The tiny‑conv smoke test (`tb_soc_pth_tiny_conv.v`) generates its own small checkpoint on the fly and does **not** require an external `.pth` file — use that for a first smoke after cloning.

## Full Setup (First Clone)

### 1. Clone the repo

```bash
git clone git@github.com:demonk69/726_NPU.git
cd 726_NPU
```

### 2. Run a self‑contained smoke (no external model needed)

This test creates its own tiny checkpoint and validates the entire SoC flow:

```bash
iverilog -g2012 -o /tmp/tb_tiny.vvp \
    sim/picorv32.v rtl/pe/*.v rtl/common/*.v rtl/buf/*.v \
    rtl/array/*.v rtl/axi/*.v rtl/ctrl/*.v rtl/power/*.v \
    rtl/soc/*.v rtl/top/*.v tb/tb_soc_pth_tiny_conv.v

python3 tools/pth/gen_tiny_conv_soc_case.py --out-dir sim/pth_tiny_conv
vvp -M. -I sim/pth_tiny_conv /tmp/tb_tiny.vvp
```

Expected output: `PASS` with classification result.

### 3. Run the main VGG flows (requires model checkpoint)

First obtain the model checkpoint. Place it at the expected default path:

```text
RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth
```

Alternatively pass `--pth <path>` to the runner scripts.

Generate the model plan from the checkpoint:

```bash
python3 tools/pth/pth_to_npu_assets.py \
    --pth RepOpt/06_RepOpt_VGG/runs/.../qat_int8_quantized.pth \
    --out-dir sim/pth_repopt_probe
```

This creates `sim/pth_repopt_probe/model_plan.json` and model asset hex files.

### 4. Quick Start

Run the verified fast baseline:

```bash
./run_vgg_e2e.sh
```

Run the unified entry point:

```bash
./run_all.sh standard 0
./run_all.sh image ./pic/test_cifar10_2.jpg
./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg
./run_all.sh closed_loop --shape 8x8 --image ./pic/test_cifar10_2.jpg
./run_vgg_closed_loop_sweep.sh --shapes 4x4,8x8 --flows os,ws
```

Run the full fast regression set:

```bash
./run_all.sh all
```

`./run_all.sh all` intentionally does not run the runtime closed-loop flow because it takes much longer than the fast baseline.

## Running Custom GEMM Cycle Sweeps

```bash
python3 tools/gemm_dataset/run_custom_gemm_cycle_sweep.py --mnk 64,128,256
```

This sweeps `M=64,K=128,N=256` across flows `os,ws` × shapes `4x4,8x8,16x16,8x32` × lanes `1,2,4`. Output goes to `data/gemm_cycles/custom_sweep_<timestamp>/` (summary CSV/TSV/Markdown + per-case logs).

Key options:
- `--mnk M,K,N` — repeatable for multiple points
- `--sim verilator` — Verilator mode (much faster, limited to `PERF_ONLY`)
- `--flows os` — single flow
- `--shapes 16x16` — single shape
- `--lanes 4` — single lane count
- `--timeout-sec 600` — per-case shell timeout
- `--continue-on-fail` — run all cases even on failure

Requires Verilator or Icarus and the usual Python environment; no external model needed.

## VGG Flow Summary

### Fast E2E Flow

`tools/pth/gen_vgg_e2e.py` generates a DRAM image containing all pre-packed A tiles for all 9 Conv layers. Firmware iterates a 1024-entry tile table, starts the NPU for each tile, then performs avgpool, classifier, and argmax on the CPU.

Use this for quick regression and baseline confidence.

### Runtime Closed-Loop Flow

`tools/pth/gen_vgg_closed_loop.py` generates static model assets and the selected input image. Firmware does the real runtime work:

- pack A tiles from dense activation buffers
- configure and run the NPU tile by tile
- read raw INT32+bias tile results
- apply ReLU and per-output-channel Q24 requant on the CPU
- scatter INT8 bytes back to dense activation buffers
- run maxpool, avgpool, classifier, and argmax

This flow avoids the old per-16-channel hardware `QUANT_CFG` approximation and validates against the exact Python model target.

The default closed-loop shape is `16x16`. The run script and generator also support `4x4`, `8x8`, and `8x32` shape selection; new tile/packed-SIMD work should keep those modes shape-aware. Use `run_vgg_closed_loop_sweep.sh` for serial OS/WS shape sweeps.

## FPGA Deployment Direction

The current board target is PYNQ-Z2. The primary route uses the Zynq PS ARM as the runtime CPU and keeps the NPU in PL:

- PC/host sends one preprocessed 3x32x32 INT8 image per inference.
- PS ARM writes image/model/runtime buffers in DDR and programs the PL NPU.
- PL NPU runs tile GEMM and exposes raw performance counters.
- PS/host returns one class plus raw counters; TOPS and bus utilization are computed on the host.

See `doc/pynq_z2_deployment.md` for the current deployment plan.

## Documentation Map

| Document | Purpose |
|---|---|
| `doc/verification_status.md` | Current verified commands and results |
| `doc/architecture.md` | Current SoC/NPU architecture and address map |
| `doc/vgg_e2e_flow.md` | Fast e2e flow details |
| `doc/vgg_closed_loop_flow.md` | Runtime closed-loop flow details |
| `doc/pynq_z2_deployment.md` | PYNQ-Z2 deployment and counter readback plan |
| `doc/known_issues.md` | Current limitations and risks |
| `doc/user_manual.md` | Register and firmware-facing ABI reference |
| `doc/rtl_reference.md` | RTL module reference |
| `doc/conv_gemm_mapping.md` | Conv2D-to-GEMM mapping reference |
| `doc/archive/` | Historical plans and obsolete worklogs |

## Important Distinctions

- `run_vgg_e2e.sh` is the fast baseline, but Python pre-generates all Conv A tiles.
- `run_vgg_closed_loop.sh` is the runtime closed-loop path and is the basis for FPGA I/O deployment.
- PYNQ-Z2 deployment uses PS ARM plus PL NPU as the primary route; pure-PL UART/SPI/Boot ROM bring-up is deferred.
- Do not use archived Windows/Icarus documents as current run instructions.
