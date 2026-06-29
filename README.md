# NPU OpenAI Lab Prototype

Version: v4.2.0 - multi-core implemented.

Current state: Linux + Verilator simulation of a PicoRV32-controlled NPU SoC for RepOpt VGG-like CIFAR-10 inference. Supports single-core and multi-core configurations (`--num-cores 1|2|4`; 1/2-core VGG verified), DFS clock-enable throttling, and INT8 SIMD lane config (1/2/4).

## What Works Now

| Flow | Entry point | Purpose | Current status |
|---|---|---|---|
| Fast VGG e2e | `./run_vgg_e2e.sh` or `./run_all.sh standard` | Python pre-generates all Conv A tiles; CPU firmware runs 1024 NPU tiles plus avgpool/classifier/argmax | PASS, cat/class 3, 10,768,727 cycles |
| Runtime closed-loop VGG (1-core) | `./run_vgg_closed_loop.sh` | CPU firmware packs A tiles at runtime, runs NPU, performs per-channel requant/scatter, maxpool, avgpool, classifier | PASS, about 114M cycles for default `16x16` |
| Runtime closed-loop VGG (multi-core) | `./run_vgg_closed_loop.sh --num-cores 2` | Same as above, distributed across multiple NPU cores via N-tile split | PASS, 1-core ≈114M, 2-core ≈110M |
| DFS (clock-enable throttling) | `--clk-div 0|1|2|3` | CE-based compute rate control (1x, 1/2, 1/4, 1/8) | PASS, tile golden-check + VGG closed-loop |
| Multi-dimensional sweep | `./run_vgg_closed_loop_sweep.sh` | Sweep shapes×flows×lanes×cores×clk_div×PPB depth | Supported |
| Arbitrary image e2e | `./run_all.sh image <file>` | Classify an image after host resize/normalize/quantize | Supported |

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

Or run the setup script to check and auto-install everything:

```bash
bash setup.sh          # check mode: prints which items are missing
bash setup.sh --install # auto-installs missing Python, Icarus, Verilator, and model files
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
|---|---|---|---|
| `RepOpt/06_RepOpt_VGG/runs/…/qat_int8_quantized.pth` | Model checkpoint (~5 MB), gitignored | Download from GitHub Releases (see step 3) |
| `sim/pth_repopt_probe/model_plan.json` | Generated from the checkpoint | Run `pth_to_npu_assets.py` (see step 3) |

The repo does **not** ship the trained RepOpt VGG checkpoint because it is a ~5 MB binary file hosted via GitHub Releases. The tiny‑conv smoke test (`tb_soc_pth_tiny_conv.v`) generates its own small checkpoint on the fly and does **not** require an external `.pth` file — use that for a first smoke after cloning.

## Full Setup (First Clone)

### 1. Clone and install prerequisites

```bash
git clone git@github.com:demonk69/726_NPU.git
cd 726_NPU

bash setup.sh --install
```

This installs missing system packages (Icarus, Verilator from source), Python packages (PyTorch CPU, Pillow), downloads the model checkpoint from GitHub Releases, and generates `model_plan.json` — all in one step.

Run `bash setup.sh` without `--install` to check prerequisites only — it prints a status table showing which items are present and which are missing, without modifying anything.

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

### 3. Run the main VGG flows

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

The default closed-loop shape is `16x16`. The run script and generator also support `4x4`, `8x8`, and `8x32` shape selection. Use `run_vgg_closed_loop_sweep.sh` for serial sweeps across shapes, flows, lanes, core count, CLK_DIV divisor, and PPB depth.

### Multi-Core Flow

`./run_vgg_closed_loop.sh --num-cores N` distributes tiles across multiple NPU cores via N-tile splitting. Accepts `--num-cores 1|2|4`, with the same shape/flow/lanes/clk-div options as the single-core runner. This is the maintained multi-core entry point as of v4.2.0.

### DFS (Clock-Enable Throttling)

CE-based compute rate control via `--clk-div 0|1|2|3` (1x, 1/2, 1/4, 1/8). DMA and AXI-Lite run at full speed; only the PE array and controller compute state are throttled. Default `0` maintains backward compatibility.

### PPB Depth Sweeps

`--ppb-depth <words>` changes the W/A ping-pong buffer depth used by the VGG closed-loop testbench and generator. The generator recomputes `KT_ELEMS` from the same depth, so RTL buffer capacity and firmware K-splitting stay matched.

### Sweep

```bash
# lanes=4, all cores/shapes/flows
./run_vgg_closed_loop_sweep.sh --lanes 4 --num-cores 1,2

# DFS sweep on 16x16/os
./run_vgg_closed_loop_sweep.sh --shapes 16x16 --flows os --clk-divs 0,1,2,3

# Buffer-depth sweep on 16x16/os/lanes=4
./run_vgg_closed_loop_sweep.sh --shapes 16x16 --flows os --lanes 4 --ppb-depths 1024,4096,8192
```

## Documentation Map

| Document | Purpose |
|---|---|
| `doc/verification_status.md` | Current verified commands and results |
| `doc/architecture.md` | Current SoC/NPU architecture, DFS, multi-core |
| `doc/vgg_e2e_flow.md` | Fast e2e flow details |
| `doc/vgg_closed_loop_flow.md` | Runtime closed-loop flow details |
| `doc/known_issues.md` | Current limitations and risks |
| `doc/user_manual.md` | Register and firmware-facing ABI reference, DFS usage |
| `doc/rtl_reference.md` | RTL module reference, CE/DFS integration |
| `doc/conv_gemm_mapping.md` | Conv2D-to-GEMM mapping reference |
| `doc/archive/` | Historical plans and obsolete worklogs |

## Important Distinctions

- `run_vgg_e2e.sh` is the fast baseline, but Python pre-generates all Conv A tiles.
- `run_vgg_closed_loop.sh` is the runtime closed-loop path for both single-core and multi-core runs (`--num-cores 1|2|4`).
- DFS is CE-based; `--clk-div` works with all runners and sweep scripts.
- PPB depth is compile/testbench configuration, not an MMIO register; use `--ppb-depth`/`--ppb-depths` before generation.
- Do not use archived Windows/Icarus documents as current run instructions.
