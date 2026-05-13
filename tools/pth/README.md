# PTH Conversion Tools

Host-side utilities for turning constrained PyTorch checkpoints into NPU-facing assets and CPU-scheduled inference plans.

## Dependency

The converter runs on the host PC, not on the reference CPU. It needs PyTorch only to read `.pth` files:

```powershell
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

PyTorch `2.11.0+cpu` failed to import in the current Windows Anaconda environment with a `c10.dll` initialization error; `2.5.1+cpu` imported successfully.

## RepOpt VGG INT8 Probe

```powershell
python tools\pth\pth_to_npu_assets.py `
  --pth .06_RepOpt_VGG\06_RepOpt_VGG\runs\cifar10_repopt_vgglike_qat\qat_int8_quantized.pth `
  --spec tools\pth\examples\repopt_vgg_int8_spec.json `
  --out-dir sim\pth_repopt_probe `
  --mode OS
```

The converter emits:

```text
model_plan.json              # CPU/NPU layer plan
checkpoint_inventory.json    # checkpoint key/type/shape inventory
summary.txt                  # concise conversion summary
assets/*_w_col.hex           # NPU W_col INT8 weights
assets/*_bias_int32.hex      # NPU accumulator-unit bias
assets/*_linear_w_int8.hex   # CPU Linear INT8 weights, when present
assets/*_linear_bias_int32.hex # CPU Linear accumulator-unit bias, when present
```

Current result for `qat_int8_quantized.pth`:

```text
layers      : 15
conv layers : 9
warnings    : 11
```

All 9 Conv layers are convertible to NPU direct Conv2D assets, but their exact PyTorch INT8 requantization is per output channel. The current NPU `QUANT_CFG` is one scale/shift per layer, so V1 must run:

```text
NPU: Conv2D + bias + ReLU
CPU: per-channel requant + NCHW repack + pooling + classifier
```

## CPU Runtime Descriptor Generation

After `model_plan.json` exists:

```powershell
python tools\pth\gen_cpu_runtime.py `
  --plan sim\pth_repopt_probe\model_plan.json `
  --out-dir sim\pth_repopt_probe\cpu_runtime
```

Generated files:

```text
model_plan_generated.h       # layer table and fixed-point requant constants
npu_pth_runtime.h            # header-only CPU/NPU helper runtime
runtime_smoke.c              # include smoke file
runtime_summary.json         # memory-fit report
```

For the current RepOpt VGG probe, generation succeeds but the full model is not
SoC-runnable with the default testbench memory:

```text
asset bytes       : 5,287,424
current DRAM bytes: 61,440
assets fit DRAM   : false
```

So this branch now has the conversion/runtime path, but the given RepOpt model
still requires either a much larger DRAM configuration and loader, or a smaller
checkpoint for the present `tb_soc.v` memory size.

## Tiny SoC Smoke

The default SoC DRAM can run a deliberately small `.pth` case:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pth_tiny_conv_soc.ps1
```

This script:

1. Generates `sim\pth_tiny_conv\tiny_conv_int8.pth`.
2. Converts it through `pth_to_npu_assets.py`.
3. Emits a DRAM image and RV32I firmware hex.
4. Runs `tb\tb_soc_pth_tiny_conv.v`.

Current passing result:

```text
[PASS] PTH tiny Conv SoC test PASSED!
R = [10, 2, 8, 2, 9, 0, 13, 0]
```

This proves the small end-to-end path:

```text
.pth -> host converter -> DRAM assets + CPU firmware -> CPU MMIO schedule -> NPU Conv2D/ReLU -> CPU-visible result
```

## 3-Layer Small Model Smoke

For a slightly more realistic CPU/NPU split, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pth_multilayer_soc.ps1
```

This generates and runs:

```text
Conv3x3(1->1) + ReLU
CPU repack: row-major int32 OFM -> NCHW int8 IFM
Conv1x1(1->2) + ReLU
CPU repack: row-major int32 OFM -> NCHW int8 IFM
Conv1x1(2->1) + ReLU
```

Current passing result:

```text
[PASS] PTH multilayer Conv SoC test PASSED!
R = [28, 22, 25, 37]
```

This smoke covers multi-layer CPU scheduling and the required layer-to-layer
CPU repack step, while staying inside the default SoC DRAM.

## RepOpt VGG Host Run

The RepOpt int8 checkpoint uses signed qint8 activations. Stock PyTorch CPU
quantized Conv expects quint8 activations, so the first full-network run uses a
host-side interpreter for the planned CPU/NPU split:

```powershell
python tools\pth\run_repopt_vgg_host.py `
  --index 0 `
  --out-json sim\pth_repopt_host_run\host_run_idx0.json
```

An arbitrary RGB image can use the same entry point. The image is resized to
`32x32`, normalized with the CIFAR-10 training mean/std, quantized to qint8,
and then classified into the CIFAR-10 classes:

```powershell
python tools\pth\run_repopt_vgg_host.py `
  --image path\to\image.png `
  --conv-backend tile4 `
  --out-json sim\pth_repopt_host_run\image_tile4.json
```

`--conv-backend direct` models each Conv with a full accumulator kernel.
`--conv-backend tile4` runs a software 4x4 Conv-as-GEMM tile scheduler for all
Conv layers, then performs CPU bias/ReLU, per-channel requant, pooling, and
classifier stages.

Current CIFAR-10 test samples:

```text
index 0: true=cat  pred=cat  logits_int8=-14 -8 -3 96 -28 9 6 -11 -17 -30
index 1: true=ship pred=ship logits_int8=7 16 -17 -14 -22 -21 -16 -20 92 -6
index 2: true=ship pred=ship logits_int8=8 52 -16 -16 -16 -24 -16 -20 59 -11
```

This confirms the full RepOpt graph can run in the V1 software split:

```text
NPU-modeled: Conv2D int8*int8 -> int32 accumulator + bias + ReLU
CPU-modeled: per-channel requant, NCHW repack, MaxPool, AvgPool, Flatten, Linear
```

Current image/tile scheduler smoke uses a PNG exported from CIFAR-10 test
sample 0:

```text
image_idx0_direct: pred=cat
image_idx0_tile4 : pred=cat
tile4 full-network conv tile counts:
  4096, 4096, 2048, 2048, 1024, 1024, 1024, 512, 512
```

## RepOpt VGG Staged RTL Conv Case

The first direct Conv2D RTL step uses the real RepOpt checkpoint, real
CIFAR-10 input, and the existing `npu_top` direct Conv2D path. The default
command verifies a top-left `4x4` output window of `stage1_0_conv`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_layer_case.ps1 -Index 0
```

Current passing result:

```text
layer=stage1_0_conv sample_index=0 cifar_label=3
output_window=4x4 of full 32x32
GEMM M=16 K=27 N=64 results=1024
ALL 1024 CHECKS PASSED
```

This compares the NPU-written int32 accumulator after bias and ReLU, before
the CPU per-channel requant step. Use `-TileOH 0 -TileOW 0` to request the full
`32x32` layer, but that is slow in the current scalar/1x1 Verilog simulation.

For actual `ARR_CFG[7]` 4x4 tile-mode verification, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_case.ps1 `
  -Index 0 -MBase 0 -NBase 0
```

Current passing tile-mode results:

```text
repopt_stage1_0_conv_tile4_m0_n0_idx0 : ALL 16 CHECKS PASSED
repopt_stage1_0_conv_tile4_m33_n4_idx0: ALL 16 CHECKS PASSED
```

This path pre-packs one local Conv-as-GEMM tile as `A_TILE[k][r]` and
`W_TILE[k][c]`, then compares raw int32 MAC accumulators. The current tile
writeback path does not yet apply direct-scalar bias/ReLU; that remains a
separate post-processing step for now.

## RepOpt RTL Tile Window + CPU Postprocess

To schedule multiple real RTL tile runs and stitch their outputs back into a
partial layer window:

```powershell
python tools\pth\run_repopt_layer_tile_rtl.py `
  --index 0 `
  --m-base 0 --n-base 0 `
  --m-tiles 2 --n-tiles 2 `
  --out-json sim\pth_repopt_tile_rtl\stage1_0_m0_n0_2x2.json
```

Current result:

```text
layer=stage1_0_conv sample_index=0 window=M[0:8) N[0:8)
tiles_run=4 raw_min=-19045 raw_max=24423
q_min=0 q_max=24
```

This runs four `ARR_CFG[7]` RTL tile cases, collects `npu_output.hex` from each
case, stitches an `8x8` raw accumulator window, then performs CPU
`bias/ReLU/per-channel requant`. The resulting qint8 window has been compared
against the host first-layer golden with zero mismatches.

## RepOpt Two-Layer SoC Window

To exercise the first real two-conv chain in SoC RTL:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_two_layer_soc.ps1 `
  -Index 0 -MBase2 0 -NBase2 0 -MTiles2 1 -NTiles2 1
```

Current passing result:

```text
[PASS] RepOpt two-layer SoC test PASSED!
stage1_0 tiles: 64 or 96 depending on the requested stage1_1 dependency window
stage1_1 tiles: passing windows now include M[0:4) N[0:4), M[0:8) N[0:8), and M[32:36) N[4:12)
```

This flow runs:

```text
stage1_0_conv selected dependency tiles -> CPU bias/ReLU/requant -> NCHW repack
-> CPU builds stage1_1 tile input from repacked IFM -> stage1_1_conv RTL tile
-> CPU bias/ReLU/per-channel requant -> compare second-layer q output
```

## RepOpt SoC Tile MMIO Scheduling + CPU Postprocess

To move tile scheduling and first-layer postprocess into the reference CPU
firmware, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 `
  -Index 0 -MBase 0 -NBase 0 -MTiles 2 -NTiles 2
```

This generates one DRAM image and one RV32I firmware image. During the single
RTL simulation, the reference CPU writes NPU MMIO registers for each tile:

```text
M_DIM=4, N_DIM=4, K_DIM=27
ARR_CFG=0x80, CFG_SHAPE=0
CTRL=0x11
```

After all tiles finish, the firmware reads the raw int32 MAC window from DRAM,
adds the real RepOpt bias, applies ReLU, runs per-channel fixed-point requant,
and writes the qint8 window back to DRAM. The testbench verifies both raw MAC
results and the firmware postprocess q results.

Current passing result:

```text
[PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!
Cycles: 8743
window: M[0:8) N[0:8)
tiles scheduled by CPU: 4
first tile result[0] = 7383
first postprocess q[0] = 11
```

The firmware uses nested loops instead of per-tile instruction expansion, so
the same path can run the complete first layer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 `
  -Index 0 -FullLayer
```

Current full-layer passing result:

```text
[PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!
Cycles: 8851650
window: M[0:1024) N[0:64)
tiles scheduled by CPU: 4096
first tile result[0] = 7383
first postprocess q[0] = 11
firmware words: 91
q_base: 0x00049500, q_count: 65536
marker: 0x00089500, dram_words: 141312
```

Use `-CompileOnly` with the same command when only generation and Verilog
compile checking are needed. The full run is intentionally not the default
smoke because it schedules 4096 NPU tile jobs and takes a long time under
Icarus Verilog.
