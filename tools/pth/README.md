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
