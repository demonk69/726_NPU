# RepOpt Full SoC Inference Worklog

This is a temporary implementation log for making RepOpt VGG inference run as:

```text
image/input tensor -> reference CPU firmware -> NPU RTL tile jobs -> CPU firmware postprocess/repack/pool/linear -> class id
```

## Reuse Boundary

Changing to another `.pth` does not mean rebuilding the whole flow from zero.

Reusable for compatible static INT8 CNN checkpoints:

- RTL NPU datapath, AXI-Lite/MMIO path, SoC wrapper, DRAM model, and PicoRV32 reference CPU integration.
- Existing `.pth` host converter framework and model plan format.
- Tile-mode RTL testbench infrastructure.
- SoC tile scheduling script and generated firmware structure.
- CPU firmware primitives for MMIO tile launch, polling, bias, ReLU, fixed-point requant, and DRAM writeback.
- Image preprocessing entry on host, if the model still expects CIFAR-like `32x32` normalized input.

Regenerated per checkpoint/model:

- `model_plan.json` and checkpoint inventory.
- NPU weight tile/column assets.
- int32 bias and per-channel requant multipliers.
- DRAM image and address map.
- Expected raw/q golden files for RTL/testbench comparison.
- Firmware constants for layer count, shapes, addresses, scales, and classifier parameters.

Needs code changes when model structure changes:

- New operators beyond Conv/ReLU/Pool/Flatten/Linear.
- Different input preprocessing or class labels.
- Non-static shapes, grouped/depthwise conv, residual/add/concat, or unsupported quantization.
- Full model memory map if assets/buffers exceed current DRAM sizing.

## Current Confirmed State

- Host RepOpt VGG interpreter can classify images using the intended CPU/NPU split semantics.
- Real RTL tile-mode `ARR_CFG[7]` works for RepOpt first-layer tiles.
- Reference CPU firmware can schedule multiple NPU tile jobs in one SoC RTL simulation.
- CPU firmware can postprocess tile outputs with bias/ReLU/per-channel fixed-point requant.
- Full first-layer window passes RTL simulation:

```text
[PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!
Cycles: 8851650
tiles: 4096 window M[0:1024) N[0:64)
firmware words: 91
q_count: 65536
dram_words: 141312
```

## Definition Of Done

Full CPU+NPU(RTL) inference is done when one command can:

1. Load a given supported `.pth` and image/input tensor.
2. Generate all model assets and a SoC DRAM image.
3. Run CPU firmware in RTL simulation.
4. Let the firmware dispatch every Conv tile to NPU RTL.
5. Let CPU firmware handle requant, repack, pooling, flatten, and classifier.
6. Read back logits/class id.
7. Compare against host golden for the same input.

## Step Plan

### Step 1 - Full First Layer RTL Run

Status: completed.

Run record:

```text
started_at: 2026-05-08 19:45:35 +08:00
command   : powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 -Index 0 -FullLayer
pid_file  : sim\repopt_full_layer_step1.pid
stdout    : sim\repopt_full_layer_step1.log
stderr    : sim\repopt_full_layer_step1.err.log
state     : generation and compile completed; vvp was stopped before PASS/FAIL result
```

Manual terminal result reported on 2026-05-09:

```text
[PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!
Cycles: 8851650
window: M[0:1024) N[0:64)
tiles scheduled by CPU: 4096
first tile result[0] = 7383
first postprocess q[0] = 11
```

Command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 `
  -Index 0 -FullLayer
```

Acceptance:

- Done: 4096 tile jobs complete in one RTL simulation.
- Done: raw MAC surface and q surface both match generated golden.
- Done: no timeout.

### Step 2 - First Layer Output Repack

Status: pending.

Implement CPU firmware logic that converts first-layer q surface into the next layer input layout:

```text
tile/q surface -> NHWC logical view -> NCHW int8 IFM buffer
```

Acceptance:

- Repacked buffer matches host golden first-layer output tensor.
- Testbench can compare the IFM buffer for the next layer.

### Step 3 - Two RepOpt Conv Layers In SoC RTL

Status: pending.

Extend the SoC generator to emit assets and firmware for:

```text
stage1_0_conv -> CPU postprocess/repack -> stage1_1_conv
```

Acceptance:

- Both Conv layers run through NPU RTL tile jobs.
- Second-layer q output matches host golden.

### Step 4 - Pooling And Stage Boundary Support

Status: pending.

Add CPU firmware MaxPool handling at RepOpt stage boundaries.

Acceptance:

- First stage including pool matches host golden.

### Step 5 - Generic Layer Scheduler

Status: pending.

Replace first-layer-specific SoC generator logic with a layer loop derived from `model_plan.json`.

Acceptance:

- The same generator can emit firmware/DRAM for multiple Conv layers without hardcoded layer names.

### Step 6 - Full Model DRAM Planner And Loader

Status: pending.

Plan weights, intermediate buffers, pooling buffers, classifier weights, logits, and markers in one DRAM map.

Acceptance:

- Full RepOpt assets and buffers fit generated SoC DRAM.
- Address ranges do not overlap.

### Step 7 - Classifier And Final Result

Status: pending.

Implement CPU firmware AvgPool/Flatten/Linear/argmax or equivalent classifier path.

Acceptance:

- Final logits and predicted class match host golden.

### Step 8 - Image Input Command

Status: pending.

Add one command that accepts an image, performs host preprocessing, generates input tensor/DRAM, runs SoC RTL, and prints prediction.

Acceptance:

- Example image produces the same class as host golden.
