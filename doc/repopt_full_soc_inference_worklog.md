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
- CPU firmware can repack first-layer q outputs into the next-layer NCHW int8 IFM buffer.
- Full first-layer window passes RTL simulation:

```text
[PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!
Cycles: 8851650
tiles: 4096 window M[0:1024) N[0:64)
firmware words: 91
q_count: 65536
dram_words: 141312
```

## 阵列形态边界

这一节专门把三个容易混淆的概念拆开说明：

- 物理阵列规模：RTL 里实际实例化的是 `16x16` 个 PE。
- 逻辑 tile 形状：一次 tile-mode NPU 任务当前可按 `CFG_SHAPE` 选择 `4x4`、`8x8`、`16x16` 或 `8x32` 的活跃 lane 形状。
- 已验证的推理窗口：更大的层输出区域，可以通过调度多个较小 tile 拼出来。

### 当前分支状态

| Shape / path | 接手后状态 | 矩阵乘法 | 边沿检测 / visual CNN | 推理 | 备注 |
|---|---|---|---|---|---|
| `4x4` tile | 已处理 | 已处理：当前主分支可复现的 tile GEMM / SoC 调度主线 | 已处理：可通过 direct scalar Conv2D 可视化入口做边沿检测类任务，但不是 4x4 tile inference 主线 | 已处理：当前 RepOpt SoC inference 主线就是 `4x4` tile，已推进到 Step4 window 级验证 | 当前真正跑通的推理闭环 |
| `8x8` tile | 未处理 | 未处理：仓库历史文档声称 active lane feed 已验证，但本轮未复验，也未接入当前 SoC inference 主线 | 未处理：当前 visual CNN 入口不走 8x8 tile 路径 | 未处理：当前没有 8x8 tile 的 RepOpt SoC inference 闭环 | 只能引用历史文档，不作为本轮完成项 |
| `16x16` tile | 未处理 | 未处理：仓库历史文档声称 active lane feed 已验证，但完整 16x16 tile 结果收集/写回未在本轮复验 | 未处理：当前 visual CNN 入口不走 16x16 tile 路径 | 未处理：当前没有 16x16 tile 的 RepOpt SoC inference 闭环 | 我们已验证的 `16x16` 是 output window，不是 `16x16 tile` |
| `8x32` tile | 未处理 | 未处理：仓库历史文档只到阵列折叠路由 / 输出顺序级别，本轮未复验完整 GEMM | 未处理 | 未处理 | 当前离 inference 主线最远 |

### 单个 PE 内多 lane 说明

上面这些“阵列形态”验证，和“单个 PE 内部的多 lane SIMD”验证是两条不同的线。

- 历史仓库状态里有、但本轮没有重新复验的内容：`tb/tb_pe_top.v` 配合 `scripts/run_sim.ps1` / `scripts/run_regression.ps1`，覆盖了单个 PE 的 packed INT8 `2-lane` 和 `4-lane` SIMD，包含：
  - OS / WS 两种执行语义
  - 带负数 lane 的测试向量
  - 与旧 sign-extended scalar INT8 输入风格的兼容性
- 当前 RepOpt SoC inference 闭环仍然跑在现有 `DATA_W=16`、`INT8_SIMD_LANES=1` 的已验证路径上。所以即使物理阵列是 `16x16`，当前端到端已跑通的 RepOpt 推理，也还没有把“单个 PE 内 packed 多 lane SIMD”纳入已验证闭环。

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

Status: completed.

Implement CPU firmware logic that converts first-layer q surface into the next layer input layout:

```text
tile/q surface -> NHWC logical view -> NCHW int8 IFM buffer
```

Acceptance:

- Done: repacked buffer matches host golden first-layer output tensor in the generated SoC case.
- Done: testbench compares the next-layer IFM buffer.

### Step 3 - Two RepOpt Conv Layers In SoC RTL

Status: in progress.

Extend the SoC generator to emit assets and firmware for:

```text
stage1_0_conv -> CPU postprocess/repack -> stage1_1_conv
```

Acceptance:

- In progress: windowed `stage1_0_conv -> CPU repack -> stage1_1_conv` SoC RTL smoke is now passing for real stage1_1 windows including `M[0:8), N[0:8)` and `M[32:36), N[4:12)`.
- In progress: larger stage1_1 windows now also pass in SoC RTL, including `M[0:16), N[0:16)` and `M[128:136), N[8:24)`.
- Pending: extend the passing windowed flow to the remaining Step3 target coverage and decide whether to promote this step to full-layer stage1_1 execution before Step4.

### Step 4 - Pooling And Stage Boundary Support

Status: in progress.

Add CPU firmware MaxPool handling at RepOpt stage boundaries.

Acceptance:

- In progress: first-stage `stage1_0_conv -> stage1_1_conv -> maxpool` SoC RTL windows are now passing for pooled outputs including:
  - rows `[0:1)`, cols `[0:1)`, channels `[0:4)`
  - rows `[0:1)`, cols `[0:4)`, channels `[0:4)`
- Pending: extend the passing pool windows to wider stage coverage and then promote Step4 to completed.

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
