# 未解决问题清单

最后更新：2026-05-19 (v0.5.0)

---

## 一、已修复（原列于 2026-05-09 版本）

| 原编号 | 问题 | 修复方式 | 验证 |
|--------|------|----------|------|
| #1 | 后处理未接入 tile 路径 | `npu_top.v:957-964` — tile_bias_buf → tile_with_bias → apply_scalar_activation → apply_scalar_quant 全线已接 | 8×32 bias PASS, K-split bias PASS, RepOpt VGG PASS |
| #9 | 16×16 rows 6-11 数据偏移 | PPBuf lane 映射 + pipe 修复 | 16×16 全规格 256/256 PASS |
| #10 | 8×32 two-pass 权重调度 | `npu_ctrl.v` two-pass FSM + `reconfig_pe_array.v` half_en | 8×32 256/256 PASS (含 bias) |
| #13 | DATA_W=16 限制吞吐 | `soc_top.v` NPU_DATA_W 16→32, 4-lane INT8 SIMD | 全回归 19/20 PASS + SoC 2/2 PASS |
| K-split | K=21/17/18/19 失败 | `seq1_len_bytes_w` 补 SIMD padding + K-split bias race 修复 | K=17/18/19/21/40 全部 PASS |

---

## 二、设计决策（非 bug，架构选择）

### 1. 外部 PSUM surface 读写未接入 (原 #6)

**现状**：K-split 时 PE accumulator 跨 k_tile 保持（`pe_acc_init_en=0` in tile mode）。32-bit acc 可容纳 K≤576 的 INT8 内积累（max partial sum ~ 576×127² ≈ 9.3M < 2³¹），无需外部 DRAM PSUM surface。

**何时需要**：当 K > 2000 或支持 FP16/FP32 时，accumulator 可能溢出，需外部 PSUM。当前 INT8 模型（max K=576 for stage1_1_conv）不触发。

### 2. Tile-mode Conv2D descriptor 未实现 (原 #8)

**现状**：`OP=CONV2D_IM2COL` 在 descriptor v1 中已定义（op=2），controller 的 descriptor FSM 骨架存在。当前验证走 GPU-style 固件调度（PicoRV32 MMIO direct write），不使用 descriptor 链。两者等效——descriptor 路径省 CPU 指令但不影响正确性。

### 3. K-split 靠 PE accumulator 保持 (非 issue)

**验证**：16×16 K=576（36 k_tiles）已通过 NPU 验证（RepOpt VGG L1, L1 raw[-21498] matches PyTorch）。跨 k_tile 的 partial sum 正确积累。

---

## 三、功能缺失

### 4. CPU 固件未实现真实 512→10 Linear 分类器 (新)

**现状**：v0.5.0 端到端测试中，分类标签由 Python 预写入 DRAM，固件仅转抄至 marker。PicoRV32 上未实现完整的 512 特征 × 10 类别矩阵乘 + argmax。

**影响**：声称"硬件端到端分类"时不完整——Conv 推理链在硬件上，分类器在 Python 里。

**修复**：固件需实现循环结构（~200 条指令）完成 5120 MAC + 10 路 argmax。

### 5. 全模型 9 层未验证 (新)

**现状**：当前验证到 2 Conv 层 + MaxPool + classifier placeholder。完整 RepOpt VGG 需 9 Conv + 3 MaxPool + Flatten + Linear。

**修复**：扩展现有 `gen_vgg_e2e.py` 至全部 9 层，逐层 IFM→OFM repack 在 Python 预计算或固件循环中实现。

### 6. MaxPool / Flatten 固件仅 demonstration (新)

**现状**：MaxPool 2×2 固件仅比较 4 个值，未做完整的 16×16→8×8 下采样。Flatten 未在固件中实现。

### 7. FP16 未在 8x8/16x16/8x32 测试 (原 #2)

**现状**：4×4 FP16 tile GEMM 已验证，8×8/16×16/8×32 FP16 未测。`gen_multi_shape_data.py` 有 IS_FP16 支持。

### 8. WS 数据流无端到端 GEMM golden (原 #3)

**现状**：所有 golden 测试用 OS 模式。WS 模式 controller 可达 done，但未跑过真实 GEMM 验证。

### 9. 4×4 边界 tile 无端到端 golden (原 #4)

**现状**：边界 tile（M=5,N=6 等）通过 `gen_multi_shape_data.py` 生成可测，未跑。

### 10. 时钟门控/DFS 未接入 PE (原 #7)

**现状**：`npu_power.v` 输出悬空。

---

## 优先级

| P | 问题 | 理由 |
|----|------|------|
| P0 | #4 CPU Linear 分类器 | 完成"端到端分类"闭环 |
| P0 | #5 全模型 9 层 chain | 完整推理管线 |
| P1 | #6 MaxPool/Flatten 完整固件 | 层间数据流 |
| P2 | #7 FP16 扩展测试 | 验证即可 |
| P2 | #8 WS golden | 非当前主线 |
| P2 | #9 边界 tile golden | 低工作量 |
| P3 | #10 时钟门控 | 低功耗评测 |
