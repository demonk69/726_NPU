# 未解决问题清单

本文档整理 DeepSeek 修改会话（2026-05-09）中发现的全部未解决/未完成问题。

---

## 一、功能缺失（电路有逻辑，但未接入或未测试）

### 1. 后处理（bias/ReLU/ReLU6/quant/saturate）未接入 tile 路径

**现状**：bias（CTRL[9]）、ReLU/ReLU6（CTRL[11:10]）、INT8 quant/saturate（QUANT_CFG 0x9C）在 **direct scalar 路径**（`npu_top.v` 的 `scalar_post_result` 链）已实现并验证通过。但 **tile mode**（`ARR_CFG[7]=1`）的 serializer→result FIFO 路径直接输出 raw int32 MAC 累加器，未经过任何后处理。

**修复思路**：在 `npu_top.v` 的 serializer `tile_result_buf[tile_ser_idx]` → FIFO 之间插入与 scalar 路径相同的 bias/activation/quant 逻辑。需要 controller 在 tile mode 下也锁存 `lk_bias_en`/`lk_post_act_mode`/`lk_quant_cfg` 并下发给 `npu_top`。

**影响**：当前 tile mode 的 RepOpt VGG 第一层仿真必须由 CPU 固件做后处理，无法声称 NPU 硬件支持 Conv+ReLU+quant 的端到端推理。

---

### 2. FP16 未在 8x8/16x16/8x32 测试

**现状**：4x4 FP16 tile GEMM 在 `tb_npu_tile_gemm.v + fp16_4x4x4` 已验证。8x8/16x16/8x32 的 INT8 已通过，但 FP16 未测试。

**修复思路**：`gen_multi_shape_data.py` 和 `tb_npu_tile_gemm_wide.v` 有 `IS_FP16` 宏支持，只需生成 FP16 测试数据并运行。需注意 FP16 时 `vector_elem_bytes` 为 `2 * shape_lanes`（每 FP16 占 2 bytes，每 32-bit word 装 2 个 FP16）。

---

### 3. WS 数据流无端到端 GEMM golden 验证

**现状**：`tb_npu_ctrl_dataflow_modes.v` 验证了 WS controller 分支可达 done，但从未运行过一个真实的 WS tile GEMM golden 测试。当前所有 4x4/8x8 golden 测试用的都是 OS 模式。

**修复思路**：生成 WS tile-pack 测试数据（weight 驻留格式不同于 OS 的 W_TILE），编写 WS 专用 check 逻辑。工作量中等。

---

### 4. 4x4 边界 tile 无端到端 golden

**现状**：`tb_npu_ctrl_tile.v` 只检查了 controller 的 tile planner 输出（tile_m_base、tile_n_base、row/col mask），没有真正跑 GEMM 并验证结果。

**修复思路**：用已有的 `gen_multi_shape_data.py` 生成 M=5,N=6 等边界 case，通过 `tb_npu_tile_gemm_wide.v` 验证 golden。工作量低。

---

### 5. Descriptor 模式未测试 8x8/16x16/8x32

**现状**：`npu_ctrl.v` 的 descriptor shape 检查已放宽到 `<=4'd3`（接受 4x4/8x8/16x16/8x32），但没有对应的 descriptor 链式测试。`tb_npu_desc_two_layer.v` 和 `tb_npu_desc_ofm_chain.v` 仍使用 4x4 的 descriptor。

**修复思路**：编写 8x8 双 descriptor 测试（类似于现有的 4x4 descriptor 测试）。CPU firmware 需能组装 descriptor v1 格式。

---

### 6. 外部 PSUM surface 读写未接入

**现状**：K-split 的 tile 内部 PSUM 累计（PE accumulator 保持跨 k_tile）可用并通过验证（`tb_npu_tile_ksplit_gemm.v`）。但外部 DRAM 中的 PSUM surface 读写路径未实现——当前靠 PE accumulator 保持而非从外部恢复。

**修复思路**：controller 在 k_tile loop 中增加 PSUM read/write DMA 请求；DMA 增加 PSUM 读写目标；`psum_out_buf` 接入 `npu_top`。

---

### 7. 时钟门控/DFS 未接入 PE

**现状**：`npu_power.v` 有 DFS 行为时钟和 row/col gating 输出，但在 `npu_top.v` 中端口悬空（`.row_clk_gated()`、`.col_clk_gated()`、`.npu_clk()`）。

**修复思路**：将 `npu_power` 的输出接入 PE 的 clock enable 或至少记录为功耗优化状态。

---

### 8. Tile-mode Conv2D descriptor 未实现

**现状**：`OP=CONV2D_IM2COL` 在 descriptor v1 中已定义（op=2），但 controller 仅支持 `OP=GEMM_TILEPACK`（op=1）。on-the-fly im2col 目前只能在 direct scalar 路径中使用（CTRL[8]）。

---

## 二、已定位但未修复的 Bug

### 9. 16x16 全尺寸 GEMM：rows 6-11 数据偏移 4 rows

**现象**：16x16 tile GEMM 中，rows 0-5 和 12-15 结果正确，rows 6-11 的 A 数据发生了 4-row 偏移（读取了 rows 10-15 的 A 值而非 rows 6-11）。

**定位**：PPBuf 的 lane 映射（verilog generate 展开）对不同 lane 值返回正确数据。A skew pipe 的数学延迟模型与 weight w_v chain 的延迟在分析中一致。疑似 **Icarus Verilog 12.0 的 generate elaborate 行为异常**，导致 LANE≥6 时的 pipe 连线错位。需要 GTKWave 波形或更换仿真工具（如 Verilator）确认。

**文件**：`rtl/top/npu_top.v` → `gen_a_skew` generate 循环

---

### 10. 8x32 完整 GEMM 不可用

**现象**：
- aw_count 为 0（已修复 active_cols 宽度问题后变为 16，即 8 rows × 2 DMA burst/row）
- Columns 0-15 计算了但值不正确（与 16x16 同源的 pipe 问题）
- Columns 16-31 全部为 0（bottom half 从未收到 cols 16-31 的权重数据）

**根因**：8x32 的 fold 架构要求 controller 支持**两轮权重调度**：
1. 第一轮：feed W[k, 0:16) 到 16 物理列
2. 第二轮：feed W[k, 16:32) 到同一 16 物理列

当前 controller 只做一轮（32 列权重一次打包到 PPBuf，但 PPBuf 只输出 16 lane，cols 16-31 数据从未到达 PE 阵列）。

**修复思路**：controller 需要将 8x32 tile 的 compute 分为两个 sub-pass。第一轮 top half 计算 cols 0-15，第二轮 bottom half 计算 cols 16-31。sub-pass 之间不 flush（PE accumulator 保持）。约需 50-100 行 controller 代码。

---

### 11. A skew pipe 在 drain 期间清零的潜在风险

**现状**：`gen_a_skew` 中当 `tile_feed_step=1 && tile_vec_fire=0` 时，`pipe[0]` 被清零。这对 LANE≥2 理论上是安全的（0 在有效数据用完后才传播到输出），但 LANE=1 的已修复（改用寄存器）。目前未发现对 4x4/8x8 的可见影响，但在极端边界条件（大 K、深度 pipe）下可能成为隐患。

---

## 三、设计局限性（非 bug，而是架构选择）

### 12. PPBuf VEC_LANES=16 限制

**现状**：`pingpong_buf` 的 `VEC_LANES = MAX_TILE_LANES = 16`。对 16x16 和 8x32 足够，但若未来想支持 32x32 等更大阵列，需要扩展。当前可通过 `MAX_TILE_LANES` 参数调整。

### 13. DATA_W=16 限制顶层吞吐

**现状**：PE 内部已实现 INT8 2/4-lane SIMD（T7.3/T7.4），但 `npu_top` 默认 `DATA_W=16`、`INT8_SIMD_LANES=1`。要解锁 0.5-1 TOPS，需要将 PPBuf/feeder/serializer 升级到 32-bit packed lane。

---

## 优先级建议

| 优先级 | 问题 | 理由 |
|--------|------|------|
| P0 | #1 后处理入 tile 路径 | 消除 tile mode 对 CPU 后处理的依赖 |
| P0 | #4 4x4 边界 golden | 修复已知 gap，低工作量 |
| P1 | #9 16x16 全尺寸 | 需要波形/换仿真器定位 |
| P1 | #10 8x32 两轮权重 | 需要 controller 新功能 |
| P2 | #2 FP16 for 8x8 | 验证即可，低工作量 |
| P2 | #5 Descriptor for 8x8 | 验证 descriptor 链 |
| P3 | #3 WS golden | WS 不是当前主线 |
| P3 | #6 外部 PSUM surface | 需要多层调度配合 |
| P3 | #7 时钟门控接入 | 低功耗评测 |
| P3 | #8 Tile-mode Conv2D descriptor | 可与 #1 配合 |
