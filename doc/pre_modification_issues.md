# 修改前电路的原始问题

本文档记录 DeepSeek 在修改前对 NPU 电路审查发现的所有问题。

---

## 问题总览

修改前电路存在三种类型的问题：
1. **硬编码 4x4 假设**：电路逻辑正确但在 4 以外的 tile 形态上因宽度/参数的硬编码截断而失效
2. **实现不完整**：功能框架存在但只完成了 4x4 的数据通路，更大形态的代码仅写了一半
3. **设计缺陷**：逻辑本身有 bug，即使 4x4 也会在某些条件下触发

---

## 一、硬编码 4x4 假设

### 1.1 controller tile_active 钳位

**文件**：`rtl/ctrl/npu_ctrl.v:453,485-490`

```verilog
localparam [31:0] TILE4_LANES = 32'd4;

assign tile_active_rows = !tile_mode ? 3'd1 :
    (tile_active_rows_32 >= TILE4_LANES) ? 3'd4 :
    tile_active_rows_32[2:0];
assign tile_active_cols = !tile_mode ? 3'd1 :
    (tile_active_cols_32 >= TILE4_LANES) ? 3'd4 :
    tile_active_cols_32[2:0];
```

**问题**：无论 cfg_shape 设为 1（8x8）、2（16x16）、3（8x32），只要 active_rows/cols ≥ 4 就被钳位到 3'd4。

**影响**：8x8/16x16/8x32 永远只有 4×4 的 active mask，导致：
- `wb_row` 循环最多迭代 4 次
- `tile_row_r_len` 最多 16 bytes
- 每个 tile 只写回左上角 4×4 的子块
- serializer 只输出 16 个词而非全 tile 的结果

---

### 1.2 端口宽度不足以容纳大 shape 值

| 信号 | 文件 | 原始宽度 | 可表示范围 | 8x32 需要 |
|------|------|---------|-----------|----------|
| `tile_active_cols` | npu_ctrl.v:95 | [2:0] | 0–7 | **32** |
| `tile_active_rows` | npu_ctrl.v:94 | [2:0] | 0–7 | 16 |
| `wb_row` | npu_ctrl.v:451 | [2:0] | 0–7 | 16 |
| `tile_row_valid` | npu_ctrl.v:92 | [3:0] | 4 bit | 16 |
| `tile_col_valid` | npu_ctrl.v:93 | [3:0] | 4 bit | 32 |
| `dma_a_ofm_active_rows` | npu_dma.v:58 | [2:0] | 0–7 | — |

---

### 1.3 serializer 硬编码 16 个输出槽

**文件**：`rtl/top/npu_top.v:913-920`

```verilog
reg [ACC_W-1:0] tile_result_buf [0:15];   // 16 槽
reg [1:0]       tile_ser_row;             // 2-bit，最多 4 行
reg [1:0]       tile_ser_col;             // 2-bit，最多 4 列
wire [3:0] tile_ser_idx = {tile_ser_row, tile_ser_col};  // 4-bit 索引
```

```verilog
for (tile_ser_i = 0; tile_ser_i < 16; tile_ser_i = tile_ser_i + 1)  // 只读 lanes [0:15]
```

**问题**：buffer 只有 16 个槽位，行/列计数器最多到 3。对 8x8（64 结果）、16x16（256 结果）、8x32（256 结果），90% 以上的 PE 结果被丢弃。8x32 右半 16 列的数据因 `pe_array_result` 只读到 lane 15 而完全丢失。

---

### 1.4 descriptor 只接受 4x4

**文件**：`rtl/ctrl/npu_ctrl.v:307`

```verilog
(desc_shape_field == 4'd0)   // 4x4 only
```

**问题**：descriptor v1 定义了 4x4/8x8/16x16/8x32 四种 shape（desc_shape_field 的 0/1/2/3），但 controller 把 1/2/3 都当成 `ERR_DESC_UNSUPPORTED` 拒绝。

---

### 1.5 单个 `vector_elem_bytes` 用于 W 和 A

**文件**：`rtl/ctrl/npu_ctrl.v:439-440`

```verilog
wire [15:0] vector_elem_bytes = scalar_elem_bytes * {11'd0, tile_shape_lanes};
wire [15:0] bytes_per_k = tile_mode ? vector_elem_bytes : scalar_elem_bytes;
```

**问题**：8x32 的 W 需要 32 columns/k，A 需要 8 rows/k。但源码用一个 `tile_shape_lanes`（8x32 时返回 16）乘以 `scalar_elem_bytes` 得到 16 bytes/k，对 W 和 A 都不对。DMA load 长度和 prefetch 地址都基于这个单一值。

---

## 二、实现不完整

### 2.1 PE 阵列输出映射只有底部一行

**文件**：`rtl/array/reconfig_pe_array.v:362-385`

```verilog
MODE_8x8: begin
    // 只输出底部一行的 8 个列结果
    for (ci = 0; ci < PHY_COLS; ci = ci+1) begin
        if (ci < 8) begin
            acc_out[ci*ACC_W +: ACC_W]   = acc_v[8][ci];
            valid_out[ci]                  = valid_v[8][ci];
        end
    end
end
MODE_16x16: begin
    // 只输出底部一行的 16 个列结果
    for (ci = 0; ci < PHY_COLS; ci = ci+1) begin
        acc_out[ci*ACC_W +: ACC_W]   = acc_v[PHY_ROWS][ci];
        valid_out[ci]                  = valid_v[PHY_ROWS][ci];
    end
end
MODE_8x32: begin
    // 只输出底部一行的 32 个列结果
    ...
end
```

**问题**：4x4 有完整的 per-PE grid 输出（`acc_out[(ri*4+ci)*ACC_W]`），但 8x8/16x16/8x32 只输出每个列底部的 PE 累加器。在 OS 模式下，每个 PE 有独立的累加器值，需要全部 64/256 个输出。当前的列输出只对 WS 模式有意义（列底部 PE 有完整点积），但 WS 尚未端到端验证。

**根源**：T7.2（8x32 折叠路由验证）和 T7.1（8x8/16x16 向量供数）完成后，写回路径的开发停在这里——供数和阵列路由已验证，但输出映射和写回没有继续完成。

---

### 2.2 8x8/16x16/8x32 仅供数验证，未端到端 golden 测试

| 测试文件 | 验证内容 | 缺少 |
|---------|---------|------|
| `tb_npu_tile_lane_feed.v` | W/A 数据到达 PE 阵列边界 | 不检查 PE 输出、不检查写回、不对比 golden |
| `tb_reconfig_pe_8x32.v` | 8x32 输出顺序、折叠路由、WS load wrap | 不检查数值正确性、不接 npu_top |
| 无 | 8x8/16x16 端到端 GEMM | — |

---

### 2.3 tile 路径无后处理

**文件**：`rtl/top/npu_top.v:963-965`

```verilog
assign r_fifo_din   = ctrl_tile_mode ? tile_result_buf[tile_ser_idx] : scalar_post_result;
assign r_fifo_wr_en = ctrl_tile_mode ? tile_ser_fire : (scalar_valid && !r_fifo_full);
```

**问题**：bias、ReLU/ReLU6、INT8 quant/saturate 电路在 `scalar_post_result` 路径中已完整实现（T6.3-T6.5），但 tile mode 的三目运算符 `ctrl_tile_mode ? tile_result_buf : scalar_post_result` 短路了所有后处理。tile mode 输出的是 raw int32 MAC 累加器，与 scalar 路径的后处理逻辑完全隔离。

---

### 2.4 `npu_power` 输出悬空

**文件**：`rtl/top/npu_top.v:973-985`

```verilog
npu_power #(...) u_power (
    ...
    .npu_clk     (),
    .row_clk_gated(),
    .col_clk_gated()
);
```

**问题**：DFS 行为时钟和 row/col clock gating 输出在 `npu_top` 中悬空，不连接任何 PE 时钟。这意味着低功耗设计无法在仿真或 FPGA 中实际生效。

---

## 三、设计缺陷（Bug）

### 3.1 serializer 索引用 active_cols 而非 grid_cols

**文件**：`rtl/top/npu_top.v:920`

```verilog
wire [3:0] tile_ser_idx = {tile_ser_row, tile_ser_col};
```

**问题**：当 `active_cols < grid_cols`（边界 tile，如 N=3 而 grid_cols=4 或 8），索引 `row * active_cols + col` 与 PE grid 的存储 `row * grid_cols + col` 错位。导致边界 tile 的 serializer 输出的行顺序完全错乱。

**重现**：M=2, K=1, N=3 8x8（行 1 的结果为全零，行 0 值正确）。

---

### 3.2 A skew pipe LANE=1 在 drain 时清零

**文件**：`rtl/top/npu_top.v:664-666`

```verilog
pipe[0] <= tile_vec_fire ? a_ppb_rd_vec[LANE*DATA_W +: DATA_W]
                         : {DATA_W{1'b0}};
```

**问题**：LANE=1 只有一级 pipe（`pipe[0]`），在 drain 周期（`tile_feed_step=1, tile_vec_fire=0`）`pipe[0]` 被清零。但 `a_skew_vec[1]` 恰好在同一拍需要读取 `pipe[0]` 中的有效数据，非阻塞赋值导致读到新写的 0。对 LANE≥2 不触发（有中间 pipe 缓冲）。

---

### 3.3 tile_capture_cnt 截断

**文件**：`rtl/top/npu_top.v:936`（原始代码）

```verilog
wire [7:0] tile_capture_cnt = tile_grid_rows[3:0] * tile_grid_cols[5:0];
```

**问题**：`tile_grid_rows` 为 5'd16（二进制 10000），`[3:0]` 截取低 4 位得到 0。导致 16x16 capture 0 个结果，所有 PE 输出被丢弃。

---

### 3.4 tile_len_raw 标量模式损失对齐

**文件**：`rtl/ctrl/npu_ctrl.v`（在 Vector_elem_bytes 拆分后引入）

```verilog
wire [15:0] tile_len_raw_w = tile_k_len * vector_elem_bytes_w;
```

**问题**：`vector_elem_bytes_w = scalar_elem_bytes * tile_shape_n_lanes`。标量模式下 `tile_shape_n_lanes=4`，`scalar_elem_bytes=1`，所以 `tile_len_raw_w = K * 4`。但标量模式的 DRAM 布局是每行/列 `K` bytes（对齐到 4），不是 `K * 4` bytes。导致 prefetch 地址步长翻倍，标量 matmul/Conv2D 全部失败。

**重现**：`run_matmul_case.ps1 -M 2 -K 2 -N 2` 在修改后返回 1/4 PASS，原始代码返回 ALL 4 PASS。

---

### 3.5 8x32 两轮权重调度缺失

**文件**：controller FSM 无相关逻辑

**问题**：8x32 的 fold 架构（`reconfig_pe_array.v` 行 332）要求 controller 分两轮喂权重：
- 第一轮：feed W[k, 0:16)，top half 计算 logical cols 0-15
- 第二轮：feed W[k, 16:32)，bottom half 计算 logical cols 16-31

当前 controller 将 32 列权重一次性打包到 PPBuf（32 bytes/k → 8 PPBuf words），但 PPBuf 只输出 16 lanes（4 words/cycle），cols 16-31 的 4 个 words 从未到达 PE 阵列。bottom half 的 128 个 PE 输出恒为 0。

---

### 3.6 16x16 大 shape 下 A 数据偏移（未修复）

**文件**：`rtl/top/npu_top.v` gen_a_skew generate 循环

**现象**：16x16 tile GEMM 中，rows 6-11 的 A 数据发生 4-row 偏移（读取了 rows 10-15 的 A 值而非 rows 6-11）。rows 0-5 和 12-15 正确。

**分析**：PPBuf lane 映射在 verilog generate 展开中对于不同 lane_i 值返回正确数据。A skew pipe 的延迟数学分析显示时序应正确。偏移恰好对应一个 PPBuf word（4 lanes）。疑似 Icarus Verilog 12.0 对深度 generate（16 级 pipe × 16 lane）elaborate 时的连线异常。需 GTKWave 波形或换仿真器确认。

---

## 修复状态

| # | 问题 | 状态 |
|---|------|------|
| 1.1 | TILE4_LANES 钳位 | ✅ 已修复 |
| 1.2 | 端口宽度不足 | ✅ 已修复 |
| 1.3 | serializer 16 槽 | ✅ 已修复 |
| 1.4 | descriptor 只接受 4x4 | ✅ 已修复 |
| 1.5 | 单 vector_elem_bytes | ✅ 已修复 |
| 2.1 | PE 输出映射不完整 | ✅ 已修复 |
| 2.2 | 无端到端 golden 测试 | ✅ 8x8 已补全 |
| 2.3 | tile 路径无后处理 | ❌ 未修复 |
| 2.4 | npu_power 悬空 | ❌ 未修复 |
| 3.1 | serializer 索引 bug | ✅ 已修复 |
| 3.2 | A skew LANE=1 bug | ✅ 已修复 |
| 3.3 | capture_cnt 截断 | ✅ 已修复 |
| 3.4 | tile_len scalar 回归 | ✅ 已修复 |
| 3.5 | 8x32 两轮权重 | ❌ 未修复 |
| 3.6 | 16x16 A 数据偏移 | ❌ 未修复 |
