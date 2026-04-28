# NPU 目标架构方案

更新时间：2026-04-27

本文描述项目应收敛到的目标架构，并标注当前 RTL 与目标之间的差距。评分要求以 4x4 脉动阵列、AXI-Lite/Burst AXI、DMA、低功耗、RTL/FPGA 验证为基础；动态可重构阵列和更高带宽/算力作为优化项。

## 设计原则

1. NPU 不猜测矩阵规模。CPU 或 descriptor 必须提供每层的维度、地址、模式和后处理配置。
2. 卷积统一映射为 GEMM：`A_im2col[M,K] * W_col[K,N] = C[M,N]`。
3. 多层卷积必须有层间输出存储和 K-split 部分和存储。
4. AXI 只负责外部搬运，PE 阵列必须靠片上 buffer 供数。
5. 先做正确的 4x4 tile，再扩展到 16x16 和 8x32。

## 系统框图

```text
PicoRV32 CPU
  |
  | AXI4-Lite
  v
NPU register file
  |
  v
Descriptor controller
  |
  +--> AXI4 DMA master ---- DRAM/SRAM
  |         |
  |         +--> descriptor read
  |         +--> W tile read
  |         +--> IFM/A tile read
  |         +--> PSUM read/write
  |         +--> OFM write
  |
  +--> A_BUF ping/pong
  +--> W_BUF ping/pong
  +--> PSUM/OUT_BUF ping/pong
  |
  v
PE array 4x4 baseline, 16x16 optimized
  |
  v
post-process: bias, ReLU/ReLU6, quant/saturate
```

## 卷积到 GEMM 的映射

```text
M = OH * OW * batch
K = Cin * KH * KW
N = Cout

A_im2col[M,K] : 每一行是一个输出像素位置展开后的输入窗口
W_col[K,N]    : 每一列是一个输出通道的卷积核
C[M,N]        : 输出特征图按空间位置和输出通道排列
```

建议不要在 DRAM 中完整物理展开 im2col。更好的方式是 DMA/地址发生器按窗口顺序读取 IFM，并在片上形成 `A_BUF` tile。第一阶段为了降低复杂度，可以先由 CPU/脚本在 DRAM 中预展开 im2col，等 GEMM 路径正确后再做 on-the-fly im2col。

## Descriptor

推荐新增 descriptor 队列，让 CPU 一次提交多层任务：

```text
desc {
  op_type        // GEMM, CONV2D, FC
  dtype          // INT8, FP16
  dataflow       // WS, OS
  M, N, K
  IH, IW, OH, OW, Cin, Cout, KH, KW, stride, pad
  ifm_addr
  weight_addr
  bias_addr
  psum_addr
  ofm_addr
  activation     // none, ReLU, ReLU6
  flags          // first_k, last_k, last_layer, irq_en
  next_desc
}
```

层结束条件：

```text
all m_tile done && all n_tile done && all k_tile done
```

网络结束条件：

```text
desc.last_layer == 1 or desc.next_desc == 0
```

## PE 阵列

### 基础形态

基础验收先实现真正的 4x4：

- 16 个 PE 同时参与计算。
- 每拍向阵列边界提供 4 个 activation 和 4 个 weight。
- 每个 tile 产生最多 16 个输出。
- 支持 INT8 和 FP16。
- 支持 WS 和 OS。

### 优化形态

物理 16x16 阵列可配置为：

| 形态 | 含义 | 需要的控制 |
|---|---|---|
| 4x4 | 只开左上 4x4 | active mask + 输出映射 |
| 8x8 | 只开左上 8x8 | active mask + 输出映射 |
| 16x16 | 全开 | 满阵列供数和写回 |
| 8x32 | 两个 8x16 半阵列横向折叠 | 路由、valid 对齐、输出重映射 |

动态可调不是简单改参数。4x4/8x8/16x16 可主要依赖 active mask 和 clock enable；8x32 需要改变 PE 间数据路由和输出收集方式。

### PE 内部

推荐 PE 结构：

```text
input regs -> multiplier -> accumulator -> output regs
```

INT8 目标：

- baseline：1-lane INT8 MAC。
- 性能优化：2-lane 或 4-lane INT8 SIMD MAC，INT32 accumulate。

FP16 目标：

- 1-lane FP16 multiply。
- FP32 accumulate。

算力估算：

```text
4x4 scalar @500MHz   = 16 GOPS
16x16 scalar @500MHz = 256 GOPS
16x16 2-lane @500MHz = 0.512 TOPS
16x16 4-lane @500MHz = 1.024 TOPS
```

因此 0.5-1 TOPS 目标要求 16x16 阵列配合 PE 内 INT8 SIMD。

## 4x4 Tile 数据布局

T2.1 采用固定的 4x4 逻辑 tile，先让 16 个 PE 在 OS 模式下真正同时产生 16 个输出。后续 8x8/16x16 只把 `TILE_M/TILE_N` 从 4 扩展到 8/16，不改变基本地址和 lane 口径。

### Tile 坐标

```text
TILE_M = 4
TILE_N = 4

m0 = m_tile * 4
n0 = n_tile * 4
k0 = 0              // Phase 2 暂不做 K-split
k_len = K

active_rows = min(4, M - m0)
active_cols = min(4, N - n0)
row_valid[r] = (r < active_rows)
col_valid[c] = (c < active_cols)
out_valid[r,c] = row_valid[r] && col_valid[c]
```

`m_tile` 沿 M 方向递增，`n_tile` 沿 N 方向递增。推荐遍历顺序为：

```text
for m_tile in 0 .. ceil(M/4)-1:
  for n_tile in 0 .. ceil(N/4)-1:
    compute C[m0:m0+3, n0:n0+3]
```

### 外部逻辑矩阵布局

软件和 golden model 使用标准 GEMM 语义：

```text
C[M,N] = A[M,K] * W[K,N]
```

逻辑地址定义为：

```text
A_row_major(i,k) = A_ADDR + (i * K + k) * elem_bytes
W_row_major(k,j) = W_ADDR + (k * N + j) * elem_bytes
C_row_major(i,j) = R_ADDR + (i * N + j) * 4
```

其中：

```text
elem_bytes = 1 for INT8
elem_bytes = 2 for FP16
```

### Phase 2 片上 tile-pack 格式

为了先验证 PE 阵列和控制器，Phase 2 的 testbench/CPU 可以把 A/W 预打包为 lane-major tile 流，避免在 T2 同时实现 gather DMA。后续 on-the-fly im2col 或 row-major gather 可在 Phase 6/T3 之后补上。

A tile pack：

```text
A_TILE[m_tile][k][r] = row_valid[r] ? A[m0+r, k] : 0
```

W tile pack：

```text
W_TILE[n_tile][k][c] = col_valid[c] ? W[k, n0+c] : 0
```

地址：

```text
A_VEC_BYTES = 4 * elem_bytes
W_VEC_BYTES = 4 * elem_bytes

A_TILE_ADDR(m_tile,k) = A_ADDR + (m_tile * K + k) * A_VEC_BYTES
W_TILE_ADDR(n_tile,k) = W_ADDR + (n_tile * K + k) * W_VEC_BYTES
```

INT8 时一个 32-bit AXI beat 正好携带 4 个 lane：

```text
beat[ 7: 0] = lane0
beat[15: 8] = lane1
beat[23:16] = lane2
beat[31:24] = lane3
```

FP16 时一个 4-lane vector 需要两个 32-bit AXI beat：

```text
beat0[15: 0] = lane0
beat0[31:16] = lane1
beat1[15: 0] = lane2
beat1[31:16] = lane3
```

INT8 lane 进入 PE 前符号扩展到 `DATA_W=16`；FP16 lane 保持原始 16-bit bit pattern。

### OS 模式 lane 顺序

OS 是 T2 的主验收路径。逻辑映射为：

```text
PE row r -> M lane r -> C[m0+r, *]
PE col c -> N lane c -> C[*, n0+c]
PE(r,c)  -> C[m0+r, n0+c]
```

因为当前阵列的 OS 权重从顶部进入并向下传播，row `r` 会比 row 0 晚 `r` 拍看到同一个 `W[k,c]`。因此 feeder 必须给 activation 做行 skew：

```text
logical cycle t = 0 .. K+3

w_in[c]   = (t < K && col_valid[c]) ? W_TILE[n_tile][t][c] : 0
act_in[r] = (0 <= t-r < K && row_valid[r])
              ? A_TILE[m_tile][t-r][r]
              : 0
```

这样 PE(r,c) 在物理周期 `t = k + r` 同时看到：

```text
A[m0+r, k]
W[k, n0+c]
```

并在内部累加：

```text
C[m0+r,n0+c] += A[m0+r,k] * W[k,n0+c]
```

OS 计算周期数：

```text
compute_cycles = K + active_rows - 1
```

随后进入 drain/flush，等待 `valid_out` 对齐后交给 serializer。

### WS 模式 lane 顺序

WS 用于权重复用，但在当前 4x4 物理映射下，它天然产生一个 M row 对应的 1x4 输出向量。完整 4x4 C tile 由 4 个 M row pass 组成。

权重加载映射：

```text
PE row r -> K lane r inside current K block
PE col c -> N lane c
PE(r,c) holds W[kb+r, n0+c]
```

加载顺序：

```text
for r in 0..3:
  load_w = 1
  w_in[c] = W[kb+r, n0+c] if kb+r < K and col_valid[c] else 0
```

计算某一个 M lane `mr`：

```text
act_in[0] = A[m0+mr, kb+t] if kb+t < K and row_valid[mr] else 0
```

当前 `reconfig_pe_array` 的 WS 路径会把 `act_in[0]` 通过行延迟送到后续 row，底部输出 4 个列结果：

```text
C[m0+mr, n0+c] partial += sum_r A[m0+mr,kb+r] * W[kb+r,n0+c]
```

Phase 2 如果只实现无 K-split WS，建议先限制 `K <= 4`。`K > 4` 的 WS 需要 `PSUM/OUT_BUF` 或 accumulator init，归入 Phase 4。

### C tile 输出顺序

阵列输出到 serializer 的目标逻辑顺序固定为行主序：

```text
result_index = r * 4 + c
result[result_index] = C[m0+r, n0+c]
valid[result_index]  = out_valid[r,c]
```

外部 DRAM 仍按 C row-major 地址写回：

```text
C_ADDR(r,c) = R_ADDR + ((m0+r) * N + (n0+c)) * 4
```

因此 serializer 不能简单假设 16 个 word 在全局 C 矩阵中总是连续。正确写回策略是：

```text
for r in 0..active_rows-1:
  write burst base = R_ADDR + ((m0+r) * N + n0) * 4
  write beats      = active_cols
  data order       = C[m0+r,n0], C[m0+r,n0+1], ...
```

当 `N == 4` 或 `active_cols == N` 时，这退化为一个连续 16 word 写回；一般矩阵需要每个有效 row 一个小 burst。

### T2.2/T2.3/T2.4 接口结论

后续 RTL 按以下接口推进：

```text
a_vec[4]       // A lane r
w_vec[4]       // W lane c
row_valid[4]
col_valid[4]
result_vec[16] // result[r*4+c]
valid_vec[16]
```

`pingpong_buf` 的 Phase 2 输出不再是一个 `rd_data` 标量，而是一个 4-lane vector；`npu_ctrl` 的计数单位不再是单个 `C[i,j]`，而是 `(m_tile,n_tile)`；`npu_top` 的结果 FIFO 前必须增加 16-output serializer 和 row-wise writeback 控制。

T2.3 规定 `ARR_CFG[7]` 为 4x4 tile mode enable。该位为 0 时保持 Phase-1 标量兼容路径；该位为 1 时：

```text
tile_i/tile_j -> m_tile/n_tile
tile_m_base   -> m_tile * 4
tile_n_base   -> n_tile * 4
vec_consume   -> 每个 k 周期推进一组 A/W vector
```

## WS 数据流

Weight Stationary 适合卷积，因为同一权重会被多个输出像素复用。

逻辑映射：

```text
PE row -> K tile lane
PE col -> output channel lane
time   -> M tile 内的输出位置
```

流程：

1. `W_BUF` 将 `W[k0+r, n0+c]` 载入 PE(r,c) 的 weight register。
2. `A_BUF` 每拍送入 `A[m, k0+r]`。
3. 每列对 row 方向乘积求和，得到 `C[m,n0+c]` 的部分和。
4. 若 K 未完成，写 `PSUM_BUF`。
5. 若 K 完成，进入 bias/activation/quant，写 `OFM`。

WS 下重新载权重的时机由 tile 计数器决定：

```text
new n_tile or new k_tile -> LOAD_W_TILE
all tiles in current layer done -> NEXT_LAYER -> LOAD_W_TILE for next desc
```

## OS 数据流

Output Stationary 适合输出 tile 能完整留在 PE 内部的 GEMM/FC。

逻辑映射：

```text
PE row -> M tile lane
PE col -> N tile lane
PE(r,c) holds C[m0+r,n0+c]
```

流程：

1. `A_BUF` 从左侧送 `A[m0+r,k]`。
2. `W_BUF` 从上侧送 `W[k,n0+c]`。
3. PE 内部 `psum += A * W`。
4. `k=0..K-1` 完成后 flush。
5. 输出 tile 写到 `OUT_BUF` 或 `PSUM_BUF`。

如果 K 被切分，OS 必须支持从 `PSUM_BUF` 初始化 PE accumulator，不能每个 K tile 都清零。

## 控制 FSM

推荐顶层 FSM：

```text
IDLE
FETCH_DESC
DECODE_DESC
INIT_LAYER
INIT_TILE
LOAD_W_TILE
LOAD_A_TILE
LOAD_PSUM
WAIT_LOAD
COMPUTE
DRAIN
WRITE_PSUM_OR_OUT
POST_PROCESS
NEXT_K_TILE
NEXT_MN_TILE
NEXT_LAYER
DONE
ABORT
ERROR
```

核心跳转：

```text
IDLE -> FETCH_DESC              start
FETCH_DESC -> DECODE_DESC       descriptor ready
DECODE_DESC -> INIT_LAYER       latch config
INIT_LAYER -> INIT_TILE         clear tile counters
INIT_TILE -> LOAD_W/A/PSUM      issue DMA
WAIT_LOAD -> COMPUTE            buffers ready
COMPUTE -> DRAIN                k_count done
DRAIN -> WRITE_PSUM_OR_OUT      valid drained
WRITE -> NEXT_K_TILE            more K tiles
WRITE -> POST_PROCESS           last K tile
POST_PROCESS -> NEXT_MN_TILE    output tile done
NEXT_MN_TILE -> INIT_TILE       more M/N tiles
NEXT_MN_TILE -> NEXT_LAYER      layer done
NEXT_LAYER -> FETCH_DESC        more desc
NEXT_LAYER -> DONE              network done
```

## AXI 和 DMA

通路划分：

- AXI4-Lite slave：CPU 配置寄存器、状态、中断、descriptor base。
- AXI4 master DMA：读 descriptor、读 W、读 IFM/A、读写 PSUM、写 OFM。

DMA 必须支持 INCR burst：

```text
ARLEN/AWLEN   = burst_beats - 1
ARSIZE/AWSIZE = 3'b010    // 32-bit beat
ARBURST/AWBURST = 2'b01   // INCR
```

32-bit AXI @ 500 MHz 理论带宽：

```text
raw = 2.0 GB/s
60% = 1.2 GB/s
80% = 1.6 GB/s
```

如果每个 burst 约 2 拍地址/响应开销，效率近似：

```text
8-beat  burst -> 8/(8+2)  = 80.0%
16-beat burst -> 16/(16+2)= 88.9%
32-beat burst -> 32/(32+2)= 94.1%
```

所以读通道单拍事务无法达到 80% 目标。

## 当前 RTL 差距摘要

| 目标 | 当前状态 | 后续任务 |
|---|---|---|
| 顶层标量 dot product | Phase 1 已通过，`u_scalar_pe` 兼容路径可写回正确结果 | 作为后续回归基线保留 |
| 4-lane A/W buffer | T2.2 已增加 `rd_vec`/`rd_vec_en`，顶层已接入阵列左上 4x4 | T2.3 切换 controller 到 vector consume |
| 4x4 tile 计数和 mask | T2.3 已通过 `ARR_CFG[7]` 启用 tile planner，支持 M/N 边界 mask | 保持回归 |
| 4x4 真并行 tile | T2.4/T2.5/T2.6 已完成 serializer/writeback 和 4x4 INT8/FP16 golden 测试 | Phase 3 进入 AXI burst DMA |
| 多层卷积 | 无 descriptor、无 OUT/PSUM buffer | 增加 descriptor controller 和 PSUM/OUT_BUF |
| AXI burst | 读通道 single-beat | 实现 INCR burst、4KB 边界处理 |
| 16x16/8x32 | 阵列模块存在 | 补齐供数、valid 对齐、写回 |
| 低功耗 | 行为模块存在但未接入 | 使用 clock enable/BUFGCE/ICG 接入 |
| SoC 验证 | `npu_top` 参数已对齐；当前卡在 DRAM `axi_arlen` 和 PicoRV32 PCPI 端口 | 先修 SoC 编译，再跑 CPU 启动 NPU smoke |
