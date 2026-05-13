# NPU 目标架构方案

更新时间：2026-05-03

本文描述项目应收敛到的目标架构，并标注当前 RTL 与目标之间的差距。评分要求以 4x4 脉动阵列、AXI-Lite/Burst AXI、DMA、低功耗、RTL/FPGA 验证为基础；动态可重构阵列和更高带宽/算力作为优化项。

## 术语和缩写

本文后续会频繁使用以下缩写。若一个名字同时有硬件模块含义和数据语义，下面会分开说明。

| 名词 | 含义 |
|---|---|
| CPU / PicoRV32 | 软件控制侧，负责准备输入数据、descriptor、寄存器配置，并通过 AXI4-Lite 启动 NPU。 |
| AXI4-Lite | 低带宽控制总线，用于访问 NPU 寄存器、状态、中断和性能计数。 |
| AXI4 master / DMA | NPU 主动访问外部 DRAM/SRAM 的数据搬运通路，用于读 descriptor、读 A/W/PSUM、写 OFM。 |
| descriptor | 一层或一个 tile-pack GEMM 任务的 64-byte 描述符。它把维度、地址、数据类型、数据流、后处理、层间 IFM 来源和 `next_desc` 串在一起。 |
| IFM / OFM | input feature map / output feature map。卷积语境下分别是输入特征图和输出特征图；GEMM_TILEPACK 语境下，`ifm_addr` 当前指向预打包后的 A tile stream。 |
| A / W / C | GEMM 语境下的 activation/input matrix、weight matrix、output matrix：`A[M,K] * W[K,N] = C[M,N]`。 |
| M / N / K | GEMM 维度。M 是输出行，N 是输出列，K 是归约维度；卷积中分别对应 `batch*OH*OW`、`Cout`、`Cin*KH*KW`。 |
| tile | 大矩阵切成的小块。当前主验证路径是 4x4 C tile，即一次产生最多 16 个输出元素。 |
| lane | 同一拍送入阵列的一条并行数据通道。4-lane A/W vector 表示每个逻辑 k 取 4 个 A 行 lane 和 4 个 W 列 lane。 |
| PE | processing element。一个乘加单元，内部有 multiplier 和 accumulator。4x4 tile 使用左上 16 个 PE。 |
| PPBuf / ping-pong buffer | 片上双缓冲。DMA 写入一侧，PE 消费另一侧，用来隐藏一部分读数据和计算的时序差。 |
| PSUM | partial sum，部分和。K 被拆分、层间需要保留中间累计值、或后处理尚未完成时，C 的中间 accumulator 值都称为 PSUM。INT8 路径按 signed int32 bit pattern 保存，FP16 路径按 FP32 bit pattern 保存。 |
| PSUM/OUT_BUF | 片上 4x4 tile-local accumulator buffer。它只保存当前 tile 的 16 个 32-bit word，不等于整层 C 矩阵；整层 PSUM/OFM surface 仍在外部 DRAM/SRAM。 |
| K-split | 当 K 太长或调度需要分段时，把一次 dot-product 拆成多个 k_tile。中间 k_tile 产出 PSUM，最后 k_tile 产出最终 OUT/OFM。 |
| OS / WS | output stationary / weight stationary。OS 让 C/PSUM 留在 PE accumulator 内，WS 让 W 留在 PE 内。当前 4x4 tile 主路径是 OS。 |
| row-skew | OS 脉动阵列中，为了让 row `r` 和 weight wavefront 对齐，A lane 会比 row 0 延迟 `r` 拍送入。 |
| accumulator init | 在计算开始前给 PE accumulator 写入初值。用于 bias 或从外部 PSUM surface 恢复上一个 k_tile 的部分和。 |
| post-process | 最后一个 k_tile 后对 32-bit accumulator 做 bias、activation、quant/saturate 等处理，再写成目标 OFM 格式。当前 direct scalar 路径已验证 bias、ReLU/ReLU6 和 INT8 quant/saturate；tile/descriptor 主线仍主要验证 32-bit 写回。 |

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

## 端到端执行流程

从系统角度看，NPU 有两条启动路径，但进入计算后的数据通路相同。

```text
direct register mode:
CPU -> AXI4-Lite -> npu_axi_lite -> npu_ctrl
    -> npu_dma -> A/W PPBuf -> npu_top feeder -> reconfig_pe_array/pe_top
    -> serializer/result FIFO -> npu_dma -> DRAM OFM

descriptor mode:
CPU prepares descriptor list in DRAM
CPU -> AXI4-Lite DESC_BASE/DESC_COUNT/CTRL[7]
    -> npu_ctrl requests descriptor fetch
    -> npu_dma reads 64-byte descriptor
    -> npu_ctrl decodes M/N/K/A/W/R/mode
    -> same compute/writeback path as direct mode
    -> next_desc or LAST_LAYER decides the next layer
```

各大模块在流程中的职责如下。

1. CPU 只提交“任务是什么”。它写寄存器或 descriptor，不直接驱动 PE 时序。
2. `npu_axi_lite` 是控制面入口，把 CPU 写入的 `M/N/K`、地址、模式、`DESC_BASE/DESC_COUNT`、启动位和中断控制暴露给 `npu_ctrl`。
3. `npu_ctrl` 是调度器。它决定当前 layer、`m_tile/n_tile/k_tile`、边界 mask、A/W load、compute、flush、writeback 和 descriptor 跳转。
4. `npu_dma` 是数据搬运器。它通过 AXI4 master 读 descriptor、读 W tile、读 A/IFM tile，后续还应读写外部 PSUM surface，并把结果写回 OFM。
5. `pingpong_buf` 是 PE 前的片上供数缓冲。当前 A/W 使用 4-lane tile-pack 格式，DMA 写入后由 controller 用 `vec_consume` 推进读指针。
6. `npu_top` 是集成边界。它把控制信号、PPBuf 数据、row-skew feeder、PE array、serializer、result FIFO 和 DMA 写回握手接起来。
7. `reconfig_pe_array`/`pe_top` 执行乘加。4x4 OS 路径中，PE(r,c) 对应 `C[m0+r,n0+c]`，K 方向的乘积累加留在 PE accumulator 内。
8. serializer/result FIFO 把 16 个 PE 输出转换成 row-major 写回序列。一般矩阵不能假设 16 word 全局连续，所以写回按有效 row 发短 burst。

T5.4 后，descriptor 链表已经可以顺序执行多个 4x4 OS tile-pack GEMM descriptor，并支持后一层用上一层 32-bit row-major OFM 作为 IFM。该路径由 `desc_ctrl[23] IFM_FROM_PREV_OFM` 触发：controller 记录上一层 `ofm_addr`，DMA 按 `A[m0+r,k]` 从上一层 OFM surface gather 4 个 row lane，再 repack 到 A PPBuf。当前已验证 INT8 4x4 GEMM 串联；direct scalar 路径已验证 bias、ReLU/ReLU6 和 INT8 quant/saturate；外部 PSUM surface 的 read/modify/write 以及 tile/descriptor 后处理仍属于后续工作。

## 卷积到 GEMM 的映射

普通 dense Conv2D（`groups=1`）可以统一映射为 `A_im2col[M,K] * W_col[K,N] = C[M,N]`。详细推导、stride/padding/dilation 公式和 layout 说明见 [conv_gemm_mapping.md](conv_gemm_mapping.md)。

```text
# M: GEMM 行数；卷积中表示 batch 内所有输出空间位置。
# K: GEMM 归约维度；卷积中表示一个卷积窗口内的输入元素数。
# N: GEMM 列数；卷积中表示输出通道数，也就是卷积核个数。
M = batch * OH * OW
K = Cin * KH * KW
N = Cout

# A_im2col 的每一行对应一个输出像素位置 (b,oh,ow) 的完整输入窗口。
A_im2col[M,K]

# W_col 的每一列对应一个输出通道 cout 的卷积核，展平顺序必须和 A 的 K 维一致。
W_col[K,N]

# C 的每一行对应一个输出空间位置，每一列对应一个输出通道。
C[M,N]
```

索引关系：

```text
# b/oh/ow: 输出元素所在的 batch、输出行、输出列。
# cin/kh/kw: 卷积窗口中的输入通道、核行、核列。
# cout: 输出通道。
m = (b * OH + oh) * OW + ow
k = (cin * KH + kh) * KW + kw
n = cout

ih = oh * stride_h + kh * dilation_h - pad_h
iw = ow * stride_w + kw * dilation_w - pad_w

A_im2col[m,k] = (0 <= ih < IH and 0 <= iw < IW) ? IFM[b,cin,ih,iw] : 0
W_col[k,n]    = WEIGHT[cout,cin,kh,kw]
C[m,n]        = OFM[b,oh,ow,cout]  // NHWC 视角；NCHW 只改变外部存储顺序
```

建议不要在 DRAM 中完整物理展开 im2col。T6.1 已先由 CPU/脚本在 DRAM 中预展开 `A_im2col`；T6.2 已把 direct scalar 路径的 A 侧地址计算搬进 DMA，`CTRL[8]` 置位后，DMA 从 raw NCHW IFM 按 `m/k` 计算窗口地址，padding/越界位置写 0，并在片上形成当前 A 行。T6.3-T6.5 在同一 direct scalar 路径上支持 `bias[j]`、ReLU/ReLU6 和 INT8 quant/saturate，执行顺序为 dot -> bias -> activation -> quant/saturate。T6.6 已验证 layer0 的量化 OFM 可直接作为 layer1 输入。当前 on-the-fly im2col 和后处理仍限于 direct scalar、非 tile/descriptor 路径。

## Descriptor

推荐新增 descriptor 队列，让 CPU 一次提交多层任务。T5.1 已固定 descriptor v1 ABI：每个 descriptor 为 16 个 32-bit little-endian word，总长 64 byte，`DESC_BASE` 和 `next_desc` 必须 64-byte aligned。T5.2 已实现 AXI-Lite `DESC_BASE/DESC_COUNT` 和 `CTRL[7] desc_mode` 的写入/读回；T5.3 已实现第一版 controller fetch/decode/next-layer，可顺序执行多个 4x4 OS tile-pack GEMM descriptor。

```text
word0  desc_ctrl       // version/op/dtype/dataflow/shape/tile-packed/first-k/last-k/irq/psum/ifm-prev flags
word1  M
word2  N
word3  K
word4  ifm_addr        // GEMM_TILEPACK: A tile-pack stream base
word5  weight_addr     // GEMM_TILEPACK: W tile-pack stream base
word6  bias_addr
word7  psum_addr
word8  ofm_addr        // GEMM C/result base
word9  ifm_shape       // [15:0]=IH, [31:16]=IW
word10 ofm_shape       // [15:0]=OH, [31:16]=OW
word11 channel_shape   // [15:0]=Cin, [31:16]=Cout
word12 kernel_stride   // KH/KW/stride_h/stride_w, 8 bit each
word13 pad_dilation    // pad_h/pad_w/dilation_h/dilation_w, 8 bit each
word14 post_cfg        // activation/quant/out_shift
word15 next_desc       // next descriptor byte address, 0 means end
```

层结束条件：

```text
all m_tile done && all n_tile done && all k_tile done
```

网络结束条件：

```text
desc.last_layer == 1 or desc.next_desc == 0
```

`DESC_COUNT` 是链表 fetch 上限，用于防止 `next_desc` 跑飞；如果 count 耗尽但没有遇到 `last_layer` 或 `next_desc=0`，T5.5 会置位 `ERR_STATUS.DESC_COUNT_EXHAUSTED`。当前 descriptor v1 可直接映射到已有直配寄存器：`M/N/K` -> `M_DIM/N_DIM/K_DIM`，`ifm_addr` -> `A_ADDR`，`weight_addr` -> `W_ADDR`，`ofm_addr` -> `R_ADDR`，`desc_ctrl.dtype/dataflow/shape/tile_packed` -> `CTRL[3:2]`、`CTRL[5:4]`、`CFG_SHAPE` 和 `ARR_CFG[7]`。T5.4 增加 `desc_ctrl[23] IFM_FROM_PREV_OFM`：置位时本层 `A_ADDR` 由上一层 `ofm_addr` 覆盖，并由 DMA 执行 row-major OFM 到 A tile stream 的 gather/repack。当前支持并验证 `OP=GEMM_TILEPACK`、`DTYPE=INT8`、`DATAFLOW=OS`、`SHAPE=4x4`、`TILE_PACKED=1` 的两层串联；独立 descriptor 路径仍支持 INT8/FP16 4x4 OS tile-pack GEMM，unsupported descriptor 会置位 `ERR_STATUS.DESC_UNSUPPORTED`。T7.3/T7.4 的 INT8 2/4-lane SIMD 目前是 PE 级能力，descriptor/tile 主线尚未声明 32-bit packed K lane 供数语义。

## PE 阵列

### 基础形态

基础验收先实现真正的 4x4：

- 16 个 PE 同时参与计算。
- 每个逻辑 `k` 周期从 PPBuf 读取 4 个 activation lane 和 4 个 weight lane；activation 进入 PE row 前要做 row-skew，前几个物理周期不是 4 个 row 全部有效。
- 每个 tile 产生最多 16 个输出。
- 支持 INT8 和 FP16；INT8 PE 内已支持 packed 2/4-lane SIMD，并兼容旧 sign-extended scalar 输入。
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
- T7.3：PE 级 2-lane INT8 SIMD MAC，INT32 accumulate。
- T7.4：PE 级 4-lane INT8 SIMD MAC，INT32 accumulate。
- 性能优化剩余项：配套上游 32-bit packed K lane 供数、阵列 valid 对齐和写回。

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

因此 0.5-1 TOPS 目标要求 16x16 阵列配合 PE 内 INT8 SIMD。T7.3/T7.4 已提供 PE 级 2-lane/4-lane 理论 0.512/1.024 TOPS 基础；T7.5 已提供 `TOPS_X1E6`、compute/e2e utilization 和 peak ops/cycle 性能计数口径。端到端高吞吐还取决于后续 packed K lane 供数、结果收集和写回。

### 4-lane SIMD 加速方向

```text
单 lane PE（每拍 1 个 K 值）:
  K=4 需要 4 拍完成一个 C 元素的点积

4-lane PE（每拍 4 个 K 值）:
  K=4 只需 1 拍完成一个 C 元素的点积

4-lane 加速的是 K 方向（归约维度），不替代其他 PE。
```

以 4×4 阵列做 4×4×4 GEMM 为例：

```text
PE(r,c) → C[r,c] = Σ_k A[r,k]·W[k,c]

PE[0,0] 内部 4 lane:
  a_in = {A[0,3], A[0,2], A[0,1], A[0,0]}   // 同一行，不同 K
  w_in = {W[3,0], W[2,0], W[1,0], W[0,0]}   // 同一列，不同 K
  lane3·lane2·lane1·lane0 并行乘加 → C[0,0]

PE[0,1] 内部 4 lane（同时）:
  a_in = {A[0,3], A[0,2], A[0,1], A[0,0]}   // 与 PE[0,0] 相同的 A 行
  w_in = {W[3,1], W[2,1], W[1,1], W[0,1]}   // 不同的 W 列
  → C[0,1]
```

16 个 PE 并行算 16 个 C 元素（不同行列），每个 PE 内部 4 lane 并行算 4 个 K 位置。**总计 16×4=64 MAC/fire，等于 GEMM 总 MAC 数——没有空闲 PE，也没有空闲 lane。**

K 不足 4 的倍数时，末尾会用 pad 零填充废 lane（硬件自动，不产生多余乘积）。

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

active_rows = min(4, M - m0)	//GEMM行数，m_tile为Tile行向坐标，m0为本Tile之前的行数；即计算不能被4整除时，剩下的行数
active_cols = min(4, N - n0)	//同上
row_valid[r] = (r < active_rows)//考虑边界-->若active为3，则有r==0、1、2时，valid
col_valid[c] = (c < active_cols)//同上
out_valid[r,c] = row_valid[r] && col_valid[c]	//行列均有效时，输出有效
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
C_row_major(i,j) = R_ADDR + (i * N + j) * 4				//4是由于输出为了防止精度丢失，统一使用32bits
```

其中：

```text
elem_bytes = 1 for INT8
elem_bytes = 2 for FP16
```

当前 direct scalar 仿真为了让 DMA 以 32-bit word 读取，A 的每一行和 W 的每一列在 DRAM 中都按 4-byte 对齐保存：

```text
stride_bytes = align4(K * elem_bytes)
A_direct(i,k) = A_ADDR + i * stride_bytes + k * elem_bytes
W_direct(k,j) = W_ADDR + j * stride_bytes + k * elem_bytes
```

也就是说，direct scalar 的 W 物理布局是按列连续存放；T6.1/T6.2 的 `W_col[K,N]` 使用同一布局。T6.2 只把 A 侧从预展开 `A_im2col` 改为 raw IFM on-the-fly gather；T6.3-T6.5 的 bias/ReLU/ReLU6/INT8 quant 发生在 direct scalar 输出写入 result FIFO 前；T6.6 的两层 Conv2D E2E 复用该 32-bit word-aligned 输出布局；tile-pack 路径仍使用下面的 4-lane tile 流格式。

### Phase 2 片上 tile-pack 格式

为了先验证 PE 阵列和控制器，Phase 2 的 testbench/CPU 可以把 A/W 预打包为 lane-major tile 流，避免在 T2 同时实现 gather DMA。后续 on-the-fly im2col 或 row-major gather 可在 Phase 6/T3 之后补上。

A tile pack：

```text
A_TILE[m_tile][k][r] = row_valid[r] ? A[m0+r, k] : 0

r是PE阵列中，PE的行序号；
m_tile是该子矩阵在原矩阵中的位置
可以把m0理解为基地址，m0是之前已经处理过的行，+r就是PE的中第r行该输入的数据；
A的第k列的数据应该是同一时间步（由于错拍，所以可能不是同一时间）被输入到PE阵列中的，所以A_TILE[m_tile][k][r]的k是次高维；
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

# w_in[c] 是送入 column c 顶部的 W lane。
# act_in[r] 是已经过 row-skew 后送入 row r 左侧/边界的 A lane。
w_in[c]   = (t < K && col_valid[c]) ? W_TILE[n_tile][t][c] : 0
act_in[r] = (0 <= t-r < K && row_valid[r])
              ? A_TILE[m_tile][t-r][r]
              : 0
```

因此启动阶段有 bubble，不是第 0 拍就 4 个 row 全部有效。以 `active_rows=4` 为例：

```text
physical t | row0 act_in | row1 act_in | row2 act_in | row3 act_in
-----------|-------------|-------------|-------------|------------
0          | A[0,0]      | 0           | 0           | 0
1          | A[0,1]      | A[1,0]      | 0           | 0
2          | A[0,2]      | A[1,1]      | A[2,0]      | 0
3          | A[0,3]      | A[1,2]      | A[2,1]      | A[3,0]
...
K          | 0           | A[1,K-1]    | A[2,K-2]    | A[3,K-3]
```

`4-lane vector` 指 PPBuf/feeder 在逻辑 `k` 维度上每次取出 `A_TILE[k][0..3]` 和 `W_TILE[k][0..3]`；它不表示 4 个 A row 从第 0 个物理周期就同时送到 PE。

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
//这里可能只算完了左上角的一小块，例如算完了C[0,0]C[0,1]C[0,2]C[1,0]C[1,1]...C[2,2]，共九个元素
//存储时，C矩阵的第0行应该放在一起，所以以C[2,2]存储时为例
//其地址计算应该是跳过“2个完整行（也就是N（总列数）个元素）+自己的列数（2）”个元素
//m0+r就是以0为起始，跳过的总行数，此处即为2（因为此时的m0 n0都是0）
//×4和之前的原因相同
```

因此 serializer 不能简单假设 16 个 word 在全局 C 矩阵中总是连续。正确写回策略是：

```text
for r in 0..active_rows-1:
  write burst base = R_ADDR + ((m0+r) * N + n0) * 4
  write beats      = active_cols
  data order       = C[m0+r,n0], C[m0+r,n0+1], ...
```

当 `N == 4` 或 `active_cols == N` 时，这退化为一个连续 16 word 写回；一般矩阵需要每个有效 row 一个小 burst。

### PSUM/OUT_BUF 和 K-split 规格

PSUM 是 “partial sum”。只要一个输出元素 `C[i,j]` 还没有完成全部 K 方向累计，或者累计结果还没有经过最终后处理，它就不是最终 OFM，而是 PSUM。这个区分很重要：PE accumulator、片上 `PSUM/OUT_BUF`、外部 DRAM 里的 PSUM surface 都可能保存 PSUM，但它们的容量和生命周期不同。

T4.1 规定 `PSUM/OUT_BUF` 是 4x4 tile 级别的 accumulator 存储，不是整层 C 矩阵的片上缓存。大矩阵的完整 partial sum surface 仍放在外部 DRAM/SRAM，由 DMA 按 tile row 读写。

数据宽度：

```text
ACC_W = 32
PSUM_WORD_BYTES = 4

INT8 path:
  PSUM word = signed int32 accumulator, two's-complement bit pattern

FP16 path:
  PSUM word = FP32 accumulator bit pattern

OUT_BUF before post-process:
  same as PSUM word, 32-bit per C element
```

T4 阶段不在 `PSUM/OUT_BUF` 内做 INT8 量化打包。后续 `bias/ReLU/quant` 可以在 last k_tile 后把 32-bit OUT 转成目标输出格式。

片上 tile buffer 深度：

```text
TILE_M = 4
TILE_N = 4
PSUM_TILE_WORDS = TILE_M * TILE_N = 16
PSUM_TILE_BYTES = 16 * 4 = 64

PSUM_BANKS = 2
PSUM_BANK_WORDS >= 16
OUT_BANK_WORDS  >= 16
```

建议第一版将 `PSUM_BUF` 和 `OUT_BUF` 做成同一个 2-bank tile SRAM 的两种角色：一个 bank 可被 compute/serializer 使用，另一个 bank 可被 DMA 读写或准备下一 tile。后续扩展到 8x8/16x16/8x32 时，只把 `PSUM_BANK_WORDS` 参数扩为 `MAX_TILE_M * MAX_TILE_N`，地址公式不变。

片上 tile-local index：

```text
psum_idx(r,c) = r * TILE_N + c       // 0..15
valid(r,c)    = row_valid[r] && col_valid[c]
```

无效 lane 不读、不写、不参与累加。

外部 PSUM/OFM surface 地址模式：

```text
PSUM_ADDR(i,j) = PSUM_BASE + (i * N + j) * 4
OFM_ADDR(i,j)  = OFM_BASE  + (i * N + j) * 4   // T4 阶段仍写 32-bit accumulator
```

4x4 tile 的外部读写仍按 row burst：

```text
for r in 0 .. active_rows-1:
  psum_row_addr = PSUM_BASE + ((m0+r) * N + n0) * 4
  ofm_row_addr  = OFM_BASE  + ((m0+r) * N + n0) * 4
  beats         = active_cols
  data order    = [m0+r,n0], [m0+r,n0+1], ...
```

因此 `PSUM_BASE` 指向一个 row-major `M*N*4` byte surface。以 `M=N=1000` 为例，完整 PSUM surface 需要 `1000*1000*4 = 4,000,000` byte，不能放入片上 tile buffer。

K-split 计数：

```text
K_TILE_MAX_BYTES = PPB_DEPTH * 4
K_TILE_ELEMS_INT8 = K_TILE_MAX_BYTES / (4 * 1)  // default PPB_DEPTH=64 -> 64
K_TILE_ELEMS_FP16 = K_TILE_MAX_BYTES / (4 * 2)  // default PPB_DEPTH=64 -> 32

k_tile_count = ceil(K / K_TILE_ELEMS)
k0           = k_tile * K_TILE_ELEMS
k_len        = min(K_TILE_ELEMS, K - k0)
```

`A_TILE_ADDR` 和 `W_TILE_ADDR` 在 K-split 下加入 `k0`：

```text
A_TILE_ADDR(m_tile,k_tile,k) = A_ADDR + (m_tile * K + (k0+k)) * A_VEC_BYTES
W_TILE_ADDR(n_tile,k_tile,k) = W_ADDR + (n_tile * K + (k0+k)) * W_VEC_BYTES
```

其中 `k = 0 .. k_len-1`。

K-split 调度语义：

```text
for m_tile in 0 .. ceil(M/4)-1:
  for n_tile in 0 .. ceil(N/4)-1:
    for k_tile in 0 .. k_tile_count-1:
      load A/W tile slice for k0 .. k0+k_len-1

      if k_tile == 0:
        init psum_tile[r,c] = 0 or bias[n0+c]				//初始化
      else:
        read psum_tile[r,c] from PSUM_ADDR(m0+r,n0+c)		//读中间结果

      compute partial over k_len							//计算新的中间结果
      psum_tile[r,c] += partial[r,c]						//把旧中间结果和新中间结果累加（原理参照矩阵分块乘法）

      if k_tile + 1 < k_tile_count:
        write psum_tile to PSUM_ADDR(m0+r,n0+c)
      else:
        write final tile to OFM_ADDR(m0+r,n0+c)
```

T4.2 已实现独立 `rtl/buf/psum_out_buf.v`，采用 2-bank tile-local SRAM 加外部 read-modify-write 语义：buffer 只保存 32-bit accumulator bit pattern，不在内部区分 INT32/FP32 加法。`valid_mask` 负责边界 tile lane 过滤，`tile_clear_en` 可清空指定 bank。T4.3 已在 `pe_top`/`reconfig_pe_array` 中实现 accumulator init：`acc_init_en` 脉冲会把 `acc_init` 写入 PE 内部 accumulator，后续 MAC 从该初值继续累加。T4.4/T4.5 已由 controller 和顶层验证驱动 k_tile loop：同一个 C tile 的多个 k_tile 连续执行，中间 k_tile 不 flush、不写回，PE accumulator 保持部分和，最后 k_tile 才 flush/writeback；`tb_npu_tile_ksplit_gemm.v` 已验证 4x4x10 INT8 OS GEMM 按 4/4/2 切分后等于未切分 golden。外部 PSUM surface 的 read/write 仍留给后续 descriptor/多层调度接入；外部可见行为必须等价于上面的调度语义。

### T2.2/T2.3/T2.4 接口结论

后续 RTL 按以下接口推进：

```text
a_vec[4]       // PPBuf 输出的 A_TILE[k][r]；进入 PE row 前会做 row-skew
w_vec[4]       // PPBuf 输出的 W_TILE[k][c]
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

如果 K 被切分，OS 必须避免每个 K tile 都清零。T4.4 当前对同一个 C tile 连续执行所有 k_tile，让 PE accumulator 在中间 k_tile 之间保持部分和；T4.3 的 PE/array accumulator init 接口保留给后续从外部 `PSUM_BUF` 恢复部分和的场景。

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
| 4-lane A/W buffer | T2.2/T2.3 已增加 `rd_vec`/`rd_vec_en`，并由 controller 的 `vec_consume` 推进 | 保持回归 |
| 4x4 tile 计数和 mask | T2.3 已通过 `ARR_CFG[7]` 启用 tile planner，支持 M/N 边界 mask | 保持回归 |
| 4x4 真并行 tile | T2.4/T2.5/T2.6 已完成 serializer/writeback 和 4x4 INT8/FP16 golden 测试 | 保持回归 |
| 多层卷积 | descriptor v1 ABI、AXI-Lite 提交寄存器、controller fetch/decode/next-layer 和 T5.4 INT8 OFM->IFM 串联已具备；T6.2 已有 direct scalar raw IFM on-the-fly im2col；T6.3-T6.5 已有 direct scalar bias、ReLU/ReLU6 和 INT8 quant/saturate；T6.6 已验证 direct scalar 两层 Conv2D E2E；单 tile 内 k_tile loop 和顶层 K-split GEMM golden 已完成，外部 PSUM surface 还未接入 descriptor 流 | 增加外部 PSUM read/write，并把 Conv2D im2col 与后处理接入 tile/descriptor 主线 |
| AXI burst | T3.1-T3.5 已完成读写通道 INCR burst、4KB 边界切分、AXI perf counters、混合正确性测试和带宽目标报告 | 后续若冲更高 write util，需多 outstanding 或 B response 重叠 |
| 16x16/8x32 | 8x8/16x16 active lane 供数已验证；8x32 阵列级折叠路由和 32-lane 输出顺序已验证；PE 级 INT8 2/4-lane SIMD 已验证 | 补齐 32-bit packed K lane 供数、更大 tile 的 valid 对齐、边界 mask 和顶层写回 |
| 低功耗 | 行为模块存在但未接入 | 使用 clock enable/BUFGCE/ICG 接入 |
| SoC 验证 | `run_soc_sim.ps1` 已通过；PicoRV32 配置 NPU 完成 2x2 INT8 GEMM，结果 `19,22,43,50` | 扩展到 descriptor/Conv2D 系统级 smoke |
