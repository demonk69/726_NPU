# 卷积到 GEMM 的映射

更新时间：2026-06-15

适用于普通 dense Conv2D（`groups=1`）：把每个 batch 中的每个输出空间位置展开成 GEMM 的一行，把每个输出通道的卷积核展开成 GEMM 的一列。

需要补充的边界条件是：`OH/OW` 由 stride、padding、dilation 决定；bias、ReLU、量化属于 GEMM 后处理；group/depthwise 卷积需要按 group 分开映射，不能直接套一个全局 `Cin * KH * KW`。

## 基本公式

```text
# A_im2col[M,K] * W_col[K,N] = C[M,N]
#
# M: GEMM 行数。卷积中表示所有输出空间位置的总数。
# K: GEMM 归约维度。卷积中表示一个卷积窗口内参与点积的输入元素数。
# N: GEMM 列数。卷积中表示输出通道数，也就是卷积核个数。
M = batch * OH * OW
K = Cin * KH * KW
N = Cout
```

| 变量 | 含义 |
|---|---|
| `batch` | 一次处理的样本数 |
| `IH/IW` | 输入特征图高度/宽度 |
| `OH/OW` | 输出特征图高度/宽度 |
| `Cin` | 输入通道数，例如 RGB 输入的 `Cin=3` |
| `Cout` | 输出通道数，也等于卷积核个数 |
| `KH/KW` | 卷积核高度/宽度 |
| `stride_h/stride_w` | 卷积窗口在高/宽方向每次移动的步长 |
| `pad_h/pad_w` | 输入特征图高/宽方向两侧补零数量 |
| `dilation_h/dilation_w` | 卷积核采样间隔，普通卷积为 1 |

输出尺寸：

```text
# floor 表示向下取整。
# 可以理解为卷积核能滑动多少次，结果的行/列就有多少个元素-1
#（-1是因为初始位置还有一个元素不因滑动而产生，所以公式最后加上了这个“1”）
# 滑动次数的计算即行/列方向的总元素数-卷积核本身的大小，再除以步长
OH = floor((IH + 2*pad_h - dilation_h*(KH-1) - 1) / stride_h) + 1
OW = floor((IW + 2*pad_w - dilation_w*(KW-1) - 1) / stride_w) + 1
```

## 矩阵含义

| 矩阵 | 形状 | 含义 |
|---|---:|---|
| `A_im2col` | `[M,K]` | 每一行是生成一个输出像素点所需的完整输入窗口，按 `Cin/KH/KW` 展平 |
| `W_col` | `[K,N]` | 每一列是一个输出通道的卷积核，展平顺序必须和 `A_im2col` 的 K 维一致 |
| `C` | `[M,N]` | GEMM 输出；可按布局恢复成 `[batch,OH,OW,Cout]` 或 `[batch,Cout,OH,OW]` |

## 索引映射

下面用 NHWC/NCHW 混合说明逻辑关系，实际 RTL/testbench 只关心最终的 `A[M,K]`、`W[K,N]`、`C[M,N]` 地址顺序。

```text
# b:   batch 内的样本编号，范围 0..batch-1
# oh:  输出特征图行坐标，范围 0..OH-1
# ow:  输出特征图列坐标，范围 0..OW-1
# cin: 输入通道编号，范围 0..Cin-1
# kh:  卷积核行坐标，范围 0..KH-1
# kw:  卷积核列坐标，范围 0..KW-1
# cout: 输出通道编号，范围 0..Cout-1

m = (b * OH + oh) * OW + ow
k = (cin * KH + kh) * KW + kw
n = cout

# 卷积窗口映射回输入特征图坐标。
ih = oh * stride_h + kh * dilation_h - pad_h
iw = ow * stride_w + kw * dilation_w - pad_w

# 超出输入边界的位置来自 padding，值为 0。
A_im2col[m,k] = (0 <= ih < IH and 0 <= iw < IW) ? IFM[b,cin,ih,iw] : 0
W_col[k,n]    = WEIGHT[cout,cin,kh,kw]
C[m,n]        = sum_k A_im2col[m,k] * W_col[k,n]
Y[m,n]        = quant(activation(C[m,n] + bias[n]))  // optional T6.3-T6.5 direct scalar postprocess
```

如果输出存为 NHWC：

```text
OFM[b,oh,ow,cout] = C[(b * OH + oh) * OW + ow, cout]
```

如果输出存为 NCHW：

```text
OFM[b,cout,oh,ow] = C[(b * OH + oh) * OW + ow, cout]
```

## GEMM 验证数据流

当前 tile GEMM 回归入口是 `tb/tile4/run_verilator.sh`。它不是只验证 4x4，而是用同一套 tile 数据格式覆盖 `4x4`、`8x8`、`16x16`、`8x32` 多种 shape。

验证数据流如下：

```text
Python case generator
  |
  +-- 生成 dense A[M,K]、W[K,N]、可选 bias[N]
  +-- 计算 golden C[M,N]
  +-- 按 RTL tile stream 规则把 W/A 打包到 dram_init.hex
  +-- 写 test_params.vh 和 expected.hex

Verilator/Icarus testbench
  |
  +-- 加载 dram_init.hex 到 DRAM model
  +-- 写 NPU 寄存器：M/N/K、W/A/R 地址、ARR_CFG、CFG_SHAPE、bias
  +-- 等待 NPU done
  +-- 从 DRAM result 区读取 C tile 写回结果
  +-- 与 expected.hex 比较
```

RTL 内部数据流如下：

```text
DRAM packed W stream -> npu_dma -> W pingpong_buf -> W vector -> PE array
DRAM packed A stream -> npu_dma -> A pingpong_buf -> A vector -> PE array
PE array accumulates INT32 C tile -> serializer/postprocess -> result FIFO -> DRAM
```

## 当前 RTL 的 Tile-Pack 关系

当前维护主线是 direct-register tile GEMM：firmware 或 testbench 把 W/A tile stream 放入 DRAM，配置 `M/N/K`、W/A/R 地址、`ARR_CFG[7]` tile mode 和 `CFG_SHAPE`，RTL DMA 把 W/A stream 拉入 PPBuf，PE array 计算 INT32 tile 结果并写回 DRAM。direct scalar im2col 路径仍保留，用于 legacy smoke 和 DMA 地址发生器覆盖。

本文有两种 lane，需要区分：

| 名称 | 含义 |
|---|---|
| tile lane | tile 内的 row/col lane，例如 4x4 shape 有 4 个 row lane 和 4 个 col lane |
| INT8 SIMD lane | 每个 PE 输入 word 内沿 K 维打包的 INT8 lane，默认 `INT8_SIMD_LANES=4` |

当前 4x4 tile-pack 约定：

```text
# m0/n0 是当前 4x4 输出 tile 的左上角坐标。
# r/c 是 tile 内部行列坐标，范围 0..3。
# k 是 GEMM 归约维度坐标，范围 0..K-1。
A_TILE[m_tile][k][r] = A[m0+r, k]
W_TILE[n_tile][k][c] = W[k, n0+c]

# 阵列输出按 row-major 暴露给 serializer。
result_index = r * 4 + c
result[result_index] = C[m0+r, n0+c]
```

注意这里的 tile lane 是 PPBuf/feeder 每个逻辑 `k` 周期取出的 row/col 向量，不表示 4 个 A row 在第 0 个物理周期同时进入 PE。当前 OS feeder 会对 A row 做错拍：

```text
# t 是物理周期；r 是 tile 内 row。
# row r 比 row 0 晚 r 拍收到同一个逻辑 k 的 A 值。
act_in[r,t] = (0 <= t-r < K) ? A_TILE[m_tile][t-r][r] : 0
w_seen_by_row_r[c,t] = (0 <= t-r < K) ? W_TILE[n_tile][t-r][c] : 0
```

所以 PE(r,c) 在 `t = k + r` 时同时看到 `A[m0+r,k]` 和 `W[k,n0+c]`。如果 A row 不做这个错拍，row 1/2/3 会把 `A[m0+r,k]` 和错误的 `W[k-r,n0+c]` 对上，结果会错。

## 4-Lane INT8 SIMD 打包

默认 Vivado/NPU 配置是 `DATA_W=32`、`INT8_SIMD_LANES=4`。每个 PE 在一个计算周期内处理同一个输出元素的 4 个 K 维 INT8 乘法，并把 4 个乘积求和后累加到 INT32 accumulator。

对一个 PE(r,c) 来说，逻辑计算是：

```text
C[m0+r,n0+c] = sum_k A[m0+r,k] * W[k,n0+c]
```

在 4-lane 模式下，K 维按 4 个一组进入 PE：

```text
cycle g handles k = 4*g + 0, 4*g + 1, 4*g + 2, 4*g + 3

a_in = {A[m,4*g+3], A[m,4*g+2], A[m,4*g+1], A[m,4*g+0]}
w_in = {W[4*g+3,n], W[4*g+2,n], W[4*g+1,n], W[4*g+0,n]}

PE adds:
  A[m,4*g+0] * W[4*g+0,n]
+ A[m,4*g+1] * W[4*g+1,n]
+ A[m,4*g+2] * W[4*g+2,n]
+ A[m,4*g+3] * W[4*g+3,n]
```

如果 `K` 不是 4 的倍数，最后一个 SIMD group 用 0 补齐。补零只影响无效 lane，不改变 golden 结果。

## 非整倍数矩阵如何部署到 4-Lane/Tile 上

矩阵尺寸不需要是 4、8、16 的倍数。controller 和 testbench generator 按 tile 形状做 ceil 切分，并用 valid mask 和 zero padding 处理边界。

通用规则：

```text
tile_rows = shape rows, e.g. 4 for 4x4, 16 for 16x16
tile_cols = shape cols, e.g. 4 for 4x4, 32 for 8x32

num_m_tiles = ceil(M / tile_rows)
num_n_tiles = ceil(N / tile_cols)
num_k_groups = ceil(K / INT8_SIMD_LANES)

valid_row(r) = (m0 + r) < M
valid_col(c) = (n0 + c) < N
valid_k_lane(l) = (4*g + l) < K
```

无效 row、col、K lane 的处理：

| 无效类型 | 处理方式 | 写回行为 |
|---|---|---|
| `m0+r >= M` | A lane 填 0，PE 结果忽略 | 不写回该 row |
| `n0+c >= N` | W lane 填 0，PE 结果忽略 | 不写回该 col |
| `4*g+l >= K` | A/W 的该 SIMD lane 填 0 | 该 lane 对 sum 无贡献 |

### 简单例子：`M=5, K=7, N=6` 跑在 4x4 shape 上

4x4 shape 的 tile rows=4、tile cols=4、INT8 lanes=4。

```text
num_m_tiles = ceil(5/4) = 2
num_n_tiles = ceil(6/4) = 2
num_k_groups = ceil(7/4) = 2
```

Tile 划分：

| Tile | m0 | n0 | 有效 rows | 有效 cols |
|---|---:|---:|---:|---:|
| T00 | 0 | 0 | rows 0..3 | cols 0..3 |
| T01 | 0 | 4 | rows 0..3 | cols 4..5 |
| T10 | 4 | 0 | row 4 | cols 0..3 |
| T11 | 4 | 4 | row 4 | cols 4..5 |

K 维分组：

| K group | 有效 K | padded lane |
|---:|---|---|
| 0 | 0,1,2,3 | none |
| 1 | 4,5,6 | lane for K=7 is 0 |

以 T11 为例，PE 阵列仍然按 4x4 计算，但只有 `r=0` 和 `c=0,1` 的输出有效：

```text
PE(0,0) -> C[4,4]
PE(0,1) -> C[4,5]
PE(0,2) -> invalid, n=6 out of range
PE(0,3) -> invalid, n=7 out of range
PE(1..3,*) -> invalid, m=5..7 out of range
```

每个有效 PE 只跑两个 4-lane SIMD cycles：

```text
cycle 0: k=0,1,2,3
cycle 1: k=4,5,6,pad0
```

结果写回时只写 `C[4,4]` 和 `C[4,5]`，不会写越界位置。

### 16x16 和 8x32 的边界行为

16x16 的边界规则相同，只是 `tile_rows=16`、`tile_cols=16`。例如 `M=17,N=18` 会产生 2x2 个 tile，右下 tile 只有 1 行和 2 列有效。

8x32 shape 的逻辑 tile 是 8 行 x 32 列。物理阵列仍是 16x16，RTL 用两 pass 覆盖 32 列：

```text
pass 0: logical cols n0+0  .. n0+15
pass 1: logical cols n0+16 .. n0+31
```

如果 `N` 不足 32 或右边界不足 32，pass 内的无效 col 同样通过 valid mask 不写回。

T6.2 已把上面的 `A_im2col[m,k]` 地址计算搬进 direct scalar DMA 地址发生器，避免在 DRAM 中保存完整展开矩阵；T6.3-T6.5 已把 direct scalar 的 `C[m,n]` 后处理扩展到 bias、ReLU/ReLU6 和 INT8 quant/saturate；T6.6 已完成 direct scalar 两层 Conv2D E2E。当前维护重点是 tile/direct-register VGG 路径和 closed-loop runtime packing。
