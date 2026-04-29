# 卷积到 GEMM 的映射

更新时间：2026-04-28

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
```

如果输出存为 NHWC：

```text
OFM[b,oh,ow,cout] = C[(b * OH + oh) * OW + ow, cout]
```

如果输出存为 NCHW：

```text
OFM[b,cout,oh,ow] = C[(b * OH + oh) * OW + ow, cout]
```

## 和当前 RTL 的关系

当前 T2.4-T2.6 已验证的是 4x4 GEMM tile 核心，不是完整硬件 im2col 前端。也就是说，testbench/脚本先把卷积或矩阵数据准备成 `A[M,K]` 和 `W[K,N]` 的 tile-pack 流，再交给 NPU 计算。

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

注意这里的 `4-lane` 是 PPBuf/feeder 每个逻辑 `k` 周期取出的向量，不表示 4 个 A row 在第 0 个物理周期同时进入 PE。当前 OS feeder 会对 A row 做错拍：

```text
# t 是物理周期；r 是 tile 内 row。
# row r 比 row 0 晚 r 拍收到同一个逻辑 k 的 A 值。
act_in[r,t] = (0 <= t-r < K) ? A_TILE[m_tile][t-r][r] : 0
w_seen_by_row_r[c,t] = (0 <= t-r < K) ? W_TILE[n_tile][t-r][c] : 0
```

所以 PE(r,c) 在 `t = k + r` 时同时看到 `A[m0+r,k]` 和 `W[k,n0+c]`。如果 A row 不做这个错拍，row 1/2/3 会把 `A[m0+r,k]` 和错误的 `W[k-r,n0+c]` 对上，结果会错。

后续 T6 的 on-the-fly im2col 会把上面的 `A_im2col[m,k]` 地址计算搬进硬件地址发生器，避免在 DRAM 中保存完整展开矩阵。
