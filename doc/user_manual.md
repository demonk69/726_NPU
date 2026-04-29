# 使用说明

更新时间：2026-04-28

本文面向 CPU 侧软件和 testbench。当前实现还处于原型阶段，寄存器直配模式可用于调试；多层卷积建议按 descriptor 模式设计。

## 当前可用能力

可以依赖：

- 单 PE INT8/FP16 MAC。
- AXI-Lite 寄存器读写。
- PPBuf 对 INT8/FP16 数据拆包。
- 顶层标量 INT8 dot product 兼容路径，可用于 Phase 1 回归。
- `ARR_CFG[7]=1` 的 4x4 tile-mode GEMM 路径，已覆盖 INT8/FP16 4x4x4 golden 测试。

不能依赖：

- 16x16/8x32 高吞吐。
- DMA 读通道 burst。
- 多层卷积自动调度。
- INT16。

## 寄存器直配模式

NPU 基地址建议：

```text
0x0200_0000
```

寄存器：

| 偏移 | 名称 | 作用 |
|---:|---|---|
| `0x00` | `CTRL` | start/abort/mode/stat_mode/irq clear |
| `0x04` | `STATUS` | busy/done |
| `0x08` | `INT_EN` | 中断使能 |
| `0x0C` | `INT_CLR/PENDING` | 清 pending / 读 pending |
| `0x10` | `M_DIM` | GEMM 行数；卷积中为 `batch*OH*OW` |
| `0x14` | `N_DIM` | GEMM 列数；卷积中为 `Cout` |
| `0x18` | `K_DIM` | 归约维度；卷积中为 `Cin*KH*KW` |
| `0x20` | `W_ADDR` | weight/B 基地址；tile mode 下指向 W tile-pack 流 |
| `0x24` | `A_ADDR` | activation/A 基地址；tile mode 下指向 A tile-pack 流 |
| `0x28` | `R_ADDR` | result/C 基地址；结果按 32-bit word 写回 |
| `0x30` | `ARR_CFG` | bit7=4x4 tile mode enable |
| `0x34` | `CLK_DIV` | DFS 配置，当前未接入 PE 时钟 |
| `0x38` | `CG_EN` | clock gating 配置 |
| `0x3C` | `CFG_SHAPE` | 00=4x4, 01=8x8, 10=16x16, 11=8x32 |

`CTRL` 编码：

```text
bit0     start
bit1     abort
bit[3:2] data mode: 00=INT8, 10=FP16
bit[5:4] dataflow: 00=WS, 01=OS
bit6     irq clear
```

常用值：

| 模式 | 值 |
|---|---:|
| INT8 + WS + start | `0x01` |
| INT8 + OS + start | `0x11` |
| FP16 + WS + start | `0x09` |
| FP16 + OS + start | `0x19` |

## 当前 GEMM 数据布局意图

当前控制器意图表达：

```text
# M: C 的行数，也是 A 的行数。
# N: C 的列数，也是 B/W 的列数。
# K: A 的列数，也是 B/W 的行数。
C[M,N] = A[M,K] * B[K,N]
```

布局：

- A：行主序。
- B/W：列主序。
- C：行主序，每个输出 32-bit。

地址：

```text
A(i,:) = A_ADDR + i * K * element_bytes
B(:,j) = W_ADDR + j * K * element_bytes
C(i,j) = R_ADDR + (i * N + j) * 4
```

注意：非 tile mode 保留 Phase 1 标量兼容路径；tile mode 使用下面的 4x4 tile-pack 格式。当前已验证 4x4 INT8/FP16 GEMM，16x16/8x32 高吞吐仍未完成。

## 4x4 Tile-Pack 约定

T2.1 之后，4x4 GEMM 测试优先使用预打包 A/W tile，而不是直接让 DMA 从 row-major 矩阵中 gather。

OS 模式逻辑映射：

```text
PE row r -> M lane r -> C[m0+r,*]
PE col c -> N lane c -> C[*,n0+c]
PE(r,c)  -> C[m0+r,n0+c]
```

A/W tile-pack：

```text
# m_tile/n_tile: 当前 4x4 输出 tile 在 M/N 方向的编号。
# m0/n0: 当前 tile 左上角坐标，m0=m_tile*4，n0=n_tile*4。
# k: GEMM 归约维度坐标。
# r/c: tile 内部行/列 lane 编号，范围 0..3。
A_TILE[m_tile][k][r] = A[m0+r,k]
W_TILE[n_tile][k][c] = W[k,n0+c]
```

INT8 packed word：

```text
word[ 7: 0] = lane0
word[15: 8] = lane1
word[23:16] = lane2
word[31:24] = lane3
```

FP16 packed words：

```text
word0[15: 0] = lane0
word0[31:16] = lane1
word1[15: 0] = lane2
word1[31:16] = lane3
```

C 输出仍按 row-major 地址检查：

```text
C_ADDR(r,c) = R_ADDR + ((m0+r) * N + (n0+c)) * 4
```

## Descriptor 模式规划

多层网络应使用 descriptor，而不是 CPU 每层手工轮询配置。

建议 descriptor 以 32-bit word 对齐：

| Word | 字段 | 说明 |
|---:|---|---|
| 0 | `op_dtype_flow_flags` | 操作类型、数据类型、WS/OS、first/last/irq 等 flags |
| 1 | `M` | GEMM M；卷积中为 `batch*OH*OW` |
| 2 | `N` | GEMM N；卷积中为 `Cout` |
| 3 | `K` | GEMM reduce 维度；卷积中为 `Cin*KH*KW` |
| 4 | `ifm_addr` | 输入特征图基地址，或预展开/预打包 A 基地址 |
| 5 | `weight_addr` | 权重/卷积核基地址 |
| 6 | `bias_addr` | bias 基地址，可为 0 |
| 7 | `psum_addr` | K-split 部分和基地址，可为 0 |
| 8 | `ofm_addr` | 输出特征图或 GEMM C 基地址 |
| 9 | `shape0` | `IH/IW/OH/OW` 或 tile-pack 索引参数 |
| 10 | `shape1` | `Cin/Cout/KH/KW` |
| 11 | `stride_pad_act` | stride、padding、dilation、activation 等后处理配置 |
| 12 | `next_desc` | 下一个 descriptor 地址，0 表示结束 |

后续 AXI-Lite 建议新增：

| 偏移 | 名称 | 说明 |
|---:|---|---|
| `0x40` | `DESC_BASE` | descriptor 链表基地址 |
| `0x44` | `DESC_COUNT` | descriptor 数量 |
| `0x50` | `ERR_STATUS` | 错误状态 |

CPU 提交流程：

```c
prepare_desc_list(desc_base);
npu_write(DESC_BASE, desc_base);
npu_write(DESC_COUNT, count);
npu_write(CTRL, START | DESC_MODE);
while ((npu_read(STATUS) & DONE) == 0) {}
```

## 多层卷积的数据生命周期

```text
Layer0:
  IFM0 + W0 -> GEMM/CONV -> OUT0

Layer1:
  IFM1 = OUT0
  IFM1 + W1 -> GEMM/CONV -> OUT1
```

如果 K 被切分：

```text
first k_tile:
  clear psum or load bias
middle k_tile:
  read psum -> accumulate -> write psum
last k_tile:
  read psum -> accumulate -> activation -> write ofm
```

因此 NPU 需要 `PSUM/OUT_BUF`，不能只靠 W/A ping-pong buffer。

## 推荐当前使用方式

当前如果要继续开发，先保持以下回归通过：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

同时运行 `tb_npu_scalar_smoke.v`、`tb_pingpong_buf_vec.v`、`tb_npu_ctrl_tile.v`、`tb_npu_tile_writeback.v`、`tb_npu_tile_gemm.v` 和 `tb_comprehensive.v`。然后按 [task_breakdown.md](task_breakdown.md) 进入 Phase 3 的 AXI burst DMA 和带宽统计。不要先写多层卷积固件，因为 NPU 还没有 descriptor、PSUM/OUT buffer 和 K-split 支持。
