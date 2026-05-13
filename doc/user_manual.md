# 使用说明

更新时间：2026-05-03

本文面向 CPU 侧软件和 testbench。当前实现还处于原型阶段，寄存器直配模式可用于调试；多层卷积建议按 descriptor 模式设计。

`.pth` 推理原型的模型子集、数据布局和 CPU/NPU 分工见 [pth_inference_subset.md](pth_inference_subset.md)。该路径当前定义为 host Python 离线转换 `.pth + model_spec.json`，参考 CPU 用 direct register mode 逐层调度 NPU。

当前已有一个最小 SoC 闭环入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pth_tiny_conv_soc.ps1
```

该入口自动生成 tiny quantized `.pth`，转换为 NPU 权重/DRAM image，并由参考 CPU 固件调度 NPU 执行 `Conv2D + ReLU`。

三层小模型闭环入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pth_multilayer_soc.ps1
```

该入口额外验证参考 CPU 连续调度多个 NPU Conv/ReLU 层，并在层间执行 NPU 输出到下一层 NCHW int8 输入的 repack。

RepOpt VGG host 整网基准入口：

```powershell
python tools\pth\run_repopt_vgg_host.py `
  --index 0 `
  --out-json sim\pth_repopt_host_run\host_run_idx0.json
```

该入口不调用 PyTorch quantized Conv 后端，而是按当前 V1 CPU/NPU 分工解释执行整网，用于生成后续 RTL/SoC 分段验证的 golden。

图片输入和全网 4x4 tile 调度入口：

```powershell
python tools\pth\run_repopt_vgg_host.py `
  --image path\to\image.png `
  --conv-backend tile4 `
  --out-json sim\pth_repopt_host_run\image_tile4.json
```

该入口会把图片 resize 到 `32x32`，按 CIFAR-10 mean/std 归一化并量化，然后输出 10 类分类结果。`tile4` backend 会遍历所有 Conv 层的 4x4 GEMM tile，CPU 负责 bias/ReLU、per-channel requant、pooling 和 classifier。

RepOpt VGG 第一层 RTL 分段验证入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_layer_case.ps1 -Index 0
```

默认验证 `stage1_0_conv` 的 `4x4` 输出窗口；当前已通过 `ALL 1024 CHECKS PASSED`。该入口比较的是 `Conv2D + bias + ReLU` 后的 int32 accumulator，CPU per-channel requant 仍在后续软件步骤中完成。

RepOpt VGG 第一层 4x4 tile-mode 入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_case.ps1 `
  -Index 0 -MBase 0 -NBase 0
```

该入口启用 `ARR_CFG[7]`，把真实 Conv 数据预打包为一个局部 `4x27x4` GEMM tile；当前已通过 `ALL 16 CHECKS PASSED`。注意该 tile-mode case 比较 raw int32 MAC accumulator，不包含 bias/ReLU。

多个 RTL tile 拼接和 CPU 后处理入口：

```powershell
python tools\pth\run_repopt_layer_tile_rtl.py `
  --index 0 `
  --m-base 0 --n-base 0 `
  --m-tiles 2 --n-tiles 2 `
  --out-json sim\pth_repopt_tile_rtl\stage1_0_m0_n0_2x2.json
```

该入口会连续运行多个真实 RTL tile case，收集 `npu_output.hex`，拼接 raw accumulator window，并由 CPU 执行 `bias/ReLU/per-channel requant`。当前 `2x2` tile window 已通过，并与 host 第一层 golden 对齐。

参考 CPU 固件/MMIO 调度多个 tile 并执行后处理的 SoC 入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 `
  -Index 0 -MBase 0 -NBase 0 -MTiles 2 -NTiles 2
```

该入口在一次 RTL 仿真中由参考 CPU 连续配置 4 个 NPU tile，并由固件执行 bias/ReLU/per-channel 固定点 requant 后写回 qint8 window。testbench 同时校验 raw int32 MAC 输出和固件后处理输出；当前已通过 `RepOpt tile-window SoC MMIO + CPU postprocess test PASSED`。

完整第一层窗口使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 `
  -Index 0 -FullLayer
```

该命令运行 `M[0:1024) N[0:64)` 的 4096 个 tile；当前完整第一层已通过，周期数为 8851650，固件仍为 91 words，DRAM 自动扩到 141312 words。若只想检查生成和编译，可加 `-CompileOnly`。

## 当前可用能力

可以依赖：

- 单 PE INT8/FP16 MAC；INT8 PE 内支持 packed 2/4-lane SIMD，并兼容旧 sign-extended scalar 输入。
- AXI-Lite 寄存器读写。
- PPBuf 对 INT8/FP16 数据拆包。
- 顶层标量 INT8 dot product 兼容路径，可用于 Phase 1 回归。
- `ARR_CFG[7]=1` 的 4x4 tile-mode GEMM 路径，已覆盖 INT8/FP16 4x4x4 golden 测试。
- DMA 读通道 INCR burst，当前覆盖 W/A 读请求。
- DMA 写通道多 INCR burst 和 4KB 边界切分。
- 混合读写场景下 8/16 beat burst 正确性测试。
- 长 burst 带宽利用率目标测试。
- AXI perf counters，可通过 AXI-Lite `0x48..0x70` 读出 burst/beat/byte/cycle、带宽和利用率。
- 独立 `psum_out_buf` 模块，支持 4x4 tile PSUM/OUT read-modify-write、边界 mask 和双 bank 隔离。
- `pe_top`/`reconfig_pe_array` accumulator init，支持从 PSUM 初值继续 MAC。
- `npu_ctrl` k_tile loop，支持 K 超过 PPB 深度时按 K slice 加载 A/W，并在最后 k_tile 写回。
- Descriptor v1 二进制格式已固定。
- AXI-Lite `DESC_BASE(0x40)`、`DESC_COUNT(0x44)` 和 `CTRL[7] desc_mode` 可写入并读回。
- 第一版 descriptor fetch/decode/next-layer 已接入：支持多个 `GEMM_TILEPACK + INT8/FP16 + OS + 4x4 + TILE_PACKED` descriptor 顺序执行。
- T5.4 已支持 descriptor 链中后一层使用上一层 32-bit row-major OFM 作为 IFM：`desc_ctrl[23]=IFM_FROM_PREV_OFM` 时，DMA 会把上一层 OFM gather/repack 成当前 4-lane A tile stream。当前已验证 INT8 4x4 GEMM 串联。
- T6.2 已支持 direct scalar on-the-fly Conv2D im2col：`CTRL[8]` 置位后，DRAM 只需保存 raw NCHW IFM 和 `W_col`，DMA 按卷积参数生成 A 行并写入 A PPBuf。
- T6.3-T6.6 已支持 direct scalar 32-bit bias、ReLU/ReLU6、INT8 quant/saturate 和两层 Conv2D E2E：`BIAS_ADDR(0x98)` 指向每输出列一个 32-bit bias word，`CTRL[9]` 启用 bias，`CTRL[11:10]` 选择 activation，`QUANT_CFG(0x9C)` 配置量化；T6.6 验证 layer0 量化 OFM 可直接作为 layer1 输入。

不能依赖：

- 16x16/8x32 端到端高吞吐写回；T7.3/T7.4 只完成 PE 级 2/4-lane SIMD，不等于整机 2x/4x 吞吐已验证。
- 多层卷积自动调度。
- FP16 OFM 自动转换成下一层 FP16 IFM；这需要后续 post-process/format conversion。
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
| `0x04` | `STATUS` | busy/done/error |
| `0x08` | `INT_EN` | bit0=done IRQ enable, bit1=error IRQ enable |
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
| `0x40` | `DESC_BASE` | descriptor 链表基地址 |
| `0x44` | `DESC_COUNT` | descriptor 数量 / fetch 上限 |
| `0x48` | `PERF_CYCLES` | monitor cycles since reset |
| `0x4C` | `PERF_RD_BEATS` | DMA AXI read data beats |
| `0x50` | `PERF_WR_BEATS` | DMA AXI write data beats |
| `0x54` | `PERF_RD_BYTES` | DMA AXI read bytes |
| `0x58` | `PERF_WR_BYTES` | DMA AXI write bytes |
| `0x5C` | `PERF_RD_BW` | read bytes/cycle x1000 |
| `0x60` | `PERF_WR_BW` | write bytes/cycle x1000 |
| `0x64` | `PERF_RD_UTIL` | read data-channel utilization, basis points |
| `0x68` | `PERF_WR_UTIL` | write data-channel utilization, basis points |
| `0x6C` | `PERF_RD_BURSTS` | DMA AXI read burst count |
| `0x70` | `PERF_WR_BURSTS` | DMA AXI write burst count |
| `0x74` | `ERR_STATUS` | W1C error status |
| `0x80` | `CONV_IFM_SHAPE` | `[15:0]=IH, [31:16]=IW` |
| `0x84` | `CONV_CHANNELS` | `[15:0]=Cin, [31:16]=Batch` |
| `0x88` | `CONV_KERNEL` | `[15:0]=KH, [31:16]=KW` |
| `0x8C` | `CONV_OUT_SHAPE` | `[15:0]=OH, [31:16]=OW` |
| `0x90` | `CONV_STRIDE_PAD` | `[7:0]=stride_h, [15:8]=stride_w, [23:16]=pad_h, [31:24]=pad_w` |
| `0x94` | `CONV_DILATION` | `[7:0]=dilation_h, [15:8]=dilation_w` |
| `0x98` | `BIAS_ADDR` | direct scalar bias vector base，每输出列一个 32-bit word |
| `0x9C` | `QUANT_CFG` | direct scalar INT8 quant：bit0 enable，bit1 round，`[15:8]` right shift，`[31:16]` signed scale |
| `0xA0` | `PERF_MAC_OPS_LO` | useful MAC operations low word |
| `0xA4` | `PERF_MAC_OPS_HI` | useful MAC operations high word |
| `0xA8` | `PERF_OPS_LO` | useful operations low word，1 MAC = 2 ops |
| `0xAC` | `PERF_OPS_HI` | useful operations high word |
| `0xB0` | `PERF_BUSY_CYCLES` | NPU busy cycles |
| `0xB4` | `PERF_COMPUTE_CYCLES` | compute-active cycles |
| `0xB8` | `PERF_DMA_CYCLES` | busy cycles not in compute |
| `0xBC` | `PERF_TOPS_X1E6` | TOPS fixed point，`TOPS * 1,000,000` |
| `0xC0` | `PERF_COMPUTE_UTIL` | compute utilization，basis points |
| `0xC4` | `PERF_E2E_UTIL` | end-to-end utilization，basis points |
| `0xC8` | `PERF_PEAK_OPS_CYC` | peak operations per cycle used by utilization |

`CTRL` 编码：

```text
bit0     start
bit1     abort
bit[3:2] data mode: 00=INT8, 10=FP16
bit[5:4] dataflow: 00=WS, 01=OS
bit6     irq clear
bit7     desc_mode: 0=direct register mode, 1=descriptor mode; T5.3 起由 controller fetch/decode descriptor
bit8     conv_im2col: direct scalar on-the-fly Conv2D im2col enable; tile/descriptor mode 当前忽略
bit9     bias_en: direct scalar 32-bit bias enable; tile/descriptor mode 当前忽略
bit[11:10] activation: 00=none, 01=ReLU, 10=ReLU6; tile/descriptor mode 当前忽略
```

`STATUS` 编码：

```text
bit0 busy
bit1 done
bit2 error
```

`ERR_STATUS` 编码：

```text
bit0 DESC_COUNT_ZERO
bit1 DESC_UNSUPPORTED
bit2 DESC_COUNT_EXHAUSTED
bit3 IFM_PREV_MISSING
```

`ERR_STATUS` 为 W1C：CPU 写 1 清对应错误位。`INT_CLR(0x0C)` 写 bit0 或 `CTRL[6]` 写 1 都会清中断 pending；`CTRL[6]` 读回始终为 0。

常用值：

| 模式 | 值 |
|---|---:|
| INT8 + WS + start | `0x01` |
| INT8 + OS + start | `0x11` |
| FP16 + WS + start | `0x09` |
| FP16 + OS + start | `0x19` |
| INT8 + WS + Conv2D on-the-fly + start | `0x101` |
| INT8 + OS + Conv2D on-the-fly + start | `0x111` |
| FP16 + OS + Conv2D on-the-fly + start | `0x119` |
| INT8 + OS + bias + start | `0x211` |
| INT8 + OS + ReLU + start | `0x411` |
| INT8 + OS + bias + ReLU + start | `0x611` |
| INT8 + OS + bias + ReLU6 + start | `0xA11` |
| FP16 + OS + Conv2D on-the-fly + bias + ReLU6 + start | `0xB19` |

### Direct Conv2D on-the-fly im2col

T6.2 的 direct Conv2D 路径复用 direct scalar matmul checker。CPU 仍需配置 GEMM 维度：

```text
M_DIM = Batch * OH * OW
N_DIM = Cout
K_DIM = Cin * KH * KW
A_ADDR = raw IFM base, NCHW contiguous: IFM[b][cin][ih][iw]
W_ADDR = W_col base, column-major with the same direct scalar 32-bit aligned column stride
R_ADDR = output C/OFM base, 32-bit accumulator row-major C[M,N]
CTRL[8] = 1
ARR_CFG[7] = 0
```

DMA 内部生成：

```text
m -> b, oh, ow
k -> cin, kh, kw
ih = oh * stride_h + kh * dilation_h - pad_h
iw = ow * stride_w + kw * dilation_w - pad_w
A[m,k] = in_bounds ? IFM[b,cin,ih,iw] : 0
```

当前限制：T6.2 只覆盖 direct scalar、非 tile mode；descriptor `OP=CONV2D_IM2COL` 尚未映射到该硬件路径，tile-pack/4x4 主线仍使用已有 GEMM/tile-pack 数据流。

### Direct scalar bias, activation, INT8 quant and two-layer Conv2D

T6.3-T6.6 的后处理和两层 Conv2D E2E 只覆盖 direct scalar、非 tile mode。CPU 配置方式：

```text
BIAS_ADDR = bias vector base
CTRL[9] = 1     // enable bias
CTRL[11:10] = 01 for ReLU, 10 for ReLU6
QUANT_CFG[0] = 1       // enable INT8 quant/saturate
QUANT_CFG[1] = 1       // optional signed rounding before shift
QUANT_CFG[15:8] = S    // arithmetic right shift, 0..31
QUANT_CFG[31:16] = Q   // signed 16-bit scale
```

Bias 布局：

```text
bias[j] at BIAS_ADDR + j * 4
```

执行语义：

```text
acc = dot(A[m,:], W[:,j])
if CTRL[9]: acc = acc + bias[j]
if CTRL[11:10] == 01: acc = ReLU(acc)
if CTRL[11:10] == 10: acc = ReLU6(acc)
if QUANT_CFG[0] and INT8:
    q = acc * signed_scale
    if QUANT_CFG[1] and S > 0: q = signed_round(q, S)
    q = q >>> S
    q = clamp(q, -128, 127)
    acc = sign_extend(q[7:0])
write R_ADDR + (m*N+j)*4
```

INT8 direct scalar 未开启 quant 时仍输出 signed 32-bit accumulator word；开启 quant 后输出 sign-extended signed int8 word，范围为 `[-128,127]`。FP16 direct scalar 输出是 FP32 word，ReLU6 clamp 到 `[0.0,6.0]`，`QUANT_CFG` 对 FP16 无效。当前 tile/descriptor 硬件主线尚未接入该 postprocess；RepOpt 第一层 SoC tile-window 用例先由参考 CPU 固件完成等价的 bias/ReLU/per-channel requant。

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

注意：非 tile mode 保留 Phase 1 标量兼容路径；tile mode 使用下面的 4x4 tile-pack 格式。当前已验证 4x4 INT8/FP16 GEMM、8x8/16x16 active lane 供数、阵列级 8x32 折叠路由和 PE 级 INT8 2/4-lane SIMD；更大 tile 的完整结果收集/写回、32-bit packed K lane 供数和端到端 2x/4x 吞吐仍未完成。

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

### P2 packed K-lane tile edge 约定

P2 的 multi-lane tile edge case 使用 PE 内 INT8 SIMD，把 GEMM 的 K 维打包到同一个 PE 的 `a_in/w_in` 中。它和 tile 的空间 lane 不同：

```text
空间 lane: r/c -> 不同 PE，覆盖 M/N tile
packed K lane: s -> 同一个 PE 内的一拍多个 INT8 product
```

single-lane：

```text
INT8_SIMD_LANES = 1
K groups        = K
A_TILE[m_tile][k][r] = A[m0+r,k]
W_TILE[n_tile][k][c] = W[k,n0+c]
```

multi-lane：

```text
L = INT8_SIMD_LANES  // 2 or 4
g = floor(k / L)
s = k % L

A_PACK[m_tile][g][r].lane[s] = (g*L+s < K) ? A[m0+r,g*L+s] : 0
W_PACK[n_tile][g][c].lane[s] = (g*L+s < K) ? W[g*L+s,n0+c] : 0
```

packed byte order:

```text
lane0 -> bits[ 7: 0]
lane1 -> bits[15: 8]
lane2 -> bits[23:16]
lane3 -> bits[31:24]
```

2-lane uses `DATA_W=16`:

```text
word[ 7: 0] = lane0
word[15: 8] = lane1
```

4-lane uses `DATA_W=32`:

```text
word[ 7: 0] = lane0
word[15: 8] = lane1
word[23:16] = lane2
word[31:24] = lane3
```

K tail rule:

```text
valid_k(g,s) = (g*L+s) < K
invalid tail lane must be packed as 0 and must not contribute MAC
```

CPU/testbench 生成 multi-lane case 时，不能把旧 sign-extended scalar 输入当作 packed 输入。当前 PE 会把 `16'hFFFF` 这类高位全是 lane0 符号扩展的输入识别成 single-lane 兼容编码，而不是两个 `-1` lane。multi-lane 验收必须让 lane1/lane2/lane3 对 golden 有可观测贡献。

P2.0.3 只是固定软件/生成器/descriptor 应遵守的 packed K 语义。当前已验证事实仍然是：PE 级 2/4-lane SIMD 通过；tile/descriptor 主线的 packed K-lane 供数、valid 对齐、结果写回和端到端吞吐尚未闭环。

## Descriptor 模式规划

多层网络应使用 descriptor，而不是 CPU 每层手工轮询配置。

T5.1 固定 descriptor v1 ABI：每个 descriptor 是 16 个 32-bit little-endian word，总长 64 byte。CPU 侧写入的 word 序和 NPU 侧 DMA 读取/解码的 word 序必须完全一致。

全局约束：

- `DESC_BASE` 和每个 `next_desc` 必须 64-byte aligned。
- descriptor 内所有地址字段都是 32-bit byte address。
- W/A/PSUM/OFM/Bias 数据地址至少 4-byte aligned，因为当前 AXI master 数据宽度是 32 bit。
- 保留位和保留 word 必须写 0；NPU 以后可以把非 0 保留位作为 `ERR_BAD_DESC`。
- 当前已验证的基础可执行组合是 `OP=GEMM_TILEPACK`、`DTYPE=INT8/FP16`、`DATAFLOW=OS`、`SHAPE=4x4`、`TILE_PACKED=1`。
- 当前已验证的层间串联组合是 `OP=GEMM_TILEPACK`、`DTYPE=INT8`、`DATAFLOW=OS`、`SHAPE=4x4`、`TILE_PACKED=1`、`IFM_FROM_PREV_OFM=1`。

### Descriptor v1 word layout

| Word | 字段 | 位定义 | 说明 |
|---:|---|---|---|
| 0 | `desc_ctrl` | 见下表 | 版本、op、dtype、dataflow、shape 和 flags |
| 1 | `M` | `[31:0]` | GEMM M；卷积中为 `batch*OH*OW` |
| 2 | `N` | `[31:0]` | GEMM N；卷积中为 `Cout` |
| 3 | `K` | `[31:0]` | GEMM reduce 维度；卷积中为 `Cin*KH*KW` |
| 4 | `ifm_addr` | `[31:0]` | 输入特征图基地址；tile-pack GEMM 下是 A tile-pack stream 基地址；`IFM_FROM_PREV_OFM=1` 时由上一层 `ofm_addr` 覆盖 |
| 5 | `weight_addr` | `[31:0]` | 权重基地址；tile-pack GEMM 下是 W tile-pack stream 基地址 |
| 6 | `bias_addr` | `[31:0]` | bias 基地址，`USE_BIAS=0` 时写 0 |
| 7 | `psum_addr` | `[31:0]` | 外部 PSUM surface 基地址，`USE_PSUM=0` 时写 0 |
| 8 | `ofm_addr` | `[31:0]` | 输出特征图或 GEMM C 基地址；直配模式中等价 `R_ADDR` |
| 9 | `ifm_shape` | `[15:0]=IH, [31:16]=IW` | GEMM 可写 0；on-the-fly conv 地址生成会使用 |
| 10 | `ofm_shape` | `[15:0]=OH, [31:16]=OW` | GEMM 可写 0；conv 输出空间尺寸 |
| 11 | `channel_shape` | `[15:0]=Cin, [31:16]=Cout` | GEMM 可写 0；conv 中 `Cout` 应等于 `N` |
| 12 | `kernel_stride` | `[7:0]=KH, [15:8]=KW, [23:16]=stride_h, [31:24]=stride_w` | GEMM 可写 0；conv/im2col 使用 |
| 13 | `pad_dilation` | `[7:0]=pad_h, [15:8]=pad_w, [23:16]=dilation_h, [31:24]=dilation_w` | GEMM 可写 0；默认 dilation 为 1 |
| 14 | `post_cfg` | `[3:0]=activation, [7:4]=quant_mode, [15:8]=out_shift, [31:16]=reserved` | descriptor/tile 后处理尚未实现，未使用字段写 0 |
| 15 | `next_desc` | `[31:0]` | 下一个 descriptor byte address；0 表示 descriptor 链结束 |

### `desc_ctrl` bit layout

| Bits | 字段 | 编码 |
|---:|---|---|
| `[3:0]` | `OP` | `0=NOP`, `1=GEMM_TILEPACK`, `2=GEMM_ROWMAJOR`, `3=CONV2D_IM2COL`, `4=FC` |
| `[7:4]` | `DTYPE` | `0=INT8`, `2=FP16`；低 2 bit 可直接映射到 `CTRL[3:2]` |
| `[11:8]` | `DATAFLOW` | `0=WS`, `1=OS`；低 2 bit 可直接映射到 `CTRL[5:4]` |
| `[15:12]` | `SHAPE` | `0=4x4`, `1=8x8`, `2=16x16`, `3=8x32`；低 2 bit 可直接映射到 `CFG_SHAPE` |
| `16` | `TILE_PACKED` | 1 表示 A/W 已按 4-lane tile-pack 布局，NPU 设置 `ARR_CFG[7]=1` |
| `17` | `FIRST_K` | descriptor 覆盖的 K range 是该输出 tile/layer 的第一段；可清 PSUM 或加载 bias |
| `18` | `LAST_K` | descriptor 覆盖的 K range 是最后一段；可写 OFM 并执行后处理 |
| `19` | `LAST_LAYER` | 1 表示执行完本 descriptor 后整个网络结束 |
| `20` | `IRQ_EN` | 本 descriptor 或网络结束后允许产生中断 |
| `21` | `USE_BIAS` | `bias_addr` 有效 |
| `22` | `USE_PSUM` | `psum_addr` 有效；用于外部 PSUM read/write |
| `23` | `IFM_FROM_PREV_OFM` | 本层 IFM 来自前一个已完成 descriptor 的 `ofm_addr`；DMA 按 row-major 32-bit OFM gather/repack 为 A tile stream |
| `[27:24]` | `reserved` | 必须写 0 |
| `[31:28]` | `VERSION` | v1 固定为 `1` |

当前 T4.5 已实现的是单 descriptor 内部 K-split：CPU 对整层写一个 descriptor，`K` 是完整 reduce 维度，`FIRST_K=1` 且 `LAST_K=1`。未来如果做 descriptor 级外部 K-split，则可用多个 descriptor 覆盖同一个 C/OFM surface 的不同 K range，并通过 `PSUM_ADDR` 保存跨 descriptor 部分和。

### CPU 侧结构体

CPU 侧必须按下面的结构体写 descriptor。字段类型固定为 `uint32_t`，不能使用 C bitfield。

```c
#include <stdint.h>

#define NPU_DESC_VERSION             1u
#define NPU_DESC_WORDS               16u
#define NPU_DESC_BYTES               64u

#define NPU_DESC_OP_NOP              0u
#define NPU_DESC_OP_GEMM_TILEPACK    1u
#define NPU_DESC_OP_GEMM_ROWMAJOR    2u
#define NPU_DESC_OP_CONV2D_IM2COL    3u
#define NPU_DESC_OP_FC               4u

#define NPU_DESC_DTYPE_INT8          0u
#define NPU_DESC_DTYPE_FP16          2u

#define NPU_DESC_FLOW_WS             0u
#define NPU_DESC_FLOW_OS             1u

#define NPU_DESC_SHAPE_4X4           0u
#define NPU_DESC_SHAPE_8X8           1u
#define NPU_DESC_SHAPE_16X16         2u
#define NPU_DESC_SHAPE_8X32          3u

#define NPU_DESC_FLAG_TILE_PACKED    (1u << 16)
#define NPU_DESC_FLAG_FIRST_K        (1u << 17)
#define NPU_DESC_FLAG_LAST_K         (1u << 18)
#define NPU_DESC_FLAG_LAST_LAYER     (1u << 19)
#define NPU_DESC_FLAG_IRQ_EN         (1u << 20)
#define NPU_DESC_FLAG_USE_BIAS       (1u << 21)
#define NPU_DESC_FLAG_USE_PSUM       (1u << 22)
#define NPU_DESC_FLAG_IFM_PREV_OFM   (1u << 23)

#define NPU_DESC_CTRL(op, dtype, flow, shape, flags) \
    ((NPU_DESC_VERSION << 28) | (((shape) & 0xfu) << 12) | \
     (((flow) & 0xfu) << 8) | (((dtype) & 0xfu) << 4) | \
     ((op) & 0xfu) | (flags))

typedef struct {
    uint32_t desc_ctrl;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t ifm_addr;
    uint32_t weight_addr;
    uint32_t bias_addr;
    uint32_t psum_addr;
    uint32_t ofm_addr;
    uint32_t ifm_shape;
    uint32_t ofm_shape;
    uint32_t channel_shape;
    uint32_t kernel_stride;
    uint32_t pad_dilation;
    uint32_t post_cfg;
    uint32_t next_desc;
} npu_desc_v1_t;

/* C11: _Static_assert(sizeof(npu_desc_v1_t) == NPU_DESC_BYTES, "bad NPU descriptor size"); */
```

一个当前可执行的 4x4 tile-pack GEMM descriptor：

```c
npu_desc_v1_t d = {0};
d.desc_ctrl = NPU_DESC_CTRL(
    NPU_DESC_OP_GEMM_TILEPACK,
    NPU_DESC_DTYPE_INT8,
    NPU_DESC_FLOW_OS,
    NPU_DESC_SHAPE_4X4,
    NPU_DESC_FLAG_TILE_PACKED |
    NPU_DESC_FLAG_FIRST_K |
    NPU_DESC_FLAG_LAST_K |
    NPU_DESC_FLAG_LAST_LAYER |
    NPU_DESC_FLAG_IRQ_EN);
d.m = 4;
d.n = 4;
d.k = 10;
d.ifm_addr = a_tile_pack_base;
d.weight_addr = w_tile_pack_base;
d.ofm_addr = c_base;
d.next_desc = 0;
```

### NPU RTL 解码常量

RTL 侧 T5.3 解码使用同一组 word index 和 bit range：

```verilog
localparam integer NPU_DESC_WORDS = 16;
localparam integer NPU_DESC_BYTES = 64;

localparam integer DESC_W_CTRL          = 0;
localparam integer DESC_W_M             = 1;
localparam integer DESC_W_N             = 2;
localparam integer DESC_W_K             = 3;
localparam integer DESC_W_IFM_ADDR      = 4;
localparam integer DESC_W_WEIGHT_ADDR   = 5;
localparam integer DESC_W_BIAS_ADDR     = 6;
localparam integer DESC_W_PSUM_ADDR     = 7;
localparam integer DESC_W_OFM_ADDR      = 8;
localparam integer DESC_W_IFM_SHAPE     = 9;
localparam integer DESC_W_OFM_SHAPE     = 10;
localparam integer DESC_W_CHANNEL_SHAPE = 11;
localparam integer DESC_W_KERNEL_STRIDE = 12;
localparam integer DESC_W_PAD_DILATION  = 13;
localparam integer DESC_W_POST_CFG      = 14;
localparam integer DESC_W_NEXT_DESC     = 15;

localparam integer DESC_CTRL_OP_LSB       = 0;
localparam integer DESC_CTRL_DTYPE_LSB    = 4;
localparam integer DESC_CTRL_FLOW_LSB     = 8;
localparam integer DESC_CTRL_SHAPE_LSB    = 12;
localparam integer DESC_CTRL_TILE_PACKED  = 16;
localparam integer DESC_CTRL_FIRST_K      = 17;
localparam integer DESC_CTRL_LAST_K       = 18;
localparam integer DESC_CTRL_LAST_LAYER   = 19;
localparam integer DESC_CTRL_IRQ_EN       = 20;
localparam integer DESC_CTRL_USE_BIAS     = 21;
localparam integer DESC_CTRL_USE_PSUM     = 22;
localparam integer DESC_CTRL_IFM_PREV_OFM = 23;
localparam integer DESC_CTRL_VERSION_LSB  = 28;
```

### Descriptor 到当前寄存器的映射

T5.3 的第一版 descriptor controller 会把 descriptor 映射到当前直配寄存器语义：

| 当前寄存器/控制 | descriptor 来源 |
|---|---|
| `M_DIM` | `word1 M` |
| `N_DIM` | `word2 N` |
| `K_DIM` | `word3 K` |
| `A_ADDR` | `word4 ifm_addr`，或在 `IFM_FROM_PREV_OFM=1` 且已有上一层时使用上一 descriptor 的 `word8 ofm_addr` |
| `W_ADDR` | `word5 weight_addr` |
| `R_ADDR` | `word8 ofm_addr` |
| `CTRL[3:2]` | `desc_ctrl[5:4]`，即 `DTYPE[1:0]` |
| `CTRL[5:4]` | `desc_ctrl[9:8]`，即 `DATAFLOW[1:0]` |
| `ARR_CFG[7]` | `desc_ctrl[16] TILE_PACKED` |
| `CFG_SHAPE` | `desc_ctrl[13:12]`，即 `SHAPE[1:0]` |

descriptor 链结束条件：

```text
normal_stop = desc_ctrl.LAST_LAYER or next_desc == 0
next_addr   = next_desc
```

`DESC_COUNT` 是防止链表跑飞的上限，不是正常结束条件。如果 `DESC_COUNT` 先耗尽但当前 descriptor 没有 `LAST_LAYER` 且 `next_desc != 0`，NPU 应报告 descriptor count exhausted error。

AXI-Lite descriptor 相关寄存器：

| 偏移 | 名称 | 说明 |
|---:|---|---|
| `0x40` | `DESC_BASE` | 已实现；descriptor 链表基地址 |
| `0x44` | `DESC_COUNT` | 已实现；descriptor 数量 / fetch 上限 |
| `0x74` | `ERR_STATUS` | 已实现；W1C descriptor/参数错误状态 |

T5.2 保证 CPU 可提交 descriptor list 的入口寄存器：`DESC_BASE/DESC_COUNT` 和 `CTRL[7]` 可写入并读回。T5.3 已支持 descriptor 自动 fetch/decode/execute 的第一版，当前限制为已验证的 4x4 OS tile-pack GEMM 组合。T5.5 已让 unsupported descriptor、descriptor count exhausted 等错误通过 `STATUS.error` 和 `ERR_STATUS` 对 CPU 可见。

T5.4 的 `IFM_FROM_PREV_OFM` 语义：

```text
descriptor i:
  ofm_addr = OUT_i surface, row-major 32-bit word

descriptor i+1:
  desc_ctrl.IFM_FROM_PREV_OFM = 1
  ifm_addr is ignored by RTL
  npu_ctrl uses OUT_i base as A source base
  npu_dma reads A[m0+r,k] from OUT_i[(m0+r) * K + k]
  npu_dma packs lanes r=0..3 into the A PPBuf tile stream
```

当前该路径用于 INT8 链式 GEMM/FC 验证：DMA 取上一层 32-bit OFM word 的低 8 bit 作为下一层 INT8 lane，再由 PPBuf 符号扩展。真实网络仍需要把后处理接入 tile/descriptor 主线，并补齐 quant/format conversion，确保写入 OFM 的值就是下一层期望的数据格式。

CPU descriptor 提交流程：

```c
prepare_desc_list(desc_base);
npu_write(DESC_BASE, desc_base);
npu_write(DESC_COUNT, count);
npu_write(CTRL, START | DESC_MODE);
uint32_t st;
do {
    st = npu_read(STATUS);
} while ((st & (DONE | ERROR)) == 0);
if (st & ERROR) {
    uint32_t err;
    err = npu_read(ERR_STATUS);
    npu_write(ERR_STATUS, err); // W1C
}
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

T4.1 规定 PSUM/OFM 在外部内存中使用 32-bit word、row-major surface：

```text
PSUM_ADDR(i,j) = PSUM_BASE + (i * N + j) * 4
OFM_ADDR(i,j)  = OFM_BASE  + (i * N + j) * 4
```

片上 `PSUM/OUT_BUF` 只缓存当前 4x4 tile 的 accumulator，完整大矩阵的 partial sum surface 仍在外部内存中。T4.2 已提供独立 `psum_out_buf` RMW 存储模块，T4.3 已提供 accumulator init，T4.4 已提供 controller k_tile loop，T4.5 已完成顶层 K-split GEMM golden。当前同一 C tile 的多个 k_tile 连续执行，中间 k_tile 不 flush/writeback，最后 k_tile 写回最终 tile；外部 PSUM surface 仍留给后续 descriptor 级 K-split 接入。T5.4 已接入外部 OFM surface 作为下一层 IFM 的 INT8 descriptor 链路。

## 推荐当前使用方式

当前如果要继续开发，先保持以下回归通过：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

同时运行 `tb_pe_top.v`、`tb_npu_scalar_smoke.v`、`tb_dma_read_burst.v`、`tb_dma_write_burst.v`、`tb_dma_burst.v`、`tb_dma_perf.v`、`tb_op_counter_perf.v`、`tb_psum_out_buf.v`、`tb_reconfig_pe_acc_init.v`、`tb_reconfig_pe_8x32.v`、`tb_npu_ctrl_ksplit.v`、`tb_npu_ctrl_dataflow_modes.v`、`tb_npu_ctrl_error_status.v`、`tb_npu_tile_ksplit_gemm.v`、`tb_npu_axi_lite_desc.v`、`tb_npu_desc_two_layer.v`、`tb_npu_desc_ofm_chain.v`、`tb_pingpong_buf_vec.v`、`tb_npu_tile_lane_feed.v`、`tb_npu_ctrl_tile.v`、`tb_npu_tile_writeback.v`、`tb_npu_tile_gemm.v` 和 `tb_comprehensive.v`。需要临时验证大矩阵功能时，使用 `scripts/run_matmul_case.ps1` 生成并运行 direct scalar matmul case；需要验证 T6.1 DRAM 预展开卷积时，使用 `scripts/run_conv2d_im2col_case.ps1`；需要验证 T6.2 raw IFM on-the-fly im2col 时，使用 `scripts/run_conv2d_otf_case.ps1`；需要验证 T6.3-T6.5 后处理时，加 `-Bias -Activation relu|relu6 -Quant -QuantScale <q> -QuantShift <s> [-QuantRound]`；需要验证 T6.6 两层 Conv2D E2E 时，使用 `scripts/run_conv2d_two_layer_case.ps1`。SoC smoke 使用 `scripts/run_soc_sim.ps1`，默认无 VCD，`-DumpVcd` 可选。T5.5 已补齐 IRQ/error status；T6.1/T6.2 已完成两种 Conv2D im2col golden；T6.3-T6.6 已完成 direct scalar bias、ReLU/ReLU6、INT8 quant/saturate 和两层 Conv2D E2E；T7.1-T7.5 已完成宽 lane 供数、阵列级 8x32 路由、PE 级 INT8 2/4-lane SIMD 和 TOPS/util 性能计数器报告；当前 tile/descriptor 主线是 OS，direct scalar matmul/Conv2D 回归覆盖 OS 和 WS。下一步进入 descriptor 化卷积、packed K lane 供数或更大 tile 完整写回。
