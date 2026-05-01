# RTL 模块参考

更新时间：2026-05-01

本文按“当前状态 + 目标职责 + 待改造点”描述各模块，避免把历史目标误写成已完成事实。

## 阅读方式

本文件从 RTL 模块角度解释“谁连到谁、谁负责什么”。概念定义以 [architecture.md](architecture.md) 的术语表为准；这里保留一份更短的模块阅读地图。

常见缩写：

| 名词 | 模块语境里的含义 |
|---|---|
| descriptor | `npu_ctrl` 通过 `npu_dma` 从 DRAM fetch 的 16-word 任务描述。T5.4 后可解码 `M/N/K/A/W/R`、dtype、dataflow、shape、tile-pack 和 `IFM_FROM_PREV_OFM` 层间来源配置。 |
| PPBuf | `pingpong_buf`，A/W 的片上 ping-pong 缓冲。DMA 写入，PE feeder 读取。 |
| PSUM | partial sum，K 尚未完全累计或尚未后处理的 C 中间值。PE accumulator、`psum_out_buf`、外部 PSUM surface 都可能承载 PSUM。 |
| OUT/OFM | 最后一个 k_tile 后的输出。当前 direct scalar 路径可在 32-bit accumulator 后接 bias 和 ReLU/ReLU6；tile/descriptor 主路径仍主要写回 32-bit accumulator word。 |
| tile | 当前主路径为 4x4 C tile。`m_tile/n_tile` 决定 C tile 左上角，`k_tile` 决定当前 K slice。 |
| lane | 送入阵列的一条并行通道。4-lane A/W vector 由 PPBuf 输出，进入 4x4 PE array。 |
| row-skew | `npu_top` feeder 对 A lane 做的错拍，使 OS 阵列中 row `r` 在 `t=k+r` 时看到正确的 `A[m0+r,k]`。 |

## 模块串联流程

顶层数据和控制可以拆成五条线看：

```text
control path:
CPU -> npu_axi_lite -> npu_ctrl

descriptor path:
npu_ctrl -> npu_dma(desc_start/desc_addr) -> AXI read DRAM
         -> npu_dma(desc_words/desc_done) -> npu_ctrl decode

read data path:
npu_ctrl(load_w/load_a) -> npu_dma -> AXI read DRAM
                         -> W/A pingpong_buf

compute path:
npu_ctrl(vec_consume/compute/flush/mask)
  -> npu_top feeder(row-skew)
  -> reconfig_pe_array
  -> pe_top accumulator

writeback path:
reconfig_pe_array -> npu_top serializer/result FIFO
  -> npu_dma -> AXI write DRAM OFM
```

直接寄存器模式从 CPU 写 `M_DIM/N_DIM/K_DIM/W_ADDR/A_ADDR/R_ADDR/ARR_CFG/CFG_SHAPE/CTRL.start` 开始，`npu_ctrl` 直接使用这些 shadow config 调度 tile。T6.1 的 DRAM 预展开 Conv2D im2col 仿真也走这条 direct scalar 路径：软件/testbench 先生成 `A_im2col[M,K]` 和 `W_col[K,N]`，再把 Conv2D golden 作为 `expected.hex` 校验。T6.2 在同一路径中增加 `CTRL[8]`：A 侧由 `npu_dma` 按 Conv2D 参数从 raw NCHW IFM on-the-fly gather，不再读取完整 `A_im2col` 中间矩阵。T6.3-T6.5 又在 direct scalar 路径中增加 `CTRL[9]` bias、`CTRL[11:10]` ReLU/ReLU6 和 `QUANT_CFG(0x9C)` INT8 quant/saturate。T6.6 验证两层 Conv2D 可在 DRAM 中把 layer0 量化 OFM 直接作为 layer1 A 输入。

Descriptor 模式从 CPU 写 `DESC_BASE/DESC_COUNT` 并置位 `CTRL[7] desc_mode` 开始。`npu_ctrl` 先让 `npu_dma` 读 64-byte descriptor，再把 descriptor 字段映射到同一套 shadow config，之后复用 direct mode 的 tile 调度和写回路径。一个 descriptor 完成后，`last_layer` 或 `next_desc==0` 结束；否则继续 fetch `next_desc`。`DESC_COUNT` 是防止链表跑飞的 fetch 上限。

当前 descriptor 流覆盖 `GEMM_TILEPACK + INT8/FP16 + OS + 4x4 + TILE_PACKED` 的独立 descriptor 顺序执行；T5.4 额外验证了 `INT8 + IFM_FROM_PREV_OFM` 的两层 GEMM 串联。T5.5 已接入 unsupported descriptor、descriptor count exhausted 等错误状态。外部 PSUM surface 的 descriptor 化读写、FP16 OFM 格式转换仍未完整接入。

## 模块状态表

| 模块 | 文件 | 当前状态 | 目标职责 |
|---|---|---|---|
| `pe_top` | `rtl/pe/pe_top.v` | 单 PE 测试通过 | INT8/FP16 MAC，WS/OS 累加 |
| `reconfig_pe_array` | `rtl/array/reconfig_pe_array.v` | 16x16 阵列存在 | 4x4/8x8/16x16/8x32 形态输出 |
| `npu_top` | `rtl/top/npu_top.v` | 标量兼容路径、4x4 tile-mode GEMM、descriptor fetch、T5.4 OFM->IFM、T6.2 im2col 和 T6.4 direct scalar postprocess 已通过 | 连接 DMA、buffer、阵列、结果收集 |
| `npu_axi_lite` | `rtl/axi/npu_axi_lite.v` | 寄存器文件、descriptor 提交寄存器、direct Conv2D shape 和 bias 地址寄存器可用 | CPU 配置、状态、中断、descriptor base/count、Conv2D 参数 |
| `npu_ctrl` | `rtl/ctrl/npu_ctrl.v` | 标量 loop、32-bit 对齐 direct stride、4x4 tile planner、descriptor fetch/decode/next-layer、上一层 OFM 地址跟踪、direct Conv2D im2col 参数下发、direct bias/activation 控制锁存 | descriptor/tile/layer 调度 |
| `npu_dma` | `rtl/axi/npu_dma.v` | 读/写侧 INCR burst、4KB 边界切分、64-byte descriptor fetch、OFM row-major gather/repack、raw IFM on-the-fly im2col gather、32-bit bias fetch | 多通路 INCR burst DMA 和带宽统计 |
| `pingpong_buf` | `rtl/buf/pingpong_buf.v` | 支持标量拆包和 4-lane vector read | 向量化 A/W tile buffer |
| `psum_out_buf` | `rtl/buf/psum_out_buf.v` | 2-bank 4x4 tile RMW 测试通过 | K-split PSUM/OUT tile storage |
| `npu_power` | `rtl/power/npu_power.v` | 输出未接 PE | clock enable/DFS/门控接入 |
| `soc_top` | `rtl/soc/soc_top.v` | SoC smoke 已通过 | PicoRV32 + NPU + DRAM 验证平台 |

## `pe_top`

当前功能：

- `mode=0`：INT8 signed MAC，`w_in[7:0] * a_in[7:0]`，32-bit accumulate。
- `mode=1`：FP16 multiply，FP32 accumulate。
- `stat_mode=0`：WS，使用内部 weight register。
- `stat_mode=1`：OS，weight 和 activation 流入，内部保持输出 psum。
- `flush` 输出累加结果并清 accumulator。
- `acc_init_en/acc_init` 已实现 accumulator 初始化；INT8 路径按 int32 bit pattern，FP16 路径按 FP32 bit pattern 继续累加。

连接关系：

```text
upstream:  reconfig_pe_array routes a_in/w_in/load_w/flush/acc_init
internal:  multiplier + accumulator
downstream: reconfig_pe_array collects psum_out and valid timing
```

在 OS 路径里，`pe_top` 的 accumulator 保存当前 C 元素的 PSUM；只有 `flush` 时才把它交给上层 serializer。若后续接入外部 PSUM readback，`acc_init_en/acc_init` 就是把旧 PSUM 恢复进 PE 的入口。

后续改造：

1. 明确 pipeline latency，在阵列 valid 对齐中使用统一参数。
2. INT8 性能优化时增加 2-lane 或 4-lane SIMD。

## `reconfig_pe_array`

当前功能：

- 实例化 16x16 个 `pe_top`。
- `cfg_shape` 支持 4x4、8x8、16x16、8x32 输出映射。
- `acc_init_en/acc_init/acc_init_mask` 支持 per-PE accumulator 初始化，T4.3 已用 4x4 OS continued-MAC 测试验证。

当前限制：

- 顶层在 tile mode 只喂左上 4x4 lane；8x8/16x16/8x32 还没有对应的宽向量供数。
- OS 模式的 4x4 路径已通过 row-skew feeder 验证；更大形态仍需要重新做 valid 对齐和测试。
- 8x32 需要更严格的折叠路由、valid 对齐和测试。

连接关系：

```text
upstream:  npu_top feeder provides a_vec/w_vec, valid masks, mode, flush, acc_init
internal:  instantiates 16x16 pe_top and masks unused rows/cols
downstream: npu_top serializer consumes result_vec[16] and valid_vec[16]
```

当前 4x4 tile 只使用左上角 PE(0..3,0..3)。`cfg_shape` 虽然能选择更大形态，但如果没有对应的 A/W 宽向量供数和写回映射，不能把它理解成 8x8/16x16 已完成。

目标接口应支持：

```text
a_vec[4]        // PPBuf 输出 A_TILE[k][r]；进入 PE row 前由 feeder 错拍
w_vec[4]        // PPBuf 输出 W_TILE[k][c]
row_valid[4]
col_valid[4]
result_vec[16]  // result[r*4+c] -> C[m0+r,n0+c]
valid_vec[16]
```

T2.1 决定：OS 模式作为第一条真正 4x4 GEMM tile 路径，PE row 对应 M lane，PE col 对应 N lane。由于当前权重从 top 向下传播，A lane 由 feeder 做 row skew：物理周期 `t` 向 row `r` 输入 `A[m0+r,t-r]`，越界时输入 0。因此前几个周期是 ramp-up bubble，不是 4 个 row 从第 0 拍全部有效。WS 模式保留为权重驻留的 1x4 row-vector 子流程，完整 4x4 tile 由 4 个 M row pass 组成；`K>4` 的 WS 累加归入 PSUM/K-split 阶段。

## `npu_top`

当前职责：

- 实例化 AXI-Lite、controller、PPBuf、DMA、PE array、power。
- 非 tile mode 把 PPBuf 标量输出接到 `u_scalar_pe`，保留 Phase 1 兼容回归路径。
- tile mode 把 PPBuf 4-lane vector 接到阵列左上 4x4，并通过 serializer 收集 16 个阵列输出。
- 结果 FIFO 在非 tile mode 写入 `u_scalar_pe` 输出，在 tile mode 写入 `pe_array_result` 的 row-major 序列。
- `cfg_shape` 已由 `npu_ctrl` 在 start 时锁存，运行中修改 `CFG_SHAPE` 不影响当前任务。
- `axi_monitor` 已接入 AXI-Lite 和 DMA AXI4 master 通道，性能计数通过 `npu_axi_lite` 暴露。
- T5.3 已将 `DESC_BASE/DESC_COUNT` 从 AXI-Lite 接到 controller，并通过 DMA read 侧 fetch 64-byte descriptor v1。
- T5.4 已将 controller 的 `dma_a_ofm_*` 元数据接到 DMA，使上一层 OFM surface 可作为下一层 A 源。
- T6.2 已将 AXI-Lite direct Conv2D 寄存器和 controller 的 `dma_a_im2col_*` 元数据接到 DMA，使 raw IFM 可作为 direct scalar Conv2D A 源。
- T6.3-T6.5 已将 `BIAS_ADDR`、`CTRL[9]`、`CTRL[11:10]` 和 `QUANT_CFG(0x9C)` 接到 direct scalar 路径：bias fetch 后作为 scalar PE accumulator init，ReLU/ReLU6 和 INT8 quant/saturate 在 result FIFO 前执行。
- T6.6 已验证 direct scalar 两层 Conv2D E2E：layer0 量化输出按 sign-extended int8 word 写回，layer1 可直接以该 `R_ADDR` 作为 `A_ADDR`。
- T2.2 已把 W/A PPBuf 的 `rd_vec` 接到阵列左上 4 行/列。
- T2.3 已让 `ctrl_vec_consume` 驱动 `rd_vec_en`，并给 A lane1/2/3 增加 1/2/3 拍延迟形成 OS row-skew；这些延迟会在启动阶段自然插入 bubble。
- T4.1 已明确 `PSUM/OUT_BUF` 的 32-bit word、4x4 tile-local index 和外部 row-major 地址模式。
- T4.2 已实现独立 `psum_out_buf` 模块。
- T4.3 已实现 PE/array accumulator init 接口；顶层当前 tie-off，后续外部 PSUM readback 可接入该接口。
- T4.4 已实现 controller k_tile loop；同一 C tile 的中间 k_tile 不 flush/writeback，最后 k_tile 才 row-wise 写回。
- T4.5 已通过 `tb_npu_tile_ksplit_gemm.v` 验证顶层 4x4x10 INT8 OS K-split GEMM，结果等于未切分 golden。
- T6.5 已在 direct scalar INT8 输出进入 result FIFO 前支持 quant/saturate；量化输出为 sign-extended signed int8 word，范围 `[-128,127]`。

连接关系：

```text
upstream control: npu_axi_lite registers and npu_ctrl schedule signals
upstream data:    npu_dma writes W/A pingpong_buf
compute bridge:   PPBuf rd_vec -> row-skew feeder -> reconfig_pe_array
write bridge:     PE results -> serializer/result FIFO -> npu_dma writeback
sideband:         npu_dma desc_words -> npu_ctrl descriptor decode
```

`npu_top` 不应该承担“决定下一层是什么”的职责。它负责把模块接成可运行的数据面；层级、tile、descriptor 链表和结束条件由 `npu_ctrl` 决定。

必须修复：

1. `npu_power` 输出或 clock enable 真正作用于 PE。
2. 将外部 PSUM surface 接入 descriptor 级 readback/writeback 路径。
3. 为 8x8/16x16/8x32 增加对应供数、valid 对齐和写回测试。

T2.1 后的顶层数据组织目标：

```text
A_TILE[m_tile][k][r] -> a_vec[r]
W_TILE[n_tile][k][c] -> w_vec[c]
PE(r,c)              -> C[m0+r,n0+c]
```

C 写回不能默认 16 word 全局连续。一般情况下每个有效 row 发一个短 burst：

```text
base = R_ADDR + ((m0+r) * N + n0) * 4
beats = active_cols
```

T2.4 已完成：

```text
pe_array_result/valid -> 16-output serializer -> result FIFO/DMA writeback
4x4 result order = r*4+c
row-wise writeback base = R_ADDR + ((m0+r) * N + n0) * 4
row-wise writeback beats = active_cols
```

T2.5/T2.6 已完成：

```text
tb/tile4/gen_tile4_data.py -> INT8/FP16 4x4x4 Python golden
tb/tb_npu_tile_gemm.v      -> ARR_CFG[7] tile-mode GEMM checker
```

## `npu_axi_lite`

当前寄存器：

| 偏移 | 名称 | 说明 |
|---:|---|---|
| `0x00` | `CTRL` | start/abort/mode/stat_mode/irq clear |
| `0x04` | `STATUS` | busy/done/error |
| `0x08` | `INT_EN` | bit0=done IRQ enable, bit1=error IRQ enable |
| `0x0C` | `INT_CLR/PENDING` | 清/读 pending |
| `0x10` | `M_DIM` | GEMM 行数；卷积中为 `batch*OH*OW` |
| `0x14` | `N_DIM` | GEMM 列数；卷积中为 `Cout` |
| `0x18` | `K_DIM` | 归约维度；卷积中为 `Cin*KH*KW` |
| `0x20` | `W_ADDR` | weight/B 或 W tile-pack 基地址 |
| `0x24` | `A_ADDR` | activation/A 或 A tile-pack 基地址 |
| `0x28` | `R_ADDR` | result/C 基地址 |
| `0x30` | `ARR_CFG` | bit7=4x4 tile mode enable |
| `0x34` | `CLK_DIV` | DFS 配置 |
| `0x38` | `CG_EN` | clock gating 配置 |
| `0x3C` | `CFG_SHAPE` | 阵列形态 |
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
| `0x74` | `ERR_STATUS` | W1C descriptor/参数错误状态 |
| `0x80` | `CONV_IFM_SHAPE` | `[15:0]=IH, [31:16]=IW` |
| `0x84` | `CONV_CHANNELS` | `[15:0]=Cin, [31:16]=Batch` |
| `0x88` | `CONV_KERNEL` | `[15:0]=KH, [31:16]=KW` |
| `0x8C` | `CONV_OUT_SHAPE` | `[15:0]=OH, [31:16]=OW` |
| `0x90` | `CONV_STRIDE_PAD` | stride/pad, 8 bit each |
| `0x94` | `CONV_DILATION` | dilation_h/dilation_w, 8 bit each |
| `0x98` | `BIAS_ADDR` | direct scalar bias vector base |
| `0x9C` | `QUANT_CFG` | bit0 enable, bit1 round, `[15:8]` right shift, `[31:16]` signed scale |

T5.1 已固定 descriptor v1 ABI：`DESC_BASE` 指向 64-byte aligned 的 16-word descriptor，`DESC_COUNT` 是 controller 最多允许 fetch 的 descriptor 数量。T5.2 已实现这两个寄存器的 reset/write/readback；T5.3 已让 controller 使用它们发起 descriptor fetch/decode，并按 `next_desc` 顺序执行多个 descriptor；T5.5 已把 done/error 通过 `STATUS`、`INT_EN/INT_CLR` 和 `ERR_STATUS` 暴露给 CPU。

`ERR_STATUS` 当前 bit 定义：bit0=`DESC_COUNT_ZERO`，bit1=`DESC_UNSUPPORTED`，bit2=`DESC_COUNT_EXHAUSTED`，bit3=`IFM_PREV_MISSING`。CPU 对 `0x74` 写 1 清对应错误位。

连接关系：

```text
upstream:   CPU/PicoRV32 writes AXI4-Lite registers
internal:   register file, status/irq/perf counter mux
downstream: npu_ctrl consumes config/start/abort/desc_base/desc_count
feedback:   npu_ctrl/npu_dma/npu_top return busy/done/irq/perf/status
```

`npu_axi_lite` 不解析 descriptor 内容，也不直接产生 DMA burst。它只把 CPU 可见寄存器稳定地交给控制器，并把运行状态映射回 CPU 可读空间。

## `npu_ctrl`

当前行为：

```text
if ARR_CFG[7] == 0:
  for i in 0..M-1:
    for j in 0..N-1:
      compute one C[i,j]

if ARR_CFG[7] == 1:
  for m_tile in 0..ceil(M/4)-1:
    for n_tile in 0..ceil(N/4)-1:
      for k_tile in 0..ceil(K/K_TILE_ELEMS)-1:
        expose tile_m_base/tile_n_base   // tile 左上角的全局 M/N 坐标
        expose tile_k_base/tile_k_len    // 当前 K slice
        expose row/col valid mask        // 边界 tile 中哪些 r/c lane 有效
        issue vec_consume for tile_k_len cycles
        flush/writeback only on final k_tile

if CTRL[7] desc_mode == 1:
  fetch 16-word descriptor v1 from DESC_BASE/next_desc
  decode M/N/K/A/W/R and desc_ctrl dtype/dataflow/shape/tile_packed/ifm_from_prev_ofm
  execute the mapped tile GEMM layer
  fetch next_desc until LAST_LAYER or next_desc == 0

if CTRL[8] conv_im2col == 1 and ARR_CFG[7] == 0:
  direct scalar A load uses raw IFM on-the-fly im2col gather
  W still uses direct scalar W_col column-major layout
```

连接关系：

```text
upstream config: npu_axi_lite direct registers or decoded descriptor fields
upstream status: npu_dma load/write/desc done, npu_top compute/writeback done
downstream DMA:  load_w_start, load_a_start, writeback_start, desc_start
downstream DMA:  for IFM_FROM_PREV_OFM, also pass row-major OFM gather metadata
downstream DMA:  for direct Conv2D, also pass raw IFM im2col gather metadata
downstream top:  tile bases, k slice, row/col masks, vec_consume, compute/flush
```

`npu_ctrl` 是唯一应该理解 loop nest 的模块：layer loop、`m_tile/n_tile` loop、`k_tile` loop 都在这里收敛。DMA 只执行一次搬运请求，PE 只执行一次被喂给它的 MAC 序列。

目标行为：

```text
for desc in descriptor_list:
  fetch 16-word descriptor v1
  decode desc_ctrl/version/op/dtype/dataflow/shape/flags
  map descriptor fields to current M/N/K/A/W/R/CTRL/ARR_CFG/CFG_SHAPE semantics
  for m_tile:
    for n_tile:
      for k_tile:
        load W tile
        load A tile
        optionally load PSUM according to T4.1 row-major address
        compute tile
        write PSUM or OUT
      post-process
```

T5.3 的第一版可执行 descriptor 组合限制为 `OP=GEMM_TILEPACK`、`DTYPE=INT8/FP16`、`DATAFLOW=OS`、`SHAPE=4x4`、`TILE_PACKED=1`；T5.4 在此基础上增加 `desc_ctrl[23] IFM_FROM_PREV_OFM` 的 INT8 两层串联。T5.5 起其它 op/shape 会置位 `ERR_STATUS.DESC_UNSUPPORTED`。

T5.3 descriptor 的具体执行顺序：

```text
IDLE
  -> FETCH_DESC       // desc_start + desc_addr to npu_dma
  -> DECODE_DESC      // latch desc_words into shadow config
  -> DESC_LAUNCH      // reuse existing start path
  -> tile load/compute/writeback loops
  -> FETCH_DESC       // if next_desc != 0 and LAST_LAYER == 0
  -> DONE
```

T5.4 后，`IFM_FROM_PREV_OFM=0` 时每层仍是独立 GEMM 任务，`ifm_addr/weight_addr/ofm_addr` 都由 descriptor 明确给出；`IFM_FROM_PREV_OFM=1` 时，`npu_ctrl` 使用上一 descriptor 的 `ofm_addr` 覆盖本层 A 源地址，并给 DMA 传入 `m_base/k_base/k_len/active_rows/stride`，由 DMA 从 row-major 32-bit OFM surface gather 出 A tile lane。该路径当前验证 INT8；FP16 仍需要明确的格式转换或后处理。T6.2 的 Conv2D im2col 元数据只在 direct scalar、非 tile mode 下使用；descriptor `OP=CONV2D_IM2COL` 尚未映射到该路径。

T4.1 已固定控制器后续需要生成的 PSUM/OFM 外部地址：

```text
PSUM_ADDR(i,j) = PSUM_BASE + (i * N + j) * 4
OFM_ADDR(i,j)  = OFM_BASE  + (i * N + j) * 4
```

T4.4 已固定当前 K-split load 地址：

```text
A_TILE_ADDR(m_tile,k_tile,k) = A_ADDR + (m_tile * K + (k0+k)) * A_VEC_BYTES
W_TILE_ADDR(n_tile,k_tile,k) = W_ADDR + (n_tile * K + (k0+k)) * W_VEC_BYTES
```

关键新增计数器：

```text
layer_id
m_tile_idx, n_tile_idx, k_tile_idx
m_inner, n_inner, k_inner
active_rows, active_cols
```

T2.3 已实现 Phase 2 的最小 tile 计数：

```text
m_tile_idx = 0 .. ceil(M/4)-1       // M 方向 tile 编号
n_tile_idx = 0 .. ceil(N/4)-1       // N 方向 tile 编号
k_cycle    = 0 .. K+active_rows-2   // OS row-skew schedule
```

并生成：

```text
row_valid[r] = (m_tile_idx*4 + r) < M  // tile 内 row r 是否对应真实 C 行
col_valid[c] = (n_tile_idx*4 + c) < N  // tile 内 col c 是否对应真实 C 列
```

## `npu_dma`

当前状态：

- Load FSM 读 W/A。
- WB FSM 写结果。
- T3.1 已完成读通道 INCR burst：`ARLEN=burst_beats-1`，一次只保留一个 outstanding read burst。
- 读 burst 按 `BURST_MAX` 和 4KB 边界切分；W/A 仍按当前控制器合约顺序读取。
- T3.2 已完成写通道 INCR burst：`AWLEN=burst_beats-1`，`WLAST` 按当前 burst 末拍产生。
- 写 burst 按 `BURST_MAX` 和 4KB 边界切分；每段等待 B response 后继续下一段，整笔完成后脉冲 `r_done`。
- T6.2 已新增 A_IM2COL 读目标：按 Conv2D `m/k` 计算 raw NCHW IFM 地址，越界/padding 生成 0，并把 INT8/FP16 元素打包写入 A PPBuf。

连接关系：

```text
upstream control: npu_ctrl issues desc/load_w/load_a/writeback requests
upstream write:   npu_top result FIFO provides writeback data
AXI side:         AXI4 master read/write channels access DRAM/SRAM
downstream read:  descriptor words return to npu_ctrl; W/A data write PPBuf
downstream done:  done pulses return to npu_ctrl for FSM advance
```

T5.3 新增的 descriptor fetch 复用读侧 AXI 通道，但输出目的不是 PPBuf，而是 `desc_words[511:0]`。因此 DMA 读侧需要区分“descriptor read”和“tile data read”两类目的地，避免把 descriptor 当成 A/W 数据写进 buffer。T5.4 又增加 A_OFM 读目标：它从上一层 row-major 32-bit OFM 发起每 lane 单 beat read，低 8 bit 被打包成 INT8 A lane 后写入 A PPBuf。T6.2 增加 A_IM2COL 读目标：它从 raw IFM 发起单 beat gather read，按字节/半字选择元素并打包为 direct scalar A 行。

目标：

1. 支持 descriptor、W、A、A_FROM_OFM、A_IM2COL、PSUM、OUT 多种传输类型。
2. 对每个通道提供 `busy/done/error/bytes`。
3. 统计 AXI 有效 beat，用于带宽利用率报告。

## `pingpong_buf`

当前功能：

- 每个 32-bit word 拆 INT8 或 FP16 子字。
- DMA 写一侧，PE 读另一侧。
- `rd_data` 保持单 lane 标量读。
- `rd_vec` 提供 4-lane vector preview，`rd_vec_en` 一拍消费 4 个 lane。
- INT8 vector：一个 32-bit word 拆成 4 个 sign-extended `OUT_WIDTH` lane。
- FP16 vector：两个 32-bit word 拆成 4 个 half-word lane。

连接关系：

```text
upstream write: npu_dma writes packed A/W words
read control:   npu_ctrl vec_consume advances rd_vec_en
downstream:     npu_top feeder consumes rd_vec and applies row-skew
```

PPBuf 只负责“按当前格式吐出 lane”。它不判断 `m_tile/n_tile/k_tile` 的全局含义，也不处理边界 mask；这些由 `npu_ctrl` 和 `npu_top` feeder 提供。

目标：

- T2.3 已由 controller 使用 vector consume。
- 后续从 4 lane 扩展为 `LANES=8/16`。
- 支持 row/col skew 所需的读地址模式。
- 支持 tile 双缓冲：DMA 写下一个 tile，PE 消费当前 tile。

T2.1 规定的 4-lane packing：

```text
INT8 beat[ 7: 0] = lane0
INT8 beat[15: 8] = lane1
INT8 beat[23:16] = lane2
INT8 beat[31:24] = lane3

FP16 beat0 = lane0,lane1
FP16 beat1 = lane2,lane3
```

T2.2 已增加 4-lane read path，输出 `rd_vec[4*DATA_W-1:0]`。T2.3 已让 `npu_top` feeder 基于 `rd_vec` 施加 OS row skew，并让 `npu_ctrl` 用 `rd_vec_en` 推进 buffer。

## `psum_out_buf`

当前功能：

- 每个 bank 保存一个 tile-local accumulator tile，默认 `4x4 * 32-bit = 16 words`。
- port A 面向 DMA/load/drain，port B 面向 compute/serializer。
- 支持同步 read/write；K-split 累加由外部执行 `read old psum -> modify -> write new psum`。
- `valid_mask` 用于边界 tile：invalid lane 读 0、`rvalid=0`、写入被忽略。
- `tile_clear_en` 可清空指定 bank。
- 同地址双写时 `write_conflict` 置位，port B 优先。

连接关系：

```text
target upstream:  npu_dma loads external PSUM surface into tile-local bank
compute bridge:   npu_top/reconfig_pe_array use acc_init to resume PSUM
target downstream:npu_dma writes intermediate PSUM or final OUT rows
```

当前该模块已独立验证，但还没有完整挂到 descriptor 多层主路径。它的定位是“当前 tile 的 16 word 暂存/RMW 单元”，不是整层输出缓存。

T4.2 验证：

```text
tb/tb_psum_out_buf.v -> PASS
```

后续接入：

1. T4.3 已完成 PE/array accumulator init 接口。
2. T4.4/T4.5 已由 `npu_ctrl` 生成 k_tile loop 并完成顶层 K-split GEMM golden；外部 PSUM/OUT surface 的 read/write 留给后续 descriptor/多层调度。

## `npu_power`

当前功能：

- 有 DFS 行为时钟和 row/col gating 输出。
- 输出在 `npu_top` 中悬空。

连接关系：

```text
target upstream:  npu_ctrl/npu_top provide busy, cfg_shape, active row/col mask
target downstream:PE array clock enable or FPGA-safe clock gate
feedback:         perf counters can compare gated/ungated execution
```

当前它还只是策略输出模块，不影响实际 PE toggle。后续接入时优先用 clock enable，避免在 FPGA RTL 中直接门控时钟。

目标：

- FPGA 上优先使用 clock enable 或 BUFGCE。
- ASIC 方案用 ICG cell。
- idle 时关闭 PE array。
- 小形态运行时关闭未使用 row/col。
- 保留性能计数，便于估算低功耗收益。

## `soc_top`

当前状态：

- `npu_top` 的旧 `.ROWS/.COLS` 实例化问题已修复，推荐继续使用 `.PHY_ROWS/.PHY_COLS`。
- `dram_model` 已补齐 `axi_arlen` 读突发并修正写突发 WLAST 地址推进；`soc_top` 的 PicoRV32 PCPI 端口已对齐参考核。
- `axi_lite_bridge` 已分离 AW/W 握手，`soc_mem`/`dram_model` CPU 读口与 PicoRV32 ready/rdata 时序对齐。
- `scripts/run_soc_sim.ps1` 当前通过：PicoRV32 配置 NPU 完成 2x2 INT8 GEMM，结果 `19,22,43,50`。

连接关系：

```text
CPU side: PicoRV32 -> AXI4-Lite -> npu_axi_lite
data side:NPU AXI4 master -> shared DRAM model
software: firmware prepares data/descriptor, starts NPU, polls done or waits IRQ
```

`soc_top` 是系统级验证壳，不应该复制 `npu_ctrl` 或 `npu_dma` 的逻辑。它的主要价值是证明 CPU、寄存器、DMA 内存访问和中断/轮询能在同一个地址空间闭环。

目标：

- PicoRV32 通过 AXI-Lite 配置 NPU。
- NPU DMA 通过 AXI4 master 访问共享 DRAM。
- CPU 等待 IRQ 或轮询 `STATUS.done`。
- testbench 检查 DRAM 输出和 PASS 标记。
