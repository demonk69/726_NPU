# RTL 模块参考

更新时间：2026-04-28

本文按“当前状态 + 目标职责 + 待改造点”描述各模块，避免把历史目标误写成已完成事实。

## 模块状态表

| 模块 | 文件 | 当前状态 | 目标职责 |
|---|---|---|---|
| `pe_top` | `rtl/pe/pe_top.v` | 单 PE 测试通过 | INT8/FP16 MAC，WS/OS 累加 |
| `reconfig_pe_array` | `rtl/array/reconfig_pe_array.v` | 16x16 阵列存在 | 4x4/8x8/16x16/8x32 形态输出 |
| `npu_top` | `rtl/top/npu_top.v` | 标量兼容路径和 4x4 tile-mode GEMM 路径已通过 | 连接 DMA、buffer、阵列、结果收集 |
| `npu_axi_lite` | `rtl/axi/npu_axi_lite.v` | 寄存器文件可用 | CPU 配置、状态、中断、descriptor base |
| `npu_ctrl` | `rtl/ctrl/npu_ctrl.v` | 标量 loop + 4x4 tile planner | descriptor/tile/layer 调度 |
| `npu_dma` | `rtl/axi/npu_dma.v` | 读 single-beat | 多通路 INCR burst DMA |
| `pingpong_buf` | `rtl/buf/pingpong_buf.v` | 支持标量拆包和 4-lane vector read | 向量化 A/W tile buffer |
| `npu_power` | `rtl/power/npu_power.v` | 输出未接 PE | clock enable/DFS/门控接入 |
| `soc_top` | `rtl/soc/soc_top.v` | NPU 参数已修复，SoC 编译仍阻塞 | PicoRV32 + NPU + DRAM 验证平台 |

## `pe_top`

当前功能：

- `mode=0`：INT8 signed MAC，`w_in[7:0] * a_in[7:0]`，32-bit accumulate。
- `mode=1`：FP16 multiply，FP32 accumulate。
- `stat_mode=0`：WS，使用内部 weight register。
- `stat_mode=1`：OS，weight 和 activation 流入，内部保持输出 psum。
- `flush` 输出累加结果并清 accumulator。

后续改造：

1. 明确 pipeline latency，在阵列 valid 对齐中使用统一参数。
2. INT8 性能优化时增加 2-lane 或 4-lane SIMD。
3. 增加 accumulator init 接口，用于 K-split 读取 `PSUM_BUF` 后继续累加。

## `reconfig_pe_array`

当前功能：

- 实例化 16x16 个 `pe_top`。
- `cfg_shape` 支持 4x4、8x8、16x16、8x32 输出映射。

当前限制：

- 顶层在 tile mode 只喂左上 4x4 lane；8x8/16x16/8x32 还没有对应的宽向量供数。
- OS 模式的 4x4 路径已通过 row-skew feeder 验证；更大形态仍需要重新做 valid 对齐和测试。
- 8x32 需要更严格的折叠路由、valid 对齐和测试。

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
- T2.2 已把 W/A PPBuf 的 `rd_vec` 接到阵列左上 4 行/列。
- T2.3 已让 `ctrl_vec_consume` 驱动 `rd_vec_en`，并给 A lane1/2/3 增加 1/2/3 拍延迟形成 OS row-skew；这些延迟会在启动阶段自然插入 bubble。

必须修复：

1. `npu_power` 输出或 clock enable 真正作用于 PE。
2. 增加 `PSUM/OUT_BUF`。
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
| `0x04` | `STATUS` | busy/done |
| `0x08` | `INT_EN` | 中断使能 |
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

建议新增：

| 偏移 | 名称 | 说明 |
|---:|---|---|
| `0x40` | `DESC_BASE` | descriptor 链表基地址 |
| `0x44` | `DESC_COUNT` | descriptor 数量 |
| `0x48` | `PERF_CYCLES` | 运行周期 |
| `0x4C` | `PERF_AXI_BEATS` | AXI beat 计数 |
| `0x50` | `ERR_STATUS` | DMA/协议/参数错误 |

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
      expose tile_m_base/tile_n_base     // tile 左上角的全局 M/N 坐标
      expose row/col valid mask          // 边界 tile 中哪些 r/c lane 有效
      issue vec_consume for K cycles     // 每个逻辑 k 周期消费一组 A/W 4-lane vector
```

目标行为：

```text
for desc in descriptor_list:
  for m_tile:
    for n_tile:
      for k_tile:
        load W tile
        load A tile
        optionally load PSUM
        compute tile
        write PSUM or OUT
      post-process
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
- 读通道 `ARLEN=0`。

目标：

1. 支持 8/16/32 beat INCR burst。
2. 支持 4KB 边界切分。
3. 支持 descriptor、W、A、PSUM、OUT 多种传输类型。
4. 对每个通道提供 `busy/done/error/bytes`。
5. 统计 AXI 有效 beat，用于带宽利用率报告。

## `pingpong_buf`

当前功能：

- 每个 32-bit word 拆 INT8 或 FP16 子字。
- DMA 写一侧，PE 读另一侧。
- `rd_data` 保持单 lane 标量读。
- `rd_vec` 提供 4-lane vector preview，`rd_vec_en` 一拍消费 4 个 lane。
- INT8 vector：一个 32-bit word 拆成 4 个 sign-extended `OUT_WIDTH` lane。
- FP16 vector：两个 32-bit word 拆成 4 个 half-word lane。

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

## `npu_power`

当前功能：

- 有 DFS 行为时钟和 row/col gating 输出。
- 输出在 `npu_top` 中悬空。

目标：

- FPGA 上优先使用 clock enable 或 BUFGCE。
- ASIC 方案用 ICG cell。
- idle 时关闭 PE array。
- 小形态运行时关闭未使用 row/col。
- 保留性能计数，便于估算低功耗收益。

## `soc_top`

当前问题：

- `npu_top` 的旧 `.ROWS/.COLS` 实例化问题已修复，推荐继续使用 `.PHY_ROWS/.PHY_COLS`。
- `scripts/run_soc_sim.ps1` 当前仍因 `dram_model.v` 中 `axi_arlen` 未绑定，以及 PicoRV32 实例 PCPI 端口名与参考核不匹配而编译失败。

目标：

- PicoRV32 通过 AXI-Lite 配置 NPU。
- NPU DMA 通过 AXI4 master 访问共享 DRAM。
- CPU 等待 IRQ 或轮询 `STATUS.done`。
- testbench 检查 DRAM 输出和 PASS 标记。
