# 架构修复路线

更新时间：2026-04-28

本文把当前 RTL 原型演进到目标 NPU 的修复路线按依赖关系排列。详细任务表见 [task_breakdown.md](task_breakdown.md)。

## 当前根问题

Phase 1 之前的根问题不是某个单独 bug，而是顶层架构连接还停留在标量点积路径：

```text
W PPBuf -> one scalar -> pe_w_in[0]
A PPBuf -> one scalar -> pe_a_in[0]
PE array -> result[0] only -> result FIFO
```

当前已经为这条路径补上 `u_scalar_pe` 兼容计算和写回，使标量 dot product 可验证通过。T2.4-T2.6 进一步补上了 4x4 tile-mode GEMM 的 4-lane 供数、16-output 收集和 INT8/FP16 golden 测试；16x16/8x32 高吞吐仍需要后续扩展。

## 修复层级

### L0：基线恢复（DONE）

目标：确保当前 RTL 可以被脚本稳定编译，并跑通一个非零标量点积。

工作：

1. 修复旧 testbench/SoC 的 `.ROWS/.COLS` 参数。
2. 统一所有脚本的 RTL 源列表。
3. 建立 `tb_npu_scalar_smoke`。
4. 修复当前非零结果为 0 的直接问题。

完成标准：

```text
PE 单测通过
标量 INT8 dot product 通过
无参数名编译错误
```

当前验证记录：

```text
scripts/run_sim.ps1      -> PASS=19 FAIL=0
tb_npu_scalar_smoke.v    -> PASS
tb_comprehensive.v       -> ALL 28 TESTS PASSED
scripts/run_full_sim.ps1 -> compile and simulation completed
```

剩余 SoC 编译问题不属于 L0 标量数据通路：`dram_model.v` 中 `axi_arlen` 未绑定，PicoRV32 PCPI 端口与参考核不匹配。

### L1：真实 4x4 tile

目标：满足基础评分项。

工作：

1. 已完成 T2.1：定义 4x4 tile 数据布局、A/W tile-pack、OS row skew 和 C serializer 顺序。
2. 已完成 T2.2：A/W buffer 支持 4-lane `rd_vec`，顶层接入阵列左上 4 行/列。
3. 已完成 T2.3：`npu_ctrl` 支持 `m_tile/n_tile`、row/col mask、`vec_consume`，顶层接入 OS row-skew feeder。
4. 已完成 T2.4：结果 serializer 写回 16 个输出，按 row-wise burst 写回 DRAM。
5. 边界 tile 用 mask 处理。

完成标准：

```text
4x4 INT8 GEMM 通过
非 4 整数倍 M/N 测试通过
4x4 FP16 GEMM 通过或给出明确容差
```

### L2：DMA burst

目标：满足 AXI Burst 和带宽要求。

工作：

1. 读通道 `ARLEN` 从固定 0 改为动态 burst。
2. 写通道支持连续 tile 写回。
3. 处理 4KB 边界。
4. 加 AXI beat/cycle 统计。

完成标准：

```text
8-beat/16-beat INCR burst 测试通过
带宽利用率报告可在仿真中输出
```

### L3：PSUM 和 K-split

目标：支持大 K 和多层卷积中间结果。

工作：

1. 增加 `PSUM/OUT_BUF`。
2. PE 支持 accumulator init。
3. controller 增加 K tile 循环。
4. K 未结束写 psum，K 结束写 final output。

完成标准：

```text
K > buffer depth 的 GEMM 通过
K-split 结果等于未切分 golden
```

### L4：Descriptor 和多层

目标：解决“什么时候下一层开始、什么时候所有卷积结束”。

工作：

1. 定义 descriptor 格式。
2. CPU 写 `DESC_BASE/DESC_COUNT`。
3. NPU fetch descriptor。
4. `NEXT_LAYER` 自动切换 IFM/OFM/weight 地址。
5. layer done 和 network done 分开报告。

完成标准：

```text
两层 GEMM/FC 串联通过
两层卷积预展开 im2col 测试通过
```

### L5：卷积前端和后处理

目标：从 GEMM 原语走向卷积推理。

工作：

1. 第一版允许 DRAM 中预展开 im2col。
2. 第二版增加 on-the-fly im2col 地址发生器。
3. 增加 bias、ReLU、ReLU6、quant/saturate。

完成标准：

```text
conv + ReLU 单层通过
conv + ReLU + conv 两层通过
```

### L6：可重构和性能

目标：拿优化分。

工作：

1. 8x8/16x16 lane 扩展。
2. 8x32 折叠路由验证。
3. INT8 SIMD PE。
4. clock gating/DFS 接入。

完成标准：

```text
16x16 full-lane 测试通过
8x32 输出顺序正确
性能计数达到可解释的 GOPS/TOPS
```

## 关键设计取舍

### 先预展开 im2col，再做 on-the-fly

直接做 on-the-fly im2col 会同时引入窗口地址、边界 padding、stride、DMA 非连续访问等复杂度。建议先让 CPU/脚本在 DRAM 中准备 `A_im2col`，验证 GEMM 核心正确后再把 im2col 前端搬到硬件。卷积变量含义和 `M/N/K` 映射见 [conv_gemm_mapping.md](conv_gemm_mapping.md)。

### 先 4x4，再 16x16

16x16 的难点不是 PE 数量，而是每拍供数和结果收集。4x4 跑通后，接口可以自然推广到 `LANES=8/16`。

### 先 clock enable，再真实门控时钟

FPGA 中直接门控时钟容易引入时钟树问题。第一版应使用 PE 内 `en` 或 clock enable；报告中说明 ASIC 可替换为 ICG。

## 下一步

`T2.4/T2.5/T2.6` 已完成 16-output serializer、row-wise writeback、4x4 INT8 GEMM golden 测试和 4x4 FP16 GEMM golden 测试。当前下一步进入 Phase 3：AXI burst DMA 和带宽统计，继续保留标量兼容路径作为回归基线。
