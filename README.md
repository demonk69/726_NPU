# NPU_prj

面向边缘 AI 推理的异构处理器原型：PicoRV32 CPU 负责控制和任务编排，NPU 负责矩阵乘和卷积中的高负载计算。项目目标是通过 AXI4-Lite 配置通路和 AXI4 DMA 数据通路，实现可验证、可扩展、低功耗的 NPU 加速器。

更新时间：2026-04-27

## 当前结论

当前仓库已经具备 PE、AXI-Lite、DMA、Ping-Pong Buffer、16x16 可重构阵列外壳和 SoC 框架。Phase 1 已恢复顶层最小正确性：标量 INT8 dot product 可通过 `npu_top` 完成 DMA 读入、PPBuf 供数、单 PE 计算、FIFO 写入和 DMA 写回。

已确认可依赖：

- `pe_top` 单 PE INT8/FP16、WS/OS 基本功能通过单元测试。
- AXI-Lite 寄存器文件、DMA、PPBuf、阵列模块均有 RTL 框架。
- `reconfig_pe_array` 已实例化 16x16 物理 PE，并有 4x4/8x8/16x16/8x32 形态选择接口。
- `npu_top` 当前有一条标量兼容计算路径，`tb_npu_scalar_smoke.v` 与 `tb_comprehensive.v` 已通过。

当前不能作为完成项声明：

- 顶层 NPU 已完成真正 4x4/16x16 并行矩阵乘法。
- 16x16 或 8x32 阵列已被数据喂满并产生并行吞吐。
- DMA 读通道已经支持多拍 INCR burst。
- SoC 集成当前已通过验证。
- DFS/时钟门控已经实际驱动 PE 主时钟。
- INT16 已实现。

## 目标架构

最终目标不是让 NPU 猜测卷积规模，而是由 CPU 或 descriptor 明确描述每层任务：

```text
CPU writes descriptor list
  -> NPU fetches layer descriptor
  -> DMA loads activation/weight tiles
  -> PE array computes GEMM tiles
  -> PSUM/OUT buffer stores partial or final results
  -> post-process applies bias/ReLU/quant
  -> next layer uses previous output as input
```

卷积映射为：

```text
A_im2col[M,K] * W_col[K,N] = C[M,N]
M = OH * OW * batch
K = Cin * KH * KW
N = Cout
```

完整设计需要三类片上存储：

- `A_BUF ping/pong`：activation tile，建议由窗口地址发生器生成 im2col 流。
- `W_BUF ping/pong`：weight tile。
- `PSUM/OUT_BUF ping/pong`：K-split 部分和、层间输出、后处理输入。

## 文档入口

| 文档 | 内容 |
|---|---|
| [doc/current_status.md](doc/current_status.md) | 当前 RTL 事实、已验证项和不应再引用的旧结论 |
| [doc/architecture.md](doc/architecture.md) | 目标 NPU 架构、PE 数据流、FSM、AXI/DMA 带宽设计 |
| [doc/module_reference.md](doc/module_reference.md) | 各 RTL 模块职责、当前差距和改造方向 |
| [doc/task_breakdown.md](doc/task_breakdown.md) | 后续任务拆分，按优先级一个一个解决 |
| [doc/user_manual.md](doc/user_manual.md) | CPU 侧编程模型和寄存器/descriptor 规划 |
| [doc/simulation_guide.md](doc/simulation_guide.md) | 当前可运行仿真、失败入口和后续验证顺序 |
| [doc/npu_debug_checklist.md](doc/npu_debug_checklist.md) | 调试检查清单 |
| [doc/soc_integration_plan.md](doc/soc_integration_plan.md) | PicoRV32 + NPU + AXI/DRAM 集成计划 |
| [doc/architecture_fix_plan.md](doc/architecture_fix_plan.md) | 从当前原型演进到目标架构的修复路线 |
| [doc/git_guide.md](doc/git_guide.md) | Git 协作说明 |

## 目录结构

```text
rtl/
  pe/       PE 与 FP16/FP32 算术
  array/    PE 阵列与可重构阵列
  axi/      AXI-Lite 寄存器与 AXI4 DMA
  buf/      Ping-Pong Buffer
  ctrl/     NPU 控制 FSM
  power/    DFS/clock gating 行为模型
  soc/      PicoRV32 SoC 外壳与存储模型
  top/      NPU 顶层集成
tb/         testbench 与测试数据
scripts/    仿真脚本
doc/        架构、任务、验证、使用文档
constraints/ FPGA 约束
```

## 当前仿真基线

Windows 环境中可用工具位于：

```powershell
E:\iverilog\bin\iverilog.exe
E:\iverilog\bin\vvp.exe
```

单 PE 基线：

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

当前结果：

```text
PASS=19 FAIL=0
```

当前顶层标量基线：

```text
tb/tb_npu_scalar_smoke.v -> PASS
tb/tb_pingpong_buf_vec.v -> PASS
tb/tb_npu_ctrl_tile.v    -> PASS
tb_comprehensive.v       -> ALL 28 TESTS PASSED
scripts/run_full_sim.ps1 -> compile and simulation completed
```

SoC 仿真入口已经越过旧 `.ROWS/.COLS` 参数问题，但仍被 `dram_model.v` 的 `axi_arlen` 绑定问题和 PicoRV32 PCPI 端口不匹配阻塞。T2.1/T2.2/T2.3/T2.4/T2.5/T2.6 已完成：4x4 tile 的 A/W tile-pack、4-lane vector read、tile planner、vector consume、OS row-skew feeder、16-output serializer、row-wise writeback，以及 4x4 INT8/FP16 GEMM golden 测试已落地。下一步进入 Phase 3 的 AXI burst DMA 和带宽统计。

## 性能目标口径

32-bit AXI @ 500 MHz 理论带宽：

```text
raw = 32 bit * 500 MHz = 2.0 GB/s
80% target = 1.6 GB/s
```

INT8 峰值算力按 MAC 计两个操作：

```text
4x4 scalar @500MHz     = 16 GOPS
16x16 scalar @500MHz   = 256 GOPS
16x16 2-lane @500MHz   = 0.512 TOPS
16x16 4-lane @500MHz   = 1.024 TOPS
```

因此要达到 0.5-1 TOPS，不能只依赖 16x16 scalar PE；PE 内部至少需要 2-lane 或 4-lane INT8 SIMD。
