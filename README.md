# NPU_prj

面向边缘 AI 推理的异构处理器原型：PicoRV32 CPU 负责控制和任务编排，NPU 负责矩阵乘和卷积中的高负载计算。项目目标是通过 AXI4-Lite 配置通路和 AXI4 DMA 数据通路，实现可验证、可扩展、低功耗的 NPU 加速器。

更新时间：2026-04-29

## 当前结论

当前仓库已经具备 PE、AXI-Lite、DMA、Ping-Pong Buffer、16x16 可重构阵列外壳和 SoC 框架。Phase 1 已恢复顶层标量最小正确性；Phase 2 已完成可验证的 4x4 tile-mode GEMM 路径，包括 16-output serializer、row-wise writeback、INT8/FP16 4x4 golden 测试。

已确认可依赖：

- `pe_top` 单 PE INT8/FP16、WS/OS 基本功能通过单元测试。
- AXI-Lite 寄存器文件、DMA、PPBuf、阵列模块均有 RTL 框架。
- `npu_dma` 读/写通道已支持 INCR burst，T3.1/T3.2 定向测试覆盖 `ARLEN/AWLEN>0` 和 4KB 边界切分。
- `tb_dma_burst.v` 已覆盖混合读写场景下 8/16 beat burst 地址和数据正确性。
- `tb_dma_perf.v` 已输出长 burst 带宽利用率报告：read 85.04%，write 77.34%，并解释 write 低于 80% 的原因。
- `axi_monitor` 已接入 `npu_top`，AXI-Lite `0x48..0x70` 可读出 DMA burst/beat/byte/cycle、bytes-per-cycle 和 utilization counters。
- `psum_out_buf` 已实现 2-bank 4x4 tile PSUM/OUT RMW 存储，并通过边界 mask、bank isolation 和 K-split RMW 单测。
- `pe_top` 和 `reconfig_pe_array` 已支持 accumulator init，4x4 PE array 可从 per-PE psum 初值继续 MAC。
- `npu_ctrl` 已支持 tile-mode k_tile loop，K 超过 PPB 深度时按 K slice 生成 A/W DMA 地址和长度，最后一个 k_tile 才 flush/writeback。
- `tb_npu_tile_ksplit_gemm.v` 已验证顶层 4x4x10 INT8 OS GEMM 按 4/4/2 K-split 执行后等于未切分 golden。
- Descriptor v1 ABI 已固定为 16 个 32-bit word / 64 byte，CPU `npu_desc_v1_t` 和 NPU RTL word/bit localparam 口径一致。
- AXI-Lite `DESC_BASE(0x40)`、`DESC_COUNT(0x44)` 和 `CTRL[7] desc_mode` 已可写入并读回。
- `npu_ctrl` 已支持第一版 descriptor fetch/decode/next-layer，可顺序执行多个 4x4 OS tile-pack GEMM descriptor。
- T5.4 已支持 INT8 descriptor 链中上一层 32-bit row-major OFM 作为下一层 IFM，DMA 会 gather/repack 成 A tile stream。
- `reconfig_pe_array` 已实例化 16x16 物理 PE，并有 4x4/8x8/16x16/8x32 形态选择接口。
- `npu_top` 保留标量兼容路径，同时在 `ARR_CFG[7]=1` 时可走 4x4 tile-mode 阵列路径。
- `tb_npu_scalar_smoke.v`、`tb_npu_tile_writeback.v`、`tb_npu_tile_gemm.v` 与 `tb_comprehensive.v` 已通过。

当前不能作为完成项声明：

- 16x16 或 8x32 阵列已被数据喂满并产生并行吞吐。
- FP16/量化后的通用层间 OFM 已自动作为下一层 IFM。
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

卷积映射为 GEMM。你的理解是正确的，普通 dense Conv2D 可按下面的行/列/归约维度展开；更完整的变量含义、stride/pad/dilation 公式和索引关系见 [doc/conv_gemm_mapping.md](doc/conv_gemm_mapping.md)。

```text
# A_im2col[M,K] * W_col[K,N] = C[M,N]
# M: GEMM 行数，卷积中等于 batch 内所有输出空间位置数量。
# K: GEMM 归约维度，卷积中等于一个输入卷积窗口的元素数量。
# N: GEMM 列数，卷积中等于输出通道数/卷积核个数。
A_im2col[M,K] * W_col[K,N] = C[M,N]
M = batch * OH * OW
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
| [doc/conv_gemm_mapping.md](doc/conv_gemm_mapping.md) | 卷积映射为 GEMM 的公式、变量含义和索引展开 |
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
tb/tb_npu_tile_writeback.v -> PASS
tb/tb_npu_tile_gemm.v + tb/tile4/int8_4x4x4 -> PASS
tb/tb_npu_tile_gemm.v + tb/tile4/fp16_4x4x4 -> PASS
tb/tb_dma_read_burst.v -> PASS
tb/tb_dma_write_burst.v -> PASS
tb/tb_dma_burst.v -> PASS
tb/tb_dma_perf.v -> PASS
tb/tb_psum_out_buf.v -> PASS
tb/tb_reconfig_pe_acc_init.v -> PASS
tb/tb_npu_ctrl_ksplit.v -> PASS
tb/tb_npu_ctrl_dataflow_modes.v -> PASS
tb/tb_npu_ctrl_error_status.v -> PASS
scripts/run_matmul_case.ps1 custom 32x32x32 INT8 OS/WS -> PASS
tb/tb_npu_tile_ksplit_gemm.v -> PASS
tb/tb_npu_axi_lite_desc.v -> PASS
tb/tb_npu_desc_two_layer.v -> PASS
tb/tb_npu_desc_ofm_chain.v -> PASS
tb_comprehensive.v       -> ALL 28 TESTS PASSED
scripts/run_full_sim.ps1 -> compile and simulation completed
```

SoC 仿真入口已经越过旧 `.ROWS/.COLS` 参数问题，但仍被 `dram_model.v` 的 `axi_arlen` 绑定问题和 PicoRV32 PCPI 端口不匹配阻塞。T2.1-T2.6 已完成 4x4 tile 的 A/W tile-pack、4-lane vector read、tile planner、vector consume、OS row-skew feeder、16-output serializer、row-wise writeback，以及 4x4 INT8/FP16 GEMM golden 测试。T3.1-T3.5 已完成 AXI read/write burst、4KB 边界切分、AXI perf counters、混合 burst 正确性测试和带宽利用率目标测试；T4.1-T4.5 已明确并实现 PSUM/OUT Buffer 的 tile-local RMW 存储、accumulator init、controller k_tile loop 和顶层 K-split GEMM golden。T5.1 已固定 descriptor v1 二进制格式，T5.2 已给 AXI-Lite 增加 `DESC_BASE/DESC_COUNT`，T5.3 已实现第一版 descriptor fetch/decode/next-layer，T5.4 已实现 INT8 OFM->IFM 两层 GEMM 串联，T5.5 已补齐 `STATUS.error`、done/error IRQ 和 `ERR_STATUS(0x74)` W1C 错误状态。当前 tile/descriptor 主线是 OS；direct scalar matmul 回归同时覆盖 OS 和 WS，并已通过自定义 32x32x32 INT8 OS/WS 功能测试。下一步进入外部 PSUM surface、卷积前端/后处理或 16x16 高吞吐阵列。

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
