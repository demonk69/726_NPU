# 当前实现状态

更新时间：2026-04-29



## 验证环境

```text
E:\iverilog\bin\iverilog.exe
E:\iverilog\bin\vvp.exe
Icarus Verilog 12.0
```

如果命令行找不到 `iverilog`，先执行：

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
```

## 实测结果

| 项目 | 当前结果 | 结论 |
|---|---:|---|
| `scripts/run_sim.ps1` | 19 PASS / 0 FAIL | 单 PE 基线可用 |
| `tb/tb_npu_scalar_smoke.v` | PASS | 顶层标量 INT8 OS 路径可用，`CFG_SHAPE` start 锁存有效 |
| `tb/tb_pingpong_buf_vec.v` | PASS | PPBuf 4-lane INT8/FP16 vector read 可用 |
| `tb/tb_npu_ctrl_tile.v` | PASS | 4x4 tile planner 和 M/N 边界 mask 可用 |
| `tb/tb_npu_tile_writeback.v` | PASS | 4x4 tile 16 个阵列输出可通过 serializer 按 row-wise burst 写回 |
| `tb/tb_npu_tile_gemm.v` + `tb/tile4/int8_4x4x4` | PASS | 4x4 INT8 GEMM 与 Python golden 一致 |
| `tb/tb_npu_tile_gemm.v` + `tb/tile4/fp16_4x4x4` | PASS | 4x4 FP16 GEMM 与 FP32 golden 容差一致 |
| `tb/tb_dma_read_burst.v` | PASS | DMA 读通道可产生 INCR burst，支持连续地址和 4KB 边界切分 |
| `tb/tb_dma_write_burst.v` | PASS | DMA 写通道可产生 INCR burst，支持多 burst 和 4KB 边界切分 |
| `tb/tb_dma_burst.v` | PASS | 混合读写场景下 8/16 beat burst 地址和数据正确 |
| `tb/tb_dma_perf.v` | PASS | 长 burst 场景 read util 85.04%，write util 77.34% 并解释 80% 差距 |
| `tb/tb_psum_out_buf.v` | PASS | PSUM/OUT buffer 支持 4x4 tile read-modify-write、边界 mask 和双 bank 隔离 |
| `tb/tb_reconfig_pe_acc_init.v` | PASS | 4x4 PE array 可从 per-PE psum 初始化后继续 OS MAC |
| `tb/tb_npu_ctrl_ksplit.v` | PASS | controller 可将 K=10 按 PPB_DEPTH=4 切成 4/4/2 三个 k_tile，最终 k_tile 后才写回 |
| `tb/tb_npu_ctrl_dataflow_modes.v` | PASS | direct scalar OS/WS controller 分支均可跑到 done，WS `pe_load_w` 覆盖 K 个周期 |
| `tb/tb_npu_ctrl_error_status.v` | PASS | controller 可锁存 desc_count=0、unsupported descriptor、descriptor count exhausted 并由 W1C 清除 |
| `scripts/run_matmul_case.ps1` | PASS | 自定义 direct matmul 可生成并运行大矩阵：32x32x32 INT8 OS/WS 均 1024 checks PASS，16x16x16 FP16 OS 256 checks PASS |
| `tb/tb_npu_tile_ksplit_gemm.v` | PASS | 顶层 4x4x10 INT8 OS GEMM 可按 4/4/2 K-split 累加，最终结果等于未切分 golden |
| `tb/tb_npu_axi_lite_desc.v` | PASS | AXI-Lite descriptor 寄存器、STATUS busy/done/error、done/error IRQ 和 ERR_STATUS W1C 可用 |
| `tb/tb_npu_desc_two_layer.v` | PASS | descriptor mode 可顺序 fetch/decode 两个 4x4 INT8 OS tile GEMM descriptor，并在 LAST_LAYER 后 done |
| `tb/tb_npu_desc_ofm_chain.v` | PASS | descriptor bit23 可让第二层使用第一层 32-bit row-major OFM 作为 IFM，DMA 完成 INT8 gather/repack |
| `tb/tb_npu_scalar_smoke.v` | PASS | 标量路径可用，且 AXI perf counters 寄存器可读 |
| `scripts/run_regression.ps1` | 1078 PASS / 12 FAIL | 新增 OS/WS dataflow 单测通过；剩余失败集中在旧 matmul 2x3x2 OS/WS case |
| 当前源列表手工编译运行 `tb_comprehensive.v` | 28 PASS / 0 FAIL | Phase 1 顶层标量兼容路径已恢复 |
| `scripts/run_full_sim.ps1` | 编译和仿真完成 | 脚本源列表与 testbench 参数已对齐 |
| `scripts/run_soc_sim.ps1` | 编译失败 | `dram_model.v` 的 `axi_arlen` 绑定问题和 PicoRV32 PCPI 端口不匹配 |

## 当前已经具备

1. `pe_top` 支持 INT8 和 FP16 MAC，含 WS/OS 累加语义。
2. `npu_axi_lite` 提供 CPU 配置寄存器，包括 `CTRL`、维度、地址、`CFG_SHAPE`。
3. `npu_ctrl` 有按 `C[i][j]` 单点输出推进的标量 FSM，也有 `ARR_CFG[7]` 控制的 4x4 tile planner。
4. `npu_dma` 有 AXI4 master 读写框架；读侧已支持 INCR burst，可把 W/A 读入 PPBuf；写侧已支持多 INCR burst 和 4KB 边界切分，可把结果 FIFO 连续写回 DRAM。
5. `pingpong_buf` 支持 DMA 写 32-bit word，PE 侧按 INT8/FP16 拆出 16-bit 标量或 4-lane vector 数据。
6. `reconfig_pe_array` 有 16x16 物理阵列和 4x4/8x8/16x16/8x32 形态选择接口。
7. `npu_power` 有 DFS 和 row/col clock gating 行为模型。
8. `npu_top` 有一条 Phase 1 标量兼容路径：PPBuf 标量输出进入 `u_scalar_pe`，结果通过 FIFO 写回。
9. `npu_ctrl` 已输出 `cfg_shape_latched`，当前任务启动后修改 `CFG_SHAPE` 不影响正在运行的阵列配置。
10. T2.1 已定义 4x4 tile 数据布局：OS 模式采用 `PE row -> M lane`、`PE col -> N lane`，A/W 使用 4-lane tile-pack，A row 由 feeder 错拍输入，C 按 `result[r*4+c]` 输出。
11. T2.2 已给 `pingpong_buf` 增加 4-lane vector read port，并在 `npu_top` 中把 W/A vector 接入 `reconfig_pe_array` 左上 4x4 边界；其中 A lane1/2/3 进入 PE 前会延迟 1/2/3 拍。
12. T2.3 已实现 `ARR_CFG[7]` 控制的 4x4 tile planner：输出 tile base、row/col mask、`vec_consume` 和 OS row-skew feeder。
13. T2.4 已实现阵列 16-output serializer 和 row-wise writeback，4x4 tile 输出顺序为 `result[r*4+c]`。
14. T2.5/T2.6 已建立 4x4 INT8/FP16 tile-mode GEMM golden 测试。
15. T3.1 已实现 DMA 读通道 INCR burst，`ARLEN` 不再固定为 0，并有 4KB 边界切分。
16. T3.2 已实现 DMA 写通道多 burst 和 4KB 边界切分，`AWLEN/WLAST` 按当前 burst 生成。
17. T3.3 已接入 `axi_monitor`，通过 AXI-Lite `0x48..0x70` 暴露 AXI master burst/beat/byte/cycle、bytes-per-cycle 和 utilization counters。
18. T3.4 已建立 `tb_dma_burst.v` 混合读写 burst 正确性测试，覆盖 8/16 beat 数据无误。
19. T3.5 已建立 `tb_dma_perf.v` 带宽利用率目标测试：read 85.04%，write 77.34%，write 未达 80% 的原因已记录为 single-outstanding/B-response gap。
20. T4.1 已在 `doc/architecture.md` 明确 `PSUM/OUT_BUF` 规格：32-bit accumulator word、4x4 tile-local depth、row-major 外部 PSUM/OFM 地址和 K-split 调度语义。
21. T4.2 已新增 `rtl/buf/psum_out_buf.v` 和 `tb/tb_psum_out_buf.v`，支持 2-bank 4x4 tile PSUM/OUT 存储、RMW 累加、边界 mask 和 bank 隔离。
22. T4.3 已给 `pe_top` 和 `reconfig_pe_array` 增加 accumulator init，支持 PE/array 从 PSUM 初值继续 MAC。
23. T4.4 已给 `npu_ctrl` 增加 k_tile loop，A/W DMA 地址和长度按 K slice 生成，中间 k_tile 不 flush/writeback，最后 k_tile 写回最终 tile。
24. T4.5 已建立顶层 K-split GEMM golden：`tb_npu_tile_ksplit_gemm.v` 用 `PPB_DEPTH=4` 将 4x4x10 INT8 OS GEMM 切成 4/4/2，并验证最终 C 矩阵等于未切分 golden。
25. T5.1 已在 `doc/user_manual.md` 固定 descriptor v1 ABI：16 个 32-bit word、64 byte、`desc_ctrl` bitfield、CPU `npu_desc_v1_t` 和 RTL word/bit localparam 口径一致。
26. T5.2 已在 `npu_axi_lite` 增加 `DESC_BASE(0x40)`、`DESC_COUNT(0x44)` 和 `CTRL[7] desc_mode` 读写路径。
27. T5.3 已给 `npu_dma` 增加 64-byte descriptor fetch 目标，并给 `npu_ctrl` 增加 `FETCH_DESC/DECODE_DESC/NEXT_LAYER` 控制流，可把 descriptor v1 映射到当前 4x4 OS tile GEMM 直配语义并顺序执行多个 descriptor。
28. T5.4 已增加 `desc_ctrl[23] IFM_FROM_PREV_OFM`：`npu_ctrl` 记录上一层 `ofm_addr`，下一层置位后由 `npu_dma` 从上一层 row-major 32-bit OFM gather/repack 为 4-lane INT8 A tile stream。
29. T5.5 已接入 done/error 可见性：`STATUS[2]`、`INT_EN/INT_CLR`、`ERR_STATUS(0x74)` W1C 和 controller descriptor 错误锁存可用。
30. 仿真口径已明确：4x4 tile、K-split、descriptor 主线当前基于 OS；direct scalar matmul 回归同时覆盖 OS 和 WS，`tb_npu_ctrl_dataflow_modes.v` 额外显式覆盖 OS/WS 控制分支。
31. `scripts/run_matmul_case.ps1` 可按参数生成并运行 direct scalar 大矩阵功能测试，已验证 32x32x32 INT8 OS/WS 和 16x16x16 FP16 OS。

## 当前关键差距

1. 当前 4x4 tile-mode GEMM 已通过 INT8/FP16 基础验证，但 8x8/16x16/8x32 还没有宽向量供数和完整写回验证。
2. 顶层 K-split GEMM golden 已通过；外部 PSUM surface 的 DMA read/writeback 尚未接入多层 descriptor 流。
3. Descriptor v1 fetch/decode/next-layer 已具备第一版：支持 `OP=GEMM_TILEPACK`、`DTYPE=INT8/FP16`、`DATAFLOW=OS`、`SHAPE=4x4`、`TILE_PACKED=1` 映射到现有 tile GEMM 数据通路；T5.4 已验证 INT8 层间 OFM 作为下一层 IFM；T5.5 已让 unsupported descriptor 和 descriptor count exhausted 等错误对 CPU 可见。外部 PSUM surface、FP16 OFM 格式转换、bias/activation/quant 仍未接入 descriptor 流。
4. DMA 读写 burst、混合 8/16 beat 正确性、基础带宽计数和 60%/80% 利用率目标报告已具备；后续若要冲 80% write util，需要允许多 outstanding write burst 或重叠 B response 间隔。
5. `npu_power` 输出没有接入 PE 主时钟路径。
6. SoC 仿真仍存在 DRAM 模型信号绑定和 PicoRV32 PCPI 端口不匹配。
7. 当前没有硬件 on-the-fly im2col 地址发生器；卷积需要先由软件/testbench 预展开或预打包为 GEMM/tile 流。

## 不应继续引用的旧结论

| 旧说法 | 当前判断 |
|---|---|
| `tb_comprehensive` 失败 2/28 PASS | 已过期，当前是 28/28 PASS |
| 全量回归 903 PASS | 当前不能作为事实引用 |
| SoC 集成已验证 | 当前需要重新修复和验证 |
| 4x4 tile 结果还没有从阵列写回 | 已过期，T2.4 已完成 serializer 和 row-wise writeback |
| 16x16/8x32 已完成高吞吐矩阵乘 | 阵列存在，但顶层尚未喂满更大形态，也缺少完整验证 |
| DMA 读通道固定 `ARLEN=0` | 已过期，T3.1 已支持 INCR read burst |
| DMA 写通道固定单 burst | 已过期，T3.2 已支持多 INCR write burst 和 4KB 边界切分 |
| DFS/时钟门控已实际降功耗 | 行为模块输出悬空 |
| 支持 INT16 | 当前 PE 只有 INT8/FP16 |

## 当前一句话定位

这是一个具备 PE 算术、NPU 外围框架、可验证 4x4 tile GEMM 路径、顶层 K-split GEMM golden、DMA read/write burst、混合 burst 正确性测试、AXI perf counters、带宽利用率报告、PSUM/OUT buffer RMW 模块、accumulator-init PE array、controller k_tile loop、descriptor v1 ABI、AXI-Lite descriptor 提交寄存器、descriptor 多任务顺序执行、INT8 OFM->IFM 两层串联以及 IRQ/error status 的原型。下一阶段重点是外部 PSUM surface、多层卷积和 16x16 高吞吐阵列。
