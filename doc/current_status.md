# 当前实现状态

更新时间：2026-05-13



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
| `scripts/run_sim.ps1` | 28 PASS / 0 FAIL | 单 PE INT8/FP16、accumulator init、T7.3 2-lane 和 T7.4 4-lane INT8 SIMD 可用 |
| `tb/tb_npu_scalar_smoke.v` | PASS | 顶层标量 INT8 OS 路径可用，`CFG_SHAPE` start 锁存有效 |
| `tb/tb_pingpong_buf_vec.v` | PASS | PPBuf 4/8/16-lane INT8 vector read 和 4/16-lane FP16 vector read 可用 |
| `tb/tb_npu_ctrl_tile.v` | PASS | 4x4 tile planner 和 M/N 边界 mask 可用 |
| `tb/tb_npu_tile_writeback.v` | PASS | 4x4 tile 16 个阵列输出可通过 serializer 按 row-wise burst 写回 |
| `tb/tb_npu_tile_lane_feed.v` | PASS | 8x8/16x16 tile feeder 可通过真实 AXI/DMA 路径把 active W/A lane 送到阵列边界 |
| `tb/tb_npu_tile_gemm.v` + `tb/tile4/int8_4x4x4` | PASS | 4x4 INT8 GEMM 与 Python golden 一致 |
| `tb/tb_npu_tile_gemm.v` + `tb/tile4/fp16_4x4x4` | PASS | 4x4 FP16 GEMM 与 FP32 golden 容差一致 |
| `scripts/run_tile_edge_case.ps1 -Shape 4x4 -Mode OS -Lanes 1` | PASS | P2 tile edge 4x4 single-lane OS 走 tile mode，16 个 OFM 与 golden 一致 |
| `scripts/run_tile_edge_case.ps1 -Shape 4x4 -Mode WS -Lanes 1 -DumpResult` | PASS | P2.3.1 以 WS row-vector micro-run 方式闭合 4x4 single-lane edge tile，testbench 累加 16 个结果并输出 `npu_output.hex` |
| `tb/tb_dma_read_burst.v` | PASS | DMA 读通道可产生 INCR burst，支持连续地址和 4KB 边界切分 |
| `tb/tb_dma_write_burst.v` | PASS | DMA 写通道可产生 INCR burst，支持多 burst 和 4KB 边界切分 |
| `tb/tb_dma_burst.v` | PASS | 混合读写场景下 8/16 beat burst 地址和数据正确 |
| `tb/tb_dma_perf.v` | PASS | 长 burst 场景 read util 85.04%，write util 77.34% 并解释 80% 差距 |
| `tb/tb_op_counter_perf.v` | PASS | T7.5 TOPS fixed-point 和 compute/e2e utilization 公式可验证 |
| `tb/tb_psum_out_buf.v` | PASS | PSUM/OUT buffer 支持 4x4 tile read-modify-write、边界 mask 和双 bank 隔离 |
| `tb/tb_reconfig_pe_acc_init.v` | PASS | 4x4 PE array 可从 per-PE psum 初始化后继续 OS MAC |
| `tb/tb_reconfig_pe_8x32.v` | PASS | 8x32 阵列级折叠路由、32-lane 输出顺序和 WS 8-row load wrap 均通过 |
| `tb/tb_npu_ctrl_ksplit.v` | PASS | controller 可将 K=10 按 PPB_DEPTH=4 切成 4/4/2 三个 k_tile，最终 k_tile 后才写回 |
| `tb/tb_npu_ctrl_dataflow_modes.v` | PASS | direct scalar OS/WS controller 分支均可跑到 done，WS `pe_load_w` 覆盖 K 个周期 |
| `tb/tb_npu_ctrl_error_status.v` | PASS | controller 可锁存 desc_count=0、unsupported descriptor、descriptor count exhausted 并由 W1C 清除 |
| `scripts/run_matmul_case.ps1` | PASS | 自定义 direct matmul 可生成并运行大矩阵：32x32x32 INT8 OS/WS 均 1024 checks PASS，16x16x16 FP16 OS 256 checks PASS |
| `scripts/run_conv2d_im2col_case.ps1` | PASS | T6.1 DRAM 预展开 Conv2D im2col：默认 1x5x5x2, 3x3, Cout=3 case 在 INT8 OS/WS 和 FP16 OS 下均 75 checks PASS |
| `scripts/run_conv2d_otf_case.ps1` | PASS | T6.2 on-the-fly Conv2D im2col：DRAM 只保存 raw NCHW IFM，默认 case 在 INT8 OS/WS 和 FP16 OS 下均 75 checks PASS |
| `scripts/run_matmul_case.ps1 -Bias` | PASS | T6.3 direct scalar 32-bit bias：INT8 OS/WS 和 FP16 OS/WS targeted case 均通过 |
| `scripts/run_matmul_case.ps1 -Bias -Activation relu/relu6` | PASS | T6.4 direct scalar activation：INT8/FP16、OS/WS 的 ReLU/ReLU6 targeted case 均通过 |
| `scripts/run_conv2d_otf_case.ps1 -Bias -Activation relu/relu6` | PASS | T6.4 Conv2D on-the-fly 后处理：INT8 OS ReLU、INT8 WS ReLU6、FP16 OS ReLU6 均 75 checks PASS |
| `scripts/run_matmul_case.ps1 -Quant` | PASS | T6.5 direct scalar INT8 quant/saturate：OS/WS targeted case 均通过 |
| `scripts/run_conv2d_otf_case.ps1 -Quant` | PASS | T6.5 Conv2D on-the-fly INT8 quant/saturate：OS/WS targeted case 均 75 checks PASS |
| `scripts/run_conv2d_two_layer_case.ps1` | PASS | T6.6 两层 Conv2D E2E：layer0 量化 OFM 直接作为 layer1 A 输入，layer0+layer1 共 48 checks PASS |
| `tb/tb_npu_tile_ksplit_gemm.v` | PASS | 顶层 4x4x10 INT8 OS GEMM 可按 4/4/2 K-split 累加，最终结果等于未切分 golden |
| `tb/tb_npu_axi_lite_desc.v` | PASS | AXI-Lite descriptor 寄存器、STATUS busy/done/error、done/error IRQ 和 ERR_STATUS W1C 可用 |
| `tb/tb_npu_desc_two_layer.v` | PASS | descriptor mode 可顺序 fetch/decode 两个 4x4 INT8 OS tile GEMM descriptor，并在 LAST_LAYER 后 done |
| `tb/tb_npu_desc_ofm_chain.v` | PASS | descriptor bit23 可让第二层使用第一层 32-bit row-major OFM 作为 IFM，DMA 完成 INT8 gather/repack |
| `tb/tb_npu_scalar_smoke.v` | PASS | 标量路径可用，AXI perf counters 和 T7.5 TOPS/util 寄存器可读 |
| `scripts/run_regression.ps1` | 2330 PASS / 0 FAIL | direct scalar matmul、4x4 tile、T7.1 lane feed、T7.2 8x32 折叠路由、T7.3/T7.4 PE INT8 SIMD、T7.5 op_counter_perf、T6.1/T6.2 Conv2D、T6.3 bias、T6.4 ReLU/ReLU6、T6.5 INT8 quant/saturate 和 T6.6 两层 Conv2D E2E 默认 case 均通过 |
| 当前源列表手工编译运行 `tb_comprehensive.v` | 28 PASS / 0 FAIL | Phase 1 顶层标量兼容路径已恢复 |
| `scripts/run_full_sim.ps1` | 编译和仿真完成 | 脚本源列表与 testbench 参数已对齐 |
| `scripts/run_soc_sim.ps1` | PASS | PicoRV32 配置 NPU 完成 2x2 INT8 GEMM，testbench 独立确认 `C00=19 C01=22 C10=43 C11=50` |

## 当前已经具备

1. `pe_top` 支持 INT8 和 FP16 MAC，含 WS/OS 累加语义；T7.3/T7.4 已支持 packed INT8 2-lane/4-lane SIMD，并保持旧 sign-extended scalar INT8 兼容。
2. `npu_axi_lite` 提供 CPU 配置寄存器，包括 `CTRL`、维度、地址、`CFG_SHAPE`。
3. `npu_ctrl` 有按 `C[i][j]` 单点输出推进的标量 FSM，也有 `ARR_CFG[7]` 控制的 4x4 tile planner。
4. `npu_dma` 有 AXI4 master 读写框架；读侧已支持 INCR burst，可把 W/A 读入 PPBuf；写侧已支持多 INCR burst 和 4KB 边界切分，可把结果 FIFO 连续写回 DRAM。
5. `pingpong_buf` 支持 DMA 写 32-bit word，PE 侧按 INT8/FP16 拆出 16-bit 标量或运行时 4/8/16-lane vector 数据。
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
30. 仿真口径已明确：4x4 tile、K-split、descriptor 主线当前基于 OS；P2.3.1 已额外验证 `4x4 / single-lane / WS row-vector micro-run` tile edge case；direct scalar matmul 回归同时覆盖 OS 和 WS，`tb_npu_ctrl_dataflow_modes.v` 额外显式覆盖 OS/WS 控制分支。
31. `scripts/run_matmul_case.ps1` 可按参数生成并运行 direct scalar 大矩阵功能测试，已验证 32x32x32 INT8 OS/WS、16x16x16 FP16 OS，以及 K 非 32-bit 对齐的 25x18x3 INT8 OS。
32. T6.1 已完成第一版 DRAM 预展开 im2col 仿真：`tb/conv2d/gen_conv2d_im2col_data.py` 生成 IFM/weight、`A_im2col`、`W_col`、DRAM image 和 Conv2D golden，`scripts/run_conv2d_im2col_case.ps1` 复用 direct matmul testbench 做端到端校验。
33. T6.2 已完成 direct scalar on-the-fly im2col：`CTRL[8]` 启用后，`npu_dma` 从 raw NCHW IFM 按 Conv2D 窗口地址 gather A 行并写入 A PPBuf，不再要求 DRAM 保存完整 `A_im2col` 中间矩阵；`scripts/run_conv2d_otf_case.ps1` 已覆盖默认 INT8 OS/WS 和 FP16 OS case。
34. T6.3 已完成 direct scalar 32-bit bias：`BIAS_ADDR(0x98)` 指向每输出列一个 32-bit bias word，`CTRL[9]` 启用后 controller 对当前输出列 fetch bias，作为 scalar PE accumulator init。
35. T6.4 已完成 direct scalar ReLU/ReLU6：`CTRL[11:10]` 为 `00=none, 01=ReLU, 10=ReLU6`，`npu_top` 在 scalar result FIFO 前执行 activation。
36. T6.5 已完成 direct scalar INT8 quant/saturate：`QUANT_CFG(0x9C)` bit0 启用，bit1 选择 signed rounding，`[15:8]` 为 arithmetic right shift，`[31:16]` 为 signed scale；语义顺序为 accumulator -> optional bias -> activation -> optional quantize/saturate，量化输出为 sign-extended signed int8 word。
37. T6.6 已完成两层 Conv2D 端到端仿真：layer0 使用 direct scalar on-the-fly im2col + bias + ReLU + INT8 quant，layer1 直接消费 layer0 `R_ADDR`，最终 golden 正确。
38. SoC smoke 已恢复：`soc_top`、`dram_model`、`axi_lite_bridge`、`soc_mem` 与 PicoRV32 ready/rdata/PCPI/AXI burst 接口已对齐，`run_soc_sim.ps1` 默认无 VCD、`-DumpVcd` 可选。
39. T7.1 已完成 8x8/16x16 向量供数：`npu_top` 按 `CFG_SHAPE` 选择 4/8/16 lane，A 侧 row-skew 延迟链扩到 16 lane，controller 的 tile DMA byte/k、K-split 容量和 OS drain 周期按 shape lane 数计算。
40. T7.2 已完成阵列级 8x32 折叠路由：top half 输出逻辑列 0..15，bottom half 输出逻辑列 16..31；activation 横向折叠、OS weight row8 重新注入、WS 8-row load wrap 和 32-lane 输出顺序均由 `tb_reconfig_pe_8x32.v` 验证。
41. T7.3 已完成 PE 级 INT8 2-lane SIMD：`pe_top` 可对 packed `{lane1,lane0}` 执行两路 signed INT8 MAC，OS/WS、负数 lane 和旧 sign-extended scalar 兼容均由 `tb_pe_top.v` 验证。
42. T7.4 已完成 PE 级 INT8 4-lane SIMD：`pe_top` 在 `DATA_W=32/INT8_SIMD_LANES=4` 下可对 packed `{lane3,lane2,lane1,lane0}` 执行四路 signed INT8 MAC，OS/WS、负数 lane 和全宽 sign-extended scalar 兼容均由 `tb_pe_top.v` 验证。
43. T7.5 已完成 `op_counter` TOPS/util 报告：AXI-Lite `0xA0..0xC8` 可读 useful MAC/ops、busy/compute/DMA cycles、`TOPS_X1E6`、compute/e2e utilization 和 peak ops/cycle；`tb_op_counter_perf.v` 与 `tb_npu_scalar_smoke.v` 均输出可引用 `[PERF]` 行。

## 当前关键差距

1. 当前 4x4 tile-mode GEMM 已通过 INT8/FP16 基础验证，P2.3.1 已闭合 4x4 single-lane WS row-vector edge tile，8x8/16x16 已有宽向量供数验证，8x32 阵列级折叠路由和 PE 级 INT8 2/4-lane SIMD 已验证；但阵列/top 仍默认 `DATA_W=16`，32-bit packed K lane 供数、8x8/16x16 完整结果收集/写回、8x8/16x16/8x32 WS tile edge 以及顶层 8x32 32-output 写回仍未完成。
2. 顶层 K-split GEMM golden 已通过；外部 PSUM surface 的 DMA read/writeback 尚未接入多层 descriptor 流。
3. Descriptor v1 fetch/decode/next-layer 已具备第一版：支持 `OP=GEMM_TILEPACK`、`DTYPE=INT8/FP16`、`DATAFLOW=OS`、`SHAPE=4x4`、`TILE_PACKED=1` 映射到现有 tile GEMM 数据通路；T5.4 已验证 INT8 层间 OFM 作为下一层 IFM；T5.5 已让 unsupported descriptor 和 descriptor count exhausted 等错误对 CPU 可见。外部 PSUM surface、FP16 OFM 格式转换、bias/activation/quant 进入 descriptor/tile 主线仍未接入。
4. DMA 读写 burst、混合 8/16 beat 正确性、基础带宽计数、60%/80% 利用率目标报告和 T7.5 TOPS/util 报告已具备；后续若要冲 80% write util，需要允许多 outstanding write burst 或重叠 B response 间隔。
5. `npu_power` 输出没有接入 PE 主时钟路径。
6. T6.2-T6.6 已有 direct scalar on-the-fly im2col、bias、ReLU/ReLU6、INT8 quant/saturate 和两层 Conv2D E2E，但 tile/descriptor 主线尚未使用这些路径；descriptor 侧 `OP=CONV2D_IM2COL` 和多层卷积后处理仍未接入。

## 不应继续引用的旧结论

| 旧说法 | 当前判断 |
|---|---|
| `tb_comprehensive` 失败 2/28 PASS | 已过期，当前是 28/28 PASS |
| 全量回归 903 PASS | 当前不能作为事实引用 |
| SoC 集成已验证 | 旧的未带独立结果检查结论已过期；当前 smoke 已重新修复并通过 |
| 4x4 tile 结果还没有从阵列写回 | 已过期，T2.4 已完成 serializer 和 row-wise writeback |
| 16x16/8x32 已完成高吞吐矩阵乘 | 阵列存在，8x8/16x16 active lane 已能喂数，8x32 阵列级折叠路由已验证，但完整结果收集/写回和顶层 8x32 32-output 写回仍未完成 |
| DMA 读通道固定 `ARLEN=0` | 已过期，T3.1 已支持 INCR read burst |
| DMA 写通道固定单 burst | 已过期，T3.2 已支持多 INCR write burst 和 4KB 边界切分 |
| DFS/时钟门控已实际降功耗 | 行为模块输出悬空 |
| 支持 INT16 | 当前 PE 只有 INT8/FP16；INT8 PE 内已有 2/4-lane SIMD，但不表示支持 INT16 |

## 当前一句话定位

这是一个具备 PE 算术、PE 级 INT8 2/4-lane SIMD、NPU 外围框架、可验证 4x4 tile GEMM 路径、8x8/16x16 active lane 供数、8x32 阵列级折叠路由、顶层 K-split GEMM golden、DMA read/write burst、混合 burst 正确性测试、AXI perf counters、带宽利用率报告、TOPS/util 报告、PSUM/OUT buffer RMW 模块、accumulator-init PE array、controller k_tile loop、descriptor v1 ABI、AXI-Lite descriptor 提交寄存器、descriptor 多任务顺序执行、INT8 OFM->IFM 两层串联、IRQ/error status、DRAM 预展开 Conv2D im2col、direct scalar on-the-fly Conv2D im2col、bias、ReLU/ReLU6 和 INT8 quant/saturate 后处理的原型。下一阶段重点是 descriptor 化卷积、上游 packed 供数配套，以及更大 tile 完整写回验证。
