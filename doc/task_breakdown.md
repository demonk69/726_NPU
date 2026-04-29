# 任务分解

更新时间：2026-04-28

本文是后续协作的主任务清单。建议按任务编号逐项解决，每完成一项就更新本文状态和验证结果。

状态定义：

- `TODO`：未开始。
- `DOING`：正在做。
- `BLOCKED`：被前置问题阻塞。
- `DONE`：已实现并通过对应验证。

## 阶段 0：文档和基线统一

| ID | 状态 | 任务 | 验收标准 |
|---|---|---|---|
| T0.1 | DONE | 更新 README 和核心文档，移除旧 PASS 口径 | 文档不再声称顶层/SoC 当前已通过 |
| T0.2 | DONE | 固化当前仿真命令和源列表 | `simulation_guide.md` 中命令可复现 |
| T0.3 | DONE | 建立一个最小 smoke test 列表 | PE、DMA、顶层 1-output 各有入口 |

## 阶段 1：恢复当前顶层最小正确性

目标：先让现有标量路径跑通一个非零 INT8 点积，不急着做 4x4 并行。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T1.1 | DONE | 修复 testbench 和 SoC 中 `npu_top` 参数名 | `tb/*.v`, `rtl/soc/soc_top.v` | 不再出现 `.ROWS/.COLS` 参数错误 |
| T1.2 | DONE | 统一仿真脚本源列表，加入 `reconfig_pe_array.v` | `scripts/*.ps1` | 脚本可编译当前 RTL |
| T1.3 | DONE | 建立 1-output INT8 点积 testbench | `tb/tb_npu_scalar_smoke.v` | 一个非零 dot product 写回正确 |
| T1.4 | DONE | 定位并修复当前 `tb_comprehensive` 非零结果为 0 的根因 | `npu_top`, `npu_ctrl`, `npu_dma` | `tb_comprehensive.v` 28/28 通过 |
| T1.5 | DONE | 修复 `cfg_shape` 运行期 live 接入问题 | `npu_ctrl`, `npu_top` | start 后修改 `CFG_SHAPE` 不影响当前任务 |

Phase 1 验证记录：

```text
scripts/run_sim.ps1      -> PASS=19 FAIL=0
tb_npu_scalar_smoke.v    -> PASS
tb_comprehensive.v       -> ALL 28 TESTS PASSED
scripts/run_full_sim.ps1 -> compile and simulation completed
scripts/run_soc_sim.ps1  -> still blocked by dram_model axi_arlen and PicoRV32 PCPI issues
```

## 阶段 2：实现真实 4x4 GEMM tile

目标：满足基础评分项“4x4 脉动阵列”，不是只实例化 4x4。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T2.1 | DONE | 定义 4x4 tile 数据布局 | `doc/architecture.md` | A/W/C tile 地址和 lane 顺序明确 |
| T2.2 | DONE | A/W buffer 从标量输出升级为 4-lane 输出 | `pingpong_buf`, `npu_top` | 每个逻辑 `k` 周期可输出一组 A/W 4-lane vector |
| T2.3 | DONE | `npu_ctrl` 从单 `C[i,j]` 改为 4x4 tile 计数 | `npu_ctrl` | 支持 M/N 非 4 整数倍，边界 mask 正确 |
| T2.4 | DONE | 阵列输出 serializer 收集 16 个结果 | `npu_top` | 一个 4x4 tile 写回 16 个 32-bit word |
| T2.5 | DONE | 建立 4x4 INT8 GEMM 测试 | `tb/` | 与 Python golden 一致 |
| T2.6 | DONE | 建立 4x4 FP16 GEMM 测试 | `tb/` | 与 FP32 golden 容差一致 |

T2.1 结论：

```text
# M/N/K: GEMM 行、列、归约维度；卷积中 M=batch*OH*OW, N=Cout, K=Cin*KH*KW。
# m0/n0: 当前 4x4 输出 tile 左上角坐标。
# r/c: tile 内部 row/col lane。
# k: 当前归约维度坐标。
OS: PE row -> M lane, PE col -> N lane, PE(r,c) -> C[m0+r,n0+c]
A_TILE[m_tile][k][r] = A[m0+r,k]
W_TILE[n_tile][k][c] = W[k,n0+c]
OS physical cycle t:
  w_in[c]   = W_TILE[t][c]
  act_in[r] = A_TILE[t-r][r]  // row skew, out of range -> 0
  // 前 r 个物理周期 row r 输入为 0；不是第 0 拍 4 个 A row 全部有效
C serializer order:
  result_index = r*4 + c
  C_ADDR = R_ADDR + ((m0+r) * N + (n0+c)) * 4
```

WS 在当前 4x4 物理映射下先定义为 1x4 row-vector 子流程，完整 4x4 tile 由 4 个 M row pass 组成；`K>4` 的 WS 累加归入 Phase 4 的 `PSUM/OUT_BUF`。

T2.2 实现记录：

```text
pingpong_buf:
  新增 rd_vec_en / rd_vec / rd_vec_valid
  INT8:  一个 32-bit word -> 4 个 sign-extended DATA_W lane
  FP16:  两个 32-bit word -> 4 个 16-bit lane
  rd_vec_en 一拍消费 4 个 lane

npu_top:
  W/A PPBuf 接出 4-lane vector
  reconfig_pe_array 左上 4 行/列接入 a_vec/w_vec
  Phase-1 scalar writeback 仍使用 u_scalar_pe，保证旧回归稳定
```

T2.2 验证记录：

```text
tb_pingpong_buf_vec.v    -> PASS
tb_npu_scalar_smoke.v    -> PASS
scripts/run_sim.ps1      -> PASS=19 FAIL=0
scripts/run_full_sim.ps1 -> compile and simulation completed
tb_comprehensive.v       -> ALL 28 TESTS PASSED
```

T2.3 实现记录：

```text
ARR_CFG[7] = 1 enables 4x4 tile planner mode
npu_ctrl:
  tile_i/tile_j 在 tile mode 下表示 m_tile/n_tile
  输出 tile_m_base/tile_n_base
  输出 tile_row_valid/tile_col_valid 和 active_rows/active_cols
  输出 vec_consume 和 tile_k_cycle
npu_top:
  vec_consume 驱动 rd_vec_en
  A lane 1/2/3 分别延迟 1/2/3 拍，形成 OS row-skew feeder
  默认 ARR_CFG=0 时仍保持 Phase-1 标量兼容路径
```

T2.3 验证记录：

```text
tb_npu_ctrl_tile.v       -> PASS
tb_pingpong_buf_vec.v    -> PASS
tb_npu_scalar_smoke.v    -> PASS
scripts/run_sim.ps1      -> PASS=19 FAIL=0
scripts/run_full_sim.ps1 -> compile and simulation completed
tb_comprehensive.v       -> ALL 28 TESTS PASSED
```

T2.4 实现记录：

```text
reconfig_pe_array:
  4x4 模式下 acc_out/valid_out 按 result_index=r*4+c 暴露 16 个 PE 输出

npu_top:
  tile mode 使用 pe_array_result/valid 捕获 16 个结果
  serializer 按 active_rows/active_cols 输出 row-major word 到 result FIFO
  非 tile mode 继续使用 u_scalar_pe 兼容路径

npu_ctrl:
  tile mode 首次 warm-up DMA load 使用 K*4*data_bytes
  结果写回按 row 发短 burst：
    addr = R_ADDR + ((m0+r) * N + n0) * 4
    len  = active_cols * 4 bytes
```

T2.4 验证记录：

```text
tb_npu_tile_writeback.v  -> PASS，4x4 K=1 tile 写回 16 个 word，4 个 row burst
tb_npu_ctrl_tile.v       -> PASS，5x6 边界 tile 产生 10 个 row writeback burst
tb_npu_scalar_smoke.v    -> PASS
tb_pingpong_buf_vec.v    -> PASS
tb_comprehensive.v       -> ALL 28 TESTS PASSED
tb_multi_rc_comprehensive.v -> ALL 13 CHECKS PASSED
```

T2.5/T2.6 实现记录：

```text
tb/tile4/gen_tile4_data.py:
  生成 tile-mode 专用 A/W memory layout：
    W_TILE[k][c] -> w_vec[c]
    A_TILE[k][r] -> a_vec[r]
  INT8 golden 使用 signed INT8 dot product，输出 32-bit two's complement
  FP16 golden 使用 FP16 product + FP32 accumulation，testbench 用 ULP 容差比较

tb/tb_npu_tile_gemm.v:
  参数化读取 tb/tile4/<case>/test_params.vh
  启用 ARR_CFG[7] 和 CFG_SHAPE=4x4
  检查 4x4 C tile 的 16 个 32-bit 结果
```

T2.5/T2.6 验证记录：

```text
tb_npu_tile_gemm.v + tb/tile4/int8_4x4x4 -> PASS，ALL 16 CHECKS PASSED
tb_npu_tile_gemm.v + tb/tile4/fp16_4x4x4 -> PASS，ALL 16 CHECKS PASSED
```

## 阶段 3：AXI Burst DMA 和带宽统计

目标：满足 AXI Burst 地址递增和 DMA 控制器要求，并为带宽利用率评分提供数据。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T3.1 | TODO | DMA 读通道支持 INCR burst | `npu_dma.v` | `ARLEN > 0`，连续地址递增 |
| T3.2 | TODO | DMA 写通道支持多 burst 和 4KB 边界切分 | `npu_dma.v` | 不跨 4KB，结果连续写回 |
| T3.3 | TODO | 增加 AXI beat/cycle 计数 | `axi_monitor`, `npu_top` | 输出带宽利用率 |
| T3.4 | TODO | 建立 burst 正确性测试 | `tb/` | 8/16 beat burst 数据无误 |
| T3.5 | TODO | 带宽利用率目标测试 | `tb/` | burst 场景达到或解释 60%/80% 差距 |

## 阶段 4：PSUM/OUT Buffer 和 K-split

目标：解决大 K、层间暂存、多层卷积的核心缺口。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T4.1 | TODO | 增加 `PSUM/OUT_BUF` 规格 | `doc/architecture.md` | 数据宽度、深度、地址模式明确 |
| T4.2 | TODO | 实现 psum read/modify/write 或片上 psum SRAM | `rtl/buf/` | K-split 能累加正确 |
| T4.3 | TODO | PE 支持 accumulator init | `pe_top`, `reconfig_pe_array` | 从 psum 初始化后继续 MAC |
| T4.4 | TODO | `npu_ctrl` 增加 k_tile loop | `npu_ctrl` | K 大于 buffer 深度时正确 |
| T4.5 | TODO | K-split GEMM 测试 | `tb/` | 多个 K tile 结果等于未切分 golden |

## 阶段 5：Descriptor 和多层调度

目标：让 NPU 知道每层何时结束、下一层何时开始、整个网络何时完成。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T5.1 | TODO | 定义 descriptor 二进制格式 | `doc/user_manual.md` | CPU 和 NPU 字段一致 |
| T5.2 | TODO | AXI-Lite 增加 `DESC_BASE/DESC_COUNT` | `npu_axi_lite.v` | CPU 可提交 descriptor list |
| T5.3 | TODO | 控制器增加 `FETCH_DESC/DECODE_DESC/NEXT_LAYER` | `npu_ctrl.v` | 多 descriptor 顺序执行 |
| T5.4 | TODO | 支持层间 OFM 作为下一层 IFM | `npu_ctrl`, `npu_dma` | 两层 GEMM/FC 串联通过 |
| T5.5 | TODO | 增加 IRQ 和错误状态 | `npu_axi_lite`, `npu_ctrl` | done/error 可由 CPU 轮询或中断获取 |

## 阶段 6：卷积前端和后处理

目标：支持多层卷积 + 激活，而不是只做预展开 GEMM。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T6.1 | TODO | 第一版使用 DRAM 预展开 im2col | `tb/`, `scripts/` | conv golden 通过 |
| T6.2 | TODO | 增加 on-the-fly im2col 地址发生器 | `rtl/ctrl/`, `rtl/axi/` | 不需要完整 im2col 中间矩阵 |
| T6.3 | TODO | 增加 bias 加法 | `rtl/top/` | bias 后结果正确 |
| T6.4 | TODO | 增加 ReLU/ReLU6 | `rtl/top/` | 激活输出正确 |
| T6.5 | TODO | INT8 quant/saturate | `rtl/top/` | 输出范围和缩放正确 |
| T6.6 | TODO | 两层卷积端到端测试 | `tb/` | layer0 输出被 layer1 使用，最终结果正确 |

## 阶段 7：16x16、8x32 和性能优化

目标：从基础分提升到优化分。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T7.1 | TODO | 8x8/16x16 向量供数 | `npu_top`, `buf` | 阵列 active lane 全部有数据 |
| T7.2 | TODO | 8x32 折叠路由修正和验证 | `reconfig_pe_array` | 逻辑 8x32 输出顺序正确 |
| T7.3 | TODO | INT8 2-lane SIMD PE | `pe_top` | 16x16 @500MHz 理论 0.512 TOPS |
| T7.4 | TODO | INT8 4-lane SIMD PE | `pe_top` | 16x16 @500MHz 理论 1.024 TOPS |
| T7.5 | TODO | 性能计数器输出 TOPS 和利用率 | `op_counter`, `npu_top` | 仿真报告可直接引用 |

## 阶段 8：低功耗和 FPGA 验证

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T8.1 | TODO | 将 clock gating 从悬空输出接入设计 | `npu_power`, `npu_top` | idle PE 不翻转或 clock enable 关闭 |
| T8.2 | TODO | DFS 寄存器行为验证 | `tb/` | div1/2/4/8 下状态机正确 |
| T8.3 | TODO | FPGA 约束和综合脚本 | `constraints/`, `scripts/` | 能完成综合 |
| T8.4 | TODO | FPGA 板级 smoke test | `fpga/` 或说明文档 | CPU 配置 NPU 并读回结果 |

## 推荐执行顺序

当前 T2.1-T2.6 已完成。下一步进入 Phase 3 的 AXI burst DMA 和带宽统计，不要先做 descriptor 或 16x16 性能优化。建议顺序：

```text
T3.1 -> T3.2 -> T3.3
```

当 burst DMA 正确性和统计口径通过后，再进入 PSUM/OUT Buffer、K-split 和 descriptor。
