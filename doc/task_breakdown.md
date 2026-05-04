# 任务分解

更新时间：2026-05-03

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
scripts/run_soc_sim.ps1  -> PASS，CPU configures NPU, NPU writes 2x2 INT8 GEMM result, CPU/testbench verify
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
| T3.1 | DONE | DMA 读通道支持 INCR burst | `npu_dma.v` | `ARLEN > 0`，连续地址递增 |
| T3.2 | DONE | DMA 写通道支持多 burst 和 4KB 边界切分 | `npu_dma.v` | 不跨 4KB，结果连续写回 |
| T3.3 | DONE | 增加 AXI beat/cycle 计数 | `axi_monitor`, `npu_top` | 输出带宽利用率 |
| T3.4 | DONE | 建立 burst 正确性测试 | `tb/` | 8/16 beat burst 数据无误 |
| T3.5 | DONE | 带宽利用率目标测试 | `tb/` | burst 场景达到或解释 60%/80% 差距 |

T3.1 实现记录：

```text
npu_dma:
  读侧 Load-FSM 每次只发一个 outstanding AR burst
  m_axi_arlen 根据剩余 byte、BURST_MAX 和 4KB 边界动态生成
  m_axi_arburst = INCR，m_axi_arsize = 32-bit beat
  W/A 读保持顺序执行，兼容当前 npu_ctrl 的 W then A 合约
  RREADY 在目标 PPBuf full 时做背压

tb/tb_dma_read_burst.v:
  case1: W 80 bytes -> 16-beat + 4-beat burst，A 32 bytes -> 8-beat burst
  case2: W 从 0x0ff0 读 32 bytes -> 4KB 边界切成两个 4-beat burst
```

T3.1 验证记录：

```text
tb_dma_read_burst.v       -> PASS，INCR read bursts and 4KB split passed
tb_npu_scalar_smoke.v     -> PASS
tb_npu_tile_writeback.v   -> PASS
tb_npu_tile_gemm.v INT8   -> PASS，ALL 16 CHECKS PASSED
tb_npu_tile_gemm.v FP16   -> PASS，ALL 16 CHECKS PASSED
tb_comprehensive.v        -> ALL 28 TESTS PASSED
tb_multi_rc_comprehensive.v -> ALL 13 CHECKS PASSED
```

T3.2 实现记录：

```text
npu_dma:
  写侧 WB-FSM 每次只发一个 outstanding AW/W burst
  m_axi_awlen 根据剩余 byte、BURST_MAX 和 4KB 边界动态生成
  m_axi_awburst = INCR，m_axi_awsize = 32-bit beat
  WLAST 按当前 burst 末拍产生，不再按整笔 r_len_bytes 产生
  每个 write burst 等待 B response 后继续下一段，r_done 在整笔写回完成后脉冲

tb/tb_dma_write_burst.v:
  case1: R 80 bytes -> 16-beat + 4-beat burst
  case2: R 从 0x0ff0 写 32 bytes -> 4KB 边界切成两个 4-beat burst
```

T3.2 验证记录：

```text
tb_dma_write_burst.v      -> PASS，INCR write bursts and 4KB split passed
tb_dma_read_burst.v       -> PASS，INCR read bursts and 4KB split passed
tb_npu_tile_writeback.v   -> PASS
tb_npu_tile_gemm.v INT8   -> PASS，ALL 16 CHECKS PASSED
tb_npu_tile_gemm.v FP16   -> PASS，ALL 16 CHECKS PASSED
scripts/run_regression.ps1 -> 926 PASS / 12 FAIL；剩余失败集中在旧 matmul 2x3x2 OS/WS case
```

T3.3 实现记录：

```text
axi_monitor:
  统计 AXI master read/write burst 数、data beat 数、byte 数和 total_cycles
  输出 read/write bytes-per-cycle x1000
  输出 read/write data-channel utilization，单位为 basis points

npu_top + npu_axi_lite:
  在 npu_top 内实例化 axi_monitor，监控 AXI-Lite 和 DMA AXI4 master 通道
  通过 AXI-Lite 只读寄存器暴露性能计数：
    0x48 PERF_CYCLES
    0x4C PERF_RD_BEATS
    0x50 PERF_WR_BEATS
    0x54 PERF_RD_BYTES
    0x58 PERF_WR_BYTES
    0x5C PERF_RD_BW
    0x60 PERF_WR_BW
    0x64 PERF_RD_UTIL
    0x68 PERF_WR_UTIL
    0x6C PERF_RD_BURSTS
    0x70 PERF_WR_BURSTS
```

T3.3 验证记录：

```text
tb_npu_scalar_smoke.v     -> PASS，标量结果/cfg_shape/perf counters 均通过
tb_dma_read_burst.v       -> PASS
tb_dma_write_burst.v      -> PASS
tb_npu_tile_writeback.v   -> PASS
tb_npu_tile_gemm.v INT8   -> PASS，ALL 16 CHECKS PASSED
tb_npu_tile_gemm.v FP16   -> PASS，ALL 16 CHECKS PASSED
scripts/run_regression.ps1 -> 929 PASS / 12 FAIL；剩余失败集中在旧 matmul 2x3x2 OS/WS case
```

T3.4 实现记录：

```text
tb/tb_dma_burst.v:
  同一轮请求内同时启动 W/A read DMA 和 result writeback DMA
  W read 覆盖 16-beat INCR burst，A read 覆盖 8-beat INCR burst
  R write 覆盖 24 beats -> 16-beat + 8-beat INCR burst
  检查 AR/AW addr、len、size、burst，检查 W/A PPBuf 写入数据和 R 写回数据顺序
  验证 Load-FSM 和 WB-FSM 可在 AXI read/write 独立通道上并行推进
```

T3.4 验证记录：

```text
tb_dma_burst.v            -> PASS，mixed 8/16-beat read/write burst data passed
scripts/run_regression.ps1 -> 929 PASS / 12 FAIL；剩余失败集中在旧 matmul 2x3x2 OS/WS case
```

T3.5 实现记录：

```text
tb/tb_dma_perf.v:
  使用 256-beat long burst 场景测试 DMA read/write data-channel utilization
  read: 16 个 16-beat INCR burst，窗口从首个 R beat 到最后 R beat
  write: 16 个 16-beat INCR burst，窗口从首个 W beat 到最后 W beat
  read 目标为 >=80%；write 硬门槛为 >=60%，若低于 80% 输出差距说明
  当前 write 低于 80% 的原因是 DMA 每个 write burst 等待 B response 后才发下一段 AW
```

T3.5 验证记录：

```text
tb_dma_perf.v:
  read  beats=256 cycles=301 bursts=16 util=85.04% bw=3.401 B/cyc
  write beats=256 cycles=331 bursts=16 util=77.34% bw=3.093 B/cyc
  PASS，write 低于 80% 的 single-outstanding/B-response gap 已解释
scripts/run_regression.ps1 -> 929 PASS / 12 FAIL；剩余失败集中在旧 matmul 2x3x2 OS/WS case
```

## 阶段 4：PSUM/OUT Buffer 和 K-split

目标：解决大 K、层间暂存、多层卷积的核心缺口。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T4.1 | DONE | 增加 `PSUM/OUT_BUF` 规格 | `doc/architecture.md` | 数据宽度、深度、地址模式明确 |
| T4.2 | DONE | 实现 psum read/modify/write 或片上 psum SRAM | `rtl/buf/` | K-split 能累加正确 |
| T4.3 | DONE | PE 支持 accumulator init | `pe_top`, `reconfig_pe_array` | 从 psum 初始化后继续 MAC |
| T4.4 | DONE | `npu_ctrl` 增加 k_tile loop | `npu_ctrl` | K 大于 buffer 深度时正确 |
| T4.5 | DONE | K-split GEMM 测试 | `tb/` | 多个 K tile 结果等于未切分 golden |

T4.1 实现记录：

```text
doc/architecture.md:
  定义 PSUM/OUT_BUF 为 4x4 tile 级 accumulator 存储，不是整层 C 的片上缓存
  数据宽度固定为 ACC_W=32，INT8 路径为 int32 accumulator，FP16 路径为 FP32 bit pattern
  片上最小深度为 2-bank * 16 words，tile-local index = r*4+c
  外部 PSUM/OFM surface 按 row-major M*N*4 byte 编址
  K-split 的 k_tile_count/k0/k_len、A/W tile slice 地址和 first/middle/last k_tile 调度语义已明确
```

T4.1 验证记录：

```text
文档规格任务，无 RTL 行为变化；不新增仿真。
```

T4.2 实现记录：

```text
rtl/buf/psum_out_buf.v:
  2-bank tile-local PSUM/OUT SRAM，默认每 bank 16 个 32-bit accumulator word
  port A 面向 DMA/load/drain，port B 面向 compute/serializer
  支持同步 read/write，用外部 read-modify-write 完成 K-split 累加
  valid_mask 过滤边界 tile：invalid lane 读 0、rvalid=0、写入被忽略
  tile_clear_en 可清空指定 bank
  同地址双写时 write_conflict 置位，port B 优先

tb/tb_psum_out_buf.v:
  覆盖 first k_tile load、second k_tile RMW accumulation、edge mask、bank isolation 和 write conflict
```

T4.2 验证记录：

```text
tb_psum_out_buf.v         -> PASS，K-split read-modify-write accumulation passed
scripts/run_regression.ps1 -> 930 PASS / 12 FAIL；新增 psum_out_buf 通过，剩余失败仍集中在旧 matmul 2x3x2 OS/WS case
```

T4.3 实现记录：

```text
rtl/pe/pe_top.v:
  增加 acc_init_en/acc_init
  acc_init_en 置位时清本地 pipeline valid，并按 stat_mode 初始化 os_acc 或 ws_acc
  后续 MAC 从该 accumulator 初值继续累加

rtl/array/reconfig_pe_array.v:
  增加 per-PE acc_init bus 和 acc_init_mask
  每个 active PE 可单独从 PSUM/OUT buffer 值初始化 accumulator

rtl/top/npu_top.v:
  暂时将 acc_init_en/acc_init/acc_init_mask tie-off 为 0
  T4.4 已由 controller 驱动 k_tile loop；外部 PSUM/OUT surface 接入留给后续 descriptor/多层调度

tb/tb_pe_top.v:
  增加 INT8 OS、FP16 OS、INT8 WS accumulator init 直接测试

tb/tb_reconfig_pe_acc_init.v:
  4x4 OS array 中 16 个 PE 使用不同 psum 初始化值，随后继续 MAC 并检查 row-major 输出
```

T4.3 验证记录：

```text
tb_pe_top.v              -> PASS=22 FAIL=0，新增 accumulator init case 全部通过
tb_reconfig_pe_acc_init.v -> PASS，4x4 PE array accumulator init continued MAC passed
scripts/run_regression.ps1 -> 931 PASS / 12 FAIL；新增 reconfig_pe_acc_init 通过，剩余失败仍集中在旧 matmul 2x3x2 OS/WS case
```

T4.4 实现记录：

```text
rtl/ctrl/npu_ctrl.v:
  增加 PPB_DEPTH 参数，按数据类型计算每个 k_tile 可容纳的 K 元素数
    INT8:  K_TILE_ELEMS = PPB_DEPTH
    FP16:  K_TILE_ELEMS = PPB_DEPTH / 2
  增加 k_tile_idx、tile_k_base、tile_k_len、tile_k_index 输出
  调度顺序变为 m_tile -> n_tile -> k_tile
  A/W DMA 地址按 (tile_idx*K + k0) * vector_elem_bytes 生成
  每个 k_tile 使用自己的 dma_len 和 compute_cycles
  中间 k_tile 不 flush、不写回，PE accumulator 保持部分和
  最后一个 k_tile 才 flush，并按 row-wise burst 写回最终 C tile

rtl/top/npu_top.v:
  传入 PPB_DEPTH，并接出 controller 的 k_tile debug outputs

tb/tb_npu_ctrl_ksplit.v:
  使用 PPB_DEPTH=4、K=10 强制切成 4/4/2 三个 k_tile
  检查三次 A/W load 地址和长度、10 次 vec_consume、最终才触发 4 个 row writeback
```

T4.4 验证记录：

```text
tb_npu_ctrl_ksplit.v     -> PASS，3 个 K-split load pair，final k_tile 后 4 个 row writeback
tb_npu_ctrl_tile.v       -> PASS，K 未超过 PPB 深度时旧 tile planner 行为保持不变
scripts/run_regression.ps1 -> 939 PASS / 12 FAIL；新增 npu_ctrl_ksplit 通过，剩余失败仍集中在旧 matmul 2x3x2 OS/WS case
```

T4.5 实现记录：

```text
rtl/ctrl/npu_ctrl.v:
  修复真实 DMA/PPBuf 时序下的 k_tile prefetch：
  后续 k_tile 的 prefetch 不再和 PPBuf swap 同周期发起，
  而是在 S_PRELOAD 延后一拍发起，避免 DMA 采样到刚 swap 前的 stale full。

tb/tb_npu_tile_ksplit_gemm.v:
  新增端到端 4x4x10 INT8 OS GEMM golden 测试
  npu_top 以 PPB_DEPTH=4/PPB_THRESH=4 实例化，强制 K=10 切成 4/4/2
  检查 6 个 W/A AXI read burst、final-only 4 个 row writeback
  检查最终 16 个 C[r,c] 等于未切分 GEMM golden

scripts/run_regression.ps1:
  新增 npu_tile_ksplit_gemm 回归项
```

T4.5 验证记录：

```text
tb_npu_tile_ksplit_gemm.v -> PASS，ALL 28 CHECKS PASSED
tb_npu_ctrl_ksplit.v      -> PASS，controller K-split 行为保持不变
tb_npu_tile_gemm.v INT8   -> PASS，旧 4x4x4 tile GEMM 行为保持不变
tb_npu_tile_gemm.v FP16   -> PASS，旧 FP16 tile GEMM 行为保持不变
scripts/run_regression.ps1 -> 967 PASS / 12 FAIL；新增 K-split GEMM 通过，剩余失败仍集中在旧 matmul 2x3x2 OS/WS case
```

## 阶段 5：Descriptor 和多层调度

目标：让 NPU 知道每层何时结束、下一层何时开始、整个网络何时完成。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T5.1 | DONE | 定义 descriptor 二进制格式 | `doc/user_manual.md` | CPU 和 NPU 字段一致 |
| T5.2 | DONE | AXI-Lite 增加 `DESC_BASE/DESC_COUNT` | `npu_axi_lite.v` | CPU 可提交 descriptor list |
| T5.3 | DONE | 控制器增加 `FETCH_DESC/DECODE_DESC/NEXT_LAYER` | `npu_ctrl.v` | 多 descriptor 顺序执行 |
| T5.4 | DONE | 支持层间 OFM 作为下一层 IFM | `npu_ctrl`, `npu_dma` | 两层 GEMM/FC 串联通过 |
| T5.5 | DONE | 增加 IRQ 和错误状态 | `npu_axi_lite`, `npu_ctrl` | done/error 可由 CPU 轮询或中断获取 |

T5.1 实现记录：

```text
doc/user_manual.md:
  固定 descriptor v1 ABI：16 个 32-bit little-endian word，总长 64 byte
  定义 desc_ctrl bit layout：VERSION/OP/DTYPE/DATAFLOW/SHAPE/TILE_PACKED/FIRST_K/LAST_K/LAST_LAYER/IRQ/BIAS/PSUM
  定义 word0..word15 的字段、位宽、对齐规则、保留位规则和链结束规则
  给出 CPU 侧 npu_desc_v1_t 结构体和 NPU RTL 解码 localparam
  明确 descriptor 到当前直配寄存器的映射：M/N/K、A/W/R 地址、CTRL mode/stat、ARR_CFG[7]、CFG_SHAPE

doc/architecture.md:
  将旧 descriptor 伪结构同步为 v1 16-word 摘要

doc/module_reference.md:
  记录 T5.3 应按 descriptor v1 fetch/decode，并先支持 GEMM_TILEPACK/4x4/OS
```

T5.1 验证记录：

```text
文档一致性检查 -> PASS，user_manual/architecture/module_reference/task_breakdown/current_status 中的 descriptor v1 口径一致
```

T5.2 实现记录：

```text
rtl/axi/npu_axi_lite.v:
  新增 DESC_BASE(0x40) 和 DESC_COUNT(0x44) 可读写寄存器
  CTRL bit7 作为 desc_mode 保持可读写，bit6 仍为 irq_clr W1C

rtl/top/npu_top.v:
  将 desc_base/desc_count 从 AXI-Lite 接出到顶层 wire，暂不驱动 controller

tb/tb_npu_axi_lite_desc.v:
  覆盖 DESC_BASE/DESC_COUNT reset/readback/write/readback
  覆盖 CTRL desc_mode bit readback
  T5.2 阶段 ERR_STATUS(0x74) 仅作为预留项，后续已在 T5.5 补齐

scripts/run_regression.ps1:
  加入 npu_axi_lite_desc 回归入口
```

T5.2 验证记录：

```text
tb_npu_axi_lite_desc -> PASS
tb_npu_scalar_smoke  -> PASS
scripts/run_regression.ps1 -> 968 PASS / 12 FAIL
12 个 FAIL 仍为旧 matmul 2x3x2 OS/WS case，未新增失败。
```

T5.3 实现记录：

```text
rtl/axi/npu_dma.v:
  新增 descriptor fetch 读目标：desc_start/desc_base_addr 触发一次 64-byte AXI read
  读取 16 个 32-bit word 到 desc_words，desc_done 单拍完成
  descriptor fetch 复用 read-side burst engine，不写入 W/A PPBuf

rtl/ctrl/npu_ctrl.v:
  CTRL[7] desc_mode=1 且 start 上升沿时进入 S_FETCH_DESC
  S_DECODE_DESC 按 descriptor v1 word layout 映射到当前 shadow config：
    M/N/K、ifm_addr->A、weight_addr->W、ofm_addr->R
    desc_ctrl dtype/dataflow/shape/tile_packed -> mode/stat/cfg_shape/ARR_CFG[7]
  单 descriptor 执行完成后根据 LAST_LAYER 或 next_desc 判断 network done
  next_desc 非 0 且 DESC_COUNT 仍有余量时继续 fetch 下一 descriptor

rtl/top/npu_top.v:
  连接 npu_ctrl 和 npu_dma 的 descriptor fetch 接口

tb/tb_npu_desc_two_layer.v:
  新增两 descriptor 顶层测试：
    desc0 -> 4x4x4 INT8 OS tile GEMM 写 R0，next_desc 指向 desc1
    desc1 -> 4x4x4 INT8 OS tile GEMM 写 R1，LAST_LAYER=1
  检查 2 次 descriptor fetch、4 次 W/A tile read、8 次 row-wise writeback 和 32 个 C 结果

scripts/run_regression.ps1:
  加入 npu_desc_two_layer 回归入口
```

T5.3 验证记录：

```text
tb_npu_desc_two_layer -> PASS，ALL 48 CHECKS PASSED
scripts/run_regression.ps1 -> 1016 PASS / 12 FAIL
12 个 FAIL 仍为旧 matmul 2x3x2 OS/WS case，未新增失败。
```

T5.4 实现记录：

```text
rtl/ctrl/npu_ctrl.v:
  增加 desc_ctrl[23] IFM_FROM_PREV_OFM 解码
  descriptor 链中每层完成后记录上一层 ofm_addr
  下一层置 IFM_FROM_PREV_OFM 时，用上一层 ofm_addr 覆盖 A 源地址
  对 A DMA 请求附带 row-major OFM gather 元数据：
    stride=当前 K，m_base/k_base/k_len/active_rows/fp16_mode

rtl/axi/npu_dma.v:
  A 通道新增 A_OFM 读目标
  从上一层 32-bit row-major OFM surface 读取 A[m0+r,k]
  INT8 路径取每个 OFM word 的低 8 bit，打包为 4-lane A tile word 后写入 A PPBuf
  保留 FP16 低 16 bit pack 结构，但当前未作为已验证网络格式声明

rtl/top/npu_top.v:
  连接 controller 和 DMA 的 dma_a_ofm_* 元数据接口

tb/tb_npu_desc_ofm_chain.v:
  新增两 descriptor 链式 GEMM：
    desc0 -> 4x4x4 INT8 OS GEMM，W0 为 identity，写 R0
    desc1 -> desc_ctrl[23]=IFM_FROM_PREV_OFM，使用 R0 作为 A，乘 W1，写 R1
  检查 descriptor/W/A/OFM-gather read、row-wise writeback、layer0/layer1 golden

scripts/run_regression.ps1:
  加入 npu_desc_ofm_chain 回归入口
```

T5.4 验证记录：

```text
tb_npu_desc_ofm_chain -> PASS，ALL 55 CHECKS PASSED
tb_npu_desc_two_layer -> PASS，ALL 48 CHECKS PASSED
tb_dma_read_burst/tb_dma_write_burst/tb_dma_burst/tb_dma_perf -> PASS
scripts/run_regression.ps1 -> 1071 PASS / 12 FAIL
12 个 FAIL 仍为旧 matmul 2x3x2 OS/WS case，未新增失败。
```

T5.5 实现记录：

```text
rtl/ctrl/npu_ctrl.v:
  增加 error/err_status 输出和 err_clear/err_clear_mask W1C 清除输入
  对 desc_count=0、unsupported descriptor、descriptor count exhausted 和首层 IFM_FROM_PREV_OFM 错误置位 ERR_STATUS
  descriptor done IRQ 由 desc_ctrl[20] IRQ_EN 控制，direct mode done 仍产生 irq_flag

rtl/axi/npu_axi_lite.v:
  STATUS(0x04) bit2 映射 controller error
  INT_EN(0x08) bit0/bit1 分别使能 done/error IRQ
  INT_CLR(0x0C) 和 CTRL bit6 都可清 npu_irq pending
  ERR_STATUS(0x74) 读 controller err_status，写 1 产生 W1C clear mask

rtl/top/npu_top.v:
  连接 status_error、err_status、err_clear 和 err_clear_mask

tb/tb_npu_axi_lite_desc.v:
  覆盖 STATUS busy/done/error、done/error IRQ、INT_CLR/CTRL bit6 清 pending 和 ERR_STATUS W1C 请求

tb/tb_npu_ctrl_error_status.v:
  新增 controller 错误状态测试，覆盖 desc_count=0、unsupported descriptor 和 descriptor count exhausted

scripts/run_regression.ps1:
  加入 npu_ctrl_error_status 回归入口
```

T5.5 验证记录：

```text
tb_npu_axi_lite_desc -> PASS
tb_npu_ctrl_error_status -> PASS，3 类 descriptor 错误均可由 ERR_STATUS 观察并 W1C 清除
tb_npu_ctrl_tile/tb_npu_ctrl_ksplit -> PASS，旧 controller tile/K-split 行为保持不变
tb_npu_desc_two_layer -> PASS，ALL 48 CHECKS PASSED
tb_npu_desc_ofm_chain -> PASS，ALL 55 CHECKS PASSED
tb_npu_scalar_smoke -> PASS
scripts/run_regression.ps1 -> 1075 PASS / 12 FAIL
12 个 FAIL 仍为旧 matmul 2x3x2 OS/WS case，未新增失败。
```

## 阶段 6：卷积前端和后处理

目标：支持多层卷积 + 激活，而不是只做预展开 GEMM。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T6.1 | DONE | 第一版使用 DRAM 预展开 im2col | `tb/`, `scripts/` | conv golden 通过 |
| T6.2 | DONE | 增加 on-the-fly im2col 地址发生器 | `rtl/ctrl/`, `rtl/axi/` | 不需要完整 im2col 中间矩阵 |
| T6.3 | DONE | 增加 bias 加法 | `rtl/top/`, `rtl/ctrl/`, `rtl/axi/`, `tb/`, `scripts/` | bias 后结果正确 |
| T6.4 | DONE | 增加 ReLU/ReLU6 | `rtl/top/`, `rtl/ctrl/`, `tb/`, `scripts/` | 激活输出正确 |
| T6.5 | DONE | INT8 quant/saturate | `rtl/top/`, `rtl/ctrl/`, `rtl/axi/`, `tb/`, `scripts/` | 输出范围和缩放正确 |
| T6.6 | DONE | 两层卷积端到端测试 | `tb/`, `scripts/` | layer0 输出被 layer1 使用，最终结果正确 |

## 阶段 7：16x16、8x32 和性能优化

目标：从基础分提升到优化分。

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T7.1 | DONE | 8x8/16x16 向量供数 | `npu_top`, `buf` | 阵列 active lane 全部有数据 |
| T7.2 | DONE | 8x32 折叠路由修正和验证 | `reconfig_pe_array` | 逻辑 8x32 输出顺序正确 |
| T7.3 | DONE | INT8 2-lane SIMD PE | `pe_top` | 16x16 @500MHz 理论 0.512 TOPS |
| T7.4 | DONE | INT8 4-lane SIMD PE | `pe_top` | 16x16 @500MHz 理论 1.024 TOPS |
| T7.5 | DONE | 性能计数器输出 TOPS 和利用率 | `op_counter`, `npu_top` | 仿真报告可直接引用 |

## 阶段 8：低功耗和 FPGA 验证

| ID | 状态 | 任务 | 主要文件 | 验收标准 |
|---|---|---|---|---|
| T8.1 | TODO | 将 clock gating 从悬空输出接入设计 | `npu_power`, `npu_top` | idle PE 不翻转或 clock enable 关闭 |
| T8.2 | TODO | DFS 寄存器行为验证 | `tb/` | div1/2/4/8 下状态机正确 |
| T8.3 | TODO | FPGA 约束和综合脚本 | `constraints/`, `scripts/` | 能完成综合 |
| T8.4 | TODO | FPGA 板级 smoke test | `fpga/` 或说明文档 | CPU 配置 NPU 并读回结果 |

## 推荐执行顺序

T6.1 实现记录：

- 新增 `tb/conv2d/gen_conv2d_im2col_data.py`，生成 dense Conv2D 的 IFM/weight、`A_im2col[M,K]`、`W_col[K,N]`、DRAM image、`expected.hex` 和 `test_params.vh`。
- 新增 `scripts/run_conv2d_im2col_case.ps1`，可按参数生成并运行单个 Conv2D im2col case，复用 direct matmul testbench 对 Conv2D golden。
- 修正 direct scalar 路径的 32-bit 对齐 stride：K 不是 4-byte 对齐时，下一行 A 和下一列 W 的预取地址按 word-aligned DMA row/column stride 推进；旧 `2x3x2` 非方阵 matmul case 已恢复通过。
- `scripts/run_regression.ps1` 已接入 T6.1 默认 case：INT8 OS、INT8 WS、FP16 OS。

T6.1 验证记录：

```text
scripts/run_conv2d_im2col_case.ps1 -Mode OS -Name conv2d_im2col_int8_os_default  -> ALL 75 CHECKS PASSED
scripts/run_conv2d_im2col_case.ps1 -Mode WS -Name conv2d_im2col_int8_ws_default  -> ALL 75 CHECKS PASSED
scripts/run_conv2d_im2col_case.ps1 -Dtype fp16 -Mode OS -Name conv2d_im2col_fp16_os_default -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1 -> TOTAL: 1315 PASS, 0 FAIL
```

T6.2 实现记录：

- 在 `npu_axi_lite` 增加 direct Conv2D on-the-fly im2col 配置寄存器：`0x80..0x94` 分别描述 IFM shape、batch/Cin、kernel、OFM shape、stride/pad、dilation；`CTRL[8]` 作为 direct scalar on-the-fly im2col enable。
- `npu_ctrl` 在 direct scalar、非 tile mode 下锁存 Conv2D 参数，并在每个输出行 `m` 的 A load 中向 DMA 传递 `m_index`、`k_len` 和卷积 shape/stride/pad/dilation；descriptor/tile 主线暂不启用该路径。
- `npu_dma` 新增 `A_IM2COL` 读目标：DRAM 中只保存原始 NCHW IFM，DMA 按 `m -> b/oh/ow`、`k -> cin/kh/kw` 计算 IFM 地址，padding 或越界位置写 0，并按 INT8/FP16 32-bit word 打包写入 A PPBuf。
- `tb/conv2d/gen_conv2d_im2col_data.py --on-the-fly` 生成 raw IFM + W_col 的 DRAM image、Conv2D golden 和 direct Conv2D 寄存器参数；新增 `scripts/run_conv2d_otf_case.ps1`。
- `scripts/run_regression.ps1` 已接入 T6.2 默认 case：INT8 OS、INT8 WS、FP16 OS。
- 补齐独立 DMA testbench 对新增 `a_im2col_*` 端口的 0 连接，避免普通 A 读单测中未连接输入变成 `x/z`。

T6.2 验证记录：

```text
scripts/run_conv2d_otf_case.ps1 -Mode OS -Name conv2d_otf_int8_os_default -> ALL 75 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Mode WS -Name conv2d_otf_int8_ws_default -> ALL 75 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype fp16 -Mode OS -Name conv2d_otf_fp16_os_default -> ALL 75 CHECKS PASSED
tb_dma_read_burst/tb_dma_write_burst/tb_dma_burst/tb_dma_perf -> PASS
```

T6.3 实现记录：

- `npu_axi_lite` 新增 `BIAS_ADDR(0x98)`，`CTRL[9]` 作为 direct scalar bias enable。
- `npu_ctrl` 在 direct scalar、非 tile mode 下为每个输出列 `j` 发起 `bias_addr + j*4` 的 bias fetch，并等待 `dma_bias_done` 后再进入 compute。
- `npu_dma` 新增 one-beat 32-bit bias fetch 目标，和当前 W/A load 串行使用读通道，返回的 `bias_data` 作为 scalar PE `acc_init`。
- `tb/matmul/gen_matmul_data.py --bias` 和 `tb/conv2d/gen_conv2d_im2col_data.py --bias` 生成 32-bit bias vector、带 bias 的 expected 和 `BIAS_ADDR/BIAS_EN` 参数。
- `scripts/run_matmul_case.ps1 -Bias`、`scripts/run_conv2d_im2col_case.ps1 -Bias`、`scripts/run_conv2d_otf_case.ps1 -Bias` 可单独运行 T6.3 case；`scripts/run_regression.ps1` 已接入 direct matmul bias 和 on-the-fly Conv2D bias 默认 case。

T6.3 验证记录：

```text
scripts/run_matmul_case.ps1 -M 3 -K 5 -N 4 -Dtype int8 -Mode OS -Bias -> ALL 12 CHECKS PASSED
scripts/run_matmul_case.ps1 -M 3 -K 5 -N 4 -Dtype int8 -Mode WS -Bias -> ALL 12 CHECKS PASSED
scripts/run_matmul_case.ps1 -M 3 -K 4 -N 3 -Dtype fp16 -Mode OS -Bias -> ALL 9 CHECKS PASSED
scripts/run_matmul_case.ps1 -M 2 -K 3 -N 2 -Dtype fp16 -Mode WS -Bias -> ALL 4 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype int8 -Mode OS -Bias -> ALL 75 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype int8 -Mode WS -Bias -> ALL 75 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype fp16 -Mode OS -Bias -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1 -> TOTAL: 1802 PASS, 0 FAIL
```

T6.4 实现记录：

- `CTRL[11:10]` 定义 direct scalar activation mode：`00=none`、`01=ReLU`、`10=ReLU6`；controller 在 start 时锁存并输出给 top。
- `npu_top` 在 direct scalar result FIFO 前执行 activation，顺序为 accumulator -> optional bias -> activation；tile/descriptor 主线暂不启用。
- INT8 输出仍是 32-bit accumulator word，ReLU6 语义为 signed int32 clamp 到 `[0,6]`；FP16 输出为 FP32 word，ReLU6 clamp 到 `[0.0,6.0]`。
- `tb/matmul/gen_matmul_data.py --activation relu|relu6` 和 `tb/conv2d/gen_conv2d_im2col_data.py --activation relu|relu6` 生成 activation 后 expected。
- `scripts/run_matmul_case.ps1`、`scripts/run_conv2d_im2col_case.ps1`、`scripts/run_conv2d_otf_case.ps1` 新增 `-Activation none|relu|relu6`；`scripts/run_regression.ps1` 已接入 T6.4 direct scalar matmul 和 Conv2D on-the-fly case。

T6.4 验证记录：

```text
scripts/run_matmul_case.ps1 -M 3 -K 5 -N 4 -Dtype int8 -Mode OS -Bias -Activation relu -> ALL 12 CHECKS PASSED
scripts/run_matmul_case.ps1 -M 3 -K 5 -N 4 -Dtype int8 -Mode WS -Bias -Activation relu6 -> ALL 12 CHECKS PASSED
scripts/run_matmul_case.ps1 -M 3 -K 4 -N 3 -Dtype fp16 -Mode OS -Bias -Activation relu -> ALL 9 CHECKS PASSED
scripts/run_matmul_case.ps1 -M 2 -K 3 -N 2 -Dtype fp16 -Mode WS -Bias -Activation relu6 -> ALL 4 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype int8 -Mode OS -Bias -Activation relu -> ALL 75 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype int8 -Mode WS -Bias -Activation relu6 -> ALL 75 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype fp16 -Mode OS -Bias -Activation relu6 -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1 -> TOTAL: 2064 PASS, 0 FAIL
```

T6.5 实现记录：

- `npu_axi_lite` 新增 `QUANT_CFG(0x9C)`：bit0 enable，bit1 round enable，`[15:8]` arithmetic right shift，`[31:16]` signed scale。
- `npu_ctrl` 在 direct scalar start 时锁存 `QUANT_CFG`，tile mode 和 FP16 路径默认关闭 quant。
- `npu_top` 在 direct scalar result FIFO 前执行 INT8 quant/saturate，顺序为 accumulator -> optional bias -> activation -> optional quantize/saturate；输出为 sign-extended signed int8 word。
- 量化口径为 `scaled = value * signed_scale`，可选 signed rounding 后算术右移，再饱和到 `[-128, 127]`。
- `tb/matmul/gen_matmul_data.py --quant --quant-scale --quant-shift --quant-round` 和 `tb/conv2d/gen_conv2d_im2col_data.py --quant ...` 生成量化后的 expected。
- `scripts/run_matmul_case.ps1`、`scripts/run_conv2d_im2col_case.ps1`、`scripts/run_conv2d_otf_case.ps1` 新增 `-Quant/-QuantScale/-QuantShift/-QuantRound`；`scripts/run_regression.ps1` 已接入 T6.5 direct scalar matmul 和 Conv2D on-the-fly case。

T6.5 验证记录：

```text
scripts/run_matmul_case.ps1 -M 3 -K 5 -N 4 -Dtype int8 -Mode OS -Bias -Activation relu -Quant -QuantScale 3 -QuantShift 5 -QuantRound -> ALL 12 CHECKS PASSED
scripts/run_matmul_case.ps1 -M 3 -K 5 -N 4 -Dtype int8 -Mode WS -Bias -Quant -QuantScale 1 -QuantShift 3 -> ALL 12 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype int8 -Mode OS -Bias -Activation relu -Quant -QuantScale 2 -QuantShift 3 -QuantRound -> ALL 75 CHECKS PASSED
scripts/run_conv2d_otf_case.ps1 -Dtype int8 -Mode WS -Bias -Quant -QuantScale 1 -QuantShift 2 -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1 -> TOTAL: 2238 PASS, 0 FAIL
```

T6.6 实现记录：

- 新增 `tb/conv2d/gen_conv2d_two_layer_data.py` 生成两层 INT8 Conv2D E2E 数据：layer0 为 raw NCHW IFM + on-the-fly im2col + bias + ReLU + INT8 quant，layer1 直接使用 layer0 `R_ADDR` 作为 `A_ADDR`。
- 新增 `tb/conv2d/tb_conv2d_two_layer.v`，在同一个 DRAM 模型中顺序运行两层，先检查 layer0 量化 OFM，再检查 layer1 最终 golden。
- 新增 `scripts/run_conv2d_two_layer_case.ps1`，并将默认 case 接入 `scripts/run_regression.ps1`。
- 同步修复 SoC 系统仿真：`dram_model` 支持 `axi_arlen` 和 AXI write burst，`soc_top` 的 PicoRV32 PCPI 端口对齐参考核，`axi_lite_bridge` 分离 AW/W 握手，SoC SRAM/DRAM CPU 读口改为组合读以匹配 PicoRV32 ready/rdata 时序；`run_soc_sim.ps1` 默认关闭 VCD，`-DumpVcd` 可选。

T6.6 验证记录：

```text
scripts/run_conv2d_two_layer_case.ps1 -> ALL 48 CHECKS PASSED
scripts/run_regression.ps1 -> TOTAL: 2286 PASS, 0 FAIL
scripts/run_soc_sim.ps1 -> [PASS] SoC integration test PASSED, Cycles: 247, C00=19 C01=22 C10=43 C11=50
```

当前 T2.1-T2.6、T3.1-T3.5、T4.1-T4.5、T5.1-T5.5、T6.1-T6.6、T7.1-T7.5 已完成。下一步进入 descriptor 化 Conv2D、外部 PSUM surface 接入、packed K lane 供数或更大 tile 完整写回。建议顺序：

```text
descriptor 化 Conv2D / packed K lane 供数
```

T7.1 实现记录：
- `pingpong_buf` 的 vector read 增加运行时 `rd_vec_lanes`，同一实例支持 4/8/16 lane 读取，INT8/FP16 都按实际 lane 数推进读指针。
- `npu_top` 将 tile feeder 扩展到最多 16 lane，并按 `CFG_SHAPE` 选择 4/8/16 lane；A 侧 row-skew 延迟链从固定 4 lane 改为 16 lane 生成式结构。
- `npu_ctrl` 的 tile DMA byte/k、K-split 容量和 OS drain 周期按 shape lane 数计算，避免 8x8/16x16 供数长度仍停留在 4-lane。
- 新增 `tb/tb_npu_tile_lane_feed.v`，通过真实 AXI/DMA 路径检查 8x8 与 16x16 的 active W/A lane 均送达阵列边界；`scripts/run_regression.ps1` 已接入该用例。

T7.1 验证记录：
```text
tb_pingpong_buf_vec -> INT8_VEC8 / INT8_VEC16 / FP16_VEC16 PASS
tb_npu_tile_lane_feed -> 8x8 lane feed observed, 16x16 lane feed observed, PASS
scripts/run_regression.ps1 -> TOTAL: 2289 PASS, 0 FAIL
```

T7.2 实现记录：
- `reconfig_pe_array` 的 8x32 mode 将 16x16 物理阵列拆成两个 8x16 半阵列；top half 对应逻辑列 0..15，bottom half 对应逻辑列 16..31。
- bottom half 的 activation 输入改为从对应 top-half row 的 16-column 水平链末端接入，避免只把 row7/col15 单点折到 row8/col0。
- 8x32 OS weight 垂直链在 row7/row8 之间断开，bottom half 从同一拍 `w_in[c]` 重新进入，形成独立右半阵列。
- 8x32 WS load row 以 8 行为周期，同时装载 top half row r 和 bottom half row r+8；`ws_load_row_out` 在 0..7 间回卷。
- 输出映射保持 32-lane 逻辑顺序：`acc_out[0..15]` 来自 top half，`acc_out[16..31]` 来自 bottom half。
- 折叠源线在最后一列 shift register 处生成，避免 Icarus 对 `act_h[*][PHY_COLS]` 末列索引产生越界 warning。

T7.2 验证记录：
```text
tb_reconfig_pe_8x32 -> output order PASS, folded activation route PASS, WS load row wrap PASS
manual compile/run tb_reconfig_pe_8x32_current.vvp -> 4 PASS, 0 FAIL
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -> TOTAL: 2293 PASS, 0 FAIL
```

T7.3 实现记录：
- `pe_top` 新增 `INT8_SIMD_LANES=2` 参数，INT8 packed 输入按 `{lane1,lane0}` 执行两路 signed 8-bit multiply，并在 PE 内合并为一个 32-bit accumulator 加数。
- OS 路径支持每拍 packed `w_in/a_in` 的 2-lane dot；WS 路径支持 packed weight latch 后对 packed activation 连续 MAC。
- 旧 direct scalar/PPBuf feeder 仍会把单个 INT8 sign-extend 到 16 bit；当 W/A 两侧都呈现旧 sign-extended scalar 编码时，PE 保持单 lane 兼容，避免 `16'hFFFF * 16'hFFFF` 被误算成两次 MAC。
- `tb/tb_pe_top.v` 增加 T7.3 覆盖：INT8 packed OS、INT8 packed WS、负数 lane 和旧 scalar 兼容；`scripts/run_regression.ps1` 已接入 `pe_top` 回归入口。
- 当前 T7.3 是 PE 级 packed SIMD 能力；tile/descriptor 主线的 packed K lane 供数、valid 对齐和端到端 2x 吞吐验证仍是后续工作。

T7.3 验证记录：
```text
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -> PASS=25 FAIL=0
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -> TOTAL: 2318 PASS, 0 FAIL
```

T7.4 实现记录：
- `pe_top` 的 `INT8_SIMD_LANES` 默认值改为随 `DATA_W` 选择：`DATA_W=16` 保持 2-lane，`DATA_W>=32` 使用 4-lane。
- INT8 datapath 扩展到 lane0..lane3，packed 约定为 `{lane3,lane2,lane1,lane0}`；四个 signed 8-bit product 在 PE 内合并为一个 18-bit 中间和，再 sign-extend 到 32-bit accumulator。
- 旧 sign-extended scalar 兼容检测扩展到全 `DATA_W` 宽度：当 W/A 的高位都只是 lane0 符号扩展时，只计算 lane0，避免 `32'hFFFFFFFF * 32'hFFFFFFFF` 被误算成 4 次 MAC。
- `tb/tb_pe_top.v` 新增 32-bit `u_pe_simd4` DUT，覆盖 INT8 packed 4-lane OS、WS packed weight latch、负数 lane 和全宽 scalar 兼容；原 16-bit DUT 的 T7.3/FP16/acc_init 回归保持不变。
- 当前 T7.4 是 PE 级 4-lane SIMD 能力；阵列/PPBuf/top/descriptor 主线仍默认 `DATA_W=16`，端到端 1.024 TOPS 还需要后续 32-bit packed K lane 供数、valid 对齐和写回配套。

T7.4 验证记录：
```text
powershell -ExecutionPolicy Bypass -File scripts/run_sim.ps1 -> PASS=28 FAIL=0
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -> TOTAL: 2321 PASS, 0 FAIL
```

T7.5 实现记录：
- `op_counter` 改为在 `ctrl_done` 时按 `M*N*K` 累计 useful MAC，并输出 `total_ops`、`peak_ops_per_cycle`、`tops_x1e6`、`compute_util_bp` 和 `e2e_util_bp`；其中 1 MAC = 2 ops，`tops_x1e6` 表示 `TOPS * 1,000,000`，util 单位为 basis points。
- `npu_top` 实例化 `op_counter`，非 tile 路径按 1x1 scalar peak 统计；tile 路径按当前 `CFG_SHAPE` 的 active row/col 统计。由于 packed K lane 供数尚未接入顶层，`INT8_SIMD_LANES` 默认保持 1，避免把 PE 级 SIMD 能力误报为端到端吞吐。
- `npu_axi_lite` 新增 `0xA0..0xC8` 只读寄存器，暴露 MAC/OPS、busy/compute/DMA cycles、TOPS fixed-point、compute/e2e utilization 和 peak ops/cycle。
- 新增 `tb/tb_op_counter_perf.v`，用固定 16x16x16 workload 验证公式：4096 MAC、8192 ops、16 busy/compute cycles、`TOPS_X1E6=256000`、util=10000bp。
- `tb/tb_npu_scalar_smoke.v` 增加顶层 AXI-Lite 读取检查，并打印可引用的 `[PERF] scalar_smoke ...` 报告行；`scripts/run_regression.ps1` 已接入 `op_counter_perf`。

T7.5 验证记录：
```text
tb_op_counter_perf -> [PERF] op_counter MAC_OPS=4096 OPS=8192 BUSY_CYCLES=16 COMPUTE_CYCLES=16 TOPS_X1E6=256000 COMPUTE_UTIL_BP=10000 E2E_UTIL_BP=10000 PEAK_OPS_CYCLE=512
tb_op_counter_perf -> ALL 9 CHECKS PASSED
tb_npu_scalar_smoke -> [PERF] scalar_smoke MAC_OPS=4 OPS=8 BUSY_CYCLES=34 COMPUTE_CYCLES=5 DMA_CYCLES=29 TOPS_X1E6=117 COMPUTE_UTIL_BP=8000 E2E_UTIL_BP=1176 PEAK_OPS_CYCLE=2
tb_npu_scalar_smoke -> PASS
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_regression.ps1 -> TOTAL: 2330 PASS, 0 FAIL
```
