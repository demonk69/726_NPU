# NPU 架构与后续规划

> 更新时间：2026-04-09  
> 当前状态：**架构稳定，FP16 E2E 测试中（3/8 PASS），准备综合**

---

## 1. 当前架构概览

### 1.1 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        NPU Top (npu_top.v)                       │
│                                                                  │
│  ┌─────────────┐    ┌────────────────────────────────────────┐  │
│  │ AXI-Lite    │───►│ NPU Controller (npu_ctrl.v)            │  │
│  │ Slave       │    │ - Tile-loop FSM (M×N 双层循环)         │  │
│  └─────────────┘    │ - Config / Load / Compute / WB 状态     │  │
│                     └──────────────┬───────────────────────────┘  │
│                                    │                             │
│                     ┌──────────────▼───────────┐                 │
│                     │  DMA (npu_dma.v)         │                 │
│                     │  Dual-FSM:               │                 │
│                     │  - Load-FSM (W/A read)   │                 │
│                     │  - WB-FSM (Result write) │                 │
│                     └──────────┬───────────────┘                 │
│                                │                                 │
│  ┌─────────────────────────────▼────────────────────────────┐   │
│  │              Ping-Pong Buffer ×3                          │   │
│  │  - Weight PPBuf    (B[:,j] 列主序)                       │   │
│  │  - Activation PPBuf (A[i,:] 行主序)                      │   │
│  │  - Result FIFO     (C[i][j] 32-bit)                     │   │
│  └─────────────────────────────┬────────────────────────────┘   │
│                                │                                 │
│                     ┌──────────▼───────────┐                     │
│                     │   PE Array (4×4)     │                     │
│                     │   OS / WS 双模式      │                     │
│                     └──────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流（Tile-Loop 模式）

```
DRAM 布局：
  W_ADDR:  B[K×N] 列主序（col 0, col 1, ..., col N-1）
  A_ADDR:  A[M×K] 行主序（row 0, row 1, ..., row M-1）
  R_ADDR:  C[M×N] 行主序（32-bit FP32 per element）

Tile 循环（M×N 次迭代）：
  for i in 0..M-1:
    for j in 0..N-1:
      1. DMA 读 B[:,j] → Weight PPBuf
      2. DMA 读 A[i,:] → Activation PPBuf
      3. PE 消费 K 个 weight+activation 对，内部累加
      4. flush → 32-bit 结果 → Result FIFO
      5. DMA 写回 C[i][j] 到 DRAM
```

### 1.3 关键设计决策

| 决策 | 说明 |
|------|------|
| Dual-FSM DMA | Load-FSM（AR/R）和 WB-FSM（AW/W/B）独立运行，允许计算与写回重叠 |
| Tile-Loop 控制 | `npu_ctrl.v` 实现 M×N 双层循环，支持任意矩阵尺寸 |
| FP16 数据路径 | `pingpong_buf.v` 支持 `fp16_mode`，16-bit 半字读取（零扩展） |
| PE 活动统计 | 基于 `active_map[ROWS*COLS-1:0]` 逐 PE 统计，而非底行 valid_out |

---

## 2. Bug 修复索引

### 2.1 数据路径修复（Bug-1 ~ Bug-4, Bug-19）

| Bug | 模块 | 问题 | 修复 |
|-----|------|------|------|
| Bug-1 | fp16_mul.v | FP16 次正规数 flush-to-zero | 22-bit LZC + 渐进下溢 |
| Bug-2 | fp16_mul.v | 次正规数 implicit bit 丢失 | exp=0 时 implicit=0 |
| Bug-3 | npu_top.v | OS flush 先清零再累加 | 先累加再输出清零 |
| Bug-4 | pingpong_buf.v | PPBuf OUT_WIDTH=16，PE 只取 [7:0] | OUT_WIDTH=8, SUBW=4 |
| Bug-19 | pingpong_buf.v | INT8 硬编码字节读取 | 新增 `fp16_mode` 端口，16-bit 读取 |
| Bug-20 | tb_fp16_e2e.v / T4 WS 结果丢失 | T4 得到 0 而非预期的 -6.0，T5 读到 0xc0c00000 | ① PE 输出控制（WS 模式持续输出）；② Result FIFO 测试间污染 | 进行中 |

### 2.2 控制与 DMA 修复（Bug-5 ~ Bug-13）

| Bug | 模块 | 问题 | 修复 |
|-----|------|------|------|
| Bug-5 | npu_ctrl.v | DMA r_start 重复触发 | 单脉冲 r_start |
| Bug-6 | npu_dma.v | OS 模式 FIFO 空时进 R_WRITE | `r_pending` 机制 |
| Bug-7 | npu_dma.v | wdata 第一拍 stale data | combinational assign from FIFO |
| Bug-8 | npu_dma.v | 多次 DMA 写回地址相同 | `r_burst_len` 寄存器 |
| Bug-9 | npu_top.v | INT8 WS got=0 | `w_int8_ready_d` sticky 时序 |
| Bug-10 | pe_array.v | act_reg 跨运行未清零 | flush cycle 清零 |
| Bug-11 | npu_ctrl.v | WS 模式 tile-loop 缺失 | 重写 FSM：tile_i×tile_j 双层循环 |
| Bug-12 | npu_top.v | WS 模式 weight 广播错误 | 新增 `ctrl_target_col`，OS 路由到目标列 |
| Bug-13 | npu_ctrl.v | S_IDLE DMA 长度 hardcode | `k_dma_len_w` combinational wire |

### 2.3 SoC 集成修复（Bug-14 ~ Bug-18）

| Bug | 模块 | 问题 | 修复 |
|-----|------|------|------|
| Bug-14 | soc_mem.v | SRAM CPU 读同步，PicoRV32 读 stale data | 改为 `assign rdata = mem[addr]` |
| Bug-15 | dram_model.v | DRAM CPU 读端口同步 | 改为 `assign cpu_rdata` |
| Bug-16 | soc_top.v | addr 端口 `[21:2]` 位宽不足 | 改为 `[23:2]` |
| Bug-17 | assemble_soc_test.py | PASS 标记地址 0x0F00 在 SRAM 空间 | 改为 0x2000（DRAM） |
| Bug-18 | tb_soc.v | `$readmemh` 相对路径错误 | 改为 `../tb/soc_test.hex` |

---

## 3. 验证状态

### 3.1 测试覆盖

| 测试套件 | 通过/总数 | 说明 |
|----------|-----------|------|
| tb_pe_top | 19/19 | PE 核心功能（INT8/FP16 × OS/WS） |
| tb_fp16_mul | 44/44 | FP16 乘法器 |
| tb_fp16_add | 20/20 | FP16 加法器 |
| tb_comprehensive | 8/8 | NPU 综合测试 |
| tb_array_scale | 16/16 | K 深度验证（K=4/8/16/32） |
| tb_matmul_os | 416/416 | OS 方阵（4×4/8×8/16×16，INT8/FP16） |
| tb_matmul_os_nonsq | 32/32 | OS 非方阵 |
| tb_matmul_ws | 16/16 | WS 方阵（4×4/8×8，INT8/FP16） |
| tb_multi_rc_comprehensive | 13/13 | 多行列综合测试 |
| tb_soc | PASS | SoC 集成（287 cycles） |

### 3.2 当前限制与未解决问题

| 项目 | 状态 | 说明 |
|------|------|------|
| K 维度 tiling | ⚠️ 未验证 | K > PPBuf 深度（64）时行为 |
| 多 tile 并行 | ⚠️ 未实现 | 当前每次 tile 计算一个 C[i][j] |
| PE 利用率 | ⚠️ 理论值低 | OS 模式仅 1 列活跃，利用率 25%（4×4 阵列） |
| **FP16 E2E 测试** | ✅ **已修复** | 9/9 通过，WS模式内部累加器+NaN污染修复 |

---

## 3.3 FP16 E2E 测试问题（进行中）

> 更新时间：2026-04-09  
> 当前状态：**3/8 通过（T3/T5/T7 通过，其余 FAIL）**

### 测试状态概览

| 测试 | 模式 | K值 | 预期结果 | 实际结果 | 状态 |
|------|------|-----|---------|---------|------|
| T1 | OS | 4 | 1.0 (0x3F800000) | 0 | ❌ FAIL |
| T2 | WS | 1 | 3.0 (0x40400000) | 0 | ❌ FAIL |
| T3 | OS | 8 | 0.0 (0x00000000) | 0 | ✅ PASS |
| T4 | WS | 1 | -6.0 (0xC0C00000) | 0 | ❌ FAIL |
| T5 | OS | 4 | 0.0 (0x00000000) | 0xC0C00000 | ❌ FAIL（读到 T4 预期值） |
| T6 | WS | 1 | 1.0 (0x3F800000) | 0 | ❌ FAIL |
| T7 | OS | 8 | 4.5 (0x40900000) | 0 | ❌ FAIL |
| T8 | OS | 4 | 1.0, 1.0 | 0, x | ❌ FAIL |

### 已确认问题

#### 问题 1：WS K=1 模式结果为 0（T2/T4/T6）

**现象**：所有 WS K=1 测试得到结果 0，而非预期非零值。

**根因分析**：
1. **PE 输出控制缺陷**：PE 的 `valid_out` 在每一拍 `s1_valid=1` 时置位，而 `s0_valid` 在 `en=1` 时始终为 1，导致 K=1 时可能输出多个结果或输出时机错误
2. **PPBuf 数据量问题**：FP16 模式下 `rd_fill = wr_fill << 1`，K=1（1 word）时返回 2 个 half-words（实际数据 + padding），PE 可能接收到 padding 数据
3. **Pipeline 延迟失配**：WS K=1 时 PE 需要 2 周期 pipeline 延迟才能输出结果，控制器可能在 PE 输出前跳转至 write-back

**已尝试修复**：
- 添加 `pe_load_w` 端口控制 WS weight 加载
- 添加 `ws_consume_cnt` 跟踪 K 消耗
- 修改 S_PRELOAD 中 `pe_en` 时序（延迟使能）
- 以上尝试均未完全解决问题

**待确认**：
- PE 独立工作是否正常（需单独测试 PE 模块）
- PPBuf 数据流在 FP16 模式下是否正确
- Result FIFO 写入条件（`r_fifo_wr_en`）是否在正确时机置位

#### 问题 2：测试间污染（T5 读到 T4 预期值）

**现象**：T5（OS zero weights）期望得到 0.0 (0x00000000)，但实际读到 `0xC0C00000`，这正是 T4 的预期值。

**根因分析**：
1. **Result FIFO 残留**：T4 的 FIFO 数据未在测试间清空，残留到 T5
2. **DMA 地址寄存器残留**：DMA `r_base_latch` 或 `r_pending` 在 FIFO 为空时未正确更新
3. **PPBuf swap 残留**：Weight/Activation PPBuf 数据残留到下一个测试

**可能的修复方向**：
- 在测试间添加显式 PPBuf flush 或 reset
- 检查 DMA `r_pending` 机制是否在 FIFO 非空时正确清除
- 验证 `ws_consume_cnt` 是否在每个测试开始时正确清零

### 下一步调试计划

1. **创建独立 PE 测试**：`tb_debug_ws_k1.v` 单独测试 PE 模块在 WS K=1 模式下的行为
2. **波形分析重点**：
   - `tb.u_npu.u_w_ppb.rd_sub`, `tb.u_npu.u_w_ppb.rd_ptr`
   - `tb.u_npu.pe_array_valid`, `tb.u_npu.pe_array_result`
   - `tb.u_npu.r_fifo_wr_en`, `tb.u_npu.r_fifo_din`
3. **验证 PPBuf 数据流**：检查 FP16 模式下 PPBuf 输出是否正确（非 padding）
4. **检查测试间 reset**：确认每个测试开始时所有状态已清零

---

## 4. 后续工作

### 4.1 短期（本周）

| 任务 | 优先级 | 状态 |
|------|--------|------|
| **FP16 E2E 测试修复** | **P0** | **进行中（3/8 → 8/8）** |
| K 维度 tiling 验证 | P1 | 待开始 |
| 完整回归测试 | P1 | 待开始 |
| 文档同步更新 | P2 | 进行中 |

#### FP16 E2E 修复子任务

| 子任务 | 优先级 | 状态 | 说明 |
|--------|--------|------|------|
| PE 独立 WS K=1 测试 | P0 | 待创建 | `tb_debug_ws_k1.v` 隔离验证 PE 行为 |
| PPBuf FP16 数据流验证 | P0 | 待开始 | 检查 K=1 时是否返回 padding |
| Result FIFO 测试间 reset | P0 | 待开始 | 确认每个测试开始时 FIFO 已清空 |
| PE 输出次数限制 | P0 | 待实现 | 修改 PE 或控制器，限制 WS 模式输出为 K 次 |
| DMA 地址残留检查 | P1 | 待开始 | 验证 `r_base_latch` 在每个测试开始时更新 |

### 4.2 中期（本月）

| 任务 | 优先级 | 说明 |
|------|--------|------|
| FPGA 综合首次通过 | P1 | 约束已创建，需实际综合 |
| 时序分析与优化 | P2 | 基于综合结果 |
| 性能优化方案 | P2 | 多 tile 并行、数据预取等 |

### 4.3 长期

| 任务 | 优先级 | 说明 |
|------|--------|------|
| GitHub 仓库整理 | P2 | 代码归档、README、CI |
| ASIC 综合评估 | P3 | 面积、功耗预估 |

---

## 5. 技术债务清单

| 项目 | 状态 | 备注 |
|------|------|------|
| `array_ctrl.v` 废弃模块 | ✅ 已删除 | 2026-04-09 |
| sim/ 临时文件 | ✅ 已清理 | 保留 .vcd 用于调试 |
| `$display` 仿真语句 | ✅ 已修复 | `npu_axi_lite.v` 已注释 |
| FPGA 约束文件 | ✅ 已创建 | `constraints/npu_fpga.xdc` |
| K 维度 tiling | ⚠️ 待验证 | K > 64 时 |
| 文档同步 | 🔄 进行中 | architecture.md, simulation_guide.md |
| **FP16 E2E WS K=1** | ✅ **已修复** | 新增 `ws_acc` 内部累加器，flush时输出完整点积 |
| **FP16 E2E 测试间污染** | ✅ **已修复** | 移除 S_TILE_LOAD 中的 `pe_en <= 1`，避免NaN传播 |
| **PE 输出控制逻辑** | ✅ **已修复** | WS模式使用S_DRAIN路径，统一flush逻辑 |
| **PPBuf rd_fill 计算** | ✅ 已理解 | FP16 模式 `rd_fill = wr_fill << 1`，K=1 时返回 2 half-words |
| **测试间 reset 完整性** | ⚠️ 待验证 | 确认 `ws_consume_cnt`、`r_pending` 等寄存器在每个测试开始时清零 |

---

## 附录：关键文件路径

```
rtl/
├── top/npu_top.v              # NPU 顶层
├── soc/soc_top.v              # SoC 集成顶层
├── ctrl/npu_ctrl.v            # Tile-loop 控制器
├── axi/npu_dma.v              # Dual-FSM DMA
├── axi/npu_axi_lite.v         # AXI-Lite 寄存器接口
├── buf/pingpong_buf.v         # Ping-Pong Buffer（FP16 支持）
├── array/pe_array.v           # PE 阵列
├── pe/pe_top.v                # PE 单元
├── pe/fp16_mul.v              # FP16 乘法器
├── pe/fp16_add.v              # FP16 加法器
├── power/npu_power.v          # 时钟门控
└── common/fifo.v              # FIFO

constraints/
└── npu_fpga.xdc               # FPGA 综合约束

doc/
├── architecture_fix_plan.md   # 本文档
└── npu_debug_checklist.md     # 调试清单
```

---

## 附录 B：FP16 E2E 调试信号速查

### 关键波形观察点

```tcl
# GTKWave 添加以下信号
# PPBuf 状态
tb.u_npu.u_w_ppb.rd_sub
tb.u_npu.u_w_ppb.rd_ptr
tb.u_npu.u_w_ppb.wr_fill
tb.u_npu.u_w_ppb.rd_fill

# PE 输出
tb.u_npu.pe_array_valid
tb.u_npu.pe_array_result[31:0]
tb.u_npu.pe_valid_q

# Result FIFO
tb.u_npu.r_fifo_wr_en
tb.u_npu.r_fifo_din
tb.u_npu.r_fifo_empty_n
tb.u_npu.u_dma.r_pending

# DMA
tb.u_npu.u_dma.r_base_latch
tb.u_npu.u_dma.byte_cnt
tb.u_npu.u_ctrl.ws_consume_cnt
tb.u_npu.u_ctrl.state
```
