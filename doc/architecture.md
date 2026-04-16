# NPU_prj — 系统架构文档

> 最后更新：2026-04-14
> 文档状态：已按可重配置 PE 阵列（16×16 + 四形态 cfg_shape）与双权重寄存器升级同步

---

## 目录

1. [项目概览](#1-项目概览)
2. [顶层系统结构](#2-顶层系统结构)
3. [NPU 内部结构详解](#3-npu-内部结构详解)
4. [数据流与 DRAM 布局](#4-数据流与-dram-布局)
5. [控制寄存器映射](#5-控制寄存器映射)
6. [控制器状态机](#6-控制器状态机真-ping-pong-版)
7. [地址锁存与层间切换](#7-地址锁存与层间切换)
8. [SoC 地址空间与关键时序约束](#8-soc-地址空间与关键时序约束)
9. [能力矩阵与性能估算](#9-能力矩阵与性能估算)
10. [配套文档阅读顺序](#10-配套文档阅读顺序)

---

## 1. 项目概览

NPU_prj 是一个面向 SoC 集成的 Verilog NPU 原型，实现了：

- **16×16 可重配置 PE 阵列**（物理 256 PE，通过 `cfg_shape` 运行时切换 4×4 / 8×8 / 16×16 / 8×32 四种形态）
- **INT8 / FP16** 两条已验证数据通路
- **AXI4-Lite 配置接口** + **AXI4 DMA 数据通路**
- **WS（Weight-Stationary）/ OS（Output-Stationary）** 两种计算模式
- **真 Ping-Pong 重叠执行**：DMA 搬运与 PE 计算并行（2026-04-13 升级）
- **双权重寄存器预取**：每个 PE 内置 active/prefetch 双寄存器，`swap_w` 单拍切换，隐藏权重加载延迟（2026-04-14 升级）
- **PicoRV32 + SRAM + DRAM + NPU** 的完整 SoC 集成验证

### 1.1 核心设计概念：Tile-Based 计算

本 NPU 采用 **tile-based 计算模型**，每个 tile 对应矩阵乘法 `C[M×N] = A[M×K] × B[K×N]` 中的一个输出元素 `C[i][j]`。

| 概念 | 说明 |
|------|------|
| **Tile** | 一个输出元素 `C[i][j]` 的完整计算任务 |
| **Tile 数据** | 需要 B 的第 j 列 `B[:,j]` 和 A 的第 i 行 `A[i,:]` |
| **Tile 内积** | K 次乘累加（MAC），产生 1 个 32-bit 结果 |
| **Tile 循环** | 控制器按 `M×N` 次外层循环逐一处理 |

**重要**：当前 NPU 不是"一次吐出整块 4×4 输出矩阵"的并行架构，而是按 `C[i][j]` 单点推进的 tile-loop 架构。每个 tile 只使用 PE 阵列的一部分资源。

### 1.2 当前验证状态

| 测试套件 | 状态 | 通过/总数 | 备注 |
|---------|------|---------|------|
| PE 单元测试（`tb_pe_top`） | ✅ PASS | 19/19 | 双权重寄存器版全通过 |
| 综合测试（`tb_comprehensive`） | ✅ PASS | 28/28 | 含复杂 FP16 场景 |
| NPU 顶层冒烟（`tb_npu_top`） | ✅ PASS | 4/4 | reconfig_pe_array 集成 |

**总计：51/51 测试通过（2026-04-14，可重配置阵列升级后）**

> 注：旧版 array_scale (16/16)、FP16 E2E (9/9)、multi_rc (13/13)、SoC (1/1) 共计 86 个测试在重构前已全部通过。当前 51 个为重构后的核心回归基线。

### 1.3 WS flush 语义说明

WS 模式下 `flush` 信号采用**纯触发输出**语义：
- flush beat 本身**不参与累加计算**，直接将 `ws_acc` 输出并清零
- 驱动 flush beat 时，`a_in` 和 `w_in` 必须设为 0（纯标记，无实际数据）
- 这与 OS 模式不同：OS 模式的 flush beat 携带最后一个真实计算数据

---

## 2. 顶层系统结构

```
                +--------------------------------------------------+
                |                    soc_top                       |
                |                                                  |
                |  +------------+                                  |
CPU firmware -->|  | PicoRV32   |  RISC-V 软核，执行固件           |
                |  +------+-----+                                  |
                |         | mem_valid/mem_ready (native bus)       |
                |         v                                        |
                |  +------+-----+   +------------------+          |
                |  |  soc_mem   |   |   dram_model     |          |
                |  |  SRAM 4KB  |   |  dual-port DRAM  |          |
                |  | 组合读     |   |  CPU端 组合读     |          |
                |  +------------+   +---------+--------+          |
                |                             ^                    |
                |         +-------------------+                    |
                |         | AXI4-Lite                              |
                |  +------+----------+                             |
                |  | axi_lite_bridge |  mem_if → AXI4-Lite         |
                |  +------+----------+  (3拍写: AW→W→B)           |
                |         | AXI4-Lite (slave)                      |
                |         v                                        |
                |  +------+----------+                             |
                |  |    npu_top      |                             |
                |  +------+----------+                             |
                |         | AXI4 Master (DMA 数据搬运)             |
                +---------+-----------------------------------------+
                          |
                          v (到 dram_model AXI4 端口)
                       DRAM 数据
```

### 2.1 各层职责

| 模块 | 职责 |
|------|------|
| **PicoRV32** | 执行固件：初始化 DRAM、配置 NPU 寄存器、轮询 done/处理中断、校验结果 |
| **soc_mem** | CPU 指令 + 数据 SRAM，组合读（异步读口，与 PicoRV32 同周期返回数据） |
| **dram_model** | CPU 与 NPU DMA 共享的行为级 DRAM，CPU 端组合读，AXI4 端支持 burst |
| **axi_lite_bridge** | 将 PicoRV32 原生 mem 总线转换为 AXI4-Lite，3拍写：`S_WRITE_AW → S_WRITE_W → done` |
| **npu_top** | NPU 顶层：集成寄存器文件、控制器、DMA、双 PPBuf、PE 阵列、结果 FIFO、电源管理 |

---

## 3. NPU 内部结构详解

```
                AXI4-Lite (CPU 配置)
                     |
                     v
          +-------------------+
          |   npu_axi_lite    |   寄存器文件，AXI4-Lite 从机
          |  CTRL/STATUS/DIM  |   暴露所有配置与状态给 CPU
          +--------+----------+
                   |  ctrl_reg, m/n/k_dim, w/a/r_addr, mode, stat
                   v
          +--------+----------+
          |     npu_ctrl      |   tile-loop 控制器 + 真 Ping-Pong FSM
          |  (Shadow Regs)    |   所有配置在 start 脉冲时锁存
          +----+----+----+----+
               |    |    |
        DMA控制|    |IRQ |PE 控制
               |    |    |     pe_en/flush/mode/stat/load_w/target_col
               v    v    v
          +----+----+  +-+------------+       +------------------+
          |  npu_dma |  |             |       |    pe_array      |
          | 三通道   |  |  （中断到   |       |   4×4 PE 网格    |
          | AXI4 主机|  |  npu_axi_lite)      |                  |
          +---+---+--+  |             |       |  +---------+     |
              |   |     +-------------+       |  |  pe_top |×16  |
    W读  A读  |   | R写                       |  +---------+     |
              |   |                           |  FP16/INT8 MAC   |
              |   |                           |  3级流水线        |
              v   v                           +---------+--------+
          +---+---+--+                                  |
          | PPBuf_W  |  权重双 Bank                      | valid_out/acc_out
          | (BufA/B) |  DMA 写 Pong，PE 读 Ping          v
          +----------+                        +---------+---------+
          +----------+                        |    sync_fifo      |
          | PPBuf_A  |  激活双 Bank             |   结果 FIFO       |
          | (BufA/B) |  DMA 写 Pong，PE 读 Ping |  PE → DMA 缓冲   |
          +----------+                        +---------+---------+
                  |  swap/clear (来自 npu_ctrl)          |
                  +----------------------------------------+
                  |  rd_en (ppb_rd_active = pe_en)         |
                  |                                        |
                  | PPBuf 读侧输出 8-bit 子字              |
                  v                                        |
          +-------+--------+                              |
          |   npu_top       |   FP16 子字节装配             |
          |  (拼 FP16)      |   2 拍 → 1 个 16-bit FP16   |
          +-------+--------+                              |
                  | pe_data 送入 pe_array                  | r_start
                  v                                        v
             pe_array                               npu_dma R_WRITE → DRAM
```

### 3.1 关键子模块功能总结

#### `npu_ctrl`（控制核心）
- 维护 `tile_i`、`tile_j` 循环计数器（0≤i<M, 0≤j<N）
- 在 `cfg_start_rise` 时**一次性锁存**所有影子寄存器（Shadow Regs）
- 驱动 DMA 启停、PPBuf swap/clear、PE en/flush
- 实现真 Ping-Pong 重叠：PE 计算 tile(i,j) 时，DMA 同步加载 tile(i,j+1)

#### `npu_dma`（三通道 AXI4 主机）
- **W 通道**：从 DRAM 读权重 B[:,j] → 写入 PPBuf_W 的 Pong Bank
- **A 通道**：从 DRAM 读激活 A[i,:] → 写入 PPBuf_A 的 Pong Bank
- **R 通道**：从结果 FIFO 读结果 → 写回 DRAM C[i][j]
- 三通道**共用一条 AXI4 总线**（时分复用），支持多拍 INCR burst

#### `pingpong_buf`（双 Bank 乒乓缓冲）
- **Bank A / Bank B** 交替切换：一侧供 DMA 写入，另一侧供 PE 读取
- 写侧：32-bit 宽（来自 DMA AXI4 读数据）
- 读侧：8-bit 子字宽（逐字节拆解给 PE，INT8=4子字/字，FP16=2×2字节/字）
- `swap` 脉冲（来自 `npu_ctrl`）：切换 wr_sel / rd_sel，本质是时序电路，需等 1 拍稳定

#### `pe_array` / `reconfig_pe_array` / `pe_top`（计算核心）
- **16×16 物理可重配置 PE 阵列**（`reconfig_pe_array.v`，2026-04-14 新增）
  - `cfg_shape` 运行时切换：4×4 / 8×8 / 16×16 / 8×32（折叠模式）
  - 时钟门控：非工作区域 PE 自动关闭（`en && row_active && col_active`）
  - **256 个** `pe_top` 实例（每个含双权重寄存器 active/prefetch）
- 每个 PE 有独立的 3 级流水线：Stage-0（输入寄存器+双权重选择）→ Stage-1（乘法）→ Stage-2（累加/输出）
- OS 模式：权重和激活都 systolic 流动（水平 shift register），PE 内部各自累加 `os_acc`
- WS 模式：权重锁存在 PE 内（`load_w` 脉冲 + `swap_w` 交换），K 个激活流入，PE 内部累加 `ws_acc`，flush 触发输出
- **双权重预取**：`load_w` 同时写入 active 和 prefetch 寄存器（同周期 bypass 可用）；后台加载下一组 weight 后 `swap_w` 单拍切换

#### `npu_axi_lite`（寄存器文件）
- AXI4-Lite 从机，接收 CPU 写操作，维护所有配置寄存器
- 向 `npu_ctrl` 输出 `ctrl_reg`、维度、地址等信号
- 管理 `int_pending`，在 `irq_flag` 上升沿置位，CPU 写 `INT_CLR` 或 `CTRL[6]` 清除
- **写时序**：AW→W 两阶段（不是同拍），与 `axi_lite_bridge` 匹配

---

## 4. 数据流与 DRAM 布局

### 4.1 DRAM 内存布局

| 符号 | 矩阵 | 存储序 | DRAM 地址 |
|------|------|------|-----------|
| `W_ADDR` | `B[K×N]`（权重） | **列主序**：`B[0][j]..B[K-1][j]` 连续存放 | 基地址 |
| `A_ADDR` | `A[M×K]`（激活） | **行主序**：`A[i][0]..A[i][K-1]` 连续存放 | 基地址 |
| `R_ADDR` | `C[M×N]`（结果） | **行主序**：`C[i][j]` 在 `R_ADDR + (i×N+j)×4` | 基地址 |

### 4.2 元素打包规则

| 模式 | DRAM 32-bit 字内容 | PPBuf 拆解 | PE 输入 |
|------|------|------|------|
| INT8 | `[b3 b2 b1 b0]`，4 个 8-bit 元素 | 逐字节，`SUBW=4` | 低 8 位有效 |
| FP16 | `[f1_hi f1_lo f0_hi f0_lo]`，2 个 FP16 | `SUBW=4`（按字节），npu_top 再拼回 16-bit | 16-bit FP16 |
| Result | 1 个 32-bit（INT8: 有符号整数；FP16: FP32） | — | 直接写回 DRAM |

**FP16 子字节装配**（在 `npu_top` 中完成）：

```
PPBuf 输出 8-bit/拍：
  拍 2k  : FP16[7:0]  → 存入 w_fp16_shift[7:0]
  拍 2k+1: FP16[15:8] → 存入 w_fp16_shift[15:8]，w_ppb_phase 1→0
                          ↓
                        pe_data_ready = 1 → pe_consume 触发
```

INT8 模式：每拍一个 8-bit 直接送 PE，`pe_data_ready = w_int8_ready_d && a_int8_ready_d`。

### 4.3 Tile-Loop 调度

#### 地址公式（元素字节数：INT8=1，FP16=2）

```
权重地址：dma_w_addr = W_ADDR + tile_j × K × 元素字节数
激活地址：dma_a_addr = A_ADDR + tile_i × K × 元素字节数
结果地址：dma_r_addr = R_ADDR + (tile_i × N + tile_j) × 4
```

#### Tile 物理映射

**OS 模式（每个 tile 计算 `C[i][j]`，一个输出元素）：**
```
target_col = tile_j % COLS

权重路由：B[:,j] 只送到 target_col 列（其余列 w_in=0）
激活广播：A[i,:] 广播到所有行
有效 PE：PE[i][target_col] 产生有效累加（os_acc）
输出：PE[i][target_col].acc_out → FIFO → DRAM
```

**WS 模式（每个 tile 计算 `C[i][j]`，一个输出元素）：**
```
target_col = tile_j % COLS

权重广播：B[:,j] 广播到整列 PE[*][target_col]（load_w 脉冲锁存 weight_reg）
激活路由：A[i,:] 按行广播（行错位 1 拍，形成脉动流）
累加：K 拍内 ws_acc += s1_mul
输出：flush 触发，ws_acc 输出到 FIFO（flush beat 自身 w_in=a_in=0）
```

---

## 5. 控制寄存器映射

基地址：**`0x0200_0000`**

| 偏移 | 名称 | R/W | 位定义 |
|-----:|------|:---:|--------|
| 0x00 | CTRL | RW | `[0]` start（上升沿触发）<br>`[1]` abort（中止运算）<br>`[3:2]` data_mode：`00`=INT8，`10`=FP16<br>`[5:4]` stat_mode：`00`=WS，`01`=OS<br>`[6]` irq_clr（**W1C**，写 1 清 IRQ，读回 0） |
| 0x04 | STATUS | RO | `[0]` busy，`[1]` done（sticky，清 start 后回零） |
| 0x08 | INT_EN | RW | `[0]` 中断使能 |
| 0x0C | INT_CLR | W1C | `[0]` 写 1 清 int_pending；读回 pending 状态 |
| 0x10 | M_DIM | RW | 矩阵 M 维度 |
| 0x14 | N_DIM | RW | 矩阵 N 维度 |
| 0x18 | K_DIM | RW | 矩阵 K 维度（内积长度） |
| 0x20 | W_ADDR | RW | 权重 DRAM 基地址（字节对齐） |
| 0x24 | A_ADDR | RW | 激活 DRAM 基地址 |
| 0x28 | R_ADDR | RW | 结果 DRAM 基地址 |
| 0x30 | ARR_CFG | RW | `[3:0]` rows，`[7:4]` cols（保留） |
| 0x34 | CLK_DIV | RW | `[2:0]` DFS 分频选择（接 `npu_power`） |
| 0x38 | CG_EN | RW | 时钟门控使能（保留） |
| **0x3C** | **CFG_SHAPE** | **RW** | **`[1:0]` PE 阵列形态：`00`=4×4, `01`=8×8, `10`=16×16, `11`=8×32** |

### 5.3 PE 阵列形状配置（cfg_shape）

> **2026-04-14 新增**。通过写入寄存器 `0x3C` 的低 2 位控制 PE 阵列工作形态。

| 值 | 形态 | 活跃区域 | 活跃 PE 数 | 适用场景 |
|---:|------|---------|----------:|---------|
| `2'b00` | **4×4** | Row[0:3], Col[0:3] | 16 | 小模型、极低功耗 |
| `2'b01` | **8×8** | Row[0:7], Col[0:7] | 64 | 中等规模推理 |
| `2'b10` | **16×16** | 全阵列 | 256 | 大矩阵并行计算 |
| `2'b11` | **8×32（折叠）** | 上半 Row[0:7]+下半 Row[8:15]，各 16 列拼接为逻辑 32 列 | 256 | 宽矩阵/大 N 维度 |

**使用方式**：

```c
// 配置 8×8 模式（在启动运算前设置）
NPU_REG(0x3C) = 0x01;   // cfg_shape = 8x8
NPU_REG(0x00) = mode | 0x01;  // 启动
```

**注意事项**：
- `cfg_shape` 在 `start` 脉冲上升沿被锁存到控制器影子寄存器 `lk_shape`
- 运行期间修改 `cfg_shape` 不会影响当前运算
- 8×32 折叠模式下，上半部 Row 7 Col 15 的激活输出自动路由到下半部 Row 8 Col 0

### 5.1 IRQ Clear 双路径

| 路径 | 方法 | 说明 |
|------|------|------|
| **Path A** | 写 `0x0C` (INT_CLR) bit0=1 | 传统独立清除寄存器 |
| **Path B** | 写 `0x00` (CTRL) bit6=1 | W1C 位，读回始终为 0；适合和 CTRL 写在同一次 AXI 事务 |

两条路径都清除 `npu_ctrl.irq` 输出（`npu_ctrl` 监测 `cfg_irq_clr = ctrl_reg[6]`）。

### 5.2 软件推荐编码

| 功能 | CTRL 值 | 位解码 |
|------|--------:|--------|
| INT8 + WS 启动 | `0x01` | start=1, mode=00, stat=00 |
| INT8 + OS 启动 | `0x11` | start=1, mode=00, stat=01 |
| FP16 + WS 启动 | `0x09` | start=1, mode=10, stat=00 |
| FP16 + OS 启动 | `0x19` | start=1, mode=10, stat=01 |
| IRQ 确认（兼 CTRL 清零） | `0x40` | irq_clr=1 |
| 完全清零 CTRL | `0x00` | 计算结束后建议显式清零 |

---

## 6. 控制器状态机

### 6.1 状态总览

```
S_IDLE
  │  cfg_start_rise：锁存 shadow regs，清 PPBuf，发 tile(0,0) DMA 启动
  ▼
S_WARMUP_LOAD
  │  等待 tile(0,0) W+A DMA 双完成（dma_load_done）
  │  swap PPBuf：Pong→Ping（tile(0,0) 切换到 PE 可读侧）
  ▼
S_WARMUP_WAIT
  │  1 拍 swap 传播；配置 pe_stat / pe_load_w
  │  若非末 tile：发 tile(0,1) 预取 DMA（向 Pong Bank 写下一个 tile）
  ▼
S_PRELOAD
  │  1 拍等待 PPBuf rd_fill 稳定（swap 是时序电路）
  │  pe_en 保持低
  ▼
S_OVERLAP_COMPUTE   ◄──────────────────────────────────────────┐
  │  pe_en=1，PE 从 Ping Bank 计算 tile(i,j)                   │
  │  DMA 同时向 Pong Bank 加载 tile(i,j+1)（预取，已在后台进行）  │
  │  OS 模式：等 w_ppb_empty && a_ppb_empty                    │
  │  WS 模式：等 ws_consume_cnt >= K+2                         │
  ▼
S_DRAIN
  │  pe_flush=1（1 拍），触发 PE 流水线第 1 级排空
  ▼
S_DRAIN2
  │  pe_flush=0（1 拍），流水线第 2 级传播（valid_out 在此拍出现）
  ▼
S_WRITE_BACK
  │  发 dma_r_start 脉冲，将 comp_r_addr 写给 DMA
  │  DMA R 通道从结果 FIFO 取数，发起 AXI4 写事务
  ▼
S_WB_WAIT
  │  等 dma_r_done_r
  │  WB 完成后：
  │    ├─ 若 is_last_tile → S_DONE
  │    ├─ 若预取 DMA 已完成 → swap PPBuf + 发下下一个预取 → S_PRELOAD ──►（上方循环）
  │    └─ 若预取 DMA 未完成 → S_WAIT_PREFETCH
  ▼
S_WAIT_PREFETCH（可选）
  │  等预取 DMA 完成（dma_load_done）
  │  完成后 swap PPBuf + 发下下一个预取 → S_PRELOAD ──►（上方循环）
  ▼
S_DONE
     irq=1，busy=0，done=1，清 PPBuf/FIFO，回到 S_IDLE
```

### 6.2 各状态详细说明

| 状态 | pe_en | pe_flush | 关键动作 |
|------|:-----:|:--------:|---------|
| **S_IDLE** | 0 | 0 | 等待 `cfg_start_rise`；一次性锁存所有 shadow reg；发 tile(0,0) DMA 启动；清 PPBuf/FIFO |
| **S_WARMUP_LOAD** | 0 | 0 | 等 W+A DMA 双完成（`dma_load_done`）；swap PPBuf（Pong→Ping，tile(0,0) 数据移到 PE 可读侧） |
| **S_WARMUP_WAIT** | 0 | 0 | 1 拍 swap 传播；设置 `pe_stat`/`pe_load_w`；若非末 tile 则发 tile(0,1) 预取 DMA |
| **S_PRELOAD** | 0 | 0 | 等 PPBuf `rd_fill` 稳定（swap 时序电路）；1 拍后 PE 才能正确读到数据 |
| **S_OVERLAP_COMPUTE** | 1 | 0 | PE 消费 Ping Bank；DMA 同时向 Pong Bank 写下一 tile；OS=等 PPBuf 双空；WS=计数 K+2 拍 |
| **S_DRAIN** | 1 | 1 | 发 `pe_flush=1`（1 拍）；触发 Stage-2 输出+清零 |
| **S_DRAIN2** | 1 | 0 | flush 传播第 2 拍；`valid_out` 在此拍高；结果写入 result FIFO |
| **S_WRITE_BACK** | 1 | 0 | 发 `dma_r_start` 脉冲；锁存 `comp_r_addr`；DMA 开始 AXI4 写 |
| **S_WB_WAIT** | 1→0 | 0 | 等 `dma_r_done_r`；判断末 tile / 预取状态，决定下一跳 |
| **S_WAIT_PREFETCH** | 0 | 0 | （仅 PE 比 DMA 快时出现）等预取 DMA 完成，再 swap + 预取下一个 + S_PRELOAD |
| **S_DONE** | 0 | 0 | `irq=1`，`done=1`，`busy=0`；清 PPBuf/FIFO；回 S_IDLE |

### 6.3 真 Ping-Pong 重叠时序图（稳态）

```
时钟周期： T0          T1~TX          TX+1    TX+2    TX+3    TX+4
状态：     WU_WAIT     PRELOAD
           OVERLAP_COMPUTE(0,0)        DRAIN  DRAIN2   WB     WB_WAIT
                       OVERLAP         ...    ...
DMA W/A：  prefetch    → Pong bank                     ← tile(0,1) done
PPBuf：    Ping=tile(0,0) → PE读                Pong=tile(0,1) → swap → Ping
PE 计算：              ← tile(0,0) K 拍 →      flush  valid
FIFO：                                                  ← 写入结果
DRAM R：                                                         → 写回 C[0][0]
                                       ↑ WB_WAIT: swap + 发 tile(0,2) 预取
                                                       ↑ S_PRELOAD
                                                              ↑ OVERLAP_COMPUTE(0,1)
```

### 6.4 首个/末个 Tile 的特殊处理

| 场景 | 特殊处理 |
|------|---------|
| **首个 Tile（暖机）** | S_WARMUP_LOAD 等待 DMA 完成后 swap；S_WARMUP_WAIT 发 tile(0,1) 预取 |
| **末个 Tile** | WB 完成后直接进 S_DONE，不发新预取 DMA |
| **单 Tile 层（M=N=1）** | S_WARMUP_WAIT 检测 `is_last_tile` 为真，跳过预取，直接 S_PRELOAD |
| **PE 比 DMA 快** | 计算结束+WB 完成，但预取未到，进 S_WAIT_PREFETCH 等待 |

---

## 7. 地址锁存与层间切换

### 7.1 Shadow Register 机制

在 `cfg_start_rise` 的**同一个时钟周期**，以下寄存器被锁存到影子寄存器中：

```verilog
// npu_ctrl.v 中的锁存逻辑
if (cfg_start_rise) begin
    lk_m_dim  <= m_dim;
    lk_n_dim  <= n_dim;
    lk_k_dim  <= k_dim;
    lk_w_addr <= w_addr;
    lk_a_addr <= a_addr;
    lk_r_addr <= r_addr;
    lk_mode   <= cfg_mode;   // ctrl_reg[3:2]
    lk_stat   <= cfg_stat;   // ctrl_reg[5:4]
end
```

FSM 运行期间**只使用影子寄存器**（`lk_*`）计算地址和模式。这意味着：
- CPU 可以在 NPU 运行时安全地预配置下一层的寄存器
- 不会出现中途地址错乱

### 7.2 层间切换流程

```
第 n 层运行中：
  CPU 可安全写 M_DIM/N_DIM/K_DIM/W_ADDR/A_ADDR/R_ADDR（这些是"活寄存器"，影子寄存器已锁住）

第 n 层完成（S_DONE）：
  irq 拉高 → CPU 中断处理（或轮询 STATUS.done）
  CPU 写 INT_CLR bit0=1 或 CTRL bit6=1 清中断
  CPU 写 CTRL[0]=0（清 start，done sticky 回零）

配置第 n+1 层：
  CPU 写 M_DIM/N_DIM/K_DIM（如果维度变化）
  CPU 写 W_ADDR/A_ADDR/R_ADDR（新层数据地址）
  CPU 写 CTRL = 新 mode/stat | 0x01（start 脉冲）
  → 触发 cfg_start_rise → 新的影子寄存器锁存 → 新层开始
```

### 7.3 abort 机制

任意时刻写 `CTRL[1]=1` 可中止当前运算，FSM 跳回 S_IDLE，`busy=0`。

---

## 8. SoC 地址空间与关键时序约束

### 8.1 地址空间

以默认参数 `MEM_WORDS=1024`（4KB SRAM）、`DRAM_WORDS=15360` 为例：

| 地址范围 | 区域 | 容量 | 说明 |
|---------|------|-----:|------|
| `0x0000_0000 – 0x0000_0FFF` | SRAM | 4 KB | CPU 指令 + 数据，`addr < 0x1000` |
| `0x0000_1000 – 0x0001_FFFF` | DRAM | ~124 KB | CPU + NPU DMA 共用 |
| `0x0200_0000 – 0x0200_003F` | NPU Reg | 64 B | AXI4-Lite 配置空间 |

> ⚠️ **重要边界**：`addr_is_ram = mem_addr < 32'h1000`。`0x0F00` 属于 **SRAM**，不是 DRAM。SoC 测试使用 `0x2000` 作为 PASS 标记地址。

### 8.2 PicoRV32 读时序要求

PicoRV32 要求 **`mem_ready` 与 `mem_rdata` 同周期有效**。因此：

- `soc_mem.rdata`：**组合读**（`assign rdata = mem[addr]`）
- `dram_model.cpu_rdata`：**组合读**（`assign cpu_rdata = mem[cpu_addr>>2]`）

若改为同步寄存器读，CPU 每次访存都读到 stale data，导致指令错乱、NPU 配置偏移、轮询死循环。

### 8.3 AXI-Lite 桥接写时序

`npu_axi_lite` 的写通道为两段式（AW→W 非同拍）：

```
S_IDLE ──awvalid──► S_WRITE_AW ──wvalid──► S_WRITE_W ──► S_IDLE
  (锁存 awaddr)                    (写入寄存器，发 bvalid)
```

- `awready = !aw_q`（只有 AW 未锁存时接收新地址）
- `wready = aw_q`（AW 已锁存后接收数据）

`axi_lite_bridge` 采用匹配的 3-cycle 写 FSM：`S_WRITE_AW → S_WRITE_W → done`。

---

## 9. 能力矩阵与性能估算

### 9.1 当前接入状态

| 能力 | 状态 | 说明 |
|------|------|------|
| INT8 数据通路 | ✅ 已接入 | 完整验证 |
| FP16 数据通路 | ✅ 已接入 | FP32 混合精度累加，完整验证 |
| WS / OS 模式 | ✅ 已接入 | 均已验证 |
| 真 Ping-Pong 重叠 | ✅ 已接入 | 2026-04-13 升级 |
| 地址影子寄存器 | ✅ 已接入 | start 脉冲时锁存，防层间污染 |
| IRQ W1C 清除（CTRL[6]） | ✅ 已接入 | 双路径清中断 |
| SoC 集成 | ✅ 已接入 | PicoRV32 固件驱动通过 |
| tile-loop 地址调度 | ✅ 已接入 | 核心执行模型 |
| AXI 多拍 INCR burst | ✅ 已接入 | `calc_arlen()` 动态计算，读写均支持 |
| **可重配置 PE 阵列 (cfg_shape)** | **✅ 已接入** | **4×4/8×8/16×16/8×32 四形态运行时切换** |
| **双权重寄存器预取** | **✅ 已接入** | **active/prefetch 双寄存器 + swap_w 原子切换** |
| 时钟门控（per-PE） | ✅ 已接入 | `reconfig_pe_array` 内部 per-PE `pe_clk_en` |
| DFS / 全局时钟门控 | ⚠️ 接口存在 | `npu_power` 已实例化，`div_sel=0` bypass；DFS 使用时需 CDC 同步器 |

### 9.2 性能分析

**串行时代（旧 FSM）**单 tile 周期：
```
~K（DMA 读）+ K（计算）+ 固定开销（~7 拍）≈ 2K + 7 拍
```

**真 Ping-Pong（新 FSM）**稳态每 tile 周期（DMA 与 PE 重叠）：
```
max(K_DMA_cycles, K_PE_cycles) + WB_cycles
```
DMA 搬运 K 个元素（INT8: K/4 字，FP16: K/2 字）与 PE 计算 K 拍同时进行，写回（约 2~3 拍）是串行的。带宽受限时实际增益取决于 burst 长度和 DRAM 延迟。

**以 INT8，K=8，4×4 矩阵（M=N=4）为例（新 FSM）：**
- 总 tile 数 = M×N = 16
- PE 计算：K=8 拍/tile
- DMA 读：K/4 = 2 字，多拍 burst 约 3~4 拍
- WB：~3 拍
- 稳态每 tile ≈ max(8, 4) + 3 = 11 拍（旧 FSM: 2×8+7=23 拍）
- **提升约 2×**

---

## 10. 配套文档阅读顺序

| 顺序 | 文档 | 内容 |
|:----:|------|------|
| 1 | `README.md` | 项目总览与快速入口 |
| 2 | `doc/user_manual.md` | 使用指南、固件编程、端到端推理调度流程 |
| 3 | `doc/simulation_guide.md` | 仿真脚本与测试项说明 |
| 4 | `doc/module_reference.md` | 模块级信号与时序细节 |
| 5 | `doc/npu_debug_checklist.md` | Bug 定位与历史修复记录 |
| 6 | `doc/architecture_fix_plan.md` | 架构演进历史 |

---

## 11. 总结

理解本工程最关键的四件事：

1. **数据布局**：B 列主序（`W_ADDR`），A 行主序（`A_ADDR`），C 行主序（`R_ADDR`）
2. **计算模型**：tile-loop 按 `C[i][j]` 单点推进，每 tile = 一个 K 维点积
3. **执行重叠**：真 Ping-Pong FSM，PE 计算 tile(i,j) 的同时 DMA 加载 tile(i,j+1)；start 时锁存 shadow reg，防层间地址污染
4. **可重配置阵列**：16×16 物理 PE 阵列通过 `cfg_shape`(0x3C) 运行时切换 4×4/8×8/16×16/8×32 四种形态；每个 PE 内置双权重寄存器支持预取隐藏

### SoC 约束（同前）

- CPU 读口必须是组合读，AXI-Lite 写必须按 AW→W 两拍完成
