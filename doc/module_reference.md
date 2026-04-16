# NPU RTL 模块说明文档

> 最后更新：2026-04-14
> 说明：本文件按当前 RTL 实现同步（含可重配置 PE 阵列、双权重寄存器预取升级），重点描述"真正接入并参与主流程"的行为。

---

## 目录

1. [阅读约定](#1-阅读约定)
2. [npu_top — NPU 顶层](#2-npu_top--npu-顶层)
3. [npu_axi_lite — 配置寄存器文件](#3-npu_axi_lite--配置寄存器文件)
4. [npu_ctrl — tile-loop 控制器](#4-npu_ctrl--tile-loop-控制器)
5. [npu_dma — 三通道 DMA](#5-npu_dma--三通道-dma)
6. [pingpong_buf — 双 Bank Ping-Pong 缓冲](#6-pingpong_buf--双-bank-ping-pong-缓冲)
7. [reconfig_pe_array — 可重配置 PE 阵列](#7-reconfig_pe_array--可重配置-pe-阵列)
8. [pe_top — 单个处理单元（双权重寄存器版）](#8-pe_top--单个处理单元双权重寄存器版)
9. [pe_array — 旧版 PE 阵列封装](#9-pe_array--旧版-pe-阵列封装)
10. [npu_power — 电源与时钟管理](#10-npu_power--电源与时钟管理)
11. [SoC 模块](#11-soc-模块)
12. [废弃与保留模块](#12-废弃与保留模块)
13. [模块关系速查](#13-模块关系速查)

---

## 1. 阅读约定

### 1.1 数据类型

| 标识 | 含义 | DRAM 打包 |
|------|------|-----------|
| INT8 | 有符号 8-bit 整数 | 4 元素/32-bit 字 |
| FP16 | IEEE 754 半精度浮点 | 2 元素/32-bit 字 |
| FP32 | 内部累加精度 | 1 元素/32-bit 字（结果） |

`CTRL[3:2]` 编码：`00` = INT8，`10` = FP16（硬件内部只区分非零 vs 零）

### 1.2 Tile-loop 约定

所有主测试遵循：

| 矩阵 | 布局 | 地址 |
|------|------|------|
| B（权重）`[K×N]` | 列主序 | `W_ADDR + j * k_bytes` |
| A（激活）`[M×K]` | 行主序 | `A_ADDR + i * k_bytes` |
| C（结果）`[M×N]` | 行主序 | `R_ADDR + (i*N+j) * 4` |

控制器按 `C[i][j]` 单个输出元素推进 tile，外层循环 `M×N` 次。

---

## 2. `npu_top` — NPU 顶层

- **文件**：`rtl/top/npu_top.v`
- **功能**：集成并连接所有 NPU 子模块：寄存器文件、控制器、DMA、双 PPBuf、PE 阵列、结果 FIFO、电源管理。

### 2.1 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `PHY_ROWS` | 16 | PE 阵列物理行数 |
| `PHY_COLS` | 16 | PE 阵列物理列数 |
| `DATA_W` | 16 | PE 输入位宽（INT8/FP16 均用 16-bit 接口） |
| `ACC_W` | 32 | 累加结果位宽 |
| `PPB_DEPTH` | 64 | Ping-Pong 每 Bank 32-bit 字深度 |
| `PPB_THRESH` | 16 | PPBuf 早启动阈值（字数） |

> **2026-04-14 升级**：参数名从 `ROWS/COLS`（逻辑尺寸）改为 `PHY_ROWS/PHY_COLS`（物理尺寸），实际工作区域由 `cfg_shape` 运行时控制。

### 2.2 接口信号

#### 系统接口

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `sys_clk` | input | 1 | 系统时钟（AXI 域 + DMA 控制域） |
| `sys_rst_n` | input | 1 | 低有效同步复位 |
| `npu_irq` | output | 1 | 计算完成中断（来自 `npu_axi_lite`） |

#### AXI4-Lite 从机（CPU 配置端口）

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `s_axi_awaddr` | input | 32 | 写地址 |
| `s_axi_awvalid` | input | 1 | 写地址有效 |
| `s_axi_awready` | output | 1 | 写地址就绪 |
| `s_axi_wdata` | input | 32 | 写数据 |
| `s_axi_wstrb` | input | 4 | 字节使能 |
| `s_axi_wvalid` | input | 1 | 写数据有效 |
| `s_axi_wready` | output | 1 | 写数据就绪 |
| `s_axi_bresp` | output | 2 | 写响应（固定 OKAY） |
| `s_axi_bvalid` | output | 1 | 写响应有效 |
| `s_axi_bready` | input | 1 | 写响应就绪 |
| `s_axi_araddr` | input | 32 | 读地址 |
| `s_axi_arvalid` | input | 1 | 读地址有效 |
| `s_axi_arready` | output | 1 | 读地址就绪 |
| `s_axi_rdata` | output | 32 | 读数据 |
| `s_axi_rresp` | output | 2 | 读响应（固定 OKAY） |
| `s_axi_rvalid` | output | 1 | 读数据有效 |
| `s_axi_rready` | input | 1 | 读数据就绪 |

#### AXI4 主机（DMA 数据端口）

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `m_axi_awaddr` | output | 32 | 写地址 |
| `m_axi_awlen` | output | 8 | burst 长度（beats-1） |
| `m_axi_awsize` | output | 3 | beat 字节宽度编码 |
| `m_axi_awburst` | output | 2 | burst 类型（`01`=INCR） |
| `m_axi_awvalid` | output | 1 | |
| `m_axi_awready` | input | 1 | |
| `m_axi_wdata` | output | ACC_W | 写数据（32-bit） |
| `m_axi_wstrb` | output | ACC_W/8 | 字节使能 |
| `m_axi_wlast` | output | 1 | 最后一拍标志 |
| `m_axi_wvalid` | output | 1 | |
| `m_axi_wready` | input | 1 | |
| `m_axi_bresp` | input | 2 | |
| `m_axi_bvalid` | input | 1 | |
| `m_axi_bready` | output | 1 | |
| `m_axi_araddr` | output | 32 | 读地址 |
| `m_axi_arlen` | output | 8 | burst 长度（beats-1，动态计算） |
| `m_axi_arsize` | output | 3 | |
| `m_axi_arburst` | output | 2 | |
| `m_axi_arvalid` | output | 1 | |
| `m_axi_arready` | input | 1 | |
| `m_axi_rdata` | input | ACC_W | 读数据 |
| `m_axi_rresp` | input | 2 | |
| `m_axi_rvalid` | input | 1 | |
| `m_axi_rready` | output | 1 | |
| `m_axi_rlast` | input | 1 | burst 最后一拍 |

### 2.3 关键内部信号

| 信号 | 位宽 | 说明 |
|------|-----:|------|
| `ctrl_target_col` | `$clog2(COLS)` | OS 模式：本 tile 权重路由到的目标列索引 |
| `pe_consume` | 1 | PE 消费使能（`pe_en && (pe_data_ready || pe_flush)`） |
| `ppb_rd_active` | 1 | `= pe_en`，PPBuf 读使能的门控 |
| `w/a_ppb_phase` | 1 | FP16 模式子字节装配状态（0=低字节，1=高字节） |
| `npu_clk_out` | 1 | 来自 `npu_power` 的 DFS 分频时钟，驱动 PE 阵列 |
| `row_clk_gated` | ROWS | 来自 `npu_power` 的行门控时钟（当前预留为后端 ICG 使用） |

### 2.4 FP16 子字节装配机制

PPBuf 读侧输出 8-bit 子字节，`npu_top` 内部用两级状态机将连续两个 8-bit 子字节拼成一个 16-bit FP16 值再送入 PE：

```
PPBuf 输出 8-bit/拍：
  拍 2k:   FP16[7:0]  → 存入 w_fp16_shift[7:0]
  拍 2k+1: FP16[15:8] → 存入 w_fp16_shift[15:8]
             ↓ w_ppb_phase 1→0 边沿
           pe_data_ready 置 1 → pe_consume
```

INT8 模式：每拍一个 8-bit 直接送 PE（无需装配，`pe_data_ready = w_int8_ready_d && a_int8_ready_d`）。

---

## 3. `npu_axi_lite` — 配置寄存器文件

- **文件**：`rtl/axi/npu_axi_lite.v`
- **功能**：AXI4-Lite 从机接口 + NPU 控制/状态寄存器组，提供 CPU→NPU 的配置通路和 NPU→CPU 的状态/中断反馈。

### 3.1 接口信号

#### AXI4-Lite 从机（与 npu_top 端口同名）

（见第 2.2 节 AXI4-Lite 部分，此处省略重复列举）

#### NPU 控制输出

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `ctrl_reg` | output | 32 | CTRL 寄存器（`[0]=start, [1]=abort, [3:2]=mode, [5:4]=stat_mode`） |
| `cfg_shape` | output | 2 | PE 阵列形状配置（见第 7 节 reconfig_pe_array） |
| `m_dim` | output | 32 | 矩阵 M 维度 |
| `n_dim` | output | 32 | 矩阵 N 维度 |
| `k_dim` | output | 32 | 矩阵 K 维度 |
| `w_addr` | output | 32 | 权重 DRAM 基地址 |
| `a_addr` | output | 32 | 激活 DRAM 基地址 |
| `r_addr` | output | 32 | 结果 DRAM 基地址 |
| `arr_cfg` | output | 8 | `[3:0]=act_rows, [7:4]=act_cols`（保留） |
| `clk_div` | output | 3 | DFS 分频选择，接 `npu_power.div_sel` |
| `cg_en` | output | 1 | 门控使能（保留） |

#### NPU 状态输入

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `status_busy` | input | 1 | 来自 `npu_ctrl.busy` |
| `status_done` | input | 1 | 来自 `npu_ctrl.done` |
| `irq_flag` | input | 1 | 来自 `npu_ctrl.irq`（运算完成脉冲） |
| `npu_irq` | output | 1 | 中断输出（到 SoC 中断控制器） |

### 3.2 寄存器映射

| 偏移 | 名称 | 位域 | 读/写 | 说明 |
|-----:|------|------|-------|------|
| 0x00 | CTRL | `[0]` start | R/W | 写 1 启动；控制器检测上升沿 |
| | | `[1]` abort | R/W | 写 1 中止当前运算 |
| | | `[3:2]` data_mode | R/W | `00`=INT8，`10`=FP16 |
| | | `[5:4]` stat_mode | R/W | `00`=WS，`01`=OS |
| 0x04 | STATUS | `[0]` busy | RO | 运算进行中 |
| | | `[1]` done | RO | 运算完成（sticky，软件清 start 后回零） |
| 0x08 | INT_EN | `[0]` int_en | R/W | 中断使能 |
| 0x0C | INT_CLR | `[0]` | W1C | 写 1 清中断 pending |
| 0x10 | M_DIM | `[31:0]` | R/W | 矩阵 M 维度 |
| 0x14 | N_DIM | `[31:0]` | R/W | 矩阵 N 维度 |
| 0x18 | K_DIM | `[31:0]` | R/W | 矩阵 K 维度（= 内积长度） |
| 0x20 | W_ADDR | `[31:0]` | R/W | 权重基地址（字节对齐） |
| 0x24 | A_ADDR | `[31:0]` | R/W | 激活基地址 |
| 0x28 | R_ADDR | `[31:0]` | R/W | 结果基地址 |
| 0x30 | ARR_CFG | `[7:0]` | R/W | 阵列配置（保留，未接入主流程） |
| 0x34 | CLK_DIV | `[2:0]` | R/W | DFS 分频选择（接 npu_power） |
| 0x38 | CG_EN | `[0]` | R/W | 门控使能（保留） |
| **0x3C** | **CFG_SHAPE** | **`[1:0]`** | **R/W** | **PE 阵列形状配置：`00`=4×4, `01`=8×8, `10`=16×16, `11`=8×32（折叠模式）** |
| 0x38 | CG_EN | `[0]` | R/W | 门控使能（保留） |

> 未定义地址读回 `0xDEADBEEF`。

### 3.3 写时序（AW→W 两阶段）

```
       CLK  ___/‾\_/‾\_/‾\_/‾\_/‾\_/‾\___
   AWVALID  _____/‾‾‾‾‾\_______
   AWREADY  _________/‾‾‾\____
    WVALID  _____________/‾‾‾‾‾\____
    WREADY  _____________/‾‾‾‾‾\____
    BVALID  _________________/‾\___
    BREADY  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

  ①  AW 握手（awvalid & awready）：锁存 awaddr → aw_q=1
  ②  W  握手（wvalid & wready， wready=aw_q=1）：写入寄存器
  ③  B  握手：同周期发出 bvalid，等待 bready
```

- `awready = !aw_q`（只有在 AW 未锁存时才接收新地址）
- `wready  = aw_q` （只有在 AW 已锁存后才接收数据）

### 3.4 读时序（单拍）

```
      CLK  ___/‾\_/‾\_/‾\_/‾\___
  ARVALID  _____/‾‾‾\_______
  ARREADY  _____/‾‾‾\_______   (组合 arready，与 arvalid 同周期成立)
   RVALID  _________/‾‾‾\___
   RREADY  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

  AR 握手同拍返回 RVALID+RDATA（零等待读）
```

---

## 4. `npu_ctrl` — tile-loop 控制器（真 Ping-Pong 版）

- **文件**：`rtl/ctrl/npu_ctrl.v`
- **功能**：驱动矩阵乘法 tile-loop 调度，实现 DMA 加载与 PE 计算的真正重叠（Ping-Pong Overlap）。每次迭代计算 `C[i][j]` 的一个元素，控制 DMA 启停、PPBuf swap/clear、PE en/flush，并管理中断清除和地址影子寄存器。

> **架构升级记录（2026-04-13）**：原串行 FSM（IDLE→LOAD→PRELOAD→COMPUTE→DRAIN→WB→NEXT_TILE）已替换为真 Ping-Pong 重叠 FSM，DMA 加载下一 tile 与 PE 计算当前 tile 同步进行。

### 4.1 核心概念：Tile-Based 调度 + Ping-Pong 重叠

**Tile 定义**：一个 tile 代表一个输出元素 `C[i][j]` 的完整计算任务。

**执行管线三阶段**：
```
Phase 0 — Warm-up（暖机）：
    加载 tile(0,0) 到 Pong Bank → swap 到 Ping Bank
    发起 tile(0,1) 预取到 Pong Bank

Phase 1 — Overlap 稳态（每个非末 tile）：
    Ping Bank → PE 计算 tile(i,j)
    Pong Bank → DMA 同时加载 tile(i,j+1)
    WB 完成后：swap，Pong→Ping，发下一预取

Phase 2 — Drain（末 tile）：
    PE 计算最后一个 tile，DMA 写回，无新预取
    完成后 irq 拉高，进 S_DONE
```

**Tile 状态寄存器**：
- `tile_i`：当前输出行索引（0 ≤ tile_i < M，使用影子 `lk_m_dim`）
- `tile_j`：当前输出列索引（0 ≤ tile_j < N，使用影子 `lk_n_dim`）

**地址公式**：
```
权重地址 = lk_w_addr + tile_j × tile_len     (tile_len = K × 元素字节数)
激活地址 = lk_a_addr + tile_i × tile_len
结果地址 = lk_r_addr + (tile_i × N + tile_j) × 4
```

### 4.2 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `ROWS` | 4 | 逻辑 PE 行数（tile-loop 循环使用） |
| `COLS` | 4 | 逻辑 PE 列数（OS 模式 target_col 范围） |
| `DATA_W` | 16 | PE 数据位宽（用于计算 DMA 字节数） |
| `ACC_W` | 32 | 结果宽度（`dma_r_len = 4 bytes`） |

### 4.3 接口信号

#### 配置输入（来自 npu_axi_lite，Live 信号，仅在 start 时采样）

| 信号 | 位宽 | 说明 |
|------|-----:|------|
| `cfg_shape_in` | 2 | PE 阵列形状（`00`=4×4, `01`=8×8, `10`=16×16, `11`=8×32），在 start 时锁存为 `lk_shape` |
| `ctrl_reg` | 32 | `[0]`=start, `[1]`=abort, `[3:2]`=mode, `[5:4]`=stat, `[6]`=irq_clr |
| `m_dim` | 32 | 矩阵 M 维度 |
| `n_dim` | 32 | 矩阵 N 维度 |
| `k_dim` | 32 | 矩阵 K 维度 |
| `w_addr` | 32 | 权重基地址 |
| `a_addr` | 32 | 激活基地址 |
| `r_addr` | 32 | 结果基地址 |
| `arr_cfg` | 8 | 阵列配置（当前未消费） |

#### 影子寄存器（FSM 运行期间使用的锁存副本）

| 影子寄存器 | 对应 Live 信号 | 锁存时机 |
|-----------|------------|---------|
| `lk_shape` | `cfg_shape_in` | `cfg_start_rise` |
| `lk_m_dim` | `m_dim` | `cfg_start_rise` |
| `lk_n_dim` | `n_dim` | `cfg_start_rise` |
| `lk_k_dim` | `k_dim` | `cfg_start_rise` |
| `lk_w_addr` | `w_addr` | `cfg_start_rise` |
| `lk_a_addr` | `a_addr` | `cfg_start_rise` |
| `lk_r_addr` | `r_addr` | `cfg_start_rise` |
| `lk_mode` | `ctrl_reg[3:2]` | `cfg_start_rise` |
| `lk_stat` | `ctrl_reg[5:4]` | `cfg_start_rise` |

#### 状态输出（到 npu_axi_lite）

| 信号 | 位宽 | 说明 |
|------|-----:|------|
| `busy` | 1 | 运算进行中（非 S_IDLE 均为高） |
| `done` | 1 | sticky 完成信号（CPU 清 start=0 后回零） |
| `irq` | 1 | 层完成中断，CPU 写 INT_CLR bit0=1 或 CTRL bit6=1 清除 |

#### DMA 控制接口

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `dma_w_start` | output | 1 | 权重 DMA 启动脉冲（单拍高）|
| `dma_w_done` | input | 1 | 权重 DMA 完成（锁存到 `dma_w_done_r`） |
| `dma_w_addr` | output | 32 | 权重 DRAM 地址 |
| `dma_w_len` | output | 16 | 权重 DMA 字节数 |
| `dma_a_start` | output | 1 | 激活 DMA 启动脉冲 |
| `dma_a_done` | input | 1 | 激活 DMA 完成 |
| `dma_a_addr` | output | 32 | 激活 DRAM 地址 |
| `dma_a_len` | output | 16 | 激活 DMA 字节数 |
| `dma_r_start` | output | 1 | 结果 DMA 启动脉冲 |
| `dma_r_done` | input | 1 | 结果 DMA 完成 |
| `dma_r_addr` | output | 32 | 结果 DRAM 地址（`comp_r_addr`） |
| `dma_r_len` | output | 16 | 结果 DMA 字节数（固定 `4`） |

#### PE 控制接口

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `pe_en` | output | 1 | PE 流水线使能 |
| `pe_flush` | output | 1 | 累加器 flush（OS: 带末 beat 数据；WS: 纯触发输出 ws_acc） |
| `pe_mode` | output | 1 | `0=INT8, 1=FP16`（来自 lk_mode） |
| `pe_stat` | output | 1 | `0=WS, 1=OS`（来自 lk_stat） |
| `pe_load_w` | output | 1 | WS 模式：weight_reg 锁存脉冲（ws_consume_cnt < K 时持续高） |
| `pe_swap_w` | output | 1 | WS 模式：双权重寄存器原子交换脉冲（预取隐藏用） |

#### PPBuf 控制接口

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `w_ppb_ready` | input | 1 | 权重 PPBuf 读侧已达阈值 |
| `w_ppb_empty` | input | 1 | 权重 PPBuf 读侧为空 |
| `a_ppb_ready` | input | 1 | 激活 PPBuf 读侧已达阈值 |
| `a_ppb_empty` | input | 1 | 激活 PPBuf 读侧为空 |
| `w_ppb_swap` | output | 1 | 权重 PPBuf bank 切换脉冲（单拍高）|
| `a_ppb_swap` | output | 1 | 激活 PPBuf bank 切换脉冲 |
| `w_ppb_clear` | output | 1 | 权重 PPBuf 全部指针复位（单拍高） |
| `a_ppb_clear` | output | 1 | 激活 PPBuf 全部指针复位 |
| `r_fifo_clear` | output | 1 | 结果 FIFO 复位脉冲 |

### 4.4 FSM 状态与转移（真 Ping-Pong 版）

#### 状态编码

| 宏名 | 编码 | 说明 |
|------|-----:|------|
| `S_IDLE` | 4'd0 | 空闲，等待 start |
| `S_WARMUP_LOAD` | 4'd1 | 暖机：等待 tile(0,0) DMA 完成 |
| `S_WARMUP_WAIT` | 4'd2 | swap 传播 + 发 tile(0,1) 预取 |
| `S_PRELOAD` | 4'd9 | PPBuf swap 时序稳定等待（1 拍） |
| `S_OVERLAP_COMPUTE` | 4'd3 | PE 计算 + DMA 后台预取 |
| `S_DRAIN` | 4'd4 | pe_flush=1，第 1 拍 |
| `S_DRAIN2` | 4'd5 | pe_flush=0，第 2 拍（valid_out 在此出现） |
| `S_WRITE_BACK` | 4'd6 | 发 dma_r_start |
| `S_WB_WAIT` | 4'd7 | 等 dma_r_done，判断下跳 |
| `S_WAIT_PREFETCH` | 4'd10 | 等预取 DMA 到位（PE 快于 DMA 时） |
| `S_DONE` | 4'd8 | irq 拉高，回 S_IDLE |

#### 状态转移图

```
                     cfg_start_rise
  S_IDLE  ──────────────────────────────────────────► S_WARMUP_LOAD
    ▲                                                       │
    │   S_DONE → S_IDLE（同拍）                            │ dma_load_done → swap
    │                                                       ▼
  S_DONE ◄─── is_last_tile ───── S_WB_WAIT ◄─── S_WRITE_BACK ◄── S_DRAIN2 ◄── S_DRAIN
                                      │                              ▲
                      dma_load_done ──┤──────────────────────────────┤
                      (预取已完成)    │swap+prefetch  S_OVERLAP_COMPUTE
                                      │                    ▲
                  !dma_load_done ──►  S_WAIT_PREFETCH      │ (下拍)
                                      │                    │
                                      └──dma_done──► S_PRELOAD ◄── S_WARMUP_WAIT
                                                          (来自 S_WB_WAIT / S_WAIT_PREFETCH)
```

#### 各状态行为表

| 状态 | pe_en | pe_flush | 核心动作 |
|------|:-----:|:--------:|---------|
| S_IDLE | 0 | 0 | 等 `cfg_start_rise`；锁存 shadow reg；清 PPBuf/FIFO；发 tile(0,0) DMA start（`dma_w_addr=w_addr`, `dma_a_addr=a_addr`） |
| S_WARMUP_LOAD | 0 | 0 | 等 `dma_load_done`（W+A 双完成）；swap PPBuf（Pong→Ping）；清 `dma_*_done_r` |
| S_WARMUP_WAIT | 0 | 0 | 设 `pe_stat`/`pe_load_w`；若 `!is_last_tile`：发 tile(0,1) 预取 DMA |
| S_PRELOAD | 0 | 0 | 等 1 拍，让 `rd_fill` 在 swap 后稳定 |
| S_OVERLAP_COMPUTE | 1 | 0 | PE 消费 Ping Bank；OS：等 `w_ppb_empty && a_ppb_empty`；WS：`ws_consume_cnt < K+2` 时持续计数 |
| S_DRAIN | 1 | 1 | `pe_flush=1`（触发 Stage-2 输出）|
| S_DRAIN2 | 1 | 0 | `pe_flush=0`（`valid_out` 在此拍出现，结果进 FIFO）|
| S_WRITE_BACK | 1 | 0 | 发 `dma_r_start` 脉冲，锁存 `comp_r_addr`，清 `dma_r_done_r` |
| S_WB_WAIT | 1→0 | 0 | 等 `dma_r_done_r`；末 tile→S_DONE；非末 tile+预取完成→swap+发下一预取+S_PRELOAD；预取未完→S_WAIT_PREFETCH |
| S_WAIT_PREFETCH | 0 | 0 | 等 `dma_load_done`；完成后 swap+发下一预取+S_PRELOAD |
| S_DONE | 0 | 0 | `irq=1`，`done=1`，`busy=0`；清 PPBuf/FIFO；回 S_IDLE |

### 4.5 时序图：稳态 Ping-Pong 重叠

```
时钟周期：  T0     T1         T2~T(K)     T(K+1)   T(K+2)   T(K+3)   T(K+4)
状态：     [WU_WAIT][PRELOAD][OVERLAP...] [DRAIN]  [DRAIN2] [WB]     [WB_WAIT]
pe_en：      0       0        1            1        1         1        1→0
pe_flush：   0       0        0            1        0         0        0
DMA(Pong)： prefetch → Pong Bank（tile i,j+1）----完成----|
PPBuf：    Ping=tile(i,j) → PE读取         Pong=tile(i,j+1) 就绪
结果 FIFO：                              写入      valid
DRAM 写：                                                   dma_r_start  → 写回 C[i][j]
```

### 4.6 IRQ 机制

```
S_DONE：irq <= 1'b1;

CPU 清除方法：
  Path A: 写 0x0C（INT_CLR） bit0=1  → npu_axi_lite 清 int_pending
  Path B: 写 0x00（CTRL）    bit6=1  → cfg_irq_clr=1 → npu_ctrl: irq <= 1'b0

done sticky：仅当 CPU 写 CTRL[0]=0 时回零
  if (!cfg_start) done <= 1'b0;
```

### 4.7 关键实现说明

- **`cfg_start_rise`**：`cfg_start && !cfg_start_d1`，防止 start 保持为 1 时重复触发。
- **`dma_load_done`**：`dma_w_done_r && dma_a_done_r`，W 和 A 通道均完成时才为真。
- **`is_last_tile`**：`(tile_i == lk_m_dim-1) && (tile_j == lk_n_dim-1)`，决定是否还需要预取。
- **`ws_consume_cnt`**：WS 模式下计数 PE 消费拍数，`>= K+2` 时退出 `S_OVERLAP_COMPUTE`（K 拍数据 + 2 拍流水线排空）。
- **`S_PRELOAD`**：swap 是时序电路，swap 脉冲后需 1 拍才能让 rd_sel 和 rd_fill 稳定，`pe_en` 在此拍保持低。
- **`target_col`**：OS 模式下等于 `tile_j % COLS`，位宽为 `$clog2(COLS)`（COLS=4 时 2-bit），在 npu_top 中路由权重到目标列。

---

## 5. `npu_dma` — 三通道 DMA

- **文件**：`rtl/axi/npu_dma.v`
- **功能**：AXI4 主机 DMA，管理权重读、激活读、结果写回三条独立通道，共用一条 AXI4 总线（时分复用）。

### 5.1 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `DATA_W` | 32 | AXI 数据总线宽度（字节） |
| `PE_DATA_W` | 16 | PE 数据位宽（用于 PPBuf 接口） |
| `BURST_MAX` | 16 | 最大 AXI burst 长度（beats） |
| `PPB_DEPTH` | 64 | PPBuf 深度（用于地址计算） |
| `PPB_THRESH` | 16 | PPBuf 早启动阈值 |
| `R_FIFO_DEPTH` | 64 | 结果 FIFO 深度 |

### 5.2 接口信号

#### 控制通道（来自 npu_ctrl）

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `w_start` | input | 1 | 权重 DMA 启动脉冲 |
| `w_base_addr` | input | 32 | 权重读起始地址 |
| `w_len_bytes` | input | 16 | 权重读字节数 |
| `w_done` | output | 1 | 权重 DMA 完成（下一拍高） |
| `w_ppb_wr_en` | output | 1 | 写 PPBuf 使能 |
| `w_ppb_wr_data` | output | DATA_W | 写 PPBuf 数据（AXI rdata） |
| `w_ppb_full` | input | 1 | PPBuf 写侧满 |
| `w_ppb_buf_ready` | input | 1 | PPBuf 达到阈值 |
| `w_ppb_buf_empty` | input | 1 | PPBuf 读侧空 |
| `a_start / a_done / ...` | — | — | 激活通道，结构同 W 通道 |
| `r_start` | input | 1 | 结果 DMA 启动脉冲 |
| `r_base_addr` | input | 32 | 结果写起始地址 |
| `r_len_bytes` | input | 16 | 结果写字节数 |
| `r_done` | output | 1 | 结果 DMA 完成 |
| `r_fifo_wr_en` | input | 1 | PE→结果 FIFO 写使能 |
| `r_fifo_din` | input | DATA_W | PE 结果数据 |
| `r_fifo_full` | output | 1 | 结果 FIFO 满 |

#### AXI4 主机（见第 2.2 节，信号完全透传至 npu_top）

### 5.3 状态机

```
  IDLE ──w_start──► W_READ ──w_all_done──► (a_len>0 ? A_READ : IDLE)
       ──a_start──► A_READ ──a_all_done──► IDLE
       ──r_start──► (r_pending=1; FIFO非空时进入) R_WRITE ──写完──► IDLE

  W_READ/A_READ 内部 AR 子状态：
    !ar_sent: 发出 arvalid（arlen=calc_arlen），等 arready → ar_sent=1
    ar_sent:  等 rvalid 数据拍；rlast 后判断剩余量 →
                有剩余: ar_sent=0，更新 addr_cnt，发下一个 AR
                无剩余: 通道完成
```

### 5.4 读通道多拍 INCR Burst 时序

```
  CLK     _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_
  arvalid  _/‾‾‾\___________________________
  arready  ___/‾\_________________________
  arlen    ═[N-1]═══════════════════════════
  rvalid   _______/‾‾‾\/‾‾‾\/‾‾‾\...  /‾\___
  rready   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾...‾‾‾‾‾‾
  rlast    _________________________...  /‾\___
  ar_sent  ____/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾...‾‾\_____

  ① arvalid 高，同拍 arlen=N-1（N = min(剩余字数, BURST_MAX)）
  ② arready 握手后 ar_sent=1，arvalid 撤销
  ③ 连续 N 拍 rvalid 数据；rlast 后 ar_sent=0
  ④ 如有剩余数据，更新 addr_cnt，重新发 arvalid
```

### 5.5 结果写通道时序

```
  CLK     _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_
  awvalid  _/‾‾\______________________________
  awready  _/‾‾\____                          (FIFO非空后进入R_WRITE)
  wvalid   _____/‾‾‾\/‾‾‾\/‾‾‾\...  /‾\______
  wlast    ________________________...  /‾\____
  bvalid   _____________________________/‾\___
  r_done   ________________________________/‾\_

  awlen = r_burst_len = (r_len_bytes / DATA_BYTES) - 1
  wdata 直接来自 FIFO 组合读口（rd_data），无额外延迟
```

### 5.6 关键实现说明

- **`r_pending` 机制**：收到 `r_start` 时先锁存地址和长度并置 `r_pending=1`；仅当结果 FIFO 非空时才进入 `R_WRITE`，防止 PE flush 后结果尚未写入 FIFO 时误判完成。
- **`wdata` 组合连线**：`m_axi_wdata` 直接 assign 自 FIFO 的 `rd_data`（组合读），不经寄存器，避免"第一拍 stale data"。
- **`calc_arlen()` function**：`arlen = min(ceil(remaining_bytes / DATA_BYTES), BURST_MAX) - 1`，每次 AR 事务动态计算。
- **`ar_sent` 标志**：AR 握手完成后置 1，防止在等待数据拍期间重复发出 arvalid。

---

## 6. `pingpong_buf` — 双 Bank Ping-Pong 缓冲

- **文件**：`rtl/buf/pingpong_buf.v`
- **功能**：两个独立 SRAM Bank（BufA / BufB），DMA 侧写入 32-bit 宽字，PE 侧按子字宽度读出，实现 DMA 加载与 PE 计算的流水线重叠（overlap）。

### 6.1 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `DATA_W` | 32 | 写入位宽（DMA 侧，32-bit 字） |
| `DEPTH` | 64 | 每 Bank 字深度（需为 2 的幂） |
| `OUT_WIDTH` | 8 | 读出位宽（PE 侧，INT8 模式 8-bit） |
| `THRESHOLD` | 16 | `buf_ready` 阈值（DMA 写入多少字后 PE 可开始读） |
| `SUBW` | 4 | `DATA_W / OUT_WIDTH`（每字子字数；INT8: 4，FP16: 2） |

> 当前 `npu_top` 实例化时固定 `OUT_WIDTH=8, SUBW=4`（INT8 正确性关键）。FP16 的字节分解同样在 8-bit 粒度进行，由 `npu_top` 再拼为 16-bit。

### 6.2 接口信号

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `clk` | input | 1 | 系统时钟 |
| `rst_n` | input | 1 | 低有效同步复位 |
| `wr_en` | input | 1 | DMA 写使能 |
| `wr_data` | input | DATA_W | DMA 写数据 |
| `rd_en` | input | 1 | PE 读使能 |
| `rd_data` | output | OUT_WIDTH | PE 读出子字（空时输出 0） |
| `swap` | input | 1 | bank 切换脉冲（单拍高） |
| `clear` | input | 1 | 复位所有指针（单拍高） |
| `buf_empty` | output | 1 | PE 侧当前 bank 无数据可读 |
| `buf_full` | output | 1 | DMA 侧当前 bank 已满 |
| `buf_ready` | output | 1 | DMA 侧写入量 ≥ THRESHOLD |
| `rd_fill` | output | `$clog2(DEPTH*SUBW)+1` | PE 侧剩余可读子字数 |
| `wr_fill` | output | `$clog2(DEPTH)+1` | DMA 侧已写入字数 |

### 6.3 Bank 切换机制

```
  初始:  wr_sel=0 (DMA→BufA),  rd_sel=0 (PE←BufA)  [初始空，等 DMA 填充]

  一次完整的 Ping-Pong 切换流程：

  1. DMA 写满 BufA（wr_sel=0），PE 仍在处理上一 Bank
  2. npu_ctrl 检测 buf_ready=1，发出 swap 脉冲
  3. swap:  wr_sel: 0→1 (DMA→BufB),  rd_sel: 0→1 (PE←BufA 旧数据)
  4. DMA 填充 BufB，PE 消费 BufA
  5. PE 消费完 BufA（buf_empty=1），下次 swap 后 PE 读 BufB
```

### 6.4 子字读取时序

```
  CLK    _/‾\_/‾\_/‾\_/‾\_/‾\_
  rd_en  _/‾‾‾‾‾‾‾‾‾‾‾‾‾‾\____
  rd_sub  ═[0]═[1]═[2]═[3]═════
  rd_data ═[B0]=[B1]=[B2]=[B3]═   (一个 32-bit 字中的 4 个字节)
            ↑rd_ptr 不变    ↑rd_ptr+1 (rd_sub 回 0)

  一个 32-bit 字依次输出 4 个 8-bit 子字（SUBW=4，OUT_WIDTH=8）
```

---

## 7. `reconfig_pe_array` — 可重配置 PE 阵列

> **2026-04-14 新增模块**，替代旧版 `pe_array.v`。`npu_top.v` 已切换为实例化此模块。

- **文件**：`rtl/array/reconfig_pe_array.v`
- **功能**：物理 **16×16** WS 模式 PE 阵列，通过 `cfg_shape` 运行时控制四种工作形态：
  - **4×4**：仅左上角 16 个 PE 工作，其余时钟门控关闭
  - **8×8**：仅左上角 64 个 PE 工作，其余时钟门控关闭
  - **16×16**：全阵列工作
  - **8×32 折叠**：将 16×16 物理阵列对折为逻辑 8×32 阵列

### 7.1 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `PHY_ROWS` | 16 | 物理行数 |
| `PHY_COLS` | 16 | 物理列数 |
| `DATA_W` | 16 | 数据位宽 |
| `ACC_W` | 32 | 累加结果位宽 |

### 7.2 接口信号

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `clk` | input | 1 | 系统时钟 |
| `rst_n` | input | 1 | 低有效同步复位 |
| `cfg_shape` | input | 2 | 形状控制（见下表） |
| `mode` | input | 1 | `0=INT8, 1=FP16` |
| `stat_mode` | input | 1 | `0=WS, 1=OS` |
| `en` | input | 1 | 全局 PE 使能 |
| `flush` | input | 1 | 全局累加器 flush |
| `load_w` | input | 1 | WS 权重锁存脉冲 |
| `swap_w` | input | 1 | 双权重寄存器交换脉冲 |
| `w_in` | input | `PHY_COLS*DATA_W` | 列广播权重输入（WS） |
| `act_in` | input | `PHY_ROWS*DATA_W` | 行激活输入 |
| `acc_in` | input | `PHY_COLS*ACC_W` | 列顶端部分和（固定接 0） |
| `acc_out` | output | `32*ACC_W` | 累加结果（8×32 模式下最大 32 列输出） |
| `valid_out` | output | 32 | 各列结果有效标志 |
| `pe_active` | output | `PHY_ROWS*PHY_COLS` | 每个 PE 的活跃状态（调试用） |

### 7.3 cfg_shape 形状编码

| 值 | 名称 | 工作区域 | 活跃 PE 数量 | 说明 |
|---:|------|---------|-------------:|------|
| `2'b00` | **4×4** | Row[0:3], Col[0:3] | 16 | 最小配置，适合低功耗场景 |
| `2'b01` | **8×8** | Row[0:7], Col[0:7] | 64 | 中等规模 |
| `2'b10` | **16×16** | Row[0:15], Col[0:15] | 256 | 全阵列并行 |
| `2'b11` | **8×32（折叠）** | 上半 Row[0:7] + 下半 Row[8:15]，各 16 列 | 256 | 将 16×16 对折为 8×32 宽阵列 |

### 7.4 时钟门控机制

每个 PE 的使能信号为：

```verilog
wire pe_clk_en = en && row_active && col_active;
```

其中 `row_active` 和 `col_active` 由 `cfg_shape` 组合解码：

```verilog
wire row_active = (cfg_shape == 2'b00) ? (r < 4) :
                  (cfg_shape == 2'b01) ? (r < 8) :
                  1'b1;  // 16x16 和 8x32 使用所有行

wire col_active = (cfg_shape == 2'b00) ? (c < 4) :
                  (cfg_shape == 2'b01) ? (c < 8) :
                  1'b1;  // 16x16 和 8x32 使用所有列
```

**效果**：非活跃区域的 PE 接收 `en=0`，其内部寄存器冻结，不消耗动态功耗。

### 7.5 8×32 折叠模式详解

当 `cfg_shape = 2'b11` 时，物理 16×16 阵列被对折为逻辑 8×32 阵列：

```
物理阵列 (16x16):
┌───────────────────────────────────────┐
│ Top Half   (Row 0~7):  8行 × 16列     │
│ ┌───────────────────────────────────┐ │
│ │ Col 0 ... Col 14  →  L[0..14]    │ │
│ │ Col 15        →  fold_act_from_top│ ├──► 折叠到 Bottom Half
│ └───────────────────────────────────┘ │
│ Bottom Half (Row 8~15): 8行 × 16列    │
│ ┌───────────────────────────────────┐ │
│ │ Col 0 = from fold_act (折叠输入)  │ │
│ │ Col 1...15    →  R[17..31]       │ │
│ └───────────────────────────────────┘ │
└───────────────────────────────────────┘

逻辑输出 (8x32):
  L[0..14]  = acc_v[8][0..14]      (Top Half, Row 8 输出)
  L[15]     = acc_v[8][15]          (Top Half, Row 8 输出)
  R[16..31] = acc_v[16][0..15]      (Bottom Half, Row 16 输出)
```

**数据流细节**：

| 方面 | 实现 |
|------|------|
| **Activation 折叠** | Row 7, Col 15 的 `act_reg` 输出 → 通过 MUX 路由到 Row 8, Col 0 的激活输入 |
| **部分和截断** | 下半部（Row ≥ 8）的 `acc_in` 强制为零，无垂直链路连接上下两半 |
| **输出拼接** | 左半（L[0:15]）来自 `acc_v[8]`，右半（R[16:31]）来自 `acc_v[16]` |

### 7.6 数据传播路径

#### OS 模式（Output-Stationary）

```
  激活水平传播（无条件，不依赖 en）：
    act_h[r][c+1] = act_reg (posedge clk)
    → 行间形成 systolic 流

  权重水平传播（OS 模式专用）：
    w_h[r][c+1] = os_w_reg (posedge clk)
    → 权重像 activation 一样 systolic 流动

  部分和垂直传播：
    acc_v[r+1][c] = PE[r][c].acc_out
```

#### WS 模式（Weight-Stationary）

```
  激活水平传播（systolic shift）：
    act_h[r][c+1] = act_reg (posedge clk)

  权重广播（按列）：
    pe_w_in = w_in[c*DATA_W +: DATA_W]
    → 第 c 列的所有行接收相同权重

  部分和流同 OS 模式
```

### 7.7 关键实现说明

- **无条件数据传播**：`act_reg` 和 `os_w_reg` 不依赖 `en` 信号，始终传播数据——避免 tile 边界产生气泡
- **OS 模式权重流动**：新增 `w_h` 水平 shift register，OS 模式下 weight 像 activation 一样 systolic 流动（不再按列广播）
- **输出端口位宽**：`acc_out` 为 `32 * ACC_W = 1024 bit`，以容纳 8×32 模式的 32 列输出；非使用列输出零
- **调试接口**：`pe_active` 扁平化输出每个 PE 的活跃状态，可用于波形验证

---

## 8. `pe_top` — 单个处理单元（双权重寄存器版）

> **2026-04-14 升级**：新增双权重寄存器（Dual Weight Register Bank），支持后台预取隐藏权重加载延迟。

- **文件**：`rtl/pe/pe_top.v`
- **功能**：单 PE 的乘累加（MAC）流水线，3 级流水，支持 INT8/FP16 两条乘法路径，WS/OS 两种累加模式。**新增双权重寄存器实现预取隐藏**。

### 8.1 接口信号

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `clk` | input | 1 | |
| `rst_n` | input | 1 | 低有效同步复位 |
| `mode` | input | 1 | `0=INT8, 1=FP16` |
| `stat_mode` | input | 1 | `0=WS, 1=OS` |
| `en` | input | 1 | 流水线使能 |
| `flush` | input | 1 | 累加器 flush（输出并清零） |
| `load_w` | input | 1 | WS：将 `w_in` 锁存到**活跃+预取**权重寄存器（同周期可用） |
| `swap_w` | input | 1 | WS：原子交换 active ↔ prefetch 权重寄存器（单拍完成） |
| `w_in` | input | DATA_W | 权重输入（INT8 用 `[7:0]`） |
| `a_in` | input | DATA_W | 激活输入（INT8 用 `[7:0]`） |
| `acc_in` | input | ACC_W | 外部链路部分和（tile-loop 下 = 0） |
| `acc_out` | output | ACC_W | 累加结果输出 |
| `valid_out` | output | 1 | 输出有效（flush 拍置高） |

### 8.2 双权重寄存器架构

> **设计动机**：在 WS 模式中，PE 需要加载新 weight 后才能开始下一轮计算。传统设计中 `load_w` 的 weight 要等下一个周期才能使用，产生一个周期的气泡。双寄存器允许在当前计算进行时**后台预取**下一组 weight。

```
                    Dual Weight Register Bank
                    =========================

  w_sel = 0 (default)          w_sel = 1 (after swap)
  ┌──────────────┐            ┌──────────────┐
  │  w_reg[0]     │            │  w_reg[1]     │
  │  ★ ACTIVE    │   ←swap──→ │  ★ ACTIVE    │
  │              │            │              │
  │ 当前用于计算  │            │ 当前用于计算  │
  └──────────────┘            └──────────────┘
       ▲                           ▲
       │ load_w 同时写入           │ load_w 同时写入
       │ (向后兼容：立即生效)      │ (向后兼容：立即生效)
       │                           │
  w_in ◄───────────────────────────┘

  active_weight = w_reg[w_sel]  →  送入 Stage-0
```

#### 操作语义

| 操作 | 信号 | 效果 |
|------|------|------|
| **加载权重（同周期可用）** | `load_w = 1` | 同时写入 `w_reg[w_sel]`(active) 和 `w_reg[~w_sel]`(prefetch)，**本周期 Stage-0 可通过 bypass 使用 `w_in`** |
| **纯预取** | `load_w = 1` + 后续 `swap_w` | 写入 prefetch 寄存器；下次 `swap_w` 切换后生效 |
| **原子交换** | `swap_w = 1` | 单拍切换 `w_sel ← ~w_sel`，active 与 prefetch 身份互换 |
| **复位** | `rst_n = 0` | 所有寄存器归零，`w_sel = 0` |

#### 典型使用流程（无气泡预取）

```verilog
// 周期 T0: 加载第一组 weight 并计算 tile A
load_w = 1;  w_in = weight_A;
// 周期 T0: Stage-0 通过 bypass 直接使用 w_in (=weight_A)
//           同时 w_reg[0] 和 w_reg[1] 都被写入 weight_A

// 周期 T1~TK: 计算 tile A（使用 w_reg[0]）
// 后台同时：
load_w = 1;  w_in = weight_B;  // 在某周期预取 weight_B 到 prefetch
// ...
swap_w = 1;                     // 交换：prefetch 变 active
// 下一个 tile B 可以立即使用 weight_B，无额外等待
```

### 8.3 流水线结构

```
  Stage-0 (Input Register) —— 含双权重选择逻辑
    ┌─────────────────────────────────────────────────────┐
    │  WS (load_w 周期):                                   │
    │    s0_w = w_in          ← bypass，同周期可使用       │
    │  WS (后续周期):                                       │
    │    s0_w = active_weight (= w_reg[w_sel])             │
    │  OS:                                                   │
    │    s0_w = w_in;  s0_a = a_in                          │
    │  flush: s0_w=0, s0_a=0                                │
    └─────────────────────────────────────────────────────┘
              ↓ posedge clk
  Stage-1 (Multiply)
    ┌─────────────────────────────────────────┐
    │  INT8: s1_mul = sign_extend(s0_w[7:0]   │
    │                * s0_a[7:0])  → 32-bit   │
    │  FP16: s1_mul = fp16_mul(s0_w, s0_a)   │
    └─────────────────────────────────────────┘
              ↓ posedge clk
  Stage-2 (Accumulate / Output)
    ┌─────────────────────────────────────────────────────────┐
    │  OS 正常: os_acc += s1_mul (FP16: FP32 精度累加)        │
    │  OS flush: acc_out = os_acc + s1_mul; os_acc=0; valid=1 │
    │  WS 正常: ws_acc += s1_mul                               │
    │  WS flush: acc_out = ws_acc; ws_acc=0; valid=1           │
    └─────────────────────────────────────────────────────────┘
```

### 8.4 关键时序：WS 模式 load_w bypass

```
  CLK     _/‾\_/‾\_/‾\_/‾\_/‾\_
  load_w  ‾‾‾\____________________   (脉冲 1 拍)
  w_in    ══W0═════════════════════   (新权重值)
  en      ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  s0_w    ══W0══W0═W0══════════════   (bypass: load_w 周期直接用 w_in)
                                  (后续周期用 w_reg[w_sel])

  关键: load_w=1 且 stat_mode=0 时,
        s0_w = w_in (bypass), 而非 active_weight (需要 1 周期稳定)
        这确保了向后兼容——旧代码不需要修改即可工作
```

### 8.5 FP16 混合精度实现

- **乘法**：`fp16_mul`（FP16 × FP16 → FP16，1 级流水）
- **累加**：`fp16_to_fp32()` 将 FP16 乘积转 FP32，然后 `fp32_add`（FP32 + FP32 → FP32）
- **输出**：32-bit FP32 结果直接写入 Result FIFO → DRAM
- 指数偏置重映射：`127 + (exp16 - 15)`，正确处理零、Inf、NaN、次正规数

---

## 9. `pe_array` — 旧版 PE 阵列封装（已废弃）

> **状态**：已废弃。`npu_top.v` 已切换为实例化 `reconfig_pe_array.v`（第 7 节）。此文件保留供参考，不再参与综合。

- **文件**：`rtl/array/pe_array.v`
- **功能**：旧版 4×4 PE 阵列封装。已被 `reconfig_pe_array.v` 替代。

---

## 10. `npu_power` — 电源与时钟管理

- **文件**：`rtl/power/npu_power.v`
- **功能**：DFS（动态频率调整）分频器 + 行/列时钟门控（ICG 行为模型），为 PE 阵列提供受控时钟。

### 9.1 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `ROWS` | 4 | PE 行数，决定 `row_clk_gated` 总线宽度 |
| `COLS` | 4 | PE 列数，决定 `col_clk_gated` 总线宽度 |

### 9.2 接口信号

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `clk` | input | 1 | 系统时钟输入 |
| `rst_n` | input | 1 | 低有效同步复位 |
| `div_sel` | input | 3 | DFS 分频选择（见下表） |
| `row_cg_en` | input | ROWS | 行门控使能（`1`=该行时钟关闭） |
| `col_cg_en` | input | COLS | 列门控使能（`1`=该列时钟关闭） |
| `npu_clk` | output | 1 | DFS 分频后的 NPU 主时钟（驱动 pe_array） |
| `row_clk_gated` | output | ROWS | 各行门控时钟（ICG 输出，预留后端使用） |
| `col_clk_gated` | output | COLS | 各列门控时钟（ICG 输出，预留后端使用） |

### 9.3 DFS 分频选项

| `div_sel` | 分频比 | 说明 |
|----------:|-------:|------|
| `000` | ÷1 | bypass（`npu_clk = sys_clk`，无延迟） |
| `001` | ÷2 | 250 MHz（@500 MHz sys_clk） |
| `010` | ÷4 | 125 MHz |
| `011` | ÷8 | 62.5 MHz |

### 9.4 DFS 时序

```
  SYS_CLK   _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_

  div_sel=001 (÷2):
  NPU_CLK   _____/‾‾‾‾\____/‾‾‾‾\___________

  div_sel=010 (÷4):
  NPU_CLK   _________/‾‾‾‾‾‾‾‾‾\______________
```

### 9.5 ICG 行为模型

```
  SYS_CLK  _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_
  row_cg_en(0)  ‾‾‾‾‾‾‾‾‾\____/‾‾‾‾‾‾‾  (1=gate, 0=pass)
  LATCH_EN      ‾‾‾‾‾‾‾‾‾‾‾‾\__/‾‾‾‾‾‾‾  (低电平锁存 ~row_cg_en)
  row_clk_gated _/‾\_/‾\_/‾\_____/‾\_/‾\  (无毛刺)

  原理：
  ① latch 在 npu_clk LOW 期间对 ~row_cg_en 透明（透明锁存）
  ② npu_clk HIGH 期间 latch 锁定，使能不随 row_cg_en 变化
  ③ row_clk_gated = npu_clk AND latched_en → 无毛刺门控时钟
```

### 9.6 当前接入状态

| 功能 | 状态 | 说明 |
|------|------|------|
| DFS `npu_clk` | ✅ 已接入 | 驱动 `pe_array.clk`（`div_sel=0` 时与 `sys_clk` 同频） |
| `row_clk_gated` | ✅ 已生成 | ICG 行为模型，RTL 功能正确；预留给 ASIC 后端替换 ICG 标准单元 |
| `col_clk_gated` | ✅ 已生成 | 同上；当前 col_cg_en 固定为 0（始终使能） |
| 行级单独门控 | ⚠️ 预留 | `row_cg_en = {ROWS{~pe_en}}`；OS 模式下按需关闭非活跃行为后续增强 |

> **⚠️ CDC 注意**：当 `div_sel ≠ 0` 时，PE 阵列运行在 `npu_clk`（分频），而 `npu_ctrl`/`npu_dma`/`PPBuf` 仍运行在 `sys_clk`，形成异步时钟域。当前 `div_sel` 复位初始值为 0（bypass），默认单时钟域无 CDC 问题。如需使用 DFS，需在控制路径插入同步器。

---

## 11. SoC 模块

### 11.1 `soc_top` — SoC 顶层

- **文件**：`rtl/soc/soc_top.v`
- **功能**：连接 PicoRV32 CPU、SRAM、DRAM、AXI-Lite 桥接器与 NPU 顶层。

#### 地址映射

| 地址范围 | 区域 | 容量 | 说明 |
|---------|------|-----:|------|
| `0x0000_0000 – 0x0000_0FFF` | SRAM | 4 KB | CPU 指令 + 数据 |
| `0x0000_1000 – 0x0001_FFFF` | DRAM | 124 KB | 权重/激活/结果 |
| `0x0200_0000 – 0x0200_001F` | NPU AXI-Lite | 32 B | NPU 寄存器 |

> **关键**：`addr_is_ram = mem_addr < 32'h1000`，地址 `0x0F00` 属于 SRAM 范围（`< 0x1000`）。

### 11.2 `soc_mem` — CPU SRAM

- **文件**：`rtl/soc/soc_mem.v`
- **关键实现**：CPU 读口使用**组合（异步）读**（`assign rdata = mem[addr]`）。
- **为何关键**：PicoRV32 要求 `mem_rdata` 与 `mem_ready` 在同一周期有效；同步读会引入 1 拍延迟，导致每条指令读到 stale 数据。

### 11.3 `dram_model` — DRAM 行为模型

- **文件**：`rtl/soc/dram_model.v`
- **双端口**：CPU 读/写端口（组合读）+ NPU AXI4 Master 端口（支持 burst）
- **关键实现**：CPU 读同样使用异步读（`assign cpu_rdata = mem[cpu_addr>>2]`）

### 11.4 `axi_lite_bridge` — AXI-Lite 桥

- **文件**：`rtl/soc/axi_lite_bridge.v`
- **功能**：将 PicoRV32 `mem_valid/mem_ready/mem_wdata/mem_rdata` 接口转换为 AXI4-Lite。
- **写时序**：3 周期（S_WRITE_AW → S_WRITE_W → 完成），与 `npu_axi_lite` 的两阶段握手匹配。

---

## 12. 废弃与保留模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| `pe_array` | `rtl/array/pe_array.v` | ⚠️ 已废弃 | 旧版 4×4 PE 阵列封装，已由 `reconfig_pe_array.v` 替代 |
| `array_ctrl` | `rtl/ctrl/array_ctrl.v` | ⚠️ 废弃 | 早期独立阵列控制器，已被 `npu_ctrl` 的 tile-loop FSM 取代 |
| `axi_monitor` | `rtl/common/axi_monitor.v` | 保留 | 仿真用 AXI 总线监视器（仅 testbench 使用） |
| `op_counter` | `rtl/common/op_counter.v` | 保留 | 仿真用操作计数器（仅 testbench 使用） |

---

## 13. 模块关系速查

```
soc_top
 ├── picorv32            CPU 核（外部引用）
 ├── soc_mem             4KB SRAM，异步读
 ├── dram_model          DRAM，双端口
 ├── axi_lite_bridge     mem_if → AXI4-Lite
 └── npu_top
      ├── npu_axi_lite   寄存器文件，AXI4-Lite 从机
      ├── npu_ctrl        tile-loop FSM 控制器
      ├── npu_dma         3 通道 AXI4 主机 DMA
      ├── pingpong_buf×2  W/A 双 PPBuf（各自独立）
      ├── reconfig_pe_array  ★ 16x16 可重配置 PE 阵列
      │    └── pe_top ×256   双权重寄存器版 PE
      │          ├── fp16_mul    FP16 乘法器
      │          └── fp32_add    FP32 加法器（累加）
      ├── fifo             结果 FIFO
      └── npu_power        DFS + ICG 时钟管理
```

### 关键控制信号流

```
CPU:       AXI-Lite write → npu_axi_lite.ctrl_reg → npu_ctrl
                                                          │
           ┌─── dma_w/a_start + addr/len ──────────────► npu_dma
           │                                                 │
           │    swap/clear ──────────────────────────► pingpong_buf
           │                                                 │
           └─── pe_en/flush/mode/stat + target_col ────► npu_top(路由) + pe_array
                                                              │
           result ◄── r_fifo ◄── pe_array.valid ──────────────┘
              │
              └── npu_dma (R_WRITE) → DRAM
```
