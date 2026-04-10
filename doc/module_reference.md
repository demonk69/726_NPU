# NPU RTL 模块说明文档

> 最后更新：2026-04-08
> 说明：本文件按当前仓库中的 RTL 实现同步，重点描述"真正接入并参与主流程"的行为。

---

## 目录

1. [阅读约定](#1-阅读约定)
2. [npu_top — NPU 顶层](#2-npu_top--npu-顶层)
3. [npu_axi_lite — 配置寄存器文件](#3-npu_axi_lite--配置寄存器文件)
4. [npu_ctrl — tile-loop 控制器](#4-npu_ctrl--tile-loop-控制器)
5. [npu_dma — 三通道 DMA](#5-npu_dma--三通道-dma)
6. [pingpong_buf — 双 Bank Ping-Pong 缓冲](#6-pingpong_buf--双-bank-ping-pong-缓冲)
7. [pe_array — PE 阵列封装](#7-pe_array--pe-阵列封装)
8. [pe_top — 单个处理单元](#8-pe_top--单个处理单元)
9. [npu_power — 电源与时钟管理](#9-npu_power--电源与时钟管理)
10. [SoC 模块](#10-soc-模块)
11. [废弃与保留模块](#11-废弃与保留模块)
12. [模块关系速查](#12-模块关系速查)

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
| `ROWS` | 4 | PE 阵列行数 |
| `COLS` | 4 | PE 阵列列数 |
| `DATA_W` | 16 | PE 输入位宽（INT8/FP16 均用 16-bit 接口） |
| `ACC_W` | 32 | 累加结果位宽 |
| `PPB_DEPTH` | 64 | Ping-Pong 每 Bank 32-bit 字深度 |
| `PPB_THRESH` | 16 | PPBuf 早启动阈值（字数） |

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

## 4. `npu_ctrl` — tile-loop 控制器

- **文件**：`rtl/ctrl/npu_ctrl.v`
- **功能**：驱动矩阵乘法 tile-loop 调度。每次迭代计算 `C[i][j]` 的一个元素，控制 DMA 启动/结束、PE 工作状态、PPBuf 切换。

### 4.1 核心概念：Tile-Based 调度

**Tile 定义**：一个 tile 代表一个输出元素 `C[i][j]` 的完整计算任务。

**Tile 循环**：控制器按以下顺序处理所有 tile：
```
for i = 0 to M-1:
    for j = 0 to N-1:
        // 处理一个 tile
        启动 DMA 读 B[:,j]
        启动 DMA 读 A[i,:]
        PE 计算 K 个元素对
        结果写入 FIFO
        DMA 写结果到 C[i][j]
```

**Tile 状态**：控制器维护：
- `tile_i`：当前输出行索引（0 ≤ tile_i < M）
- `tile_j`：当前输出列索引（0 ≤ tile_j < N）
- `target_col`：OS 模式下的权重目标列（`tile_j % COLS`）

**Tile 地址计算**：
```
权重地址 = W_ADDR + tile_j × K × 元素字节数
激活地址 = A_ADDR + tile_i × K × 元素字节数
结果地址 = R_ADDR + (tile_i×N + tile_j) × 4
```

### 4.2 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `ROWS` | 4 | PE 行数（当前控制器不区分行，统一广播） |
| `COLS` | 4 | PE 列数（OS 模式下的 target_col 范围） |
| `DATA_W` | 16 | PE 数据位宽（用于计算 DMA 字节数） |
| `ACC_W` | 32 | 结果宽度（用于 `dma_r_len = ACC_W/8`） |

### 4.2 接口信号

#### 配置输入（来自 npu_axi_lite）

| 信号 | 位宽 | 说明 |
|------|-----:|------|
| `ctrl_reg` | 32 | 控制寄存器（start/abort/mode/stat_mode） |
| `m_dim` | 32 | 矩阵 M 维度 |
| `n_dim` | 32 | 矩阵 N 维度 |
| `k_dim` | 32 | 矩阵 K 维度 |
| `w_addr` | 32 | 权重基地址 |
| `a_addr` | 32 | 激活基地址 |
| `r_addr` | 32 | 结果基地址 |
| `arr_cfg` | 8 | 阵列配置（当前未消费） |

#### 状态输出（到 npu_axi_lite）

| 信号 | 位宽 | 说明 |
|------|-----:|------|
| `busy` | 1 | 运算进行中（S_IDLE 以外均为高） |
| `done` | 1 | sticky 完成信号 |
| `irq` | 1 | 单拍完成中断脉冲 |

#### DMA 控制接口

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `dma_w_start` | output | 1 | 权重 DMA 启动脉冲（单拍高） |
| `dma_w_done` | input | 1 | 权重 DMA 完成（被 `dma_w_done_r` 锁存） |
| `dma_w_addr` | output | 32 | 权重 DRAM 地址 |
| `dma_w_len` | output | 16 | 权重 DMA 字节数 |
| `dma_a_start` | output | 1 | 激活 DMA 启动脉冲 |
| `dma_a_done` | input | 1 | 激活 DMA 完成 |
| `dma_a_addr` | output | 32 | 激活 DRAM 地址 |
| `dma_a_len` | output | 16 | 激活 DMA 字节数 |
| `dma_r_start` | output | 1 | 结果 DMA 启动脉冲 |
| `dma_r_done` | input | 1 | 结果 DMA 完成 |
| `dma_r_addr` | output | 32 | 结果 DRAM 地址 |
| `dma_r_len` | output | 16 | 结果 DMA 字节数（固定 `ACC_W/8 = 4`） |

#### PE 控制接口

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `pe_en` | output | 1 | PE 工作使能 |
| `pe_flush` | output | 1 | 累加器 flush 脉冲（持续 `PIPELINE_DEPTH` 拍） |
| `pe_mode` | output | 1 | `0=INT8, 1=FP16` |
| `pe_stat` | output | 1 | `0=WS, 1=OS` |
| `target_col` | output | `$clog2(COLS)` | OS 模式权重路由目标列（已扩展位宽，2026-04-08） |

#### PPBuf 控制接口

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `w_ppb_ready` | input | 1 | 权重 PPBuf 已达阈值 |
| `w_ppb_empty` | input | 1 | 权重 PPBuf 读侧为空 |
| `a_ppb_ready` | input | 1 | 激活 PPBuf 已达阈值 |
| `a_ppb_empty` | input | 1 | 激活 PPBuf 读侧为空 |
| `w_ppb_swap` | output | 1 | 权重 PPBuf bank 切换脉冲 |
| `a_ppb_swap` | output | 1 | 激活 PPBuf bank 切换脉冲 |
| `w_ppb_clear` | output | 1 | 权重 PPBuf 复位脉冲 |
| `a_ppb_clear` | output | 1 | 激活 PPBuf 复位脉冲 |

### 4.3 FSM 状态与转移

```
                  cfg_start_rise
  S_IDLE ─────────────────────────────────────────────► S_LOAD
    ▲                                                        │ dma_w_done_r && dma_a_done_r
    │                                                        ▼
  S_DONE ◄──── tile_count+1 >= tile_total ──────── S_PRELOAD
    │                                                        │ (下拍)
    │                                                        ▼
    │                                                   S_COMPUTE
    │                                                        │ ppb_empty && dma_all_done
    │                                                        ▼
    │                                                    S_DRAIN
    │                                                        │ drain_cnt >= PIPELINE_DEPTH
    │                                                        ▼
    │                                                  S_WRITE_BACK
    │                                                        │ dma_r_start 脉冲
    │                                                        ▼
    │                                                   S_WB_WAIT
    │                                                        │ dma_r_done_r
    │                                                        ▼
    └──────── tile_count+1 < tile_total ──────────── S_NEXT_TILE
```

| 状态 | pe_en | pe_flush | 动作 |
|------|-------|----------|------|
| S_IDLE | 0 | 0 | 等待 `cfg_start_rise`；发出第 0 tile DMA，清 PPBuf |
| S_LOAD | 0 | 0 | 等待 W+A DMA 双完成；`swap` PPBuf |
| S_PRELOAD | 1 | 0 | 拉高 `pe_en` 一拍，锁存 DMA 长度 |
| S_COMPUTE | 1 | 0 | PE 持续消费 PPBuf；检测 PPBuf 双空 |
| S_DRAIN | 1 | 1 | 拉高 `pe_flush` 持续 `PIPELINE_DEPTH`（3）拍，排空流水线 |
| S_WRITE_BACK | 1 | 0 | 发出 `dma_r_start` 单拍脉冲 |
| S_WB_WAIT | 1 | 0 | 等待结果写回完成 |
| S_NEXT_TILE | 0 | 0 | 推进 tile_i/j，计算下一 tile DMA 地址，发出 DMA start |
| S_DONE | 0 | 0 | 拉高 `done`、`irq`，回 S_IDLE |

### 4.4 时序图：单 Tile 完整流程

```
  CLK   _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_...
  state [IDLE][LOAD ...........][PRELOAD][COMPUTE .....][DRAIN...][WB][WAIT][NEXT]
  pe_en  0    0                  1        1              1         1   1     0
  flush  0    0                  0        0              1  1  1   0   0     0
  w_start  1                                                           1(next)
  a_start  1                                                           1(next)
  r_start                                                    1
```

### 4.5 关键实现说明

- **`cfg_start` 边沿检测**：`cfg_start_rise = cfg_start && !cfg_start_d1`，防止软件保持 start=1 时重复触发。
- **`done` sticky**：`done` 在 S_DONE 置高后保持，直到软件将 `CTRL[0]` 清零（`if (!cfg_start) done <= 0`）。
- **`dma_xxx_done_r`**：`done` 信号被立即锁存为 sticky 寄存器，防止多拍状态丢失。
- **`k_dma_len_w`**：组合逻辑，按 INT8 或 FP16 动态计算 DMA 字节数（4B 对齐）。
- **`target_col`**：OS 模式下等于 `(tile_j % COLS)`，已扩展为 `$clog2(COLS)` 位，COLS=4 时无截断。

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

## 7. `pe_array` — PE 阵列封装

- **文件**：`rtl/array/pe_array.v`
- **功能**：实例化 `ROWS × COLS` 个 `pe_top`，组织激活与部分和的传播路径，支持 WS 和 OS 两种数据流模式。

### 7.1 模块参数

| 参数 | 默认值 | 说明 |
|------|-------:|------|
| `ROWS` | 4 | 行数 |
| `COLS` | 4 | 列数 |
| `DATA_W` | 16 | 数据位宽 |
| `ACC_W` | 32 | 累加结果位宽 |

### 7.2 接口信号

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `clk` | input | 1 | 来自 `npu_top` 的 `npu_clk_out`（DFS 分频时钟） |
| `rst_n` | input | 1 | 低有效同步复位 |
| `mode` | input | 1 | `0=INT8, 1=FP16` |
| `stat_mode` | input | 1 | `0=WS, 1=OS` |
| `en` | input | 1 | PE 使能（时钟使能等效，提供动态功耗控制） |
| `flush` | input | 1 | 累加器 flush |
| `load_w` | input | 1 | WS 模式权重锁存脉冲 |
| `w_in` | input | `COLS*DATA_W` | 权重输入（`[c*DATA_W +: DATA_W]` 为第 c 列权重） |
| `act_in` | input | `ROWS*DATA_W` | 激活输入（`[r*DATA_W +: DATA_W]` 为第 r 行激活） |
| `acc_in` | input | `COLS*ACC_W` | 列顶端部分和输入（当前固定为 0） |
| `acc_out` | output | `COLS*ACC_W` | 列底端累加结果 |
| `valid_out` | output | COLS | 各列结果有效标志 |

### 7.3 数据传播路径

#### OS 模式（Output-Stationary）

```
  激活广播（无行错位）：
    act_h[r][0] = act_in[r] → act_h[r][1] = act_h[r][0] → ... (绕过 act_reg)
    所有列的 PE[r][c] 在同一拍看到相同的激活值

  权重列路由（由 npu_top 在 w_in 接入前完成）：
    PE[r][target_col] 收到 B[:,j]；其余列 w_in=0
    → 只有 target_col 列的 os_acc 有效

  部分和垂直传播（重力方向）：
    acc_v[0][c] = 0 (顶端边界)
    acc_v[r+1][c] = PE[r][c].acc_out → 向下传播
    acc_out[c] = acc_v[ROWS][c] (最底一行的输出)
```

#### WS 模式（Weight-Stationary）

```
  激活行错位（systolic shift）：
    act_h[r][c+1] = act_reg (posedge clk 延迟一拍)
    → 行 r 比行 r-1 晚一拍处理相同激活
    → 形成 ROWS 拍的流水线斜波

  权重广播（所有列 load 同一行权重）：
    load_w=1 时所有 PE 的 weight_reg 锁存 w_in
    后续 K 拍复用 weight_reg

  部分和流同 OS 模式（向下传播）
```

### 7.4 关键实现说明

- `act_reg` 在 `flush` 周期强制清零，防止跨 tile 污染。
- `en=0` 时所有 PE 寄存器冻结（clock enable 方式的动态功耗控制，与 `npu_power` ICG 协同）。
- `acc_in` 顶端固定接 0，PE 内部 `os_acc`/`ws_acc` 各自维护累加状态。

---

## 8. `pe_top` — 单个处理单元

- **文件**：`rtl/pe/pe_top.v`
- **功能**：单 PE 的乘累加（MAC）流水线，3 级流水，支持 INT8/FP16 两条乘法路径，WS/OS 两种累加模式。

### 8.1 接口信号

| 信号 | 方向 | 位宽 | 说明 |
|------|------|-----:|------|
| `clk` | input | 1 | |
| `rst_n` | input | 1 | 低有效同步复位 |
| `mode` | input | 1 | `0=INT8, 1=FP16` |
| `stat_mode` | input | 1 | `0=WS, 1=OS` |
| `en` | input | 1 | 流水线使能 |
| `flush` | input | 1 | 累加器 flush（输出并清零） |
| `load_w` | input | 1 | WS 模式：将 `w_in` 锁存到 `weight_reg` |
| `w_in` | input | DATA_W | 权重（INT8 用 `[7:0]`） |
| `a_in` | input | DATA_W | 激活（INT8 用 `[7:0]`） |
| `acc_in` | input | ACC_W | 外部链路部分和（WS chain 测试用；tile-loop 下 = 0） |
| `acc_out` | output | ACC_W | 累加结果输出 |
| `valid_out` | output | 1 | 输出有效（flush 拍置高） |

### 8.2 流水线结构

```
  Stage-0 (Input Register)
    ┌─────────────────────────────────────────┐
    │  WS: s0_w = load_w ? w_in : weight_reg  │
    │  OS: s0_w = w_in;  s0_a = a_in         │
    │  flush: s0_w=0, s0_a=0 (清零防止二次累加)│
    └─────────────────────────────────────────┘
              ↓ posedge clk
  Stage-1 (Multiply)
    ┌─────────────────────────────────────────┐
    │  INT8: s1_mul = sign_extend(s0_w[7:0]   │
    │                * s0_a[7:0])  → 32-bit   │
    │  FP16: s1_mul = fp16_mul(s0_w, s0_a)   │
    │         (fp16_mul 内部 1 拍流水)         │
    └─────────────────────────────────────────┘
              ↓ posedge clk
  Stage-2 (Accumulate / Output)
    ┌─────────────────────────────────────────────────────────┐
    │  OS 正常: os_acc += s1_mul (FP16: FP32 精度累加)        │
    │  OS flush: acc_out = os_acc + s1_mul; os_acc=0; valid=1 │
    │  WS 正常: ws_acc += s1_mul; acc_out = acc_in + s1_mul   │
    │  WS flush: acc_out = ws_acc; ws_acc=0; valid=1 (仅首拍) │
    └─────────────────────────────────────────────────────────┘
```

### 8.3 关键时序：OS 模式 flush

```
  CLK     _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_
  en       ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
  flush    ____________/‾‾‾‾‾‾\__________   (持续 3 拍)
  w_in     ═[K-1]══════[0]=[0]=[0]=══════  (Stage-0 置零)
  a_in     ═[K-1]══════[0]=[0]=[0]=══════
                              ↑
  s1_flush _______________/‾‾‾‾‾‾\________
  valid    _________________/‾\___________  (Stage-2: 仅首个 flush 拍)
  acc_out  ═════════════════[RESULT]=══════

  注意: flush=1 且 stage-0 置零 → stage-1 product=0
       stage-2 acc_out = os_acc + 0 = os_acc 输出后清零
```

### 8.4 FP16 混合精度实现

- **乘法**：`fp16_mul`（FP16 × FP16 → FP16，1 级流水）
- **累加**：`fp16_to_fp32()` 将 FP16 乘积转 FP32，然后 `fp32_add`（FP32 + FP32 → FP32）
- **输出**：32-bit FP32 结果直接写入 Result FIFO → DRAM
- 指数偏置重映射：`127 + (exp16 - 15)`，正确处理零、Inf、NaN、次正规数

---

## 9. `npu_power` — 电源与时钟管理

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

## 10. SoC 模块

### 10.1 `soc_top` — SoC 顶层

- **文件**：`rtl/soc/soc_top.v`
- **功能**：连接 PicoRV32 CPU、SRAM、DRAM、AXI-Lite 桥接器与 NPU 顶层。

#### 地址映射

| 地址范围 | 区域 | 容量 | 说明 |
|---------|------|-----:|------|
| `0x0000_0000 – 0x0000_0FFF` | SRAM | 4 KB | CPU 指令 + 数据 |
| `0x0000_1000 – 0x0001_FFFF` | DRAM | 124 KB | 权重/激活/结果 |
| `0x0200_0000 – 0x0200_001F` | NPU AXI-Lite | 32 B | NPU 寄存器 |

> **关键**：`addr_is_ram = mem_addr < 32'h1000`，地址 `0x0F00` 属于 SRAM 范围（`< 0x1000`）。

### 10.2 `soc_mem` — CPU SRAM

- **文件**：`rtl/soc/soc_mem.v`
- **关键实现**：CPU 读口使用**组合（异步）读**（`assign rdata = mem[addr]`）。
- **为何关键**：PicoRV32 要求 `mem_rdata` 与 `mem_ready` 在同一周期有效；同步读会引入 1 拍延迟，导致每条指令读到 stale 数据。

### 10.3 `dram_model` — DRAM 行为模型

- **文件**：`rtl/soc/dram_model.v`
- **双端口**：CPU 读/写端口（组合读）+ NPU AXI4 Master 端口（支持 burst）
- **关键实现**：CPU 读同样使用异步读（`assign cpu_rdata = mem[cpu_addr>>2]`）

### 10.4 `axi_lite_bridge` — AXI-Lite 桥

- **文件**：`rtl/soc/axi_lite_bridge.v`
- **功能**：将 PicoRV32 `mem_valid/mem_ready/mem_wdata/mem_rdata` 接口转换为 AXI4-Lite。
- **写时序**：3 周期（S_WRITE_AW → S_WRITE_W → 完成），与 `npu_axi_lite` 的两阶段握手匹配。

---

## 11. 废弃与保留模块

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| `array_ctrl` | `rtl/ctrl/array_ctrl.v` | ⚠️ 废弃 | 早期独立阵列控制器，已被 `npu_ctrl` 的 tile-loop FSM 取代，仍存在于 `rtl/` 目录，不参与综合，待清理 |
| `axi_monitor` | `rtl/common/axi_monitor.v` | 保留 | 仿真用 AXI 总线监视器（仅 testbench 使用） |
| `op_counter` | `rtl/common/op_counter.v` | 保留 | 仿真用操作计数器（仅 testbench 使用） |

---

## 12. 模块关系速查

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
      ├── pe_array        ROWS×COLS PE 网格
      │    └── pe_top×16  单 PE（含 fp16_mul/fp32_add）
      ├── sync_fifo       结果 FIFO（PE→DMA）
      └── npu_power       DFS 分频 + 行/列 ICG 门控时钟
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
