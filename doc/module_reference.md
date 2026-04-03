# NPU RTL 模块详细文档

> 项目：NPU_prj — 嵌入式低功耗 NPU 加速器 IP
> 日期：2026-04-02
> 状态：RTL 功能验证通过（点积测试 310=310）

---

## 目录

1. [系统概览](#1-系统概览)
2. [NPU 核心模块](#2-npu-核心模块)
   - 2.1 [npu_top — NPU 顶层集成](#21-npu_top--npu-顶层集成)
   - 2.2 [npu_ctrl — NPU 控制器 FSM](#22-npu_ctrl--npu-控制器-fsm)
   - 2.3 [npu_axi_lite — AXI4-Lite 寄存器文件](#23-npu_axi_lite--axi4-lite-寄存器文件)
   - 2.4 [npu_dma — DMA 控制器](#24-npu_dma--dma-控制器)
   - 2.5 [pe_top — 处理单元 (PE)](#25-pe_top--处理单元-pe)
   - 2.6 [fp16_mul — FP16 乘法器](#26-fp16_mul--fp16-乘法器)
   - 2.7 [pe_array — PE 阵列](#27-pe_array--pe-阵列)
   - 2.8 [array_ctrl — 阵列控制器](#28-array_ctrl--阵列控制器)
3. [缓冲与存储模块](#3-缓冲与存储模块)
   - 3.1 [pingpong_buf — 乒乓缓冲区](#31-pingpong_buf--乒乓缓冲区)
   - 3.2 [sync_fifo — 同步 FIFO](#32-sync_fifo--同步-fifo)
4. [电源管理模块](#4-电源管理模块)
   - 4.1 [npu_power — 时钟门控与 DFS](#41-npu_power--时钟门控与-dfs)
5. [辅助模块](#5-辅助模块)
   - 5.1 [axi_monitor — AXI 总线监视器](#51-axi_monitor--axi-总线监视器)
   - 5.2 [op_counter — 操作计数器](#52-op_counter--操作计数器)
6. [SoC 集成模块](#6-soc-集成模块)
   - 6.1 [soc_top — SoC 顶层](#61-soc_top--soc-顶层)
   - 6.2 [axi_lite_bridge — AXI-Lite 桥接器](#62-axi_lite_bridge--axi-lite-桥接器)
   - 6.3 [dram_model — DRAM 模型](#63-dram_model--dram-模型)
   - 6.4 [soc_mem — SoC SRAM](#64-soc_mem--soc-sram)

---

## 1. 系统概览

### 1.1 整体数据流

```
DRAM ──DMA读取──► PPBuf(W) ──┐
                           ├──► PE阵列 ──► Result FIFO ──DMA写回──► DRAM
DRAM ──DMA读取──► PPBuf(A) ──┘
```

### 1.2 模块层次

```
npu_top (顶层)
├── npu_axi_lite   — AXI4-Lite 寄存器文件（CPU 配置接口）
├── npu_ctrl       — NPU 控制器 FSM
├── npu_dma        — AXI4 Master DMA（3 通道：W/A/R）
├── pingpong_buf   ×2 — 权重/激活值乒乓缓冲区
├── pe_array       — M×N 脉动 PE 阵列
│   └── pe_top ×M×N
│       └── fp16_mul ×M×N
├── sync_fifo      — 结果 FIFO（PE → DMA）
└── npu_power      — 时钟门控 + DFS

soc_top (SoC 顶层)
├── PicoRV32       — RISC-V CPU
├── soc_mem        — 指令/数据 SRAM
├── dram_model     — 双端口 DRAM
├── axi_lite_bridge — iomem → AXI4-Lite 桥接
└── npu_top        — NPU 加速器
```

---

## 2. NPU 核心模块

### 2.1 npu_top — NPU 顶层集成

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/top/npu_top.v` |
| **功能** | NPU 顶层集成，连接所有子模块，实现完整数据通路 |
| **实例化** | `npu_axi_lite`, `npu_ctrl`, `npu_dma`, `pingpong_buf` ×2, `pe_array`, `sync_fifo`, `npu_power` |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ROWS | 4 | PE 阵列行数（M 维度） |
| COLS | 4 | PE 阵列列数（N 维度） |
| DATA_W | 16 | 数据位宽（FP16/INT8 均为 16 位输入） |
| ACC_W | 32 | 累加器位宽 |
| PPB_DEPTH | 64 | 乒乓缓冲区每 bank 深度（32-bit 字数） |
| PPB_THRESH | 16 | 提前启动阈值（DMA 填充多少字后 PE 可开始消费） |

#### 端口列表

##### 系统信号

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| sys_clk | input | 1 | 系统时钟 |
| sys_rst_n | input | 1 | 低电平异步复位 |

##### AXI4-Lite 从机端口（CPU 配置接口）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| s_axi_awaddr | input | 32 | 写地址 |
| s_axi_awvalid | input | 1 | 写地址有效 |
| s_axi_awready | output | 1 | 写地址就绪 |
| s_axi_wdata | input | 32 | 写数据 |
| s_axi_wstrb | input | 4 | 写字节使能 |
| s_axi_wvalid | input | 1 | 写数据有效 |
| s_axi_wready | output | 1 | 写数据就绪 |
| s_axi_bresp | output | 2 | 写响应 |
| s_axi_bvalid | output | 1 | 写响应有效 |
| s_axi_bready | input | 1 | 写响应就绪 |
| s_axi_araddr | input | 32 | 读地址 |
| s_axi_arvalid | input | 1 | 读地址有效 |
| s_axi_arready | output | 1 | 读地址就绪 |
| s_axi_rdata | output | 32 | 读数据 |
| s_axi_rresp | output | 2 | 读响应 |
| s_axi_rvalid | output | 1 | 读数据有效 |
| s_axi_rready | input | 1 | 读数据就绪 |

##### AXI4 主机端口（DMA 访问 DRAM）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| m_axi_awaddr | output | 32 | 写地址 |
| m_axi_awlen | output | 8 | 写突发长度 |
| m_axi_awsize | output | 3 | 写数据大小（log2(bytes)） |
| m_axi_awburst | output | 2 | 写突发类型（INCR） |
| m_axi_awvalid | output | 1 | 写地址有效 |
| m_axi_awready | input | 1 | 写地址就绪 |
| m_axi_wdata | output | ACC_W | 写数据 |
| m_axi_wstrb | output | ACC_W/8 | 写字节使能 |
| m_axi_wlast | output | 1 | 写最后一个拍 |
| m_axi_wvalid | output | 1 | 写数据有效 |
| m_axi_wready | input | 1 | 写数据就绪 |
| m_axi_bresp | input | 2 | 写响应 |
| m_axi_bvalid | input | 1 | 写响应有效 |
| m_axi_bready | output | 1 | 写响应就绪 |
| m_axi_araddr | output | 32 | 读地址 |
| m_axi_arlen | output | 8 | 读突发长度 |
| m_axi_arsize | output | 3 | 读数据大小 |
| m_axi_arburst | output | 2 | 读突发类型（INCR） |
| m_axi_arvalid | output | 1 | 读地址有效 |
| m_axi_arready | input | 1 | 读地址就绪 |
| m_axi_rdata | input | ACC_W | 读数据 |
| m_axi_rresp | input | 2 | 读响应 |
| m_axi_rvalid | input | 1 | 读数据有效 |
| m_axi_rready | output | 1 | 读数据就绪 |
| m_axi_rlast | input | 1 | 读最后一个拍 |

##### 中断

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| npu_irq | output | 1 | NPU 中断请求（计算完成时触发） |

#### 功能描述

npu_top 是 NPU 的顶层集成模块，内部连接关系：

1. **配置路径**：CPU 通过 AXI4-Lite → `npu_axi_lite` 寄存器文件 → 配置信号传入 `npu_ctrl`
2. **DMA 路径**：`npu_ctrl` 启动 `npu_dma`，DMA 从 DRAM 读取权重/激活值，写入 `pingpong_buf`
3. **计算路径**：`pingpong_buf` 输出数据到 `pe_array`，PE 计算后结果写入 `sync_fifo`
4. **写回路径**：`npu_dma` 从 `sync_fifo` 读取结果，通过 AXI4 Master 写回 DRAM
5. **控制路径**：`npu_ctrl` FSM 协调整个流程，包括乒乓缓冲区 swap/clear 信号

关键内部连接：
- `pe_w_in = {(COLS){w_ppb_rd_data}}` — 权重 PPBuf 输出广播到所有行
- `pe_a_in = {(ROWS){a_ppb_rd_data}}` — 激活值 PPBuf 输出广播到所有行
- `r_fifo_wr_en = pe_en && |pe_array_valid` — 任一列有效即写入 FIFO

---

### 2.2 npu_ctrl — NPU 控制器 FSM

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/ctrl/npu_ctrl.v` |
| **功能** | NPU 顶层控制器，协调 DMA 加载、PE 计算和结果写回 |
| **关键特性** | 乒乓缓冲区感知、cfg_start 上升沿检测、done 信号 sticky |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ROWS | 4 | PE 阵列行数 |
| COLS | 4 | PE 阵列列数 |
| DATA_W | 16 | 数据位宽 |
| ACC_W | 32 | 累加器位宽 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |
| ctrl_reg | input | 32 | 控制寄存器（来自 AXI-Lite 寄存器文件） |
| m_dim | input | 32 | 矩阵 M 维度 |
| n_dim | input | 32 | 矩阵 N 维度 |
| k_dim | input | 32 | 矩阵 K 维度 |
| w_addr | input | 32 | 权重 DRAM 基地址 |
| a_addr | input | 32 | 激活值 DRAM 基地址 |
| r_addr | input | 32 | 结果 DRAM 基地址 |
| arr_cfg | input | 8 | 阵列配置 |
| busy | output | 1 | 忙碌状态 |
| done | output | 1 | 完成状态（sticky，CPU 清除 ctrl_reg[0] 后才复位） |
| dma_w_start | output | 1 | 权重 DMA 启动（脉冲） |
| dma_w_done | input | 1 | 权重 DMA 完成 |
| dma_w_addr | output | 32 | 权重 DMA 地址 |
| dma_w_len | output | 16 | 权重 DMA 长度（字节） |
| dma_a_start | output | 1 | 激活值 DMA 启动（脉冲） |
| dma_a_done | input | 1 | 激活值 DMA 完成 |
| dma_a_addr | output | 32 | 激活值 DMA 地址 |
| dma_a_len | output | 16 | 激活值 DMA 长度（字节） |
| dma_r_start | output | 1 | 结果 DMA 启动（脉冲） |
| dma_r_done | input | 1 | 结果 DMA 完成 |
| dma_r_addr | output | 32 | 结果 DMA 地址 |
| dma_r_len | output | 16 | 结果 DMA 长度（字节） |
| pe_en | output | 1 | PE 使能 |
| pe_flush | output | 1 | PE 刷新累加器 |
| pe_mode | output | 1 | 数据类型（0=INT8, 1=FP16） |
| pe_stat | output | 1 | 静止模式（0=WS, 1=OS） |
| w_ppb_ready | input | 1 | 权重 PPBuf 就绪（填充 ≥ THRESHOLD） |
| w_ppb_empty | input | 1 | 权重 PPBuf 空 |
| a_ppb_ready | input | 1 | 激活值 PPBuf 就绪 |
| a_ppb_empty | input | 1 | 激活值 PPBuf 空 |
| w_ppb_swap | output | 1 | 权重 PPBuf 交换（脉冲） |
| a_ppb_swap | output | 1 | 激活值 PPBuf 交换（脉冲） |
| w_ppb_clear | output | 1 | 权重 PPBuf 清除（脉冲） |
| a_ppb_clear | output | 1 | 激活值 PPBuf 清除（脉冲） |
| irq | output | 1 | 中断请求 |

#### FSM 状态机

```
         cfg_start_rise
S_IDLE ──────────────► S_LOAD
                          │
                  dma_w_done && dma_a_done
                          │
                          ▼
                       S_PRELOAD (1 cycle)
                          │
                          ▼
                       S_COMPUTE
                          │
             dma_done && ppb_empty
                          │
                          ▼
                       S_DRAIN (pe_flush=1)
                          │
                          ▼
                       S_DRAIN2 (pipeline drain)
                          │
                          ▼
                     S_WRITE_BACK
                          │
                          ▼
                       S_WB_WAIT
                          │
                     dma_r_done
                          │
                          ▼
                       S_DONE (done=1, irq=1)
                          │
                          └──► S_IDLE
```

| 状态 | 编码 | 描述 |
|------|------|------|
| S_IDLE | 4'd0 | 等待 CPU 启动命令（cfg_start 上升沿） |
| S_LOAD | 4'd1 | DMA 加载权重+激活值到 PPBuf |
| S_PRELOAD | 4'd2 | 等待 1 周期让 swap 数据传播到 PE 输入 |
| S_COMPUTE | 4'd3 | PE 计算中，DMA 可能仍在填充 PPBuf |
| S_DRAIN | 4'd4 | PE flush 脉冲（输出累加结果） |
| S_DRAIN2 | 4'd9 | 流水线排空（flush 结果传播到 Stage-2） |
| S_WRITE_BACK | 4'd5 | 启动 DMA 写回结果 |
| S_WB_WAIT | 4'd6 | 等待 DMA 写回完成 |
| S_DONE | 4'd7 | 置位 done 和 irq，返回 IDLE |

#### 关键设计要点

- **cfg_start 上升沿检测**：通过 `cfg_start_d1` 寄存器延迟一拍，`cfg_start_rise = cfg_start && !cfg_start_d1`，防止 start 位仍为高时重新触发
- **done sticky 信号**：done 在 S_DONE 置 1，直到 CPU 写 ctrl_reg[0]=0 才清除
- **DMA 长度计算**：
  - 权重：`n_dim * k_dim * data_bytes`
  - 激活值：`m_dim * k_dim * data_bytes`
  - 结果：`m_dim * n_dim * (ACC_W/8)`
- **中止支持**：S_LOAD 和 S_COMPUTE 状态下 cfg_abort 可中止当前运算

---

### 2.3 npu_axi_lite — AXI4-Lite 寄存器文件

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/axi/npu_axi_lite.v` |
| **功能** | AXI4-Lite 从机接口 + NPU 配置寄存器文件 |
| **实例化** | 在 `npu_top` 中实例化为 `u_axi_lite` |

#### 端口列表

##### AXI4-Lite 从机接口

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| aclk | input | 1 | AXI 时钟 |
| aresetn | input | 1 | 复位 |
| awaddr | input | 32 | 写地址 |
| awvalid | input | 1 | 写地址有效 |
| awready | output | 1 | 写地址就绪 |
| wdata | input | 32 | 写数据 |
| wstrb | input | 4 | 写字节使能 |
| wvalid | input | 1 | 写数据有效 |
| wready | output | 1 | 写数据就绪 |
| bresp | output | 2 | 写响应（固定 OKAY） |
| bvalid | output | 1 | 写响应有效 |
| bready | input | 1 | 写响应就绪 |
| araddr | input | 32 | 读地址 |
| arvalid | input | 1 | 读地址有效 |
| arready | output | 1 | 读地址就绪 |
| rdata | output | 32 | 读数据 |
| rresp | output | 2 | 读响应（固定 OKAY） |
| rvalid | output | 1 | 读数据有效 |
| rready | input | 1 | 读数据就绪 |

##### NPU 配置输出

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| ctrl_reg | output | 32 | 控制寄存器 |
| m_dim | output | 32 | 矩阵 M 维度 |
| n_dim | output | 32 | 矩阵 N 维度 |
| k_dim | output | 32 | 矩阵 K 维度 |
| w_addr | output | 32 | 权重 DRAM 地址 |
| a_addr | output | 32 | 激活值 DRAM 地址 |
| r_addr | output | 32 | 结果 DRAM 地址 |
| arr_cfg | output | 8 | 阵列配置 |
| clk_div | output | 3 | 时钟分频选择 |
| cg_en | output | 1 | 时钟门控使能 |
| status_busy | input | 1 | NPU 忙碌状态（来自 ctrl） |
| status_done | input | 1 | NPU 完成状态（来自 ctrl） |
| irq_flag | input | 1 | 中断标志（来自 ctrl） |
| npu_irq | output | 1 | NPU 中断输出 |

#### 寄存器地址映射

| 偏移 | 名称 | 位定义 | 说明 |
|------|------|--------|------|
| 0x00 | CTRL | [0]=start, [1]=abort, [3:2]=data_mode, [5:4]=stat_mode | 控制寄存器 |
| 0x04 | STATUS | [0]=busy, [1]=done | 状态寄存器（只读） |
| 0x08 | INT_EN | [0]=enable | 中断使能 |
| 0x0C | INT_CLR | [0]=clear | 中断清除（写 1 清除） |
| 0x10 | M_DIM | [31:0] | 矩阵 M 维度 |
| 0x14 | N_DIM | [31:0] | 矩阵 N 维度 |
| 0x18 | K_DIM | [31:0] | 矩阵 K 维度 |
| 0x20 | W_ADDR | [31:0] | 权重基地址 |
| 0x24 | A_ADDR | [31:0] | 激活值基地址 |
| 0x28 | R_ADDR | [31:0] | 结果基地址 |
| 0x30 | ARR_CFG | [3:0]=rows, [7:4]=cols | 阵列配置 |
| 0x34 | CLK_DIV | [2:0]=div_sel | 时钟分频 |
| 0x38 | CG_EN | [0]=enable | 时钟门控 |

#### 功能描述

1. **写 FSM**：先锁存 AW 地址，等待 W 通道数据到达后同时完成写操作（1 周期完成写事务）
2. **读 FSM**：先锁存 AR 地址，下一周期返回 R 数据
3. **中断逻辑**：`irq_flag && int_en_reg[0]` 置位 `int_pending`，写 `INT_CLR` 寄存器清除
4. **地址默认值**：未定义地址读返回 `0xDEADBEEF`

---

### 2.4 npu_dma — DMA 控制器

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/axi/npu_dma.v` |
| **功能** | AXI4 Master DMA，3 通道（权重读取、激活值读取、结果写回） |
| **关键特性** | W→A 链式读取、共享 AXI 总线时分复用、内部 Result FIFO |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DATA_W | 32 | AXI 数据总线宽度 |
| PE_DATA_W | 16 | PE 数据宽度 |
| BURST_MAX | 16 | 最大 AXI 突发长度 |
| PPB_DEPTH | 32 | PPBuf 深度 |
| PPB_THRESH | 16 | PPBuf 提前启动阈值 |
| R_FIFO_DEPTH | 64 | 结果 FIFO 深度 |

#### 端口列表

##### 通用

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |

##### 通道 0：权重（DRAM → PPBuf → PE）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| w_start | input | 1 | 权重通道启动 |
| w_base_addr | input | 32 | 权重基地址 |
| w_len_bytes | input | 16 | 权重字节长度 |
| w_done | output | 1 | 权重 DMA 完成 |
| w_ppb_wr_en | output | 1 | 权重 PPBuf 写使能 |
| w_ppb_wr_data | output | 32 | 权重 PPBuf 写数据 |
| w_ppb_full | input | 1 | 权重 PPBuf 满 |
| w_ppb_buf_ready | input | 1 | 权重 PPBuf 就绪（≥ THRESHOLD） |
| w_ppb_buf_empty | input | 1 | 权重 PPBuf 空 |
| w_ppb_drain_done | input | 1 | 权重 PPBuf 排空完成（当前恒接 1） |

##### 通道 1：激活值（DRAM → PPBuf → PE）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| a_start | input | 1 | 激活值通道启动 |
| a_base_addr | input | 32 | 激活值基地址 |
| a_len_bytes | input | 16 | 激活值字节长度 |
| a_done | output | 1 | 激活值 DMA 完成 |
| a_ppb_wr_en | output | 1 | 激活值 PPBuf 写使能 |
| a_ppb_wr_data | output | 32 | 激活值 PPBuf 写数据 |
| a_ppb_full | input | 1 | 激活值 PPBuf 满 |
| a_ppb_buf_ready | input | 1 | 激活值 PPBuf 就绪 |
| a_ppb_buf_empty | input | 1 | 激活值 PPBuf 空 |
| a_ppb_drain_done | input | 1 | 激活值 PPBuf 排空完成（当前恒接 1） |

##### 通道 2：结果（PE → FIFO → DRAM）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| r_start | input | 1 | 结果通道启动 |
| r_base_addr | input | 32 | 结果基地址 |
| r_len_bytes | input | 16 | 结果字节长度 |
| r_done | output | 1 | 结果 DMA 完成 |
| r_fifo_wr_en | input | 1 | 结果 FIFO 写使能（来自 PE） |
| r_fifo_din | input | 32 | 结果 FIFO 写数据 |
| r_fifo_full | output | 1 | 结果 FIFO 满 |

##### AXI4 Master（共享总线）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| m_axi_awaddr | output | 32 | 写地址 |
| m_axi_awlen | output | 8 | 写突发长度 |
| m_axi_awsize | output | 3 | 写数据大小 |
| m_axi_awburst | output | 2 | 写突发类型（INCR） |
| m_axi_awvalid | output | 1 | 写地址有效 |
| m_axi_awready | input | 1 | 写地址就绪 |
| m_axi_wdata | output | 32 | 写数据 |
| m_axi_wstrb | output | 4 | 写字节使能 |
| m_axi_wlast | output | 1 | 写最后一拍 |
| m_axi_wvalid | output | 1 | 写数据有效 |
| m_axi_wready | input | 1 | 写数据就绪 |
| m_axi_bresp | input | 2 | 写响应 |
| m_axi_bvalid | input | 1 | 写响应有效 |
| m_axi_bready | output | 1 | 写响应就绪 |
| m_axi_araddr | output | 32 | 读地址 |
| m_axi_arlen | output | 8 | 读突发长度 |
| m_axi_arsize | output | 3 | 读数据大小 |
| m_axi_arburst | output | 2 | 读突发类型（INCR） |
| m_axi_arvalid | output | 1 | 读地址有效 |
| m_axi_arready | input | 1 | 读地址就绪 |
| m_axi_rdata | input | 32 | 读数据 |
| m_axi_rresp | input | 2 | 读响应 |
| m_axi_rvalid | input | 1 | 读数据有效 |
| m_axi_rready | output | 1 | 读数据就绪 |
| m_axi_rlast | input | 1 | 读最后一拍 |

#### DMA FSM 状态

| 状态 | 编码 | 描述 |
|------|------|------|
| IDLE | 4'd0 | 空闲，等待启动信号（优先级：R_WRITE > WA_READ > W_READ > A_READ） |
| W_READ | 4'd1 | 从 DRAM 读取权重，写入 PPBuf，完成后链式跳转到 A_READ |
| A_READ | 4'd3 | 从 DRAM 读取激活值，写入 PPBuf |
| WA_READ | 4'd5 | 交错读取权重和激活值（交替突发） |
| R_WRITE | 4'd7 | 从 FIFO 读取 PE 结果，写入 DRAM |

#### 功能描述

1. **权重读取（W_READ）**：AXI 读事务从 `w_base_addr` 开始，每次读取 32-bit 数据写入 `w_ppb`。读取完成后，如果 `a_len_bytes > 0` 则自动跳转到 `A_READ`（W→A 链式读取）
2. **激活值读取（A_READ）**：同权重读取，从 `a_base_addr` 开始
3. **结果写回（R_WRITE）**：从内部 FIFO 读取数据，通过 AXI 写事务写入 `r_base_addr` 开始的 DRAM 区域
4. **PPBuf 写使能逻辑**：
   - `w_ppb_wr_en = (dma_state == W_READ || (dma_state == WA_READ && reading_w)) && m_axi_rvalid && !w_ppb_full`
   - AXI 读取数据直接送入 PPBuf
5. **内部 FIFO**：使用 `sync_fifo` 实例，深度 64，PE 有效输出时写入

---

### 2.5 pe_top — 处理单元 (PE)

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/pe/pe_top.v` |
| **功能** | 支持 WS/OS 双模式、FP16/INT8 混合精度的乘累加处理单元 |
| **流水线** | 3 级：Stage-0（输入寄存）→ Stage-1（乘法）→ Stage-2（累加/输出） |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DATA_W | 16 | 数据位宽（FP16/INT8 输入均为 16-bit） |
| ACC_W | 32 | 累加器位宽 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |
| mode | input | 1 | 数据类型（0=INT8, 1=FP16） |
| stat_mode | input | 1 | 静止模式（0=Weight-Stationary, 1=Output-Stationary） |
| en | input | 1 | 流水线使能 |
| flush | input | 1 | 刷新累加器（脉冲） |
| w_in | input | DATA_W | 权重输入（INT8 使用 [7:0]） |
| a_in | input | DATA_W | 激活值输入（INT8 使用 [7:0]） |
| acc_in | input | ACC_W | 输入部分和（OS 模式，当前未使用） |
| acc_out | output | ACC_W | 累加结果输出 |
| valid_out | output | 1 | 输出有效（flush 周期产生） |

#### 流水线阶段详细

##### Stage-0：输入寄存

| 信号 | 位宽 | 说明 |
|------|------|------|
| s0_w | DATA_W | 锁存的权重 |
| s0_a | DATA_W | 锁存的激活值 |
| s0_valid | 1 | 有效标志 |
| s0_flush | 1 | flush 标志（传递到下一级） |
| s0_mode | 1 | mode 传递到下一级 |
| s0_stat | 1 | stat_mode 传递到下一级 |

- en=1 时锁存输入，置 s0_valid=1
- en=0 时 s0_valid=0（防止无效数据污染 Stage-1）

##### Stage-1：乘法

| 信号 | 位宽 | 说明 |
|------|------|------|
| s1_mul | ACC_W | 乘法结果 |
| s1_valid | 1 | 有效标志 |
| s1_flush | 1 | flush 标志 |
| s1_stat | 1 | stat_mode |
| s1_acc_in | ACC_W | acc_in 传递 |

**INT8 乘法**：
```verilog
wire signed [7:0] int8_w = s0_w[7:0];
wire signed [7:0] int8_a = s0_a[7:0];
wire signed [15:0] int8_mul_16 = $signed(int8_w) * $signed(int8_a);
assign int8_prod = {{(ACC_W-16){int8_mul_16[15]}}, int8_mul_16};
```
- 正确的符号扩展：基于 `int8_mul_16[15]`（乘积实际符号位）

**FP16 乘法**：实例化 `fp16_mul` 模块

- 仅当 s0_valid=1 时更新 s1_mul，避免无效数据污染

##### Stage-2：累加/输出

| 信号 | 位宽 | 说明 |
|------|------|------|
| os_acc | ACC_W | Output-Stationary 累加器 |
| acc_out | ACC_W | 累加结果输出 |
| valid_out | 1 | 输出有效 |

- **正常计算**（s1_valid=1 && !s1_flush）：`os_acc <= os_acc + s1_mul`
- **刷新输出**（s1_valid=1 && s1_flush）：`acc_out <= os_acc`，`valid_out <= 1`，同时用 `s1_mul` 初始化下一轮累加

---

### 2.6 fp16_mul — FP16 乘法器

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/pe/fp16_mul.v` |
| **功能** | IEEE 754 半精度（FP16）乘法器，组合逻辑输出（寄存器延迟在 pe_top Stage-1） |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ACC_W | 32 | 输出位宽 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟（保留，未使用，用于接口兼容） |
| rst_n | input | 1 | 复位（保留，未使用） |
| en | input | 1 | 使能（保留，未使用） |
| a | input | 16 | FP16 操作数 A |
| b | input | 16 | FP16 操作数 B |
| result | output | ACC_W | 乘积结果（FP16 零扩展到 ACC_W，**不进行符号扩展**） |

#### FP16 格式

```
[15]    [14:10]   [9:0]
符号    指数      尾数
(S)     (E)       (M)
偏移量 = 15
```

#### 功能描述

1. **拆包**：从两个 16-bit 输入提取符号、指数、尾数
2. **特殊情况检测**：NaN、Inf、Zero 及其组合（如 0×Inf=NaN）
3. **次正规数输入支持**（动态隐式位）：
   - 次正规数输入（exp=0, mant≠0）：隐式位=0，有效指数=1
   - 正规数输入（exp≠0）：隐式位=1，有效指数=存储值
4. **乘法**：11-bit 尾数相乘，结果 22-bit
5. **完整 22-bit LZC**（前导零计数）：
   - 范围 0..22（22=全零）
   - 支持次正规数乘积的正确归一化
6. **Barrel 右移归一化**：右移量 `RS = 11 - lzc`，归一化后隐式 1 在 bit[10]
7. **指数计算**：`biased_exp = eff_ea + eff_eb - 14 - lzc`
8. **正常路径**（biased_exp > 0）：
   - Guard/Sticky 位提取 + RN 舍入（round-to-nearest-even）
   - 溢出钳位到 Inf
9. **次正规数输出路径**（biased_exp ≤ 0）——**渐进下溢**：
   - 对归一化后的 11-bit mantissa 继续右移 `extra_shift = 1 - biased_exp` 位
   - 使用独立的 barrel shifter + guard/sticky RN 舍入
   - `biased_exp < -10`：结果过小，flush to zero
   - 舍入导致 mantissa 溢出到 0x400 时，自动进位到最小正规数（exp=1, mant=0）
10. **打包**：组合符号、指数、尾数为 FP16 结果
11. **零扩展**：FP16 结果**零扩展**到 ACC_W 位宽（**禁止符号扩展**，避免破坏 IEEE 754 位模式）

---

### 2.7 pe_array — PE 阵列

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/array/pe_array.v` |
| **功能** | M×N 脉动阵列，激活值水平流动，部分和垂直流动 |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ROWS | 4 | 行数（M 维度） |
| COLS | 4 | 列数（N 维度） |
| DATA_W | 16 | 数据位宽 |
| ACC_W | 32 | 累加器位宽 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |
| mode | input | 1 | 全局数据类型（0=INT8, 1=FP16） |
| stat_mode | input | 1 | 全局静止模式（0=WS, 1=OS） |
| en | input | 1 | 使能 |
| flush | input | 1 | 刷新 |
| w_in | input | COLS×DATA_W | 权重输入（每列一个，广播到所有行） |
| act_in | input | ROWS×DATA_W | 激活值输入（每行一个） |
| acc_in | input | COLS×ACC_W | 部分和输入（每列一个，当前接 0） |
| acc_out | output | COLS×ACC_W | 累加输出（每列一个，来自最底行） |
| valid_out | output | COLS | 输出有效（每列一个） |

#### 拓扑结构

```
act_in[0] ──► PE[0][0] ──► PE[0][1] ──► ... ──► PE[0][N-1]
                 │                               │
act_in[1] ──► PE[1][0] ──► PE[1][1] ──► ... ──► PE[1][N-1]
                 │                               │
       ...         ...                             ...
act_in[M-1]► PE[M-1][0]─► PE[M-1][1]─► ... ──► PE[M-1][N-1]
                 │                               │
              acc_out[0]                      acc_out[N-1]
```

- **激活值**：水平流动（行方向），每个 PE 内部有一级寄存器延迟
- **部分和**：垂直流动（列方向），直接传递
- **权重**：按列广播，所有行使用相同权重
- **输出**：每列最底行 PE 的 acc_out 和 valid_out

---

### 2.8 array_ctrl — 阵列控制器

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/array/array_ctrl.v` |
| **功能** | 简化的 PE 阵列控制器，基于计数器的调度器 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |
| start | input | 1 | 启动计算 |
| busy | output | 1 | 忙碌 |
| pe_en | output | 1 | PE 使能 |
| pe_flush | output | 1 | PE 刷新 |
| pe_mode | output | 1 | PE 模式 |
| pe_stat | output | 1 | PE 静止模式 |
| pe_w_out | output | COLS×DATA_W | PE 权重输出 |
| pe_act_out | output | ROWS×DATA_W | PE 激活值输出 |
| pe_acc_out | output | COLS×ACC_W | PE 累加输出 |
| pe_result | input | COLS×ACC_W | PE 结果 |
| pe_valid | input | COLS | PE 有效 |

#### 功能描述

基于计数器的简单调度器：
- start 信号触发后开始计数
- `load_done = cycle_cnt >= ROWS + COLS`：开始 flush
- `drain_done = cycle_cnt >= ROWS + COLS + 5`：完成
- 当前版本未实际驱动数据总线，仅控制 PE 的 en/flush 信号

---

## 3. 缓冲与存储模块

### 3.1 pingpong_buf — 乒乓缓冲区

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/buf/pingpong_buf.v` |
| **功能** | 双 bank 缓冲区，支持子字读取，实现 DMA 加载与 PE 计算重叠 |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DATA_W | 32 | 写数据位宽（DMA 侧，32-bit AXI 字） |
| DEPTH | 64 | 每 bank 深度（字数，必须为 2 的幂） |
| OUT_WIDTH | 16 | 读数据位宽（PE 侧） |
| THRESHOLD | 16 | DMA 填充多少字后 buf_ready 置位 |
| SUBW | 4 | 每字子字数（INT8：32/8=4） |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |
| wr_en | input | 1 | 写使能（DMA 侧） |
| wr_data | input | DATA_W | 写数据 |
| rd_en | input | 1 | 读使能（PE 侧） |
| rd_data | output | OUT_WIDTH | 读数据（符号扩展后的子字） |
| swap | input | 1 | 切换 bank（脉冲） |
| clear | input | 1 | 复位所有指针（脉冲） |
| buf_empty | output | 1 | 读者 bank 空（无更多子字可读） |
| buf_full | output | 1 | 写者 bank 满 |
| buf_ready | output | 1 | 写者 bank 填充 ≥ THRESHOLD |
| rd_fill | output | $clog2(DEPTH×SUBW)+1 | 剩余子字数 |
| wr_fill | output | $clog2(DEPTH)+1 | 已写字数 |

#### Bank 切换机制

```
初始状态：wr_sel=0(BufA), rd_sel=0(BufA)

swap 时：
  wr_sel ← ~wr_sel （写入切换到新 bank）
  rd_sel ← wr_sel (旧值) （读取切换到刚被写满的 bank）

效果：DMA 开始写入 BufB，PE 从 BufA 消费数据
```

#### 子字读取机制

```
32-bit 字: [byte3 | byte2 | byte1 | byte0]

rd_sub=0 → byte0 [7:0]  → 符号扩展到 16-bit
rd_sub=1 → byte1 [15:8] → 符号扩展到 16-bit
rd_sub=2 → byte2 [23:16]→ 符号扩展到 16-bit
rd_sub=3 → byte3 [31:24]→ 符号扩展到 16-bit
```

#### 关键设计要点

- **空缓冲区输出 0**：`rd_data = buf_empty ? {OUT_WIDTH{1'b0}} : rd_data_raw`，防止 PE 锁存陈旧数据
- **swap 时重置新写者 bank 指针**：写入切换到新 bank 时清零该 bank 的 wr_ptr 和 wr_fill
- **swap 时复制填充计数到读侧**：`rd_fill = wr_fill × SUBW`

---

### 3.2 sync_fifo — 同步 FIFO

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/common/fifo.v` |
| **功能** | 参数化单时钟同步 FIFO，用于 PE 结果缓冲 |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DATA_W | 16 | 数据位宽 |
| DEPTH | 16 | FIFO 深度（必须为 2 的幂） |
| ALMOST_FULL | 2 | almost_full 阈值（剩余槽位数） |
| ALMOST_EMPTY | 2 | almost_empty 阈值（剩余条目数） |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |
| wr_en | input | 1 | 写使能 |
| wr_data | input | DATA_W | 写数据 |
| full | output | 1 | 满标志 |
| almost_full | output | 1 | 几乎满 |
| rd_en | input | 1 | 读使能 |
| rd_data | output | DATA_W | 读数据（组合逻辑，无延迟） |
| empty | output | 1 | 空标志 |
| almost_empty | output | 1 | 几乎空 |
| fill_count | output | $clog2(DEPTH)+1 | 填充计数 |

#### 功能描述

- 环形缓冲区实现，头/尾指针各比地址多 1 bit（MSB 区分满/空）
- 读数据为组合逻辑（0 周期读延迟）
- `full = head == tail && MSB 不同`
- `empty = head == tail && MSB 相同`

---

## 4. 电源管理模块

### 4.1 npu_power — 时钟门控与 DFS

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/power/npu_power.v` |
| **功能** | 动态频率缩放（DFS）+ 按行/列时钟门控 |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ROWS | 4 | PE 行数 |
| COLS | 4 | PE 列数 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 系统时钟 |
| rst_n | input | 1 | 复位 |
| div_sel | input | 3 | DFS 配置 |
| row_cg_en | input | ROWS | 每行时钟门控（1=门控/关闭） |
| col_cg_en | input | COLS | 每列时钟门控（1=门控/关闭） |
| npu_clk | output | 1 | 分频时钟 |
| row_clk_gated | output | ROWS | 行门控时钟 |
| col_clk_gated | output | COLS | 列门控时钟 |

#### DFS 配置

| div_sel | 分频 | 说明 |
|---------|------|------|
| 3'b000 | ÷1 | 直通（bypass） |
| 3'b001 | ÷2 | 2 分频 |
| 3'b010 | ÷4 | 4 分频 |
| 3'b011 | ÷8 | 8 分频 |

#### 功能描述

- **DFS**：基于计数器的简单时钟分频，`npu_clk = (div_sel==0) ? clk : dfs_clk_r`
- **行时钟门控**：`row_clk_gated[i] = row_cg_en[i] ? 1'b0 : npu_clk`（行为模型，ASIC 中替换为 ICG 单元）
- **列时钟门控**：同上
- 当前 npu_top 中 `row_cg = {ROWS{~pe_en}}`，PE 不工作时自动门控

---

## 5. 辅助模块

### 5.1 axi_monitor — AXI 总线监视器

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/common/axi_monitor.v` |
| **功能** | 监视 AXI4-Lite 和 AXI4 Master 端口，统计带宽和延迟 |

#### 端口列表

##### AXI4-Lite 监视输入

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| s_awvalid ~ s_rready | input | 各1~8 | AXI4-Lite 握手信号 |

##### AXI4 Master 监视输入

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| m_awlen, m_arlen | input | 8 | 突发长度 |
| m_awvalid ~ m_rready, m_awlast, m_rlast | input | 各1 | AXI4 握手信号 |

##### 统计输出

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| s_axi_wr_cnt | output | 32 | AXI4-Lite 写事务数 |
| s_axi_rd_cnt | output | 32 | AXI4-Lite 读事务数 |
| s_axi_wr_beats | output | 32 | AXI4-Lite 写数据拍数 |
| s_axi_rd_beats | output | 32 | AXI4-Lite 读数据拍数 |
| s_axi_wr_lat | output | 32 | AXI4-Lite 写延迟总和（周期） |
| s_axi_rd_lat | output | 32 | AXI4-Lite 读延迟总和（周期） |
| m_axi_wr_cnt | output | 32 | AXI4 Master 写突发数 |
| m_axi_rd_cnt | output | 32 | AXI4 Master 读突发数 |
| m_axi_wr_bytes | output | 32 | AXI4 Master 总写字节数 |
| m_axi_rd_bytes | output | 32 | AXI4 Master 总读字节数 |
| m_axi_wr_beats | output | 32 | AXI4 Master 写数据拍数 |
| m_axi_rd_beats | output | 32 | AXI4 Master 读数据拍数 |
| m_axi_wr_lat | output | 32 | AXI4 Master 写延迟总和 |
| m_axi_rd_lat | output | 32 | AXI4 Master 读延迟总和 |
| total_cycles | output | 32 | 监视持续周期数 |
| m_axi_rd_bw | output | 32 | 读带宽（字节/周期 ×1000） |
| m_axi_wr_bw | output | 32 | 写带宽（字节/周期 ×1000） |

---

### 5.2 op_counter — 操作计数器

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/common/op_counter.v` |
| **功能** | NPU 性能分析器，跟踪 MAC 操作、PE 利用率、效率指标 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| rst_n | input | 1 | 复位 |
| pe_en | input | 1 | PE 使能 |
| pe_flush | input | 1 | PE flush |
| ctrl_busy | input | 1 | 控制器忙碌 |
| ctrl_done | input | 1 | 控制器完成 |
| dma_w_done | input | 1 | 权重 DMA 完成 |
| dma_a_done | input | 1 | 激活值 DMA 完成 |
| dma_r_done | input | 1 | 结果 DMA 完成 |
| pe_valid | input | COLS | PE 阵列有效输出 |
| m_dim, n_dim, k_dim | input | 32 | 矩阵维度配置 |
| total_mac_ops | output | 64 | 总 MAC 操作数 |
| total_pe_cycles | output | 32 | PE 使能周期数 |
| total_busy_cycles | output | 32 | NPU 忙碌总周期数 |
| total_compute_cycles | output | 32 | 计算状态周期数 |
| total_dma_cycles | output | 32 | DMA 周期数 |
| active_pe_cnt | output | 32 | 当前活跃 PE 数 |
| peak_active_pe | output | 32 | 峰值 PE 利用率 |
| fsm_transitions | output | 32 | FSM 状态转换次数 |
| utilization_pct | output | 32 | PE 利用率 %（×100） |
| mac_per_cycle | output | 32 | 平均每周期 MAC 数（×100） |
| efficiency_pct | output | 32 | 整体效率 %（×100） |

---

## 6. SoC 集成模块

### 6.1 soc_top — SoC 顶层

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/soc/soc_top.v` |
| **功能** | SoC 顶层：PicoRV32 CPU + NPU 加速器 + SRAM + DRAM |
| **实例化** | `picorv32`, `soc_mem`, `dram_model`, `axi_lite_bridge`, `npu_top` |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MEM_WORDS | 1024 | SRAM 大小（4KB） |
| DRAM_WORDS | 15360 | DRAM 大小（~60KB） |
| NPU_ROWS | 4 | NPU 行数 |
| NPU_COLS | 4 | NPU 列数 |
| NPU_DATA_W | 16 | NPU 数据位宽 |
| NPU_ACC_W | 32 | NPU 累加器位宽 |
| NPU_PPB_DEPTH | 32 | NPU PPBuf 深度 |
| NPU_PPB_THRESH | 16 | NPU PPBuf 阈值 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 系统时钟 |
| rst_n | input | 1 | 复位 |

#### 地址映射

| 地址范围 | 区域 | 访问者 | 说明 |
|----------|------|--------|------|
| 0x0000_0000 - 0x0000_0FFF | SRAM | CPU only | 指令 + 数据存储（4KB） |
| 0x0000_0100 - 0x0000_FFFF | DRAM | CPU + NPU DMA | NPU 数据区（权重、激活值、结果） |
| 0x0200_0000 - 0x0200_003F | NPU 寄存器 | CPU only | 通过 AXI-Lite Bridge 配置 NPU |

#### 功能描述

1. **地址译码**：PicoRV32 的 `mem_if` 地址经过译码后分别访问 SRAM、DRAM 或 NPU
2. **SRAM**：PicoRV32 的指令和数据存储，使用 `soc_mem` 模块
3. **DRAM**：CPU 侧通过简单 valid/ready 接口访问，NPU DMA 侧通过 AXI4 接口访问
4. **NPU 配置**：CPU 访问 NPU 基地址以上的地址，通过 `axi_lite_bridge` 转换为 AXI4-Lite 协议
5. **中断**：NPU 完成时 `npu_irq → cpu_irq[7]`

---

### 6.2 axi_lite_bridge — AXI-Lite 桥接器

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/soc/axi_lite_bridge.v` |
| **功能** | PicoRV32 iomem 接口 → AXI4-Lite 协议转换 |

#### 端口列表

##### PicoRV32 iomem 侧

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| iomem_valid | input | 1 | 存储器访问有效 |
| iomem_ready | output | 1 | 存储器就绪 |
| iomem_wstrb | input | 4 | 写字节使能（非零=写，零=读） |
| iomem_addr | input | 32 | 地址（含 NPU 基地址） |
| iomem_wdata | input | 32 | 写数据 |
| iomem_rdata | output | 32 | 读数据 |

##### AXI4-Lite Master 侧（到 NPU）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| m_axi_awaddr ~ m_axi_rready | 各方向 | 各位 | 标准 AXI4-Lite 信号 |

##### 配置

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| npu_base_addr | input | 32 | NPU 寄存器基地址（如 0x02000000） |

#### FSM

| 状态 | 编码 | 描述 |
|------|------|------|
| S_IDLE | 2'd0 | 等待 iomem_valid |
| S_WRITE | 2'd1 | AW+W 同时发出 |
| S_READ | 2'd2 | AR 发出，等待 R 数据 |

#### 关键设计

- `npu_offset = iomem_addr - npu_base_addr`：在 AXI 地址中剥离基地址偏移
- 写操作：iomem_wstrb 非零时进入 S_WRITE
- 读操作：iomem_wstrb 为零时进入 S_READ

---

### 6.3 dram_model — DRAM 模型

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/soc/dram_model.v` |
| **功能** | 行为级双端口 DRAM 模型 |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| WORDS | 15360 | 字数（~60KB） |
| DATA_W | 32 | 数据位宽 |

#### 端口列表

##### 端口 1：CPU 侧（简单存储器接口）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| cpu_valid | input | 1 | 访问有效 |
| cpu_ready | output | 1 | 就绪（恒为 1） |
| cpu_we | input | 1 | 写使能 |
| cpu_wstrb | input | 4 | 字节使能 |
| cpu_addr | input | 32 | 地址 |
| cpu_wdata | input | 32 | 写数据 |
| cpu_rdata | output | 32 | 读数据 |

##### 端口 2：NPU DMA 侧（AXI4 从机接口）

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| axi_awaddr ~ axi_rready | 各方向 | 各位 | 标准 AXI4 信号 |

#### 功能描述

- 两个端口共享同一存储阵列
- CPU 端口：始终就绪，支持字节通道写
- AXI 端口：支持突发读/写，AW+W 顺序握手，B 响应，AR+R 突发读取
- 仿真中无需仲裁，时序由 TB 自然串行化

---

### 6.4 soc_mem — SoC SRAM

| 属性 | 说明 |
|------|------|
| **文件** | `rtl/soc/soc_mem.v` |
| **功能** | PicoRV32 的指令 + 数据 SRAM |

#### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| WORDS | 1024 | 32-bit 字数（4KB） |
| INIT_HEX | "" | 可选 hex 初始化文件 |

#### 端口列表

| 端口名 | 方向 | 位宽 | 说明 |
|--------|------|------|------|
| clk | input | 1 | 时钟 |
| wen | input | 4 | 字节通道写使能 |
| addr | input | 22 | 字地址（byte_addr >> 2） |
| wdata | input | 32 | 写数据 |
| rdata | output | 32 | 读数据（1 周期延迟） |

#### 功能描述

- 单端口 SRAM，读有 1 周期延迟
- 支持字节通道写（wen[3:0] 分别控制 4 个字节）
- 兼容 picosoc_mem 接口

---

## 附录 A：参数化配置参考

### 默认配置

| 参数 | 值 | 说明 |
|------|----|------|
| ROWS × COLS | 4×4 | 16 个 PE |
| DATA_W | 16 | 支持 FP16/INT8 |
| ACC_W | 32 | 32-bit 累加器 |
| PPB_DEPTH | 64 | 每 bank 64 个 32-bit 字 |
| PPB_THRESH | 16 | DMA 填充 16 字后 PE 可开始 |
| FIFO_DEPTH | 64 | 结果 FIFO 64 深度 |
| BURST_MAX | 16 | AXI 最大突发 16 拍 |

### 点积测试配置

| 参数 | 值 | 说明 |
|------|----|------|
| ROWS × COLS | 1×1 | 单 PE |
| DATA_W | 16 | INT8 模式 |
| ACC_W | 32 | 32-bit 累加 |
| M=1, N=1, K=4 | — | 1×4 点积 |

---

## 附录 B：仿真验证状态

| 测试 | 模式 | 数据 | 期望 | 实际 | 状态 |
|------|------|------|------|------|------|
| tb_classifier | INT8 OS | [3,7,-2,5]·[10,20,30,40] | 310 | 310 | ✅ PASS |
| tb_pe_top (4场景) | INT8/FP16 × WS/OS | 各自测试数据 | — | — | ✅ PASS |
| tb_npu_top | INT8 WS | 4×4 矩阵乘 | — | — | ✅ PASS |
