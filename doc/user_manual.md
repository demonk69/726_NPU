# NPU_prj 用户手册

> 嵌入式低功耗 NPU 加速器 IP — 使用与仿真指南
>
> 最后更新：2026-04-01（新增 SoC 集成 + Ping-Pong Buffer）

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境搭建](#2-环境搭建)
3. [目录结构](#3-目录结构)
4. [快速仿真](#4-快速仿真)
5. [寄存器编程指南](#5-寄存器编程指南)
6. [SoC 集成指南](#6-soc-集成指南)
7. [Ping-Pong Buffer 使用](#7-ping-pong-buffer-使用)
8. [仿真测试详解](#8-仿真测试详解)
9. [性能分析与监控](#9-性能分析与监控)
10. [模块参考](#10-模块参考)
11. [扩展与定制](#11-扩展与定制)
12. [常见问题](#12-常见问题)

---

## 1. 项目简介

NPU_prj 是一款面向**边缘计算**的低功耗 NPU 加速器 IP，通过 AXI 总线挂载于 SoC（Cortex-M0 / RISC-V）。

### 核心特性

| 特性 | 说明 |
|---|---|
| 精度 | INT8 / INT16 / FP16 混合精度 |
| 阵列 | 4×4 脉动 PE 阵列（参数可扩展） |
| 数据流 | Weight-Stationary (WS) / Output-Stationary (OS) |
| PE 流水线 | 3 级（Input Reg → MUL → ACC） |
| 接口 | AXI4-Lite（寄存器配置）+ AXI4 Master（DMA） |
| 电源管理 | 时钟门控（per-row/col）+ DFS（÷1/2/4/8） |
| 目标算力 | ≥ 0.5 TOPS INT8 |

### 系统架构

```
┌──────────────────────────────────────────────────────┐
│  SoC (Cortex-M0 / RISC-V)                           │
│       │ AXI Bus                                      │
└───────┼──────────────────────────────────────────────┘
        │
   AXI4-Lite ──► NPU Top
        │         ┌────────────────────────────────┐
        │         │  AXI-Lite 寄存器 (14 个)       │
        │         └──────────┬─────────────────────┘
        │         ┌──────────▼─────────────────────┐
        │         │  NPU Controller (FSM)          │
        │         └──┬──────────────┬──────────────┘
        │      ┌───▼────┐    ┌──────▼────────┐
        │      │  DMA   │    │  4×4 PE Array  │
        │      │ 3-ch   │    │  16×32bit MAC  │
        │      └───┬────┘    └──────┬─────────┘
        │          │               │
        │    AXI4 Master     Result FIFO
        │          │               │
        └──────────┼───────────────┘
                   ▼
               DRAM (SRAM/DDR)
```

---

## 2. 环境搭建

### 2.1 必需工具

| 工具 | 版本 | 用途 | 下载 |
|---|---|---|---|
| Icarus Verilog (iverilog) | ≥ 11.0 | RTL 编译与仿真 | https://bleyer.org/icarus/ |
| GTKWave | 最新 | VCD 波形查看 | https://gtkwave.sourceforge.net/ |

### 2.2 Windows 安装步骤

1. **安装 iverilog**：从官网下载 Windows 安装包，安装时勾选 "Add to PATH"
2. **安装 GTKWave**：下载安装包，同样添加到 PATH
3. **验证安装**：
   ```powershell
   iverilog -V          # 应显示版本信息
   gtkwave --version    # 应显示版本信息
   ```

### 2.3 Linux 安装步骤

```bash
# Ubuntu/Debian
sudo apt install iverilog gtkwave

# 验证
iverilog -V
gtkwave --version
```

---

## 3. 目录结构

```
NPU_prj/
├── rtl/
│   ├── pe/
│   │   ├── pe_top.v              # 32-bit PE 单元 (INT8/INT16/FP16, WS/OS, 3-stage)
│   │   └── fp16_mul.v            # IEEE 754 FP16 乘法器
│   ├── array/
│   │   └── pe_array.v            # M×N 可参数化 systolic 脉动阵列
│   ├── axi/
│   │   ├── npu_axi_lite.v        # AXI4-Lite 从机 + 14 个配置寄存器
│   │   └── npu_dma.v             # AXI4 Master DMA（Ping-Pong 版本，3 通道）
│   ├── ctrl/
│   │   └── npu_ctrl.v            # NPU 控制器 FSM（流水化：加载与计算重叠）
│   ├── buf/
│   │   └── pingpong_buf.v        # Ping-Pong 双缓冲区（DMA 与 PE 并行）
│   ├── soc/
│   │   ├── soc_top.v             # SoC 顶层集成（PicoRV32 + NPU + SRAM + DRAM）
│   │   ├── soc_mem.v             # SoC SRAM 模块
│   │   ├── dram_model.v          # 双端口 DRAM 行为模型
│   │   └── axi_lite_bridge.v     # PicoRV32 iomem → AXI4-Lite 桥接
│   ├── power/
│   │   └── npu_power.v           # 时钟门控 + DFS
│   ├── top/
│   │   └── npu_top.v             # NPU 顶层集成（含 PPBuf）
│   └── common/
│       ├── fifo.v                # 参数化同步 FIFO (sync_fifo)
│       ├── axi_monitor.v         # AXI 总线带宽监控
│       └── op_counter.v          # NPU 操作计数与性能分析
├── tb/
│   ├── tb_pe_top.v               # PE 单元测试（4 场景：INT8/FP16 × WS/OS）
│   ├── tb_npu_top.v              # NPU 系统级测试（含 AXI BFM + DRAM 模型）
│   ├── tb_soc.v                  # SoC 集成测试（CPU 固件驱动 NPU）
│   └── soc_test.S                # RISC-V 测试固件（汇编）
├── sim/
│   └── wave/                     # VCD 波形输出目录
├── doc/
│   ├── architecture.md           # 系统架构设计文档
│   ├── soc_integration_plan.md   # SoC 集成方案（含 Ping-Pong 设计）
│   └── user_manual.md            # 本文档
├── scripts/
│   ├── run_sim.ps1               # Windows PE 单元仿真脚本
│   ├── run_sim.sh                # Linux PE 单元仿真脚本
│   ├── run_full_sim.ps1          # Windows NPU 全系统仿真脚本
│   └── run_soc_sim.ps1           # Windows SoC 集成仿真脚本
├── picorv32_ref/                 # PicoRV32 参考源码
│   └── picorv32.v
└── README.md                     # 项目概述
```

---

## 4. 快速仿真

### 4.1 PE 单元仿真

测试单个 PE 的 INT8/FP16 乘加功能，包含 4 个测试场景：

```powershell
# Windows
cd NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

```bash
# Linux
cd NPU_prj
bash scripts/run_sim.sh
```

**预期输出**：
```
--- Test 1: INT8 Weight-Stationary ---
[PASS] Test 1: got=30 exp=30
--- Test 2: INT8 Output-Stationary ---
[PASS] Test 2: got=20 exp=20
--- Test 3: FP16 Weight-Stationary ---
[INFO] FP16 WS result = 0x00003C00 (expected 0x00003C00 for 1.0 * 1.0)
--- Test 4: FP16 Output-Stationary ---
[INFO] FP16 OS result = 0x00003C00
=== Summary: PASS=2  FAIL=0 ===
ALL TESTS PASSED
```

### 4.2 NPU 全系统仿真

测试完整的 NPU 数据通路（AXI-Lite → 控制器 → DMA → PE 阵列），包含带宽和操作统计：

```powershell
# Windows（仅支持 Windows 脚本）
cd NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1
```

**预期输出**：
```
========================================
  NPU Full System Simulation
========================================
[OK] Icarus Verilog version ...

[1/3] Compiling...
[OK] Compilation successful.

[2/3] Running simulation...
--- TEST 1: INT8 4x4 WS ---
    [BFM] W 0x00000010 = 0x00000004 OK
    [BFM] W 0x00000014 = 0x00000004 OK
    ...
    -> NPU DONE at cycle 77
================================================================
  NPU PERFORMANCE REPORT: INT8 4x4 WS
================================================================
  [AXI4-Lite CPU Port]
    Write Txns      :      8    Read Txns      :     21
  [AXI4 Master DMA Port]
    Read Bursts     :     42    Read Bytes      :    136
    Rd Bandwidth    :    1.72 B/cyc
  [NPU Compute]
    Total MAC Ops   :     24
    Peak Active PEs :      4 / 16
  [Performance]
    PE Utilization  :   XX.XX %
    Total Cycles    :     79
================================================================

[3/3] Results:
  VCD waveform : sim/wave/tb_npu_top.vcd (XX KB)
  Open with    : gtkwave sim/wave/tb_npu_top.vcd
```

### 4.3 查看波形

```powershell
# Windows
gtkwave sim\wave\tb_npu_top.vcd

# Linux
gtkwave sim/wave/tb_npu_top.vcd
```

**推荐关注的信号**：

| 信号路径 | 说明 |
|---|---|
| `u_npu.state` (npu_ctrl) | 控制器 FSM 状态 |
| `u_npu.pe_en` | PE 阵列使能 |
| `u_npu.status_busy / status_done` | 忙/完成标志 |
| `s_awvalid / s_awready` | AXI-Lite 写地址握手 |
| `m_arvalid / m_arready` | DMA 读请求握手 |
| `m_rvalid / m_rdata` | DMA 读数据返回 |

---

## 5. 寄存器编程指南

NPU 通过 AXI4-Lite 从机接口配置，CPU 按以下顺序操作：

### 5.1 寄存器映射

| 偏移 | 名称 | R/W | 位域 | 描述 |
|---:|---|:---:|---|---|
| 0x00 | `CTRL` | RW | `[0]` start | 写 1 启动一次推理 |
| | | | `[1]` abort | 写 1 中止当前推理 |
| | | | `[3:2]` mode | 数据模式：00=INT8, 01=INT16, 10=FP16 |
| | | | `[5:4]` stat_mode | 数据流：00=WS, 01=OS |
| 0x04 | `STATUS` | RO | `[0]` busy | NPU 忙碌中 |
| | | | `[1]` done | 计算完成 |
| 0x08 | `INT_EN` | RW | `[0]` | 中断使能 |
| 0x0C | `INT_CLR` | W | `[0]` | 中断清除（写 1 清） |
| 0x10 | `M_DIM` | RW | `[31:0]` | 矩阵 C 的行数 M |
| 0x14 | `N_DIM` | RW | `[31:0]` | 矩阵 C 的列数 N |
| 0x18 | `K_DIM` | RW | `[31:0]` | 内积维度 K |
| 0x20 | `W_ADDR` | RW | `[31:0]` | 权重在 DRAM 中的起始地址 |
| 0x24 | `A_ADDR` | RW | `[31:0]` | 激活在 DRAM 中的起始地址 |
| 0x28 | `R_ADDR` | RW | `[31:0]` | 结果在 DRAM 中的起始地址 |
| 0x30 | `ARR_CFG` | RW | `[3:0]` act_rows | 活跃行数 |
| | | | `[7:4]` act_cols | 活跃列数 |
| 0x34 | `CLK_DIV` | RW | `[2:0]` div_sel | 时钟分频：000=÷1, 001=÷2, 010=÷4, 011=÷8 |
| 0x38 | `CG_EN` | RW | `[0]` | 时钟门控使能 |

### 5.2 典型操作流程（C = A × B，INT8，Weight-Stationary）

```
步骤 1: 配置矩阵维度
  写 0x10 = M        // C 的行数
  写 0x14 = N        // C 的列数
  写 0x18 = K        // 内积维度

步骤 2: 配置 DRAM 地址
  写 0x20 = W_addr   // 权重 B 的 DRAM 地址 (N×K 个元素)
  写 0x24 = A_addr   // 激活 A 的 DRAM 地址 (M×K 个元素)
  写 0x28 = R_addr   // 结果 C 的 DRAM 地址 (M×N 个元素)

步骤 3: 启动推理
  写 0x00 = 0x01     // bit0=start, [3:2]=00=INT8, [5:4]=00=WS

步骤 4: 等待完成
  轮询读 0x04，直到 bit1 (done) = 1

步骤 5: 清除控制寄存器（必须！）
  写 0x00 = 0x00     // 清零 ctrl_reg，防止自动重启

步骤 6: 读取结果
  从 R_addr 开始读取 M×N 个 32-bit 累加结果
```

### 5.3 控制寄存器位域详解

```
CTRL (0x00) 位域布局：
  Bit  0   : START    — 写 1 触发推理（自清除）
  Bit  1   : ABORT    — 写 1 中止当前计算
  Bit  3:2 : MODE     — 00=INT8, 01=INT16, 10=FP16
  Bit  5:4 : STAT_MODE— 00=Weight-Stationary, 01=Output-Stationary
  其他位   : 保留
```

**CTRL 值示例**：

| 值 | 含义 |
|---|---|
| `0x01` | INT8, WS 启动 |
| `0x05` | INT8, OS 启动 |
| `0x11` | INT16, WS 启动 |
| `0x05` | INT16, OS 启动 |
| `0x00` | 清零（复位后必须执行） |

### 5.4 注意事项

> **重要**：计算完成后，`ctrl_reg[0]` 仍为 1。控制器进入 IDLE 后会检测到 `cfg_start` 并**自动重新启动**。必须在 `done` 后立即写 `0x00` 到 CTRL 寄存器清零。

> **DMA 数据大小计算**：
> - 权重 DMA 传输：`N × K × (DATA_W/8)` 字节
> - 激活 DMA 传输：`M × K × (DATA_W/8)` 字节
> - 结果 DMA 传输：`M × N × (ACC_W/8)` 字节（ACC_W=32）
> - INT8 模式下 DATA_W=16，每个元素占 2 字节
> - 结果固定为 32-bit 累加值

---

## 6. SoC 集成指南

本节描述 PicoRV32 CPU 与 NPU 的 SoC 集成方案，包括地址映射设计原理、各设备地址为什么这样分配，以及如何在固件中通过 CPU 配置 NPU 参数寄存器。

### 6.1 系统架构总览

```
                          ┌──────────────────────────────────────────┐
                          │              SoC (soc_top.v)              │
                          │                                            │
                          │  PicoRV32 CPU                              │
                          │  mem_valid / mem_addr / mem_wdata          │
                          │  mem_ready / mem_rdata                     │
                          │       │                                    │
                          │       ▼  (地址译码)                        │
                          │  ┌─────────────────────────────────────┐  │
                          │  │     addr[31:24] 决定目标设备         │  │
                          │  └──┬──────────┬───────────┬───────────┘  │
                          │     │          │           │              │
                          │     ▼          ▼           ▼              │
                          │  ┌──────┐  ┌───────┐  ┌──────────┐      │
                          │  │ SRAM │  │ DRAM  │  │ NPU Reg  │      │
                          │  │4KB   │  │ ~60KB │  │(AXI-Lite │      │
                          │  │      │  │       │  │ bridge)  │      │
                          │  └──────┘  └───┬───┘  └────┬─────┘      │
                          │                │           │             │
                          │                │    ┌──────┴──────┐      │
                          │                │    │   npu_top   │      │
                          │                │    │             │      │
                          │                │    │  DMA (AXI4  │      │
                          │                └────┤  Master)    │      │
                          │                     │     │       │      │
                          │                     │  PPBuf_W   │      │
                          │                     │  PPBuf_A   │      │
                          │                     │     │       │      │
                          │                     │  PE Array  │      │
                          │                     │     │       │      │
                          │                     │  Result    │      │
                          │                     └─────┼───────┘      │
                          │                           │              │
                          │                     ──────►DRAM           │
                          └──────────────────────────────────────────┘
```

### 6.2 地址映射表

| 地址范围 | 大小 | 设备 | 访问者 | 用途 |
|---|---|---|---|---|
| `0x0000_0000 - 0x0000_0FFF` | 4KB | **SRAM** | 仅 CPU | CPU 指令存储 + 数据 |
| `0x0000_1000 - 0x0000_FFFF` | ~60KB | **DRAM** | CPU + NPU DMA | NPU 输入（权重/激活）+ 输出结果 |
| `0x0200_0000 - 0x0200_003F` | 64B | **NPU 寄存器** | 仅 CPU | NPU 配置参数 |

### 6.3 为什么地址是这样分配的

#### SRAM：`0x0000_0000` 起始

- **PicoRV32 硬件决定**：PicoRV32 的 `PROGADDR_RESET` 参数决定 CPU 复位后从哪个地址取第一条指令。我们设置 `PROGADDR_RESET = 0x0000_0000`，所以 SRAM 必须从 0 开始
- **参考 picosoc**：picosoc 的 SRAM 也是从 `0x0000_0000` 开始，大小由 `MEM_WORDS` 参数控制（默认 256 words = 1KB，我们扩展到 1024 words = 4KB）
- **CPU 取指**：CPU 的所有指令都从 SRAM 取，所以固件（RISC-V 机器码）必须预先加载到 SRAM 的 0 地址
- **CPU 栈**：栈指针初始化在 SRAM 末尾 `STACKADDR = 4 * MEM_WORDS = 0x0000_1000`，栈从高地址向低地址增长

#### DRAM：`0x0000_1000` 起始（紧接 SRAM 之后）

- **简单连续地址空间**：DRAM 紧接在 SRAM 之后，地址译码只需一个比较器 `addr >= 4*MEM_WORDS`
- **不需要对齐到 2 的幂边界**：DRAM 在仿真中是行为模型，没有真实 DRAM 的行/bank 对齐要求。所以从 `0x1000`（紧接 SRAM）开始最省空间
- **CPU 和 NPU DMA 共享访问**：这是关键设计——CPU 先把权重和激活矩阵写入 DRAM，然后 NPU DMA 从 DRAM 读取数据送入 PE 阵列。计算完成后 NPU DMA 再把结果写回 DRAM，CPU 最后从 DRAM 读取结果验证
- **为什么不用单独的 AXI 交叉开关**：DRAM 模型在仿真中有两个端口（CPU 端口 + NPU DMA 端口），不需要总线仲裁。CPU 在 NPU 启动前完成 DRAM 写入，NPU 运行时 CPU 只轮询寄存器，两者时序上错开

#### NPU 寄存器：`0x0200_0000` 起始

- **参考 picosoc 的 iomem 空间**：PicoRV32 没有独立的 IO 地址空间，而是通过地址高位来区分内存映射 IO。picosoc 中 `mem_addr[31:24] > 8'h01` 时激活 `iomem_valid`，即地址 `>= 0x0200_0000` 的区域被视为 IO 设备
- **为什么是 0x0200_0000 而不是更小**：这个地址是 picosoc 的设计惯例。picosoc 的 SPI Flash 配置寄存器在 `0x0200_0000`，UART 寄存器在 `0x0200_0004/0x0200_0008`。我们继承了同样的 IO 地址空间，把 NPU 寄存器也放在 `0x0200_0000` 开始的区域
- **地址译码**：`addr >= 0x0200_0000` → `iomem_valid=1` → 通过 AXI-Lite 桥接访问 NPU 寄存器
- **CPU 独占访问**：只有 CPU 需要读写 NPU 寄存器（配置 PE 规模、DMA 地址、矩阵维度等），NPU DMA 从不访问这个区域

#### 地址译码逻辑（在 soc_top.v 中）

```verilog
wire addr_is_ram  = mem_valid && (mem_addr < 4 * MEM_WORDS);        // < 0x1000
wire addr_is_dram = mem_valid && (mem_addr >= 4*MEM_WORDS) &&       // 0x1000 ~ 0x01FFFFFF
                    (mem_addr < 32'h0200_0000);
wire addr_is_npu  = mem_valid && (mem_addr >= 32'h0200_0000);       // >= 0x02000000
```

### 6.4 CPU 固件配置 NPU 的完整步骤

CPU 通过 store 指令（`sw`）向 NPU 寄存器地址写入配置值。NPU 寄存器的访问路径是：

```
CPU sw 指令 → PicoRV32 mem_if → 地址译码 → AXI-Lite 桥接 → NPU AXI-Lite 从机 → 寄存器文件
```

**完整操作流程**（C = A × B，INT8，Weight-Stationary，4×4 矩阵）：

#### 第一步：准备 DRAM 数据

在启动 NPU 之前，CPU 必须先把权重矩阵和激活矩阵写入 DRAM。

```
DRAM 布局：

  0x0000_1000  ┌──────────────────┐
               │  权重矩阵 W      │  N×K 个 INT8 元素
               │  (4×4 = 32 bytes)│
  0x0000_1020  ├──────────────────┤
               │  (padding)       │
               │                  │
  0x0000_1200  ├──────────────────┤
               │  激活矩阵 A      │  M×K 个 INT8 元素
               │  (4×4 = 32 bytes)│
  0x0000_1220  ├──────────────────┤
               │  (padding)       │
               │                  │
  0x0000_1400  ├──────────────────┤
               │  结果矩阵 C      │  M×N 个 INT32 累加值
               │  (4×4 = 64 bytes)│
  0x0000_1440  └──────────────────┘
```

CPU 用普通 `sw` 指令写入 DRAM 地址即可：

```c
// C 伪代码：写权重矩阵到 DRAM[0x1000]
volatile uint32_t *w_ptr = (uint32_t *)0x00001000;
w_ptr[0] = 0x00010000;  // W[0][0]=0, W[0][1]=1  (identity matrix)
w_ptr[1] = 0x00010000;  // W[0][2]=0, W[0][3]=1
// ...

// 写激活矩阵到 DRAM[0x1200]
volatile uint32_t *a_ptr = (uint32_t *)0x00001200;
a_ptr[0] = 0x04030201;  // A[0] = {1, 2, 3, 4}
a_ptr[1] = 0x08070605;  // A[1] = {5, 6, 7, 8}
// ...
```

#### 第二步：配置 NPU 寄存器

NPU 寄存器基址为 `0x0200_0000`，CPU 通过 `sw` 指令写入：

```c
#define NPU_BASE  0x02000000
#define NPU_REG(offset) (*((volatile uint32_t *)(NPU_BASE + (offset))))

// 矩阵维度
NPU_REG(0x10) = 4;        // M_DIM = 4 (C 矩阵行数)
NPU_REG(0x14) = 4;        // N_DIM = 4 (C 矩阵列数)
NPU_REG(0x18) = 4;        // K_DIM = 4 (内积维度)

// DMA 源地址和目标地址
NPU_REG(0x20) = 0x00001000;  // W_ADDR = DRAM 中权重的起始地址
NPU_REG(0x24) = 0x00001200;  // A_ADDR = DRAM 中激活的起始地址
NPU_REG(0x28) = 0x00001400;  // R_ADDR = DRAM 中结果的目标地址

// PE 阵列配置
NPU_REG(0x30) = 0x44;     // ARR_CFG = 4 行 × 4 列

// 中断使能（可选）
NPU_REG(0x08) = 0x01;     // INT_EN = 1
```

**DMA 地址说明**：
- `W_ADDR` 告诉 NPU DMA：从 DRAM 的哪个地址开始读取权重数据
- `A_ADDR` 告诉 NPU DMA：从 DRAM 的哪个地址开始读取激活数据
- `R_ADDR` 告诉 NPU DMA：把计算结果写到 DRAM 的哪个地址
- DMA 自动计算传输长度：`权重字节数 = N × K × sizeof(element)`
- 这些地址必须指向 DRAM 区域（`≥ 0x0000_1000`），不能指向 SRAM 或 NPU 寄存器区域

#### 第三步：启动 NPU

```c
// CTRL = 0x01: start=1, mode=INT8(00), stat_mode=WS(00)
NPU_REG(0x00) = 0x01;
```

此时 NPU 控制器开始执行：
1. DMA 同时从 DRAM[W_ADDR] 读权重、从 DRAM[A_ADDR] 读激活
2. 数据通过 Ping-Pong Buffer 送入 PE 阵列
3. PE 阵列计算完毕后，DMA 把结果写到 DRAM[R_ADDR]

#### 第四步：等待完成

**方式一：轮询**
```c
while ((NPU_REG(0x04) & 0x02) == 0) {
    // STATUS bit1 (done) = 0，继续等待
}
// done = 1，计算完成
```

**方式二：中断**（需启用 INT_EN）
```c
// CPU 收到 irq[7] 中断
// 在中断处理函数中：
void npu_irq_handler(void) {
    uint32_t status = NPU_REG(0x04);
    if (status & 0x02) {
        // 计算完成
        NPU_REG(0x0C) = 0x01;  // 清除中断标志
    }
}
```

#### 第五步：读取结果

```c
volatile uint32_t *r_ptr = (uint32_t *)0x00001400;
int32_t c[4][4];
for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
        c[i][j] = r_ptr[i * 4 + j];
```

#### 第六步：清理

```c
NPU_REG(0x00) = 0x00;     // 清零 CTRL，防止自动重启
```

### 6.5 DMA 地址配置详解

#### DMA 传输方向

```
                    ┌──────────────────────────────────────┐
  DRAM[W_ADDR] ──► │  DMA Read (Weight)   │
                    │         │           │
                    │         ▼           │
                    │  Ping-Pong Buffer   │──► PE Array ──► Result
                    │         │           │      │
                    │         ▼           │      ▼
                    │  DMA Read (Activ)   │  Result FIFO
                    │         │           │      │
                    │         ▼           │      ▼
  DRAM[R_ADDR] ◄── │  DMA Write (Result) │
                    └──────────────────────────────────────┘
```

| 寄存器 | 方向 | 说明 |
|---|---|---|
| `W_ADDR` (0x20) | DMA **从 DRAM 读** | 权重矩阵在 DRAM 中的起始地址 |
| `A_ADDR` (0x24) | DMA **从 DRAM 读** | 激活矩阵在 DRAM 中的起始地址 |
| `R_ADDR` (0x28) | DMA **向 DRAM 写** | 计算结果在 DRAM 中的目标地址 |

#### DMA 自动计算传输长度

DMA 传输的**总字节数**由矩阵维度寄存器自动推算：

```
权重传输字节数 = N_DIM × K_DIM × bytes_per_element
激活传输字节数 = M_DIM × K_DIM × bytes_per_element
结果传输字节数 = M_DIM × N_DIM × 4  (结果固定 32-bit)
```

其中 `bytes_per_element` 由 CTRL 寄存器的 `mode` 位决定：
- INT8 (mode=00)：每个元素 1 字节，但 DMA 按 32-bit 字传输（一个字含 4 个 INT8）
- INT16/FP16 (mode=01/10)：每个元素 2 字节，一个字含 2 个元素

> **注意**：`W_ADDR`、`A_ADDR`、`R_ADDR` 指向的是 DRAM 的字节地址，DMA 内部按 32-bit 字对齐访问。

### 6.6 AXI-Lite 桥接

PicoRV32 使用简单的 `mem_valid/mem_ready` 握手协议，NPU 使用标准的 AXI4-Lite 协议。`axi_lite_bridge.v` 负责协议转换：

```
PicoRV32                           NPU
mem_valid ──► ┌──────────────┐ ──► s_axi_awvalid / s_axi_arvalid
mem_ready ◄── │ axi_lite     │ ◄── s_axi_awready / s_axi_arready
mem_wdata  ──► │   _bridge   │ ──► s_axi_wdata
mem_wstrb  ──► │              │ ──► s_axi_wstrb
mem_rdata  ◄── │              │ ◄── s_axi_rdata
             └──────────────┘
```

- CPU 写操作（`wstrb != 0`）：bridge 自动生成 AW+W 通道事务
- CPU 读操作（`wstrb == 0`）：bridge 自动生成 AR 通道事务，等待 R 通道返回数据
- 地址偏移处理：bridge 自动减去 `NPU_BASE_ADDR`，NPU 看到的是从 0 开始的寄存器偏移

---

## 7. Ping-Pong Buffer 使用

### 7.1 为什么需要 Ping-Pong Buffer

在原始设计中，DMA 从 DRAM 读数据后写入单个 FIFO，PE 从 FIFO 消费数据。这是一个**串行**过程：

```
无 Ping-Pong：
|  DMA 读权重  |  DMA 读激活  |  PE 计算  |  DMA 写回  |
               串行，PE 经常等数据（FIFO 空闲）
```

Ping-Pong Buffer 让 DMA 和 PE **并行**工作：

```
有 Ping-Pong：
| DMA 填充 BufA    | DMA 填充 BufB    | DMA 填充 BufA    |
|                   | PE 消费 BufA     | PE 消费 BufB     |
                    ↑ 重叠！           ↑ 重叠！
```

### 7.2 工作原理

`pingpong_buf.v` 内部有两组独立的 SRAM（BufA 和 BufB）：

- **写端（DMA 侧）**：向当前写缓冲区写入数据
- **读端（PE 侧）**：从当前读缓冲区读出数据
- **swap**：当写缓冲区填满时，DMA 和 PE 自动切换到另一组缓冲区

```
  DMA 写入 ──► BufA ──► PE 消费
              BufB     (PE 等待)
  
  [swap 触发]（BufA 填满）
  
  DMA 写入 ──► BufB ──► PE 消费
              BufA     (PE 消费 BufA 的剩余数据)
```

### 7.3 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DEPTH` | 32 | 每个缓冲区的深度（32 个数据字） |
| `THRESHOLD` | 16 | DMA 填充多少个数据后 PE 可以开始消费 |
| `DATA_W` | 32 | DMA 侧数据宽度（32-bit AXI 总线） |
| `OUT_WIDTH` | 16 | PE 侧数据宽度（16-bit） |

**THRESHOLD = 16 的含义**：DMA 不需要等整个 BufA（32 个字）全部填满才让 PE 开始。只要填满 16 个，PE 就可以开始消费。这进一步减少了 PE 的等待时间。

### 7.4 状态信号

| 信号 | 方向 | 说明 |
|---|---|---|
| `buf_ready` | output | 写缓冲区已达到 THRESHOLD，PE 可以开始 |
| `buf_empty` | output | 读缓冲区已空，PE 无数据可读 |
| `buf_full` | output | 写缓冲区已满，DMA 必须暂停 |
| `swap` | input | 外部触发缓冲区切换（DMA 填满时自动触发） |

### 7.5 在数据通路中的位置

```
DRAM ──DMA Read──► pingpong_buf (Weight) ──► PE Array
DRAM ──DMA Read──► pingpong_buf (Activ)  ──► PE Array
PE Array ──► sync_fifo (Result) ──DMA Write──► DRAM
```

权重和激活各有独立的 Ping-Pong Buffer，允许 W 和 A 同时并行加载。

---

## 8. 仿真测试详解

### 8.1 PE 单元测试 (`tb/tb_pe_top.v`)

测试单个 PE 的基本功能，验证 3 级流水线的正确性。

| 测试 | 模式 | 场景 | 验证方法 |
|---|---|---|---|
| Test 1 | INT8, WS | weight=3, activations=1,2,3,4 | 期望 acc_out = 30 |
| Test 2 | INT8, OS | weights=1,2,3,4, activation=2 | 期望 acc_out = 20 |
| Test 3 | FP16, WS | w=1.0, a=1.0 | 期望结果 0x00003C00 |
| Test 4 | FP16, OS | w=1.0, a=2.0 | 烟雾测试 |

**运行命令**：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

**波形输出**：`sim/wave/tb_pe_top.vcd`

### 8.2 NPU 系统级测试 (`tb/tb_npu_top.v`)

完整的系统级验证，包含：

- **AXI4-Lite BFM**：`axi_write` / `axi_read` task 模拟 CPU 读写
- **DRAM 模型**：16K 深度 × 32-bit 行为级存储器
- **AXI Monitor**：实时跟踪总线带宽和事务统计
- **Op Counter**：跟踪 MAC 操作数、PE 利用率、FSM 耗时

| 测试 | 模式 | 矩阵规模 | DRAM 布局 |
|---|---|---|---|
| Test 1 | INT8, WS | 4×4 × 4 | W@0x0000, A@0x0100, R@0x0200 |
| Test 2 | INT8, OS | 4×4 × 4 | W@0x1000, A@0x1100, R@0x1200 |
| Test 3 | INT8, WS | 8×8 × 8 (tiled) | W@0x2000, A@0x3000, R@0x4000 |

> **注意**：当前 main sequence 仅运行 Test 1。Test 2/3 已定义但未启用，可取消注释添加到 main sequence。

**运行命令**：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1
```

**波形输出**：`sim/wave/tb_npu_top.vcd`

### 8.3 Testbench 关键 Task

| Task | 功能 | 参数 |
|---|---|---|
| `axi_write(addr, data)` | 写 AXI-Lite 寄存器 | 32-bit 地址, 32-bit 数据 |
| `axi_read(addr, data)` | 读 AXI-Lite 寄存器 | 地址输入, 数据输出 |
| `wait_done(timeout)` | 等待 NPU 完成 | 超时周期数 |
| `init_dram_seq(base, count)` | 初始化 DRAM 数据 | 起始地址, 元素个数 |
| `print_report(label)` | 打印性能报告 | 测试名称字符串 |

---

## 9. 性能分析与监控

### 9.1 AXI 总线监控 (`rtl/common/axi_monitor.v`)

实时监控 AXI4-Lite（CPU 配置端口）和 AXI4 Master（DMA 端口）的带宽和延迟。

**监控指标**：

| 指标 | 说明 |
|---|---|
| `s_axi_wr_cnt` / `s_axi_rd_cnt` | CPU 端口写/读事务数 |
| `s_axi_wr_lat` / `s_axi_rd_lat` | CPU 端口写/读延迟总和（周期） |
| `m_axi_rd_cnt` / `m_axi_wr_cnt` | DMA 端口读/写突发数 |
| `m_axi_rd_bytes` / `m_axi_wr_bytes` | DMA 传输总字节数 |
| `m_axi_rd_bw` / `m_axi_wr_bw` | 平均带宽（bytes/cycle × 1000） |
| `total_cycles` | 监控总周期数 |

**带宽计算**：
```
实际带宽 (B/cycle) = m_axi_rd_bw / 1000
例如：m_axi_rd_bw = 1720 → 1.72 B/cycle
```

### 9.2 NPU 操作计数器 (`rtl/common/op_counter.v`)

跟踪 NPU 计算性能和效率。

**性能指标**：

| 指标 | 说明 |
|---|---|
| `total_mac_ops` | 总 MAC 操作数（64-bit） |
| `total_pe_cycles` | PE 使能的总周期数 |
| `total_busy_cycles` | NPU 忙碌总周期数 |
| `total_compute_cycles` | 纯计算周期（pe_en && !pe_flush） |
| `total_dma_cycles` | DMA 传输周期（busy && !pe_en） |
| `peak_active_pe` | 峰值活跃 PE 数 |
| `utilization_pct` | PE 利用率百分比（×100） |
| `mac_per_cycle` | 每 cycle 平均 MAC 数（×100） |
| `efficiency_pct` | 计算效率 = compute_cycles / busy_cycles（×100） |

**指标解读**：
- `utilization_pct`：PE 利用率。100% 意味着所有 16 个 PE 在每个使能周期都在产出有效结果
- `mac_per_cycle`：吞吐率。理论上限为 ROWS×COLS = 16（4×4 阵列）
- `efficiency_pct`：整体效率。DMA 开销越小，效率越高

### 9.3 性能报告示例

仿真结束时自动输出 `NPU PERFORMANCE REPORT`：

```
================================================================
  NPU PERFORMANCE REPORT: INT8 4x4 WS
================================================================
  [AXI4-Lite CPU Port]
    Write Txns      :      8    Read Txns      :     21
    Write Beats     :      8    Read Beats     :     21
    Avg Wr Latency  :      3 cyc  Avg Rd Latency  :      2 cyc
  [AXI4 Master DMA Port]
    Write Bursts    :      1    Read Bursts     :     42
    Write Beats     :      4    Read Beats      :    136
    Write Bytes     :     16    Read Bytes      :    136
    Rd Bandwidth    :    1.72 B/cyc
    Wr Bandwidth    :    0.20 B/cyc
  [NPU Compute]
    Total MAC Ops   :     24
    PE Active Cycles:      6    Busy Cycles    :     72
    Compute Cycles  :      4    DMA Cycles     :     66
    Peak Active PEs :      4 / 16
  [Performance]
    PE Utilization  :    6.25 %
    MACs/Cycle      :    4.00
    Efficiency      :    5.56 %
    Total Cycles    :     79
================================================================
```

---

## 10. 模块参考

### 10.1 `npu_top` — 顶层集成

| 参数 | 默认值 | 说明 |
|---|---|---|
| `ROWS` | 4 | PE 阵列行数 |
| `COLS` | 4 | PE 阵列列数 |
| `DATA_W` | 16 | 数据位宽（FP16/INT8 均为 16-bit） |
| `ACC_W` | 32 | 累加器位宽 |

**端口**：

| 端口 | 方向 | 说明 |
|---|---|---|
| `sys_clk` / `sys_rst_n` | input | 系统时钟 / 复位 |
| `s_axi_*` | — | AXI4-Lite 从机（CPU 配置端口） |
| `m_axi_*` | — | AXI4 Master（DMA 端口，数据宽度 = ACC_W） |
| `npu_irq` | output | 中断输出 |

### 10.2 `npu_ctrl` — 控制器 FSM（Ping-Pong 流水化）

**状态机**（Ping-Pong 流水化版本）：

```
S_IDLE → S_LOAD → S_COMPUTE → S_DRAIN → S_WRITE_BACK → S_DONE → S_IDLE
```

| 状态 | 说明 |
|---|---|
| `S_IDLE` | 等待 `ctrl_reg[0]=1` 启动信号 |
| `S_LOAD` | DMA 同时加载 W 和 A 到 Ping-Pong Buffer；等待 PPBuf 达到 THRESHOLD 后 PE 开始消费 |
| `S_COMPUTE` | PE 消费 PPBuf 数据进行计算；DMA 可能仍在填充 PPBuf 另一侧（**并行重叠**） |
| `S_DRAIN` | DMA 完成，PE 消费 PPBuf 剩余数据，刷出累加器 |
| `S_WRITE_BACK` | DMA 将 PE 结果写回 DRAM (R_WRITE) |
| `S_DONE` | 置位 done/irq，返回 IDLE |

> **与旧版区别**：旧版是严格串行的 `LOAD_W → LOAD_A → COMPUTE`。新版 DMA 加载和 PE 计算通过 Ping-Pong Buffer 重叠执行。

### 10.3 `npu_dma` — DMA 控制器（Ping-Pong 版本）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATA_W` | 32 | AXI 数据位宽（= ACC_W） |
| `PE_DATA_W` | 16 | PE 侧数据位宽 |
| `BURST_MAX` | 16 | 最大突发长度 |
| `PPB_DEPTH` | 32 | Ping-Pong 缓冲区深度 |
| `PPB_THRESH` | 16 | Ping-Pong 早期启动阈值 |
| `R_FIFO_DEPTH` | 64 | 结果 FIFO 深度 |

**通道**：
- **通道 0 (Weight)**：从 DRAM 读取权重 → 写入 Weight Ping-Pong Buffer
- **通道 1 (Activation)**：从 DRAM 读取激活 → 写入 Activation Ping-Pong Buffer
- **通道 2 (Result)**：从 Result FIFO 读取 PE 结果 → 写回 DRAM

**DMA 状态机**：`IDLE → W_READ / A_READ / WA_READ / R_WRITE → IDLE`
- `WA_READ`：Weight 和 Activation 交替突发读取，最大化总线利用率

### 10.4 `pe_top` — 处理单元

**3 级流水线**：

```
Stage 0: Input Register（锁存 weight + activation）
Stage 1: Multiplier（INT8 有符号乘 / FP16 乘）
Stage 2: Accumulator（累加或 OS 内部累加）
```

| 端口 | 说明 |
|---|---|
| `mode` (1-bit) | 0=INT8, 1=FP16 |
| `stat_mode` (1-bit) | 0=Weight-Stationary, 1=Output-Stationary |
| `en` | 流水线使能 |
| `flush` | 刷出累加器 / 加载新权重 |
| `w_in[15:0]` | 权重输入 |
| `a_in[15:0]` | 激活输入 |
| `acc_in[31:0]` | 部分和输入（WS 模式传递用） |
| `acc_out[31:0]` | 累加结果输出 |
| `valid_out` | 输出有效标志 |

### 10.5 `pe_array` — 脉动阵列

```
      act_in[0] ──► PE[0][0] ──► PE[0][1] ──► … ──► PE[0][N-1]
                       │                               │
      act_in[1] ──► PE[1][0] ──► PE[1][1] ──► … ──► PE[1][N-1]
                       │                               │
             …         …                               …
      act_in[M-1]► PE[M-1][0]─► PE[M-1][1]─► … ──► PE[M-1][N-1]
```

- 激活水平脉动（行方向）
- 部分和垂直传递（列方向）
- 权重按列广播

### 10.6 `fp16_mul` — FP16 乘法器

IEEE 754 半精度乘法器：
- 格式：1 sign + 5 exponent + 10 mantissa（bias=15）
- 支持特殊值：NaN, Infinity, Zero
- 输出符号扩展到 ACC_W 位

### 10.7 `sync_fifo` — 同步 FIFO

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATA_W` | 16 | 数据位宽 |
| `DEPTH` | 16 | 深度（2 的幂） |
| `ALMOST_FULL` | 2 | almost_full 阈值 |
| `ALMOST_EMPTY` | 2 | almost_empty 阈值 |

### 10.8 `npu_power` — 电源管理

| 功能 | 说明 |
|---|---|
| DFS (动态频率调整) | div_sel: 000=÷1, 001=÷2, 010=÷4, 011=÷8 |
| 行时钟门控 | 每行独立门控，1=关闭时钟 |
| 列时钟门控 | 每列独立门控，1=关闭时钟 |

> **注意**：当前为行为模型。FPGA 实现需替换为 BUFGCE 原语，ASIC 实现需使用 ICG 单元。

---

## 11. 扩展与定制

### 11.1 扩展阵列规模

修改 `npu_top` 的参数即可扩展阵列：

```verilog
npu_top #(
    .ROWS(8),      // 8 行
    .COLS(8),      // 8 列
    .DATA_W(16),
    .ACC_W(32)
) u_npu ( ... );
```

### 11.2 添加新的测试场景

在 `tb/tb_npu_top.v` 的 main sequence 中添加：

```verilog
initial begin
    ...
    test1;
    test2;           // 添加 test2
    #(CLK_T*100);
    test3;           // 添加 test3
    #(CLK_T*100);
    ...
end
```

### 11.3 自定义 DRAM 数据

替换 `init_dram_seq` 为实际数据加载：

```verilog
task load_matrix;
    input [31:0] base_addr;
    // ... 逐元素写入 dram 数组
endtask
```

---

## 12. 常见问题

### Q1: 仿真卡在 `wait_done` 不返回

**原因**：NPU 未发出 done 信号。常见原因：
- 未正确配置所有寄存器（M_DIM/N_DIM/K_DIM/W_ADDR/A_ADDR/R_ADDR）
- DRAM 数据区域重叠
- 控制器 FSM 死锁

**排查**：查看波形中 `u_npu.state` 是否正常流转。

### Q2: NPU 无限重启

**原因**：`ctrl_reg[0]` 未清零。控制器完成 S_DONE→S_IDLE 后检测到 `cfg_start` 仍为 1，自动重新启动。

**解决**：确保 `wait_done` 中读取到 done 后立即 `axi_write(32'h00, 32'h00)` 清零。

### Q3: 编译报错 `$clog2` 未定义

**原因**：iverilog 版本过旧。

**解决**：升级到 iverilog ≥ 11.0，或在编译时加 `-g2012` 参数。

### Q4: iverilog 不支持 SystemVerilog 语法

本项目所有 RTL 均使用 Verilog-2001 兼容语法。已知的 iverilog 限制：
- 不支持 task 内 `return` 语句（用 `cnt = timeout_val` 替代 break）
- 不支持 `8'(i)` 类型转换（用位拼接替代）
- 不支持 task 内 `reg` 声明（在 module 级声明）

### Q5: 如何修改仿真超时时间

在 `tb_npu_top.v` 中修改：
- `wait_done(20000)` — 单次推理超时周期数
- `#(CLK_T * 200000)` — 全局超时

### Q6: FP16 结果如何解读

`acc_out` 中低 16 位为 FP16 格式，高 16 位为符号扩展。解读方法：
- 取 `acc_out[15:0]` 得到 FP16 值
- Bit 15 = 符号, Bit 14:10 = 指数 (bias=15), Bit 9:0 = 尾数
- 例如 `0x3C00` = +1.0, `0x4000` = +2.0, `0xC000` = −2.0
