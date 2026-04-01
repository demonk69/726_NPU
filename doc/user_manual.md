# NPU_prj 用户手册

> 嵌入式低功耗 NPU 加速器 IP — 使用与仿真指南
>
> 最后更新：2026-04-01

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境搭建](#2-环境搭建)
3. [目录结构](#3-目录结构)
4. [快速仿真](#4-快速仿真)
5. [寄存器编程指南](#5-寄存器编程指南)
6. [仿真测试详解](#6-仿真测试详解)
7. [性能分析与监控](#7-性能分析与监控)
8. [模块参考](#8-模块参考)
9. [扩展与定制](#9-扩展与定制)
10. [常见问题](#10-常见问题)

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
│   │   ├── pe_array.v            # M×N 可参数化 systolic 脉动阵列
│   │   └── array_ctrl.v          # [已废弃] 由 npu_ctrl 替代
│   ├── axi/
│   │   ├── npu_axi_lite.v        # AXI4-Lite 从机 + 14 个配置寄存器
│   │   └── npu_dma.v             # AXI4 Master DMA（权重/激活/结果 3 通道）
│   ├── ctrl/
│   │   └── npu_ctrl.v            # NPU 控制器 FSM
│   ├── power/
│   │   └── npu_power.v           # 时钟门控 + DFS
│   ├── top/
│   │   └── npu_top.v             # NPU 顶层集成
│   └── common/
│       ├── fifo.v                # 参数化同步 FIFO (sync_fifo)
│       ├── axi_monitor.v         # AXI 总线带宽监控
│       └── op_counter.v          # NPU 操作计数与性能分析
├── tb/
│   ├── tb_pe_top.v               # PE 单元测试（4 场景：INT8/FP16 × WS/OS）
│   └── tb_npu_top.v              # NPU 系统级测试bench（含 AXI BFM + DRAM 模型）
├── sim/
│   └── wave/                     # VCD 波形输出目录
├── doc/
│   ├── architecture.md           # 系统架构设计文档
│   └── user_manual.md            # 本文档
├── scripts/
│   ├── run_sim.ps1               # Windows PE 单元仿真脚本
│   ├── run_sim.sh                # Linux PE 单元仿真脚本
│   └── run_full_sim.ps1          # Windows NPU 全系统仿真脚本
├── constraints/
│   └── npu_fpga.xdc              # [TODO] FPGA 综合约束
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

## 6. 仿真测试详解

### 6.1 PE 单元测试 (`tb/tb_pe_top.v`)

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

### 6.2 NPU 系统级测试 (`tb/tb_npu_top.v`)

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

### 6.3 Testbench 关键 Task

| Task | 功能 | 参数 |
|---|---|---|
| `axi_write(addr, data)` | 写 AXI-Lite 寄存器 | 32-bit 地址, 32-bit 数据 |
| `axi_read(addr, data)` | 读 AXI-Lite 寄存器 | 地址输入, 数据输出 |
| `wait_done(timeout)` | 等待 NPU 完成 | 超时周期数 |
| `init_dram_seq(base, count)` | 初始化 DRAM 数据 | 起始地址, 元素个数 |
| `print_report(label)` | 打印性能报告 | 测试名称字符串 |

---

## 7. 性能分析与监控

### 7.1 AXI 总线监控 (`rtl/common/axi_monitor.v`)

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

### 7.2 NPU 操作计数器 (`rtl/common/op_counter.v`)

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

### 7.3 性能报告示例

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

## 8. 模块参考

### 8.1 `npu_top` — 顶层集成

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

### 8.2 `npu_ctrl` — 控制器 FSM

**状态机**：

```
S_IDLE → S_LOAD_W → S_LOAD_A → S_COMPUTE → S_DRAIN → S_WRITE_BACK → S_DONE → S_IDLE
```

| 状态 | 说明 |
|---|---|
| `S_IDLE` | 等待 `ctrl_reg[0]=1` 启动信号 |
| `S_LOAD_W` | DMA 搬运权重 (W_READ) |
| `S_LOAD_A` | DMA 搬运激活 (A_READ) |
| `S_COMPUTE` | PE 阵列计算（1 周期简化模型） |
| `S_DRAIN` | 刷出累加器（pe_flush=1） |
| `S_WRITE_BACK` | DMA 写回结果 (R_WRITE) |
| `S_DONE` | 置位 done/irq，返回 IDLE |

### 8.3 `npu_dma` — DMA 控制器

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATA_W` | 32 | 数据位宽（= ACC_W） |
| `BURST_MAX` | 16 | 最大突发长度 |
| `FIFO_DEPTH` | 64 | 内部 FIFO 深度 |

**通道**：
- **通道 0 (Weight)**：从 DRAM 读取权重到 W FIFO
- **通道 1 (Activation)**：从 DRAM 读取激活到 A FIFO
- **通道 2 (Result)**：从 R FIFO 写回结果到 DRAM

**DMA 状态机**：`IDLE → W_READ / A_READ / R_WRITE → IDLE`（时分复用 AXI 总线）

### 8.4 `pe_top` — 处理单元

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

### 8.5 `pe_array` — 脉动阵列

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

### 8.6 `fp16_mul` — FP16 乘法器

IEEE 754 半精度乘法器：
- 格式：1 sign + 5 exponent + 10 mantissa（bias=15）
- 支持特殊值：NaN, Infinity, Zero
- 输出符号扩展到 ACC_W 位

### 8.7 `sync_fifo` — 同步 FIFO

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATA_W` | 16 | 数据位宽 |
| `DEPTH` | 16 | 深度（2 的幂） |
| `ALMOST_FULL` | 2 | almost_full 阈值 |
| `ALMOST_EMPTY` | 2 | almost_empty 阈值 |

### 8.8 `npu_power` — 电源管理

| 功能 | 说明 |
|---|---|
| DFS (动态频率调整) | div_sel: 000=÷1, 001=÷2, 010=÷4, 011=÷8 |
| 行时钟门控 | 每行独立门控，1=关闭时钟 |
| 列时钟门控 | 每列独立门控，1=关闭时钟 |

> **注意**：当前为行为模型。FPGA 实现需替换为 BUFGCE 原语，ASIC 实现需使用 ICG 单元。

---

## 9. 扩展与定制

### 9.1 扩展阵列规模

修改 `npu_top` 的参数即可扩展阵列：

```verilog
npu_top #(
    .ROWS(8),      // 8 行
    .COLS(8),      // 8 列
    .DATA_W(16),
    .ACC_W(32)
) u_npu ( ... );
```

### 9.2 添加新的测试场景

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

### 9.3 自定义 DRAM 数据

替换 `init_dram_seq` 为实际数据加载：

```verilog
task load_matrix;
    input [31:0] base_addr;
    // ... 逐元素写入 dram 数组
endtask
```

---

## 10. 常见问题

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
