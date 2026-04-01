# NPU_prj — 嵌入式 NPU 加速器系统架构

## 1. 项目概述

面向边缘计算的 **低功耗 NPU 加速器 IP**，通过 AXI 总线挂载于 SoC（Cortex-M0 / RISC-V）之下，
提供 INT8 / INT16 / FP16 混合精度矩阵乘加加速能力。

### 核心指标

| 指标 | 目标值 |
|---|---|
| 算力 (INT8) | ≥ 0.5 TOPS (基础), 目标 1 TOPS |
| 精度 | INT8 / INT16 / FP16 |
| 阵列规模 | 4×4 PE（可参数扩展至 8×8） |
| 频率 | 200 MHz ~ 1 GHz |
| 接口 | AXI4-Lite（寄存器配置）+ AXI4（DMA 数据搬移） |
| 功耗策略 | 时钟门控 + 动态频率调整 (DFS) + 多电压域 |

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        SoC (CPU)                                 │
│  ┌──────────┐                                                   │
│  │ Cortex-M0 │                                                 │
│  │ / RISC-V  │                                                 │
│  └────┬──────┘                                                   │
│       │ AXI Bus                                                  │
└───────┼──────────────────────────────────────────────────────────┘
        │
        ├─── AXI4-Lite ───► ┌──────────────────────────────────┐
        │                    │       NPU Top (npu_top)          │
        │                    │  ┌────────────────────────────┐  │
        │                    │  │  AXI-Lite Register File    │  │
        │                    │  │  (npu_axi_lite)             │  │
        │                    │  └──────────┬─────────────────┘  │
        │                    │             │ ctrl/status          │
        │                    │  ┌──────────▼─────────────────┐  │
        │                    │  │  NPU Controller            │  │
        │                    │  │  (npu_ctrl)                 │  │
        │                    │  │  - 配置解析                │  │
        │                    │  │  - 计算调度 FSM            │  │
        │                    │  │  - 中断生成                │  │
        │                    │  └──┬─────────┬──────────────┘  │
        │                    │     │         │                  │
        │                    │  ┌──▼──────┐ ┌▼──────────────┐  │
        │                    │  │  DMA    │ │  Systolic     │  │
        │                    │  │ (npu_dma)│ │  PE Array     │  │
        │                    │  │         │ │ (pe_array)    │  │
        │                    │  │ AXI4   │ │ 4×4 = 16 PE  │  │
        │                    │  │ Master │ │ 32-bit MAC   │  │
        │                    │  └────┬────┘ └──────┬───────┘  │
        │                    │       │              │           │
        │                    │  ┌────▼──────────────▼───────┐  │
        │                    │  │  Result Buffer (FIFO)     │  │
        │                    │  └───────────────────────────┘  │
        │                    │                                  │
        │                    │  ┌───────────────────────────┐  │
        │                    │  │  Power Management         │  │
        │                    │  │  - Clock Gating (cg_en)   │  │
        │                    │  │  - DFS (div_sel)          │  │
        │                    │  └───────────────────────────┘  │
        └────────────────────┼──────────────────────────────────┘
                             │
                    AXI4 Master (DMA)
                             │
                        ┌────▼────┐
                        │  DRAM   │
                        │ (SRAM/  │
                        │  DDR)   │
                        └─────────┘
```

---

## 3. 模块说明

### 3.1 AXI4-Lite 寄存器接口 (`rtl/axi/npu_axi_lite.v`)

| 偏移 | 名称 | 位宽 | R/W | 描述 |
|---:|---|:---:|:---:|---|
| 0x00 | CTRL | 32 | RW | 控制寄存器：bit0=start, bit1=abort, [3:2]=mode(INT8/INT16/FP16), [5:4]=stat_mode(WS/OS) |
| 0x04 | STATUS | 32 | RO | 状态寄存器：bit0=busy, bit1=done, bit2=int_en |
| 0x08 | INT_EN | 32 | RW | 中断使能 |
| 0x0C | INT_CLR | 32 | W | 中断清除（写1清） |
| 0x10 | M_DIM | 32 | RW | 矩阵 M 维度 |
| 0x14 | N_DIM | 32 | RW | 矩阵 N 维度 |
| 0x18 | K_DIM | 32 | RW | 矩阵 K 维度 |
| 0x20 | W_ADDR | 32 | RW | 权重 DRAM 起始地址 |
| 0x24 | A_ADDR | 32 | RW | 激活 DRAM 起始地址 |
| 0x28 | R_ADDR | 32 | RW | 结果 DRAM 起始地址 |
| 0x30 | ARR_CFG | 32 | RW | 阵列配置：[3:0]=act_rows, [7:4]=act_cols |
| 0x34 | CLK_DIV | 32 | RW | 时钟分频：[2:0]=div_sel (÷1/÷2/÷4/÷8) |
| 0x38 | CG_EN | 32 | RW | 时钟门控使能 |

### 3.2 DMA 控制器 (`rtl/axi/npu_dma.v`)

- AXI4 Master 接口，支持 INCR 突发传输
- 3 通道：权重 DMA、激活 DMA、结果 DMA
- FIFO 缓冲解耦 AXI 带宽与 PE 阵列吞吐
- 可配置突发长度 (BLEN = 1/4/8/16)

### 3.3 NPU 控制器 (`rtl/ctrl/npu_ctrl.v`)

FSM 状态机：
```
IDLE → LOAD_WEIGHT → COMPUTE → DRAIN_RESULT → DONE → IDLE
```

- 解析 AXI-Lite 寄存器配置
- 协调 DMA 数据搬移与 PE 阵列计算时序
- 生成中断信号通知 CPU 计算完成
- 支持 abort 中止当前计算

### 3.4 PE 阵列 (`rtl/array/pe_array.v`)

- 4×4 = 16 个 PE 单元，可参数化为 ROWS × COLS
- Weight-Stationary：权重从顶部预加载，激活水平脉动
- Output-Stationary：权重+激活流入，内部累加
- 支持动态配置活跃行列（低功耗）

### 3.5 PE 单元 (`rtl/pe/pe_top.v`)

32-bit MAC 单元，3 级流水线：
```
Stage0: Input Reg
Stage1: MUL (INT8 8b×8b / INT16 16b×16b / FP16)
Stage2: ACC (32-bit accumulate)
```

### 3.6 电源管理 (`rtl/power/npu_power.v`)

- **时钟门控**：per-row / per-column 独立门控
- **DFS**：可编程分频 ÷1 / ÷2 / ÷4 / ÷8
- **门控控制**：由 `CG_EN` 寄存器和阵列活跃配置自动推导

---

## 4. 数据流

### Weight-Stationary (GEMM: C = A × B)

```
1. CPU 写配置寄存器 (M, N, K, 地址, 模式)
2. DMA 从 DRAM 加载权重 B → PE 阵列（从顶部注入）
3. DMA 逐行从 DRAM 加载激活 A → 从左侧注入
4. PE 阵列脉动计算，部分和向下累加
5. 结果从底部输出 → Result FIFO → DMA 写回 DRAM
6. NPU 控制器置位 done 标志 → 中断通知 CPU
```

---

## 5. 目录结构

```
NPU_prj/
├── rtl/
│   ├── pe/
│   │   ├── pe_top.v          # 32-bit PE (INT8/INT16/FP16 MAC)
│   │   └── fp16_mul.v        # FP16 乘法器
│   ├── array/
│   │   ├── pe_array.v        # M×N systolic 阵列（含时钟门控）
│   │   └── array_ctrl.v      # [废弃→由 npu_ctrl 替代]
│   ├── axi/
│   │   ├── npu_axi_lite.v    # AXI4-Lite 从机 + 寄存器文件
│   │   └── npu_dma.v         # AXI4 Master DMA（3通道）
│   ├── ctrl/
│   │   └── npu_ctrl.v        # NPU 顶层控制器 FSM
│   ├── power/
│   │   └── npu_power.v       # 时钟门控 + DFS
│   ├── top/
│   │   └── npu_top.v         # NPU 顶层集成
│   └── common/
│       ├── fifo.v            # 参数化同步 FIFO (sync_fifo)
│       ├── axi_monitor.v     # AXI 总线带宽监控
│       └── op_counter.v      # NPU 操作计数与性能分析
├── tb/
│   ├── tb_pe_top.v           # PE 单元测试
│   └── tb_npu_top.v          # NPU 系统级测试bench
├── sim/wave/
├── doc/
│   ├── architecture.md       # 本文档（架构设计）
│   └── user_manual.md        # 用户手册（使用指南）
├── scripts/
│   ├── run_sim.sh            # Linux PE 仿真
│   ├── run_sim.ps1           # Windows PE 仿真
│   └── run_full_sim.ps1      # Windows NPU 全系统仿真
├── constraints/
│   └── npu_fpga.xdc          # FPGA 综合约束 [TODO]
└── README.md
```

---

## 6. 性能估算

### 算力 (INT8, 4×4 阵列, WS 模式)

- 每 PE 每 cycle 执行 1 次 INT8 MAC (2 OP)
- 16 PE × 2 OP/cycle = 32 OP/cycle
- @ 500 MHz: 32 × 500M = **16 GOPS**
- @ 1 GHz, 8×8 阵列: 64 × 2 × 1G = **128 GOPS**

> **注**：要达到 1 TOPS，需要 8×8 阵列 @ 1 GHz 或更大阵列/更高频率。
> 基础版本以 0.5 TOPS 为目标需 32×32 阵列 @ 500 MHz。
> 当前实现为可扩展架构，通过参数 ROWS/COLS 可直接扩展。

### 功耗优化

- 时钟门控：闲置行列零切换功耗
- DFS：低负载时降频运行
- 多电压域：SoC 侧 / NPU 侧 独立供电

---

## 7. 参考

- *Efficient Processing of Deep Neural Networks* (Sze et al., 2017)
- *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow* (Chen et al., 2016)
- AMBA AXI4 / AXI4-Lite Protocol Specification (ARM)
- IEEE 754-2008 Half Precision Standard
