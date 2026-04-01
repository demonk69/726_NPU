# NPU_prj — 嵌入式低功耗 NPU 加速器

> 面向边缘计算的 NPU 加速器 IP，通过 AXI 总线挂载于 SoC（Cortex-M0 / RISC-V），
> 支持 INT8 / INT16 / FP16 混合精度矩阵乘加，4×4 脉动阵列，目标算力 ≥0.5 TOPS。

---

## 目录结构

```
NPU_prj/
├── rtl/
│   ├── pe/
│   │   ├── pe_top.v              # 32-bit PE (INT8/INT16/FP16, WS/OS, 3-stage pipeline)
│   │   └── fp16_mul.v            # IEEE 754 FP16 乘法器
│   ├── array/
│   │   ├── pe_array.v            # M×N 动态可调 systolic 脉动阵列
│   │   └── array_ctrl.v          # [deprecated → npu_ctrl]
│   ├── axi/
│   │   ├── npu_axi_lite.v        # AXI4-Lite 从机 + 14 个配置寄存器
│   │   └── npu_dma.v             # AXI4 Master DMA（权重/激活/结果 3 通道）
│   ├── ctrl/
│   │   └── npu_ctrl.v            # NPU 控制器 FSM（IDLE→LOAD→COMPUTE→DRAIN→DONE）
│   ├── power/
│   │   └── npu_power.v           # 时钟门控（per-row/col）+ DFS（÷1/2/4/8）
│   ├── top/
│   │   └── npu_top.v             # NPU 顶层集成
│   └── common/
│       ├── fifo.v                # 参数化同步 FIFO (sync_fifo)
│       ├── axi_monitor.v         # AXI 总线带宽监控
│       └── op_counter.v          # NPU 操作计数与性能分析
├── tb/
│   ├── tb_pe_top.v               # PE 单元测试（4 场景）
│   └── tb_npu_top.v              # NPU 系统级测试bench（AXI BFM + DRAM）
├── sim/wave/                     # VCD 波形输出
├── doc/
│   ├── architecture.md           # 系统架构设计文档
│   └── user_manual.md            # 用户手册
├── constraints/
│   └── npu_fpga.xdc              # [TODO] FPGA 约束
├── scripts/
│   ├── run_sim.sh                # Linux PE 单元仿真
│   ├── run_sim.ps1               # Windows PE 单元仿真
│   └── run_full_sim.ps1          # Windows NPU 全系统仿真
└── README.md
```

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│  SoC (Cortex-M0 / RISC-V)                                   │
│       │ AXI Bus                                              │
└───────┼──────────────────────────────────────────────────────┘
        │
   AXI4-Lite ──► NPU Top
        │         ┌─────────────────────────────────────┐
        │         │  AXI-Lite 寄存器 (14 个配置寄存器)    │
        │         └──────────┬──────────────────────────┘
        │                    │
        │         ┌──────────▼──────────────────────────┐
        │         │  NPU Controller (FSM)                │
        │         └──┬──────────────────┬───────────────┘
        │      ┌───▼────┐      ┌────────▼────────┐
        │      │  DMA   │      │  4×4 PE Array   │
        │      │ 3-ch   │      │  16×32bit MAC   │
        │      └───┬────┘      └───────┬─────────┘
        │          │                  │
        │    AXI4 Master        Result FIFO
        │          │                  │
        └──────────┼──────────────────┘
                   ▼
               DRAM (SRAM/DDR)
```

详细架构说明见 [doc/architecture.md](doc/architecture.md)，使用指南见 [doc/user_manual.md](doc/user_manual.md)。

---

## 寄存器映射（AXI4-Lite）

| 偏移 | 名称 | R/W | 描述 |
|---:|---|:---:|---|
| 0x00 | CTRL | RW | bit0=start, bit1=abort, [3:2]=mode, [5:4]=stat_mode |
| 0x04 | STATUS | RO | bit0=busy, bit1=done |
| 0x08 | INT_EN | RW | 中断使能 |
| 0x0C | INT_CLR | W | 中断清除 |
| 0x10 | M_DIM | RW | 矩阵 M |
| 0x14 | N_DIM | RW | 矩阵 N |
| 0x18 | K_DIM | RW | 矩阵 K |
| 0x20 | W_ADDR | RW | 权重 DRAM 地址 |
| 0x24 | A_ADDR | RW | 激活 DRAM 地址 |
| 0x28 | R_ADDR | RW | 结果 DRAM 地址 |
| 0x30 | ARR_CFG | RW | [3:0]=act_rows, [7:4]=act_cols |
| 0x34 | CLK_DIV | RW | [2:0]=div_sel (÷1/2/4/8) |
| 0x38 | CG_EN | RW | 时钟门控使能 |

---

## 性能估算

| 配置 | 算力 (INT8) |
|---|---|
| 4×4 @ 500 MHz | 16 GOPS |
| 8×8 @ 1 GHz | 128 GOPS |
| 16×16 @ 1 GHz | 512 GOPS |
| 32×32 @ 1 GHz | **2 TOPS** |

功耗策略：时钟门控 + DFS + 多电压域。

---

## 快速仿真

```powershell
# Windows — PE 单元测试
cd NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1

# Windows — NPU 全系统仿真（含带宽与操作统计）
powershell -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1
```

```bash
# Linux — PE 单元测试
bash scripts/run_sim.sh
```

依赖：[Icarus Verilog](https://bleyer.org/icarus/) >= 11.0, [GTKWave](https://gtkwave.sourceforge.net/)

详细使用说明见 [doc/user_manual.md](doc/user_manual.md)。

---

## 后续 TODO

- [ ] `tb_npu_top.v` — 启用 Test 2 (OS) 和 Test 3 (8×8 tiled)
- [ ] `constraints/npu_fpga.xdc` — FPGA 综合约束
- [ ] 形式验证约束
- [ ] ASIC 综合脚本
- [ ] 电源管理行为模型替换（FPGA: BUFGCE, ASIC: ICG）
- [ ] DMA 多 tile 调度（大矩阵自动分块）
