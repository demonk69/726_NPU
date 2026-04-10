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
│   │   ├── fp16_mul.v            # IEEE 754 FP16 乘法器（含渐进下溢）
│   │   ├── fp16_add.v            # FP16 加法器
│   │   └── fp32_add.v            # FP32 累加器（WS/OS FP16 精度提升）
│   ├── array/
│   │   ├── pe_array.v            # ROWS×COLS 参数化 PE 阵列（支持 OS/WS，验证规模 COLS=4/8/16/32）
│   │   └── array_ctrl.v          # [deprecated → npu_ctrl]
│   ├── axi/
│   │   ├── npu_axi_lite.v        # AXI4-Lite 从机 + 14 个配置寄存器
│   │   └── npu_dma.v             # AXI4 Master DMA（3通道，r_pending 机制）
│   ├── ctrl/
│   │   └── npu_ctrl.v            # NPU 控制器 FSM（9状态 tile-loop，含 S_DRAIN 流水线排空）
│   ├── buf/
│   │   └── pingpong_buf.v        # Ping-Pong 双缓冲区（DMA 与 PE 并行）
│   ├── soc/
│   │   ├── soc_top.v             # SoC 顶层（PicoRV32 + NPU + SRAM + DRAM）
│   │   ├── soc_mem.v             # SoC SRAM
│   │   ├── dram_model.v          # 双端口 DRAM 模型
│   │   └── axi_lite_bridge.v     # PicoRV32 iomem → AXI4-Lite 桥接
│   ├── power/
│   │   └── npu_power.v           # 时钟门控（per-row/col）+ DFS（÷1/2/4/8）[行为模型，npu_clk/row_clk_gated/col_clk_gated 当前悬空未驱动 PE]
│   ├── top/
│   │   └── npu_top.v             # NPU 顶层集成
│   └── common/
│       ├── fifo.v                # 参数化同步 FIFO (sync_fifo)
│       ├── axi_monitor.v         # AXI 总线带宽监控
│       └── op_counter.v          # NPU 操作计数与性能分析
├── tb/
│   ├── tb_pe_top.v               # PE 单元测试（19 检查点）
│   ├── tb_fp16_mul.v             # FP16 乘法器（44 用例）
│   ├── tb_fp16_add.v             # FP16 加法器（20 用例）
│   ├── tb_comprehensive.v        # NPU 综合测试（8 场景）
│   ├── tb_classifier.v           # 三层 FC 网络推理（Tiny-FC-Net）
│   ├── tb_npu_top.v              # NPU 系统级测试
│   ├── tb_soc.v                  # SoC 集成测试
│   ├── tb_array_scale.v          # PE 阵列规模验证框架
│   ├── tb_array_scale_core.v     # 阵列规模验证核心逻辑
│   ├── array_scale/              # 阵列规模验证数据生成
│   └── soc_test.S                # RISC-V SoC 测试固件
├── sim/wave/                     # VCD 波形输出
├── doc/
│   ├── architecture.md           # 系统架构设计
│   ├── module_reference.md       # RTL 模块详细文档
│   ├── simulation_guide.md       # 仿真快速上手教程
│   └── user_manual.md            # 用户手册
├── constraints/
│   └── npu_fpga.xdc              # [TODO] FPGA 综合约束
├── scripts/
│   ├── run_sim.sh                # Linux PE 单元仿真
│   ├── run_sim.ps1               # Windows PE 单元仿真
│   ├── run_full_sim.ps1          # NPU 全系统仿真
│   ├── run_classifier_sim.ps1    # 分类器推理仿真
│   ├── run_soc_sim.ps1           # SoC 集成仿真
│   ├── run_array_scale.ps1       # K 深度规模验证（1x1 PE，K=4/8/16/32）

│   └── gen_classifier_data.py    # 分类器数据生成
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

| 配置 | 理论峰值算力 (INT8) | 备注 |
|---|---|---|
| 4×4 @ 500 MHz | 16 GOPS | 全阵列满载理论值 |
| 8×8 @ 1 GHz | 128 GOPS | 扩展配置 |
| 16×16 @ 1 GHz | 512 GOPS | 扩展配置 |
| 32×32 @ 1 GHz | **2 TOPS** | 扩展配置 |

> ⚠️ **当前实现说明**：当前 tile-loop 按 `C[i][j]` 单个输出元素串行推进，每个 tile 占用 K+若干额外周期，实际有效吞吐远低于理论峰值。DMA 读写通道当前使用单拍 burst（`arlen/awlen=0`），AXI 带宽利用率偏低，"目标带宽利用率 >80%"为设计目标，尚未达到。  
> 功耗策略：时钟门控 + DFS 寄存器接口已实现，但 `npu_power` 输出当前悬空，未驱动 PE 主时钟路径。

---

## 快速仿真

```powershell
# Windows — PE 单元测试
cd NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1

# Windows — NPU 综合测试（8 场景，推荐日常回归）
iverilog -g2012 -DDUMP_VCD -o sim\tb_comprehensive.vvp rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v rtl\array\pe_array.v rtl\buf\pingpong_buf.v rtl\power\npu_power.v rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\top\npu_top.v tb\tb_comprehensive.v
vvp sim\tb_comprehensive.vvp

# Windows — NPU 全系统仿真（含带宽与操作统计）
powershell -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1

# Windows — 三层 FC 网络推理测试
powershell -ExecutionPolicy Bypass -File scripts\run_classifier_sim.ps1

# Windows — K 深度规模验证（1x1 PE，K=4/8/16/32 × INT8/FP16 × WS/OS）
powershell -ExecutionPolicy Bypass -File scripts\run_array_scale.ps1


# Windows — SoC 集成测试（PicoRV32 + NPU）
powershell -ExecutionPolicy Bypass -File scripts\run_soc_sim.ps1

# Windows — 全量回归（预期：903 PASS, 0 FAIL）
powershell -ExecutionPolicy Bypass -File scripts\run_regression.ps1
```

```bash
# Linux — PE 单元测试
bash scripts/run_sim.sh
```

依赖：[Icarus Verilog](https://bleyer.org/icarus/) >= 11.0, [GTKWave](https://gtkwave.sourceforge.net/)

详细使用说明见 [doc/user_manual.md](doc/user_manual.md)。

---

## 后续 TODO

- [ ] `constraints/npu_fpga.xdc` — FPGA 综合约束
- [ ] 形式验证约束
- [ ] ASIC 综合脚本
- [ ] 电源管理行为模型替换（FPGA: BUFGCE, ASIC: ICG），并将 `npu_clk`/`row_clk_gated`/`col_clk_gated` 真正接入 PE 主时钟路径
- [ ] DMA burst 优化：当前 `arlen/awlen=0`（单拍），需实现多拍 INCR burst 以提升带宽利用率至目标 >80%
- [ ] K 维度 tiling：当前 K > PPBuf 深度时行为未验证，需补充 K-split 机制
- [ ] `target_col` 位宽扩展：当前为 1-bit，当 COLS > 2 时需扩展为 `$clog2(COLS)` 位

## 已完成验证

- [x] `tb_pe_top.v` — PE 单元测试（INT8/FP16 × WS/OS，19 检查点 PASS）
- [x] `tb_fp16_mul.v` — FP16 乘法器（44 用例 PASS，含次正规数/渐进下溢）
- [x] `tb_fp16_add.v` — FP16 加法器（20 用例 PASS）
- [x] `tb_comprehensive.v` — NPU 综合测试（8 场景 PASS，含 back-to-back 和边界值）
- [x] `tb_classifier.v` — 三层 FC 网络推理（FC1→ReLU→FC2→ReLU→FC3，PASS）
- [x] `tb_npu_top.v` — NPU 系统级测试（INT8 4×4 WS，含性能报告 PASS）
- [x] `tb_array_scale` — K 深度规模验证（ROWS=1, COLS=1, K=4/8/16/32 × INT8/FP16 × WS/OS，16/16 PASS）

- [x] `tb_multi_rc_comprehensive` — 多行多列综合测试（ROWS=2, COLS=2，13/13 PASS）
- [x] `tb_ws_multibeat` — WS 多轮点积验证（K=4/8/16，17/17 PASS）
- [x] `tb_fp16_e2e` — FP16 全链路端到端验证（9/9 PASS）
- [x] Ping-Pong Buffer（DMA 与 PE 并行，THRESHOLD=16 提前启动）
- [x] SoC 集成验证（PicoRV32 + NPU + SRAM + DRAM，tb_soc.v ✅ 287 cycles PASS，C=[19,22,43,50]）
- [x] Bug 修复记录：
  - OS flush 数据黑洞：flush 周期先累加当前乘积再输出清零
  - FP16 乘法器：次正规数打包丢失隐式 1（flush-to-zero → 渐进下溢）
  - FIFO phantom beat：`pe_valid_q` 延迟一拍避免 NBA 未更新就读出
  - DMA SUBW=2 打包：FP16/INT16 每 32-bit 字 2 元素
  - **Bug 5**（WS DMA 重复触发）：`S_WB_WAIT` 不再持续拉 `dma_r_start`
  - **Bug 6**（OS TIMEOUT）：`npu_dma` 新增 `r_pending` 机制，等待 FIFO 非空后进入 R_WRITE
  - **Bug 7**（INT8 WS got=0）：PPBuf `rd_data` 组合输出 vs `w_fp16_shift` 寄存器 1-cycle 延迟，引入 `w_int8_ready_d` sticky 信号
  - **Bug 8**（act_reg 跨运行污染）：`pe_array.v` systolic `act_reg` flush 时清零，防止 COLS>1 的 col+1 首元素残留
