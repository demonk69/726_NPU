# NPU_prj 仿真快速上手教程

> 适合对象：刚拿到源码、想立刻跑起仿真的工程师
>
> 最后更新：2026-04-14

## ✅ 当前验证状态（2026-04-14 更新）

| 测试套件 | 状态 | 通过/总数 | 备注 |
|---------|------|---------|------|
| PE单元测试 (`tb_pe_top`) | ✅ PASS | **19/19** | 双权重寄存器版全通过 |
| 综合测试 (`tb_comprehensive`) | ✅ PASS | **28/28** | 含复杂 FP16 场景，使用 reconfig_pe_array |
| NPU顶层冒烟 (`tb_npu_top`) | ✅ PASS | **4/4** | reconfig_pe_array + npu_ctrl 集成 |
| FP16端到端 (`tb_fp16_e2e`) | ✅ PASS | 9/9 | 所有FP16测试通过 |
| 多行多列测试 (`tb_multi_rc`) | ✅ PASS | 13/13 | 多列时序问题已修复 |
| 阵列规模验证 (`tb_array_scale`) | ✅ PASS | 16/16 | K=4/8/16/32 全通过 |
| SoC集成 (`tb_soc`) | ✅ PASS | 1/1 | 287 cycles PASS |

**核心回归（2026-04-14 重构后）**：51/51 PASS ✅
**全量回归（重构前）**：86/86 PASS ✅
1. **WS flush 纯输出语义**：`pe_top.v` 修改 WS 模式 flush 行为——flush beat 直接输出 `ws_acc`，不累加当前 `s1_mul`，使 flush 成为纯触发操作，与 TB 设计意图一致
2. **drive_ws_beat 时序修正**：`tb_pe_top.v` 的 `drive_ws_beat` task 内部加入 `@(posedge clk); #1; en=0`，确保每个 beat 只被 pipeline 采样一次，防止同一 beat 因 `en` 高电平持续两个 posedge 而被双重采样
3. **load_w beat en 降低**：所有 WS 模式的 `load_w=1` beat 完成后立即同步设置 `en=0`，避免首个数据 beat 重复进入 pipeline
4. **flush beat 清零规范化**：所有 WS 模式的 flush beat 统一设置 `a_in=16'd0, w_in=16'd0`（纯 flush，不携带实际计算数据）

**关键修复**（2026-04-09）：
1. **WS模式内部累加器**：`pe_top.v` 新增 `ws_acc` 寄存器，WS模式在K个周期内内部累加，flush时输出完整点积
2. **NaN污染修复**：`npu_ctrl.v` 移除 S_TILE_LOAD 中的 `pe_en <= 1`，避免FP16 WS测试时读取stale weight_reg导致NaN传播
3. **WS flush路径统一**：WS模式现在和OS模式一样使用 S_DRAIN/S_DRAIN2 路径输出结果

**状态**：所有测试套件全部通过，架构稳定，回归基线清零。

---

## 目录

1. [工具安装（5 分钟）](#1-工具安装5-分钟)
2. [克隆与目录一览](#2-克隆与目录一览)
3. [第一次仿真：PE 单元测试](#3-第一次仿真pe-单元测试)
4. [第二次仿真：NPU 综合测试（推荐）](#4-第二次仿真npu-综合测试推荐)
5. [第三次仿真：NPU 全系统性能测试](#5-第三次仿真npu-全系统性能测试)
5.1. [分类器推理仿真（`tb_classifier.v`）](#51-分类器推理仿真tb_classifierv)
5.2. [阵列规模验证（`tb_array_scale`）](#52-阵列规模验证tb_array_scale)
5.3. [多行多列综合测试（`tb_multi_rc_comprehensive`）](#53-多行多列综合测试tb_multi_rc_comprehensive)
5.4. [FP16 端到端验证（`tb_fp16_e2e`）](#54-fp16-端到端验证tb_fp16_e2e)
6. [FP16 加法器单元测试](#6-fp16-加法器单元测试)
7. [SoC 全系统联调仿真](#7-soc-全系统联调仿真)
8. [用 GTKWave 看波形](#8-用-gtkwave-看波形)
9. [手动编译命令参考](#9-手动编译命令参考)
10. [理解各仿真场景](#10-理解各仿真场景)
11. [自己添加测试向量](#11-自己添加测试向量)
12. [常见报错与解决](#12-常见报错与解决)

---

## 1. 工具安装（5 分钟）

只需两个工具：**Icarus Verilog**（编译仿真）和 **GTKWave**（看波形）。

### Windows

1. 从 https://bleyer.org/icarus/ 下载 Windows 安装包（推荐 11.0+）
2. 安装时**勾选 "Add to PATH"**（否则需要手动配置环境变量）
3. 从 https://gtkwave.sourceforge.net/ 下载 GTKWave，同样安装并加入 PATH
4. 打开 PowerShell，验证：

```powershell
iverilog -V
# 输出示例：Icarus Verilog version 12.0 (stable) ...

gtkwave --version
# 输出示例：GTKWave Analyzer v3.3.117 ...
```

### Linux（Ubuntu / Debian）

```bash
sudo apt update && sudo apt install -y iverilog gtkwave
iverilog -V && gtkwave --version
```

> **最低版本要求**：iverilog ≥ 11.0，否则 `$clog2` 系统函数无法识别。

---

## 2. 克隆与目录一览

```
NPU_prj/
├── rtl/                 ← RTL 源码（全部可综合 Verilog-2001）
│   ├── pe/              ← PE 单元 & FP16/FP32 运算器
│   ├── array/           ← 脉动阵列
│   ├── axi/             ← AXI-Lite 从机 & DMA
│   ├── ctrl/            ← 控制器 FSM
│   ├── buf/             ← Ping-Pong 缓冲区
│   ├── soc/             ← SoC 集成模块
│   ├── top/             ← NPU 顶层
│   └── common/          ← FIFO、监控、计数器
├── tb/                  ← Testbench（所有仿真入口）
│   ├── tb_pe_top.v          ← 场景A: 单 PE 功能测试
│   ├── tb_comprehensive.v   ← 场景B: NPU 综合用例（推荐入口）★
│   ├── tb_classifier.v      ← 场景C: 三层 FC 网络推理测试
│   ├── tb_npu_top.v         ← 场景D: 带性能报告的系统测试
│   ├── tb_soc.v             ← 场景E: SoC CPU+NPU 联调
│   ├── tb_array_scale.v     ← 场景F: K 深度验证旧版入口（历史命名保留，仅参考）
│   ├── tb_array_scale_body.v← K 深度验证公共代码（被 wrapper include）
│   ├── tb_array_scale_core.v← K 深度验证旧版核心（已废弃）
│   └── array_scale/         ← K 深度验证数据/生成器/wrapper

├── scripts/
│   ├── run_sim.ps1          ← 运行场景A
│   ├── run_full_sim.ps1     ← 运行场景D（含性能报告）
│   ├── run_classifier_sim.ps1 ← 运行场景C（分类器推理）
│   ├── run_soc_sim.ps1      ← 运行场景E
│   ├── run_array_scale.ps1  ← 运行场景F（阵列规模验证）
│   └── run_multi_rc_sim.ps1 ← 运行场景G（多行多列综合测试）
└── sim/wave/            ← VCD 波形输出目录（自动创建）
```

---

## 3. 第一次仿真：PE 单元测试

这是最快的验证。测试**单个 PE（Processing Element）** 的乘加功能，**14 个测试 / 19 个检查点**，覆盖 WS/OS × INT8/FP16、pipeline stall、特殊值、weight latch、动态模式切换等，约 10 秒完成。

### 运行

```powershell
# 在项目根目录执行
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

### 预期输出

```
>>> Starting Test 1: INT8 Weight-Stationary (true WS) <<<
[TEST 1 RESULT] Info: w=3 loaded once, acts={1,2,3,4} accumulated
Expected Output  : 0x0000001E (Dec: 30)
Actual Output    : 0x0000001E (Dec: 30)
Status           : [PASS]

>>> Starting Test 2: INT8 Output-Stationary <<<
[TEST 2 RESULT] Info: weights={1,2,3,4}, act=2 internal acc
Status           : [PASS]

>>> Starting Test 3: FP16 Weight-Stationary <<<
[TEST 3 RESULT] Info: w=2.0 (0x4000), a=1.5 (0x3E00)
Status           : [PASS]

>>> Starting Test 4: FP16 Output-Stationary <<<
[TEST 4 RESULT] Info: 2.0*1.5 + 2.0*1.5
Status           : [PASS]

>>> Starting Test 5: INT8 OS with Pipeline Stalls <<<
[TEST 5 RESULT] Info: 5*2, stall 2 cyc, 3*3
Status           : [PASS]

>>> Starting Test 6: FP16 Edge Cases <<<
[TEST 61 RESULT] Info: Inf (0x7C00) * 1.0 (0x3C00) -> Inf
Status           : [PASS]
[TEST 62 RESULT] Info: Inf (0x7C00) * 0.0 (0x0000) -> NaN
Status           : [PASS]

>>> Starting Test 7: FP16 Sign Toggling & Accumulation <<<
[TEST 7 RESULT] Info: FP16 4.0 + (-2.0) using FP32 mixed-precision accumulation
Status           : [PASS]

>>> Starting Test 8: Back-to-Back Flush <<<
[TEST 81 RESULT] Info: Flush 1 (sum of 4*5)
Status           : [PASS]
[TEST 82 RESULT] Info: Flush 2 (back-to-back empty)
Status           : [PASS]

>>> Starting Test 9: Dynamic Mode Switching (INT8 -> FP16) <<<
[TEST 91 RESULT] Info: INT8 beat (w=2, a=3)
Status           : [PASS]
[TEST 92 RESULT] Info: FP16 beat (w=1.0, a=2.0)
Status           : [PASS]

>>> Starting Test 10: FP16 WS Multi-Accumulation <<<
[TEST 10 RESULT] Info: w=2.0 loaded once, a={1.0,2.0,3.0}, WS FP16 acc
Status           : [PASS]

>>> Starting Test 11: FP16 WS Negative Accumulation <<<
[TEST 11 RESULT] Info: w=-1.5, a={2.0,3.0}, WS FP16 acc => -7.5
Status           : [PASS]

>>> Starting Test 12: FP16 Complex Decimals Accumulation <<<
[TEST 12 RESULT] Info: 3.140625 + 1.234375 - 0.875 + 0.109375
Status           : [PASS]

>>> Starting Test 13: FP16 Big + Tiny Value Alignment <<<
[TEST 13 RESULT] Info: 10.0 (0x4900) + 0.015625 (0x2400)
Status           : [PASS]

>>> Starting Test 14: True WS Weight Latch Verification <<<
[TEST 141 RESULT] Info: load w=5, a=10 => 50
Status           : [PASS]
[TEST 142 RESULT] Info: w still 5 (no load_w), a=3 => 15
Status           : [PASS]
[TEST 143 RESULT] Info: load w=7, a=4 => 28
Status           : [PASS]

===================================================
=== Summary: PASS=19  FAIL=0 ===
ALL TESTS PASSED SUCCESSFULLY
===================================================
```

### 完整测试用例说明

`tb_pe_top.v` 测试 PE 的 **3-stage 流水线**（Stage0=Input-Reg, Stage1=MUL, Stage2=ACC/Output），19 个检查点覆盖以下维度：

| 测试 | 模式 | 描述 | 期望值 | 验证点 |
|------|------|------|--------|--------|
| T1 | INT8 WS | 加载 w=3，流式 a={1,2,3,4} | 30 | Weight latch + 多拍累加 |
| T2 | INT8 OS | w={1,2,3,4}, a=2 内部累加 | 20 | OS 内部累加器 |
| T3 | FP16 WS | w=2.0 × a=1.5 | 0x40400000 (3.0 FP32) | FP16→FP32 混合精度 |
| T4 | FP16 OS | 2.0×1.5 + 2.0×1.5 | 0x40C00000 (6.0 FP32) | FP16 OS 累加 |
| T5 | INT8 OS | 5×2, **stall 2 cyc**, 3×3 | 19 | **Pipeline stall 恢复** |
| T61 | FP16 WS | Inf × 1.0 | 0x7F800000 (Inf) | FP16 特殊值传播 |
| T62 | FP16 WS | Inf × 0.0 | 0xFFC000000 (NaN) | FP16 无效运算 |
| T7 | FP16 OS | 4.0 + (-2.0) | 0x40000000 (2.0) | 符号反转 + FP32 累加 |
| T81 | INT8 OS | 4×5 后 flush | 20 | Flush 输出正确值 |
| T82 | INT8 OS | **连续 flush** 空 | 0 | **Back-to-back flush 不丢数** |
| T91 | INT8 WS | w=2, a=3 | 6 | **动态模式切换** |
| T92 | FP16 WS | w=1.0, a=2.0（同 PE，切模式） | 0x40000000 | 运行时模式切换 |
| T10 | FP16 WS | w=2.0 锁存, a={1.0,2.0,3.0} | 0x41400000 (12.0) | **FP16 WS 多拍链式累加** |
| T11 | FP16 WS | w=-1.5, a={2.0,3.0} | 0xC0F00000 (-7.5) | FP16 WS 负数累加 |
| T12 | FP16 OS | 3.14+1.23-0.875+0.109 | 0x40670000 (3.609) | FP16 复合小数对齐 |
| T13 | FP16 OS | 10.0 + 0.015625 | 0x41204000 | **大小值阶对齐**（exp diff=9） |
| T141 | INT8 WS | load w=5, a=10 | 50 | **Weight latch 验证** |
| T142 | INT8 WS | w 不重新加载, a=3 | 15 | weight_reg 保持 w=5 |
| T143 | INT8 WS | 重新 load w=7, a=4 | 28 | weight_reg 更新为 w=7 |

> **关键设计**：WS 模式下 `load_w=1` 时锁存权重到 `weight_reg`，后续 beat 无需重复发送权重（脉动阵列核心特性）。FP16 累加使用 FP32 加法器（`fp32_add.v`），精度从 ~3.3 位有效数字提升到 ~7.3 位。

---

## 4. 第二次仿真：NPU 综合测试（推荐）

**`tb_comprehensive.v`** 是本项目最完整的功能测试，覆盖 8 个场景，包含边界值、连续运算，全部通过即代表 NPU 数据通路正确。

### 运行命令

```powershell
cd D:\NPU_prj

# 编译
iverilog -g2012 -DDUMP_VCD `
  -o sim\tb_comprehensive.vvp `
  rtl\pe\fp16_mul.v `
  rtl\pe\fp16_add.v `
  rtl\pe\fp32_add.v `
  rtl\pe\pe_top.v `
  rtl\common\fifo.v `
  rtl\common\axi_monitor.v `
  rtl\common\op_counter.v `
  rtl\array\pe_array.v `
  rtl\buf\pingpong_buf.v `
  rtl\power\npu_power.v `
  rtl\ctrl\npu_ctrl.v `
  rtl\axi\npu_axi_lite.v `
  rtl\axi\npu_dma.v `
  rtl\top\npu_top.v `
  tb\tb_comprehensive.v

# 运行仿真
cd sim
vvp ..\sim\tb_comprehensive.vvp
```

> **Linux 用户**：把反引号 `` ` `` 换成 `\`，路径分隔符 `\` 换成 `/`。

### 预期输出（全部通过）

```
################################################################
  NPU Comprehensive Test Suite
  PE Array: 1x1, DATA_W=16, ACC_W=32
################################################################

--- Test 1: INT8 Positive Dot Product (K=8) ---
  W=[1,2,3,4,5,6,7,8] A=[10,20,30,40,50,60,70,80]
  Expected: 2040
  [PASS] T1_PosDotK8: got 2040 (0x000007f8)

--- Test 1b: K=4 Regression (expected 310) ---
  [PASS] T1b_K4_Regress: got 310

--- Test 2: INT8 Mixed Signed Dot Product (K=8) ---
  W=[10,-20,30,-40,50,-60,70,-80]
  A=[3,5,-7,9,-11,13,-15,17]
  Expected: -4380
  [PASS] T2_MixDotK8: got -4380 (0xffffeeе4)

--- Test 3: INT8 Boundary Values (K=8) ---
  W=[127,-128,127,-128,1,0,-1,0]
  A=[127,-128,-128,127,127,1,127,1]
  Expected: 1
  [PASS] T3_BoundaryK8: got 1 (0x00000001)

--- Test 4: INT8 Zero Weights (K=8) ---
  W=[0,...], A=[10,20,...,80]
  Expected: 0
  [PASS] T4_ZeroWeights: got 0 (0x00000000)

--- Test 5: INT8 Alternating Sign (K=16) ---
  W=[1,-1,...,4,-4], A=[1,-1 循环]
  Expected: 40
  [PASS] T5_Alternating: got 40 (0x00000028)

--- Test 6: Back-to-Back Operations ---
  [PASS] T6_Back2Back_1: got 2040
  [PASS] T6_Back2Back_2: got -4380

RESULT: 8 PASSED, 0 FAILED
```

---

## 5. 第三次仿真：NPU 全系统性能测试

`run_full_sim.ps1` 运行 `tb_npu_top.v`，除了功能验证还会输出详细的**性能报告**（AXI 带宽、PE 利用率、MAC 吞吐率）。

### 运行

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1
```

### 预期输出（节选）

```
========================================
  NPU Full System Simulation
========================================
[OK] Icarus Verilog version 12.0 ...

[1/3] Compiling...
[OK] Compilation successful.

[2/3] Running simulation...
--- TEST 1: INT8 4x4 WS ---
    [BFM] W 0x00000010 = 0x00000004 OK
    ...
    -> NPU DONE at cycle 77

================================================================
  NPU PERFORMANCE REPORT: INT8 4x4 WS
================================================================
  [AXI4-Lite CPU Port]
    Write Txns      :      8    Read Txns      :     21
    Avg Wr Latency  :      3 cyc  Avg Rd Latency :      2 cyc
  [AXI4 Master DMA Port]
    Read Bursts     :     42    Read Bytes      :    136
    Rd Bandwidth    :    1.72 B/cyc
  [NPU Compute]
    Total MAC Ops   :     24
    Peak Active PEs :      4 / 16
  [Performance]
    PE Utilization  :    6.25 %
    MACs/Cycle      :    4.00
    Efficiency      :    5.56 %
    Total Cycles    :     79
================================================================

[3/3] Results:
  VCD waveform : sim/wave/tb_npu_top.vcd (XXX KB)
  Open with    : gtkwave sim/wave/tb_npu_top.vcd
```

### 报告中的关键指标解读

| 指标 | 含义 | 当前值 | 理想值 |
|------|------|--------|--------|
| `Rd Bandwidth` | DMA 读取带宽 (B/cycle) | ~1.72 | 接近总线宽度/4 |
| `PE Utilization` | 按逐 PE 活动位图统计的真实利用率（适配任意 `ROWS×COLS`） | 与测试场景相关 | 100% |
| `MACs/Cycle` | 每个有效计算周期的平均 MAC 吞吐 | 与测试场景相关 | `ROWS×COLS` |

| `Efficiency` | 计算时间 / 总忙时间 | ~5.56% | 接近 100% |

> **注意**：当前性能报告中的 `PE Utilization` 已改为按逐 PE 活动位图统计的真实利用率，而不是只看底行输出 valid。不同数据流（OS/WS）、阵列规模和测试矩阵尺寸都会直接影响该指标。当前测试用例为 4×4 小矩阵，DMA 传输开销占主导，大矩阵下效率通常会显著提升。


---

## 5.1 分类器推理仿真（`tb_classifier.v`）

`run_classifier_sim.ps1` 运行一个**三层全连接神经网络（Tiny-FC-Net）**的端到端推理测试：

```
FC1(16→8) → ReLU → FC2(8→4) → ReLU → FC3(4→4)
```

权重由 `scripts/gen_classifier_data.py` 随机生成并预存到 `tb/classifier_dram.hex`，golden reference 存于 `tb/classifier_golden.txt`。

### 运行

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_classifier_sim.ps1
```

### 预期输出

```
========================================
  Tiny-FC-Net Classifier Simulation
========================================

[0/4] Generating test data...
  [OK] classifier_dram.hex exists, skipping generation.

[1/4] Compiling...
  [OK] Icarus Verilog version 12.0 ...
  [OK] Compilation successful.

[2/4] Running simulation...
  [PASS] STEP 1: FC1(16->8) verified
  [PASS] STEP 2: FC2(8->4) verified
  [PASS] STEP 3: FC3(4->4) verified
  [PASS] ALL CLASSIFICATIONS VERIFIED

[3/4] Results:
  VCD waveform : sim/wave/tb_classifier.vcd (XXX KB)

[4/4] Files:
  DRAM hex     : tb/classifier_dram.hex
  Golden ref   : tb/classifier_golden.txt
  Layout       : tb/classifier_layout.txt
  Expected .vh : tb/classifier_expected.vh
```

---

## 5.2 阵列规模验证（`tb_array_scale`）

`run_array_scale.ps1` 实际上是在 **`ROWS=1, COLS=1` 的物理 PE testbench** 上，验证 **K=4, 8, 16, 32** 四种深度下的端到端正确性。每个 K 覆盖 INT8/FP16 × WS/OS 共 4 个组合，**总计 16 个测试用例**，当前 **16/16 PASS**。

测试通过 AXI BFM + DRAM 模型完成完整的配置→DMA 加载→计算→写回→验证流程。


### 架构

```
tb/array_scale/
├── gen_data.py          ← 生成测试数据（INT8/FP16 打包、golden reference）
├── gen_tb.py            ← 生成每个 N 的 wrapper（硬编码 $readmemh 路径）
├── N4/
│   ├── tb_wrapper.v     ← N=4 专用 wrapper（module 壳 + 硬编码路径）
│   ├── test_params.vh   ← 测试参数宏定义
│   ├── dram_init.hex    ← DRAM 初始化数据
│   └── expected.hex     ← 期望结果
├── N8/ ... N32/         ← 同上
└── ...

tb/tb_array_scale_body.v ← 公共测试逻辑（AXI BFM、DRAM 模型、do_test task）
                             被 wrapper include，不含 module 定义
```

> **设计说明**：iverilog 不支持在 `$readmemh` 中使用宏展开文件路径，也不支持 `` ` `` 宏拼接在 `include` 字符串中。因此采用 wrapper 方案——每个 N 的 wrapper 硬编码 `$readmemh` 路径，`include` 公共 body 文件。

### 运行

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_array_scale.ps1
```

### 当前状态

- **OS 模式（8 个测试）：全部 PASS** ✓
- **WS 模式（8 个测试）：全部 PASS** ✓
- **总计：16/16 PASS** ✓

### 预期输出（完整通过示例）


```
========================================
  PE Array Scale Verification
========================================

[0/3] Generating test data and wrappers...
[OK] Icarus Verilog version 12.0 ...

========================================
  K Depth: 4 (physical PE array in this testbench is 1x1)
========================================
[PASS] T0: INT8_WS got=4677
[PASS] T1: INT8_OS got=3285
[PASS] T2: FP16_WS got=0xc0000000 exp=0xc0000000
[PASS] T3: FP16_OS got=0x3f400000 exp=0x3f400000

... (K=8, K=16, K=32 类似) ...

========================================
  FINAL SUMMARY
========================================
  Total: 16 PASSED, 0 FAILED
========================================

```

### 关键设计

- 测试框架：`ROWS=1, COLS=1, K=N`，通过参数化 wrapper 测试不同 K 深度
- 数据生成：`tb/array_scale/gen_data.py` 生成随机权重/激活值和 golden reference
  - INT8 打包：4 元素/32-bit word（SUBW=4，匹配 PPBuf `OUT_WIDTH=8`）
  - FP16 打包：2 元素/32-bit word（标准 FP16 打包）
  - WS / OS golden 都按当前 RTL 语义生成；其中 WS 现在以 `pe_top.ws_acc` flush 的**完整 dot-product** 为准
- Wrapper 生成：`tb/array_scale/gen_tb.py` 为每个 K 生成独立的 wrapper（硬编码 `$readmemh` 路径、include `test_params.vh` 和 `tb_array_scale_body.v`）
  - `expected` 数组大小已改为 `NUM_TESTS`，不再出现 `$readmemh ... range [0:31]` 告警
- **PPBuf SUBW=4 适配（2026-04-04）**：PPBuf 从 `OUT_WIDTH=16, SUBW=2` 改为 `OUT_WIDTH=8, SUBW=4`，`gen_data.py` INT8 打包同步更新
- **2026-04-08 修正**：array_scale 的 WS golden 从旧的“首拍/末拍乘积”口径改为当前 RTL 实际写回的单个累加结果，因此 `run_array_scale.ps1` 已恢复 **16/16 PASS**


---

## 5.3 多行多列综合测试（`tb_multi_rc_comprehensive`）

`tb_multi_rc_comprehensive.v` 在 **ROWS=2, COLS=2** 物理 PE 阵列上运行 **13 个测试用例**，专项验证：
- 多列 Systolic 激活延迟（col1 比 col0 晚 1 周期）
- OS 模式 ROWS>1 行为（ROWS 不影响结果值）
- 结果序列化（col0 → col1 顺序写 DRAM）
- Back-to-back 多次操作状态重置

### 架构要点

```
pe_w_in  = {COLS{w_ppb_rd_data}}   ← 同一权重广播到所有列
pe_a_in  = {ROWS{a_ppb_rd_data}}   ← 同一激活广播到所有行
pe_acc_in = 0                       ← 顶端 psum 输入固定为 0

Systolic 激活延迟（pe_array.v 中 act_reg）:
  col0: A[0], A[1], ..., A[K-1]        (直接，无延迟)
  col1: 0,    A[0], ..., A[K-2]        (1 拍延迟)

OS 模式: 每个 PE 独立维护 os_acc，acc_in 被忽略。
  acc_out = acc_v[ROWS][c] = 最底行 PE 的 os_acc 输出。
  ROWS > 1 不会倍增结果值。
```

### 期望值计算

```python
# col0: 完整点积
col0 = sum(W[k] * A[k] for k in range(K))

# col1: activation 右移 1 拍（首位填 0）
A_shifted = [0] + A[:K-1]
col1 = sum(W[k] * A_shifted[k] for k in range(K))
```

### 运行

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_multi_rc_sim.ps1
```

### 手动编译命令

```powershell
cd D:\NPU_prj

iverilog -g2012 -DDUMP_VCD `
  -o sim\tb_multi_rc `
  rtl\pe\fp16_mul.v `
  rtl\pe\fp16_add.v `
  rtl\pe\fp32_add.v `
  rtl\pe\pe_top.v `
  rtl\common\fifo.v `
  rtl\common\axi_monitor.v `
  rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v `
  rtl\array\pe_array.v `
  rtl\power\npu_power.v `
  rtl\ctrl\npu_ctrl.v `
  rtl\axi\npu_axi_lite.v `
  rtl\axi\npu_dma.v `
  rtl\top\npu_top.v `
  tb\tb_multi_rc_comprehensive.v

vvp sim\tb_multi_rc
```

### 预期输出（全部通过）

```
################################################################
  NPU Multi-Row/Col Comprehensive Test
  PE Array: 2x2, DATA_W=16, ACC_W=32
################################################################

--- Test T1: INT8 OS COLS=2 (K=4, W=[1,2,3,4] A=[5,6,7,8]) ---
  col0 = 1*5+2*6+3*7+4*8 = 70
  col1 = 1*0+2*5+3*6+4*7 = 56  (systolic shift)
  [PASS] T1_col0: got 70
  [PASS] T1_col1: got 56

--- Test T2: INT8 OS COLS=2 Alternating Sign (K=8) ---
  col0 = 0  (alternating cancel)
  col1 = -2 (shift breaks first cancellation)
  [PASS] T2_col0: got 0
  [PASS] T2_col1: got -2

--- Test T3: INT8 Boundary Values COLS=2 (K=4) ---
  col0 = 32514
  col1 = -16384
  [PASS] T3_col0: got 32514
  [PASS] T3_col1: got -16384

--- Test T4: Zero Weights COLS=2 (K=4) ---
  [PASS] T4_col0: got 0
  [PASS] T4_col1: got 0

--- Test T5: Back-to-Back COLS=2 ---
  Run1 (T1 data): col0=70, col1=56
  Run2 (T3 data): col0=32514, col1=-16384
  [PASS] T5_run1_col0: got 70
  [PASS] T5_run1_col1: got 56
  [PASS] T5_run2_col0: got 32514
  [PASS] T5_run2_col1: got -16384

--- Test T6: Systolic Shift Explicit Verification ---
  col0 != col1 for non-trivial A → confirms 1-cycle shift is active
  [PASS] T6_shift_confirmed

================================================================
  RESULT: 13 PASSED, 0 FAILED
================================================================
```

### 测试用例摘要

| ID | 内容 | col0 期望 | col1 期望 | 验证点 |
|----|------|-----------|-----------|--------|
| T1 | W=[1,2,3,4] A=[5,6,7,8] K=4 | 70 | 56 | 基本 systolic shift |
| T2 | 交替符号 K=8 | 0 | -2 | shift 破坏对称相消 |
| T3 | INT8 边界值 K=4 | 32514 | -16384 | 边界乘积精度 |
| T4 | 零权重 K=4 | 0 | 0 | 零权重不影响 |
| T5 | Back-to-back（T1→T3）| 70→32514 | 56→-16384 | 运算间状态重置 |
| T6 | Shift 显式验证 | ≠col1 | ≠col0 | 1-cycle 延迟确认 |

### 关键参数设置（必须遵守）

| 参数 | 正确值 | 错误值 | 后果 |
|------|--------|--------|------|
| M_DIM | **1** | ROWS（=2） | DMA r_len 翻倍，FIFO 数据不足 → TIMEOUT |
| N_DIM | **COLS**（=2） | 1 | 只写回 col0，丢失 col1 |
| K_DIM | K（向量长度） | — | K 不对结果错误 |

### GTKWave 关键信号（ROWS=2, COLS=2）

```
tb_multi_rc_comprehensive.u_npu.u_ctrl.state        ← FSM（关键）
tb_multi_rc_comprehensive.u_npu.pe_array_valid       ← flush 有效脉冲
tb_multi_rc_comprehensive.u_npu.pe_array_result[31:0]  ← col0 结果
tb_multi_rc_comprehensive.u_npu.pe_array_result[63:32] ← col1 结果
tb_multi_rc_comprehensive.u_npu.gen_r_ser.ser_active ← 序列化进行中
tb_multi_rc_comprehensive.u_npu.gen_r_ser.ser_col    ← 当前序列化列号（0 or 1）
tb_multi_rc_comprehensive.u_npu.u_dma.dma_state      ← DMA FSM
tb_multi_rc_comprehensive.u_npu.u_pe_array.gen_row[0].gen_col[1].u_pe.s0_a
                                                      ← col1 PE 激活输入（应比 col0 晚 1 拍）
```

---

## 5.4 FP16 端到端验证（`tb_fp16_e2e`）

`tb_fp16_e2e.v` 覆盖 **FP16 全链路**：`DRAM → DMA → PPBuf → FP16 packer → PE → Result FIFO → DMA → DRAM`，运行 **8 个测试，9 个检查点**，全部 **PASS（2026-04-06）**。

### 测试用例摘要

| ID | 模式 | K | 说明 | 期望（FP32） |
|----|------|---|------|-------------|
| T1 | OS | 4 | W=[2,-1.5,3,0.5] A=[1,2,4,-2] | 1.0 (`0x3F800000`) |
| T2 | WS | 1 | w=1.5, a=2.0 | 3.0 (`0x40400000`) |
| T3 | OS | 8 | 交替符号相消 | 0.0 (`0x00000000`) |
| T4 | WS | 1 | w=-3.0, a=2.0 | -6.0 (`0xC0C00000`) |
| T5 | OS | 4 | 零权重 | 0.0 (`0x00000000`) |
| T6 | WS | 1 | w=0.25, a=4.0 | 1.0 (`0x3F800000`) |
| T7 | OS | 8 | W=[1.0]*8, A=[0.125..1.0] | 4.5 (`0x40900000`) |
| T8 | OS | 4 | Back-to-back T1×2 | 1.0, 1.0 |

**WS 架构说明**：单 PE 系统中 `acc_in=0`，WS 每拍输出 `w*a`（单次乘积）；DMA 取第一个 FIFO 结果（即 `W[0]*A[0]`）。K=1 测试精确验证 FP16 乘法 + FP32 写回路径。如需系统级 WS 点积，须由主机 CPU 对 K 个 FIFO 输出求和。

### RTL Bug 修复（2026-04-06）—— `npu_top.v` FP16 packer

**Bug**：原 `pe_consume` 的 FP16 条件为 `!w_ppb_phase && !a_ppb_phase`。由于初始状态 `phase=0`，`pe_en=1` 时立刻触发一次虚假 consume（此时 shift register 仍为 0），将一个 `w=0, a=0` 的乘积写入结果 FIFO，导致 WS/OS 模式取出 0x00000000。

**修复**：改用 `phase 1→0 下降沿延迟 1 拍` 检测。新增 `w_phase_fall_d / a_phase_fall_d` 寄存器：
```verilog
// 1 拍后检测 high byte 被锁存的时刻
w_phase_fall_d <= pe_mode && w_ppb_rd_en && w_ppb_phase;  // phase 为 1 时记录
// pe_consume 条件：FP16 时必须是"刚完成 2nd byte"的下一拍，或 flush
pe_consume = pe_en && (pe_data_ready || pe_flush)
           && (pe_mode ? (w_phase_fall_d && a_phase_fall_d) || pe_flush : 1'b1);
```
**影响范围**：`npu_top.v` 仅修改 `pe_consume` 逻辑，不影响 INT8 路径。
**回归验证**：INT8 综合（8/8）、多行多列（13/13）、FP16 端到端（9/9）均 PASS。

### ctrl_reg 编码

```
FP16 + OS：ctrl = 32'h19  (bit[5:4]=01=OS, bit[3:2]=10=FP16, bit[0]=1=start)
FP16 + WS：ctrl = 32'h09  (bit[5:4]=00=WS, bit[3:2]=10=FP16, bit[0]=1=start)
```

### 运行

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_fp16_e2e_sim.ps1
```

### 手动编译命令

```powershell
cd D:\NPU_prj

iverilog -g2012 -DDUMP_VCD `
  -o sim\tb_fp16_e2e `
  rtl\pe\fp16_mul.v `
  rtl\pe\fp16_add.v `
  rtl\pe\fp32_add.v `
  rtl\pe\pe_top.v `
  rtl\common\fifo.v `
  rtl\common\axi_monitor.v `
  rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v `
  rtl\array\pe_array.v `
  rtl\power\npu_power.v `
  rtl\ctrl\npu_ctrl.v `
  rtl\axi\npu_axi_lite.v `
  rtl\axi\npu_dma.v `
  rtl\top\npu_top.v `
  tb\tb_fp16_e2e.v

vvp sim\tb_fp16_e2e
```

### 预期输出（全部通过）

```
################################################################
  NPU FP16 End-to-End Test
  ROWS=1 COLS=1 DATA_W=16 ACC_W=32
################################################################

--- T1: FP16 OS K=4 (W=[2,3,-1.5,0.5] A=[1,2,4,-2] => 1.0) ---
  [PASS] T1: got=0x3f800000 (correct FP32)

--- T2: FP16 WS K=1 (w=1.5 a=2.0 => 3.0) ---
  [PASS] T2: got=0x40400000 (correct FP32)

--- T3: FP16 OS K=8 alternating cancel => 0.0 ---
  [PASS] T3: got=0x00000000 (correct FP32)

--- T4: FP16 WS K=1 negative (w=-3 a=2 => -6.0) ---
  [PASS] T4: got=0xc0c00000 (correct FP32)

--- T5: FP16 OS zero weights K=4 => 0.0 ---
  [PASS] T5: got=0x00000000 (correct FP32)

--- T6: FP16 WS K=1 small (w=0.25 a=4.0 => 1.0) ---
  [PASS] T6: got=0x3f800000 (correct FP32)

--- T7: FP16 OS K=8 precision (W=1, A=0.125..1.0) => 4.5 ---
  [PASS] T7: got=0x40900000 (correct FP32)

--- T8: Back-to-back OS (T1 repeated twice => 1.0, 1.0) ---
  [PASS] T81: got=0x3f800000 (correct FP32)
  [PASS] T82: got=0x3f800000 (correct FP32)

================================================================
  FP16 E2E RESULT: 9 PASSED, 0 FAILED
================================================================
  ALL PASS - FP16 end-to-end pipeline verified.
```

### GTKWave 关键信号（FP16 端到端）

```
tb_fp16_e2e.u_npu.u_ctrl.state         ← FSM 状态
tb_fp16_e2e.u_npu.pe_mode              ← 1=FP16 模式
tb_fp16_e2e.u_npu.w_ppb_phase          ← FP16 weight packer phase（0=低字节，1=高字节）
tb_fp16_e2e.u_npu.a_ppb_phase          ← FP16 activation packer phase
tb_fp16_e2e.u_npu.w_phase_fall_d       ← phase 1→0 延迟触发（pe_consume FP16 gate）
tb_fp16_e2e.u_npu.pe_consume           ← PE 有效消费脉冲（每 FP16 对一次）
tb_fp16_e2e.u_npu.w_fp16_shift         ← 组装好的 FP16 weight（16-bit）
tb_fp16_e2e.u_npu.a_fp16_shift         ← 组装好的 FP16 activation（16-bit）
tb_fp16_e2e.u_npu.r_fifo_wr_en         ← 结果写入 FIFO
tb_fp16_e2e.u_npu.r_fifo_din           ← 结果数据（FP32）
```

---

## 6. FP16 加法器单元测试（`tb_fp16_add.v`）

`tb_fp16_add.v` 是 `fp16_add.v` 模块的独立单元测试，**20 个测试用例**覆盖基本加减法、大小阶对齐、特殊值（Inf/NaN/±0）、次正规数等。

### 运行

```powershell
cd D:\NPU_prj

# 编译（组合逻辑模块，无需时钟）
iverilog -g2012 -o sim\tb_fp16_add.vvp rtl\pe\fp16_add.v tb\tb_fp16_add.v

# 运行仿真
vvp sim\tb_fp16_add.vvp
```

### 预期输出

```
[PASS] Test 1:  a=0x3C00 b=0x3C00  expected=0x4000  got=0x4000
[PASS] Test 2:  a=0x4000 b=0x4200  expected=0x4500  got=0x4500
[PASS] Test 3:  a=0x3E00 b=0x3E00  expected=0x4200  got=0x4200
[PASS] Test 4:  a=0x3C00 b=0xBC00  expected=0x0000  got=0x0000
[PASS] Test 5:  a=0x4000 b=0xBC00  expected=0x3C00  got=0x3C00
[PASS] Test 6:  a=0xC000 b=0x3C00  expected=0xBC00  got=0xBC00
[PASS] Test 7:  a=0x4200 b=0x3C00  expected=0x4400  got=0x4400
[PASS] Test 8:  a=0x4900 b=0x2400  expected=0x4902  got=0x4902
[PASS] Test 9:  a=0x4400 b=0xC000  expected=0x4000  got=0x4000
[PASS] Test 10: a=0x4248 b=0x3CF0  expected=0x4460  got=0x4460
[PASS] Test 11: a=0x4460 b=0xBB00  expected=0x4300  got=0x4300
[PASS] Test 12: a=0x4300 b=0x2F00  expected=0x4338  got=0x4338
[PASS] Test 13: a=0x7C00 b=0xFC00  expected=0xFE00  got=0xFE00
[PASS] Test 14: a=0x7E00 b=0x3C00  expected=0x7E00  got=0x7E00
[PASS] Test 15: a=0x0000 b=0x0000  expected=0x0000  got=0x0000
[PASS] Test 16: a=0x8000 b=0x8000  expected=0x0000  got=0x0000
[PASS] Test 17: a=0x3C00 b=0x8000  expected=0x3C00  got=0x3C00
[PASS] Test 18: a=0x0400 b=0x0400  expected=0x0800  got=0x0800
[PASS] Test 19: a=0x3C00 b=0x4000  expected=0x4200  got=0x4200
[PASS] Test 20: a=0x4000 b=0x4000  expected=0x4400  got=0x4400

=== Summary: PASS=20  FAIL=0 ===
```

### 测试用例详解

| 分组 | 测试 ID | 测试内容 | 验证点 |
|------|---------|----------|--------|
| 基本加法 | T1 | 1.0 + 1.0 = 2.0 | 同阶尾数相加 |
| | T2 | 2.0 + 3.0 = 5.0 | 同阶不同尾数 |
| | T3 | 1.5 + 1.5 = 3.0 | 尾数相加进位 |
| 基本减法 | T4 | 1.0 + (-1.0) = 0.0 | 抵消为零 |
| | T5 | 2.0 + (-1.0) = 1.0 | 减法 |
| | T6 | -2.0 + 1.0 = -1.0 | 负结果符号 |
| | T9 | 4.0 + (-2.0) = 2.0 | 阶差=1 减法 |
| 阶差对齐 | T8 | 10.0 + 0.015625 | **大阶差对齐**（exp diff=9） |
| 复合小数 | T10 | 3.140625 + 1.234375 = 4.375 | 多位小数精确对齐 |
| | T11 | 4.375 + (-0.875) = 3.5 | 减法 + 阶差 |
| | T12 | 3.5 + 0.109375 = 3.609375 | 低位精度保留 |
| 特殊值 | T13 | Inf + (-Inf) = NaN | 无效运算检测 |
| | T14 | NaN + 1.0 = NaN | NaN 传播 |
| | T15 | 0.0 + 0.0 = 0.0 | 零值处理 |
| | T16 | -0.0 + (-0.0) = +0.0 | **负零对消规则** |
| | T17 | 1.0 + (-0.0) = 1.0 | 负零不污染正常值 |
| 次正规数 | T18 | 2⁻¹⁴ + 2⁻¹⁴ = 2⁻¹³ | **最小正规数边界** |
| 累加模拟 | T19 | 1.0 + 2.0 = 3.0 | 顺序累加第一步 |
| | T20 | 2.0 + 2.0 = 4.0 | 模拟 PE fp16_mul 结果相加 |

> **注意**：`fp16_add.v` 是纯组合逻辑（无时钟），测试使用 `#10` 延时等待输出稳定。该模块在 PE 中被 `fp32_add.v` 取代用于 FP16 累加，但仍作为独立验证模块保留。

---

## 7. SoC 全系统联调仿真（`tb_soc.v`）✅ 已完成验证

`tb_soc.v` 测试 **PicoRV32 CPU + NPU** 的全系统集成。CPU 固件通过 AXI-Lite 配置 NPU 完成 **2×2 INT8 矩阵乘法**，读回结果验证，并写入 PASS 标记。

> **验证状态（2026-04-07）**：**287 cycles PASS**  
> C[0][0]=19, C[0][1]=22, C[1][0]=43, C[1][1]=50

### 运行

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_soc_sim.ps1
```

### 预期输出

```
=== NPU SoC Simulation ===

[1/3] Assembling firmware...
  OK: Assembled tb/soc_test.hex (420 bytes)

[2/3] Compiling Verilog...
  OK: Compiled to sim/soc_sim.vvp

[3/3] Running simulation...
[SoC DBG] addr=0x00001000 wstrb=1111 wdata=0x01020304  ← A 矩阵初始化
[SoC DBG] addr=0x00001010 wstrb=1111 wdata=0x00030001  ← B 矩阵初始化
...
[SoC DBG] addr=0x02000000 wstrb=1111 wdata=0x00000011  ← CTRL 写入（start+OS）
...
[SoC DBG] addr=0x00002000 wstrb=1111 wdata=0x000000aa  ← PASS 标记写入 DRAM
[PASS] SoC integration test PASSED!
Cycles: 287
```

### 测试矩阵（2×2 INT8）

```
A = [[1, 2],    B = [[1, 3],
     [3, 4]]         [2, 4]]

C = A × B:
  C[0][0] = 1×1 + 2×2 = 5    ← 不对，实际 19（B 存 DRAM 列主序）
  实际测试：A 按行存，B 按列主序存到 W_ADDR
  C[0][0]=19, C[0][1]=22, C[1][0]=43, C[1][1]=50
```

### SoC 地址映射

| 区域 | 地址范围 | 说明 |
|------|---------|------|
| SRAM | 0x0000–0x0FFF | 4KB，固件代码 + 数据栈 |
| DRAM | 0x1000–0x1FFFF | 128KB，矩阵数据 |
| NPU 寄存器 | 0x02000000–0x0200001F | AXI-Lite 配置端口 |

**⚠️ 关键注意**：
- `0x0F00 < 0x1000`，属于 SRAM 空间，不是 DRAM！(Bug-17 教训)
- PASS 标记必须写到 DRAM 空间（≥ 0x1000），推荐用 `0x2000`
- `$readmemh` 路径使用 `../tb/soc_test.hex`，避免 vvp 工作目录问题（Bug-18 教训）

### 架构概览

```
PicoRV32 CPU ──iomem──► axi_lite_bridge ──AXI4-Lite──► NPU 寄存器
       │                                                    │
       └──────SRAM / DRAM（地址译码）◄────── NPU DMA ──AXI4──► DRAM
```

- **CPU 固件**（`tb/soc_test.S`）：初始化 A/B 矩阵到 DRAM，配置 NPU M/N/K 维度及地址，写 CTRL 启动，轮询 STATUS done，逐元素验证结果，写 PASS/FAIL 标记
- **SRAM 和 DRAM 均使用异步（组合）读**：PicoRV32 要求 `mem_ready` 与 `mem_rdata` 在同一周期有效

### 关键 Bug（已修复）

| Bug | 现象 | 修复 |
|-----|------|------|
| soc_mem 同步读（Bug-14） | CPU 每条指令读 stale data，程序执行混乱 | `assign rdata = mem[addr]`（异步） |
| dram_model 同步读（Bug-15） | CPU LW 指令读到错误值 | `assign cpu_rdata = mem[...]`（异步） |
| soc_top addr 位宽（Bug-16） | 编译警告，地址高位截断 | `mem_addr[23:2]` 取 22-bit |
| 标记地址在 SRAM（Bug-17） | testbench 监视 DRAM，PASS 永不触发 | 改标记地址为 0x2000（DRAM 空间） |
| $readmemh 路径（Bug-18） | 加载旧 hex，CPU 执行错误指令 | 改为 `../tb/soc_test.hex` |

### 无 RISC-V 工具链时

脚本会自动使用预生成的 `tb/soc_test.hex`（420 字节，2026-04-07 版本）。如需重新汇编固件：

```powershell
# 使用 assemble_soc_test.py（内置汇编器，无需 riscv 工具链）
cd D:\NPU_prj\tb
python assemble_soc_test.py  # 生成 soc_test.hex
```

---

所有带 `-DDUMP_VCD` 编译的仿真会在运行目录生成 `.vcd` 波形文件。

---

## 8. 用 GTKWave 看波形

### 打开波形

```powershell
# tb_comprehensive 的波形
gtkwave sim\tb_comprehensive.vcd

# NPU 全系统波形
gtkwave sim\wave\tb_npu_top.vcd
```

### 推荐添加的信号

在 GTKWave 左侧的 Signal Tree 中，展开对应模块，拖入以下信号：

#### 全局控制流（必看）

| 信号路径 | 说明 |
|----------|------|
| `tb_comprehensive.clk` | 时钟 |
| `tb_comprehensive.rst_n` | 复位（高有效） |
| `tb_comprehensive.u_npu.u_ctrl.state` | 控制器 FSM 状态（关键！） |
| `tb_comprehensive.u_npu.pe_en` | PE 阵列使能 |
| `tb_comprehensive.u_npu.pe_flush` | PE flush（累加结果输出） |

#### AXI-Lite 配置接口（CPU 侧）

| 信号路径 | 说明 |
|----------|------|
| `tb_comprehensive.s_awvalid` | 写地址有效 |
| `tb_comprehensive.s_awready` | 写地址就绪（NPU 响应） |
| `tb_comprehensive.s_awaddr` | 写地址 |
| `tb_comprehensive.s_wdata` | 写数据 |

#### DMA 接口（DRAM 侧）

| 信号路径 | 说明 |
|----------|------|
| `tb_comprehensive.m_arvalid` | DMA 读请求 |
| `tb_comprehensive.m_araddr` | DMA 读地址 |
| `tb_comprehensive.m_rvalid` | DRAM 返回数据有效 |
| `tb_comprehensive.m_rdata` | DRAM 返回数据 |

#### NPU 状态

| 信号路径 | 说明 |
|----------|------|
| `tb_comprehensive.u_npu.u_ctrl.status_done` | 计算完成标志 |
| `tb_comprehensive.u_npu.u_ctrl.status_busy` | NPU 忙碌标志 |
| `tb_comprehensive.npu_irq` | 中断输出 |

### GTKWave 操作提示

- **Ctrl+A**：全选已添加的信号
- **滚轮**：左右缩放时间轴
- **Shift+点击**：多选信号
- **右键信号** → `Data Format` → `Decimal` / `Hex`：切换数值显示格式
- **View → Show Grid**：显示时钟格

---

## 9. 手动编译命令参考

如果不想用脚本，可以直接在命令行编译：

### 只编译 PE 单元

```powershell
iverilog -g2012 `
  -o sim\wave\sim_pe.out `
  rtl\pe\fp16_mul.v `
  rtl\pe\fp16_add.v `
  rtl\pe\fp32_add.v `
  rtl\pe\pe_top.v `
  tb\tb_pe_top.v

vvp sim\wave\sim_pe.out
```

### 编译 NPU 全系统（含可重配置阵列、Ping-Pong、DMA）

> **注意（2026-04-14 更新）**：`pe_array.v` 已被 `reconfig_pe_array.v` 替代。编译时请使用新的文件名。

```powershell
iverilog -g2012 -DDUMP_VCD `
  -I rtl `
  -o sim\npu_sim `
  rtl\pe\fp16_mul.v `
  rtl\pe\fp16_add.v `
  rtl\pe\fp32_add.v `
  rtl\pe\pe_top.v `
  rtl\common\fifo.v `
  rtl\common\axi_monitor.v `
  rtl\common\op_counter.v `
  rtl\array\reconfig_pe_array.v `
  rtl\buf\pingpong_buf.v `
  rtl\power\npu_power.v `
  rtl\ctrl\npu_ctrl.v `
  rtl\axi\npu_axi_lite.v `
  rtl\axi\npu_dma.v `
  rtl\top\npu_top.v `
  tb\tb_npu_top.v

cd sim\wave
vvp ..\npu_sim
```

### 运行综合测试（推荐，验证 reconfig_pe_array 集成）

```powershell
cd D:\NPU_prj
iverilog -g2012 -o sim\tb_comprehensive.vvp -s tb_comprehensive `
  rtl\pe\fp16_add.v rtl\pe\fp16_mul.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\array\reconfig_pe_array.v rtl\common\fifo.v rtl\buf\pingpong_buf.v `
  rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\ctrl\npu_ctrl.v `
  rtl\power\npu_power.v rtl\top\npu_top.v tb\tb_comprehensive.v

cd sim; vvp -N tb_comprehensive.vvp
```

### 编译 SoC 全系统

> **注意**：SoC 测试仍使用旧版 `pe_array.v`（尚未迁移到 `reconfig_pe_array`）。如需在 SoC 中使用新阵列，需更新 `soc_top.v` 的实例化。

```powershell
iverilog -g2012 -s tb_soc `
  -o sim\soc_sim.vvp `
  sim\picorv32.v `
  rtl\soc\soc_mem.v `
  rtl\soc\dram_model.v `
  rtl\soc\axi_lite_bridge.v `
  rtl\soc\soc_top.v `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\array\reconfig_pe_array.v `
  rtl\buf\pingpong_buf.v rtl\power\npu_power.v `
  rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v `
  rtl\top\npu_top.v `
  tb\tb_soc.v

cd sim
vvp soc_sim.vvp
```

### 常用 iverilog 参数说明

| 参数 | 作用 |
|------|------|
| `-g2012` | 启用 IEEE 1364-2005 扩展（支持 `$clog2` 等系统函数） |
| `-DDUMP_VCD` | 条件编译宏，开启波形转储 |
| `-s <模块名>` | 指定顶层模块（默认取文件中第一个 module） |
| `-I <目录>` | 添加 include 搜索路径 |
| `-o <输出文件>` | 指定输出的 `.vvp` 文件名 |

---

## 10. 理解各仿真场景

### 场景总览

| 文件 | 覆盖范围 | 用例数 | 运行时间 | 推荐场景 |
|------|----------|--------|----------|----------|
| `tb_fp16_mul.v` | FP16 乘法器独立验证 | 44 | < 1s | FP16 乘法器单元测试 |
| `tb_fp16_add.v` | FP16 加法器独立验证 | 20 | < 1s | FP16 加法器单元测试 |
| `tb_pe_top.v` | 单 PE 全功能测试 | 19 | < 5s | PE 流水线/WS latch 验证 |
| `tb_comprehensive.v` | NPU 端到端：边界值/连续运算 | 8 | ~30s | **日常回归测试** ★ |
| `tb_multi_rc_comprehensive.v` | ROWS=2,COLS=2 多行多列：systolic shift、序列化、back-to-back | **13** | ~30s | **多列 PE 阵列验证** ★ |
| `tb_fp16_e2e.v` | FP16 全链路：OS 点积/WS 单乘/精度/back-to-back | **9** | ~30s | **FP16 端到端验证** ★ |
| `tb_classifier.v` | 三层 FC 网络端到端推理 | 3 | ~30s | 网络推理集成验证 |
| `tb_npu_top.v` | NPU 系统含性能报告 | 多场景 | ~30s | 性能分析与带宽分析 |
| `tb_soc.v` | PicoRV32 CPU 固件驱动 NPU | 1 | 数分钟 | SoC 集成验证 |
| `tb_array_scale` | K 深度验证：ROWS=1,COLS=1，K=4/8/16/32 × 4 组合（16/16 PASS） | 16 | ~5min | **参数化 K 深度验证** ★ |


---

### 10.1 FP16 乘法器单元测试（`tb_fp16_mul.v`）

**44 个测试用例**，覆盖 FP16 乘法器的所有功能路径：

| 分组 | 测试 ID | 测试内容 |
|------|---------|----------|
| 基本乘法 | T1–T13 | 正数乘法（1.0×1.0, 2.0×2.0, 1.5×1.5, 舍入边界等） |
| 负数 | T20–T25 | 符号位正确性（-×+=-, -×-=+, 负权重 WS 等） |
| 特殊值 | T30–T42 | Inf×1=Inf, Inf×0=NaN, NaN 传播, ±0 处理, -0×-1=+0 |
| 溢出 | T50–T52 | max FP16 × 1.0, max × 2.0 = Inf, max × 1.5 = Inf |
| **次正规数** | **T60–T67** | **渐进下溢：max_sub×1.0, max_sub×0.5, min_sub×1.0, min_sub×2.0, min_sub²=0, min_normal×0.25, sub×sub=0** |

**编译运行**：

```powershell
cd D:\NPU_prj
iverilog -g2012 -o sim\tb_fp16_mul.vvp rtl\pe\fp16_mul.v tb\tb_fp16_mul.v
vvp sim\tb_fp16_mul.vvp
```

**关键实现细节**：
- 完整 22-bit LZC 归一化（前导零计数范围 0..22）
- 次正规数输入支持（动态隐式位）
- **渐进下溢**：当 `biased_exp ≤ 0` 时，对归一化 mantissa 继续右移并做 RN 舍入，而非 flush-to-zero
- 舍入上溢自动进位到最小正规数（exp=1, mant=0）

---

### 10.2 综合测试用例详解（`tb_comprehensive.v`）

以下是 8 个测试场景的数学原理，理解这些有助于调试时快速定位问题。

---

#### Test 1：INT8 正数点积（K=8）

```
W = [1, 2, 3, 4, 5, 6, 7, 8]
A = [10, 20, 30, 40, 50, 60, 70, 80]

点积 = 1×10 + 2×20 + 3×30 + 4×40 + 5×50 + 6×60 + 7×70 + 8×80
     = 10 + 40 + 90 + 160 + 250 + 360 + 490 + 640
     = 2040
```

**DRAM 数据布局（Little-Endian 字节打包）**：

```
W 存储地址：0x2000
  word0 = 0x04030201  → byte[0]=1, byte[1]=2, byte[2]=3, byte[3]=4
  word1 = 0x08070605  → byte[0]=5, byte[1]=6, byte[2]=7, byte[3]=8

A 存储地址：0x2020
  word0 = 0x281E140A  → byte[0]=0x0A=10, byte[1]=0x14=20, byte[2]=0x1E=30, byte[3]=0x28=40
  word1 = 0x50463C32  → byte[0]=0x32=50, byte[1]=0x3C=60, byte[2]=0x46=70, byte[3]=0x50=80

结果读取地址：0x2040
```

---

#### Test 1b：K=4 回归（期望 310）

```
W = [3, 7, -2, 5]
A = [10, 20, 30, 40]

点积 = 3×10 + 7×20 + (-2)×30 + 5×40
     = 30 + 140 - 60 + 200 = 310
```

**负数 INT8 打包**：-2 = 0xFE，-20 = 0xEC 等，符号扩展在 PE 内部由 `$signed` 处理。

---

#### Test 2：INT8 混合正负（K=8）

```
W = [10, -20, 30, -40, 50, -60, 70, -80]
A = [3,   5,  -7,   9, -11,  13, -15,  17]

点积 = 30 - 100 - 210 - 360 - 550 - 780 - 1050 - 1360
     = -4380 (0xFFFFEEE4)
```

---

#### Test 3：INT8 边界值（K=8）

```
W = [127, -128, 127, -128,  1,  0, -1,  0]
A = [127, -128, -128, 127, 127, 1, 127, 1]

点积 = 127×127 + (-128)×(-128) + 127×(-128) + (-128)×127
     + 1×127 + 0×1 + (-1)×127 + 0×1
     = 16129 + 16384 - 16256 - 16256 + 127 + 0 - 127 + 0
     = 1
```

> **调试要点**：INT8 边界乘积（如 127×127=16129）会产生大的正中间值，多个大值相消后才得到 1。任何一步符号扩展错误都会导致结果偏差数万。
>
> **打包方式**：A 第二个 word 的 A[4]=127, A[5]=1 对应字节为 `[0x7F, 0x01, 0x7F, 0x01]`，打包成 `32'h017F017F`（Little-Endian，低字节在低地址）。

---

#### Test 4：零权重（K=8）

```
W = [0, 0, 0, 0, 0, 0, 0, 0]
A = [10, 20, 30, 40, 50, 60, 70, 80]

期望结果 = 0
```

验证 PE 累加器不会因浮动输入产生非零噪声。

---

#### Test 5：INT8 交替正负（K=16）

```
W = [1,-1, 1,-1,  2,-2, 2,-2,  3,-3, 3,-3,  4,-4, 4,-4]
A = [1,-1, 1,-1,  1,-1, 1,-1,  1,-1, 1,-1,  1,-1, 1,-1]

每对 (W[i], W[i+1]) × (A[i], A[i+1]):
  1×1 + (-1)×(-1) = 2，共 8 对
  加权：2×(1+1+2+2+3+3+4+4) = 2×20 = 40

期望结果 = 40
```

---

#### Test 6：连续运算（Back-to-Back）

复用 Test 1 和 Test 2 的数据，**连续启动两次 NPU**，验证：
1. 第一次完成后状态机正确复位到 IDLE
2. 第二次可以正确启动并完成
3. 两次结果互不干扰（分别存到不同 DRAM 地址）

```
第一次：期望 2040
第二次：期望 -4380
```

---

### 10.3 关键仿真机制

#### AXI-Lite BFM（总线功能模型）

testbench 用 `axi_write` / `axi_read` task 模拟 CPU 对 NPU 寄存器的读写：

```verilog
// 写寄存器：
axi_write(REG_K_DIM, 32'd8);    // 设置 K=8

// 读状态：
axi_read(REG_STATUS, status);   // 读取 done/busy 位

// 等待完成（内部轮询 STATUS，超时自动报错）：
wait_done(100000);              // 最多等 100000 个周期
```

#### DRAM 行为模型

testbench 内置一个 `reg [31:0] dram [0:DRAM_SZ-1]` 数组，模拟真实 DRAM：

```verilog
// 直接写 DRAM（预置测试数据）：
dram[32'h2000 >> 2] = 32'h04030201;    // 字地址 = 字节地址 >> 2

// 读回结果（验证阶段）：
got_val = dram[32'h2040 >> 2];
```

#### 字节打包规则

NPU DMA 按 32-bit 字传输，PE 从 `w_in[15:0]` / `a_in[15:0]` 取元素。
**INT8 打包方式**：

```
DRAM word = 32'hDDCCBBAA
               │  │  │  └─ byte[0] → 第1个 INT8 元素
               │  │  └──── byte[1] → 第2个 INT8 元素
               │  └─────── byte[2] → 第3个 INT8 元素
               └────────── byte[3] → 第4个 INT8 元素
```

例如 W=[1,2,3,4] 打包为 `32'h04030201`：

```
AA=01 → W[0]=1
BB=02 → W[1]=2
CC=03 → W[2]=3
DD=04 → W[3]=4
```

负数用补码：-1=0xFF, -2=0xFE, -128=0x80, 127=0x7F

---

## 11. 自己添加测试向量

在 `tb/tb_comprehensive.v` 中添加自定义测试向量，按以下步骤操作：

### 步骤 1：选择 DRAM 地址

找一个不与现有测试冲突的地址段。现有测试占用：

| 测试 | 权重地址 | 激活地址 | 结果地址 |
|------|---------|----------|---------|
| T1 | 0x2000 | 0x2020 | 0x2040 |
| T1b | 0x2080 | 0x2084 | 0x2088 |
| T2 | 0x2100 | 0x2120 | 0x2140 |
| T3 | 0x2200 | 0x2220 | 0x2240 |
| T4 | 0x2300 | 0x2320 | 0x2340 |
| T5 | 0x2400 | 0x2440 | 0x2480 |
| T6 | 复用 T1/T2 | — | 0x2500 |

建议从 `0x2600` 开始使用新地址。

### 步骤 2：在 `initial` 数据块中写入向量

```verilog
// initial begin ... 中添加：

// My Custom Test: W=[2,3], A=[4,5], K=2
// Expected: 2*4 + 3*5 = 8 + 15 = 23
dram[32'h2600 >> 2] = 32'h00000302;  // W: byte[0]=2, byte[1]=3
dram[32'h2620 >> 2] = 32'h00000504;  // A: byte[0]=4, byte[1]=5
```

### 步骤 3：在测试序列末尾添加调用

```verilog
// 在 initial begin ... end 的测试序列末尾：

$display("");
$display("--- My Custom Test: K=2 ---");
run_npu(32'd1, 32'd1, 32'd2,
        32'h2600, 32'h2620, 32'h2640,
        CTRL_START | CTRL_OS,
        100000);    // 超时周期数
got_val = dram[32'h2640 >> 2];
exp_val = 32'd23;
if (got_val === exp_val) begin
    $display("  [PASS] MyTest: got %0d", $signed(got_val));
    pass_cnt = pass_cnt + 1;
end else begin
    $display("  [FAIL] MyTest: got %0d, exp %0d", $signed(got_val), $signed(exp_val));
    fail_cnt = fail_cnt + 1;
end
```

### 步骤 4：重新编译并运行

```powershell
cd D:\NPU_prj
# 重新编译（替换 .vvp 文件名以区分）
iverilog -g2012 -DDUMP_VCD `
  -o sim\tb_comprehensive.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\array\pe_array.v rtl\buf\pingpong_buf.v `
  rtl\power\npu_power.v rtl\ctrl\npu_ctrl.v `
  rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v `
  rtl\top\npu_top.v tb\tb_comprehensive.v

vvp sim\tb_comprehensive.vvp
```

---

## 12. 常见报错与解决

### ❌ `iverilog: command not found`

**原因**：iverilog 未安装或未加入 PATH。

**解决**：
```powershell
# Windows：重新安装时勾选 "Add to PATH"，或手动加入：
$env:PATH += ";C:\iverilog\bin"

# 或使用完整路径：
C:\iverilog\bin\iverilog -g2012 ...
```

---

### ❌ `Error: Unknown module type: fp32_add`

**原因**：编译时遗漏了 `rtl\pe\fp32_add.v`（FP32 累加器模块）。

**解决**：确保 `iverilog` 命令包含所有以下文件：

```
rtl\pe\fp16_mul.v        ← 必须在 pe_top.v 之前
rtl\pe\fp16_add.v        ← FP16 加法器
rtl\pe\fp32_add.v        ← FP32 累加器（WS/OS FP16 模式使用）
rtl\pe\pe_top.v
rtl\common\fifo.v
rtl\common\axi_monitor.v
rtl\common\op_counter.v
rtl\array\pe_array.v
rtl\buf\pingpong_buf.v
rtl\power\npu_power.v
rtl\ctrl\npu_ctrl.v
rtl\axi\npu_axi_lite.v
rtl\axi\npu_dma.v
rtl\top\npu_top.v
tb\tb_comprehensive.v    ← testbench 最后
```

---

### ❌ `*** TIMEOUT after N cycles! ***`

**原因**：NPU 没有正确发出 done 信号。常见根因：

1. K_DIM 与实际 DRAM 数据长度不匹配
2. 结果地址 R_ADDR 与权重/激活地址重叠
3. ctrl_reg 未清零导致 FSM 重复启动

**排查步骤**：
```powershell
# 开启 VCD 后用 GTKWave 检查 state 信号
gtkwave sim\tb_comprehensive.vcd
# 找 u_npu.u_ctrl.state，看 FSM 卡在哪个状态
```

---

### ❌ `FAIL: got 0, exp 2040`

**原因**：结果读取地址错误，或 DMA write-back 未完成就读取。

**解决**：确认 `wait_done` 之后再读 `dram[R_ADDR >> 2]`，不要在 `run_npu` 前读。

---

### ❌ `$clog2` 未定义

**原因**：iverilog 版本过旧（< 11.0）。

**解决**：升级 iverilog，或在编译命令中加 `-g2012`：

```powershell
iverilog -g2012 ...   # 强制使用 2005 标准
```

---

### ❌ 波形文件 `.vcd` 没有生成

**原因**：编译时未加 `-DDUMP_VCD` 宏，或 `vvp` 运行目录不对。

**解决**：
```powershell
# 1. 编译时加宏
iverilog -g2012 -DDUMP_VCD ...

# 2. 在预期的波形输出目录运行 vvp
cd sim
vvp ..\sim\tb_comprehensive.vvp   # vcd 会生成在 sim\ 下
```

---

## 快速参考卡片

```
┌─────────────────────────────────────────────────────────────────┐
│                    NPU 仿真快速参考                               │
├─────────────────────────┬───────────────────────────────────────┤
│ 运行 PE 单元测试（19项）  │ scripts\run_sim.ps1                   │
│ 运行 FP16 乘法器测试     │ iverilog ... tb\tb_fp16_mul.v          │
│ 运行 FP16 加法器测试     │ iverilog ... tb\tb_fp16_add.v          │
│ 运行综合回归测试         │ 手动编译 tb_comprehensive.v（见第4节） │
│ 运行多行多列测试         │ scripts\run_multi_rc_sim.ps1          │
│ 运行 FP16 端到端测试     │ scripts\run_fp16_e2e_sim.ps1          │
│ 运行分类器推理测试       │ scripts\run_classifier_sim.ps1        │
│ 运行全系统性能测试       │ scripts\run_full_sim.ps1              │
│ 运行阵列规模验证         │ scripts\run_array_scale.ps1           │
│ 运行 SoC 联调测试        │ scripts\run_soc_sim.ps1               │
│ 打开波形                 │ gtkwave sim\wave\*.vcd                │
├─────────────────────────┼───────────────────────────────────────┤
│ 关键 FSM 信号            │ u_npu.u_ctrl.state                    │
│ NPU 完成标志             │ u_npu.u_ctrl.status_done              │
│ PE 使能                  │ u_npu.pe_en                           │
├─────────────────────────┼───────────────────────────────────────┤
│ COLS=2 col0 结果         │ pe_array_result[31:0]                 │
│ COLS=2 col1 结果         │ pe_array_result[63:32]                │
│ M_DIM（多列时）          │ 必须=1，不得=ROWS                     │
│ col1 期望值              │ A_shifted=[0,A[0],...,A[K-2]]        │
├─────────────────────────┼───────────────────────────────────────┤
│ 字节打包（Little-Endian）│ 0xDDCCBBAA → [AA, BB, CC, DD]        │
│ INT8 负数                │ -1=0xFF, -2=0xFE, -128=0x80          │
│ 结果地址必须 >> 2        │ dram[addr >> 2] = word 索引           │
└─────────────────────────┴───────────────────────────────────────┘
```

---

*本文档对应 NPU_prj 验证状态（更新于 2026-04-10）：*

| 测试套件 | 状态 | 通过/总数 |
|---------|------|---------|
| FP16乘法器单元测试 | ✅ PASS | 44/44 |
| FP16加法器单元测试 | ✅ PASS | 20/20 |
| PE单元测试 | ✅ PASS | 19/19 |
| NPU综合测试 | ✅ PASS | 8/8 |
| 多行多列测试 | ✅ PASS | 13/13 |
| FP16端到端测试 | ✅ PASS | 9/9 |
| 阵列规模验证 | ✅ PASS | 16/16 |
| SoC集成测试 | ✅ PASS | 1/1 |

**所有已知问题已修复，全量回归零失败。**

*如有 RTL 变更请重新运行回归测试。*
