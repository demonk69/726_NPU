# NPU_prj 仿真快速上手教程

> 适合对象：刚拿到源码、想立刻跑起仿真的工程师
>
> 最后更新：2026-04-03

---

## 目录

1. [工具安装（5 分钟）](#1-工具安装5-分钟)
2. [克隆与目录一览](#2-克隆与目录一览)
3. [第一次仿真：PE 单元测试](#3-第一次仿真pe-单元测试)
4. [第二次仿真：NPU 综合测试（推荐）](#4-第二次仿真npu-综合测试推荐)
5. [第三次仿真：NPU 全系统性能测试](#5-第三次仿真npu-全系统性能测试)
6. [用 GTKWave 看波形](#6-用-gtkwave-看波形)
7. [手动编译命令参考](#7-手动编译命令参考)
8. [理解各仿真场景](#8-理解各仿真场景)
9. [自己添加测试向量](#9-自己添加测试向量)
10. [常见报错与解决](#10-常见报错与解决)

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
│   ├── pe/              ← PE 单元 & FP16 乘法器
│   ├── array/           ← 脉动阵列
│   ├── axi/             ← AXI-Lite 从机 & DMA
│   ├── ctrl/            ← 控制器 FSM
│   ├── buf/             ← Ping-Pong 缓冲区
│   ├── top/             ← NPU 顶层
│   └── common/          ← FIFO、监控、计数器
├── tb/                  ← Testbench（所有仿真入口）
│   ├── tb_pe_top.v          ← 场景A: 单 PE 功能测试
│   ├── tb_comprehensive.v   ← 场景B: NPU 综合用例（推荐入口）★
│   ├── tb_npu_top.v         ← 场景C: 带性能报告的系统测试
│   └── tb_soc.v             ← 场景D: SoC CPU+NPU 联调
├── scripts/
│   ├── run_sim.ps1          ← 运行场景A
│   ├── run_full_sim.ps1     ← 运行场景C（含性能报告）
│   └── run_soc_sim.ps1      ← 运行场景D
└── sim/wave/            ← VCD 波形输出目录（自动创建）
```

---

## 3. 第一次仿真：PE 单元测试

这是最快的验证。测试**单个 PE（Processing Element）** 的乘加功能，4 个场景，约 10 秒完成。

### 运行

```powershell
# 在项目根目录执行
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

### 预期输出

```
[INFO] Compiling...
[INFO] Running simulation...
--- Test 1: INT8 Weight-Stationary ---
  weight=3, activations=1,2,3,4,0,0,0,0
  [PASS] Test 1: got=30 exp=30
--- Test 2: INT8 Output-Stationary ---
  weights=1,2,3,4, activation=2
  [PASS] Test 2: got=20 exp=20
--- Test 3: FP16 Weight-Stationary ---
  [INFO] FP16 WS result = 0x00003C00 (expected 0x00003C00 for 1.0 * 1.0)
--- Test 4: FP16 Output-Stationary ---
  [INFO] FP16 OS result = 0x00003C00
=== Summary: PASS=2  FAIL=0 ===
ALL TESTS PASSED
[INFO] Done. VCD: ...\sim\wave\tb_pe_top.vcd
```

### 这个测试验证了什么？

| 场景 | 权重/激活 | 期望累加值 | 验证点 |
|------|-----------|-----------|--------|
| INT8 WS | w=3, a=[1,2,3,4] | 3×(1+2+3+4)=30 | Weight-Stationary 累加 |
| INT8 OS | w=[1,2,3,4], a=2 | (1+2+3+4)×2=20 | Output-Stationary 权重流动 |
| FP16 WS | w=1.0, a=1.0 | 0x3C00 (+1.0) | FP16 IEEE 754 乘法器 |
| FP16 OS | w=1.0, a=2.0 | 0x4000 (+2.0) | FP16 OS 烟雾测试 |

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
| `PE Utilization` | PE 在使能周期的有效使用率 | ~6.25% | 100% |
| `MACs/Cycle` | 每周期 MAC 吞吐 | 4.0 | 16.0 (4×4 阵列) |
| `Efficiency` | 计算时间 / 总忙时间 | ~5.56% | 接近 100% |

> **注意**：当前测试用例为 4×4 小矩阵，DMA 传输开销占主导，PE 利用率偏低是正常的。大矩阵下效率会显著提升。

---

## 6. 用 GTKWave 看波形

所有带 `-DDUMP_VCD` 编译的仿真会在运行目录生成 `.vcd` 波形文件。

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

## 7. 手动编译命令参考

如果不想用脚本，可以直接在命令行编译：

### 只编译 PE 单元

```powershell
iverilog -g2012 `
  -o sim\wave\sim_pe.out `
  rtl\pe\fp16_mul.v `
  rtl\pe\pe_top.v `
  tb\tb_pe_top.v

vvp sim\wave\sim_pe.out
```

### 编译 NPU 全系统（含 Ping-Pong、DMA、监控）

```powershell
iverilog -g2012 -DDUMP_VCD `
  -I rtl `
  -o sim\npu_sim `
  rtl\pe\fp16_mul.v `
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
  tb\tb_npu_top.v

cd sim\wave
vvp ..\npu_sim
```

### 常用 iverilog 参数说明

| 参数 | 作用 |
|------|------|
| `-g2012` | 启用 IEEE 1364-2005 扩展（支持 `$clog2` 等系统函数） |
| `-DDUMP_VCD` | 条件编译宏，开启波形转储 |
| `-I <目录>` | 添加 include 搜索路径 |
| `-o <输出文件>` | 指定输出的 `.vvp` 文件名 |

---

## 8. 理解各仿真场景

### 场景总览

| 文件 | 覆盖范围 | 运行时间 | 推荐场景 |
|------|----------|----------|----------|
| `tb_fp16_mul.v` | FP16 乘法器单元测试：44 个测试用例 | < 1s | FP16 乘法器独立验证 |
| `tb_fp16_add.v` | FP16 加法器单元测试 | < 1s | FP16 加法器独立验证 |
| `tb_pe_top.v` | 单 PE：INT8/FP16 × WS/OS | < 5s | 初次验证 PE 功能 |
| `tb_comprehensive.v` | NPU 端到端：8 个测试用例 | ~30s | **日常回归测试** ★ |
| `tb_npu_top.v` | NPU 系统：含性能报告 | ~30s | 性能分析与带宽分析 |
| `tb_soc.v` | SoC 全系统：CPU固件驱动NPU | 数分钟 | SoC 集成验证 |

---

### 8.1 FP16 乘法器单元测试（`tb_fp16_mul.v`）

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

### 8.2 综合测试用例详解（`tb_comprehensive.v`）

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

#### Test 3：INT8 边界值（K=8）——最严格的精度测试

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

### 8.2 关键仿真机制

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

#### 字节打包规则（重要！）

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

## 9. 自己添加测试向量

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
  rtl\pe\fp16_mul.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\array\pe_array.v rtl\buf\pingpong_buf.v `
  rtl\power\npu_power.v rtl\ctrl\npu_ctrl.v `
  rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v `
  rtl\top\npu_top.v tb\tb_comprehensive.v

vvp sim\tb_comprehensive.vvp
```

---

## 10. 常见报错与解决

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

### ❌ `Error: Unknown module type: npu_top`

**原因**：编译时遗漏了某个 RTL 源文件。

**解决**：确保 `iverilog` 命令包含所有以下文件：

```
rtl\pe\fp16_mul.v        ← 必须在 pe_top.v 之前
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
│ 运行 PE 单元测试         │ scripts\run_sim.ps1                   │
│ 运行综合回归测试         │ 手动编译 tb_comprehensive.v（见第4节） │
│ 运行全系统性能测试       │ scripts\run_full_sim.ps1              │
│ 打开波形                 │ gtkwave sim\wave\*.vcd                │
├─────────────────────────┼───────────────────────────────────────┤
│ 关键 FSM 信号            │ u_npu.u_ctrl.state                    │
│ NPU 完成标志             │ u_npu.u_ctrl.status_done              │
│ PE 使能                  │ u_npu.pe_en                           │
├─────────────────────────┼───────────────────────────────────────┤
│ 字节打包（Little-Endian）│ 0xDDCCBBAA → [AA, BB, CC, DD]        │
│ INT8 负数                │ -1=0xFF, -2=0xFE, -128=0x80          │
│ 结果地址必须 >> 2        │ dram[addr >> 2] = word 索引           │
└─────────────────────────┴───────────────────────────────────────┘
```

---

*本文档对应 NPU_prj 综合仿真通过状态（8/8 PASS），如有 RTL 变更请重新运行仿真验证。*
