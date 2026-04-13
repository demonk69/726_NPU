# NPU_prj 用户手册

> 最后更新：2026-04-13
> 状态：已按真 Ping-Pong 重叠 FSM（2026-04-13）、全量回归结果与端到端调度场景同步

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境准备](#2-环境准备)
3. [快速开始](#3-快速开始)
4. [寄存器编程指南](#4-寄存器编程指南)
5. [数据布局与地址约定](#5-数据布局与地址约定)
6. [端到端推理调度详解：三层卷积+激活](#6-端到端推理调度详解三层卷积激活)
7. [SoC 集成使用说明](#7-soc-集成使用说明)
8. [常用仿真入口](#8-常用仿真入口)
9. [常见问题](#9-常见问题)
10. [配套文档索引](#10-配套文档索引)

---

## 1. 项目简介

NPU_prj 是一个 Verilog NPU 原型工程，包含：

- 4×4 可参数化 PE 阵列
- INT8 / FP16 数据通路
- WS / OS 两种计算模式
- AXI4-Lite 配置接口
- AXI4 DMA 数据搬运
- PicoRV32 + SRAM + DRAM + NPU 的 SoC 集成

### 当前可依赖的事实

| 项目 | 当前状态 |
|---|---|
| INT8 | ✅ 可用 |
| FP16 | ✅ 可用 |
| INT16 | ⚠️ 不建议依赖，当前主通路未单独实现 |
| WS 模式 | ✅ 已验证 |
| OS 模式 | ✅ 已验证 |
| SoC 固件驱动 | ✅ 已验证 |
| 全量回归 | ✅ 903 PASS / 0 FAIL |

### 当前最重要的理解方式

这个工程现在不是"整块 4×4 输出并行一次算完"的经典说明书模型，而是：

- 控制器按 `C[i][j]` **逐个输出元素**推进
- 每个 tile 读取一列 `B[:,j]` 和一行 `A[i,:]`
- 然后把这个 tile 的 32-bit 结果写回 `R_ADDR + (i*N + j)*4`

---

## 2. 环境准备

### 2.1 基本依赖

| 工具 | 用途 |
|---|---|
| Icarus Verilog | 编译与运行 RTL 仿真 |
| GTKWave | 查看波形 |
| Python | 运行部分辅助脚本（如 `assemble_soc_test.py`） |
| PowerShell | Windows 下运行仿真脚本 |

### 2.2 推荐验证

```powershell
iverilog -V
gtkwave --version
python --version
```

---

## 3. 快速开始

## 3.1 跑最小单元测试

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

用途：验证单个 PE 的 INT8 / FP16 基本功能。

## 3.2 跑 NPU 综合测试

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1
```

用途：验证 NPU 主数据通路、DMA、控制器、阵列和统计模块。

## 3.3 跑 SoC 集成测试

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_soc_sim.ps1
```

当前已验证结果：

```text
[PASS] SoC integration test PASSED!
Cycles: 287
```

## 3.4 跑全量回归

```powershell
cd D:\NPU_prj
powershell -ExecutionPolicy Bypass -File scripts\run_regression.ps1
```

当前基线结果：

```text
903 PASS, 0 FAIL
```

---

## 4. 寄存器编程指南

NPU 配置基地址：**`0x0200_0000`**

## 4.1 寄存器表

| 偏移 | 名称 | R/W | 说明 |
|---:|---|:---:|---|
| 0x00 | `CTRL` | RW | `[0] start`, `[1] abort`, `[3:2] mode`, `[5:4] stat_mode` |
| 0x04 | `STATUS` | RO | `[0] busy`, `[1] done` |
| 0x08 | `INT_EN` | RW | bit0 中断使能 |
| 0x0C | `INT_CLR / PENDING` | W1C / RO | 写 1 清 pending；读回 pending 状态 |
| 0x10 | `M_DIM` | RW | M 维度 |
| 0x14 | `N_DIM` | RW | N 维度 |
| 0x18 | `K_DIM` | RW | K 维度 |
| 0x20 | `W_ADDR` | RW | 权重基地址 |
| 0x24 | `A_ADDR` | RW | 激活基地址 |
| 0x28 | `R_ADDR` | RW | 结果基地址 |
| 0x30 | `ARR_CFG` | RW | 预留阵列配置寄存器 |
| 0x34 | `CLK_DIV` | RW | 预留 DFS 配置 |
| 0x38 | `CG_EN` | RW | 预留门控配置 |

## 4.2 推荐控制字

| 功能 | CTRL 值 |
|---|---:|
| INT8 + WS 启动 | `0x01` |
| INT8 + OS 启动 | `0x11` |
| FP16 + WS 启动 | `0x09` |
| FP16 + OS 启动 | `0x19` |
| 清零 CTRL | `0x00` |

> 建议把 `mode=10` 作为 FP16 的软件编码。虽然当前 RTL 内部"非 00"都进入 FP16 通路，但不要依赖这个宽松行为。

## 4.3 最小编程顺序

```c
#define NPU_BASE 0x02000000u
#define NPU_REG(off) (*(volatile unsigned int *)(NPU_BASE + (off)))

NPU_REG(0x10) = M;
NPU_REG(0x14) = N;
NPU_REG(0x18) = K;
NPU_REG(0x20) = W_ADDR;
NPU_REG(0x24) = A_ADDR;
NPU_REG(0x28) = R_ADDR;
NPU_REG(0x00) = 0x11;   // 例：INT8 + OS + start

while ((NPU_REG(0x04) & 0x2u) == 0) {
    ;
}

NPU_REG(0x00) = 0x00;
```

### 为什么最后要写回 `0x00`

因为 `done` 是 sticky，`start` 也不会自动由软件视角清零。计算完成后显式清掉 CTRL，可以避免下一次调试时状态残留造成误判。

---

## 5. 数据布局与地址约定

## 5.1 DRAM 布局规则

当前主流程约定：

- `W_ADDR`：矩阵 `B[K×N]`，**列主序**
- `A_ADDR`：矩阵 `A[M×K]`，**行主序**
- `R_ADDR`：矩阵 `C[M×N]`，**行主序**

### 举例

若要计算 `C = A × B`，则第 `j` 列权重 `B[:,j]` 必须连续放在 DRAM 中，因为控制器会按：

```text
dma_w_addr = W_ADDR + j * k_dma_bytes
```

而第 `i` 行激活 `A[i,:]` 也必须连续放置，因为控制器按：

```text
dma_a_addr = A_ADDR + i * k_dma_bytes
```

### 5.1.1 Tile-Based 计算模型

**Tile 是 NPU 的核心计算单元**，代表一个输出元素 `C[i][j]` 的完整计算：

1. **Tile 定义**：一个 tile = 一个 `C[i][j]` 的计算任务
2. **Tile 数据**：需要 B 的第 j 列 `B[:,j]` 和 A 的第 i 行 `A[i,:]`
3. **Tile 计算**：完成 K 次乘积累加

**Tile 循环流程**：
```
for i in [0..M-1]:
    for j in [0..N-1]:
        // 处理一个 tile
        1. DMA 读 B[:,j] → PPBuf_W
        2. DMA 读 A[i,:] → PPBuf_A
        3. PE 计算 K 个元素对
        4. flush → 结果写入 FIFO
        5. DMA 写结果到 C[i][j]
```

**Tile 地址计算**：
```
权重地址 = W_ADDR + j × K × 元素字节数
激活地址 = A_ADDR + i × K × 元素字节数
结果地址 = R_ADDR + (i×N + j) × 4
```

**元素字节数**：
- INT8：1 字节（DRAM 中 4 元素打包为 32-bit 字）
- FP16：2 字节（DRAM 中 2 元素打包为 32-bit 字）

## 5.2 打包格式

| 模式 | 32-bit 字中的元素数 | 说明 |
|---|---:|---|
| INT8 | 4 | PPBuf 逐字节拆开给 PE |
| FP16 | 2 | 顶层先拼回 16-bit FP16，再送 PE |
| 结果 | 1 | 结果始终按 32-bit 写回 |

## 5.3 SoC 地址空间

默认 SoC 地址空间如下：

| 地址范围 | 区域 | 说明 |
|---|---|---|
| `0x0000_0000 ~ 0x0000_0FFF` | SRAM | 4KB，CPU 指令/数据 |
| `0x0000_1000 ~ 0x0000_FFFF` | DRAM | 约 60KB，CPU 与 NPU DMA 共用 |
| `0x0200_0000 ~ 0x0200_0038` | NPU Reg | 配置空间 |

### 一个非常容易踩的坑

**`0x00000F00` 属于 SRAM，不属于 DRAM。**

也就是说：

- 你如果把 PASS/FAIL 标记写到 `0x0F00`
- testbench 却去监视 DRAM
- 那么它永远不会触发

SoC 测试最终采用的是 **`0x00002000`** 作为 PASS 标记地址。

---

## 6. 端到端推理调度详解：三层卷积+激活

本章以一个**三层卷积神经网络（每层后接 ReLU 激活）的前向推理**为例，完整描述：

1. 软件（固件/驱动）如何准备数据、调度 NPU、处理中断
2. 硬件（NPU FSM + DMA + PPBuf + PE）每一层内部的状态跳转

---

### 6.1 背景：Im2Col 的作用

卷积操作在送入 NPU 之前，需要由软件将输入 Feature Map 做 **Im2Col** 变换，将卷积转换为矩阵乘法 `Y = W × X_col`：

| 符号 | 含义 | 形状（示例）|
|------|------|-----------|
| `W` | 卷积核（展开后的权重矩阵）| `[M × K]`，M=输出通道，K=卷积核展开长度 |
| `X_col` | Im2Col 后的输入矩阵 | `[K × N]`，N=输出位置数（H_out × W_out）|
| `Y` | 输出矩阵（reshape 后即为输出 Feature Map） | `[M × N]` |

**NPU 执行的就是这个 `Y = W × X_col` 矩阵乘法**，每个输出元素 `Y[i][j]` 对应一个 tile。

对应 NPU 寄存器约定：
- `A_ADDR` ← W（权重矩阵，行主序，M×K）
- `W_ADDR` ← X_col（Im2Col 后的输入，列主序，K×N）
- `R_ADDR` ← Y（输出矩阵，行主序，M×N）
- `M_DIM=M, N_DIM=N, K_DIM=K`

> ⚠️ **注意**：NPU 寄存器的 `W_ADDR`（权重基地址）指向 DRAM 中**按列主序存放的 B 矩阵**（即 X_col），而不是神经网络权重。神经网络权重对应 `A_ADDR`。这是历史命名遗留。

---

### 6.2 场景设定

假设三层卷积网络，每层参数如下（简化为整数，INT8 模式，OS 模式）：

| 层 | M（输出通道） | K（内积长度） | N（输出位置） | 矩阵乘规模 |
|----|:-----------:|:-----------:|:-----------:|----------|
| Layer 1 | 4 | 8 | 4 | `A[4×8] × B[8×4] = C[4×4]` |
| Layer 2 | 4 | 4 | 4 | `A[4×4] × B[4×4] = C[4×4]` |
| Layer 3 | 4 | 4 | 2 | `A[4×4] × B[4×2] = C[4×2]` |

激活函数：每层矩阵乘完成后，CPU 对结果矩阵做 **ReLU**（逐元素 `max(0, x)`）。

DRAM 地址布局（示例，INT8）：

```
0x1000  Layer1_W[4×8]  （A_ADDR，行主序）    32 字节
0x1020  Layer1_X_col[8×4] （W_ADDR，列主序） 32 字节
0x1040  Layer1_Y[4×4]  （R_ADDR，输出）      64 字节（每元素 4B，32-bit）
0x1080  Layer2_W[4×4]  （A_ADDR）           16 字节
0x1090  Layer2_X_col（Im2Col of Layer1_Y）（W_ADDR）16 字节
0x10B0  Layer2_Y[4×4]  （R_ADDR）           64 字节
0x10F0  Layer3_W[4×4]  （A_ADDR）           16 字节
0x1100  Layer3_X_col（W_ADDR）              8 字节
0x1110  Layer3_Y[4×2]  （R_ADDR）           32 字节
```

---

### 6.3 软件部分：逐层调度流程

#### 6.3.1 全局初始化（一次性）

```c
#define NPU_BASE   0x02000000u
#define NPU_REG(x) (*(volatile unsigned int *)(NPU_BASE + (x)))

// 设置公共模式：INT8 + OS
#define CTRL_INT8_OS  0x11u   // start=1, mode=00, stat=01

// 使能中断（若使用中断模式）
NPU_REG(0x08) = 0x1u;
```

#### 6.3.2 第 1 层：Layer 1 卷积（4×8×4 矩阵乘）

**① 软件准备数据**

```c
// 1. 将卷积核 Layer1_W 以行主序写入 DRAM 0x1000
// 2. 对输入 Feature Map 做 Im2Col，将结果以列主序写入 DRAM 0x1020
//    （Im2Col 完全由 CPU 执行，NPU 不参与此步）
init_layer1_weight(0x1000);  // 写 A_ADDR
im2col_layer1(input_fm, 0x1020);  // 写 W_ADDR（列主序 X_col）
```

**② 配置 NPU 寄存器**

```c
NPU_REG(0x10) = 4;        // M_DIM = 4
NPU_REG(0x14) = 4;        // N_DIM = 4
NPU_REG(0x18) = 8;        // K_DIM = 8
NPU_REG(0x20) = 0x1020;   // W_ADDR（X_col 列主序）
NPU_REG(0x24) = 0x1000;   // A_ADDR（权重行主序）
NPU_REG(0x28) = 0x1040;   // R_ADDR（输出）
```

**③ 启动 NPU**

```c
NPU_REG(0x00) = CTRL_INT8_OS;  // 写 start=1（触发 cfg_start_rise，shadow reg 锁存）
```

> 此时影子寄存器锁存完成，CPU 可以安全开始准备 Layer 2 的数据。

**④ 等待完成（轮询模式）**

```c
while ((NPU_REG(0x04) & 0x2u) == 0) { ; }  // 等 done=1
// 或中断模式：等 irq → 进入中断服务函数
```

**⑤ 清 IRQ 与 CTRL**

```c
NPU_REG(0x0C) = 0x1u;   // INT_CLR bit0=1，清中断
NPU_REG(0x00) = 0x00u;  // 清 CTRL（done sticky 回零）
```

**⑥ 软件执行 ReLU**

```c
// Layer 1 输出在 R_ADDR=0x1040，共 4×4=16 个 32-bit 结果
for (int i = 0; i < 16; i++) {
    int v = *(volatile int *)(0x1040 + i*4);
    *(volatile int *)(0x1040 + i*4) = (v > 0) ? v : 0;  // ReLU
}
```

#### 6.3.3 第 2 层：Layer 2 卷积（4×4×4 矩阵乘）

```c
// 以 Layer1_Y（ReLU 后）作为输入做 Im2Col → 列主序写入 0x1090
im2col_layer2(0x1040, 0x1090);

NPU_REG(0x10) = 4; NPU_REG(0x14) = 4; NPU_REG(0x18) = 4;
NPU_REG(0x20) = 0x1090; NPU_REG(0x24) = 0x1080; NPU_REG(0x28) = 0x10B0;
NPU_REG(0x00) = CTRL_INT8_OS;

while ((NPU_REG(0x04) & 0x2u) == 0) { ; }
NPU_REG(0x0C) = 0x1u; NPU_REG(0x00) = 0x00u;

// ReLU on Layer2_Y at 0x10B0
for (int i = 0; i < 16; i++) {
    int v = *(volatile int *)(0x10B0 + i*4);
    *(volatile int *)(0x10B0 + i*4) = (v > 0) ? v : 0;
}
```

#### 6.3.4 第 3 层：Layer 3 卷积（4×4×2 矩阵乘）

```c
im2col_layer3(0x10B0, 0x1100);

NPU_REG(0x10) = 4; NPU_REG(0x14) = 2; NPU_REG(0x18) = 4;
NPU_REG(0x20) = 0x1100; NPU_REG(0x24) = 0x10F0; NPU_REG(0x28) = 0x1110;
NPU_REG(0x00) = CTRL_INT8_OS;

while ((NPU_REG(0x04) & 0x2u) == 0) { ; }
NPU_REG(0x0C) = 0x1u; NPU_REG(0x00) = 0x00u;
// Layer 3 输出为 logits，通常不做 ReLU，直接读取 0x1110
```

---

### 6.4 硬件部分：每层内部的状态跳转详解

以 **Layer 1**（M=4, N=4, K=8，INT8, OS 模式）为例，详细描述 NPU FSM 从 start 到 irq 的全程跳转。共有 M×N=16 个 tile。

#### 6.4.1 Layer 1 Tile 序列（行主序推进） 

```
Tile (0,0): C[0][0] = A[0,:] · B[:,0]   target_col=0
Tile (0,1): C[0][1] = A[0,:] · B[:,1]   target_col=1
Tile (0,2): C[0][2] = A[0,:] · B[:,2]   target_col=2
Tile (0,3): C[0][3] = A[0,:] · B[:,3]   target_col=3
Tile (1,0): C[1][0] = A[1,:] · B[:,0]   target_col=0
...
Tile (3,3): C[3][3] = A[3,:] · B[:,3]   target_col=3  ← is_last_tile
```

#### 6.4.2 Phase 0 — 暖机（Warm-Up）

**触发**：CPU 写 CTRL = `0x11`，`cfg_start_rise = 1`。

```
S_IDLE（触发 cfg_start_rise）
  ├─ 动作：
  │    锁存影子寄存器（lk_m/n/k_dim, lk_w/a/r_addr, lk_mode=INT8, lk_stat=OS）
  │    w_ppb_clear=1, a_ppb_clear=1, r_fifo_clear=1
  │    tile_i=0, tile_j=0
  │    dma_w_addr = W_ADDR（= 0x1020），dma_w_len = K×1 = 8 字节
  │    dma_a_addr = A_ADDR（= 0x1000），dma_a_len = 8 字节
  │    dma_w_start=1, dma_a_start=1   ← 发 tile(0,0) W/A DMA
  │    busy=1
  └─ 跳转 → S_WARMUP_LOAD
```

```
S_WARMUP_LOAD
  ├─ 等待：dma_load_done（dma_w_done_r=1 且 dma_a_done_r=1）
  │    DMA 通过 AXI4 INCR burst 从 DRAM 读取：
  │      权重 B[:,0]（8 字节，2 个 32-bit 字）→ PPBuf_W Pong Bank
  │      激活 A[0,:]（8 字节，2 个 32-bit 字）→ PPBuf_A Pong Bank
  ├─ 动作：w_ppb_swap=1, a_ppb_swap=1   ← Pong→Ping，tile(0,0) 数据切到 PE 可读侧
  │         dma_w_done_r=0, dma_a_done_r=0
  └─ 跳转 → S_WARMUP_WAIT
```

```
S_WARMUP_WAIT（1 拍 swap 传播）
  ├─ 动作：pe_stat=OS(1), pe_load_w=0（OS 模式不需要 load_w）
  │    is_last_tile? 否（tile(0,0) 不是末 tile）
  │    发预取：
  │      tile(0,1) 的 W 地址 = W_ADDR + 1×8 = 0x1028
  │      tile(0,1) 的 A 地址 = A_ADDR + 0×8 = 0x1000（同一行 i=0）
  │      dma_w_start=1, dma_a_start=1   ← 预取 tile(0,1) → PPBuf Pong Bank
  └─ 跳转 → S_PRELOAD
```

```
S_PRELOAD（1 拍 rd_fill 稳定等待）
  ├─ pe_en=0（PPBuf swap 是时序电路，刚切换的 rd_fill 需要 1 拍才稳定）
  └─ 跳转 → S_OVERLAP_COMPUTE
```

#### 6.4.3 Phase 1 — Overlap 稳态（tile(0,0) 开始）

```
S_OVERLAP_COMPUTE（tile(0,0)，OS 模式）
  ├─ pe_en=1
  │  Ping Bank 供 PE 读取 tile(0,0)（B[:,0] 和 A[0,:]）
  │  Pong Bank 正在被 DMA 写入 tile(0,1)（后台，同步进行）
  │
  │  PPBuf 读口：8-bit 子字，逐拍输出
  │    每拍输出 1 字节；INT8 无需 FP16 装配，pe_data_ready 直接有效
  │    target_col = 0 % 4 = 0
  │    pe_array：权重只路由到 col=0；激活广播所有行
  │    PE[0][0] 累加 os_acc += w×a（共 K=8 拍）
  │
  │  退出条件：OS 模式下 w_ppb_empty && a_ppb_empty（8 拍数据消费完）
  └─ 跳转 → S_DRAIN
```

```
S_DRAIN（pe_flush=1，第 1 拍）
  pe_en=1, pe_flush=1
  PE Stage-0：s0_w=0, s0_a=0（flush 置零）
  PE Stage-2：acc_out = os_acc + 0 = os_acc；valid_out（下一拍）
  └─ 跳转 → S_DRAIN2
```

```
S_DRAIN2（pe_flush=0，第 2 拍）
  pe_en=1, pe_flush=0
  PE Stage-2：valid_out=1，acc_out=C[0][0] 写入 result FIFO
  └─ 跳转 → S_WRITE_BACK
```

```
S_WRITE_BACK
  dma_r_addr = R_ADDR + (0×4 + 0)×4 = 0x1040
  dma_r_len = 4
  dma_r_start=1   ← DMA R 通道从 result FIFO 读结果，AXI4 写回 DRAM
  └─ 跳转 → S_WB_WAIT
```

```
S_WB_WAIT
  等 dma_r_done_r=1
  is_last_tile? 否
  
  若 tile(0,1) 预取 DMA 已完成（dma_load_done=1）：
    ├─ 推进计数：tile_j: 0→1
    ├─ w_ppb_swap=1, a_ppb_swap=1  ← Pong(tile(0,1))→Ping，供 PE 下次计算
    ├─ r_fifo_clear=1
    ├─ pe_stat=OS(1), ws_consume_cnt=0
    ├─ dma_w_done_r=0, dma_a_done_r=0
    ├─ next_is_last? 否（tile(0,2) 不是末）
    │    发预取 tile(0,2)：
    │      dma_w_addr = W_ADDR + 2×8 = 0x1030
    │      dma_a_addr = A_ADDR + 0×8 = 0x1000
    │      dma_w_start=1, dma_a_start=1
    └─ 跳转 → S_PRELOAD → S_OVERLAP_COMPUTE（tile(0,1)）

  若预取尚未完成（PE 比 DMA 快，K 小时可能发生）：
    └─ 跳转 → S_WAIT_PREFETCH → 等完成 → S_PRELOAD → S_OVERLAP_COMPUTE
```

#### 6.4.4 Phase 1 稳态循环（tile(0,1) … tile(3,2)）

以上模式重复执行。每轮：
- **PE 端**：从 Ping Bank 消费当前 tile 的 K=8 拍数据
- **DMA 端**：同步向 Pong Bank 预取下一个 tile

**行边界（j 回绕）**示例：tile(0,3) 完成后，下一个是 tile(1,0)：

```
WB_WAIT（tile(0,3) 写回完成）：
  tile_j: 3→0（j+1=4 >= N=4，回绕）
  tile_i: 0→1
  
  发预取 tile(1,1)：
    dma_w_addr = W_ADDR + 1×8 = 0x1028
    dma_a_addr = A_ADDR + 1×8 = 0x1008  ← i=1，换行
    dma_w_start=1, dma_a_start=1
  
  swap PPBuf（Pong=tile(1,0)→Ping）
  → S_PRELOAD → S_OVERLAP_COMPUTE（tile(1,0)）
```

#### 6.4.5 Phase 2 — 末 tile（tile(3,3)）

```
WB_WAIT（tile(3,2) 写回完成）：
  推进 tile_j: 2→3
  swap PPBuf（Pong=tile(3,3)→Ping）
  
  next_is_last = True（tile(3,3) 是末 tile）
    → 不发新预取（无需写 Pong）
  
  → S_PRELOAD → S_OVERLAP_COMPUTE（tile(3,3)）
  → S_DRAIN → S_DRAIN2（结果 C[3][3] 写入 FIFO）
  → S_WRITE_BACK → S_WB_WAIT

WB_WAIT（tile(3,3) 写回完成）：
  is_last_tile = True
  → S_DONE
```

```
S_DONE
  irq=1, done=1, busy=0
  ws_consume_cnt=0, tile_i=0, tile_j=0
  w_ppb_clear=1, a_ppb_clear=1, r_fifo_clear=1
  → S_IDLE（等待 CPU 确认中断并配置下一层）
```

#### 6.4.6 Layer 1 完整时序总览

```
时钟阶段          NPU 状态               DMA 行为              PE 行为
──────────────────────────────────────────────────────────────────────
T0 (start)        S_IDLE→WU_LOAD         读 tile(0,0) W+A       空闲
T1~Tk             S_WARMUP_LOAD          DMA 搬运（K/4 burst）   空闲
Tk+1              S_WARMUP_WAIT          发 tile(0,1) 预取        空闲
Tk+2              S_PRELOAD              预取 tile(0,1) 继续      空闲
Tk+3 ~ Tk+K+2     S_OVERLAP_COMPUTE      预取 tile(0,1) 完成      计算 tile(0,0)
Tk+K+3            S_DRAIN                WB tile(0,0) 准备        flush
Tk+K+4            S_DRAIN2               DMA R write start        valid_out
Tk+K+5            S_WRITE_BACK           AXI4 写 DRAM             —
Tk+K+6            S_WB_WAIT              swap + 发 tile(0,2) 预取 —
Tk+K+7            S_PRELOAD              预取 tile(0,2) 进行       —
Tk+K+8 ~ ...      S_OVERLAP_COMPUTE      预取 tile(0,2) 同步       计算 tile(0,1)
...（持续 16 个 tile）
末 tile 完成       S_DONE                 —                        —
                   irq=1
```

#### 6.4.7 Layer 2 / Layer 3

与 Layer 1 流程相同，差异仅在于：
- M/N/K 维度不同（影子寄存器在 start 时重新锁存）
- W/A/R 地址指向新的 DRAM 区域
- tile 总数 = M×N（Layer 2: 16 个，Layer 3: 8 个）

---

### 6.5 软件与硬件职责分工总结

| 事项 | 执行方 | 说明 |
|------|:------:|------|
| Im2Col 变换 | **软件（CPU）** | 在 NPU 启动前完成，输出列主序矩阵到 DRAM |
| 权重加载到 DRAM | **软件（CPU）** | 初始化时完成，行主序 |
| 配置 NPU 寄存器 | **软件（CPU）** | 写 M/N/K/W_ADDR/A_ADDR/R_ADDR/CTRL |
| 等待完成 | **软件（CPU）** | 轮询 STATUS.done 或等待中断 |
| IRQ 清除 | **软件（CPU）** | 写 INT_CLR bit0=1 或 CTRL bit6=1 |
| CTRL 清零 | **软件（CPU）** | 写 `0x00`，让 done sticky 回零 |
| ReLU 激活 | **软件（CPU）** | 遍历 R_ADDR 结果，原地做 max(0,x) |
| Im2Col 下一层输入 | **软件（CPU）** | 以上一层 ReLU 输出为输入，CPU 执行 |
| 地址锁存 | **硬件（npu_ctrl）** | cfg_start_rise 时锁存所有影子寄存器 |
| tile 循环调度 | **硬件（npu_ctrl）** | 自动推进 tile_i/j，计算所有地址 |
| DMA 搬运 | **硬件（npu_dma）** | AXI4 INCR burst，3 通道时分复用 |
| Ping-Pong 切换 | **硬件（npu_ctrl + PPBuf）** | swap 脉冲触发，自动切换读写 bank |
| PE 计算 | **硬件（pe_array）** | K 拍 MAC，flush 触发输出 |
| 结果写回 | **硬件（npu_dma）** | 从 result FIFO 写回 DRAM |

---

### 6.6 关键注意事项

1. **Im2Col 的列主序要求**：B 矩阵（X_col）必须以列主序存放，原因是 DMA 按 `W_ADDR + j×K×bytes` 读取第 j 列，列内元素必须连续。

2. **Shadow Reg 保护**：NPU 运行期间可安全写入下一层的配置寄存器（M_DIM/N_DIM/K_DIM/W_ADDR/A_ADDR/R_ADDR），不影响当前层运行。只有当新的 `start` 脉冲到来时才会重新锁存。

3. **ReLU 时机**：必须在 `STATUS.done=1` 且清 CTRL 之后才能读取结果（结果 DMA 写回 DRAM 在 done 置位前完成）。

4. **结果格式**：INT8 结果是 32-bit 有符号整数（K 次 8-bit × 8-bit 累加，最大值 `127×127×K`）；FP16 结果是 32-bit FP32 值。ReLU 实现需按对应格式处理。

5. **多层连续执行**：不需要完全重置 NPU，只需清中断+清 CTRL，然后配置新参数并发 start 即可。

---

## 7. SoC 集成使用说明

## 7.1 已验证场景

当前 SoC 测试使用 PicoRV32 固件完成：

1. 初始化 DRAM 中的 A / B 数据
2. 通过 AXI-Lite 配置 NPU
3. 启动 NPU
4. 轮询 `STATUS.done`
5. 校验结果
6. 在 DRAM `0x2000` 写入 PASS 标记 `0xAA`

### 当前通过结果

```text
[PASS] SoC integration test PASSED!
Cycles: 287
R_ADDR results: C[0][0]=19 C[0][1]=22 C[1][0]=43 C[1][1]=50
```

## 7.2 固件入口文件

主要文件：

- `tb/soc_test.S`
- `tb/assemble_soc_test.py`
- `tb/tb_soc.v`
- `scripts/run_soc_sim.ps1`

## 7.3 重新生成固件 hex

```powershell
cd D:\NPU_prj\tb
python assemble_soc_test.py
```

### 一个重要细节

`tb_soc.v` 必须从 **`../tb/soc_test.hex`** 读取 hex，而不是裸写 `soc_test.hex`。

原因是：

- `vvp` 的工作目录通常在 `sim/`
- `$readmemh("soc_test.hex", ...)` 会优先找 `sim/soc_test.hex`
- 如果那里残留旧文件，CPU 会执行完全错误的程序

---

## 8. 常用仿真入口

| 目标 | 命令 |
|---|---|
| PE 单元测试 | `powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1` |
| NPU 综合测试 | `powershell -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1` |
| 分类器测试 | `powershell -ExecutionPolicy Bypass -File scripts\run_classifier_sim.ps1` |
| K 深度验证（1x1 PE，K=4/8/16/32） | `powershell -ExecutionPolicy Bypass -File scripts\run_array_scale.ps1` |

| SoC 集成测试 | `powershell -ExecutionPolicy Bypass -File scripts\run_soc_sim.ps1` |
| 全量回归 | `powershell -ExecutionPolicy Bypass -File scripts\run_regression.ps1` |

### 建议的日常顺序

1. 改 PE / 浮点逻辑后：先跑 `run_sim.ps1`
2. 改控制器 / DMA 后：跑 `run_full_sim.ps1`
3. 改 SoC / bridge / SRAM / DRAM 后：跑 `run_soc_sim.ps1`
4. 准备收尾前：跑 `run_regression.ps1`

### WS 模式 flush beat 规范（重要）

**WS 模式下 flush 采用纯触发语义**，编写 Testbench 或硬件驱动时必须遵守：

```verilog
// 正确：flush beat 前先确保 en=0，然后发纯 flush
@(posedge clk); #1;
w_in = 16'd0; a_in = 16'd0;   // 必须清零
flush = 1; en = 1;
@(posedge clk); #1;
flush = 0; en = 0;
```

- `a_in` 和 `w_in` **必须为 0**，flush beat 不应携带计算数据
- 每个数据 beat（包括 `load_w=1` beat）完成后需立即降低 `en`，避免重复采样
- 结果：K 个数据 beat 完成累加后，flush beat 直接输出 `ws_acc` 并清零

---

## 9. 常见问题

### Q1：CPU 卡在轮询 `STATUS.done`

优先检查三件事：

1. `soc_mem` 是否仍是组合读
2. `dram_model` 的 CPU 读口是否仍是组合读
3. `axi_lite_bridge` 是否保持 AW -> W 两段写流程

如果这些地方被改成"更像同步 RAM/同步总线"的写法，PicoRV32 很容易读到 stale data。

### Q2：PASS 标记写了，但 testbench 没检测到

先确认地址是不是在 DRAM。

- `0x0F00`：SRAM
- `0x2000`：DRAM

### Q3：CPU 执行的程序和我写的固件不一样

通常是加载了错误的 hex 文件。

- 检查 `tb_soc.v` 的 `$readmemh` 路径
- 检查 `sim/` 下是否残留旧 `soc_test.hex`

### Q4：FP16 结果应该怎么看

不要再按"低 16 位 FP16、高 16 位符号扩展"去理解。

当前 FP16 路径的累加结果是**32-bit 结果表示**，其语义由 `pe_top` 中的 FP16/FP32 累加路径决定。简单说：

- 输入是 FP16
- 中间会做更宽的累加
- 输出/写回按 32-bit 结果处理

### Q5：`ARR_CFG` / `CLK_DIV` / `CG_EN` 能直接改运行结果吗

目前不建议把它们当成"已经完整生效的用户接口"。

- `ARR_CFG`：当前主流程未完整消费
- `CLK_DIV` / `CG_EN`：更多是保留和行为模型接口。`npu_power` 模块已实例化，但 `npu_clk`、`row_clk_gated`、`col_clk_gated` 三路输出在 `npu_top` 中全部悬空，**未驱动 PE 主时钟路径**，电源管理完全不生效。

### Q6：为什么 INT8 模式下 PE 输入还是 16-bit

因为顶层统一使用 `DATA_W=16` 接口，INT8 只消费低 8 位。真正的 8-bit 元素切分是在 PPBuf 读侧完成的。

### Q7：OS 模式下 COLS>2 时结果是否正确

✅ **已修复（2026-04-08）**：`npu_ctrl.target_col` 已由 1-bit 扩展为 `$clog2(COLS > 1 ? COLS : 2)` 位，`npu_top.v` 中 `ctrl_target_col` wire 同步扩宽。COLS=4（默认配置）下 OS 模式列路由已验证正确。

### Q8：如何清除中断？有哪些方法？

有两种方法，任选其一：

- **方法 A**：向 `0x020C`（INT_CLR）写 `0x1`，清除 `int_pending`
- **方法 B**：向 `0x0200`（CTRL）写 `0x40`（bit6=1），W1C 清 IRQ

方法 B 的优势是可以和清 CTRL 合并为一次写操作：写 `0x40` 同时清中断又不重新启动 NPU（bit0=0）。

### Q9：为什么运行时带宽利用率很低

✅ **已优化（2026-04-08）**：DMA 读通道已升级为多拍 INCR burst。新增 `calc_arlen()` function，按 `min(剩余字数, BURST_MAX) - 1` 动态计算 `arlen`；`ar_sent` 标志防止 AR 重发，`rlast` 后判断剩余量决定是否发起下一个 AR 事务。读通道不再是逐字单拍，带宽利用率已显著提升。

✅ **进一步优化（2026-04-13）**：npu_ctrl 升级为真 Ping-Pong 重叠 FSM，DMA 加载下一 tile 与 PE 计算当前 tile 同步进行，从根本上消除了 DMA 等待 PE、PE 等待 DMA 的串行开销。

---

## 10. 配套文档索引

| 文档 | 用途 |
|---|---|
| `README.md` | 项目总览与入口 |
| `doc/architecture.md` | 系统级结构、真 Ping-Pong FSM、数据流、地址与时序约束 |
| `doc/module_reference.md` | 模块级信号与时序细节 |
| `doc/simulation_guide.md` | 仿真脚本和测试说明 |
| `doc/npu_debug_checklist.md` | 排障与历史 bug 检查表 |

---

## 最后给使用者的建议

如果你刚接手这个工程，最稳妥的理解顺序是：

1. 先把 **SoC 地址空间**搞清楚
2. 再把 **B 列主序 / A 行主序 / C 行主序** 记住
3. 然后把 **tile-loop = 每次算一个 `C[i][j]`** 这个模型刻进脑子里
4. 最后理解 **真 Ping-Pong**：PE 算 tile(i,j) 时，DMA 已经在后台装下一个 tile，这是性能的关键所在

这样你看控制器、DMA、固件和 testbench 时，就不会老把它想成另一套架构。
