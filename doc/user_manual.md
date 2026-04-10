# NPU_prj 用户手册

> 最后更新：2026-04-10  
> 状态：已按当前 RTL、SoC 联调结果与全量回归结果同步

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境准备](#2-环境准备)
3. [快速开始](#3-快速开始)
4. [寄存器编程指南](#4-寄存器编程指南)
5. [数据布局与地址约定](#5-数据布局与地址约定)
6. [SoC 集成使用说明](#6-soc-集成使用说明)
7. [常用仿真入口](#7-常用仿真入口)
8. [常见问题](#8-常见问题)
9. [配套文档索引](#9-配套文档索引)

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

## 6. SoC 集成使用说明

## 6.1 已验证场景

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

## 6.2 固件入口文件

主要文件：

- `tb/soc_test.S`
- `tb/assemble_soc_test.py`
- `tb/tb_soc.v`
- `scripts/run_soc_sim.ps1`

## 6.3 重新生成固件 hex

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

## 7. 常用仿真入口

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

## 8. 常见问题

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

### Q8：为什么运行时带宽利用率很低

✅ **已优化（2026-04-08）**：DMA 读通道已升级为多拍 INCR burst。新增 `calc_arlen()` function，按 `min(剩余字数, BURST_MAX) - 1` 动态计算 `arlen`；`ar_sent` 标志防止 AR 重发，`rlast` 后判断剩余量决定是否发起下一个 AR 事务。读通道不再是逐字单拍，带宽利用率已显著提升。

---

## 9. 配套文档索引

| 文档 | 用途 |
|---|---|
| `README.md` | 项目总览与入口 |
| `doc/architecture.md` | 系统级结构、数据流、地址与时序约束 |
| `doc/module_reference.md` | 模块级说明 |
| `doc/simulation_guide.md` | 仿真脚本和测试说明 |
| `doc/npu_debug_checklist.md` | 排障与历史 bug 检查表 |

---

## 最后给使用者的建议

如果你刚接手这个工程，最稳妥的理解顺序是：

1. 先把 **SoC 地址空间**搞清楚
2. 再把 **B 列主序 / A 行主序 / C 行主序** 记住
3. 然后把 **tile-loop = 每次算一个 `C[i][j]`** 这个模型刻进脑子里

这样你看控制器、DMA、固件和 testbench 时，就不会老把它想成另一套架构。
