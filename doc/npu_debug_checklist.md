# NPU_prj 调试检查清单（Debug Checklist）

> 版本：2026-04-10（更新：补充 Bug-21 WS flush 语义修正；将第 13 节 FP16 E2E 专项分析归入历史记录）  
> 适用范围：所有仿真场景（PE 单元 / NPU 综合 / 多行多列 / SoC 集成）  
> 使用方式：遇到 FAIL / TIMEOUT / 结果错误时，按分节逐项排查。

---

## 目录

1. [快速诊断流程](#1-快速诊断流程)
2. [TIMEOUT 排查](#2-timeout-排查)
3. [结果值错误排查](#3-结果值错误排查)
4. [多行多列（ROWS/COLS > 1）专项](#4-多行多列rowscols--1专项)
5. [DMA 相关问题](#5-dma-相关问题)
6. [AXI 协议常见陷阱](#6-axi-协议常见陷阱)
7. [数据打包与地址计算](#7-数据打包与地址计算)
8. [FP16 模式专项](#8-fp16-模式专项)
9. [OS / WS 模式差异](#9-os--ws-模式差异)
10. [SoC 集成调试专项](#10-soc-集成调试专项)
11. [历史 Bug 记录（已修复）](#11-历史-bug-记录已修复)
12. [GTKWave 常用信号速查](#12-gtkwave-常用信号速查)

---

## 1. 快速诊断流程

```
仿真失败
    │
    ├─ TIMEOUT? ──────────────────► 第2节
    │
    ├─ 结果值错误（got≠exp）? ────► 第3节
    │    ├─ got=0 (全为0)?  ──────► 第5节 DMA 排查
    │    ├─ got 符号错误?   ──────► 第7节 打包 + 第9节 OS/WS
    │    ├─ col0 正确但 col1 错? ► 第4节 多行多列专项
    │    └─ T2/T3 结果写到 T1 地址? ► 第5节 DMA burst fix
    │
    ├─ SoC 集成失败? ─────────────► 第10节 SoC 专项
    │    ├─ CPU 挂在 poll_loop?  ──► 10.1 同步读 Bug
    │    ├─ PASS 标记未被检测到? ──► 10.2 地址空间 Bug
    │    └─ CPU 执行错误指令?  ───► 10.3 hex 路径 Bug
    │
    └─ 编译错误? ─────────────────► 检查 RTL 文件列表顺序（fp16_mul → pe_top → ...）
```

---

## 2. TIMEOUT 排查

**现象**：`*** TIMEOUT after N cycles! ***`

### 检查步骤

| 步骤 | 检查项 | 快速定位方法 |
|------|--------|-------------|
| 2.1 | FSM 卡在哪个状态？ | `$display ctrl_state, dma_state` 或 GTKWave 看 `u_ctrl.state` |
| 2.2 | NPU done 信号是否拉高？ | GTKWave: `u_ctrl.status_done` |
| 2.3 | DMA r_start 是否发出？ | `u_dma.dma_r_start` (应为单脉冲) |
| 2.4 | r_fifo 是否有数据？ | `u_dma.r_fifo_empty_n` |
| 2.5 | AW handshake 是否完成？ | `m_axi_awvalid & m_axi_awready` |
| 2.6 | wlast 是否拉高？ | `m_axi_wlast` (应在最后一拍 = 1) |

### 常见根因

#### TIMEOUT-A：DMA r_start 重复触发
- **现象**：FSM 反复进入 S_WB_WAIT → S_WB_WAIT，never completes
- **根因**：`npu_ctrl.v` S_WB_WAIT 状态持续拉 `dma_r_start`（已修复：只等 `dma_r_done_r`）
- **确认方法**：看 `u_dma.r_pending` 是否正确锁存

#### TIMEOUT-B：r_fifo 空时 DMA 进入 R_WRITE
- **现象**：DMA 进入 R_WRITE 但 FIFO 为空，wvalid 永不拉高
- **根因**：OS pipeline 延迟导致 r_start 到达时 FIFO 还没有数据（已修复：r_pending 机制）
- **确认方法**：`u_dma.r_fifo_empty_n` 在 R_WRITE 进入时是否为 1

#### TIMEOUT-C：M_DIM 设置错误（多列模式）
- **现象**：COLS=2 但设 M_DIM=2，DMA 等待 8 字节但 FIFO 只有 2×4=8 字节…  
  实际问题：DMA r_len = M×N×4，若设 M_DIM=ROWS（而非 1），会导致等待数据超过实际产生量
- **根因**：物理 ROWS≠逻辑 M_DIM。每次 flush 只产生 COLS 个结果，M_DIM 必须=1
- **正确设置**：`M_DIM=1, N_DIM=COLS, K_DIM=K`

#### TIMEOUT-D：AXI B channel 未响应
- **现象**：DMA 等待 bvalid 永不到来
- **检查**：testbench 的 DRAM 写模型是否正确处理 wlast → b_pending → bvalid

---

## 3. 结果值错误排查

### 3.1 got=0（全为零）

| 可能根因 | 快速确认 |
|----------|----------|
| 读取结果地址在 run_npu 之前（DRAM 初始值为 0） | 确认在 `wait_done` 之后再读 `dram[R_ADDR>>2]` |
| DMA 写回地址错误（写到别处了） | 添加 `$display("[DRAM_WR]")` 跟踪 wr_base + addr |
| R_ADDR 计算错误 | 检查 `REG_R_ADDR` 设置值 vs `dram[...]` 下标 |
| AXI wlast 错误导致 wr_phase 永不清零 | 见第5节 DMA burst fix |

### 3.2 got 符号错误（正负号相反）

- INT8 符号扩展：PE 内部用 `$signed(int8_w) * $signed(int8_a)`，确保 DRAM 打包用 8-bit 补码
- `-2` 应打包为 `0xFE`，`-128` 为 `0x80`，`127` 为 `0x7F`
- FP16 符号位：FP16 bit[15] = sign，打包时不要符号扩展到 32-bit word 的高 16 位

### 3.3 got 值偏差但符号正确

- 检查 K_DIM 是否与实际元素数匹配
- 检查 W/A 打包的字节顺序（Little-Endian：低地址 = 低字节 = 第一个元素）
- 多列：col0 正确但 col1 错 → 见第4节 systolic shift

---

## 4. 多行多列（ROWS/COLS > 1）专项

### 4.1 ROWS > 1 的行为（OS 模式）

> **关键事实：OS 模式中 ROWS 不影响每列结果值。**

```
pe_a_in  = {ROWS{a_ppb_rd_data}}  ← 相同激活广播到所有行
pe_acc_in = 0                      ← 垂直 psum 链顶端固定为 0
acc_out   = acc_v[ROWS][c]         ← 仅取最底行 PE 的 acc_out

OS 模式中每个 PE 使用独立 os_acc，acc_in 被忽略。
最底行 PE 只有自己的 os_acc = sum(W[k]*A[k])，结果不含上层 PE 的贡献。
```

**✗ 错误理解**：`result = ROWS × dot_product`  
**✓ 正确理解**：`result = dot_product`（仅最底行 PE 输出，ROWS 无倍增效应）

### 4.2 COLS > 1 的行为（systolic 激活延迟）

```
act_h[r][0] = a_ppb_rd_data          (col0 直接接收，无延迟)
act_h[r][c] = 经过 c 个 act_reg 的延迟  (每列额外一拍)

对 K 轮数据流：
  col0 看到: A[0], A[1], ..., A[K-1]
  col1 看到: 0,    A[0], ..., A[K-2]  (1 拍延迟，首位填 0)
  col2 看到: 0,    0,    A[0], ..., A[K-3]
```

**期望值计算公式（以 COLS=2 为例）**：
```
col0_result = sum_{k=0}^{K-1}  W[k] * A[k]
col1_result = sum_{k=0}^{K-1}  W[k] * A_shifted[k]
            = sum_{k=1}^{K-1}  W[k] * A[k-1]  (A[0] 对应 0)
            = W[1]*A[0] + W[2]*A[1] + ... + W[K-1]*A[K-2]
```

### 4.3 多列 M_DIM 设置规则

| 参数 | 正确值 | 说明 |
|------|--------|------|
| M_DIM | `1` | 每次 flush 产生 COLS 个结果，逻辑行数为 1 |
| N_DIM | `COLS` | 结果列数 = 物理 PE 列数 |
| K_DIM | `K` | 元素数（权重/激活向量长度） |
| r_len | `M_DIM × N_DIM × 4 = COLS × 4` 字节 | DMA 写回长度 |

**⚠️ 错误设置 M_DIM=ROWS**：会导致 r_len=ROWS×COLS×4，但 FIFO 只放入 COLS 个数据，DMA 永久等待 → TIMEOUT。

### 4.4 结果序列化顺序

COLS > 1 时，serialiser（`npu_top.v` gen_r_ser 块）按列顺序写 FIFO：
```
FIFO 写入顺序: col0, col1, col2, ..., col[COLS-1]
DRAM 写入: [R_ADDR+0×4]=col0, [R_ADDR+1×4]=col1, ...
```

### 4.5 常见 COLS>1 调试信号

```verilog
// 在 testbench 中添加 (或 GTKWave 中观察):
u_npu.pe_array_valid        // 每次 flush 有效
u_npu.pe_array_result[31:0] // col0 结果
u_npu.pe_array_result[63:32]// col1 结果 (COLS=2)
u_npu.pe_valid_q            // 1-cycle 延迟后的 valid
u_npu.gen_r_ser.ser_active  // serialiser 正在 drain
u_npu.gen_r_ser.ser_col     // 当前 drain 的列
u_npu.r_fifo_wr_en          // FIFO 写使能
u_npu.r_fifo_din            // 写入 FIFO 的数据
```

---

## 5. DMA 相关问题

### 5.1 DMA 写回地址错误（Bug 已修复）

**现象**：T2/T3 的结果写到 T1 的 DRAM 地址

**根因（双重）**：
1. `burst_len=0`（每次 burst 只 1 beat），但 R_WRITE 写了 N beats → 地址不正确
2. `m_axi_wlast` 是 `output reg`（NBA 赋值），在最后一 beat 时 wlast 仍为 0（上一拍的值）→ DRAM BFM 的 `wr_phase` 永不清零 → `wr_base` 不更新

**修复（已合入 npu_dma.v）**：
```verilog
reg [7:0] r_burst_len;  // awlen = total_beats - 1
reg aw_sent;
assign m_axi_awlen = r_burst_len;
assign m_axi_wlast = (dma_state == R_WRITE) &&
                     (byte_cnt + DATA_W/8 >= r_len_latch);
```

### 5.2 DMA 状态机 FSM 状态映射

```
npu_dma.v FSM state 编码:
  IDLE    = 0  ← 空闲
  W_READ  = 1  ← 读权重（AR + R channel）
  A_READ  = 2  ← 读激活（AR + R channel）
  R_WRITE = 3  ← 写结果（AW + W + B channel）
```

### 5.3 r_pending 机制

OS 模式 pipeline 延迟问题（已修复）：
- `r_start` 到达时 PE 结果可能尚未进入 FIFO
- `r_pending` 标志锁存地址/长度，等 FIFO 非空再进入 R_WRITE
- 如果 DMA 不进入 R_WRITE 或 FIFO 一直为空，检查 flush 信号是否正确传播

### 5.4 DMA 诊断 $display 模板

在 testbench DRAM 写模型中添加：
```verilog
$display("[DRAM_WR] wr_base=0x%08h addr=0x%08h data=0x%08h beat=%0d",
         wr_base, (wr_base + wr_cnt*4), m_wdata, wr_cnt);
```
正常多次操作时 `wr_base` 应每次 DMA 操作都更新。

---

## 6. AXI 协议常见陷阱

### 6.1 wlast 必须为 combinational（已修复）

| 错误做法 | 正确做法 |
|----------|---------|
| `output reg m_axi_wlast` + NBA `m_axi_wlast <= ...` | `output wire m_axi_wlast` + combinational `assign` |

**原因**：NBA（Non-Blocking Assignment）在同一时钟沿的 active-region 之后生效，导致 DRAM BFM 在真正的 last beat 看到 wlast=0，无法完成 burst。

### 6.2 AW handshake 必须先于 W channel 数据

- DMA 发出 awvalid 后等待 awready，成功后才发 wvalid
- `aw_sent` 标志防止重复发 awvalid

### 6.3 B channel（Write Response）处理

- 每次 burst 完成后必须等待 bvalid，DMA 才能回到 IDLE
- 如果 bready 一直为 0 或 testbench DRAM 模型没有发 bvalid → TIMEOUT

### 6.4 arlen 编码

- AXI4 中 `arlen = N_beats - 1`（0 = 1 beat，15 = 16 beats）
- DMA 计算：`arlen = (len_bytes / DATA_BYTES) - 1`

---

## 7. 数据打包与地址计算

### 7.1 INT8 DRAM 打包（Little-Endian）

```
DRAM 32-bit word = 0xDDCCBBAA
                         │  │  │  └─ byte[0] → 第1个 INT8（最低地址）
                         │  │  └──── byte[1] → 第2个 INT8
                         │  └─────── byte[2] → 第3个 INT8
                         └────────── byte[3] → 第4个 INT8（最高地址）

示例：W=[1,2,3,4] → 0x04030201
      A=[127,-128,1,0] → 0x0001807F   (-128=0x80, 127=0x7F)
      W=[1,-1,2,-2]    → 0xFE02FF01   (-1=0xFF, -2=0xFE)
```

### 7.2 DRAM 地址 vs 数组下标

```verilog
// DRAM 字节地址 → 32-bit 字数组下标（>> 2）
dram[0x1020 >> 2]  = dram[0x408]   ← 字节地址 0x1020 的 32-bit word
dram[0x1024 >> 2]  = dram[0x409]   ← 紧接着下一个 word
```

### 7.3 DMA 传输长度计算

```
INT8 模式：
  K 个 INT8 元素 = K 字节 = K/4 个 32-bit words
  w_len = K 字节, a_len = K 字节
  r_len = M_DIM * N_DIM * 4 字节（32-bit 结果）

FP16 模式：
  K 个 FP16 元素 = 2K 字节 = K/2 个 32-bit words
  w_len = 2K 字节, a_len = 2K 字节
```

### 7.4 结果地址不重叠原则

- W_ADDR / A_ADDR / R_ADDR 必须各自独立，不得相互覆盖
- 多次操作时，R_ADDR 应递增（如 0x1020, 0x1030 等）以免新结果覆盖旧结果

---

## 8. FP16 模式专项

### 8.1 FP16 常见数值速查

| FP16 值 | 16-bit hex | FP32 等效 |
|---------|-----------|----------|
| 0.0 | 0x0000 | 0.0 |
| 1.0 | 0x3C00 | 1.0 |
| 2.0 | 0x4000 | 2.0 |
| -1.0 | 0xBC00 | -1.0 |
| Inf | 0x7C00 | +Inf |
| -Inf | 0xFC00 | -Inf |
| NaN | 0x7E00 | NaN |

### 8.2 FP16 DRAM 打包

FP16 每个值 16-bit，打包到 32-bit word 时每个 word 包含 2 个元素：
```
word = {FP16[1], FP16[0]}   (FP16[0] 在低 16-bit)
如：[1.0, 2.0] → {0x4000, 0x3C00} → 0x40003C00
```

### 8.3 PPBuf 相位（pe_mode=1 时）

FP16 模式下 PPBuf 出来的是 8-bit 子字，需要 2 拍组成 1 个 FP16：
- `w_ppb_phase=0`: 接收低字节 FP16[7:0]
- `w_ppb_phase=1`: 接收高字节 FP16[15:8]，之后 PE 才能消费一次

**PE 有效消费条件**（当前实现，已修复 Bug-20）：
```verilog
// phase 1→0 下降沿延迟 1 拍
pe_consume = pe_en && (pe_data_ready || pe_flush)
           && (pe_mode ? (w_phase_fall_d && a_phase_fall_d) || pe_flush : 1'b1);
```
即 FP16 模式下，需等待 `w_phase_fall_d && a_phase_fall_d`（高字节装配完成后的下一拍）才触发消费。

### 8.4 FP16 结果在 DRAM 中是 FP32

NPU 使用 FP32 混合精度累加（`fp32_add.v`），结果 FIFO 存储 FP32（32-bit）。
因此 FP16 操作的结果仍为 32-bit 写回 DRAM。

---

## 9. OS / WS 模式差异

| 特性 | OS（Output-Stationary）| WS（Weight-Stationary）|
|------|----------------------|----------------------|
| ctrl_reg bit[4] | 1 (CTRL_OS=0x10) | 0 (CTRL_WS=0x00) |
| 权重处理 | 每拍流入（W streams in）| 第一拍 load_w 锁存，后续复用 |
| 激活处理 | 每拍流入（A streams in）| 每拍流入 |
| 累加方式 | 内部 os_acc 独立累加 | 外部 acc_in 链式累加 |
| acc_in 使用 | **忽略**（OS 只用 os_acc）| 使用（WS 求部分和链） |
| flush | 输出 os_acc 并清零 | **纯触发输出**：直接输出 ws_acc（不加当前 s1_mul），ws_acc 清零；flush beat 中 a_in/w_in 必须为 0 |
| 多行效果 | **ROWS 不影响结果值** | WS 理论上可多行累加（未验证）|

### OS 模式检查点

- `CTRL_OS = 32'h11` (start=bit0, OS=bit4)  
  注意：`tb_comprehensive.v` 用 `CTRL_OS=0x10`（不含 start），需 OR 上 `CTRL_START=0x01`  
  注意：`tb_multi_rc_comprehensive.v` 用 `CTRL_OS=0x11`（已合并 start 位）
- flush 信号必须在 K 拍数据结束后由控制器发出
- 每次 flush 仅清零 os_acc，不影响 weight_reg

---

## 10. SoC 集成调试专项

> 适用场景：`tb_soc.v` + PicoRV32 CPU + NPU AXI-Lite 联合仿真

### 10.1 CPU 挂在 poll_loop（status_done 永不为 1）

**现象**：仿真运行数万 cycle，CPU 一直在轮询 STATUS 寄存器，NPU 无响应

**检查步骤**：

| 步骤 | 检查项 | 方法 |
|------|--------|------|
| 10.1.1 | SRAM 读时序是否正确 | `soc_mem.v` 必须是异步读（`assign rdata = mem[addr]`），不能是同步 |
| 10.1.2 | DRAM CPU 读时序是否正确 | `dram_model.v` CPU 读端口必须是异步读 |
| 10.1.3 | CPU 是否正确写入 CTRL 寄存器 | `$display` 打印 AXI-Lite 写地址和数据，确认 `addr=0x02000000 wdata=0x11` |
| 10.1.4 | NPU 是否收到 start 信号 | GTKWave 观察 `u_npu.u_ctrl.state` 是否离开 S_IDLE |

**根因（已修复 Bug-14 / Bug-15）**：
- `soc_mem.v` 原为同步读：`always @(posedge clk) rdata <= mem[addr]`
  - `ram_ready` 是组合信号（= `addr_is_ram`），同一周期有效
  - PicoRV32 看到 `ready=1` 但 `rdata` 还是上一周期的旧值
  - → 每条指令都执行错误，寄存器值全乱，CTRL 写入到错误地址
- **修复**：改为 `assign rdata = mem[addr[ADDR_W-1:0]]`（组合读）
- `dram_model.v` CPU 读端口同理，改为 `assign cpu_rdata = mem[cpu_addr[ADDR_W+1:2]]`

**诊断信号**：
```
// 在 tb_soc.v 中观察
u_soc.u_bridge.axi_awaddr   // AXI-Lite 写地址
u_soc.u_bridge.axi_wdata    // AXI-Lite 写数据
u_soc.cpu.mem_addr          // PicoRV32 当前访问地址
u_soc.cpu.mem_wdata         // PicoRV32 写数据
```

---

### 10.2 SoC 仿真不 TIMEOUT 但 PASS 标记未被检测到

**现象**：CPU 完成计算并写入标记，但 testbench 始终不打印 `[PASS]`

**根因（已修复 Bug-17）**：标记地址在 SRAM 空间而非 DRAM 空间

```
SoC 地址映射：
  addr_is_ram  = mem_addr < 0x1000   → 0x0000–0x0FFF = SRAM
  addr_is_dram = (mem_addr >= 0x1000 && mem_addr < 0x20000) → 0x1000–0x1FFFF = DRAM

旧代码：固件写 0x0F00 → 实际到 SRAM（因为 0x0F00 < 0x1000）
        testbench 监视 u_dram.mem[0x0F00>>2]=mem[960] → 永远不更新 → 永不 PASS
```

**修复**：固件标记地址改为 `0x2000`（在 DRAM 空间 0x1000–0x1FFFF 内），testbench 改为监视 `u_dram.mem[0x2000>>2] = mem[2048]`

---

### 10.3 CPU 执行错误指令（PC 跑飞 / 寄存器值异常）

**现象**：DBG trace 显示 CPU 从不期望的 PC 取指，或 SW 写入错误地址

**根因（已修复 Bug-18）**：`$readmemh` 加载了错误的 hex 文件

```
vvp 工作目录为 sim/ 时：
  $readmemh("soc_test.hex", ...) → 解析为 sim/soc_test.hex（可能是旧文件！）
  正确文件在 tb/soc_test.hex

症状：
  sim/soc_test.hex 10,240 字节（旧版，含 verify 分支）
  tb/soc_test.hex  420 字节（新版，正确逻辑）
  CPU 从旧 hex 执行，行为完全不同
```

**修复**：`$readmemh("../tb/soc_test.hex", ...)` → 无论 vvp 从哪个目录运行都能正确找到

**排查方法**：
```bash
# 在 sim/ 目录下检查 hex 文件时间戳和大小
ls -la soc_test.hex     # sim/ 下的 hex
ls -la ../tb/soc_test.hex  # tb/ 下的最新 hex
# 确认两者一致，或 tb_soc.v 使用绝对/正确相对路径
```

---

### 10.4 SoC 地址空间速查

| 区域 | 地址范围 | 模块 | 注意 |
|------|---------|------|------|
| SRAM | 0x0000–0x0FFF | soc_mem.v | 4KB，inst + data |
| DRAM | 0x1000–0x1FFFF | dram_model.v | 128KB |
| NPU 寄存器 | 0x02000000–0x0200001F | npu_axi_lite.v | AXI-Lite，8 个 32-bit 寄存器 |

**⚠️ 常见误区**：`0x0F00 < 0x1000`，属于 SRAM 空间，不是 DRAM！

### 10.5 AXI-Lite 写协议时序（npu_axi_lite.v 特殊性）

`npu_axi_lite.v` 的 AW/W channel 是**互斥**的：
- Cycle 1：`awready=1`（AW handshake），`wready=0`
- Cycle 2：`awready=0`，`wready=1`（W handshake，锁存 aw_q）
- `wr_en = aw_q && wvalid && wready`（第2 cycle 才实际写寄存器）

`axi_lite_bridge.v` 的状态机需适配此 2-cycle 写协议：`S_IDLE → S_WRITE_AW → S_WRITE_W → S_IDLE`（共 3 个周期完成一次写）

---

## 11. 历史 Bug 记录（已修复）

> 保留用于参考，以便未来类似问题快速对照。

| Bug ID | 模块 | 现象 | 根因 | 修复 | 验证 |
|--------|------|------|------|------|------|
| Bug-1 | `fp16_mul.v` | FP16 次正规数结果为 0 | flush-to-zero 而非渐进下溢 | 22-bit LZC + 右移舍入 + RN | tb_fp16_mul 44/44 |
| Bug-2 | `fp16_mul.v` | 次正规数输入 implicit bit 丢失 | exp=0 时 implicit=1 错误 | 动态隐式位（exp=0 时 implicit=0） | tb_fp16_mul 44/44 |
| Bug-3 | `npu_top.v` | OS flush 数据黑洞 | flush 周期先清零再累加，丢一拍数据 | flush 周期先累加再输出清零 | tb_pe_top 19/19 |
| Bug-4 | `pingpong_buf.v` + `npu_top.v` | INT8 丢失半数元素 | PPBuf OUT_WIDTH=16 导致 PE 只取 [7:0] | PPBuf 改 OUT_WIDTH=8, SUBW=4 | tb_comprehensive 8/8 |
| Bug-5 | `npu_ctrl.v` | DMA r_start 重复触发 | S_WB_WAIT 持续拉 r_start | 只在进入 WB 时单脉冲 r_start | tb_comprehensive 8/8 |
| Bug-6 | `npu_dma.v` | OS 模式 FIFO 空时进 R_WRITE 卡死 | r_start 脉冲到时 FIFO 为空 | r_pending 机制：等 FIFO 非空再写 | tb_multi_rc 13/13 |
| Bug-7 | `npu_dma.v` | m_axi_wdata 第一拍是 stale data | output reg 赋值有 1 cycle 滞后 | 改为 combinational assign from FIFO rd_data | tb_multi_rc 13/13 |
| Bug-8 | `npu_dma.v` | 多次 DMA 写回地址相同（T2/T3 写到 T1 地址） | 双重根因：burst_len=0 + wlast NBA 滞后一拍 | r_burst_len 寄存器 + wlast 改 output wire combinational | tb_multi_rc 13/13 |
| Bug-9 | `npu_top.v` | INT8 WS got=0 | PPBuf rd_data 组合 vs w_fp16_shift 寄存器 1-cycle 延迟 | 新增 w_int8_ready_d / a_int8_ready_d sticky 信号 | tb_array_scale 16/16 |
| Bug-10 | `pe_array.v` | COLS>1 back-to-back col1 偏差 | act_reg 跨运行未清零 | flush cycle 时 `act_reg <= 0` | tb_multi_rc 13/13 |
| Bug-11 | `npu_ctrl.v` | WS 模式无 tile-loop，只能计算单个点积 | FSM 无多 tile 双层循环结构 | 重写 FSM，新增 tile_i/tile_j 计数器 | WS 模式仿真 PASS |
| Bug-12 | `npu_top.v` | OS 模式 weight 广播到所有列，无法计算正确 C[i][j] | 缺少按列路由机制 | 新增 ctrl_target_col 信号，OS 模式 weight 只路由到目标列 | OS 方阵 416/416 |
| Bug-13 | `npu_ctrl.v` | DMA w_len 硬编码为 4（bytes），大矩阵传输截断 | S_IDLE 中 `dma_w_len=16'd4` placeholder | 新增 k_dma_len_w combinational wire（= K × 元素字节数） | OS/WS 全尺寸 PASS |
| Bug-14 | `soc_mem.v` | SoC CPU 每条指令读到 stale data，寄存器值全错，CTRL 写入错误地址（0x02000020 而非 0x02000000） | SRAM 读使用同步寄存器（`always @(posedge clk) rdata <= mem[...]`），但 `ram_ready` 是组合信号 → 同一周期 ready=1 但 rdata 是上一周期的旧值 | 改为 `assign rdata = mem[addr[ADDR_W-1:0]]`（异步组合读） | SoC tb_soc PASS |
| Bug-15 | `dram_model.v` | CPU 通过 LW 读 DRAM 时读到错误数据 | DRAM CPU 读端口使用同步读（同 Bug-14 根因） | 改为 `assign cpu_rdata = mem[cpu_addr[ADDR_W+1:2]]`（异步） | SoC tb_soc PASS |
| Bug-16 | `soc_top.v` | 编译警告：SRAM addr 端口位宽不匹配 | `.addr(mem_addr[21:2])` 是 20-bit，而 soc_mem addr 端口声明 22-bit | 改为 `.addr(mem_addr[23:2])`（22-bit，对齐 4KB/4=1K 字） | SoC 编译无警告 |
| Bug-17 | `assemble_soc_test.py` / `soc_test.S` | CPU 写入 PASS 标记后 testbench 不响应，仿真 timeout | 标记地址 0x0F00 < 0x1000，属于 SRAM 空间；testbench 监视 DRAM 的 mem[960]，永远不更新 | 改标记地址为 0x2000（DRAM 空间），testbench 改监视 mem[2048] | SoC tb_soc PASS |
| Bug-18 | `tb_soc.v` | CPU 执行与预期完全不同的指令序列，从错误 PC 取指 | `$readmemh("soc_test.hex", ...)` 从 vvp 工作目录（sim/）解析，加载 sim/ 下的 10KB 旧 hex，而非 tb/ 下的最新 420B hex | 改为 `$readmemh("../tb/soc_test.hex", ...)`，明确指向 tb 目录 | SoC tb_soc PASS |
| Bug-19 | `pingpong_buf.v` + `npu_ctrl.v` | FP16 模式下 PPBuf 读取数据错误，WS K=1 结果全为 0 | PPBuf 在 FP16 模式下以 16-bit 粒度读取，但数据写入时 32-bit word 的低 16-bit 是 padding，高 16-bit 是实际数据。原代码读取了低 16-bit（padding），导致 PE 接收到 0。同时，WS 模式状态机未正确处理 pipeline 延迟，导致 Result FIFO 写入时机错误 | ① 修改 `pingpong_buf.v`：FP16 模式下读取高 16-bit（`rd_mem[31:16]`）作为第一个 half-word<br>② 修改 `npu_ctrl.v`：WS 模式保持 `pe_en` 到 K+3 周期，确保 pipeline 排空 | FP16 E2E 9/9 PASS |
| Bug-20 | `npu_top.v` FP16 packer | FP16 pe_consume 虚假触发：FP16 模式 `pe_en=1` 时立刻触发一次 consume（phase=0 初始状态），将 w=0,a=0 写入 FIFO，导致 WS/OS 所有非零结果失败 | `pe_consume` 条件为 `!w_ppb_phase && !a_ppb_phase`，初始 phase=0 即满足，未等待真正的 FP16 半字对装配完成 | 改用 phase 1→0 下降沿延迟 1 拍检测：新增 `w_phase_fall_d / a_phase_fall_d` 寄存器，确保 FP16 consume 在"刚完成第 2 字节"后才触发 | FP16 E2E 9/9 PASS |
| Bug-21 | `pe_top.v` + `tb_pe_top.v` | WS 模式 PE 单元测试 7 个用例 FAIL（期望值与实际值有 1.5–2× 偏差）；根因：TB 发出的 flush beat 携带了旧的 a_in 数据被 RTL 额外累加，同时 beat 被 pipeline 重复采样 | **双重根因**：① RTL：WS flush 时会将当前 beat 的 s1_mul 累加到 ws_acc 再输出，而 TB 设计意图是 flush beat = 纯触发，不应有实际计算；② TB：`drive_ws_beat` task 未在 beat 结束后降低 en，导致同一 beat 被 pipeline 在相邻两个 posedge 各采样一次（翻倍效果）；`load_w=1` beat 同理 | ① `pe_top.v`：WS flush 语义改为纯输出——`acc_out <= ws_acc`（不加 s1_mul），ws_acc 清零；② `tb_pe_top.v`：`drive_ws_beat` task 内部加 `@(posedge clk); #1; en=0`；所有 WS `load_w=1` beat 后同行设置 `en=0`；所有 WS flush beat 统一设置 `a_in=0, w_in=0` | PE 单元测试 19/19 PASS |

---

## 13. WS 模式 PE 时序调试专项（已归档，2026-04-10 修复）

> **状态：已修复。PE 单元测试 19/19 PASS，FP16 E2E 9/9 PASS。**  
> 本节保留历史分析过程，供同类问题参考。

### 13.1 现象描述（修复前）

WS 模式下 PE 单元测试 7 个用例失败，典型模式为：
- 期望值的 **1.5× 或 2×** 出现在实际输出
- flush beat 之前最后一个 beat 的乘积被"意外重复计入"

### 13.2 根因分析

**根因 A：RTL flush 语义**

`pe_top.v` WS 模式下，flush 时会把当前 beat 的 `s1_mul` 累加进 `ws_acc`：
```verilog
// 修复前（错误）
acc_out <= ws_acc + s1_mul;
```
TB 设计意图是 flush beat 为**纯触发**（a_in=0 表达"无数据"），RTL 却视之为"最后一个数据 beat"——两者语义不匹配。

**根因 B：TB drive_ws_beat 时序**

原 `drive_ws_beat` task 结束后未立即降低 `en`，导致：
- Task 在 posedge C0 后设置 `en=1, a_in=v`
- 调用方下一行 `@(posedge clk); #1; en=0` 才在 C1 posedge 后设置 en=0
- **C0 和 C1 的 posedge 都采样到 en=1**，同一 beat 被 pipeline 计算两次

**根因 C：load_w beat 未及时降低 en**

`load_w=1` beat 完成后，下一行只清零了 `load_w`，而 `en` 仍为高，导致首个数据 beat 被重复采样。

### 13.3 修复方案

| 修改点 | 修改内容 |
|--------|----------|
| `pe_top.v` WS flush | flush 时 `acc_out <= ws_acc`（直接输出，不加 s1_mul），ws_acc 清零 |
| `tb_pe_top.v` drive_ws_beat | task 末尾加 `@(posedge clk); #1; en=0`，每个 beat 精确 1 周期 |
| `tb_pe_top.v` load_w beat | `@(posedge clk); #1; en=0; load_w=0`（同行降低 en） |
| `tb_pe_top.v` flush beat | 所有 WS flush beat 统一 `a_in=16'd0; w_in=16'd0`（明确纯 flush）|

### 13.4 验证结果

修复后 PE 单元测试：**PASS=19 / FAIL=0**，全量回归无回归。

---

## 14. GTKWave 常用信号速查

### 全局控制

| 信号路径 | 说明 |
|---------|------|
| `tb.u_npu.u_ctrl.state` | 控制器 FSM（关键！）|
| `tb.u_npu.u_ctrl.status_done` | 完成标志 |
| `tb.u_npu.u_ctrl.status_busy` | 忙碌标志 |
| `tb.u_npu.pe_en` | PE 阵列使能 |
| `tb.u_npu.pe_flush` | PE flush 脉冲 |

### DMA 通道

| 信号路径 | 说明 |
|---------|------|
| `tb.u_npu.u_dma.dma_state` | DMA FSM（0=IDLE,1=W_READ,2=A_READ,3=R_WRITE）|
| `tb.m_axi_awaddr` | DMA 写地址 |
| `tb.m_axi_awvalid` | AW 有效（DMA 发起写）|
| `tb.m_axi_wlast` | 最后一拍（应在 last beat 高）|
| `tb.m_axi_awlen` | burst 长度（awlen=N-1 for N beats）|
| `tb.u_npu.u_dma.byte_cnt` | 当前已传字节数 |
| `tb.u_npu.u_dma.r_len_latch` | 锁存的目标长度 |
| `tb.u_npu.u_dma.aw_sent` | AW handshake 已完成 |

### PE 阵列（COLS=2 示例）

| 信号路径 | 说明 |
|---------|------|
| `tb.u_npu.pe_array_valid` | PE 输出有效 |
| `tb.u_npu.pe_array_result[31:0]` | col0 结果 |
| `tb.u_npu.pe_array_result[63:32]` | col1 结果 |
| `tb.u_npu.pe_valid_q` | 延迟 1 拍的 valid |
| `tb.u_npu.r_fifo_wr_en` | FIFO 写使能 |
| `tb.u_npu.r_fifo_din` | FIFO 写数据 |
| `tb.u_npu.gen_r_ser.ser_active` | 多列序列化进行中 |
| `tb.u_npu.gen_r_ser.ser_col` | 当前序列化列号 |

### 单 PE 内部（ROWS=2, COLS=2 示例）

| 信号路径 | 说明 |
|---------|------|
| `tb.u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.os_acc` | PE[0][0] 累加器 |
| `tb.u_npu.u_pe_array.gen_row[0].gen_col[1].u_pe.s0_a` | PE[0][1] 激活输入（应有 1 拍延迟）|
| `tb.u_npu.u_pe_array.gen_row[1].gen_col[0].u_pe.acc_out` | PE[1][0] 输出（最终结果 col0）|

---

## 快速参考：计算期望值

### OS 模式 COLS=2 期望值计算模板

```python
# Python 参考计算（INT8）
def calc_expected(W, A, K):
    # col0: 完整点积
    col0 = sum(W[k] * A[k] for k in range(K))

    # col1: activation 右移 1 位（首位填 0）
    A_shifted = [0] + A[:K-1]
    col1 = sum(W[k] * A_shifted[k] for k in range(K))

    return col0, col1

# 示例：T1
W = [1, 2, 3, 4]; A = [5, 6, 7, 8]
print(calc_expected(W, A, 4))  # → (70, 56)

# 示例：T3
W = [127, -128, 1, 0]; A = [127, -128, 1, 0]
print(calc_expected(W, A, 4))  # → (32514, -16384)
```

---

*本文档应随每次 RTL 修改同步更新。如发现新 Bug，请在第10节补充记录。*
