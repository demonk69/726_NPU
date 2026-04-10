# NPU_prj — 系统架构文档

> 最后更新：2026-04-10  
> 文档状态：已按当前 RTL 与 SoC 联调结果同步

---

## 1. 项目概览

NPU_prj 是一个面向 SoC 集成的 Verilog NPU 原型，实现了：

- **4×4 可参数化 PE 阵列**
- **INT8 / FP16** 两条已验证数据通路
- **AXI4-Lite 配置接口** + **AXI4 DMA 数据通路**
- **WS（Weight-Stationary）/ OS（Output-Stationary）** 两种计算模式
- **PicoRV32 + SRAM + DRAM + NPU** 的 SoC 集成验证

### 核心设计概念：Tile-Based 计算

本 NPU 采用 **tile-based 计算模型**，这是理解其行为的关键：

1. **什么是 Tile？**
   - 在矩阵乘法 `C[M×N] = A[M×K] × B[K×N]` 中，每个 **tile** 对应一个输出元素 `C[i][j]` 的计算
   - 每个 tile 需要完成完整的 K 维点积累加
   - 这种设计将大矩阵分解为**微小的计算单元**，逐个处理

2. **Tile 的物理意义：**
   - 物理 PE 阵列是 4×4，但**每个 tile 只使用部分 PE**
   - OS 模式：`target_col = j % COLS`，权重只送到目标列
   - WS 模式：权重广播到整个列，部分和沿列传递
   - 一个 tile ≠ 一个 PE，而是一个完整的 K 维累加任务

3. **Tile-Loop 调度：**
   - 控制器按 `M×N` 次循环推进，每次计算一个 `C[i][j]`
   - 每完成一个 tile，更新 `i` 和 `j`，计算新的地址

这种设计实现了**资源复用与内存友好**，但不同于传统"一次吐出整块输出矩阵"的并行架构。

### 当前实现边界

这份文档描述的是**当前代码实际行为**，不是早期设计目标：

- 当前 RTL **实际稳定支持 INT8 与 FP16**。
- `CTRL[3:2]` 在硬件内部只区分"INT8"与"非 INT8"。软件上建议：
  - `00`：INT8
  - `10`：FP16
- `ARR_CFG`、`CLK_DIV`、`CG_EN` 已有寄存器接口，但**尚未完整闭环到阵列规模裁剪/真实门控时钟路径**，更多属于保留能力与观测接口。
- 当前 NPU 的主工作流是 **tile-loop 矩阵乘法**：每次 tile 计算一个 `C[i][j]`，而不是一次性在 4×4 阵列上直接吐出整块 4×4 输出矩阵。

### 当前验证状态

| 项目 | 状态 |
|---|---|
| PE 单元 | ✅ 19/19 PASS |
| FP16 乘法 | ✅ 44/44 PASS |
| FP16 加法 | ✅ 20/20 PASS |
| NPU 综合测试 | ✅ 8/8 PASS |
| 阵列规模验证 | ✅ 16/16 PASS |
| 多行多列综合 | ✅ 13/13 PASS |
| FP16 端到端 | ✅ 9/9 PASS |
| SoC 集成 | ✅ 287 cycles PASS |
| 全量回归 | ✅ 903 PASS / 0 FAIL |

### WS flush 语义说明（2026-04-10 确立）

WS 模式下，`flush` 信号采用**纯触发输出**语义：
- flush beat 本身**不参与累加计算**，直接将 `ws_acc` 输出并清零
- 驱动 flush beat 时，`a_in` 和 `w_in` 必须设为 0（纯 flush 标记，无实际数据）
- 这与 OS 模式不同：OS 模式的 flush beat 是最后一个数据 beat，携带真实计算数据

此语义确保：K 个数据 beat 完整累加后，第 K+1 个 flush beat 无副作用地触发输出。

---

## 2. 顶层系统结构

```text
                +--------------------------------------+
                |              soc_top                 |
                |                                      |
                |  +-----------+                       |
CPU firmware -->|  | PicoRV32  |                       |
                |  +-----+-----+                       |
                |        | mem_if                      |
                |        v                             |
                |  +-----------+   +---------------+  |
                |  |  soc_mem  |   |  dram_model   |  |
                |  |  SRAM 4KB |   |  dual-port    |  |
                |  +-----------+   +-------+-------+  |
                |                      ^     ^         |
                |                      |     |         |
                |  +-------------------+     |         |
                |  | axi_lite_bridge         |         |
                |  +-----------+-------------+         |
                |              | AXI4-Lite             |
                |              v                       |
                |         +----------+                 |
                |         | npu_top  |                 |
                |         +----+-----+                 |
                |              | AXI4 Master           |
                +--------------+-----------------------+
                               |
                               v
                            DRAM data
```

### 分层职责

- **PicoRV32**：负责固件执行、初始化 DRAM、配置 NPU、轮询 done/处理中断。
- **soc_mem**：CPU 指令与数据 SRAM。
- **dram_model**：CPU 与 NPU DMA 共享的行为级 DRAM。
- **axi_lite_bridge**：把 PicoRV32 的 `mem_valid/mem_ready` 风格访存转换成 AXI4-Lite。
- **npu_top**：NPU 顶层，内部集成寄存器、控制器、DMA、PPBuf、PE 阵列、结果 FIFO、电源管理占位模块。

---

## 3. NPU 内部结构

```text
AXI4-Lite
   |
   v
+--------------+
| npu_axi_lite |
+------+-------+
       |
       v
+--------------+        +-------------+
|   npu_ctrl   |------->|  npu_dma    |---- AXI4 ----> DRAM
+------+-------+        +------+------+ 
       |                         |
       | swap/clear              | write 32-bit words
       v                         v
+-------------+          +-------------+
| pingpong_buf|          | pingpong_buf|
|   (Weight)  |          | (Activation)|
+------+------+          +------+------+
       |                        |
       +-----------+------------+
                   v
               +-------+
               |pe_array|
               +---+---+
                   |
                   v
               +-------+
               | FIFO  |
               +-------+
```

### 关键点

1. **配置与状态**由 `npu_axi_lite` 暴露给 CPU。
2. **npu_ctrl** 负责 tile-loop 调度、DMA 启动、PE flush、下一 tile 地址计算。
   - 维护 `tile_i` 和 `tile_j` 计数器（0≤i<M, 0≤j<N）
   - 计算每个 tile 的 DRAM 地址：`W_ADDR + j×K×元素字节数`，`A_ADDR + i×K×元素字节数`
   - OS 模式：计算 `target_col = j % COLS`，用于权重路由
3. **npu_dma** 负责：
   - 从 DRAM 读权重（B 矩阵的第 j 列）
   - 从 DRAM 读激活（A 矩阵的第 i 行）
   - 从结果 FIFO 取结果写回 DRAM（C[i][j]）
4. **pingpong_buf** 把"DMA 读入"和"PE 消费"解耦，支持 INT8（4 元素/字）和 FP16（2 元素/字）两种打包格式。
5. **pe_array** 是参数化阵列，但当前 tile-loop 调度按单个输出元素推进，每个 tile 只激活部分 PE：
   - OS 模式：激活整行，权重只送到 `target_col` 列
   - WS 模式：权重广播到整列，激活按行广播

---

## 4. 当前数据流约定

## 4.1 DRAM 布局

当前代码与测试默认使用如下布局：

- **`W_ADDR`**：矩阵 `B[K×N]`，**列主序**
- **`A_ADDR`**：矩阵 `A[M×K]`，**行主序**
- **`R_ADDR`**：矩阵 `C[M×N]`，**行主序**

### 元素打包规则

| 模式 | DRAM 中每个 32-bit 字包含 | 备注 |
|---|---|---|
| INT8 | 4 个 8-bit 元素 | PPBuf 以 `OUT_WIDTH=8, SUBW=4` 拆出 |
| FP16 | 2 个 16-bit 元素 | 在 `npu_top` 中按低字节/高字节重新拼成 16-bit FP16 |
| Result | 1 个 32-bit 结果 | INT8 为 32-bit 整数累加；FP16 路径输出 32-bit 累加结果 |

## 4.2 Tile-loop 调度

### 4.2.1 Tile 的生命周期

一个 tile 代表一个输出元素 `C[i][j]` 的完整计算过程。当前控制器的外层循环是：

```text
for i in [0 .. M-1]
  for j in [0 .. N-1]
    // 开始一个 tile 的计算
    启动 W/A DMA，读入 B[:,j] 和 A[i,:]
    PE 消费 K 个元素对
    flush → 结果写入 FIFO
    DMA 读 FIFO，写回 DRAM
    更新 i/j 地址，准备下一个 tile
```

### 4.2.2 Tile 的内部阶段

每个 tile 经历以下阶段：

| 阶段 | 控制器状态 | 作用 | 时长 |
|---|---|---|---|
| **准备** | S_LOAD | 启动 W/A DMA，填充 PPBuf | ~K 拍 + DMA 延迟 |
| **对齐** | S_PRELOAD | 给 PE 流水线对齐一拍 | 1 拍 |
| **计算** | S_COMPUTE | PE 流水线全速运行，完成 K 次 MAC | K-1 拍 |
| **排空** | S_DRAIN | 保持 flush，等待流水线清空 | 3 拍（PE 流水线深度） |
| **回写** | S_WRITE_BACK → S_WB_WAIT | 结果 FIFO → DMA → DRAM | ~1 + DMA 延迟 |
| **切换** | S_NEXT_TILE | 计算下一个 tile 的 i/j 和地址 | 1 拍 |

### 4.2.3 Tile 的物理映射

虽然 PE 阵列是 4×4，但**每个 tile 只使用部分资源**：

**OS 模式（每个 tile 计算一个输出元素）：**
```
Tile C[i][j]:
  - 激活广播：整行 PE (PE[i][*]) 接收 A[i,:]
  - 权重路由：只送到 target_col = j % COLS 列
  - 有效 PE：仅第 i 行 × 第 target_col 列的一个 PE 产生有效累加
```

**WS 模式（每个 tile 计算一列的部分和）：**
```
Tile C[i][j]:
  - 权重广播：整列 PE (PE[*][j%COLS]) 接收 B[:,j]
  - 激活路由：按行广播 A[i,:]
  - 部分和：沿列方向传递累加，最后一个 PE 产生最终结果
```

### 4.2.4 Tile 地址计算

控制器维护：
- `tile_i` (0≤i<M)：当前输出的行索引
- `tile_j` (0≤j<N)：当前输出的列索引

每个 tile 的 DRAM 地址：
```
权重地址 = W_ADDR + j × K × 元素字节数
激活地址 = A_ADDR + i × K × 元素字节数
结果地址 = R_ADDR + (i×N + j) × 4
```

其中**元素字节数**：
- INT8：1 字节（但 DRAM 打包为 4 元素/32-bit 字）
- FP16：2 字节（DRAM 打包为 2 元素/32-bit 字）

### OS 模式（Output-Stationary）

- **激活广播**：`A[i,:]` 在行内广播，所有 `PE[i][*]` 接收相同激活
- **权重路由**：`B[:,j]` 只送到 `target_col = j % COLS` 列
- **累加位置**：每个 PE 内部累加自己的 `os_acc`，只有 `target_col` 列的 PE 产生有效累加
- **Tile 输出**：每个 tile 对应一个输出元素 `C[i][j]`，由 `PE[i][target_col]` 产生
- **物理资源利用率**：低（每 tile 只激活一行中的一列 PE）

### WS 模式（Weight-Stationary）

- **权重广播**：`B[:,j]` 在列内广播，所有 `PE[*][target_col]` 接收相同权重
- **激活路由**：`A[i,:]` 按行广播
- **部分和传递**：累加值沿列方向传递（`pe_col_sum`），最终由最下方 PE 产生结果
- **Tile 输出**：每个 tile 对应一个输出元素 `C[i][j]`，由最下方 `PE[ROWS-1][target_col]` 产生
- **物理资源利用率**：中（每 tile 激活一整列 PE，但只有底部 PE 产生最终输出）

---

## 5. 控制寄存器映射

基地址：**`0x0200_0000`**

| 偏移 | 名称 | R/W | 位定义 / 说明 |
|---:|---|:---:|---|
| 0x00 | CTRL | RW | `[0] start`, `[1] abort`, `[3:2] mode`, `[5:4] stat_mode` |
| 0x04 | STATUS | RO | `[0] busy`, `[1] done` |
| 0x08 | INT_EN | RW | `[0]` 中断使能 |
| 0x0C | INT_CLR / PENDING | W1C / RO | 写 `1` 清 pending；读回当前 `int_pending` |
| 0x10 | M_DIM | RW | M 维度 |
| 0x14 | N_DIM | RW | N 维度 |
| 0x18 | K_DIM | RW | K 维度 |
| 0x20 | W_ADDR | RW | 权重基地址 |
| 0x24 | A_ADDR | RW | 激活基地址 |
| 0x28 | R_ADDR | RW | 结果基地址 |
| 0x30 | ARR_CFG | RW | `[3:0] rows, [7:4] cols`，当前主要作为保留配置 |
| 0x34 | CLK_DIV | RW | `[2:0]` 分频选择，当前主要接到 `npu_power` 观测路径 |
| 0x38 | CG_EN | RW | 时钟门控使能位，当前未完整反馈到主数据通路 |

### 软件推荐编码

| 功能 | CTRL 值 | 含义 |
|---|---:|---|
| INT8 + WS 启动 | `0x01` | `start=1, mode=00, stat=00` |
| INT8 + OS 启动 | `0x11` | `start=1, mode=00, stat=01` |
| FP16 + WS 启动 | `0x09` | 建议软件使用 `mode=10` 表示 FP16 |
| FP16 + OS 启动 | `0x19` | 建议软件使用 `mode=10, stat=01` |
| 清零 CTRL | `0x00` | 计算结束后建议显式清零 |

> 注意：当前硬件内部 `mode != 00` 都会走 FP16 通路，因此软件不要依赖 `01/10/11` 的细粒度区分。

---

## 6. 控制器状态机

当前 `npu_ctrl` 的 FSM 为：

```text
S_IDLE
  -> S_LOAD
  -> S_PRELOAD
  -> S_COMPUTE
  -> S_DRAIN
  -> S_WRITE_BACK
  -> S_WB_WAIT
  -> S_NEXT_TILE
  -> (S_LOAD for next tile / S_DONE)
  -> S_IDLE
```

### 各状态作用（按 Tile 生命周期）

| 状态 | 作用 | 对应 Tile 阶段 |
|---|---|---|
| S_IDLE | 等待 `cfg_start` 上升沿 | 空闲 |
| S_LOAD | 启动 W/A DMA，把本 tile 数据装入 PPBuf | 准备阶段 |
| S_PRELOAD | 给 PE 输入一个对齐周期 | 对齐阶段 |
| S_COMPUTE | PE 正常消费 K 个元素 | 计算阶段 |
| S_DRAIN | 保持 `flush` 若干拍，等待流水线结果完全推出 | 排空阶段 |
| S_WRITE_BACK | 脉冲 `dma_r_start` | 回写阶段 |
| S_WB_WAIT | 等待结果 DMA 完成 | 回写等待 |
| S_NEXT_TILE | 计算下一 tile 的 `i/j`、DMA 地址与 `target_col` | 切换阶段 |
| S_DONE | 置位 `done` / `irq` | 完成所有 tile |

### Tile 状态转移细节

1. **S_LOAD → S_PRELOAD**：当两个 PPBuf 都非空时切换，表示数据已准备好
2. **S_COMPUTE**：持续 K 个周期，每个周期消费一对 W/A 元素
3. **S_DRAIN**：固定 3 拍，对应 PE 3-stage 流水线深度
4. **S_NEXT_TILE**：关键逻辑：
   - `tile_j++`，如果 `tile_j == N` 则 `tile_j=0, tile_i++`
   - 计算新地址：`W_ADDR += K×元素字节数`（列步进）
   - 计算 `target_col = tile_j % COLS`（OS 模式）
   - 如果 `tile_i == M`，进入 S_DONE，否则回到 S_LOAD

### 关键实现细节

- `cfg_start` 使用**上升沿检测**，避免 CTRL 保持为 1 时重复触发。
- `done` 是**sticky** 信号；当软件把 `CTRL.start` 清零后才回落。
- `target_col` 仅在 **OS 模式**下更新，用于把权重路由到正确列。
- `k_dma_len_w` 会根据 INT8 / FP16 自动算 DMA 字节数。

> ✅ **已修复（2026-04-08）**：`target_col` 已由 1-bit 扩展为 `$clog2(COLS > 1 ? COLS : 2)` 位，`npu_top.v` 中 `ctrl_target_col` 同步扩宽，COLS=4 时不再截断。

---

## 7. SoC 地址空间与时序约束

## 7.1 地址空间

以默认参数 `MEM_WORDS=1024`、`DRAM_WORDS=15360` 为例：

| 地址范围 | 区域 | 说明 |
|---|---|---|
| `0x0000_0000` ~ `0x0000_0FFF` | SRAM | 4KB，CPU 指令 + 数据 |
| `0x0000_1000` ~ `0x0000_FFFF` | DRAM | 约 60KB，CPU 与 NPU DMA 共用 |
| `0x0200_0000` ~ `0x0200_0038` | NPU Reg | AXI4-Lite 配置空间 |

> `0x0F00 < 0x1000`，因此它属于 **SRAM**，不是 DRAM。这是 SoC 联调时已踩过的坑。

## 7.2 PicoRV32 读时序要求

PicoRV32 要求：

- `mem_ready` 与 `mem_rdata` **同周期有效**

因此当前 RTL 中：

- `soc_mem.rdata` 是**组合读**
- `dram_model.cpu_rdata` 也是**组合读**

如果把它们改成同步寄存器读，CPU 会读到上一拍 stale data，导致：

- 指令执行错乱
- 寄存器值错误
- NPU 配置地址写偏
- poll_loop 永不结束

## 7.3 AXI-Lite 桥接时序

`npu_axi_lite` 的写通道不是 AW/W 同拍接受，而是：

1. 先接收 AW
2. 下一拍再接收 W

因此 `axi_lite_bridge` 采用两段式写 FSM：

```text
S_IDLE -> S_WRITE_AW -> S_WRITE_W -> S_IDLE
```

这也是 SoC 配置路径正确工作的前提。

---

## 8. 当前实现中"已接入"与"未完全接入"的能力

| 能力 | 状态 | 说明 |
|---|---|---|
| INT8 数据通路 | ✅ 已接入 | 已完整验证 |
| FP16 数据通路 | ✅ 已接入 | 已完整验证 |
| WS / OS 模式 | ✅ 已接入 | 已验证 |
| tile-loop 地址调度 | ✅ 已接入 | 当前核心执行模型 |
| SoC 集成 | ✅ 已接入 | PicoRV32 固件驱动通过 |
| 阵列尺寸动态裁剪 | ⚠️ 部分保留 | `ARR_CFG` 已暴露，但当前主流程未完整消费 |
| DFS / 时钟门控 | ❌ 接口存在，输出悬空 | `npu_power` 已实例化，但 `npu_clk`、`row_clk_gated`、`col_clk_gated` 三路输出在 `npu_top` 中全部悬空（`.npu_clk()` 等）。PE 阵列使用 `sys_clk` 直连，电源管理未实际生效 |
| 经典整块矩阵阵列并行出块 | ❌ 当前非主路径 | 当前以 tile-loop 单输出元素推进 |
| AXI 多拍 burst | ✅ 读/写均已支持 | 读通道升级为多拍 INCR burst（`calc_arlen()` 动态计算 `arlen`，2026-04-08），写通道 `awlen` 按结果字数计算，两路均已生效 |

### 实际吞吐估算

### 8.1 Tile 的性能分析

当前 tile-loop 架构下，单次 `C[i][j]`（一个 tile）的计算路径为：

```
S_LOAD (~K 拍 DMA) + S_PRELOAD (1拍) + S_COMPUTE (K-1拍) + S_DRAIN (3拍) + S_WRITE_BACK/WB_WAIT (~2拍) + S_NEXT_TILE (1拍)
```

总周期数 ≈ K（DMA读） + K（计算） + 固定开销（~7拍）

#### 性能瓶颈分析

**计算密度低的原因：**
1. **Tile 粒度太小**：每个 tile 只计算一个输出元素，但需要完整的 K 维数据
2. **PE 利用率低**：
   - OS 模式：每 tile 只激活一行中的一列 PE（1/COLS 利用率）
   - WS 模式：每 tile 激活一整列 PE，但只有底部 PE 产生输出
3. **DMA 开销大**：每个 tile 都需要重新发起 DMA 读取

#### 示例计算

以 INT8，K=4，4×4 矩阵（M=N=4）为例：
- **总 MAC 次数** = M×N×K = 4×4×4 = 64 次（不是 128 次）
- **总周期数** ≈ M×N×(2K+7) = 16×(8+7) = 240 周期
- **有效 MAC/cycle** = 64/240 ≈ 0.27
- **理论峰值**（4×4 阵列每拍 16 MAC）：16 GOPS @ 500 MHz
- **实际吞吐** ≈ 0.27×16 = 4.3 GOPS（理论峰值的 27%）

#### 与并行架构对比

| 架构 | 特点 | 适用场景 |
|---|---|---|
| **Tile-Loop**（当前） | 逐个输出元素计算，资源复用，内存友好 | 小矩阵，内存带宽受限 |
| **全并行** | 一次性计算整块输出矩阵，资源消耗大 | 大矩阵，计算密集 |
| **混合并行** | 多个 tile 并发，平衡资源与带宽 | 通用场景 |

当前实现偏向**内存友好**而非**计算密集**，适合嵌入式 SoC 场景。

### 8.2 优化方向

1. **增大 Tile 粒度**：每个 tile 计算多个输出元素（如 4×4 块）
2. **并发多个 Tile**：流水线化 DMA 与计算，重叠不同 tile 的阶段
3. **数据重用**：在 PPBuf 中缓存可复用的数据（如权重共享）
4. **DMA 优化**：使用更大的 burst 长度，减少 DMA 启动开销

---

## 9. 与文档配套阅读顺序

建议按下面顺序阅读：

1. **`README.md`**：项目总览与快速入口
2. **`doc/user_manual.md`**：如何使用、如何跑仿真、如何写固件
3. **`doc/simulation_guide.md`**：仿真脚本与测试项说明
4. **`doc/module_reference.md`**：模块级细节
5. **`doc/npu_debug_checklist.md`**：定位 BUG 与排障

---

## 10. 总结

当前 NPU_prj 已从"早期目标架构描述"收敛为一个**可回归、可联调、可由 PicoRV32 固件驱动的 tile-loop NPU 实现**；理解它时，最重要的是把握三件事：

1. **数据布局固定：B 列主序，A 行主序，C 行主序**
2. **控制模型是按 `C[i][j]` 单点推进的 tile-loop**
3. **SoC 侧 CPU 读口必须是组合读，AXI-Lite 写必须按 AW→W 两拍完成**
