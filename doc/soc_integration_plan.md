# SoC 集成方案：PicoRV32 + NPU（含 Ping-Pong Buffer）

## 1. 系统总览

```
                          ┌──────────────────────────────────────────────────┐
                          │                   SoC (soc_top.v)               │
                          │                                                  │
                          │  ┌────────────────────────────────────────────┐  │
  CPU                      │  │         地址译码 + 数据选择               │  │
  (PicoRV32)               │  │  addr[31:24] >= 0x01 → iomem_valid       │  │
  mem_valid/addr/wdata     │  └───────┬──────────┬──────────┬─────────────┘  │
  mem_ready/rdata          │          │          │          │                │
  irq[7] ← NPU done        │          ▼          ▼          ▼                │
                          │  ┌───────────┐ ┌─────────┐ ┌──────────────┐    │
                          │  │   SRAM    │ │  DRAM   │ │  NPU Reg     │    │
                          │  │ 0x0000   │ │ 0x0100  │ │ 0x0200_0000 │    │
                          │  │ (4KB)    │ │-0xFFFF  │ │ (AXI-Lite    │    │
                          │  │          │ │ (60KB)  │ │  → 内部桥接) │    │
                          │  └───────────┘ └────┬─────┘ └──────┬───────┘    │
                          │                      │              │            │
                          │                      │    ┌─────────┴─────────┐  │
                          │                      │    │     npu_top       │  │
                          │                      │    │ ┌───────────────┐ │  │
                          │                      └────┤ │ DMA (AXI4     │ │  │
                          │                           │ │  Master)     │ │  │
                          │                           │ └───┬───────┬───┘ │  │
                          │                           │     │       │     │  │
                          │                           │     ▼       ▼     │  │
                          │                           │  Weight  Activ   │  │
                          │                           │  PPBuf   PPBuf   │  │
                          │                           │  (BufA/  (BufA/  │  │
                          │                           │   BufB)   BufB)  │  │
                          │                           │     │       │     │  │
                          │                           │     ▼       ▼     │  │
                          │                           │  ┌─────────────┐  │  │
                          │                           │  │  PE Array   │  │  │
                          │                           │  │  (4×4)      │  │  │
                          │                           │  └──────┬──────┘  │  │
                          │                           │         │         │  │
                          │                           │     Result        │  │
                          │                           │     PPBuf         │  │
                          │                           │         │         │  │
                          │                           │         ▼         │  │
                          │                           └─────────►DRAM     │  │
                          └──────────────────────────────────────────────────┘
```

## 2. 地址映射

| 地址范围 | 大小 | 设备 | 访问者 | 说明 |
|---|---|---|---|---|
| `0x0000_0000 - 0x0000_0FFF` | 4KB | SRAM | CPU only | 指令 + 数据存储 |
| `0x0000_0100 - 0x0000_FFFF` | ~60KB | DRAM | CPU + NPU DMA | NPU 输入数据（权重/激活）+ 输出结果 |
| `0x0200_0000 - 0x0200_003F` | 64B | NPU 寄存器 | CPU only | NPU 配置（通过 AXI-Lite 桥接） |

**注**：SRAM 地址 `0x0000_0000` ~ `0x0000_0FFF`，DRAM 地址从 `0x0000_0100` 开始。
CPU 固件从 `0x0000_0000` 开始执行。DRAM 的起始地址需要 256 字节对齐（简化地址计算）。

## 3. Ping-Pong Buffer 设计

### 3.1 动机

当前的 DMA→PE 数据流是串行的：
```
DMA 读 DRAM → 写入 FIFO → PE 消费 FIFO → FIFO 空 → DMA 读下一块
```
问题：DMA 读和 PE 计算不能重叠，PE 经常等数据（FIFO 空闲）。

**Ping-Pong Buffer 让 DMA 和 PE 并行工作**：
```
Clock 1-32:   DMA → BufA[0..31]     PE 空闲（BufA 填充中）
Clock 33-64:  DMA → BufB[0..31]     PE ← BufA[0..31]  ← 重叠！
Clock 65-96:  DMA → BufA[32..63]    PE ← BufB[0..31]  ← 重叠！
...
```

### 3.2 模块接口

```verilog
module pingpong_buf #(
    parameter DATA_W    = 16,     // 数据位宽
    parameter DEPTH     = 32,     // 每个缓冲区深度（必须 2 的幂）
    parameter OUT_WIDTH = 16      // 输出位宽（PE 需要的宽度）
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // 写端口（DMA 侧：AXI4 读回的数据写入）
    input  wire                  wr_en,
    input  wire [DATA_W-1:0]    wr_data,

    // 读端口（PE 侧：PE 消费数据）
    input  wire                  rd_en,
    output wire [OUT_WIDTH-1:0]  rd_data,

    // 控制信号
    input  wire                  start,     // DMA 传输开始
    input  wire                  swap,      // 切换缓冲区
    input  wire                  clear,     // 清空当前缓冲区指针

    // 状态信号
    output wire                  buf_empty, // 当前读缓冲区为空
    output wire                  buf_full,  // 当前写缓冲区为满
    output wire                  fill_count, // 当前读缓冲区填充数量
    output wire                  active_buf  // 0=BufA被PE读, 1=BufB被PE读
);
```

### 3.3 双缓冲区结构

```
         wr_en/wr_data (DMA侧)
              │
              ▼
    ┌─────────────────────┐
    │   Buffer Select     │
    │   (write_ptr)       │
    └──────┬──────┬───────┘
           │      │
     ┌─────▼──┐ ┌▼──────┐
     │ Buf A  │ │ Buf B │     每个 Buf = DEPTH × DATA_W
     │[0..31] │ │[0..31]│     简单的双端口 RAM
     └────┬───┘ └───┬────┘
          │         │
    ┌─────▼─────────▼─────┐
    │   Buffer Select     │
    │   (read_ptr)        │
    └──────────┬──────────┘
               │
               ▼
         rd_data (PE侧)
```

### 3.4 工作流程（以 Weight 通道为例）

```
时间线  →  ──────────────────────────────────────────────►

DMA:    ┌─ fill BufA ──┐─ fill BufB ──┐─ fill BufA ──┐
                     ↓ swap            ↓ swap
PE:              ┌─ drain BufA ──┐─ drain BufB ──┐─ drain BufA ──┐

时序重叠:         ████████████████████████████████████████
                 ← DMA fill BufB  →← PE drain BufA →
                 完全重叠，无空闲等待
```

### 3.5 Buffer 深度选择

| 参数 | 值 | 说明 |
|---|---|---|
| DEPTH | 32 | 每个 Ping-Pong 缓冲区 32 个数据字 |
| 总存储 | 64 × DATA_W | 两组各 32 |
| 交换阈值 | DMA 填满一组后自动切换 | 由控制器 `swap` 信号触发 |

### 3.6 DMA 控制器改造

当前 `npu_dma.v` 使用单 FIFO，需要改为：

```
当前:  DMA AXI Read → sync_fifo → PE（串行）
改为:  DMA AXI Read → pingpong_buf → PE（并行）
```

DMA 侧改动：
- 写入 pingpong_buf 的 `wr_en/wr_data`
- 当一个 buffer 写满 DEPTH 个数据后，拉高 `swap` 信号切换
- 传输结束时，等待 PE 侧消费完当前 buffer

PE 侧改动：
- 通过 `rd_en/rd_data` 从 pingpong_buf 读数据
- `buf_empty` 信号指示当前 buffer 已消费完
- 消费完后，`buf_empty` 指示 DMA 侧可以写入该 buffer

### 3.7 控制器 FSM 改造（npu_ctrl.v）

当前 FSM：
```
S_IDLE → S_LOAD_W (等dma_w_done) → S_LOAD_A (等dma_a_done) → S_COMPUTE → S_DRAIN → S_WRITE_BACK → S_DONE
```

改为支持 Ping-Pong 的流水化 FSM：
```
S_IDLE
  → S_LOAD_W_AND_A (DMA 同时加载 W 和 A 到各自的 PPBuf)
  → S_COMPUTE (PE 开始消费，DMA 持续填充 PPBuf 的另一半)
  → S_DRAIN (PE 消费 PPBuf 残余，DMA 已停止)
  → S_WRITE_BACK (结果写回 DRAM)
  → S_DONE
```

关键变化：
- **Weight 和 Activation 可以同时加载**（DMA 已有 W_READ/A_READ 两个通道）
- **加载与计算重叠**：PPBuf 的 DMA 侧和 PE 侧可以同时工作
- 不再需要等 `dma_w_done` 才能启动 `dma_a_start`

## 4. 各模块详细设计

### 4.1 新增/修改文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `rtl/buf/pingpong_buf.v` | **新增** | Ping-Pong 双缓冲区模块 |
| `rtl/buf/ppb_wrapper.v` | **新增** | NPU 专用的 PPBuf 包装（处理位宽适配） |
| `rtl/axi/npu_dma.v` | **修改** | DMA 写端对接 PPBuf，增加 swap 控制 |
| `rtl/ctrl/npu_ctrl.v` | **修改** | FSM 支持流水化加载+计算 |
| `rtl/top/npu_top.v` | **修改** | 集成 PPBuf 替换单 FIFO |
| `rtl/soc/soc_top.v` | **新增** | SoC 顶层集成 |
| `rtl/soc/axi_lite_bridge.v` | **新增** | PicoRV32 iomem → NPU AXI-Lite 桥接 |
| `rtl/soc/soc_mem.v` | **新增** | SRAM 模块（参考 picosoc_mem） |
| `rtl/soc/dram_model.v` | **新增** | 双接口 DRAM 行为模型 |
| `tb/tb_soc.v` | **新增** | SoC 系统验证测试平台 |
| `tb/soc_test.S` | **新增** | RISC-V 汇编测试固件 |
| `scripts/run_soc_sim.ps1` | **新增** | SoC 仿真脚本 |

### 4.2 pingpong_buf.v 详细设计

```verilog
module pingpong_buf #(
    parameter DATA_W    = 16,
    parameter DEPTH     = 32,
    parameter OUT_WIDTH = 16
)(
    input  wire                  clk,
    input  wire                  rst_n,

    // 写端口 (DMA 侧)
    input  wire                  wr_en,
    input  wire [DATA_W-1:0]    wr_data,

    // 读端口 (PE 侧)
    input  wire                  rd_en,
    output wire [OUT_WIDTH-1:0]  rd_data,

    // 控制
    input  wire                  swap,       // 切换活动 buffer
    input  wire                  clear,      // 清零读写指针

    // 状态
    output wire                  buf_empty,
    output wire                  buf_full,
    output wire [$clog2(DEPTH):0] fill_count
);
```

内部实现：
- 两个简单双端口 RAM（`mem_a[0:DEPTH-1]` 和 `mem_b[0:DEPTH-1]`）
- `wr_sel`：写选择器（0=写 BufA，1=写 BufB）
- `rd_sel`：读选择器（与写选择器相反，保证 DMA 和 PE 操作不同 buffer）
- `wr_ptr`：写指针（当前写 buffer 内的位置）
- `rd_ptr`：读指针（当前读 buffer 内的位置）
- `swap` 时 `wr_sel` 和 `rd_sel` 同时翻转，指针归零

### 4.3 npu_dma.v 改造要点

当前 DMA 内部已有 `sync_fifo`，改造为：

```verilog
// 替换 Weight FIFO
// 原: sync_fifo u_w_fifo (...)
// 新:
pingpong_buf #(.DATA_W(ACC_W), .DEPTH(32), .OUT_WIDTH(DATA_W)) u_w_ppb (
    .clk(clk), .rst_n(rst_n),
    .wr_en(w_fifo_wr),
    .wr_data(m_axi_rdata[DATA_W-1:0]),  // 截取低 16 位
    .rd_en(w_fifo_rd_en),
    .rd_data(w_fifo_dout),
    .swap(w_ppb_swap),
    .clear(w_ppb_clear),
    .buf_empty(w_fifo_empty),
    .buf_full(w_fifo_full),
    .fill_count()
);

// Activation FIFO 同理替换
```

新增 DMA 输出：
```verilog
output reg  w_ppb_swap,   // Weight PPBuf 切换信号
output reg  a_ppb_swap,   // Activation PPBuf 切换信号
output reg  w_ppb_clear,  // 清零 Weight PPBuf
output reg  a_ppb_clear,  // 清零 Activation PPBuf
```

swap 触发条件：当写指针达到 DEPTH 时自动拉高一周期。

### 4.4 npu_ctrl.v 改造的 FSM

```
S_IDLE
  │ cfg_start=1
  ▼
S_LOAD_PPBUF
  │ DMA 同时启动 W_READ + A_READ，数据流入各自的 PPBuf
  │ 当 PPBuf 的 "DMA 侧" 填满一组后自动 swap
  │ 当 PPBuf 的 "PE 侧" 有数据且达到启动阈值
  ▼
S_COMPUTE
  │ PE 消费 PPBuf 数据
  │ DMA 继续填充 PPBuf 的另一侧（并行！）
  │ 判断条件：
  │   - DMA 传输完成 (dma_w_done && dma_a_done)
  │   - 且 PPBuf 中待消费数据已空
  ▼
S_DRAIN
  │ PE 消费 PPBuf 中剩余数据
  │ 等待 buf_empty && dma_done
  ▼
S_WRITE_BACK
  │ PE 结果写回 DRAM
  ▼
S_DONE
  │ done=1, irq=1
  ▼
S_IDLE
```

### 4.5 axi_lite_bridge.v（iomem → AXI-Lite 桥接）

PicoRV32 的 `iomem` 接口是简单握手协议（valid/ready），NPU 寄存器需要 AXI4-Lite 协议。需要一个桥接模块：

```verilog
module axi_lite_bridge (
    input  wire        clk,
    input  wire        rst_n,
    // PicoRV32 iomem 侧
    input  wire        iomem_valid,
    output wire        iomem_ready,
    input  wire [3:0]  iomem_wstrb,
    input  wire [31:0] iomem_addr,
    input  wire [31:0] iomem_wdata,
    output reg  [31:0] iomem_rdata,
    // NPU AXI4-Lite 侧
    output wire [31:0] s_axi_awaddr,
    output wire        s_axi_awvalid,
    input  wire        s_axi_awready,
    output wire [31:0] s_axi_wdata,
    output wire [3:0]  s_axi_wstrb,
    output wire        s_axi_wvalid,
    input  wire        s_axi_wready,
    input  wire [1:0]  s_axi_bresp,
    input  wire        s_axi_bvalid,
    output wire        s_axi_bready,
    output wire [31:0] s_axi_araddr,
    output wire        s_axi_arvalid,
    input  wire        s_axi_arready,
    input  wire [31:0] s_axi_rdata,
    input  wire [1:0]  s_axi_rresp,
    input  wire        s_axi_rvalid,
    output wire        s_axi_rready
);
```

### 4.6 soc_top.v 顶层集成

```verilog
module soc_top #(
    parameter MEM_WORDS   = 1024,    // SRAM 4KB
    parameter DRAM_WORDS  = 15360,   // DRAM ~60KB
    parameter NPU_ROWS    = 4,
    parameter NPU_COLS    = 4
)(
    input  wire        clk,
    input  wire        rst_n,
    // UART (调试输出，可选)
    output wire        uart_tx,
    input  wire        uart_rx
);
```

地址译码逻辑（参考 picosoc）：

```verilog
// 地址分区
wire addr_is_ram   = mem_valid && (mem_addr < 4*MEM_WORDS);
wire addr_is_dram  = mem_valid && (mem_addr >= 4*MEM_WORDS) && (mem_addr < 32'h0200_0000);
wire addr_is_npu   = mem_valid && (mem_addr >= 32'h0200_0000) && (mem_addr < 32'h0200_0040);

assign iomem_valid = addr_is_npu;   // NPU 寄存器区域
```

### 4.7 DRAM 行为模型（双接口）

DRAM 需要同时服务 CPU 和 NPU DMA：

```verilog
module dram_model #(
    parameter WORDS  = 15360,
    parameter ADDR_W = 32
)(
    input  wire        clk,
    // 端口1: CPU 侧（简单读写）
    input  wire        cpu_en,
    input  wire        cpu_we,
    input  wire [3:0]  cpu_wstrb,
    input  wire [31:0] cpu_addr,
    input  wire [31:0] cpu_wdata,
    output reg  [31:0] cpu_rdata,
    // 端口2: NPU DMA 侧（AXI4 行为模型）
    input  wire        dma_arvalid,
    output wire        dma_arready,
    input  wire [31:0] dma_araddr,
    // ... AXI R/W 通道
);
```

### 4.8 测试固件 (soc_test.S)

RISC-V 汇编固件完成以下流程：

```asm
# 1. 初始化 DRAM 中的测试矩阵
#    将 4×4 INT8 权重矩阵写入 DRAM 地址 0x100
#    将 4×4 INT8 激活矩阵写入 DRAM 地址 0x200

# 2. 配置 NPU 寄存器（地址 0x0200_0000 基址）
    li t0, 0x02000000
    li t1, 0x01           # ctrl = start
    sw t1, 0x00(t0)       # 写 CTRL 寄存器

    li t1, 4              # M=4
    sw t1, 0x10(t0)       # M_DIM
    li t1, 4              # N=4
    sw t1, 0x14(t0)       # N_DIM
    li t1, 4              # K=4
    sw t1, 0x18(t0)       # K_DIM
    li t1, 0x00000100     # W addr = 0x100
    sw t1, 0x20(t0)       # W_ADDR
    li t1, 0x00000200     # A addr = 0x200
    sw t1, 0x24(t0)       # A_ADDR
    li t1, 0x00000300     # R addr = 0x300
    sw t1, 0x28(t0)       # R_ADDR

    # 设置 CTRL: start=1, mode=INT8(00), stat=WS(00)
    li t1, 0x01
    sw t1, 0x00(t0)

# 3. 轮询等待完成
poll:
    lw t1, 0x04(t0)       # 读 STATUS
    andi t1, t1, 2        # 检查 done 位
    beqz t1, poll

# 4. 读结果验证
    li t2, 0x00000300     # 结果地址
    lw t3, 0(t2)          # 读第一个结果
    li t4, EXPECTED_VALUE
    bne t3, t4, fail

# 5. 成功/失败标记
pass:
    li t0, 0x00000F00
    li t1, 0xAA           # magic pass value
    sw t1, 0(t0)
    j end

fail:
    li t0, 0x00000F00
    li t1, 0xFF           # magic fail value
    sw t1, 0(t0)

end:
    j end                  # 死循环，仿真通过 $finish 检测
```

## 5. 数据流完整路径

以一次 INT8 4×4 矩阵乘法为例：

```
阶段1: CPU 初始化数据
  CPU 固件 → SRAM 指令执行
  CPU → DRAM[0x100..0x13F] 写入权重矩阵 (4×4×2=32 bytes)
  CPU → DRAM[0x200..0x23F] 写入激活矩阵 (4×4×2=32 bytes)

阶段2: CPU 配置 NPU
  CPU → NPU_REG[CTRL]    = 0x01 (start)
  CPU → NPU_REG[M_DIM]   = 4
  CPU → NPU_REG[N_DIM]   = 4
  CPU → NPU_REG[K_DIM]   = 4
  CPU → NPU_REG[W_ADDR]  = 0x100
  CPU → NPU_REG[A_ADDR]  = 0x200
  CPU → NPU_REG[R_ADDR]  = 0x300
  CPU → NPU_REG[ARR_CFG] = 0x44  (4行×4列)
  CPU → NPU_REG[CTRL]    = 0x01  (start!)
  NPU FSM 进入 S_LOAD_PPBUF

阶段3: DMA 加载 + PE 计算（Ping-Pong 重叠）
  ┌─ cycle 1-16 ──────────────────────────────────────────┐
  │ DMA: AR → DRAM, 读权重 → PPBuf_W.BufA[0..15]        │
  │ PE:  空闲（BufA 未满，未启动）                         │
  └────────────────────────────────────────────────────────┘
  ┌─ cycle 17-32 ─────────────────────────────────────────┐
  │ DMA: 读权重 → PPBuf_W.BufA[16..31]                   │
  │       同时启动 Activation DMA → PPBuf_A.BufA[0..15]  │
  │ PE:  ← PPBuf_W.BufA[0..15] 开始消费权重               │
  │       ← PPBuf_A.BufA[0..15] 开始消费激活              │
  └────────────────────────────────────────────────────────┘
  ┌─ cycle 33+ ───────────────────────────────────────────┐
  │ DMA: swap → PPBuf_W.BufB, PPBuf_A.BufB              │
  │       继续填充另一半                                   │
  │ PE:  继续消费 BufA 的剩余数据                          │
  │       消费完 BufA → swap → 消费 BufB                   │
  └────────────────────────────────────────────────────────┘
  ⚡ 在此期间 DMA 和 PE 始终并行，无等待

阶段4: 排空残余
  DMA 传输完成，PE 继续消费 PPBuf 剩余数据
  buf_empty && dma_done → 进入 S_WRITE_BACK

阶段5: 结果写回
  PE 计算结果 → PPBuf_R → DMA → DRAM[0x300..0x33F]

阶段6: 完成
  NPU done=1 → irq → CPU 中断或轮询检测
  CPU 从 DRAM[0x300] 读结果验证
```

## 6. 性能对比

### 无 Ping-Pong（当前设计）

```
|  DMA Load W  |  DMA Load A  |  PE Compute  |  DMA WB  |
|◄─────────────►|◄────────────►|◄────────────►|◄────────►|
     T_dma         T_dma          T_pe         T_wb

总时间 = T_dma(W) + T_dma(A) + T_pe + T_dma(WB)
```

### 有 Ping-Pong（新设计）

```
|  DMA Load W+A (overlap with PE)  |  PE Compute (overlap)  | Drain | WB |
|◄────────────────────────────────►|◄───────────────────────►|◄─────►|◄──►|
                T_load+compute                              T_drain  T_wb

总时间 ≈ max(T_dma, T_pe) + T_drain + T_wb
```

**加速比**：对于大矩阵（DMA 时间和 PE 计算时间接近时），接近 **2x**。
实际加速取决于矩阵大小和 DEPTH 参数。

## 7. 文件结构

```
rtl/
├── pe/
│   ├── pe_top.v
│   └── fp16_mul.v
├── array/
│   └── pe_array.v
├── axi/
│   ├── npu_axi_lite.v      (现有，不变)
│   └── npu_dma.v           (修改：接入 PPBuf)
├── ctrl/
│   └── npu_ctrl.v           (修改：流水化 FSM)
├── buf/
│   ├── pingpong_buf.v       (新增)
│   └── ppb_wrapper.v        (新增)
├── common/
│   ├── fifo.v
│   ├── axi_monitor.v
│   └── op_counter.v
├── power/
│   └── npu_power.v
├── soc/
│   ├── soc_top.v            (新增)
│   ├── axi_lite_bridge.v    (新增)
│   ├── soc_mem.v            (新增)
│   └── dram_model.v         (新增)
└── top/
    └── npu_top.v            (修改：集成 PPBuf)

tb/
├── tb_soc.v                 (新增)
└── soc_test.S               (新增)

scripts/
└── run_soc_sim.ps1           (新增)

sim/                           (PicoRV32 参考源码)
└── picorv32.v
```

## 8. 实现优先级

| 阶段 | 内容 | 依赖 |
|---|---|---|
| **P0** | `pingpong_buf.v` + 单元测试 | 无 |
| **P1** | `npu_dma.v` 改造接入 PPBuf | P0 |
| **P2** | `npu_ctrl.v` 流水化 FSM | P1 |
| **P3** | `npu_top.v` 集成 PPBuf | P1+P2 |
| **P4** | SoC 基础框架：`soc_mem.v` + `dram_model.v` + `soc_top.v` | P3 |
| **P5** | `axi_lite_bridge.v` | P4 |
| **P6** | `tb_soc.v` + `soc_test.S` + 仿真验证 | P4+P5 |
