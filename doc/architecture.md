# 当前 NPU/SoC 架构

更新时间：2026-05-25

本文是当前实现状态说明，不是目标架构计划。结论以当前 RTL、生成器、固件模板、testbench 和最近一次 `./run_vgg_e2e.sh` 仿真为准。

## 文档边界

本文回答三个问题：

- 当前 SoC/NPU/CPU/DRAM/testbench 的边界是什么。
- 当前 VGG E2E 路径中 Python、CPU 固件、NPU RTL 分别做了什么。
- 哪些能力已经在当前路径中使用，哪些只是已有模块、历史测试项或未来计划。

本文不再沿用旧文档里的 Phase/T5/T6/T7 叙事，也不把计划写成已实现事实。完整 runtime 层间闭环的后续路线见 `doc/closed_loop_vgg_inference_plan.md`。

关键结论：当前 `run_vgg_e2e.sh` 已验证 VGG 9 层 Conv tile + CPU avgpool/classifier/argmax 可以通过分类测试，但它仍不是完整 9 层 runtime 层间闭环。Python 仍在仿真前生成所有 Conv 层所需的 A tile 输入数据。

## 当前系统概览

当前 SoC 由 PicoRV32、NPU、片上 SRAM、仿真 DRAM 和 Verilator testbench 组成。

```text
Verilator testbench
  |
  | load firmware hex into SRAM
  | load DRAM init hex into DRAM model
  | observe DRAM marker for PASS/FAIL/TIMEOUT
  v
soc_top
  |
  +-- PicoRV32 CPU
  |     |
  |     +-- SRAM: firmware instruction/data memory
  |     +-- DRAM: runtime data, tile table, classifier data, marker
  |     +-- AXI-Lite bridge -> NPU register file
  |
  +-- NPU
  |     |
  |     +-- AXI-Lite slave: CPU configuration/status path
  |     +-- AXI4 master DMA: reads A/W/bias, writes results
  |     +-- ping-pong buffers
  |     +-- reconfigurable PE array
  |     +-- post-process + result FIFO
  |
  +-- DRAM model
        |
        +-- CPU simple memory access port
        +-- NPU AXI4 DMA port
```

默认 SoC 地址边界来自 `rtl/soc/soc_top.v`：

| 区域 | 当前含义 |
|---|---|
| `0x0000_0000 .. 4*MEM_WORDS-1` | SRAM，PicoRV32 firmware 运行区域。VGG testbench 当前 `MEM_WORDS=1024`，即 4 KiB。 |
| `4*MEM_WORDS .. 0x01FF_FFFF` | 仿真 DRAM，可被 CPU 访问，也可被 NPU DMA 访问。 |
| `0x0200_0000 ..` | NPU MMIO 寄存器，经 `axi_lite_bridge` 转为 NPU AXI-Lite slave 访问。 |

当前 VGG testbench 使用 `DRAM_WORDS = 2*1024*1024`，即 DRAM model 为 2M 个 32-bit word。

## 当前组件职责

| 组件 | 当前职责 |
|---|---|
| `run_vgg_e2e.sh` | 清理 `sim/vgg_e2e/`，运行生成器，调用 Verilator 编译 SoC testbench，执行仿真并打印 PASS/FAIL/TIMEOUT、cycles、预测类别。 |
| `tools/pth/gen_vgg_e2e.py` | 加载 RepOpt VGG checkpoint/spec，生成 DRAM 初始数据、VGG tile table、firmware hex、Verilog 参数文件和 expected data。 |
| `tools/pth/vgg_fw_template.hex` | PicoRV32 固件模板。运行时读取 10-word VGG tile table，逐 tile 配置 NPU，最后执行 avgpool、classifier、argmax 并写 marker。 |
| `tb/tb_soc_vgg_e2e.v` | VGG SoC testbench。加载 SRAM/DRAM hex，监视 marker，区分 PASS、classification mismatch、firmware failure 和 timeout。 |
| `rtl/soc/soc_top.v` | 集成 PicoRV32、SRAM、DRAM model、AXI-Lite bridge 和 NPU。 |
| `rtl/top/npu_top.v` | NPU 顶层，连接寄存器文件、controller、DMA、PPBuf、PE array、post-process、result FIFO 和 performance counters。 |
| `rtl/ctrl/npu_ctrl.v` | NPU 调度 FSM。锁存 CPU 配置，发起 A/W/bias DMA，驱动 PE，控制 tile writeback，也包含 RTL descriptor v1 decode 路径。 |
| `rtl/axi/npu_dma.v` | AXI4 master DMA。读 W/A/descriptor/bias，写 result FIFO 数据回 DRAM。 |
| `rtl/array/reconfig_pe_array.v` | 可重构 PE array 外壳，物理 16x16 PE，支持 4x4/8x8/16x16/8x32 shape 选择接口。 |

## CPU 到 NPU 的控制面

CPU 通过 `0x0200_0000` 基地址访问 NPU AXI-Lite 寄存器。当前关键寄存器如下，完整 bit 定义以 `rtl/axi/npu_axi_lite.v` 为准。

| Offset | 名称 | 当前用途 |
|---|---|---|
| `0x00` | `CTRL` | `bit0=start`，`[3:2]=data_mode`，`[5:4]=stat_mode`，`bit7=desc_mode`，`bit8=conv_im2col`，`bit9=bias_en`，`[11:10]=activation`。 |
| `0x04` | `STATUS` | `busy/done/error`。固件用它判断 NPU tile 是否完成。 |
| `0x10/0x14/0x18` | `M_DIM/N_DIM/K_DIM` | GEMM/Conv tile 维度。 |
| `0x20/0x24/0x28` | `W_ADDR/A_ADDR/R_ADDR` | DRAM 中 W tile、A tile、result 输出地址。 |
| `0x30` | `ARR_CFG` | `bit7=1` 进入 tile mode。 |
| `0x3C` | `CFG_SHAPE` | 阵列 shape：`0=4x4`，`1=8x8`，`2=16x16`，`3=8x32`。复位默认 `16x16`。 |
| `0x40/0x44` | `DESC_BASE/DESC_COUNT` | RTL descriptor v1 路径使用；当前 VGG E2E 不使用此路径。 |
| `0x80..0x94` | Conv shape registers | direct scalar on-the-fly im2col 路径使用；当前 VGG tile 路径不依赖。 |
| `0x98` | `BIAS_ADDR` | bias 向量地址。tile mode 下 controller 按 tile column 逐 word 取 bias。 |
| `0x9C` | `QUANT_CFG` | INT8 quant/saturate 参数：enable、round、shift、signed scale。 |
| `0xA0..0xC8` | performance counters | MAC/ops、busy/compute/DMA cycles、TOPS/util 等计数口径。 |

当前 VGG tile 的 `CTRL` 值为 `0x611`，含义是：start、INT8、OS、bias enable、ReLU。量化由 `QUANT_CFG` 单独配置。

## 当前执行模式

### Direct Register Mode

这是当前 VGG E2E 使用的路径。CPU 固件对每个 tile 执行：

```text
write A_ADDR
write R_ADDR
write W_ADDR
write BIAS_ADDR
write M_DIM/N_DIM/K_DIM
write QUANT_CFG
write ARR_CFG
write CTRL with start
poll STATUS until done/error
```

进入 NPU 后的数据路径为：

```text
AXI-Lite registers
  -> npu_ctrl shadow config
  -> npu_dma reads W/A/bias from DRAM
  -> ping-pong buffers
  -> reconfigurable PE array
  -> tile result capture
  -> bias -> activation -> quant/saturate
  -> result FIFO
  -> npu_dma writes R back to DRAM
```

VGG 当前 tile shape 实际使用 16x16：生成器使用 `TR=16`、`TC=16`，`ARR_CFG=0x80` 打开 tile mode，`CFG_SHAPE` 依赖复位默认 `2'b10`。边界 tile 的有效行列由 `M_DIM/N_DIM` 和 controller 的 active row/col mask 限制。

### RTL Descriptor v1 Mode

RTL 中还有一条 descriptor v1 路径，由 `DESC_BASE`、`DESC_COUNT` 和 `CTRL[7] desc_mode` 启动。该 ABI 是 16 个 32-bit word，共 64 byte。`npu_ctrl.v` 当前支持的基础 descriptor 条件是：

```text
version = 1
op = GEMM_TILEPACK
dtype = INT8 or FP16
flow = OS
shape = 4x4/8x8/16x16/8x32
tile_packed = 1
use_bias = 0
use_psum = 0
```

这个 RTL descriptor v1 不等于当前 VGG 固件使用的 10-word tile table。当前 VGG E2E 没有通过 `DESC_BASE/DESC_COUNT` 让 NPU 自己 fetch descriptor；它由 CPU 固件读取 DRAM 中的 VGG tile table，再逐项写 AXI-Lite 寄存器。

### Direct Scalar Conv/Im2col Path

RTL 中保留 direct scalar Conv2D on-the-fly im2col 相关寄存器和 DMA 模式，例如 `CTRL[8]`、`CONV_IFM_SHAPE`、`CONV_CHANNELS`、`CONV_KERNEL`、`CONV_OUT_SHAPE`、`CONV_STRIDE_PAD`、`CONV_DILATION`。

该路径不是当前 VGG E2E 主路径。当前 VGG 使用离线生成的 packed A tile，而不是在 NPU tile path 中 runtime 从 dense IFM 生成 im2col/A tile。

## 当前 VGG E2E 数据流

当前入口：

```bash
./run_vgg_e2e.sh
```

最近验证结果：

```text
[PASS] RepOpt VGG end-to-end classification PASSED
Cycles: 10768727
Predicted: cat (class 3)
```

整体流程：

```text
Python generator
  -> creates DRAM image, W tiles, bias, A tiles for all Conv layers
  -> creates 10-word tile table for 1024 NPU tile runs
  -> creates classifier weights/bias and expected label
  -> emits dram_init.hex, soc_vgg.hex, soc_vgg_params.vh

Verilator testbench
  -> loads soc_vgg.hex into SRAM
  -> loads dram_init.hex into DRAM
  -> releases reset

PicoRV32 firmware
  -> loops over 1024 tile table entries
  -> configures and starts NPU for each tile
  -> polls NPU status
  -> after Conv tiles, reads stage4_1 results
  -> computes avgpool + classifier + argmax
  -> writes marker to DRAM

NPU RTL
  -> executes Conv/GEMM tile MACs
  -> applies tile bias/ReLU/quant where configured
  -> writes result tiles to DRAM

Testbench
  -> reads marker
  -> reports PASS, mismatch, firmware failure, or timeout
```

### 当前生成器使用的 DRAM Layout

当前 VGG 生成器使用以下基地址：

| 地址 | 名称 | 内容 |
|---|---|---|
| `0x0000_1000` | `W_BASE` | packed W tile streams。 |
| `0x0000_2000` | `B_BASE` | bias vectors。 |
| `0x0000_3000` | `TILE_TABLE_BASE` | VGG firmware tile table，10 words per tile。 |
| `0x0004_0000` | `A_BASE` | packed A tile streams。当前所有 Conv 层的 A tile 都由 Python 预生成。 |
| `0x0008_0000` | `R_BASE` | NPU result tiles。 |
| `0x0060_0000` | `FEAT_BASE` | CPU avgpool 后的 512 features。 |
| `0x0060_2000` | `CLS_W_BASE` | classifier weights。 |
| `0x0061_0000` | `CLS_B_BASE` | classifier bias。 |
| `0x0061_1000` | `SCORE_BASE` | classifier scores。 |
| `0x0061_2000` | `MARKER` | firmware/testbench marker。 |
| `0x0061_3000` | `LABEL_ADDR` | Python expected class。 |

### VGG Firmware Tile Table

当前 VGG tile table 每个 entry 是 10 个 32-bit word：

| Word | 字段 | 用途 |
|---|---|---|
| 0 | `A_ADDR` | packed A tile stream 地址。 |
| 1 | `R_ADDR` | result tile 输出地址。 |
| 2 | `W_ADDR` | packed W tile stream 地址。 |
| 3 | `BIAS_ADDR` | 当前 N tile 的 bias 起始地址。必须包含 `n_tile * TC * 4` offset。 |
| 4 | `M_DIM` | 当前 tile 有效 M 行数。 |
| 5 | `N_DIM` | 当前 tile 有效 N 列数。 |
| 6 | `K_DIM` | 当前层 GEMM K。 |
| 7 | `QUANT_CFG` | tile-level INT8 quant 配置。 |
| 8 | `ARR_CFG` | 当前为 `0x80`，打开 tile mode。 |
| 9 | `CTRL` | 当前为 `0x611`，启动 INT8 OS bias ReLU。 |

这个 10-word table 是 PicoRV32 固件 ABI，不是 RTL descriptor v1 ABI。二者不能混用。

### 当前不是完整 Runtime 层间闭环

当前 VGG 路径中，NPU 的确执行了 9 层 Conv 对应的 1024 个 tile，CPU 固件也确实在仿真运行时执行了最后的 avgpool、classifier、argmax。

但以下内容仍由 Python 在仿真前完成：

- 逐层 Conv fixed-point golden 计算。
- MaxPool 后中间 activation 计算。
- 后续层 packed A tile 生成。
- 9 层 Conv 所有 tile 的 A tile stream 写入 `dram_init.hex`。

因此，当前运行时并没有完成：

```text
上一层 NPU OFM -> CPU/NPU runtime repack/im2col -> 下一层 A tile
```

完整闭环目标是让 Python 只生成静态资产，让 CPU 固件和 NPU RTL 在运行时从输入图像开始生成所有层间 activation。该目标仍在计划中。

## NPU Tile 数据路径

当前 VGG 依赖 tile mode。逻辑 GEMM 为：

```text
A[M,K] * W[K,N] = R[M,N]
```

tile mode 下，controller 以 tile 为单位遍历：

```text
for m_tile in ceil(M / tile_rows):
  for n_tile in ceil(N / tile_cols):
    for k_tile in ceil(K / k_tile_elems):
      DMA read packed A/W slice
      PE array compute partial/final tile
      last k_tile writes active rows/cols to DRAM
```

当前 VGG 每个固件 table entry 已经是一个具体 tile，通常 `M_DIM<=16`、`N_DIM<=16`，因此 NPU 在该 run 内看到的是一个小 tile GEMM，而不是整层 descriptor。

result writeback 使用 row-major C layout：

```text
R_ADDR + ((m_base + row) * N_DIM + (n_base + col)) * 4
```

在当前 VGG 固件逐 tile 调度方式下，`R_ADDR` 已经是每个 tile 的输出基址，`M_DIM/N_DIM` 是该 tile 的局部有效尺寸。

tile post-process 顺序来自 `rtl/top/npu_top.v`：

```text
accumulator
  -> optional bias
  -> optional activation: none/ReLU/ReLU6
  -> optional INT8 quant/saturate
  -> result FIFO
  -> DMA writeback
```

当前 VGG 使用 bias、ReLU 和 quant/saturate。`QUANT_CFG` 是 tile-level scalar 配置，不是完整 per-channel quant table。

## Testbench Marker 规则

当前 VGG testbench 监视 `MARKER = 0x0061_2000`。

| Marker | 含义 |
|---|---|
| `0x100 + expected_class` | PASS。 |
| `0x100..0x109` 且不等于 expected | classification mismatch，立即 FAIL。 |
| `0x000000FF` | firmware failure。 |
| 超过 `VGG_TIMEOUT_CYCLES` | TIMEOUT。 |

这个 fail-fast 规则避免 wrong-class marker 被误报为 timeout。

## 当前已验证基线

当前可作为本架构文档依据的 VGG 基线是：

```bash
./run_vgg_e2e.sh
```

已验证内容：

- Verilator 可编译并运行 PicoRV32 + NPU + DRAM model 的 SoC testbench。
- PicoRV32 firmware 可逐 tile 配置 NPU 并等待完成。
- NPU 可完成当前 VGG 9 层 Conv 对应的 1024 个 tile。
- VGG tile table entry 宽度为 10 words，与 firmware 读取顺序一致。
- per-N-tile `BIAS_ADDR` offset 正确，否则后续 channel tile 会读错 bias。
- wrong-class marker 会被 testbench 立即归类为 classification mismatch。
- 默认样本通过分类，cycles 为 `10768727`。

## 已实现但不要在当前 VGG 中过度声明的能力

以下能力在源码中存在，部分有历史定向测试，但不是当前 VGG E2E 的核心闭环依据：

| 能力 | 当前说明 |
|---|---|
| RTL descriptor v1 | `npu_ctrl.v` 支持 64-byte descriptor fetch/decode/next-layer，但当前 VGG E2E 不通过该路径调度。 |
| `IFM_FROM_PREV_OFM` descriptor flag | RTL 中有上一层 OFM gather/repack 入口，但当前 VGG 9 层没有用它实现 runtime 层间闭环。 |
| direct scalar on-the-fly im2col | RTL 有相关寄存器和 DMA 逻辑，不是当前 VGG tile path。 |
| 4x4/8x8/16x16/8x32 shape selector | `reconfig_pe_array` 和 controller 有接口。当前 VGG 依赖 16x16 tile shape，不能据此声明所有 shape 的端到端吞吐都已验证。 |
| performance counters | AXI-Lite 计数器已接入，但本文不声明最新性能或 TOPS 数字。 |
| low-power/clock-gating model | 源码有 `rtl/power` 和相关接口，但本文不声明 FPGA/硅级功耗结论。 |

## 当前限制

当前仍不能声明为已完成的能力：

- 完整 9 层 runtime 层间闭环。
- CPU 固件 runtime 生成第 2 到第 9 层 A tile。
- CPU 固件 runtime 执行所有 MaxPool 并把输出作为下一层输入。
- Python 只提供静态资产的 VGG 推理路径。
- NPU tile path 支持 per-channel quant table。
- VGG 使用 RTL descriptor v1 一次性链式提交完整网络。
- 8x32 或完整 16x16 峰值吞吐的端到端 VGG 验证。
- 外部 PSUM surface read/modify/write 被当前 VGG 依赖。
- 固件源码级可维护性；当前 VGG 固件仍是 hex 模板。
- FPGA timing、真实功耗、真实外部 DRAM 带宽结论。

## 后续文档分工

为避免再次把计划和事实混在一起，建议按下列边界维护文档：

| 文档 | 负责内容 |
|---|---|
| `doc/architecture.md` | 当前 SoC/NPU/CPU/DRAM 结构和真实数据流。 |
| `doc/vgg_e2e_flow.md` | 当前 VGG E2E 的逐步运行流程、生成物、PASS 输出和故障分类。 |
| `doc/memory_map_and_abi.md` | NPU 寄存器、VGG DRAM layout、10-word tile table、marker、descriptor v1 ABI。 |
| `doc/firmware_runtime.md` | PicoRV32 固件启动、tile loop、avgpool/classifier/argmax、未来固件生成策略。 |
| `doc/rtl_reference.md` | 各 RTL 模块接口、职责、限制和测试入口。 |
| `doc/verification_status.md` | 最近实际执行过的命令、结果和未验证项。 |
| `doc/closed_loop_vgg_inference_plan.md` | 完整 runtime 层间闭环实现计划。 |

## 关键源码索引

| 文件 | 为什么重要 |
|---|---|
| `run_vgg_e2e.sh` | 当前 VGG E2E 入口。 |
| `tools/pth/gen_vgg_e2e.py` | 当前 VGG DRAM layout、10-word tile table、Python 离线 A tile 生成逻辑。 |
| `tools/pth/vgg_fw_template.hex` | 当前 PicoRV32 固件模板。 |
| `tb/tb_soc_vgg_e2e.v` | marker 规则和 VGG PASS/FAIL/TIMEOUT 判断。 |
| `rtl/soc/soc_top.v` | SoC 地址边界、CPU/NPU/DRAM 集成。 |
| `rtl/axi/npu_axi_lite.v` | NPU AXI-Lite 寄存器定义。 |
| `rtl/ctrl/npu_ctrl.v` | direct register mode、descriptor v1 decode、tile loop、bias fetch 和 writeback FSM。 |
| `rtl/axi/npu_dma.v` | W/A/bias/descriptor read 和 result writeback。 |
| `rtl/top/npu_top.v` | NPU 顶层数据通路、post-process、result FIFO、PE array 连接。 |
| `rtl/array/reconfig_pe_array.v` | 物理 PE array 和 shape 选择。 |
