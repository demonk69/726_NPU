# PYNQ-Z2 上板部署计划

Updated: 2026-06-02

本文是当前 PYNQ-Z2 上板工作的执行清单。当前已选择 PYNQ/Zynq 传统路线：图片和模型资产由 PS ARM 端通过 Linux、Python、Notebook、SSH 或板上文件系统管理；PL 侧 NPU 只通过 AXI 读写 DDR buffer，并通过 AXI-Lite 接受控制。

UART、SPI Flash、Boot ROM、板上 PicoRV32 和纯 PL 图片加载不属于第一版 PYNQ-Z2 主路径。

## 目标数据流

```text
PC / browser / SSH / Notebook
  -> Zynq PS ARM 上的 PYNQ Linux
  -> PYNQ Python runtime
  -> pynq.allocate 分配的 DDR/CMA buffer
  -> PL NPU 通过 AXI4 master 读写 DDR
  -> PS 读取分类结果和性能计数器
```

第一阶段目标是在 PYNQ-Z2 上完成一次 RepOpt VGG closed-loop 推理。每张图片需要返回：

- `class_id`
- NPU raw performance counters
- PS/host 计算得到的 cycles、TOPS、读带宽、写带宽、utilization

## 当前 RTL 状态

上板相关 RTL 结构如下：

- `rtl/top/npu_pynq_wrapper.v`：推荐作为 Vivado IP 顶层，用于 PYNQ/Zynq。
- `rtl/top/npu_top.v`：板无关 NPU 顶层，由 wrapper 例化。
- `rtl/axi/npu_axi_lite.v`：AXI-Lite register file、status、error clear、perf clear/snapshot。
- `rtl/axi/npu_dma.v`：NPU 内部 AXI4 master DMA，用于 DDR 读写。
- `rtl/ctrl/npu_ctrl.v`：NPU 调度 FSM、tile flow、descriptor flow、error latch。

已经具备的上板相关能力：

- AXI-Lite write response 使用 registered `BVALID`，并保持到 `BREADY`。
- `PERF_CTRL` 位于 offset `0x78`，`bit0=clear`，`bit1=snapshot`。
- 性能寄存器读 snapshot shadow，不直接读 live counter。
- RTL 派生性能指标默认关闭，`PERF_ENABLE_DERIVED=0`。
- DMA 会把 AXI `RRESP`、AXI `BRESP` 和 4-byte 对齐错误上报到 `ERR_STATUS`。
- `npu_pynq_wrapper` 已暴露常见 Xilinx AXI sideband，并把 AXI-Lite 地址裁成 NPU 本地 offset。

性能计数器分工：

- PL 只输出 raw counters。
- TOPS、带宽和 utilization 在 PS/PYNQ Python 或 host 端计算。
- 第一版 Vivado BD 不需要单独添加 performance counter IP。

## 用户需要做什么

仓库现在提供了第一版 Vivado project/BD Tcl 骨架：`scripts/create_pynq_z2_npu_project.tcl`。它用于创建 PYNQ-Z2 project、加入 NPU RTL、生成 PS+NPU Block Design wrapper；脚本仍需要在真实 Vivado 环境中验证和迭代。

用户需要完成以下板级工作：

1. 创建 PYNQ-Z2 Vivado project。
2. 加入本文列出的 RTL 源文件。
3. 将 `npu_pynq_wrapper` package 成 custom IP，或在 BD wrapper flow 中直接例化。
4. 创建 Zynq Block Design：PS GP0 连接 NPU AXI-Lite，NPU AXI master 通过 HP0 访问 PS DDR。
5. 生成同名 `.bit` 和 `.hwh`。
6. 把 `.bit`、`.hwh` 以及后续 runtime assets/scripts 复制到 PYNQ 板上。
7. 依次跑 AXI-Lite smoke、DDR/NPU smoke、single-layer smoke、full VGG。

## 我还未完成什么

仓库中尚未完成以下内容，不能假设已经可直接上板：

- 已有第一版 Vivado project/BD Tcl：`scripts/create_pynq_z2_npu_project.tcl`，但还没有在 Vivado 中跑通。
- 还没有在 Vivado IP Packager 中 package 并验证 `npu_pynq_wrapper`；当前 Tcl 第一版先用 BD module reference 直接例化 wrapper。
- 还没有跑 PYNQ-Z2 synthesis、implementation、timing 或 bitstream generation。
- 还没有生成 `.bit/.hwh` artifact。
- 还没有 PYNQ Python runtime。
- 当前 closed-loop firmware/runtime 行为还没有完整从仿真/PicoRV32 flow 移植到 PS Python 或 PS C helper。
- 还没有整理 board static assets：weights、bias、per-channel Q24 multipliers、classifier 参数、image normalization metadata。
- 还没有做任何真实板上 smoke test。

## Vivado 工程设置

建议使用：

- Board：PYNQ-Z2
- Device：`xc7z020clg400-1`
- Flow：Vivado project + IP Integrator Block Design
- 第一版 PL clock：`100 MHz`；如果 timing 不收敛，再降到 `50 MHz`

第一版自动化入口：

```bash
vivado -mode batch -source scripts/create_pynq_z2_npu_project.tcl
```

生成 bitstream 时使用：

```bash
vivado -mode batch -source scripts/create_pynq_z2_npu_project.tcl -tclargs --build-bitstream
```

默认输出：

| 项目 | 默认值 |
|---|---|
| Project dir | `build/vivado/pynq_z2_npu` |
| Project name | `npu_pynq_z2` |
| Device | `xc7z020clg400-1` |
| Board part | 自动匹配 `*pynq-z2*`，必要时用 `--board-part <name>` 指定 |
| Block Design | `system.bd` |

脚本第一版依赖 Vivado 对 `npu_pynq_wrapper` 的 AXI interface inference。如果 Vivado 没有把 `s_axi_*` 和 `m_axi_*` 识别成 `S_AXI`/`M_AXI` interface，先按报错信息在 IP Packager 中手动 map，或后续给 wrapper 补 Xilinx interface attributes。

Vivado project 需要加入这些 RTL：

| Path | 用途 |
|---|---|
| `rtl/top/npu_pynq_wrapper.v` | PYNQ/Zynq wrapper 和 Vivado-facing top |
| `rtl/top/npu_top.v` | 板无关 NPU top |
| `rtl/axi/npu_axi_lite.v` | AXI-Lite register/status/perf block |
| `rtl/axi/npu_dma.v` | AXI4 master DMA |
| `rtl/ctrl/npu_ctrl.v` | scheduler 和 error/status control |
| `rtl/array/reconfig_pe_array.v` | reconfigurable PE array |
| `rtl/buf/pingpong_buf.v` | W/A ping-pong buffers |
| `rtl/buf/psum_out_buf.v` | buffer 源文件，保留加入工程 |
| `rtl/pe/pe_top.v` | PE datapath |
| `rtl/pe/fp16_mul.v` | FP16 multiply support |
| `rtl/pe/fp16_add.v` | FP16 add support |
| `rtl/pe/fp32_add.v` | FP32 add support |
| `rtl/common/fifo.v` | result FIFO 和 common FIFO |
| `rtl/common/op_counter.v` | operation counters |
| `rtl/common/axi_monitor.v` | AXI raw counters |
| `rtl/power/npu_power.v` | power/clock-gating wrapper |

不要把下面这些文件作为 PYNQ-Z2 PL 顶层或板级系统的一部分：

- `rtl/soc/soc_top.v`
- `rtl/soc/soc_mem.v`
- `rtl/soc/dram_model.v`
- `rtl/soc/axi_lite_bridge.v`
- `tb/*`
- `sim/*`

这些是仿真基础设施，不是 Zynq/PYNQ 板级集成结构。

## NPU IP Packaging

第一版推荐把 `npu_pynq_wrapper` package 成 Vivado custom IP。

Vivado 操作：

1. 打开 `Tools -> Create and Package New IP`。
2. 选择 `Package your current project` 或指定 RTL 目录。
3. 顶层模块设为 `npu_pynq_wrapper`。
4. 检查 Vivado interface inference。
5. 如果某个接口没有自动识别，就按下表手动 map 端口。

预期 IP 接口：

| Wrapper port | Vivado interface | 说明 |
|---|---|---|
| `aclk` | Clock | 接 PS `FCLK_CLK0` |
| `aresetn` | Active-low reset | 接 `proc_sys_reset/peripheral_aresetn` |
| `s_axi_*` | AXI4-Lite slave, `S_AXI` | PS 配置 NPU register |
| `m_axi_*` | AXI4 master, `M_AXI` | NPU DMA 读写 DDR |
| `npu_irq` | Interrupt | 第一版可选 |

第一版 IP 参数保持默认值：

| Parameter | Value |
|---|---:|
| `PHY_ROWS` | `16` |
| `PHY_COLS` | `16` |
| `DATA_W` | `32` |
| `ACC_W` | `32` |
| `PPB_DEPTH` | `64` |
| `PPB_THRESH` | `16` |
| `INT8_SIMD_LANES` | `4` |
| `PERF_ENABLE_DERIVED` | `0` |
| `S_AXI_OFFSET_BITS` | `16` |
| `M_AXI_ID_WIDTH` | `1` |

`S_AXI_OFFSET_BITS=16` 表示 wrapper 只保留低 64KB register offset 再传给 `npu_top`。PYNQ runtime 仍必须从 `.hwh` 或 `overlay.ip_dict` 获取真实 IP base address，不要硬编码仿真 base address。

## Block Design

创建一个 BD，例如 `system.bd`。

需要添加的 BD IP：

| IP | 必须 | 用途 |
|---|---|---|
| `ZYNQ7 Processing System` | 是 | ARM PS、DDR、FCLK、GP/HP AXI |
| `Processor System Reset` | 是 | 生成 PL reset |
| `npu_pynq_wrapper` custom IP | 是 | PL NPU |
| AXI Interconnect 或 SmartConnect for AXI-Lite | 是 | PS `M_AXI_GP0` 到 NPU `S_AXI` |
| AXI Interconnect 或 SmartConnect for DDR | 是 | NPU `M_AXI` 到 PS `S_AXI_HP0` |
| `xlconcat` | 可选 | 使用 `npu_irq` 时做 IRQ fan-in |
| `ILA` | 可选 | 调试 AXI-Lite、M_AXI、reset 或 IRQ |

第一版图片路径不需要这些 IP/模块：

- Xilinx `AXI DMA` IP
- UART image receiver
- SPI flash reader
- Boot ROM
- PicoRV32
- standalone PL memory system

### ZYNQ7 PS 配置

对 PYNQ-Z2 运行 board automation，然后确认：

| Setting | 第一版设置 |
|---|---|
| DDR | 按 board preset 启用 |
| Fixed IO | 按 board preset 启用 |
| `FCLK_CLK0` | 启用，先用 `100 MHz` |
| `FCLK_RESET0_N` | 启用 |
| `M_AXI_GP0` | 启用，用于 NPU AXI-Lite register |
| `S_AXI_HP0` | 启用，用于 NPU DMA 访问 DDR |
| `IRQ_F2P` | 可选；第一版轮询也可以 |

NPU `M_AXI` 当前是 32-bit data width。SmartConnect/Interconnect 插入 width conversion 是可接受的。不要只为了匹配 HP port 宽度而修改 `ACC_W`。

### BD 连接

clock/reset：

```text
processing_system7_0/FCLK_CLK0
  -> proc_sys_reset_0/slowest_sync_clk
  -> npu_pynq_wrapper_0/aclk
  -> AXI interconnect clocks

processing_system7_0/FCLK_RESET0_N
  -> proc_sys_reset_0/ext_reset_in

proc_sys_reset_0/peripheral_aresetn
  -> npu_pynq_wrapper_0/aresetn
  -> AXI interconnect aresetn
```

AXI-Lite：

```text
processing_system7_0/M_AXI_GP0
  -> AXI-Lite interconnect S00_AXI
  -> AXI-Lite interconnect M00_AXI
  -> npu_pynq_wrapper_0/S_AXI
```

NPU DDR master：

```text
npu_pynq_wrapper_0/M_AXI
  -> DDR interconnect S00_AXI
  -> DDR interconnect M00_AXI
  -> processing_system7_0/S_AXI_HP0
```

可选 IRQ：

```text
npu_pynq_wrapper_0/npu_irq
  -> xlconcat/In0
  -> processing_system7_0/IRQ_F2P[0]
```

Address Editor 建议：

| Segment | 建议 |
|---|---|
| NPU AXI-Lite base | 让 Vivado 自动分配，常见为 `0x43C0_0000` |
| NPU AXI-Lite range | `64K` |
| NPU `M_AXI` to DDR | 映射到 PS DDR address space |

PYNQ runtime 应通过 `.hwh` metadata 和 `overlay.ip_dict` 获取真实 base address。

## Bitstream 输出

Vivado checklist：

1. `Validate Design`。
2. `Generate Output Products`。
3. `Create HDL Wrapper`。
4. `Run Synthesis`。
5. `Run Implementation`。
6. 检查 timing。
7. `Generate Bitstream`。
8. 把匹配的 `.bit` 和 `.hwh` 复制到 PYNQ。

示例输出名：

```text
npu_pynq.bit
npu_pynq.hwh
```

`.bit` 和 `.hwh` 的文件名主干必须一致，PYNQ 才能自动解析 IP metadata。

## PYNQ 图片传输方式

已选择 PYNQ 标准文件/Python 传图方式。

图片到板上的可选方式：

- 通过 JupyterLab 或 Notebook 上传。
- 用 `scp` 传到板上目录，例如 `~/npu/images/`。
- 在 PC 上预处理成 `3x32x32` INT8 `.bin` 或 `.npy`，再上传到板上。

这条路径只使用 PS Linux 文件系统和 DDR buffer，不使用 PL UART、SPI 或 Xilinx `AXI DMA` IP。

PYNQ runtime 需要负责：

1. `Overlay("npu_pynq.bit")` 加载 bitstream。
2. 从 `overlay.ip_dict` 找到 NPU IP。
3. 用 `.hwh` 中的 physical address 和 range 创建 `MMIO`。
4. 用 `pynq.allocate` 分配 DDR/CMA buffer。
5. 把 static assets 加载到 board buffer。
6. 把每张输入图转成 RepOpt VGG 需要的 `3x32x32` CHW INT8。
7. 把图片 bytes 写进 activation buffer。
8. PL 读取前对 buffer 执行 `flush()`。
9. 把 physical DDR address 写入 NPU register。
10. 通过 AXI-Lite 启动 NPU tile operations。
11. 轮询 `STATUS` 或等待 IRQ。
12. PS 读取 PL 写回结果前执行 `invalidate()`。
13. PS 执行 ReLU、per-channel Q24 requant、scatter、pooling、classifier、argmax。
14. snapshot 并读取 raw performance counters。

最小 Python 形态：

```python
from pynq import Overlay, MMIO, allocate
import numpy as np

ol = Overlay("npu_pynq.bit")
npu_name = next(name for name in ol.ip_dict if "npu" in name.lower())
npu_info = ol.ip_dict[npu_name]
mmio = MMIO(npu_info["phys_addr"], npu_info["addr_range"])

runtime = allocate(shape=(runtime_bytes,), dtype=np.uint8, cacheable=False)
base = runtime.physical_address

img_i8 = preprocess_to_chw_int8("image.jpg")
runtime[ACT_A_OFF:ACT_A_OFF + 3072] = img_i8.view(np.uint8)
runtime.flush()

mmio.write(0x78, 0x1)                              # clear raw counters
run_vgg_schedule(mmio, runtime, base)               # 待实现
mmio.write(0x78, 0x2)                              # snapshot raw counters

result = read_result_and_counters(mmio)
```

地址规则：

- 仿真中的 `ACT_A=0x00010000` 这类地址应成为 PYNQ allocated runtime buffer 内部 offset。
- NPU register 必须写入 `runtime.physical_address + offset`。
- 不要把 numpy index 或仿真 absolute address 写入 NPU register。

## Register Checklist

下面 offset 都是相对 NPU AXI-Lite IP base address 的本地 offset。

| Offset | Register | 用途 |
|---:|---|---|
| `0x00` | `CTRL` | start、mode、dataflow、bias、activation |
| `0x04` | `STATUS` | bit0 busy、bit1 done、bit2 error |
| `0x08` | `INT_EN` | optional interrupt enable |
| `0x0C` | `INT_CLR` | optional interrupt clear |
| `0x10` | `M_DIM` | GEMM M |
| `0x14` | `N_DIM` | GEMM N |
| `0x18` | `K_DIM` | GEMM K |
| `0x20` | `W_ADDR` | W tile physical DDR address |
| `0x24` | `A_ADDR` | A tile physical DDR address |
| `0x28` | `R_ADDR` | result physical DDR address |
| `0x30` | `ARR_CFG` | tile mode configuration |
| `0x3C` | `CFG_SHAPE` | `0=4x4`，`1=8x8`，`2=16x16`，`3=8x32` |
| `0x74` | `ERR_STATUS` | sticky error bits，写 1 清对应 bit |
| `0x78` | `PERF_CTRL` | bit0 clear counters，bit1 snapshot counters |
| `0x98` | `BIAS_ADDR` | bias physical DDR address |
| `0x9C` | `QUANT_CFG` | 第一版 VGG 板上路径保持 HW quant disabled |

第一版 VGG tile bring-up 先使用已验证的 `16x16` 路径：

| Register | Value | 含义 |
|---|---:|---|
| `ARR_CFG` | `0x00000080` | tile mode on |
| `CFG_SHAPE` | `0x00000002` | `16x16` |
| `QUANT_CFG` | `0x00010000` | hardware quant disabled |
| `CTRL` | `0x00000211` | start、INT8、OS、bias/raw INT32 path |

第一版 board smoke 不要同时扩展 shape 变量。`16x16` 稳定后，再加入 `4x4`、`8x8` 和 `8x32`。

当前上板相关 `ERR_STATUS` masks：

| Mask | 含义 |
|---:|---|
| `0x00000010` | DMA read `RRESP` 非 OKAY |
| `0x00000020` | DMA write `BRESP` 非 OKAY |
| `0x00000040` | DMA read address 或 length 非 4-byte 对齐 |
| `0x00000080` | DMA write address 或 length 非 4-byte 对齐 |
| `0x00000100` | direct mode 存在 zero `M/N/K` |
| `0x00000200` | direct Conv2D 存在 zero shape/stride/dilation field |

性能计数器读取顺序：

1. 推理前写 `0x78 = 0x1`。
2. 运行推理。
3. 完成后写 `0x78 = 0x2`。
4. 读取 `0x48..0x70` 和 `0xA0..0xC8` snapshot。

常用 raw counter offset：

| Offset | Counter |
|---:|---|
| `0x48` | total cycles |
| `0x4C` | AXI read beats |
| `0x50` | AXI write beats |
| `0x54` | AXI read bytes |
| `0x58` | AXI write bytes |
| `0x6C` | AXI read bursts |
| `0x70` | AXI write bursts |
| `0xA0/0xA4` | MAC ops low/high |
| `0xA8/0xAC` | ops low/high |
| `0xB0` | busy cycles |
| `0xB4` | compute cycles |
| `0xB8` | DMA cycles |
| `0xC8` | peak ops per cycle |

当 `PERF_ENABLE_DERIVED=0` 时，下面寄存器预期为 0，不应用作板上指标来源：

- `0xBC`: `tops_x1e6`
- `0xC0`: `compute_util_bp`
- `0xC4`: `e2e_util_bp`

软件侧计算：

```text
tops = ops * f_clk_hz / busy_cycles / 1e12
rd_bw_bytes_per_s = rd_bytes * f_clk_hz / total_cycles
wr_bw_bytes_per_s = wr_bytes * f_clk_hz / total_cycles
rd_util = rd_beats / total_cycles
wr_util = wr_beats / total_cycles
```

## Bring-Up 顺序

### Step A: AXI-Lite Smoke

目标：证明 PS 可以访问 NPU registers。

1. 用 PYNQ 加载 `npu_pynq.bit`。
2. 从 `overlay.ip_dict` 找到 NPU base address。
3. 读取 `STATUS(0x04)`。
4. 写读 `CFG_SHAPE(0x3C)`。
5. 写 `PERF_CTRL(0x78)=1`，再写 `PERF_CTRL(0x78)=2`。
6. 读取性能寄存器，确认 AXI-Lite 路径不挂死。

### Step B: DDR/NPU Smoke

目标：证明 NPU `M_AXI` 可以通过 `S_AXI_HP0` 读写 PYNQ DDR。

1. 分配一个小 DDR/CMA buffer。
2. 在 buffer 中放一个 tiny W tile、A tile、bias block 和 result area。
3. 对 buffer 执行 `flush()`。
4. 写 `M_DIM/N_DIM/K_DIM/W_ADDR/A_ADDR/R_ADDR/BIAS_ADDR`。
5. 先用 `CFG_SHAPE=2`。
6. 启动 NPU。
7. 轮询 `STATUS.done` 并检查 `STATUS.error`。
8. 对 result buffer 执行 `invalidate()`。
9. 对比 result 和 software golden。

### Step C: Single Conv Layer

目标：把仿真 closed-loop 的一层迁移到 PYNQ runtime。

1. 加载单层 static weights、bias 和 Q24 multipliers。
2. PS runtime 按仿真 runtime 的规则 pack A tiles。
3. 运行 NPU tile GEMM。
4. PS 执行 ReLU、per-channel Q24 requant 和 scatter。
5. 对比整层输出和 exact Python。

### Step D: Full VGG Closed Loop

目标：在 PYNQ-Z2 上跑完整 RepOpt VGG。

1. 一次性加载全部 static assets。
2. 每张图只更新 input activation buffer。
3. 执行全部 Conv tile layers 和 PS post-processing。
4. 执行 pooling、classifier 和 argmax。
5. 返回 `class_id` 和 raw counters。
6. 用已知 CIFAR-10 图片对比 board output 和 exact Python。

## 验收标准

第一阶段验收：

- PYNQ 能加载匹配的 `.bit/.hwh`。
- PS 能读写 NPU AXI-Lite registers。
- NPU 能通过 HP0 读写 PYNQ allocated DDR。
- Tiny tile smoke 与 software golden 一致。
- 有效 smoke run 中 error bits 保持为 0。

第二阶段验收：

- 一张已知图片返回 expected `class_id`。
- 每张图片返回 raw counters。
- PS/host 打印 cycles、TOPS、read bandwidth、write bandwidth 和 utilization。
- 连续多张图片运行不需要重新下载 bitstream。

## 延后路线

以下路线明确延后：

- UART image packet protocol
- SPI flash image/weight loading
- Boot ROM
- PicoRV32 as board control CPU
- pure-PL SoC bring-up

只有当项目需要 non-Zynq 部署或 no-Linux FPGA 路径时，再恢复这些方向。
