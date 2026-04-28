# SoC 集成计划

更新时间：2026-04-27

本文描述 PicoRV32 + NPU 的目标集成方式和当前需要修复的接口问题。

## 目标结构

```text
PicoRV32
  |
  | iomem
  v
AXI-Lite bridge
  |
  +--> NPU register file
  |
  +--> SRAM/DRAM mapped IO

NPU DMA AXI4 master
  |
  v
shared DRAM
```

CPU 职责：

1. 准备输入、权重、descriptor。
2. 写 NPU 配置寄存器。
3. 启动 NPU。
4. 轮询 `STATUS.done` 或等待 IRQ。
5. 读取输出或进入下一任务。

NPU 职责：

1. 根据寄存器或 descriptor 发起 DMA。
2. 执行 GEMM/conv tile。
3. 写回结果。
4. 更新状态和中断。

## 地址映射建议

| 地址范围 | 设备 | 访问者 | 说明 |
|---|---|---|---|
| `0x0000_0000` 起 | SRAM | CPU | 指令和小数据 |
| `0x0100_0000` 起 | DRAM | CPU + NPU DMA | 输入、权重、descriptor、输出 |
| `0x0200_0000` 起 | NPU regs | CPU | AXI-Lite 配置窗口 |

实际 testbench 可以继续使用较小内存模型，但文档和固件应统一使用同一套地址宏。

## CPU 编程流程

### 旧寄存器直配模式

适合早期 GEMM smoke test：

```c
npu_write(M_DIM, m);
npu_write(N_DIM, n);
npu_write(K_DIM, k);
npu_write(A_ADDR, a_addr);
npu_write(W_ADDR, w_addr);
npu_write(R_ADDR, r_addr);
npu_write(CFG_SHAPE, shape);
npu_write(CTRL, start | dtype | dataflow);
while ((npu_read(STATUS) & DONE) == 0) {}
```

### Descriptor 模式

适合多层网络：

```c
prepare_desc_list(desc_base);
npu_write(DESC_BASE, desc_base);
npu_write(DESC_COUNT, num_layers);
npu_write(CTRL, START | DESC_MODE | IRQ_EN);
wait_irq_or_poll_done();
```

## 当前接口问题

`rtl/soc/soc_top.v` 中实例化 `npu_top` 的旧 `.ROWS/.COLS` 问题已经修复。后续推荐继续使用物理阵列参数：

```verilog
.PHY_ROWS(...)
.PHY_COLS(...)
```

SoC 当前仍不能作为通过基线，`scripts/run_soc_sim.ps1` 的剩余编译问题是：

```text
dram_model.v: unable to bind axi_arlen
soc_top.v: PicoRV32 PCPI ports do not match the referenced CPU module
```

修复顺序：

1. 对齐 `dram_model` 的 AXI 端口和内部信号命名，确认是否需要接入 `arlen/awlen`。
2. 对齐 PicoRV32 参考核实际暴露的 PCPI 端口，或关闭 SoC 中未使用的 PCPI 连接。
3. 先运行 CPU 写寄存器 + NPU done 的空任务测试。
4. 再运行 1-output INT8 dot product。
5. 最后运行 4x4 GEMM tile。

## AXI-Lite 桥

桥接要求：

- CPU 写寄存器必须产生完整 AXI-Lite AW/W/B 握手。
- CPU 读寄存器必须产生 AR/R 握手。
- 写 `CTRL.start` 后，CPU 可轮询 `STATUS.busy/done`。
- IRQ pending 可通过 `INT_CLR` 或 `CTRL.irq_clear` 清除。

## DMA 与共享 DRAM

目标数据流：

```text
CPU writes input/weight/desc to DRAM
NPU DMA reads DRAM
NPU computes
NPU DMA writes output to DRAM
CPU reads output or checks PASS marker
```

需要注意：

1. CPU 和 DMA 同时访问 DRAM 时需要仲裁或双端口模型。
2. testbench DRAM 模型要正确处理 AXI burst。
3. NPU DMA 地址应使用字节地址，DRAM 内部下标转换统一封装。
4. descriptor、A/W/OUT 地址建议 4 字节对齐；burst 优化后建议更高对齐。

## SoC 验证阶段

| 阶段 | 目标 | 验收 |
|---|---|---|
| S0 | CPU 能读写 NPU 寄存器 | 读回配置值正确 |
| S1 | CPU 启动空 NPU 任务 | `done` 或 IRQ 正确 |
| S2 | CPU 启动标量 dot product | DRAM 输出等于 golden |
| S3 | CPU 启动 4x4 GEMM | 16 个输出正确 |
| S4 | CPU 提交 descriptor list | 两层任务连续完成 |
| S5 | FPGA smoke test | 板上读回结果正确 |

## 固件建议

固件应保留一组最小函数：

```c
static inline void npu_write(uint32_t off, uint32_t val);
static inline uint32_t npu_read(uint32_t off);
void npu_start_gemm(uint32_t m, uint32_t n, uint32_t k,
                    uint32_t a, uint32_t w, uint32_t r,
                    uint32_t dtype, uint32_t dataflow);
int npu_wait_done(uint32_t timeout);
```

后续 descriptor 模式上线后，再加入：

```c
void npu_submit_desc(uint32_t desc_base, uint32_t count);
```
