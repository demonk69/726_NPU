# 仿真指南

更新时间：2026-04-29

本文给出当前可复现的仿真入口和后续验证顺序。Phase 1 已恢复顶层标量兼容路径；Phase 2 已完成可验证的 4x4 tile-mode GEMM 路径；Phase 3 已完成 AXI read/write burst、perf counters 和带宽目标测试；T4.2-T4.5 已完成独立 PSUM/OUT buffer RMW、PE accumulator init、controller k_tile loop 单测和顶层 K-split GEMM golden；T5.1-T5.4 已完成 descriptor v1 ABI、AXI-Lite descriptor 提交寄存器、descriptor fetch/decode/next-layer 和 INT8 OFM->IFM 串联。16x16/8x32 高吞吐、外部 PSUM descriptor 流和多层卷积仍是后续任务。

## 工具

当前机器可用：

```powershell
E:\iverilog\bin\iverilog.exe
E:\iverilog\bin\vvp.exe
```

PowerShell 中建议先加入 PATH：

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
```

## 当前通过入口

### PE 单元测试

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

当前结果：

```text
PASS=22 FAIL=0
ALL TESTS PASSED SUCCESSFULLY
```

用途：

- 验证 `pe_top` INT8/FP16 MAC。
- 验证 WS/OS 基本语义。
- 验证 INT8/FP16 accumulator init 后继续 MAC。
- 作为后续改 PE 前后的回归基线。

### 顶层标量 smoke test

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_scalar_smoke.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v rtl\array\reconfig_pe_array.v rtl\power\npu_power.v `
  rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\top\npu_top.v `
  tb\tb_npu_scalar_smoke.v
vvp sim\tb_npu_scalar_smoke.vvp
```

当前结果：

```text
[PASS] tb_npu_scalar_smoke: scalar INT8 OS result=300 and cfg_shape latched
```

用途：

- 验证 `npu_top` 可完成 W/A DMA 读入、PPBuf 供数、标量 PE 计算、FIFO 写入和 DMA 写回。
- 验证 `CFG_SHAPE` 在 start 后被锁存，不受运行中寄存器写入影响。

### Ping-Pong Buffer 4-lane vector test

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_pingpong_buf_vec.vvp `
  rtl\buf\pingpong_buf.v tb\tb_pingpong_buf_vec.v
vvp sim\tb_pingpong_buf_vec.vvp
```

当前结果：

```text
[PASS] tb_pingpong_buf_vec
```

用途：

- 验证 INT8 一个 32-bit word 可输出 4 个 sign-extended lane。
- 验证 FP16 两个 32-bit word 可输出 4 个 16-bit lane。
- 验证 `rd_vec_en` 一拍消费 4 个 lane。

### NPU controller 4x4 tile planner test

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_ctrl_tile.vvp `
  rtl\ctrl\npu_ctrl.v tb\tb_npu_ctrl_tile.v
vvp sim\tb_npu_ctrl_tile.vvp
```

当前结果：

```text
[PASS] tb_npu_ctrl_tile
```

用途：

- 验证 `ARR_CFG[7]` 可启用 4x4 tile planner。
- 验证 M=5、N=6 时 tile mask 为 4x4、4x2、1x4、1x2。
- 验证 `vec_consume` 在每个 tile 中按 K 周期发出。

### `tb_comprehensive.v`

当前手工补齐源列表后运行结果为：

```text
ALL 28 TESTS PASSED
```

该测试当前可作为 Phase 1 顶层标量兼容路径的回归基线。

### 4x4 tile writeback 和 GEMM

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_regression.ps1
```

当前已覆盖：

```text
tb_npu_tile_writeback.v + 4x4 K=1 INT8 tile -> PASS
tb_npu_tile_gemm.v + tb/tile4/int8_4x4x4   -> PASS，ALL 16 CHECKS PASSED
tb_npu_tile_gemm.v + tb/tile4/fp16_4x4x4   -> PASS，ALL 16 CHECKS PASSED
tb_npu_axi_lite_desc.v                      -> PASS
```

### DMA read burst

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_dma_read_burst.vvp `
  rtl\common\fifo.v rtl\axi\npu_dma.v tb\tb_dma_read_burst.v
vvp sim\tb_dma_read_burst.vvp
```

当前结果：

```text
[PASS] tb_dma_read_burst: INCR read bursts and 4KB split passed
```

用途：

- 验证 `m_axi_arlen = burst_beats - 1`，不是固定 0。
- 验证 `m_axi_arburst = INCR`，连续 burst 地址递增。
- 验证读 burst 不跨 4KB 边界。

### DMA write burst

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_dma_write_burst.vvp `
  rtl\common\fifo.v rtl\axi\npu_dma.v tb\tb_dma_write_burst.v
vvp sim\tb_dma_write_burst.vvp
```

当前结果：

```text
[PASS] tb_dma_write_burst: INCR write bursts and 4KB split passed
```

用途：

- 验证 `m_axi_awlen = burst_beats - 1`，不是整笔写回固定一个 burst。
- 验证 `m_axi_awburst = INCR`，连续写 burst 地址递增。
- 验证写 burst 不跨 4KB 边界。
- 验证 `WLAST` 只在当前 burst 最后一拍拉高，结果数据连续写回。

### DMA mixed burst

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_dma_burst.vvp `
  rtl\common\fifo.v rtl\axi\npu_dma.v tb\tb_dma_burst.v
vvp sim\tb_dma_burst.vvp
```

当前结果：

```text
[PASS] tb_dma_burst: mixed 8/16-beat read/write burst data passed
```

用途：

- 验证同一轮请求内 W/A read DMA 和 result writeback DMA 可并行推进。
- 验证读侧 16-beat W burst、8-beat A burst 的地址和数据正确。
- 验证写侧 24 beats 被切成 16-beat + 8-beat，且写回数据连续正确。

### DMA bandwidth utilization

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_dma_perf.vvp `
  rtl\common\fifo.v rtl\axi\npu_dma.v tb\tb_dma_perf.v
vvp sim\tb_dma_perf.vvp
```

当前结果：

```text
[PERF] read  beats=256 cycles=301 bursts=16 util=85.04% bw=3.401 B/cyc
[PERF] write beats=256 cycles=331 bursts=16 util=77.34% bw=3.093 B/cyc
[INFO] write utilization is below 80% because the current DMA issues one outstanding write burst and waits for B response before the next AW.
[PASS] tb_dma_perf: bandwidth utilization target test completed
```

用途：

- 验证长 burst 场景下 read data-channel 达到 80% 以上。
- 验证 write data-channel 达到 60% 以上，并解释当前 single-outstanding/B-response 间隔导致未达到 80%。

### PSUM/OUT buffer RMW

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_psum_out_buf.vvp `
  rtl\buf\psum_out_buf.v tb\tb_psum_out_buf.v
vvp sim\tb_psum_out_buf.vvp
```

当前结果：

```text
[PASS] tb_psum_out_buf: K-split read-modify-write accumulation passed
```

用途：

- 验证 2-bank 4x4 tile-local PSUM/OUT 存储。
- 验证 first k_tile load 与后续 k_tile read-modify-write 累加。
- 验证边界 tile `valid_mask`：invalid lane 读 0、`rvalid=0`、写入被忽略。
- 验证 bank isolation 和 same-address write conflict 标志。

### PE array accumulator init

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_reconfig_pe_acc_init.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v `
  rtl\pe\pe_top.v rtl\array\reconfig_pe_array.v `
  tb\tb_reconfig_pe_acc_init.v
vvp sim\tb_reconfig_pe_acc_init.vvp
```

当前结果：

```text
[PASS] tb_reconfig_pe_acc_init: 4x4 PE array accumulator init continued MAC passed
```

用途：

- 验证 `reconfig_pe_array` 的 per-PE `acc_init` 和 `acc_init_mask`。
- 验证 4x4 OS array 从不同 psum 初值继续 MAC。
- 验证输出仍按 row-major `result[r*4+c]`。

### Controller K-split loop

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_ctrl_ksplit.vvp `
  rtl\ctrl\npu_ctrl.v tb\tb_npu_ctrl_ksplit.v
vvp sim\tb_npu_ctrl_ksplit.vvp
```

当前结果：

```text
[PASS] tb_npu_ctrl_ksplit
```

用途：

- 用 `PPB_DEPTH=4, K=10` 强制切出 4/4/2 三个 k_tile。
- 验证 A/W load 地址分别为 `base+0x0/base+0x10/base+0x20`，长度为 16/16/8 byte。
- 验证总 `vec_consume=10`。
- 验证中间 k_tile 不 flush/writeback，最后 k_tile 后才产生 4 个 row-wise writeback。

### Controller OS/WS direct dataflow

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_ctrl_dataflow_modes.vvp `
  rtl\ctrl\npu_ctrl.v `
  tb\tb_npu_ctrl_dataflow_modes.v
vvp sim\tb_npu_ctrl_dataflow_modes.vvp
```

当前结果：

```text
[PASS] tb_npu_ctrl_dataflow_modes: OS and WS control branches passed
```

用途：

- 验证 direct scalar OS 分支 `pe_stat=1`，且不产生 `pe_load_w`。
- 验证 direct scalar WS 分支 `pe_stat=0`，且 `pe_load_w` 覆盖 K 个周期。
- 避免只看 tile/descriptor OS 主线时漏掉 WS controller 分支。

### Custom direct matmul case

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_matmul_case.ps1 `
  -M 32 -K 32 -N 32 -Dtype int8 -Mode OS -Name custom_int8_os_32x32x32
```

当前结果：

```text
custom_int8_os_32x32x32 -> ALL 1024 CHECKS PASSED
custom_int8_ws_32x32x32 -> ALL 1024 CHECKS PASSED
custom_fp16_os_16x16x16 -> ALL 256 CHECKS PASSED
```

用途：

- 自动生成 A/B/C golden、DRAM image 和 `test_params.vh`。
- 编译并运行 direct scalar matmul 路径，可选 `-Dtype int8/fp16` 和 `-Mode OS/WS`。
- 当前用于功能正确性测试，不代表 4x4 tile/descriptor 高吞吐大矩阵路径已经完成。

### Top K-split GEMM golden

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_tile_ksplit_gemm.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v rtl\buf\psum_out_buf.v rtl\array\reconfig_pe_array.v `
  rtl\power\npu_power.v rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\top\npu_top.v `
  tb\tb_npu_tile_ksplit_gemm.v
vvp sim\tb_npu_tile_ksplit_gemm.vvp
```

当前结果：

```text
[PASS] tb_npu_tile_ksplit_gemm: ALL 28 CHECKS PASSED
```

用途：

- 用 `PPB_DEPTH=4, K=10` 在 `npu_top` 端到端强制切出 4/4/2 三个 k_tile。
- 验证真实 DMA/PPBuf 延迟下的 6 个 W/A AXI read burst。
- 验证中间 k_tile 不写回，最终只产生 4 个 row-wise writeback。
- 验证 16 个 `C[r,c]` 等于未切分 4x4x10 INT8 OS GEMM golden。

### AXI-Lite descriptor registers

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_axi_lite_desc.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v rtl\buf\psum_out_buf.v rtl\array\reconfig_pe_array.v `
  rtl\power\npu_power.v rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\top\npu_top.v `
  tb\tb_npu_axi_lite_desc.v
vvp sim\tb_npu_axi_lite_desc.vvp
```

当前结果：

```text
[PASS] tb_npu_axi_lite_desc: descriptor regs, IRQ, STATUS, and ERR_STATUS W1C passed
```

用途：

- 验证 `DESC_BASE(0x40)` 和 `DESC_COUNT(0x44)` 复位为 0、可写入、可读回。
- 验证 `CTRL[7] desc_mode` 可读写，`CTRL[6] irq_clr` 仍保持 W1C 读 0。
- 验证 `STATUS(0x04)` busy/done/error 读回、done/error IRQ pending、`INT_CLR(0x0C)`/`CTRL[6]` 清 pending。
- 验证 `ERR_STATUS(0x74)` 读回和写 1 清除请求。

### Controller error status

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_ctrl_error_status.vvp `
  rtl\ctrl\npu_ctrl.v `
  tb\tb_npu_ctrl_error_status.v
vvp sim\tb_npu_ctrl_error_status.vvp
```

当前结果：

```text
[PASS] tb_npu_ctrl_error_status: controller descriptor errors passed
```

用途：

- 验证 descriptor mode 下 `DESC_COUNT=0` 会置 `ERR_STATUS[0]`。
- 验证 unsupported descriptor 会置 `ERR_STATUS[1]`。
- 验证 `DESC_COUNT` 耗尽但 `next_desc!=0` 且未 `LAST_LAYER` 会置 `ERR_STATUS[2]`。
- 验证错误状态可通过 W1C 清除。

### Descriptor two-layer sequence

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_desc_two_layer.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v rtl\buf\psum_out_buf.v rtl\array\reconfig_pe_array.v `
  rtl\power\npu_power.v rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\top\npu_top.v `
  tb\tb_npu_desc_two_layer.v
vvp sim\tb_npu_desc_two_layer.vvp
```

当前结果：

```text
[PASS] tb_npu_desc_two_layer: ALL 48 CHECKS PASSED
```

用途：

- 验证 `CTRL[7] desc_mode` 下 controller 从 `DESC_BASE` 发起 64-byte descriptor fetch。
- 验证 descriptor v1 的 `M/N/K/ifm_addr/weight_addr/ofm_addr/desc_ctrl/next_desc` 映射到当前 4x4 OS tile GEMM 语义。
- 验证两个 descriptor 顺序执行，第二个 `LAST_LAYER=1` 后才置 `STATUS.done`。
- 验证 2 次 descriptor fetch、4 次 W/A tile read、8 次 row-wise writeback 和 32 个输出结果。

### Descriptor OFM->IFM chain

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_desc_ofm_chain.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v rtl\buf\psum_out_buf.v rtl\array\reconfig_pe_array.v `
  rtl\power\npu_power.v rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\top\npu_top.v `
  tb\tb_npu_desc_ofm_chain.v
vvp sim\tb_npu_desc_ofm_chain.vvp
```

当前结果：

```text
[PASS] tb_npu_desc_ofm_chain: ALL 55 CHECKS PASSED
```

用途：

- 验证 `desc_ctrl[23] IFM_FROM_PREV_OFM`。
- 验证 `npu_ctrl` 记录上一层 `ofm_addr`，并在下一层覆盖 A 源地址。
- 验证 `npu_dma` 从上一层 row-major 32-bit OFM gather `A[m0+r,k]`，并 repack 成 4-lane INT8 A tile stream。
- 验证两层 4x4x4 INT8 OS GEMM 串联 golden。

### AXI perf counters

`tb_npu_scalar_smoke.v` 现在同时检查 AXI perf counters：

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_npu_scalar_smoke.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v rtl\pe\pe_top.v `
  rtl\common\fifo.v rtl\common\axi_monitor.v rtl\common\op_counter.v `
  rtl\buf\pingpong_buf.v rtl\array\reconfig_pe_array.v rtl\power\npu_power.v `
  rtl\ctrl\npu_ctrl.v rtl\axi\npu_axi_lite.v rtl\axi\npu_dma.v rtl\top\npu_top.v `
  tb\tb_npu_scalar_smoke.v
vvp sim\tb_npu_scalar_smoke.vvp
```

当前结果：

```text
[PASS] tb_npu_scalar_smoke: scalar INT8 OS result=300, cfg_shape latched, perf counters valid
```

AXI-Lite 性能寄存器：

```text
0x48 PERF_CYCLES
0x4C PERF_RD_BEATS
0x50 PERF_WR_BEATS
0x54 PERF_RD_BYTES
0x58 PERF_WR_BYTES
0x5C PERF_RD_BW       # bytes/cycle x1000
0x60 PERF_WR_BW       # bytes/cycle x1000
0x64 PERF_RD_UTIL     # basis points
0x68 PERF_WR_UTIL     # basis points
0x6C PERF_RD_BURSTS
0x70 PERF_WR_BURSTS
```

变量含义：

```text
# M/N/K 是 GEMM 维度；卷积中 M=batch*OH*OW，N=Cout，K=Cin*KH*KW。
# m_tile/n_tile 是 4x4 输出 tile 编号。
# r/c 是 tile 内部 row/col lane。
# k 是归约维度坐标。
A_TILE[m_tile][k][r] = A[m0+r,k]
W_TILE[n_tile][k][c] = W[k,n0+c]
```

`A_TILE[k][0..3]` 是 PPBuf 每个逻辑 `k` 周期输出的 4-lane vector。进入 PE array 前，`npu_top` 会把 A lane1/2/3 分别延迟 1/2/3 拍；因此前几个物理周期包含 bubble，这正是 OS 图里的错拍输入。

### `run_full_sim.ps1`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_full_sim.ps1
```

当前结果：编译成功，仿真完成。

## 当前失败入口

### `run_soc_sim.ps1`

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_soc_sim.ps1
```

当前已修复项目根目录和旧 `.ROWS/.COLS` 问题，但 SoC 编译仍失败，主要错误类别是：

```text
dram_model.v: unable to bind axi_arlen
soc_top.v: PicoRV32 PCPI ports do not match the referenced CPU module
```

该问题归入 SoC 集成阶段，不阻塞 Phase 2 的 4x4 tile 数据通路设计。

## 推荐新增测试

按任务顺序新增：

| 测试 | 目的 | 依赖任务 |
|---|---|---|
| `tb_npu_scalar_smoke.v` | 一个非零 INT8 dot product 写回正确 | T1.3 |
| `tb_dma_read_burst.v` | DMA 读侧 INCR burst 地址和数据正确 | DONE/T3.1 |
| `tb_dma_write_burst.v` | DMA 写侧 INCR burst 地址和数据正确 | DONE/T3.2 |
| `tb_npu_scalar_smoke.v` | AXI perf counters 可读且计数非零/匹配标量场景 | DONE/T3.3 |
| `tb_dma_burst.v` | 混合读写 8/16 beat burst 数据正确 | DONE/T3.4 |
| `tb_dma_perf.v` | 长 burst 带宽利用率目标测试 | DONE/T3.5 |
| `tb_psum_out_buf.v` | PSUM/OUT buffer read-modify-write、边界 mask 和 bank 隔离 | DONE/T4.2 |
| `tb_reconfig_pe_acc_init.v` | PE array 从 per-PE psum 初始化后继续 MAC | DONE/T4.3 |
| `tb_npu_ctrl_ksplit.v` | controller k_tile loop、K slice DMA 地址/长度、final-only writeback | DONE/T4.4 |
| `tb_npu_tile_ksplit_gemm.v` | 顶层多个 k_tile 累加结果等于未切分 GEMM golden | DONE/T4.5 |
| `tb_npu_axi_lite_desc.v` | `DESC_BASE/DESC_COUNT` 和 `CTRL[7] desc_mode` readback | DONE/T5.2 |
| `tb_npu_desc_two_layer.v` | descriptor fetch/decode 和多 descriptor 顺序执行 | DONE/T5.3 |
| `tb_npu_desc_ofm_chain.v` | descriptor bit23 下 layer0 OFM 作为 layer1 IFM | DONE/T5.4 |
| `tb_npu_tile_gemm.v` + `tb/tile4/int8_4x4x4` | 使用 T2.1 tile-pack 的 4x4 INT8 GEMM tile | DONE/T2.5 |
| `tb_npu_tile_gemm.v` + `tb/tile4/fp16_4x4x4` | 使用 T2.1 tile-pack 的 4x4 FP16 GEMM tile | DONE/T2.6 |
| `tb_conv2d_relu.v` | 卷积 + 激活 | T6.4 |
| `tb_soc_smoke.v` | CPU 配置 NPU 并读回结果 | S2 |

## RTL 源列表建议

最小 NPU 顶层仿真源列表应包含：

```text
rtl/pe/fp16_mul.v
rtl/pe/fp16_add.v
rtl/pe/fp32_add.v
rtl/pe/pe_top.v
rtl/common/fifo.v
rtl/common/axi_monitor.v
rtl/common/op_counter.v
rtl/buf/pingpong_buf.v
rtl/buf/psum_out_buf.v
rtl/array/reconfig_pe_array.v
rtl/power/npu_power.v
rtl/ctrl/npu_ctrl.v
rtl/axi/npu_axi_lite.v
rtl/axi/npu_dma.v
rtl/top/npu_top.v
```

旧的 `rtl/array/pe_array.v` 已删除；当前 `npu_top` 接入的是 `reconfig_pe_array.v`。

## 验证顺序

推荐严格按下面推进：

1. 保持 PE 单元测试通过。
2. 保持 `tb_npu_scalar_smoke.v` 和 `tb_comprehensive.v` 通过。
3. 保持 `tb_pingpong_buf_vec.v` 通过。
4. 保持 `tb_npu_ctrl_tile.v` 通过。
5. 保持 `tb_npu_tile_writeback.v` 通过。
6. 保持 `tb_npu_tile_gemm.v` 的 INT8/FP16 4x4 case 通过。
7. 保持 `tb_dma_read_burst.v` 通过。
8. 保持 `tb_dma_write_burst.v` 通过。
9. 保持 `tb_dma_burst.v` 通过。
10. 保持 `tb_dma_perf.v` 通过。
11. 保持 AXI perf counters 可读，beat/byte/cycle/utilization 有效。
12. K-split GEMM 通过。
13. AXI-Lite descriptor 提交寄存器通过。
14. descriptor 两任务顺序执行通过。
15. descriptor 层间 OFM/IFM 串联通过。
16. 预展开 im2col 的卷积通过。
17. 修复 SoC 编译并完成 CPU 启动 NPU smoke。
18. FPGA smoke test。

## T2.1 4x4 测试数据约定

T2.5/T2.6 的 testbench 使用预打包 A/W tile 流：

```text
A_TILE[m_tile][k][r] = A[m0+r,k]
W_TILE[n_tile][k][c] = W[k,n0+c]
```

物理输入不是从第 0 拍开始 4 个 A row 全部有效；row `r` 实际收到的是 `A_TILE[t-r][r]`，越界时为 0。

INT8 4-lane vector 用一个 32-bit word：

```text
word = {lane3, lane2, lane1, lane0}
```

FP16 4-lane vector 用两个 32-bit word：

```text
word0 = {lane1, lane0}
word1 = {lane3, lane2}
```

输出检查按 C row-major 地址：

```text
C_ADDR(r,c) = R_ADDR + ((m0+r) * N + (n0+c)) * 4
```

## 波形建议

顶层调试优先看：

```text
u_ctrl.state
u_ctrl.tile_i/tile_j or future m_tile/n_tile/k_tile
u_ctrl.tile_m_base/tile_n_base
u_ctrl.tile_row_valid/tile_col_valid
u_ctrl.vec_consume/tile_k_cycle
u_dma.load_state
u_dma.wb_state
m_axi_arvalid/arready/araddr/arlen
m_axi_rvalid/rready/rlast/rdata
m_axi_awvalid/awready/awaddr/awlen
m_axi_wvalid/wready/wlast/wdata
u_w_ppb.rd_data/wr_en/rd_en
u_a_ppb.rd_data/wr_en/rd_en
u_w_ppb.rd_vec/rd_vec_en/rd_vec_valid
u_a_ppb.rd_vec/rd_vec_en/rd_vec_valid
pe_w_in
pe_a_in
pe_array_valid
pe_array_result
r_fifo_wr_en
r_fifo_din
```

如果结果为 0，先确认：

1. DRAM 读出的 W/A 是否非零。
2. PPBuf 读侧是否真的输出非零。
3. PE 输入边界是否拿到非零。
4. `flush` 时 valid 是否到达。
5. 结果 FIFO 是否写入。
6. DMA 写回地址是否正确。
