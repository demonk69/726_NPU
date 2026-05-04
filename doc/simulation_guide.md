# 仿真指南

更新时间：2026-05-03

本文给出当前可复现的仿真入口和后续验证顺序。Phase 1 已恢复顶层标量兼容路径；Phase 2 已完成可验证的 4x4 tile-mode GEMM 路径；Phase 3 已完成 AXI read/write burst、perf counters 和带宽目标测试；T4.2-T4.5 已完成独立 PSUM/OUT buffer RMW、PE accumulator init、controller k_tile loop 单测和顶层 K-split GEMM golden；T5.1-T5.5 已完成 descriptor v1 ABI、AXI-Lite descriptor 提交寄存器、descriptor fetch/decode/next-layer、INT8 OFM->IFM 串联和 IRQ/error status；T6.1 已完成 DRAM 预展开 Conv2D im2col golden 仿真；T6.2 已完成 direct scalar on-the-fly Conv2D im2col 仿真；T6.3-T6.5 已完成 direct scalar bias、ReLU/ReLU6 和 INT8 quant/saturate 后处理；T6.6 已完成两层 Conv2D E2E；T7.1 已完成 8x8/16x16 active lane 供数验证；T7.2 已完成阵列级 8x32 折叠路由验证；T7.3/T7.4 已完成 PE 级 INT8 2/4-lane SIMD 验证；T7.5 已完成 TOPS/util 性能计数器报告；SoC smoke 已恢复。更大 tile 完整写回、packed K lane 供数和 descriptor 化卷积仍是后续任务。

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
PASS=28 FAIL=0
ALL TESTS PASSED SUCCESSFULLY
```

用途：

- 验证 `pe_top` INT8/FP16 MAC。
- 验证 WS/OS 基本语义。
- 验证 INT8/FP16 accumulator init 后继续 MAC。
- 验证 T7.3/T7.4 INT8 packed 2/4-lane SIMD，以及旧 sign-extended scalar 兼容。
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
tb_npu_tile_lane_feed.v                     -> PASS，8x8/16x16 active lane observed
tb_reconfig_pe_8x32.v                       -> PASS，8x32 output order/fold route/load wrap
tb_pe_top.v                                 -> PASS，INT8 packed 2/4-lane OS/WS and scalar compatibility
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

### 8x32 folded PE array route

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
iverilog -g2012 -o sim\tb_reconfig_pe_8x32.vvp `
  rtl\pe\fp16_mul.v rtl\pe\fp16_add.v rtl\pe\fp32_add.v `
  rtl\pe\pe_top.v rtl\array\reconfig_pe_array.v `
  tb\tb_reconfig_pe_8x32.v
vvp sim\tb_reconfig_pe_8x32.vvp
```

当前结果：

```text
[PASS] 8x32 output order
[PASS] 8x32 folded activation route
[PASS] 8x32 WS load row wraps at 8
[PASS] tb_reconfig_pe_8x32
```

用途：

- 验证 `cfg_shape=2'b11` 下 16x16 物理阵列被映射为两个 8x16 半阵列。
- 验证逻辑列 0..15 来自 top half，逻辑列 16..31 来自 bottom half，`acc_out[0..31]` 顺序正确。
- 验证 top-half activation 水平链末端折入 bottom half，对应 row r -> row r+8。
- 验证 OS weight 链在 row7/row8 间断开，bottom half 从 `w_in[c]` 重新注入。
- 验证 WS weight load row 在 0..7 内回卷，同时覆盖上下两个半阵列的同名逻辑 row。

### PE INT8 2/4-lane SIMD

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_sim.ps1
```

当前结果：

```text
tb_pe_top.v -> PASS=28 FAIL=0
scripts/run_regression.ps1 -> TOTAL: 2330 PASS, 0 FAIL
```

用途：

- 验证 `pe_top` 在 `mode=0` 下把 16-bit `w_in/a_in` 解释为 packed `{lane1,lane0}`，执行两路 signed INT8 MAC。
- 验证 `pe_top` 在 `DATA_W=32/INT8_SIMD_LANES=4` 下把 `w_in/a_in` 解释为 packed `{lane3,lane2,lane1,lane0}`，执行四路 signed INT8 MAC。
- 验证 OS packed stream、WS packed weight latch、负数 lane 和 accumulator 输出。
- 验证旧 direct scalar/PPBuf sign-extended INT8 输入仍保持单 lane 兼容。
- 当前是 PE 级验证；端到端 2x/4x 吞吐仍需要 packed K lane 供数和更大 tile 写回配套。

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
- K 不是 32-bit word 对齐时，direct scalar 预取 stride 已按 word-aligned 行/列跨度推进；`25x18x3` INT8 OS 已通过。
- 当前用于功能正确性测试，不代表 4x4 tile/descriptor 高吞吐大矩阵路径已经完成。

### T6.1 Conv2D im2col case

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_conv2d_im2col_case.ps1 `
  -Mode OS -Name conv2d_im2col_int8_os_default
```

当前结果：

```text
conv2d_im2col_int8_os_default  -> ALL 75 CHECKS PASSED
conv2d_im2col_int8_ws_default  -> ALL 75 CHECKS PASSED
conv2d_im2col_fp16_os_default  -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1     -> TOTAL: 2330 PASS, 0 FAIL
```

用途：

- 用 `tb/conv2d/gen_conv2d_im2col_data.py` 生成 IFM/weight、DRAM 中的 `A_im2col[M,K]` 和 `W_col[K,N]`。
- 默认 case 是 `B=1, IFM=5x5, Cin=2, KHxKW=3x3, Cout=3, pad=1`，映射到 `M=25, K=18, N=3`。
- `expected.hex` 是 Conv2D golden；testbench 仍复用 direct matmul checker。
- 这是 T6.1 的 DRAM 预展开方案，不是 T6.2 的硬件 on-the-fly im2col。

### T6.2 Conv2D on-the-fly im2col case

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_conv2d_otf_case.ps1 `
  -Mode OS -Name conv2d_otf_int8_os_default
```

当前结果：

```text
conv2d_otf_int8_os_default  -> ALL 75 CHECKS PASSED
conv2d_otf_int8_ws_default  -> ALL 75 CHECKS PASSED
conv2d_otf_fp16_os_default  -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1  -> TOTAL: 2330 PASS, 0 FAIL
```

用途：

- DRAM 中只保存 raw NCHW IFM 和 `W_col[K,N]`，不再保存完整 `A_im2col[M,K]` 中间矩阵。
- `CTRL[8]` 启用 direct scalar on-the-fly im2col，`0x80..0x94` 提供 IFM/OFM/kernel/stride/pad/dilation 参数。
- `npu_dma` 按 `m -> b/oh/ow`、`k -> cin/kh/kw` 生成 IFM 地址，padding 或越界位置写 0，并按 INT8/FP16 打包到 A PPBuf。
- 当前该路径只覆盖 direct scalar、非 tile mode；tile/descriptor 主线仍保持现有 GEMM/tile-pack 路径。

### T6.3/T6.4 Bias and activation cases

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_matmul_case.ps1 `
  -M 3 -K 5 -N 4 -Dtype int8 -Mode OS -Bias -Activation relu

powershell -ExecutionPolicy Bypass -File scripts\run_conv2d_otf_case.ps1 `
  -Dtype int8 -Mode OS -Bias -Activation relu
```

当前结果：

```text
matmul_relu_int8_os_3x5x4        -> ALL 12 CHECKS PASSED
matmul_relu6_int8_ws_3x5x4       -> ALL 12 CHECKS PASSED
matmul_relu_fp16_os_3x4x3        -> ALL 9 CHECKS PASSED
matmul_relu6_fp16_ws_2x3x2       -> ALL 4 CHECKS PASSED
conv2d_relu_otf_int8_os_default  -> ALL 75 CHECKS PASSED
conv2d_relu6_otf_int8_ws_default -> ALL 75 CHECKS PASSED
conv2d_relu6_otf_fp16_os_default -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1       -> TOTAL: 2330 PASS, 0 FAIL
```

用途：

- `-Bias` 生成 32-bit per-output-column bias，配置 `BIAS_ADDR(0x98)` 和 `CTRL[9]`。
- `-Activation relu|relu6` 配置 `CTRL[11:10]`，后处理顺序为 accumulator -> optional bias -> activation。
- INT8 ReLU6 clamp 到 signed int32 `[0,6]`；FP16 ReLU6 clamp 到 FP32 `[0.0,6.0]`。
- 当前该路径只覆盖 direct scalar、非 tile mode；tile/descriptor 主线仍待接入后处理。

### T6.5 INT8 quant/saturate cases

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_matmul_case.ps1 `
  -M 3 -K 5 -N 4 -Dtype int8 -Mode OS -Bias -Activation relu `
  -Quant -QuantScale 3 -QuantShift 5 -QuantRound

powershell -ExecutionPolicy Bypass -File scripts\run_conv2d_otf_case.ps1 `
  -Dtype int8 -Mode OS -Bias -Activation relu `
  -Quant -QuantScale 2 -QuantShift 3 -QuantRound
```

当前结果：

```text
matmul_quant_int8_os_3x5x4           -> ALL 12 CHECKS PASSED
matmul_quant_int8_ws_3x5x4           -> ALL 12 CHECKS PASSED
conv2d_quant_otf_int8_os_default     -> ALL 75 CHECKS PASSED
conv2d_quant_otf_int8_ws_default     -> ALL 75 CHECKS PASSED
scripts/run_regression.ps1           -> TOTAL: 2330 PASS, 0 FAIL
```

用途：

- `-Quant` 写 `QUANT_CFG(0x9C)` 并启用 direct scalar INT8 quant/saturate。
- `-QuantScale <q>` 配置 signed 16-bit scale，`-QuantShift <s>` 配置 arithmetic right shift，`-QuantRound` 在 shift 前启用 signed rounding。
- 后处理顺序为 accumulator -> optional bias -> activation -> optional quantize/saturate；输出为 sign-extended signed int8 word。
- 当前该路径只覆盖 INT8 direct scalar、非 tile mode；FP16 和 tile/descriptor 主线忽略 `QUANT_CFG`。

### T6.6 two-layer Conv2D E2E

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -ExecutionPolicy Bypass -File scripts\run_conv2d_two_layer_case.ps1
```

当前结果：

```text
conv2d_two_layer_int8_os_default -> ALL 48 CHECKS PASSED
scripts/run_regression.ps1       -> TOTAL: 2330 PASS, 0 FAIL
```

用途：

- layer0 使用 raw NCHW IFM、direct scalar on-the-fly im2col、bias、ReLU 和 INT8 quant/saturate，写出 sign-extended int8 OFM。
- layer1 直接把 layer0 `R_ADDR` 作为 `A_ADDR`，验证层间量化输出能被下一层消费。
- testbench 同时检查 layer0 中间 OFM 和 layer1 最终 golden。

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
[PERF] scalar_smoke MAC_OPS=4 OPS=8 BUSY_CYCLES=34 COMPUTE_CYCLES=5 DMA_CYCLES=29 TOPS_X1E6=117 COMPUTE_UTIL_BP=8000 E2E_UTIL_BP=1176 PEAK_OPS_CYCLE=2
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
0xA0 PERF_MAC_OPS_LO
0xA4 PERF_MAC_OPS_HI
0xA8 PERF_OPS_LO      # 1 MAC = 2 ops
0xAC PERF_OPS_HI
0xB0 PERF_BUSY_CYCLES
0xB4 PERF_COMPUTE_CYCLES
0xB8 PERF_DMA_CYCLES
0xBC PERF_TOPS_X1E6   # TOPS * 1,000,000
0xC0 PERF_COMPUTE_UTIL # basis points
0xC4 PERF_E2E_UTIL     # basis points
0xC8 PERF_PEAK_OPS_CYC
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

## SoC smoke

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_soc_sim.ps1
```

当前结果：

```text
[PASS] SoC integration test PASSED
Cycles: 247
DRAM result area (0x1020): C00=19 C01=22 C10=43 C11=50
```

`run_soc_sim.ps1` 默认不生成 VCD；需要波形时使用 `-DumpVcd`，需要 CPU/NPU 详细日志时使用 `-VerboseLog`。

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
| `tb_op_counter_perf.v` | TOPS fixed-point 和 compute/e2e utilization 公式验证 | DONE/T7.5 |
| `tb_psum_out_buf.v` | PSUM/OUT buffer read-modify-write、边界 mask 和 bank 隔离 | DONE/T4.2 |
| `tb_reconfig_pe_acc_init.v` | PE array 从 per-PE psum 初始化后继续 MAC | DONE/T4.3 |
| `tb_reconfig_pe_8x32.v` | 8x32 折叠路由、32-lane 输出顺序和 WS 8-row load wrap | DONE/T7.2 |
| `tb_pe_top.v` | PE 级 INT8 packed 2/4-lane SIMD OS/WS、负数 lane 和旧 scalar 兼容 | DONE/T7.3/T7.4 |
| `tb_npu_ctrl_ksplit.v` | controller k_tile loop、K slice DMA 地址/长度、final-only writeback | DONE/T4.4 |
| `tb_npu_tile_ksplit_gemm.v` | 顶层多个 k_tile 累加结果等于未切分 GEMM golden | DONE/T4.5 |
| `tb_npu_axi_lite_desc.v` | `DESC_BASE/DESC_COUNT` 和 `CTRL[7] desc_mode` readback | DONE/T5.2 |
| `tb_npu_desc_two_layer.v` | descriptor fetch/decode 和多 descriptor 顺序执行 | DONE/T5.3 |
| `tb_npu_desc_ofm_chain.v` | descriptor bit23 下 layer0 OFM 作为 layer1 IFM | DONE/T5.4 |
| `tb_npu_tile_gemm.v` + `tb/tile4/int8_4x4x4` | 使用 T2.1 tile-pack 的 4x4 INT8 GEMM tile | DONE/T2.5 |
| `tb_npu_tile_gemm.v` + `tb/tile4/fp16_4x4x4` | 使用 T2.1 tile-pack 的 4x4 FP16 GEMM tile | DONE/T2.6 |
| `scripts/run_conv2d_im2col_case.ps1` | DRAM 预展开 Conv2D im2col 后复用 direct matmul checker 对 Conv2D golden | DONE/T6.1 |
| `scripts/run_conv2d_otf_case.ps1` | raw IFM on-the-fly im2col 后复用 direct matmul checker 对 Conv2D golden | DONE/T6.2 |
| `scripts/run_matmul_case.ps1 -Bias -Activation ...` | direct scalar GEMM bias/ReLU/ReLU6 后处理 | DONE/T6.3/T6.4 |
| `scripts/run_conv2d_otf_case.ps1 -Bias -Activation ...` | direct scalar Conv2D on-the-fly bias/ReLU/ReLU6 后处理 | DONE/T6.3/T6.4 |
| `scripts/run_matmul_case.ps1 -Quant ...` | direct scalar GEMM INT8 quant/saturate 后处理 | DONE/T6.5 |
| `scripts/run_conv2d_otf_case.ps1 -Quant ...` | direct scalar Conv2D on-the-fly INT8 quant/saturate 后处理 | DONE/T6.5 |
| `scripts/run_conv2d_two_layer_case.ps1` | 两层 Conv2D 端到端，layer0 OFM 作为 layer1 IFM | DONE/T6.6 |
| `scripts/run_soc_sim.ps1` | CPU 配置 NPU 并读回结果 | DONE/SoC smoke |

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
12. 保持 `tb_reconfig_pe_8x32.v` 通过。
13. K-split GEMM 通过。
14. AXI-Lite descriptor 提交寄存器通过。
15. descriptor 两任务顺序执行通过。
16. descriptor 层间 OFM/IFM 串联通过。
17. 预展开 im2col 的卷积通过。
18. on-the-fly im2col 的卷积通过。
19. 两层 Conv2D E2E 通过。
20. SoC 编译和 CPU 启动 NPU smoke 通过。
21. FPGA smoke test。

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
