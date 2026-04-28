# 仿真指南

更新时间：2026-04-27

本文给出当前可复现的仿真入口和后续验证顺序。Phase 1 已恢复顶层标量兼容路径；真正 4x4/16x16 并行 tile 仍是后续任务。

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
PASS=19 FAIL=0
ALL TESTS PASSED SUCCESSFULLY
```

用途：

- 验证 `pe_top` INT8/FP16 MAC。
- 验证 WS/OS 基本语义。
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
| `tb_dma_burst.v` | AXI INCR burst 地址和数据正确 | T3.4 |
| `tb_npu_tile_gemm.v` + `tb/tile4/int8_4x4x4` | 使用 T2.1 tile-pack 的 4x4 INT8 GEMM tile | T2.5 |
| `tb_npu_tile_gemm.v` + `tb/tile4/fp16_4x4x4` | 使用 T2.1 tile-pack 的 4x4 FP16 GEMM tile | T2.6 |
| `tb_gemm_ksplit.v` | K-split psum 累加正确 | T4.5 |
| `tb_desc_two_layer.v` | descriptor 多层顺序执行 | T5.4 |
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
rtl/array/reconfig_pe_array.v
rtl/power/npu_power.v
rtl/ctrl/npu_ctrl.v
rtl/axi/npu_axi_lite.v
rtl/axi/npu_dma.v
rtl/top/npu_top.v
```

旧的 `rtl/array/pe_array.v` 可以保留作对照，但当前 `npu_top` 接入的是 `reconfig_pe_array.v`。

## 验证顺序

推荐严格按下面推进：

1. 保持 PE 单元测试通过。
2. 保持 `tb_npu_scalar_smoke.v` 和 `tb_comprehensive.v` 通过。
3. 保持 `tb_pingpong_buf_vec.v` 通过。
4. 保持 `tb_npu_ctrl_tile.v` 通过。
5. 保持 `tb_npu_tile_writeback.v` 通过。
6. 保持 `tb_npu_tile_gemm.v` 的 INT8/FP16 4x4 case 通过。
7. AXI burst 正确性通过。
8. K-split GEMM 通过。
9. descriptor 两层 GEMM/FC 通过。
10. 预展开 im2col 的卷积通过。
11. 修复 SoC 编译并完成 CPU 启动 NPU smoke。
12. FPGA smoke test。

## T2.1 4x4 测试数据约定

T2.5/T2.6 的 testbench 使用预打包 A/W tile 流：

```text
A_TILE[m_tile][k][r] = A[m0+r,k]
W_TILE[n_tile][k][c] = W[k,n0+c]
```

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
