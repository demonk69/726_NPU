# NPU 调试检查清单

更新时间：2026-04-28

本文用于后续每次修 RTL 时快速定位问题。当前顶层标量路径和 4x4 tile-mode GEMM 路径已通过，后续优先围绕“4-lane DMA/PPBuf 供数 -> PE array -> tile serializer -> DMA 写回”这条链路排查。

## 快速判断

```text
编译失败
  -> 检查 testbench 参数名、源列表、模块顺序

仿真 TIMEOUT
  -> 看 controller state、DMA state、AXI handshake、FIFO empty/full

结果为 0
  -> 查 DRAM 读数、PPBuf 输出、PE 输入、flush valid、FIFO 写入、DMA 写回地址

结果错位
  -> 查地址计算、burst 地址递增、result serializer 顺序

多 lane 错误
  -> 查 lane packing、valid mask、边界 tile mask、阵列输出映射
```

## 编译失败检查

1. `npu_top` 推荐使用 `PHY_ROWS/PHY_COLS`；`ROWS/COLS` 仅作为旧 testbench 兼容参数保留。
2. 顶层源列表必须包含 `rtl/array/reconfig_pe_array.v`。
3. `pe_top` 依赖 `fp16_mul/fp16_add/fp32_add`。
4. testbench 如果引用内部信号，要确认层级名仍存在。

## TIMEOUT 检查

优先看：

```text
u_ctrl.state
u_dma.load_state
u_dma.wb_state
dma_w_start/done
dma_a_start/done
dma_r_start/done
r_fifo_empty/full
m_axi_awvalid/awready
m_axi_wvalid/wready/wlast
m_axi_bvalid/bready
```

常见原因：

- DMA writeback 等 FIFO 非空，但 PE 没有产生 valid。
- AXI B channel 没响应。
- `wlast` 没在最后一拍拉高。
- controller 等待的 `done` latch 没清。
- result length 大于 FIFO 实际写入数量。

## 结果为 0 检查

按顺序排：

1. DRAM 中输入是否非零。
2. AXI R channel 是否读出非零 `m_axi_rdata`。
3. W/A PPBuf 是否写入。
4. W/A PPBuf 读侧是否输出非零。
5. `pe_w_in` 和 `pe_a_in` 是否非零。
6. `pe_en` 是否覆盖了有效数据周期。
7. `flush` 是否在 pipeline 数据到达后发出。
8. `pe_array_valid` 是否拉高。
9. `r_fifo_wr_en` 是否拉高。
10. DMA 是否把 FIFO 数据写到期望 `R_ADDR`。

## AXI Burst 检查

实现 burst 后必须检查：

- `ARLEN/AWLEN = beats - 1`。
- `ARSIZE/AWSIZE = 2` 表示 4 byte beat。
- `ARBURST/AWBURST = INCR`。
- 地址每 beat 加 4。
- 不跨 4KB 边界。
- `RLAST/WLAST` 只在最后一 beat 拉高。
- 下一个 burst 地址从上一个结束地址继续。

## 数据打包检查

INT8：

```text
低地址 -> 低 8 bit -> 第一个元素
byte 需要符号扩展到 16-bit 后进入 PE
```

FP16：

```text
每个 32-bit word 含两个 half-word
低 16 bit 是第一个 FP16
高 16 bit 是第二个 FP16
```

结果：

```text
INT8 accumulate -> signed INT32
FP16 accumulate -> FP32 bit pattern
```

## WS/OS 检查

WS：

- 权重应先 load 到 PE。
- 多个 activation 复用同一权重。
- 切换 `n_tile/k_tile` 时需要重新 load weight。
- K-split 时需要从 psum 继续累加。

OS：

- PE 内部保持当前输出 psum。
- K 完成前不能 flush 清零。
- 如果 K-split，不能每个 K tile 都清 accumulator，必须加载旧 psum。

## 4x4 tile 检查

当前 4x4 tile 应满足：

```text
# m0/n0: 当前 4x4 输出 tile 左上角全局坐标。
# r/c: tile 内部 row/col lane。
# k: GEMM 归约维度坐标。
A lane: OS row r receives A[m0+r,t-r] at physical cycle t
W lane: row r sees W[t-r,n0+c] after vertical propagation
startup bubble: row r is zero for the first r physical cycles
valid mask: edge tile inactive lane 不写回
result index: result[r*4+c] = C[m0+r,n0+c]
result count: active_rows * active_cols
```

检查点：

- `ARR_CFG[7]` 是否置 1，否则仍走标量兼容路径。
- `tile_m_base/tile_n_base` 是否按 4 递增。
- `tile_row_valid/tile_col_valid` 是否匹配 M/N 边界。
- `vec_consume` 是否每个 tile 发出 K 拍。
- `rd_vec` lane 顺序是否为 lane0 在低位。
- INT8 lane 是否符号扩展到 `DATA_W`。
- FP16 lane 是否保持 16-bit bit pattern。
- 16 个输出是否都进入 FIFO。
- serializer 顺序是否与 C 行主序一致。
- M/N 非 4 整数倍时，mask 是否阻止无效输出写回。
- `row_valid[r]` 和 `col_valid[c]` 是否分别来自 `m0+r < M`、`n0+c < N`。
- C 写回是否按每个有效 row 的 row-major 地址发起，而不是错误地假设所有 16 个 word 总是连续。

## Descriptor 检查

多层模式上线后，检查：

1. descriptor 读取地址正确。
2. descriptor 字段字节序正确。
3. `NEXT_LAYER` 时 W/A/OUT 地址切换。
4. 上一层 `ofm_addr` 是否作为下一层 `ifm_addr`。
5. `last_layer` 时才产生 network done IRQ。

## 推荐波形保存

每完成一个任务，保留：

```text
sim/wave/<task_id>_<test_name>.vcd
sim/wave/<task_id>_<test_name>.log
```

并在 [task_breakdown.md](task_breakdown.md) 中记录 PASS/FAIL。
