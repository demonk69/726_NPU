# 验证状态 — 2026-05-13

## Icarus 12.0 通过

| 测试 | shape | M×K×N | k_tile | 结果 |
|------|-------|--------|--------|------|
| 4×4 | 4x4 | 4×4×4 | 1 | 16/16 PASS |
| 8×8 | 8x8 | 8×4×8 | 1 | 64/64 PASS |
| 16×16 | 16x16 | 16×4×16 | 1 | 256/256 PASS |
| 16×16 K=2 | 16x16 | 16×2×16 | 1 | 256/256 PASS |
| 16×16 K=3 | 16x16 | 16×3×16 | 1 | 256/256 PASS |
| 16×16 K=5 | 16x16 | 16×5×16 | 1 | 256/256 PASS |
| 16×16 K=7 | 16x16 | 16×7×16 | 1 | 256/256 PASS |
| 16×16 K=20 | 16x16 | 16×20×16 | 2 (16+4) | 256/256 PASS |
| 16×16 K=21 | 16x16 | 16×21×16 | 2 (16+5) | 256/256 PASS |
| 16×16 K=32 | 16x16 | 16×32×16 | 2 (16+16) | 256/256 PASS |
| 16×16 K=40 | 16x16 | 16×40×16 | 3 (16+16+8) | 256/256 PASS |
| 16×16 M=24 | 16x16 | 24×4×16 | 1 | 384/384 PASS |
| 16×16 N=20 | 16x16 | 16×4×20 | 1 | 320/320 PASS |
| 8×32 | 8x32 | 8×4×32 | 1 (two-pass) | 256/256 PASS |

## Verilator 验证命令

```bash
cd ~/726_NPU/tb/tile4
bash run_all.sh
```

单条测试：
```bash
cd ~/726_NPU/tb/tile4

# 16×16
verilator --binary +incdir+test_16x16 --top-module tb_npu_tile_gemm_v2 \
  --timing -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-PINMISSING -Wno-INITIALDLY \
  ../../rtl/pe/fp16_mul.v ../../rtl/pe/fp16_add.v ../../rtl/pe/fp32_add.v ../../rtl/pe/pe_top.v \
  ../../rtl/common/fifo.v ../../rtl/common/axi_monitor.v ../../rtl/common/op_counter.v \
  ../../rtl/buf/pingpong_buf.v ../../rtl/buf/psum_out_buf.v \
  ../../rtl/array/reconfig_pe_array.v ../../rtl/power/npu_power.v \
  ../../rtl/ctrl/npu_ctrl.v ../../rtl/axi/npu_axi_lite.v ../../rtl/axi/npu_dma.v \
  ../../rtl/top/npu_top.v ../tb_npu_tile_gemm_v2.v
./obj_dir/Vtb_npu_tile_gemm_v2

# 8×32
verilator --binary +incdir+test_8x32 --top-module tb_npu_tile_gemm_v2 \
  --timing -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-PINMISSING -Wno-INITIALDLY \
  ../../rtl/pe/fp16_mul.v ... ../../rtl/top/npu_top.v ../tb_npu_tile_gemm_v2.v
./obj_dir/Vtb_npu_tile_gemm_v2
```

## 本次修复清单

| 文件 | 改动 | 原因 |
|------|------|------|
| `tb/tb_npu_tile_gemm_v2.v` | AXI-Lite 写入用 negedge 驱动, 读/写间插 @(posedge clk) | Verilator NB 调度竞态 |
| `rtl/axi/npu_dma.v` | R_FIFO_DEPTH 64→256 | 16×16 tile 256 entries 超出 FIFO 容量, 指针环绕 |
| `rtl/buf/pingpong_buf.v` | `vec_abs_next` [5:0]→[7:0] | 16×4=64 超出 6-bit, 多 fire 时截断为 0 |
| `rtl/array/reconfig_pe_array.v` | half_en 半阵列门控 + flush bypass | 8×32 two-pass 分时启用 |
| `rtl/ctrl/npu_ctrl.v` | 8×32 two-pass FSM, W_ADDR2, A reload, pass_idx | 8×32 权重分两次加载 |
| `rtl/top/npu_top.v` | half_en 连接, capture 延迟 1 拍 | acc_out NB 对齐 |
| `tb/tile4/gen_multi_shape_data.py` | K-split 分段打包, packed_pad, W 半阵列拆分 | K 非 4 倍数 / large K 数据对齐 |
