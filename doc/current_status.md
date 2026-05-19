# 验证状态 — 2026-05-19 (v0.5.0)

## 独立 Tile GEMM（Icarus + Verilator 双仿真器）

| 测试 | shape | M×K×N | 结果 |
|------|-------|--------|------|
| 4×4 | 4x4 | 4×4×4 | 16/16 PASS |
| 8×8 | 8x8 | 8×8×8 | 64/64 PASS |
| 16×16 | 16x16 | 16×4×16 | 256/256 PASS |
| 16×16 bias | 16x16 | 16×4×16 | 256/256 PASS |
| 16×16 K=17/18/19 | 16x16 | 16×K×16 | 256/256 PASS |
| 16×16 K=20/21/32/40 | 16x16 | 16×K×16 | 256/256 PASS |
| 16×16 K=21 bias | 16x16 | 16×21×16 | 256/256 PASS |
| 8×32 | 8x32 | 8×2×32 | 256/256 PASS |
| 8×32 bias | 8x32 | 8×2×32 | 256/256 PASS |
| 8×32 K=5 bias | 8x32 | 8×5×32 | 256/256 PASS |
| 8×32 K=21 bias | 8x32 | 8×21×32 | 256/256 PASS |
| 任意形状 | 16x16 | 11×27×12 bias | 132/132 PASS |

**回归**：12/12 standalone PASS（Icarus + Verilator 双仿真器一致）

---

## SoC 闭环测试（PicoRV32 调度 NPU）

| 测试 | 说明 | Cycles | 结果 |
|------|------|--------|------|
| RepOpt tile-window | L0, 16×16 tile, real weights | 29,983 | PASS |
| 2-layer synthetic | L0 → CPU repack → L1 | 1,453 | PASS |
| 3-layer synthetic | L0 → repack → L1 → repack → L2 | 2,224 | PASS |
| 9-layer synthetic | 9 layers with CPU repack chain | 6,922 | PASS |
| RepOpt VGG L0 | real weight, NPU + CPU bias/ReLU | 853 | PASS |
| RepOpt VGG 2-layer | L0 (4-tile) + L1 (K=576, 36 ktiles) | 9,306 | PASS |
| **RepOpt VGG e2e** | L0+L1+MaxPool→class label=3 vs PyTorch | 9,375 | PASS |

---

## RTL 改动清单 (v0.1 → v0.5)

### npu_ctrl.v
- DMA done latch merged into main FSM（消除 multi-driver）
- bias_col 5→6 bit（8×32 防环绕）
- w_addr_pass1_offset ceil(K/4)*4*16（SIMD padding）
- seq1_len_bytes_w/a 含 SIMD padding（prefetch 路径）
- K-split + bias race 修复（prefetch 移至 S_OVERLAP_COMPUTE）
- 8×32 two-pass FSM, half_en, pass_idx

### npu_top.v
- tile_bias_buf depth 16→32, tile_bias_idx 5→6 bit
- tile_cap_d1 capture 延迟 1 拍
- half_en 连接
- DATA_W=32, PPB_DEPTH=64

### soc_top.v
- NPU_DATA_W 16→32
- PPB_DEPTH 32→64

### reconfig_pe_array.v
- half_en 半阵列门控
- flush bypass for fold mode

### npu_dma.v
- R_FIFO_DEPTH 64→256

### pingpong_buf.v
- vec_abs_next [5:0]→[7:0]

### tb_npu_tile_gemm_v2.v
- AXI-Lite negedge 驱动, 读/写间 @(posedge clk)

---

## 后处理路径状态

Tile-mode 后处理（bias/ReLU/ReLU6/quant/saturate）已在 `npu_top.v` 完整接入：

```
tile_result_buf → tile_with_bias (+bias) → apply_scalar_activation (ReLU/ReLU6) → apply_scalar_quant → FIFO
```

验证：8×32 bias PASS, K-split bias PASS, RepOpt VGG CPU bias/ReLU firmware PASS。

---

## 架构决策

- **K-split**：PE accumulator 跨 k_tile 保持（pe_acc_init_en=0 in tile mode），32-bit acc 足够 K≤576 的 INT8 积累，无需外部 DRAM PSUM surface
- **调度**：验证走 PicoRV32 MMIO direct write 路径，descriptor 链可作为 CPU 指令优化，非功能必需
- **SIMD**：4-lane INT8 packed（DATA_W=32），16×16/8×32 tile 已验证
