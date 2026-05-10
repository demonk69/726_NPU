# DeepSeek 修改记录 — 2026-05-09

本文记录由 DeepSeek AI 辅助完成的所有 RTL 修改、bug 修复和测试验证。

---

## 修改动机

原始设计存在三条硬编码障碍，阻止 4x4 以外的 tile 形态进行完整的端到端 GEMM 验证：

1. **控制器输出被钳死在 4**：`npu_ctrl.v` 中的 `TILE4_LANES=32'd4` 和 `tile_active_rows/cols` clamp
2. **顶层 serializer 硬编码 16 个输出**：`npu_top.v` 的 `tile_result_buf [0:15]`、2-bit 计数器
3. **PE 阵列只输出底部一行**：`reconfig_pe_array.v` 的 8x8/16x16/8x32 仅输出列底部结果

---

## 修改清单

### 一、`rtl/array/reconfig_pe_array.v`

| 改动 | 说明 |
|------|------|
| 新增参数 `MAX_TILE_RESULTS=256` | 控制 `acc_out` 和 `valid_out` 端口宽度 |
| `acc_out` 从 `[32*ACC_W-1:0]` 扩为 `[MAX_TILE_RESULTS*ACC_W-1:0]` | 支持最多 256 个 PE 结果同时输出 |
| `valid_out` 从 `[31:0]` 扩为 `[MAX_TILE_RESULTS-1:0]` | 同上 |
| 8x8 模式输出映射改为 per-PE grid（8×8=64 个结果） | 原为仅底部一行 8 个列结果 |
| 16x16 模式输出映射改为 per-PE grid（16×16=256 个结果） | 原为仅底部一行 16 个列结果 |
| 8x32 模式输出映射改为 per-PE grid：top half（ri×32+ci）+ bottom half（ri×32+ci+16）共 256 个结果 | 原为仅底部一行 32 个列结果 |

---

### 二、`rtl/ctrl/npu_ctrl.v`

| 改动 | 说明 |
|------|------|
| 删除 `localparam TILE4_LANES = 32'd4` | 移除 4x4 钳位 |
| `tile_active_rows` 端口从 `[2:0]` 扩为 `[4:0]` | 支持最多 16 行 |
| `tile_active_cols` 端口从 `[2:0]` 扩为 `[5:0]` | 支持最多 32 列（8x32） |
| `tile_row_valid`/`tile_col_valid` 从 `[3:0]` 扩为 `[15:0]` | 16 bit 有效掩码 |
| `wb_row` 从 `[2:0]` 扩为 `[4:0]` | 支持最多 16 行写回 |
| 新增函数 `shape_tile_n_lanes`，返回列方向 lane 数（4x4→4, 8x8→8, 16x16→16, 8x32→32） | 区分行列 lane 数 |
| `tile_n_tiles`/`tile_n_base`/`tile_active_cols_32` 改用 `tile_shape_n_lanes_32` | 8x32 正确处理 32 列单 tile |
| 拆分 `vector_elem_bytes` 为 `vector_elem_bytes_w` 和 `vector_elem_bytes_a` | 8x32 的 W/A 加载字节数不同 |
| 拆分 `tile_len` 为 `tile_len_w`/`tile_len_a` | DMA W/A 加载长度独立 |
| 拆分 `seq1_len_bytes` 为 W/A 版本 | prefetch 地址计算独立 |
| 拆分 `cfg_start_tile_len` 为 W/A 版本 | warmup 加载长度独立 |
| Descriptor shape 检查从 `==4'd0` 改为 `<=4'd3` | 接受 4x4/8x8/16x16/8x32 |

---

### 三、`rtl/top/npu_top.v`

| 改动 | 说明 |
|------|------|
| 新增 `localparam MAX_TILE_RESULTS = 256` | 与 reconfig_pe_array 口径一致 |
| `pe_array_result` 从 `[32*ACC_W-1:0]` 扩为 `[MAX_TILE_RESULTS*ACC_W-1:0]` | 匹配阵列输出 |
| `pe_array_valid` 从 `[31:0]` 扩为 `[MAX_TILE_RESULTS-1:0]` | 同上 |
| `ctrl_tile_active_cols` 从 `[4:0]` 扩为 `[5:0]` | 匹配 controller 输出 |
| `tile_result_buf` 从 `[0:15]` 扩为 `[0:255]` | 最多 256 个结果缓存 |
| `tile_ser_row` 从 `[1:0]` 扩为 `[4:0]` | 最多 16 行 |
| `tile_ser_col` 从 `[1:0]` 扩为 `[5:0]` | 最多 32 列 |
| `tile_ser_active_rows` 从 `[2:0]` 扩为 `[4:0]` | 匹配 active_rows |
| `tile_ser_active_cols` 从 `[2:0]` 扩为 `[5:0]` | 匹配 active_cols |
| capture 循环从硬编码 16 改为 shape-aware（使用 `tile_capture_cnt`） | 4x4/8x8/16x16/8x32 自动适配 |
| `op_counter` 的 `COLS` 参数从 32 改为 `MAX_TILE_RESULTS(256)` | 匹配宽 valid 端口 |
| `reconfig_pe_array` 实例化传入 `.MAX_TILE_RESULTS(MAX_TILE_RESULTS)` | 端口宽度一致 |

#### Bug 修复

| Bug | 位置 | 原因 | 修复 |
|-----|------|------|------|
| `tile_ser_idx` 用 `tile_ser_active_cols` 计算索引 | serializer | PE grid 按 `grid_cols` 存储，但序列化器用 `active_cols`（边界 tile < grid_cols）读取，导致行列错位 | 改为 `tile_ser_row * tile_grid_cols + tile_ser_col` |
| A skew pipe LANE=1 在 drain 时清零 | gen_a_skew | `tile_feed_step && !tile_vec_fire` 时 `pipe[0]` 被清零，LANE=1 仅有单级 pipe，数据未传递给 `a_skew_vec[1]` 即丢失 | LANE=1 改用寄存器延迟 `l1_reg`，仅在 `tile_vec_fire` 时捕获，drain 期间保持 |
| `tile_capture_cnt` 截断 5'd16 为 0 | capture 循环 | `tile_grid_rows[3:0]` 只取低 4 位，16（10000）的低 4 位为 0，导致 16x16 capture 0 个结果 | 改为 `{3'd0, tile_grid_rows} * {2'd0, tile_grid_cols}`，使用完整位宽 |
| `tile_len_raw_w/a` scalar 模式乘了 `tile_shape_lanes` | npu_ctrl.v prefetch 地址 | 拆分 W/A 后 `tile_len_raw_w = tile_k_len * vector_elem_bytes_w`，但 `vector_elem_bytes_w = scalar_elem_bytes * tile_shape_n_lanes` 在 scalar 模式下不该乘 shape lanes（导致 prefetch 步长变成 8 而非 4 bytes，scalar matmul/Conv2D 结果全错） | `tile_len_raw_w/a` 在 `tile_mode=0` 时使用 `scalar_elem_bytes` |

---

### 四、`rtl/axi/npu_dma.v`

| 改动 | 说明 |
|------|------|
| `a_ofm_active_rows` 端口从 `[2:0]` 扩为 `[4:0]` | 匹配 controller 输出 |
| 内部信号 `ofm_active_rows_latch`、`ofm_lane_pos` 同步扩宽 | 避免编译 warning |

---

### 五、新增文件

| 文件 | 说明 |
|------|------|
| `tb/tb_npu_tile_gemm_wide.v` | 基于 `tb_npu_tile_gemm.v` 的 shape-aware 版本，支持可配置的 `GRID_COLS_VAL`、`AW_EXPECT_VAL`、`CFG_SHAPE_VAL` |
| `tb/tile4/gen_multi_shape_data.py` | 多 shape（4x4/8x8/16x16/8x32）测试数据生成器，支持多 word/k 的 tile pack、边界零填充、精确 AW 计数 |
| `scripts/run_multi_shape_gemm.ps1` | 多 shape 测试运行脚本 |

---

## 已验证通过的测试

| 测试 | Shape | M×K×N | 结果 |
|------|-------|--------|------|
| 4x4 回归 | 4x4 | 4×4×4 | ALL 16 PASS |
| 8x8 基本 | 8x8 | 8×4×8 | ALL 64 PASS |
| 8x8 多 tile | 8x8 | 16×4×16 (4 tiles) | ALL 256 PASS |
| 8x8 K-split | 8x8 | 8×40×8 (2 k_tiles) | ALL 64 PASS |
| 8x8 边界列 N=3 | 8x8 | 8×4×3 | ALL 24 PASS |
| 8x8 边界行 M=3 | 8x8 | 3×4×8 | ALL 24 PASS |
| 8x8 边界+多 tile | 8x8 | 9×4×5 (2 tiles) | ALL 45 PASS |
| 8x8 非整数倍 | 8x8 | 10×4×10 (4 tiles) | ALL 100 PASS |
| 16x16 最小 | 16x16 | 1×1×1 | ALL 1 PASS |
| Direct scalar matmul | — | 3×5×4 | ALL 12 PASS |
| Direct scalar matmul | — | 2×2×2 | ALL 4 PASS |
| Visual CNN smoke | — | 256×9×6 | ALL 1536 PASS, mismatches=0 |

---

## 未完成项

| 项目 | 状态 | 原因 |
|------|------|------|
| 16x16 完整 GEMM | ❌ | A skew pipe 对 LANE≥6 存在 4-row 数据偏移（rows 6-11 读到了 rows 10-15 的 A 值），疑似 Icarus 仿真工具的 generate elaborate 行为异常，需波形定位 |
| 8x32 完整 GEMM | ❌ | Controller 不支持两轮权重调度（fold 机制需要将 32 列权重分两轮加载到 16 物理列），bottom half 永远收到 cols 0-15 的权重而非 cols 16-31 |

---

## 修改的文件完整列表

| 文件 | 状态 |
|------|------|
| `rtl/array/reconfig_pe_array.v` | 已修改 |
| `rtl/ctrl/npu_ctrl.v` | 已修改 |
| `rtl/top/npu_top.v` | 已修改 |
| `rtl/axi/npu_dma.v` | 已修改 |
| `tb/tb_npu_tile_gemm_wide.v` | 新增 |
| `tb/tile4/gen_multi_shape_data.py` | 新增 |
| `scripts/run_multi_shape_gemm.ps1` | 新增 |

---

## 当前遗留问题

详见 `doc/pre_modification_issues.md`（修改前电路原始问题）和 `doc/unresolved_issues.md`（未解决问题清单）。摘要如下：

### 已定位但未修复的 Bug

| # | 问题 | 现象 | 根因分析 |
|---|------|------|---------|
| 3.5 | **8x32 两轮权重调度缺失** | bottom half（cols 16-31）全零，aw_count=0（修复宽度后正常写回） | Controller 不支持 8x32 fold 机制的两轮权重，cols 16-31 的权重数据从未到达 PE 阵列 |
| 3.6 | **16x16 A 数据 4-row 偏移** | rows 6-11 读到了 rows 10-15 的 A 值，其他 rows 正确 | Pipe 延迟数学正确，疑似 Icarus generate elaborate 异常。偏移量恰为 1 PPBuf word（4 lanes） |

### 功能缺失（电路存在但未接入）

| # | 功能 | 现状 | 修复估计 |
|---|------|------|---------|
| 2.3 | **后处理入 tile 路径** | bias/ReLU/quant 电路在 scalar 路径已完成，tile 路径被 `ctrl_tile_mode ? tile_result_buf : scalar_post_result` 短路 | 低：在 serializer→FIFO 间插入后处理逻辑 |
| 2.4 | **时钟门控/DFS 接入 PE** | `npu_power` 输出悬空（`.row_clk_gated()` 等未接） | 低：端口连接 |
| — | **FP16 for 8x8** | INT8 已验证，FP16 未测试 | 低：生成 FP16 测试数据 |
| — | **4x4 边界 golden** | controller planner 已验证，无 GEMM 数据验证 | 低：生成边界 tile 数据 |
| — | **WS tile GEMM golden** | 仅有 OS 端到端 golden | 中：WS 数据布局不同 |
| — | **外部 PSUM surface 读写** | K-split 内部可用，外部 R/W 未接 | 中：需 DMA + controller 改动 |
| — | **Descriptor 8x8 多任务** | shape check 已放宽，无测试 | 低：编写 descriptor 测试数据 |
| — | **Tile-mode Conv2D descriptor** | OP=CONV2D_IM2COL 未映射 | 中：controller descriptor decode 扩展 |
