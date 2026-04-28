# 当前实现状态

更新时间：2026-04-27



## 验证环境

```text
E:\iverilog\bin\iverilog.exe
E:\iverilog\bin\vvp.exe
Icarus Verilog 12.0
```

如果命令行找不到 `iverilog`，先执行：

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
```

## 实测结果

| 项目 | 当前结果 | 结论 |
|---|---:|---|
| `scripts/run_sim.ps1` | 19 PASS / 0 FAIL | 单 PE 基线可用 |
| `tb/tb_npu_scalar_smoke.v` | PASS | 顶层标量 INT8 OS 路径可用，`CFG_SHAPE` start 锁存有效 |
| `tb/tb_pingpong_buf_vec.v` | PASS | PPBuf 4-lane INT8/FP16 vector read 可用 |
| `tb/tb_npu_ctrl_tile.v` | PASS | 4x4 tile planner 和 M/N 边界 mask 可用 |
| 当前源列表手工编译运行 `tb_comprehensive.v` | 28 PASS / 0 FAIL | Phase 1 顶层标量兼容路径已恢复 |
| `scripts/run_full_sim.ps1` | 编译和仿真完成 | 脚本源列表与 testbench 参数已对齐 |
| `scripts/run_soc_sim.ps1` | 编译失败 | `dram_model.v` 的 `axi_arlen` 绑定问题和 PicoRV32 PCPI 端口不匹配 |

## 当前已经具备

1. `pe_top` 支持 INT8 和 FP16 MAC，含 WS/OS 累加语义。
2. `npu_axi_lite` 提供 CPU 配置寄存器，包括 `CTRL`、维度、地址、`CFG_SHAPE`。
3. `npu_ctrl` 有按 `C[i][j]` 单点输出推进的标量 FSM，也有 `ARR_CFG[7]` 控制的 4x4 tile planner。
4. `npu_dma` 有 AXI4 master 读写框架，可把 W/A 读入 PPBuf，把结果 FIFO 写回 DRAM。
5. `pingpong_buf` 支持 DMA 写 32-bit word，PE 侧按 INT8/FP16 拆出 16-bit 标量或 4-lane vector 数据。
6. `reconfig_pe_array` 有 16x16 物理阵列和 4x4/8x8/16x16/8x32 形态选择接口。
7. `npu_power` 有 DFS 和 row/col clock gating 行为模型。
8. `npu_top` 有一条 Phase 1 标量兼容路径：PPBuf 标量输出进入 `u_scalar_pe`，结果通过 FIFO 写回。
9. `npu_ctrl` 已输出 `cfg_shape_latched`，当前任务启动后修改 `CFG_SHAPE` 不影响正在运行的阵列配置。
10. T2.1 已定义 4x4 tile 数据布局：OS 模式采用 `PE row -> M lane`、`PE col -> N lane`，A/W 使用 4-lane tile-pack，C 按 `result[r*4+c]` 输出。
11. T2.2 已给 `pingpong_buf` 增加 4-lane vector read port，并在 `npu_top` 中把 W/A vector 接入 `reconfig_pe_array` 左上 4x4 边界。
12. T2.3 已实现 `ARR_CFG[7]` 控制的 4x4 tile planner：输出 tile base、row/col mask、`vec_consume` 和 OS row-skew feeder。

## 当前关键差距

1. 4x4 tile mode 已能产生 vector consume 和 row/col mask，但结果仍未从阵列 serializer 写回。
2. 当前可验证写回路径仍是标量 `u_scalar_pe` 结果，阵列 16 个输出还没有进入结果 FIFO。
3. 当前可验证写回路径是标量 `u_scalar_pe` 结果，还没有收集多列或多行阵列输出。
4. `npu_ctrl` 的循环单位仍是单个 `C[i][j]`，不是矩阵 tile。
5. 当前没有 `PSUM/OUT_BUF`，无法支持 K-split、多层卷积中间结果暂存。
6. 当前没有 descriptor 队列，NPU 不具备自主多层调度能力。
7. DMA 读通道固定 `ARLEN=0`，未达到 AXI burst 带宽利用率目标。
8. `npu_power` 输出没有接入 PE 主时钟路径。
9. SoC 仿真仍存在 DRAM 模型信号绑定和 PicoRV32 PCPI 端口不匹配。

## 不应继续引用的旧结论

| 旧说法 | 当前判断 |
|---|---|
| `tb_comprehensive` 失败 2/28 PASS | 已过期，当前是 28/28 PASS |
| 全量回归 903 PASS | 当前不能作为事实引用 |
| SoC 集成已验证 | 当前需要重新修复和验证 |
| 16x16/8x32 已完成高吞吐矩阵乘 | 阵列存在，但顶层未喂满、未完整写回 |
| DMA 读通道支持多拍 burst | 当前读通道 `ARLEN=0` |
| DFS/时钟门控已实际降功耗 | 行为模块输出悬空 |
| 支持 INT16 | 当前 PE 只有 INT8/FP16 |

## 当前一句话定位

这是一个具备 PE 算术和 NPU 外围框架的原型。下一阶段的重点是把顶层数据组织从“标量点积路径”升级为“可验证的 4x4 tile GEMM 路径”，再扩展到 descriptor、多层卷积和 16x16 高吞吐阵列。
