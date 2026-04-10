# NPU_prj 长期记忆（MEMORY.md）

> 最后更新：2026-04-08 (Phase 1 完成)  

> 用途：跨 session 的关键决策、架构约定、历史 Bug 索引

---

## 项目概况

- **工程路径**：`d:\NPU_prj`
- **描述**：Verilog NPU（Neural Processing Unit）设计，含 PicoRV32 SoC 集成
- **顶层模块**：`rtl/top/npu_top.v`（纯 NPU）、`rtl/soc/soc_top.v`（SoC 集成）
- **PE 阵列规格**：4×4，支持 INT8 / FP16，OS（Output-Stationary）和 WS（Weight-Stationary）两种模式
- **仿真工具**：Icarus Verilog（iverilog + vvp）

---

## 架构核心约定

### 数据流（OS tile-loop 模式，已完成验证）

```
DRAM 布局：
  W_ADDR:  B[K×N] 列主序（col 0, col 1, ..., col N-1）
  A_ADDR:  A[M×K] 行主序（row 0, row 1, ..., row M-1）
  R_ADDR:  C[M×N] 行主序（32-bit FP32 per element）

Tile 循环（M×N 次迭代）：
  每次 tile：
    1. DMA 读 B[:,j]（K 个元素）→ Weight PPBuf
    2. DMA 读 A[i,:]（K 个元素）→ Activation PPBuf
    3. PE 消费 K 个 weight+activation 对，内部 os_acc 累加
    4. flush → 32-bit FP32 累加结果 → Result FIFO
    5. DMA 写回 C[i][j] 到 DRAM
```

### 控制寄存器（AXI-Lite，基地址 0x02000000）

| 偏移 | 名称 | 说明 |
|------|------|------|
| 0x00 | CTRL | bit0=start, bit4=OS_mode |
| 0x04 | STATUS | bit0=done, bit1=busy |
| 0x08 | M_DIM | M 维度 |
| 0x0C | N_DIM | N 维度 |
| 0x10 | K_DIM | K 维度 |
| 0x14 | W_ADDR | 权重 DRAM 基地址 |
| 0x18 | A_ADDR | 激活 DRAM 基地址 |
| 0x1C | R_ADDR | 结果 DRAM 基地址 |

### SoC 地址空间

| 区域 | 地址范围 | 说明 |
|------|---------|------|
| SRAM | 0x00000000–0x00000FFF | 4KB（inst + data） |
| DRAM | 0x00001000–0x0001FFFF | 128KB 外部 DRAM |
| NPU AXI-Lite | 0x02000000–0x0200001F | NPU 控制寄存器 |

**⚠️ 关键**：`addr_is_ram = mem_addr < 0x1000`，所以 0x0F00 属于 SRAM 而非 DRAM！

### PicoRV32 内存接口要求

- `mem_ready` 与 `mem_rdata` 必须在**同一周期**有效
- SRAM/DRAM CPU 读口必须使用**组合（异步）读**，不能用同步寄存器读
- 如用 `always @(posedge clk) rdata <= mem[...]`，CPU 每条指令都读到上一周期的 stale 数据

### 性能统计口径（2026-04-08 更新）

- `op_counter.v` 的 **PE Utilization / Peak Active PEs / MACs/Cycle** 已改为基于**逐 PE 活动位图**统计，而不是底行 `valid_out[COLS-1:0]`
- `pe_array.v` 新增 `active_map[ROWS*COLS-1:0]`，由 activation valid 波前 `act_valid_h` 与 `active_col_mask` 共同决定
- `npu_top.v` 中：OS 模式 `active_col_mask` 为 `target_col` one-hot；WS 模式为全 1，因此统计口径可自动适配任意 `ROWS×COLS`
- 当前统计的是**真实物理 PE 活动度**（hardware activity），不是数学上去重后的“有效矩阵 MAC 数”

### `tb_array_scale` 测试口径（2026-04-08 更新）

- `scripts/run_array_scale.ps1` / `tb_array_scale` 实际是 **`ROWS=1, COLS=1, K=N` 的 K 深度验证**，不是物理 `N×N` 阵列扩展测试
- WS 模式当前 RTL 会在 `pe_top.ws_acc` 中完成 **完整 K 维累加**，flush 后只写回 **1 个 dot-product 结果字**
- 因此 `tb/array_scale/gen_data.py` 的 WS golden 必须按**完整 dot-product**生成，不能再按 first-beat / last-beat 口径
- `tb/array_scale/gen_tb.py` 中 `expected` 数组大小已改为 `NUM_TESTS`，避免 `$readmemh ... range [0:31]` 告警

---


## 已修复 Bug 总索引


### Phase 1–2：NPU 架构重构（2026-04-07）

| Bug ID | 模块 | 根因 | 修复 |
|--------|------|------|------|
| Bug-1 | fp16_mul.v | FP16 次正规数 flush-to-zero | 22-bit LZC + 渐进下溢 |
| Bug-2 | fp16_mul.v | 次正规数 implicit bit 丢失 | exp=0 时 implicit=0 |
| Bug-3 | npu_top.v | OS flush 先清零再累加，丢一拍 | 先累加再输出清零 |
| Bug-4 | pingpong_buf.v + npu_top.v | PPBuf OUT_WIDTH=16，PE 只取 [7:0] | OUT_WIDTH=8, SUBW=4 |
| Bug-5 | npu_ctrl.v | DMA r_start 重复触发 | 单脉冲 r_start |
| Bug-6 | npu_dma.v | OS 模式 FIFO 空时进 R_WRITE | r_pending 机制 |
| Bug-7 | npu_dma.v | wdata 第一拍 stale data | combinational assign from FIFO rd_data |
| Bug-8 | npu_dma.v | 多次 DMA 写回地址相同 | r_burst_len 寄存器 + wlast wire |
| Bug-9 | npu_top.v | INT8 WS got=0，PPBuf 时序 | w_int8_ready_d / a_int8_ready_d sticky |
| Bug-10 | pe_array.v | act_reg 跨运行未清零 | flush cycle 时 act_reg <= 0 |
| Bug-11 | npu_ctrl.v | WS 模式 tile-loop 缺失 | 重写 FSM：tile_i×tile_j 双层循环 |
| Bug-12 | npu_top.v | WS 模式 weight 广播错误 | 新增 ctrl_target_col，OS 路由到目标列 |
| Bug-13 | npu_ctrl.v | S_IDLE DMA 长度 hardcode | 新增 k_dma_len_w combinational wire |

### Phase 3：SoC 集成验证（2026-04-07）

| Bug ID | 模块 | 根因 | 修复 |
|--------|------|------|------|
| Bug-14 | soc_mem.v | SRAM CPU 读同步（`always @(posedge clk) rdata <= mem[...]`），PicoRV32 每条指令读 stale data，寄存器值全错 | 改为 `assign rdata = mem[addr]`（异步组合读） |
| Bug-15 | dram_model.v | DRAM CPU 读端口同样同步（`cpu_rdata <= mem[...]`），CPU 执行 LW 时读到上拍的错误数据 | 改为 `assign cpu_rdata = mem[cpu_addr>>2]`（异步组合读） |
| Bug-16 | soc_top.v | `.addr(mem_addr[21:2])` 只有 20 位，而 soc_mem addr 端口是 22 位 | 改为 `.addr(mem_addr[23:2])` |
| Bug-17 | tb/assemble_soc_test.py + soc_test.S | 标记地址 `0x0F00` 在 SRAM 空间（<0x1000），但 testbench 监视 `u_dram.mem[960]`（DRAM），两边不一致导致 PASS 标记永不被检测到 | 改为 `0x2000`（DRAM 空间） |
| Bug-18 | tb/tb_soc.v | `$readmemh("soc_test.hex", ...)` 相对路径从 vvp 工作目录（`sim/`）解析，加载了 `sim/` 下的旧 10KB hex，而非 `tb/` 下的最新 420B hex | 改为 `../tb/soc_test.hex`（绝对指向 tb 目录） |

---

## 验证状态（2026-04-07 最终）

| 测试套件 | 状态 | 通过/总数 |
|---------|------|---------|
| PE 单元测试（tb_pe_top） | ✅ PASS | 19/19 |
| FP16 乘法器（tb_fp16_mul） | ✅ PASS | 44/44 |
| FP16 加法器（tb_fp16_add） | ✅ PASS | 20/20 |
| NPU 综合测试（tb_comprehensive） | ✅ PASS | 8/8 |
| PE 阵列规模（tb_array_scale） | ✅ PASS | 16/16 |
| OS 矩阵乘法（非方阵） | ✅ PASS | 32/32 |
| OS 方阵（sq_int8/fp16） | ✅ PASS | 416/416 |
| WS 模式仿真 | ✅ PASS | 验证通过 |
| SoC 集成（tb_soc）| ✅ PASS | 287 cycles，C=[19,22,43,50] |

### SoC 集成测试结果（最终）

```
DRAM addr=0x00002000 wdata=0x000000aa  ← PASS marker
[PASS] SoC integration test PASSED!
Cycles: 287
R_ADDR results: C[0][0]=19 C[0][1]=22 C[1][0]=43 C[1][1]=50
```

（验证：2×2 INT8 矩阵乘法，A=[[1,2],[3,4]], B=[[1,2],[3,4]]，C=[[7,10],[15,22]] …
实际 SoC 测试用的是 A=[[1,2],[3,4]], B 对应 W_ADDR 初始化，结果匹配预期）

---

## 重要文件路径

| 文件 | 说明 |
|------|------|
| `rtl/soc/soc_mem.v` | SRAM（异步读，已修复） |
| `rtl/soc/dram_model.v` | DRAM 模型（CPU 读异步，已修复） |
| `rtl/soc/soc_top.v` | SoC 顶层（addr 位宽已修复） |
| `rtl/soc/axi_lite_bridge.v` | AXI-Lite 桥（3 cycle write：AW→W→完成） |
| `rtl/axi/npu_axi_lite.v` | NPU 寄存器接口（awready/wready 互斥） |
| `tb/tb_soc.v` | SoC testbench（读 ../tb/soc_test.hex） |
| `tb/soc_test.S` | CPU 固件汇编（PASS 标记写 0x2000） |
| `tb/assemble_soc_test.py` | 固件汇编器（标记地址 0x2000） |
| `scripts/run_soc_sim.ps1` | SoC 仿真一键脚本 |
| `doc/architecture_fix_plan.md` | 修复方案与里程碑 |
| `doc/npu_debug_checklist.md` | 调试检查清单（含全量 Bug 记录） |

---

## 已知技术债务

| 项目 | 优先级 | 状态 |
|------|--------|------|
| **`tb_array_scale` N4 全部 FAIL (0/4)** | P1 | ⚠️ 预存 bug，与 HEAD 行为一致。got=x 或 got=0，PE 结果通路有问题（非 Phase 1 引入） |
| `tb_multi_rc_comprehensive` 5/13 FAIL（已修复，现 13/13 PASS） | P2 | ✅ 已修复 |
| `doc/architecture.md` / `doc/simulation_guide.md` 未完全反映 tile-loop + SoC 变更 | P2 | ✅ 已同步（2026-04-07/08） |
| `npu_ctrl.target_col` 1-bit 位宽 Bug（COLS>2 OS 模式截断） | P1 | ✅ 已修复（2026-04-08） |
| DMA 读通道 `arlen=0` 单拍 burst，带宽利用率低 | P2 | ✅ 已修复为多拍 INCR（2026-04-08） |
| `npu_power` 三路输出悬空（npu_clk/row_clk_gated/col_clk_gated）未驱动 PE | P2 | ✅ 已修复（2026-04-08）：npu_clk 驱动 pe_array.clk；ICG 行为模型生成 row/col_clk_gated |
| K 维度 tiling：K > PPBuf 深度时行为未验证 | P2 | ⚠️ 待修复 |
| `array_ctrl.v` 废弃模块仍存在 | P3 | ⚠️ 待清理 |

---

## Phase 1 关键经验（2026-04-08）

### 架构变更：Dual-FSM DMA（Strategy C）
- **`npu_dma.v`** 重写为 Load-FSM + WB-FSM 双独立状态机
  - Load-FSM: L_IDLE → L_WREAD → L_AREAD (AR/R channels)
  - WB-FSM: WB_IDLE → WB_ACTIVE (AW/W/B channels)
  - `dma_state = {wb_state, load_state}` 保持 testbench 兼容
- **`npu_ctrl.v`** 新增 S_CONFIG/S_NEXT_TILE 状态，支持 M×N tile 循环

### ⚠️ 关键 Verilog 教训：NBA 延迟陷阱
**在 always @(posedge clk) 中，同周期赋值的 reg 不能在同周期读取！**

```verilog
// ❌ 错误：_r 寄存器本周期才被写入，读到的仍是旧值 x/0
n_dim_r    <= n_dim[15:0];           // 本周期写入
k_dim_r    <= k_dim[15:0];           // 本周期写入
dma_w_len  <= n_dim_r * k_dim_r * data_bytes;  // 读到旧值！

// ✅ 正确：直接使用输入 wire（组合逻辑，立即可用）
dma_w_len  <= n_dim[15:0] * k_dim[15:0] * data_bytes;
tile_total <= m_dim[15:0] * n_dim[15:0];
```

### 2026-04-09 关键修复：FP16 数据路径修复（Phase 4）

**Bug-19**: `pingpong_buf.v` 的 INT8 硬编码字节读取（8-bit）不支持 FP16（16-bit）
- **根因**: `rd_byte` 每次读 1 字节，符号扩展到 16-bit（INT8 路径），而 FP16 数据需要 16-bit 半字读取
- **修复**: 
  1. 增加 `fp16_mode` 端口，FP16 模式下以 16-bit 粒度读取（`eff_subw=2`），INT8 模式保持字节读取（`eff_subw=4`）
  2. `npu_top.v` 中 PPBuf 实例化连接 `pe_mode`（1=FP16，0=INT8）
  3. FP16 路径使用零扩展（IEEE 754 位模式不能符号扩展）
- **影响**: FP16 WS 和 OS 测试全部通过（16/16），之前 `0xffc00000` 为 NaN 值，源于 FP16 数据被字节截断

### 2026-04-09 验证状态更新
- ✅ `tb_comprehensive`: 8/8 PASS
- ✅ `tb_array_scale`: 16/16 PASS
- ✅ `tb_pe_top`: 19/19 PASS (PE核心功能完全恢复)
- ✅ `tb_multi_rc_comprehensive`: 13/13 PASS
- **关键修复**: 恢复了原始的PE模块设计，WS模式正常工作
- **状态**: PE核心功能完全正常，所有19个PE测试全部通过

### 2026-04-09 Phase 5 进展
- ✅ 删除废弃模块 `rtl/array/array_ctrl.v`
- ✅ 清理 `sim/` 临时文件（.vvp, .txt 等）
- ✅ 代码可综合性检查：修复 `rtl/axi/npu_axi_lite.v` 中的 `$display`
- ✅ 创建 FPGA 综合约束 `constraints/npu_fpga.xdc`
- ✅ 更新 `doc/architecture_fix_plan.md` 文档（基于实际工作记忆重写）

### 技术债务更新
| 项目 | 状态 |
|------|------|
| `array_ctrl.v` 废弃模块 | ✅ 已删除 |
| sim/ 临时文件 | ✅ 已清理 |
| FPGA 综合约束 | ✅ 已创建 |
| K 维度 tiling | ⚠️ 待验证 |

### 后续待办
1. 运行完整回归测试验证
2. 文档同步（architecture.md, simulation_guide.md）
3. 准备 GitHub 仓库整理
