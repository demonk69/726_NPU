# 工程文档重做计划

更新时间：2026-05-25

本文是文档体系重做计划，目标是清理当前“计划、工作日志、旧状态、实际实现”混在一起的问题，重新建立以当前 RTL、CPU 固件、仿真脚本为准的工程文档。本文本身是计划，不是新的最终文档入口。

## 背景问题

当前文档的主要风险：

- 计划型文档、历史工作日志、当前事实文档混在 `doc/` 下。
- 部分 README 和指南仍描述旧 Windows/Icarus 流程，而当前关键 VGG E2E 路径使用 Linux + Verilator。
- 有些文档把“目标架构”或“未来任务”写得像已实现事实。
- 当前 VGG E2E 已经发生变化：NPU 执行 9 层 Conv tile，但层间 A tile 仍由 Python 离线生成；最后 avgpool/classifier/argmax 由 CPU 固件运行时完成。
- 如果不重做文档，后续开发很容易误以为工程已经具备完整 9 层 runtime 层间闭环。

## 重做原则

1. 事实文档只写已被源码和仿真验证的内容。
2. 计划文档必须明确标注“计划”，不能混在架构事实里。
3. 每个结论都要能追溯到 RTL 文件、固件生成器、脚本入口或最近一次仿真命令。
4. 根 `README.md` 只做可信入口，不堆历史细节。
5. 旧文档不要直接静默删除；先归档、加弃用说明，再按确认后的清单删除。
6. 文档中区分三种状态：`已验证`、`已实现但未完整验证`、`计划中`。
7. 不再用过期阶段编号作为主线，例如 Phase/T5/T6/T7 只能留在归档或变更记录中。

## 目标文档体系

建议最终保留以下文档。

| 文档 | 类型 | 作用 |
|---|---|---|
| `README.md` | 入口 | 项目目标、当前真实状态、快速运行、文档索引 |
| `doc/architecture.md` | 结构文档 | 当前 SoC/NPU/CPU/DRAM 数据流和控制流 |
| `doc/rtl_reference.md` | 结构文档 | RTL 模块职责、接口、当前限制 |
| `doc/firmware_runtime.md` | 固件文档 | PicoRV32 固件职责、MMIO 调度、VGG 固件流程 |
| `doc/memory_map_and_abi.md` | 接口文档 | AXI-Lite 寄存器、DRAM layout、marker、descriptor/固件 ABI |
| `doc/vgg_e2e_flow.md` | 推理文档 | 当前 VGG E2E 真实数据流、Python/CPU/NPU 分工 |
| `doc/simulation_guide.md` | 仿真手册 | 可运行命令、环境、预期输出、失败排查 |
| `doc/verification_status.md` | 验证状态 | 当前通过/失败/未验证测试矩阵 |
| `doc/closed_loop_vgg_inference_plan.md` | 计划 | 9 层全闭环推理实现计划 |
| `doc/documentation_rework_plan.md` | 计划 | 本文档重做计划，完成后可归档 |
| `doc/known_issues.md` | 风险 | 当前限制、误差来源、技术债 |
| `doc/archive/README.md` | 归档索引 | 说明旧文档来源、为何归档、哪些不能引用 |

可选文档：

| 文档 | 条件 | 要求 |
|---|---|---|
| `doc/conv_gemm_mapping.md` | 保留 | 必须确认公式和当前 layout 一致 |
| `doc/debug_checklist.md` | 保留 | 只保留仍适用于当前 RTL/Verilator 的排查步骤 |
| `tools/pth/README.md` | 保留 | 聚焦工具输入/输出/脚本，不承担总架构说明 |
| `doc/changelog.md` | 可选 | 只记录重要工程变更，不替代 Git 历史 |

## 每个文档的要求

### `README.md`

必须包含：

- 一句话项目定义。
- 当前最可信状态。
- 快速运行命令。
- 当前 VGG E2E 是否是完整层间闭环的明确说明。
- 文档索引。

禁止包含：

- 大段历史计划。
- 未验证性能指标。
- 过期 Windows-only 命令作为主入口。

### `doc/architecture.md`

必须包含：

- 当前 SoC 框图。
- CPU、NPU、DRAM、testbench 的边界。
- direct register mode 和当前 VGG 固件调度方式。
- 真实数据路径：A/W/bias/R/marker。
- 当前不支持或未验证的能力。

禁止包含：

- 把 descriptor v2、完整自动多层卷积、完整 16x16/8x32 吞吐写成已完成。
- 把 Python 预生成的中间 A tile 描述成 runtime 层间流转。

### `doc/rtl_reference.md`

必须包含：

- `rtl/pe`、`rtl/array`、`rtl/buf`、`rtl/axi`、`rtl/ctrl`、`rtl/top`、`rtl/soc` 的模块职责。
- 关键接口和寄存器来源。
- 当前通过的最小测试入口。
- 已知限制。

要求：

- 以当前源码为准逐项核对。
- 每个模块只写当前事实，不混入长期愿景。

### `doc/firmware_runtime.md`

必须包含：

- PicoRV32 固件如何启动。
- `vgg_fw_template.hex` 当前职责。
- CPU 固件每个 tile 写哪些 NPU 寄存器。
- CPU 后处理：avgpool、classifier、argmax、marker。
- 固件生成策略和未来闭环固件规划。

必须明确：

- 当前 VGG 固件不是动态生成下一层 A tile。
- 当前固件模板是 hex，不是易维护源码；后续应迁移到生成式汇编或 C/asm 源。

### `doc/memory_map_and_abi.md`

必须包含：

- NPU AXI-Lite 寄存器表。
- 当前 VGG DRAM layout。
- tile descriptor 10-word runtime template 格式。
- marker 规则：`0x100 + pred`、`0xFF`。
- classifier/feature/score 地址。
- 与 descriptor v1 ABI 的区别。

要求：

- 不允许把 VGG 固件 10-word tile table 和 RTL descriptor v1 16-word ABI 混为一谈。

### `doc/vgg_e2e_flow.md`

必须包含：

- 当前 `run_vgg_e2e.sh` 完整步骤。
- Python 生成了什么。
- CPU 固件做了什么。
- NPU 做了什么。
- 哪些层间信息仍来自 Python。
- 当前 PASS 输出和周期。
- 当前修复过的问题：10-word descriptor、per-N-tile bias address、fail-fast marker。

### `doc/simulation_guide.md`

必须包含：

- Linux 当前工作路径下的命令。
- `./run_vgg_e2e.sh`。
- `./run_all.sh standard [idx]` 或实际可用入口。
- Verilator 编译/运行说明。
- 常见失败：timeout、classification mismatch、firmware failure。
- 如何判断是 NPU 卡住还是 CPU 后处理卡住。

禁止包含：

- 只适用于旧 Windows PowerShell 的命令作为主路径。
- 未重新验证的大规模 regression 数字。

### `doc/verification_status.md`

必须包含：

- 最后验证日期。
- 每个测试命令。
- PASS/FAIL/TIMEOUT。
- 预期输出关键行。
- 未验证项和原因。

要求：

- 数字必须来自最近一次实际运行，不能沿用旧文档里的历史统计。

### `doc/known_issues.md`

必须包含：

- 当前不是完整 runtime 层间闭环。
- Python 仍预生成中间 A tile。
- 固件 hex 模板不可维护。
- VGG 当前运行时间较长。
- per-channel quant 与 NPU tile scalar quant 的差异风险。
- 16x16/8x32 端到端吞吐未作为当前 VGG 依赖。

## 现有文档处理建议

当前 Markdown 文件处理建议如下。

| 现有文件 | 建议处理 | 原因 |
|---|---|---|
| `README.md` | 重写 | 当前入口和状态描述过长，且含旧阶段结论 |
| `tools/pth/README.md` | 复查后重写 | 应聚焦工具，不承担整体架构说明 |
| `doc/architecture.md` | 重写 | 保留结构，但需要改成当前事实口径 |
| `doc/module_reference.md` | 重写并改名为 `rtl_reference.md` | 当前内容混合旧阶段和目标描述 |
| `doc/user_manual.md` | 拆分 | 寄存器/ABI 放 `memory_map_and_abi.md`，固件放 `firmware_runtime.md` |
| `doc/simulation_guide.md` | 重写 | 当前主路径应以 Linux + Verilator + VGG E2E 为准 |
| `doc/current_status.md` | 重写为 `verification_status.md` | 旧状态数字需要重新验证 |
| `doc/pth_inference_subset.md` | 复查后合并或归档 | 可能仍有模型子集价值，但需核对当前工具 |
| `doc/conv_gemm_mapping.md` | 复查后保留 | 公式型文档可保留，但必须核对 layout |
| `doc/npu_debug_checklist.md` | 复查后改名为 `debug_checklist.md` | 保留仍有效的调试步骤 |
| `doc/git_guide.md` | 可保留或移到 archive | 与当前技术文档重做关系不大 |
| `doc/task_breakdown.md` | 归档或删除 | 计划型旧文档，容易误导 |
| `doc/architecture_fix_plan.md` | 归档或删除 | 计划型旧文档，当前已部分过期 |
| `doc/soc_integration_plan.md` | 归档或删除 | 计划型旧文档，SoC 状态已变化 |
| `doc/pre_modification_issues.md` | 归档 | 历史问题记录，不应作为当前问题入口 |
| `doc/deepseek_changes_20260509.md` | 归档 | 工作日志型文档 |
| `doc/repopt_full_soc_inference_worklog.md` | 归档 | 工作日志型文档 |
| `doc/unresolved_issues.md` | 重写为 `known_issues.md` 或归档 | 当前问题需重新核对 |
| `doc/visual_cnn_verification.md` | 归档或复查 | 需确认是否仍有当前验证价值 |

## 删除/归档执行策略

建议分三步执行，最大限度避免误导：

### 第一步：新增“可信入口”

新增或重写：

```text
README.md
doc/vgg_e2e_flow.md
doc/simulation_guide.md
doc/verification_status.md
doc/known_issues.md
```

在旧文档未处理前，根 README 必须明确：

```text
doc/archive/ 或旧计划文档仅供历史参考，不代表当前实现状态。
当前事实以 README、vgg_e2e_flow、simulation_guide、verification_status 为准。
```

### 第二步：归档旧计划和日志

创建目录：

```text
doc/archive/legacy_before_2026_05_25/
```

移动以下类型文档：

```text
*_plan.md
*_worklog.md
deepseek_changes_*.md
pre_modification_issues.md
task_breakdown.md
```

归档目录下放 `README.md`，说明：

```text
这些文档是历史记录，不应作为当前架构、接口、仿真命令依据。
```

### 第三步：确认后删除

归档稳定一轮后，如果确定不再需要历史记录，再删除归档中的计划型文档。删除前应满足：

- 新文档已覆盖当前工程使用所需信息。
- 新仿真手册已能独立指导运行。
- 新 verification status 有实际命令和输出。
- Git diff 中能清楚看到删除的是历史文档，不是源码或测试资产。

## 重做顺序

推荐顺序：

1. 写 `doc/vgg_e2e_flow.md`，先把当前真实推理流固定下来。
2. 写 `doc/simulation_guide.md`，保证任何人能复现当前 PASS。
3. 写 `doc/verification_status.md`，记录最新实际命令和结果。
4. 重写根 `README.md`，只保留可信入口。
5. 写 `doc/firmware_runtime.md`。
6. 写 `doc/memory_map_and_abi.md`。
7. 重写 `doc/architecture.md`。
8. 重写 `doc/rtl_reference.md`。
9. 重写 `doc/known_issues.md`。
10. 归档旧计划/日志文档。
11. 复查 `tools/pth/README.md`。

## 每轮文档复查清单

每次文档更新前后都检查：

- 文档中的命令是否能在当前 Linux workspace 运行。
- 文档中的文件路径是否存在。
- 文档中的寄存器 offset 是否与 `rtl/axi/npu_axi_lite.v` 一致。
- 文档中的固件流程是否与 `tools/pth/vgg_fw_template.hex` 或生成器一致。
- 文档中的 VGG 数据流是否与 `tools/pth/gen_vgg_e2e.py` 一致。
- 文档是否明确区分当前事实和未来计划。
- 是否有旧文档仍被 README 索引为可信入口。

## 完成标准

文档重做完成应满足：

- 根 `README.md` 可以作为唯一入口。
- `doc/simulation_guide.md` 能独立指导当前仿真。
- `doc/vgg_e2e_flow.md` 明确说明当前不是完整 9 层 runtime 层间闭环。
- `doc/closed_loop_vgg_inference_plan.md` 明确说明如何实现完整闭环。
- 所有旧计划/日志文档已归档或删除。
- 没有文档把 Python 预生成中间 A tile 描述成 CPU/NPU runtime 层间流转。
