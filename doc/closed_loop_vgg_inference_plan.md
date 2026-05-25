# VGG 全闭环推理实现计划

更新时间：2026-05-25

本文记录从当前 `run_vgg_e2e.sh` 通过状态，演进到“CPU 固件 + NPU RTL 自己完成 9 层层间流转”的实现计划。本文是计划文档，不应被当作已实现功能说明。

## 当前基线

当前已验证入口：

```bash
./run_vgg_e2e.sh
```

当前结果：

```text
[PASS] RepOpt VGG end-to-end classification PASSED
Cycles: 10768727
Predicted: cat (class 3)
```

当前数据流：

```text
Python 生成 DRAM 初始数据
    -> CPU 固件逐 tile 读取 descriptor
    -> CPU 固件通过 MMIO 配置 NPU
    -> NPU RTL 执行 9 层 Conv tile 计算并写回 DRAM
    -> CPU 固件读取最后一层输出
    -> CPU 固件执行 avgpool + classifier + argmax
    -> CPU 固件写 marker，testbench 判 PASS/FAIL
```

当前不满足“全闭环”的原因：

- Python 在仿真前一次性生成了 9 层所有 tile 的 A tile 输入数据。
- NPU 的每层输出虽然写回 DRAM，但前面层的输出没有在运行时被 CPU 固件动态转成下一层输入。
- MaxPool、层间 repack、下一层 im2col/A tile pack 主要仍由 Python 离线完成。
- 当前真正运行时闭环的是最后一层 NPU 输出到 CPU avgpool/classifier/argmax。

## 目标定义

最终目标：

```text
Python 只生成静态资产：
输入图像、权重、bias、requant 参数、classifier 参数、layer metadata、golden label

CPU 固件运行时完成：
tile 调度、A tile pack、NPU 配置、NPU 输出后处理、层间 dense buffer 流转、maxpool、avgpool、classifier、argmax

NPU RTL 运行时完成：
9 层 Conv 的 INT8 GEMM tile 计算
```

验收时，`dram_init.hex` 不应包含第 2 到第 9 层的预生成 A tile。CPU 固件必须从输入图像 dense buffer 开始，自己生成所有层间 activation。

## 推荐总体路线

优先实现“正确性优先”的闭环版本：

```text
NPU 只算 raw Conv tile
CPU 做 bias + ReLU + per-channel requant + scatter 到 dense activation buffer
CPU 从 dense activation buffer 动态 pack 下一层 A tile
CPU 做 maxpool / avgpool / classifier
```

选择该路线的原因：

- RepOpt VGG 的 requant 是 per-channel multiplier。
- 当前 VGG E2E 使用的 NPU tile `QUANT_CFG` 是每 16-channel tile 一个 scalar 近似。
- 如果闭环后继续让 NPU 做 tile-level quant，误差会逐层累积，最终分类不稳定。
- 先由 CPU 做 per-channel requant，能最大限度接近 Python/PyTorch golden。
- 不先改 RTL，风险更小，调试边界更清楚。

## 运行时数据布局

建议统一使用 dense row-major activation buffer：

```text
addr = base + ((m * C) + c) * 4
m = h * W + w
C = output channels
```

含义：

```text
Dense OFM[M, C]，每个元素用 32-bit word 保存 sign-extended int8 或中间 int32。
```

推荐使用两个 activation buffer ping-pong：

```text
ACT_A
ACT_B
POOL_TMP 可选
```

每层结束后：

```text
如果无 pool：next_ifm = current_ofm
如果有 pool：CPU maxpool current_ofm -> next_ifm
```

## Conv 层运行时流程

每个 Conv 层按以下流程执行：

```text
for m_tile in output spatial tiles:
    CPU 从当前 dense IFM pack A tile 到 A_WORK

    for n_tile in output channel tiles:
        CPU 配置 NPU：
            A_ADDR = A_WORK
            W_ADDR = layer_w[n_tile]
            R_ADDR = R_WORK
            BIAS/QUANT 关闭或设为 raw path
            M_DIM/N_DIM/K_DIM = 当前 tile shape
            ARR_CFG = tile mode
            CTRL = INT8 + OS + start

        NPU 计算 raw MAC tile 到 R_WORK

        CPU 读取 R_WORK：
            加 bias[n]
            ReLU
            per-channel requant
            clamp int8
            scatter 到 dense OFM[M, C]
```

需要注意：

- `A_WORK` 是单个 tile 的临时 packed A stream，不是整层 A tile 表。
- `R_WORK` 是单个 tile 的临时 raw output buffer。
- 权重仍可由 Python 离线按 n tile 预打包，因为权重是静态资产。
- CPU 固件必须处理边界 tile，例如最后一个 M tile 或 N tile 不满 16。

## Pooling 与最后分类

VGG 中 MaxPool 位置：

```text
stage1_1_conv 后
stage2_1_conv 后
stage3_2_conv 后
```

CPU 固件执行：

```text
dense OFM -> maxpool2d -> dense IFM for next layer
```

最后一层 `stage4_1_conv` 后：

```text
dense final OFM: 16 spatial x 512 channels
CPU avgpool: 每个 channel 对 16 个空间点求平均
CPU classifier: 512 x 10 dot product
CPU argmax
CPU 写 marker = 0x100 + pred
```

## 生成器改造

新增或改造生成器时，Python 只负责生成静态资产：

- 输入图像 dense buffer。
- 每层 W tile-pack 数据。
- 每层 bias。
- 每层 per-channel requant multiplier/shift。
- 每层 shape、stride、padding、dilation、activation metadata。
- classifier weight/bias。
- expected label 和可选 golden dump。

Python 不再生成：

- 第 2 到第 9 层的 A tile。
- 中间层 MaxPool 后结果。
- 下一层 IFM 的 runtime repack 结果。

## 固件生成策略

不要继续手写 `vgg_fw_template.hex`。建议使用 Python assembler 生成固件 hex。

需要的固件 helper：

| Helper | 职责 |
|---|---|
| `run_npu_tile` | 写 NPU 寄存器、start、轮询 done/error |
| `pack_a_tile_from_dense` | 从 dense IFM 按 Conv2D im2col 规则生成 A tile stream |
| `postprocess_tile_to_dense` | 读取 raw tile，bias/ReLU/per-channel requant，scatter 到 dense OFM |
| `maxpool2d_dense` | 对 dense activation 做 2D maxpool |
| `avgpool_dense_4x4` | stage4_1 输出做 512-channel avgpool |
| `classifier_argmax` | 计算 10-class score 并写 marker |

需要补齐或确认的 RV32 指令 helper：

```text
LB/LBU/SB/LH/LHU
SLLI/SRLI/SRAI
SLT/SLTI/BLT/BGE
MUL/MULH 或可替代的 fixed-point 乘法序列
```

## 分阶段实施

### 阶段 A：单层闭环

目标：`stage1_0_conv` 不再使用 Python 生成的 A tile 表。

验收：

```text
CPU 从输入 dense image pack A tile
NPU 执行 stage1_0 所有 tile
CPU postprocess/scatter 到 dense OFM
dense OFM 抽样值对齐 Python fixed-point golden
```

### 阶段 B：两层闭环

目标：`stage1_0_conv -> stage1_1_conv`。

验收：

```text
stage1_1 的 A tile 由 CPU 从 stage1_0 dense OFM 动态 pack
不使用 Python 生成的 stage1_1 A tile
stage1_1 dense OFM 抽样值对齐 golden
```

### 阶段 C：第一个 MaxPool 闭环

目标：`stage1_0 -> stage1_1 -> maxpool`。

验收：

```text
CPU maxpool 输出 shape 和抽样值对齐 golden
下一层输入来自 CPU maxpool 输出
```

### 阶段 D：stage2 block 闭环

目标：`stage2_0 -> stage2_1 -> maxpool`。

验收：

```text
stage2 block 运行时层间 pack/scatter/pool 全部由 CPU 固件完成
```

### 阶段 E：stage3 block 闭环

目标：`stage3_0 -> stage3_1 -> stage3_2 -> maxpool`。

验收：

```text
stage3 block 全部 runtime 闭环
```

### 阶段 F：stage4 block 闭环

目标：`stage4_0 -> stage4_1`。

验收：

```text
stage4_1 输出 dense OFM 由 NPU+CPU runtime 生成
```

### 阶段 G：完整分类闭环

目标：9 层 Conv + pool + avgpool + classifier + argmax。

验收：

```text
./run_vgg_closed_loop.sh
[PASS]
Predicted class == Python expected class
```

## 后续优化项

正确性跑通后再考虑以下优化：

| 优化项 | 说明 |
|---|---|
| RTL im2col DMA 接入 tile path | 用硬件替代 CPU pack A tile |
| per-channel quant table | 让 NPU tile path 支持 per-channel requant |
| strided writeback/scatter | 让 NPU 直接写 dense OFM，减少 CPU scatter |
| descriptor v2 | 支持 layer/tile metadata、postprocess、pool hints |
| bigger tile shape | 8x8/16x16/8x32 完整端到端写回和验证 |

## 最终验收标准

必须满足：

- `dram_init.hex` 不包含第 2 到第 9 层的预生成 A tile。
- CPU 固件从输入 dense buffer 开始自己生成所有层间数据。
- NPU RTL 实际执行 9 层 Conv tile。
- CPU 固件完成 MaxPool、AvgPool、Classifier、Argmax。
- 至少 `img_idx=0` 通过，建议额外覆盖多个 CIFAR index。
- 当前 `./run_vgg_e2e.sh` baseline 仍保持 PASS。

## 主要风险

| 风险 | 缓解 |
|---|---|
| CPU 固件变大 | 用生成器生成 hex，不手写模板 |
| per-channel requant 溢出或 rounding 不一致 | 先建立 Python fixed-point golden，再逐层抽样比对 |
| 仿真时间增加 | 分阶段跑小模型/单层，再跑完整 VGG |
| layout 错误难定位 | 每阶段 dump dense OFM 抽样和 checksum |
| 分类偶然通过但中间错误 | 中间层必须有抽样/统计验证，不只看最终 label |
