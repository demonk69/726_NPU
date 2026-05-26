# Visual CNN 功能验证方案

本文记录 `scripts/run_visual_cnn_case.ps1` 对应的复杂可视化 NPU 验证入口。它不是新的 RTL datapath，而是把现有 direct scalar Conv2D on-the-fly im2col、bias、activation 和 INT8 quant/saturate 串成一个真实图片特征提取实验。

## 运行方式

快速 smoke：

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_visual_cnn_case.ps1 `
  -Resize 16 -Name visual_cnn_smoke16
```

默认可视化 case：

```powershell
$env:Path = 'E:\iverilog\bin;' + $env:Path
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_visual_cnn_case.ps1
```

默认参数为 `pic/test2_128.png`、`Resize=64`、`Mode=OS`、`Activation=relu`、`QuantScale=1`、`QuantShift=3`、启用 signed rounding。也可以改为 WS：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_visual_cnn_case.ps1 `
  -Mode WS -Resize 64 -Name visual_cnn_ws_64
```

## 验证内容

输入图片先转为灰度，再映射为 signed INT8：

```text
npu_ifm = grayscale - 128
```

随后生成一个 `Cout=6` 的 3x3 Conv2D feature bank：

| Channel | Kernel | Bias |
|---|---|---:|
| 0 | Sobel X | 0 |
| 1 | Sobel Y | 0 |
| 2 | Laplacian | -8 |
| 3 | Sharpen | 8 |
| 4 | Outline | -16 |
| 5 | Emboss | 0 |

对 `64x64` 默认输入，Conv2D 映射到 GEMM：

```text
M = OH * OW = 64 * 64 = 4096
K = Cin * KH * KW = 1 * 3 * 3 = 9
N = Cout = 6
NUM_RESULTS = 4096 * 6 = 24576
```

这个 case 覆盖：

- `CTRL[8]` direct scalar on-the-fly im2col。
- `W_col[K,N]` 多输出通道列布局。
- `BIAS_ADDR(0x98)` 每输出通道 32-bit bias。
- `CTRL[11:10]` ReLU/ReLU6/none activation。
- `QUANT_CFG(0x9C)` INT8 scale、shift、round 和 saturate。
- OS/WS direct scalar 控制分支。
- `DUMP_RESULT_HEX` 输出 `npu_output.hex` 后再做图像级 diff。

## 输出文件

脚本会在 `tb/image/<case name>/` 下生成：

```text
dram_init.hex          # DRAM 初始化内容
expected.hex           # bit-exact golden，row-major M*N words
npu_output.hex         # 仿真 dump 的 NPU 输出
metadata.json          # case 配置
visual_summary.json    # mismatch、max_abs_diff 和通道统计
input_gray.png         # 输入灰度图
golden_<channel>.png   # 每个通道的 golden 特征图
npu_<channel>.png      # 每个通道的 NPU 特征图
diff_<channel>.png     # 每个通道的 diff
golden_rgb.png         # 软件 golden 伪 RGB 融合
npu_rgb.png            # NPU 输出伪 RGB 融合
diff_heatmap.png       # 所有通道最大绝对差异热力图
comparison_grid.png    # 总览拼图
report.html            # HTML 汇总报告
```

判定标准仍以数值为准：

```text
ALL <NUM_RESULTS> CHECKS PASSED
mismatches=0
max_abs_diff=0
```

`golden_rgb.png` 和 `npu_rgb.png` 只是帮助直观看 feature response 是否合理；`diff_heatmap.png` 全黑才说明所有通道逐点一致。

## V1 边界

当前 V1 保持单输入灰度通道 `Cin=1`，用 `Cout=6` 验证多输出通道和后处理。伪 RGB 融合在 Python 渲染阶段完成，不代表 RTL 内部新增了 RGB/多层融合 datapath。

多通道 layer0 的 `R_ADDR` 输出是每个结果一个 32-bit word，而 direct INT8 layer1 的普通 A 矩阵输入期望按 byte packed 的 `A[M,K]`。因此这个 V1 没有把 `Cout=6` 的 OFM 直接作为下一层 `K=6` 的输入强行串接。现有 `scripts/run_conv2d_two_layer_case.ps1` 仍覆盖 `K=1` 情况下的 layer0 `R_ADDR -> layer1 A_ADDR` 两层端到端链路。
