# PTH 推理原型支持范围

本文是 `feature/pth-cpu-npu-inference` 分支的第一步：先固定 `.pth -> 参考 CPU 调度 -> NPU 执行 Conv/ReLU` 的 V1 支持范围。目标不是支持任意 PyTorch 模型，而是先让一个结构受限的小模型能从 `.pth` 权重走到 SoC 仿真推理结果。

## 结论

单独给一个任意 `.pth`，当前工程无法直接推理。原因有两个：

- `.pth` 常见内容只是 `state_dict`，只有参数名和 tensor，不包含可靠的 forward graph。
- 当前 NPU 不解析 PyTorch，也没有通用模型 runtime；CPU 只能通过 MMIO 配置 NPU 寄存器并管理 DRAM 数据。

V1 可执行目标定义为：

```text
.pth state_dict + model_spec.json
  -> host Python converter
  -> NPU 权重/输入/计划文件
  -> 参考 CPU runtime 逐层配置 NPU
  -> NPU 执行 Conv2D/ReLU/可选 quant
  -> CPU 处理层间格式转换和不支持算子
```

其中 `.pth` 在 host Python 上解析，不在参考 CPU 上解析。

## V1 输入格式

V1 接受两类 checkpoint：

```python
torch.save(model.state_dict(), "model.pth")
```

或：

```python
torch.save({"state_dict": model.state_dict()}, "model.pth")
```

模型拓扑必须由一个旁路 JSON 显式描述，文件名暂定为 `model_spec.json`。不依赖 `torch.save(model)`，因为它需要 Python 类定义可导入，跨环境不稳定。

最小 `model_spec.json` 示例：

```json
{
  "name": "tiny_conv_relu",
  "input": {
    "shape": [1, 1, 28, 28],
    "dtype": "int8",
    "layout": "NCHW",
    "scale": 1.0,
    "zero_point": 0
  },
  "layers": [
    {
      "name": "conv0",
      "op": "conv2d",
      "weight": "conv0.weight",
      "bias": "conv0.bias",
      "stride": [1, 1],
      "padding": [1, 1],
      "dilation": [1, 1],
      "activation": "relu",
      "quant": {
        "enabled": true,
        "scale": 1,
        "shift": 3,
        "round": true
      }
    }
  ]
}
```

## 支持算子

| 算子 | V1 支持 | 执行位置 | 说明 |
|---|---:|---|---|
| `Conv2d` | 是 | NPU | 使用 direct scalar on-the-fly im2col；CPU 配置 `M/N/K`、`A_ADDR/W_ADDR/R_ADDR` 和 `0x80..0x94` Conv 寄存器 |
| `ReLU` | 是 | NPU | `CTRL[11:10]=01` |
| `ReLU6` | 是 | NPU | `CTRL[11:10]=10` |
| `Bias` | 是 | NPU | `BIAS_ADDR(0x98)`，每输出通道一个 32-bit word |
| `INT8 quant/saturate` | 是 | NPU | `QUANT_CFG(0x9C)`，输出 sign-extended int8 word |
| `Linear` | 部分 | NPU 或 CPU | 可映射 direct matmul；V1 优先把最后小 FC 放 CPU，降低层间格式复杂度 |
| `Flatten` | 是 | CPU | 只做地址/索引视图或软件搬运 |
| `BatchNorm2d` | 离线支持 | converter | fold 到 Conv 权重和 bias |
| `MaxPool/AvgPool` | 暂不加速 | CPU | V1 可由 CPU 软件执行，小模型可先不用 |
| `Softmax` | 暂不加速 | CPU | 分类 demo 可直接比较 logits，先不要求 softmax |
| `Add/Residual/Concat` | 否 | - | V1 不支持 |
| 动态 shape/控制流 | 否 | - | V1 只支持静态 shape |

## 模型结构限制

V1 只接受静态顺序网络：

```text
input
  -> [Conv2d + optional Bias + optional ReLU/ReLU6 + optional INT8 quant]*
  -> optional CPU Flatten
  -> optional CPU Linear / argmax
```

限制：

- `Batch=1`。
- 输入/中间特征图使用 NCHW。
- Conv2D 支持 `groups=1`，不支持 depthwise/group conv。
- Conv2D kernel 优先支持 `1x1`、`3x3`、`5x5`；更大 kernel 需要确认 PPBuf 和仿真时间。
- stride/pad/dilation 使用现有 Conv 寄存器表达。
- 第一版只承诺 INT8 推理路径；FP16 可作为后续扩展。
- 每层由 CPU direct register mode 启动一次，不使用 descriptor mode。

## 数据布局口径

输入 IFM：

```text
A_ADDR -> raw IFM, NCHW contiguous, INT8 byte packed
```

权重：

```text
PyTorch Conv weight: [Cout, Cin, KH, KW]
NPU W_col:           [K, Cout], K = Cin * KH * KW
Column j stores output channel j.
```

输出 OFM：

```text
R_ADDR -> row-major C[M,N], M = B * OH * OW, N = Cout
每个输出是一个 32-bit word
```

开启 INT8 quant 后，32-bit word 中保存 sign-extended int8 值。CPU 如果要把该输出作为下一层 Conv2D 的 IFM，需要执行：

```text
C[M,N] row-major 32-bit words
  -> clamp/read low int8
  -> reshape to NHWC spatial/channel view
  -> transpose/repack to NCHW INT8 byte stream
  -> 写入下一层 A_ADDR
```

这是 V1 runtime 必须承担的层间格式转换。

## 第一版建议 demo

为了最快闭环，第一版 `.pth` demo 采用：

```text
Input: 1x1x16x16 INT8
Layer0: Conv2d(1, 4, kernel=3, stride=1, padding=1) + ReLU + quant
CPU: repack OFM to NCHW INT8
Layer1: Conv2d(4, 2, kernel=1, stride=1, padding=0) + ReLU + quant
CPU: global sum / argmax 或直接比较 logits
```

选择 `Layer1 1x1` 是为了验证 `Cin>1` 的真实层间输入，同时避免第二层 kernel 复杂度掩盖 runtime 问题。

## 后续步骤

1. 已完成 host converter 原型：读取 `.pth + model_spec.json`，输出 `model_plan.json`、NPU Conv 权重、CPU Linear 权重和 checkpoint inventory。
2. 已完成 CPU runtime descriptor 生成原型：输出 C layer table、NPU direct mode 寄存器参数、per-channel requant 固定点参数和 CPU pooling/linear helper。
3. 待完成 SoC 仿真脚本：编译 CPU 程序、初始化 DRAM/模型资产、运行推理、读回结果并对比 golden。
4. 若小模型 V1 闭环通过，再考虑把 Conv/ReLU/quant 接入 descriptor 主线，减少 CPU 每层轮询和 repack 开销。

## Host Converter 状态

已新增 `tools/pth/pth_to_npu_assets.py` 和 `tools/pth/examples/repopt_vgg_int8_spec.json`，可以解析本地 `.06_RepOpt_VGG` 中的 `qat_int8_quantized.pth`，并导出 9 个 Conv2D 的 NPU `W_col` 权重和 accumulator-unit `bias_int32`。

运行命令：

```powershell
python tools\pth\pth_to_npu_assets.py `
  --pth .06_RepOpt_VGG\06_RepOpt_VGG\runs\cifar10_repopt_vgglike_qat\qat_int8_quantized.pth `
  --spec tools\pth\examples\repopt_vgg_int8_spec.json `
  --out-dir sim\pth_repopt_probe `
  --mode OS
```

已确认结果：

```text
layers       : 15
conv layers  : 9
warnings     : 11
```

关键限制：该 int8 checkpoint 的 Conv 权重是 per-output-channel quantized，PyTorch 的 requant multiplier 也是 per-channel；当前 NPU `QUANT_CFG` 是每层单一 scale/shift，不能精确替代 per-channel requant。因此 V1 精确路径必须是：

```text
NPU: Conv2D + bias + ReLU
CPU: per-channel requant + NCHW repack
```

## CPU runtime 生成状态

已新增 `tools/pth/gen_cpu_runtime.py` 和 `tools/pth/runtime/npu_pth_runtime.h`。在 `model_plan.json` 生成后，可以导出参考 CPU 侧的 C layer table 和固定点 requant 参数：

```powershell
python tools\pth\gen_cpu_runtime.py `
  --plan sim\pth_repopt_probe\model_plan.json `
  --out-dir sim\pth_repopt_probe\cpu_runtime
```

本步骤同时补齐了 `classifier` 的 CPU Linear 权重导出：

```text
assets/classifier_linear_w_int8.hex
assets/classifier_linear_bias_int32.hex
```

当前生成结果明确显示 RepOpt VGG 不能直接放进现有 SoC smoke 的默认 DRAM：

```text
asset bytes       : 5,287,424
current DRAM bytes: 61,440
assets fit DRAM   : false
```

所以结论是：这条分支已经具备“给定受限 `.pth` -> host 转换 -> CPU layer table -> CPU 调度 NPU Conv/ReLU -> CPU 做 requant/pool/linear”的代码基础；但用户给的这个 RepOpt VGG checkpoint 还不能在当前 `tb_soc.v` 默认内存配置下完整推理。下一步若要真正跑通 SoC 仿真，需要二选一：扩大 SoC/testbench DRAM 并增加模型资产 loader，或先训练/准备一个小得多的 int8 checkpoint 用当前 60KB 级 DRAM 验证闭环。

## Tiny SoC smoke 状态

已新增当前 DRAM 可承载的 `.pth` 闭环 smoke：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pth_tiny_conv_soc.ps1
```

该脚本会自动生成一个单层 `Conv2D(1, 2, 3x3) + ReLU` 的 quantized tiny checkpoint，然后走同一套 host converter，生成 DRAM 初始化文件和 RV32I firmware hex。firmware 运行在参考 CPU 上，逐项写入 NPU direct Conv2D 寄存器并轮询 `STATUS.done`。

已确认仿真结果：

```text
[PASS] PTH tiny Conv SoC test PASSED!
Cycles: 687
R = [10, 2, 8, 2, 9, 0, 13, 0]
```

这说明“小 `.pth` + 现有参考 CPU + 现有 NPU 电路”已经可以完成一个真实 Conv/ReLU 推理 smoke。RepOpt VGG 不能跑通的原因仍然是模型资产和中间 buffer 规模远超当前默认 DRAM，而不是 CPU 无法配置 NPU。

## 三层小模型 smoke 状态

已新增 3 层 `.pth` 小模型 SoC 闭环：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_pth_multilayer_soc.ps1
```

模型结构：

```text
Conv3x3(1->1) + ReLU
CPU repack: row-major int32 OFM -> NCHW int8 IFM
Conv1x1(1->2) + ReLU
CPU repack: row-major int32 OFM -> NCHW int8 IFM
Conv1x1(2->1) + ReLU
```

已确认仿真结果：

```text
[PASS] PTH multilayer Conv SoC test PASSED!
Cycles: 1403
R = [28, 22, 25, 37]
```

这个用例比单层 tiny smoke 多验证了两件事：

- 参考 CPU 可以连续调度多个 `.pth` 转换出来的 NPU Conv/ReLU 层。
- 层间必须由 CPU 把 NPU row-major int32 OFM 重新打包成下一层 Conv 所需的 NCHW int8 IFM。

## RepOpt VGG host 整网状态

RepOpt VGG 的 int8 checkpoint 使用 signed qint8 activation；当前 PyTorch CPU quantized Conv 后端期望 quint8 activation，因此不能直接把该 checkpoint 当作 stock PyTorch eager int8 模型运行。已新增 host-side CPU/NPU split interpreter：

```powershell
python tools\pth\run_repopt_vgg_host.py `
  --index 0 `
  --out-json sim\pth_repopt_host_run\host_run_idx0.json
```

图片输入使用同一入口：

```powershell
python tools\pth\run_repopt_vgg_host.py `
  --image path\to\image.png `
  --conv-backend tile4 `
  --out-json sim\pth_repopt_host_run\image_tile4.json
```

图片路径会执行 `RGB -> resize 32x32 -> ToTensor -> CIFAR-10 Normalize -> qint8 quantize`。模型仍是 CIFAR-10 分类器，只会输出 10 类之一。

该解释器按当前计划分工执行整网：

```text
NPU 语义：Conv2D int8*int8 -> int32 accumulator + bias + ReLU
CPU 语义：per-channel requant、NCHW repack、MaxPool、AvgPool、Flatten、Linear
```

已确认 CIFAR-10 test 样本：

```text
index 0: true=cat  pred=cat  logits_int8=-14 -8 -3 96 -28 9 6 -11 -17 -30
index 1: true=ship pred=ship logits_int8=7 16 -17 -14 -22 -21 -16 -20 92 -6
index 2: true=ship pred=ship logits_int8=8 52 -16 -16 -16 -24 -16 -20 59 -11
```

这说明 RepOpt VGG 已经可以在 host 上按“参考 CPU + NPU 语义”跑完整推理。下一步不是立刻全网 SoC，而是抽取 RepOpt 的实际 Conv 层做 RTL 分段验证；全网 SoC 仍需要扩大 DRAM 和模型资产 loader。

当前 host 解释器已有两个 Conv backend：

```text
direct: full Conv accumulator model，等价于 NPU Conv2D + bias + ReLU 后再 CPU requant
tile4 : 全网 Conv-as-GEMM 4x4 tile 调度，CPU 做 bias/ReLU、per-channel requant、pool、linear
```

已用 CIFAR-10 样本 0 导出的 PNG 验证图片入口：

```text
image_idx0_direct: pred=cat
image_idx0_tile4 : pred=cat
```

`tile4` backend 的全网层调度已经覆盖 9 个 Conv 层，tile 数分别为：

```text
4096, 4096, 2048, 2048, 1024, 1024, 1024, 512, 512
```

## RepOpt VGG 分段 RTL 状态

已新增 `tools/pth/gen_repopt_layer_case.py` 和 `scripts/run_repopt_layer_case.ps1`，用于把 RepOpt 的真实 Conv 层抽出来送进现有 `npu_top` direct Conv2D testbench。默认先跑第一层 `stage1_0_conv` 的左上角 `4x4` 输出窗口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_layer_case.ps1 -Index 0
```

当前通过结果：

```text
layer=stage1_0_conv sample_index=0 cifar_label=3
output_window=4x4 of full 32x32
GEMM M=16 K=27 N=64 results=1024
ALL 1024 CHECKS PASSED
```

该 case 使用 `.pth` 中的真实 int8 权重、真实 int32 bias、CIFAR-10 样本 0 的真实量化输入，比较点是 NPU 输出的 `Conv2D + bias + ReLU` int32 accumulator，尚未进入 CPU per-channel requant。完整第一层是 `M=1024, K=27, N=64`，在当前 1x1 标量 Verilog 仿真下运行时间较长，因此先以 tile 方式建立可重复的 RTL golden。若要请求完整第一层，可使用 `-TileOH 0 -TileOW 0`。

注意：上面的 `4x4` 是输出空间窗口裁剪，仍走 direct scalar Conv2D 路径，并不是 `ARR_CFG[7]` 的硬件 4x4 tile-mode。

真正的 4x4 tile-mode 入口为 `tools/pth/gen_repopt_tile_case.py` 和 `scripts/run_repopt_tile_case.ps1`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_case.ps1 `
  -Index 0 -MBase 0 -NBase 0
```

当前已通过两个 RepOpt 第一层真实 tile：

```text
repopt_stage1_0_conv_tile4_m0_n0_idx0 : ALL 16 CHECKS PASSED
repopt_stage1_0_conv_tile4_m33_n4_idx0: ALL 16 CHECKS PASSED
```

这个 case 把 Conv2D 展开为一个局部 `4x27x4` GEMM tile，预打包为 `A_TILE[k][r]` 和 `W_TILE[k][c]`，并启用 `ARR_CFG[7]`。当前 tile-mode 写回比较的是 raw int32 MAC accumulator；bias/ReLU/per-channel requant 还没有并入 tile 写回路径，需要后续再做。

已新增 `tools/pth/run_repopt_layer_tile_rtl.py`，用于把多个真实 RTL tile 串起来形成一个局部 layer window，并在 host CPU 上执行后处理：

```powershell
python tools\pth\run_repopt_layer_tile_rtl.py `
  --index 0 `
  --m-base 0 --n-base 0 `
  --m-tiles 2 --n-tiles 2 `
  --out-json sim\pth_repopt_tile_rtl\stage1_0_m0_n0_2x2.json
```

当前通过结果：

```text
layer=stage1_0_conv sample_index=0 window=M[0:8) N[0:8)
tiles_run=4 raw_min=-19045 raw_max=24423
q_min=0 q_max=24
```

该入口真实运行 4 个 `ARR_CFG[7]` tile case，收集每个 case 的 `npu_output.hex`，拼成 `8x8` raw accumulator window，然后由 CPU 执行 `bias/ReLU/per-channel requant`。后处理后的 qint8 window 已和 host 第一层 golden 对比，当前 mismatch 为 0。

进一步已新增 `tools/pth/gen_repopt_tile_soc_case.py` 和 `scripts/run_repopt_tile_soc.ps1`，把 tile 调度和第一层后处理下沉到参考 CPU 固件：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 `
  -Index 0 -MBase 0 -NBase 0 -MTiles 2 -NTiles 2
```

当前通过结果：

```text
[PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!
Cycles: 8743
window: M[0:8) N[0:8)
tiles scheduled by CPU: 4
first tile result[0] = 7383
first postprocess q[0] = 11
```

这一步生成一次 DRAM image 和一次 RV32I firmware；仿真中参考 CPU 通过 MMIO 连续配置多个 `4x27x4` tile，NPU 在同一次 RTL 仿真内逐 tile 运算。随后 CPU 固件从 DRAM 读回 raw int32 MAC window，执行真实 bias、ReLU 和 per-channel 固定点 requant，并把 qint8 window 写回 DRAM。testbench 会同时校验 raw MAC 结果和固件后处理 q 结果。

固件现在使用嵌套循环调度 tile，不再按 tile 展开指令，因此已经可以跑完整第一层窗口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_repopt_tile_soc.ps1 `
  -Index 0 -FullLayer
```

当前完整第一层通过结果：

```text
[PASS] RepOpt tile-window SoC MMIO + CPU postprocess test PASSED!
Cycles: 8851650
window: M[0:1024) N[0:64)
tiles scheduled by CPU: 4096
first tile result[0] = 7383
first postprocess q[0] = 11
firmware words: 91
q_base: 0x00049500, q_count: 65536
marker: 0x00089500, dram_words: 141312
```

如果只想检查生成和编译，可加 `-CompileOnly`。完整第一层仿真需要调度 4096 个 NPU tile，在 Icarus Verilog 下耗时较长，所以不作为默认 smoke。
