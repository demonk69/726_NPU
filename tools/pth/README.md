# PyTorch/VGG Tooling

Updated: 2026-05-30

This directory contains host-side tooling for RepOpt VGG-like CIFAR-10 inference tests.

Current maintained flows are Linux + Verilator. Older PowerShell/Icarus examples in archived documents are not current guidance.

## Maintained Generators

| File | Purpose |
|---|---|
| `gen_vgg_e2e.py` | Fast VGG e2e generator. Pre-generates all Conv A tile streams for the 9-Conv model. |
| `gen_vgg_closed_loop.py` | Runtime closed-loop generator. Generates static model assets and firmware that packs A tiles at runtime. |
| `run_repopt_vgg_host.py` | Host reference implementation shared by both VGG generators. |

## Fast E2E Flow

Entry points:

```bash
./run_vgg_e2e.sh
./run_all.sh standard 0
./run_all.sh image ./pic/test_cifar10_2.jpg
```

Generator behavior:

- Loads the checkpoint and model plan.
- Computes all intermediate Conv A tiles in Python.
- Emits a DRAM init image containing all pre-packed tile streams.
- Emits firmware and Verilog parameter headers.

Use this flow for quick baseline regression.

## Runtime Closed-Loop Flow

Entry points:

```bash
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg
./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg
```

Generator behavior:

- Loads the checkpoint and model plan.
- Emits static model assets: weights, bias, per-channel Q24 multipliers, classifier params, descriptors.
- Emits the selected input image into the initial DRAM image.
- Emits firmware that performs runtime A-tile packing, NPU scheduling, per-channel requant/scatter, maxpool, avgpool, classifier, and argmax.

Use this flow for deployment-oriented validation.

## Host Reference

`run_repopt_vgg_host.py` interprets the model split used by the generators:

- NPU role: Conv2D int8*int8 accumulation with bias.
- CPU role: per-channel requant, activation packing/scatter, maxpool, avgpool, flatten, linear classifier.

The reference intentionally avoids relying on PyTorch quantized Conv kernels for the core check. It is used to create exact and fixed-runtime expected labels.

## Inputs

Default model/data paths:

| Input | Default path |
|---|---|
| Checkpoint | `RepOpt/06_RepOpt_VGG/runs/cifar10_repopt_vgglike_qat/qat_int8_quantized.pth` |
| Model plan | `sim/pth_repopt_probe/model_plan.json` |
| Layer spec | `tools/pth/examples/repopt_vgg_int8_spec.json` |
| CIFAR-10 data root | `RepOpt/06_RepOpt_VGG/data` |

When `--image <path>` is used, Pillow loads the image, resizes it to 32x32, normalizes with CIFAR-10 constants, and quantizes it using the model input scale/zero-point.

## Output Directories

| Flow | Output directory |
|---|---|
| e2e | `sim/vgg_e2e/` |
| closed-loop | `sim/vgg_closed_loop/` |

These are generated artifacts and should not be committed.
