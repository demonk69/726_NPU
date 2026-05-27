# VGG Runtime Closed-Loop Flow

Updated: 2026-05-26

This document describes the runtime closed-loop RepOpt VGG path driven by `run_vgg_closed_loop.sh` and `./run_all.sh closed_loop`.

## Purpose

The closed-loop flow verifies the firmware behavior needed for an FPGA deployment. Python generates static model assets and the input activation buffer, while the PicoRV32 firmware performs runtime Conv tile packing, NPU scheduling, post-processing, pooling, classification, and argmax.

This is the maintained deployment-oriented flow.

## Entry Points

```bash
./run_vgg_closed_loop.sh
./run_vgg_closed_loop.sh 7
./run_vgg_closed_loop.sh --image ./pic/test_cifar10_2.jpg
./run_vgg_closed_loop.sh --shape 8x8 --image ./pic/test_cifar10_2.jpg
./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg
```

Aliases accepted by `run_all.sh`: `closed_loop`, `closed-loop`, and `full`.

## Generator

`tools/pth/gen_vgg_closed_loop.py` generates:

| Artifact | Purpose | Per-image? |
|---|---|---|
| `dram_init.hex` | Input image plus static model assets | Partly |
| `soc_vgg_closed_loop.hex` | PicoRV32 runtime firmware | No |
| `soc_vgg_closed_loop_params.vh` | Testbench paths, timeouts, expected labels | Partly |
| `expected_features.hex` | Fixed-runtime feature reference | Yes |
| `metadata.json` | Scores, labels, generated sizes | Yes |

The firmware is image-independent. The input image bytes and expected labels change per test case.

## Runtime Firmware Work

For each Conv layer, firmware performs:

1. Pack an A tile at runtime from dense activation memory.
2. Configure the NPU with W tile, packed A tile, result buffer, and bias base address.
3. Run tile GEMM in the selected shape mode.
4. Read raw INT32+bias tile results from `R_WORK`.
5. Apply ReLU and per-output-channel Q24 requant on the CPU.
6. Scatter resulting INT8 bytes into the dense output activation buffer.

Firmware also performs maxpool, avgpool, classifier, and argmax on the CPU.

## Shape Modes

The default closed-loop shape is the verified `16x16` path. The generator and run script also accept:

| `--shape` | `CFG_SHAPE` | Tile rows | Tile cols |
|---|---:|---:|---:|
| `4x4` | `0` | 4 | 4 |
| `8x8` | `1` | 8 | 8 |
| `16x16` | `2` | 16 | 16 |
| `8x32` | `3` | 8 | 32 |

The `8x32` mode uses the RTL two-pass folded array format, so W is generated as contiguous cols 0-15 then cols 16-31 pass streams. New work on packed-SIMD dataflow must preserve all four shape modes, not only the default 16x16 path.

## Quantization Policy

The validation target is the exact Python model output.

The old hardware `QUANT_CFG` path used one approximate multiplier per 16 output channels. That approximation caused some images to collapse to class 3. The closed-loop flow no longer treats that approximation as a PASS target.

Current closed-loop policy:

- NPU writes raw INT32+bias output.
- CPU applies ReLU and exact per-output-channel Q24 multiplier.
- Testbench expects `VGG_CLOSED_EXACT_LABEL`.
- Testbench prints both exact-python and fixed-runtime diagnostics.

## Address Map Used By Firmware

| Symbol | Address | Purpose |
|---|---:|---|
| `ACT_A` | `0x00010000` | Dense activation buffer A |
| `ACT_B` | `0x00030000` | Dense activation buffer B |
| `A_WORK` | `0x00050000` | Runtime packed A tile buffer |
| `R_WORK` | `0x00068000` | Raw NPU tile result buffer |
| `FEAT_BASE` | `0x00069000` | Avgpool feature buffer |
| `SCORE_BASE` | `0x0006A000` | Classifier score buffer |
| `MARKER_ADDR` | `0x0006B000` | Testbench progress/result marker |
| `DESC_BASE` | `0x0006C000` | Firmware layer descriptors |
| `STATIC_BASE` | `0x00070000` | Weights, bias, multipliers, classifier params |

## Verified Results

| Command | Result | Prediction | Cycles |
|---|---|---|---:|
| `./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg` | PASS | frog/class 6 | 114,014,769 |
| `./run_all.sh closed_loop --image ./pic/test_cifar10_4.jpg` | PASS | dog/class 5 | 114,013,544 |

These runtime results are for the default `--shape 16x16` mode unless the command states otherwise.

## Correct Use

Use this flow when validating deployment behavior or changes to firmware-side runtime packing/post-processing.

Do not compare this flow against the old hardware-qcfg approximation. The expected label is exact-python output.
