# Closed-Loop VGG Inference Plan and Completion Record

Updated: 2026-05-26

This file is retained as the design record for the runtime closed-loop VGG path. The implementation now exists as `run_vgg_closed_loop.sh` and `tools/pth/gen_vgg_closed_loop.py`.

For current operational details, read `doc/vgg_closed_loop_flow.md`.

## Original Goal

Move from a host-prepacked VGG e2e test toward a deployment-style runtime flow:

- Python emits static model assets and test input only.
- PicoRV32 firmware packs Conv A tiles at runtime from dense activation buffers.
- NPU executes tile GEMM.
- Firmware performs post-processing, pooling, classifier, and argmax.
- Testbench validates the actual runtime prediction marker.

## Completion Status

| Phase | Goal | Status |
|---|---|---|
| A | Generate static assets independent of prepacked A tiles | Done |
| B | Add firmware runtime A-tile packing | Done |
| C | Run NPU tile GEMM from firmware-generated A tiles | Done |
| D | Scatter tile outputs back to dense activation buffers | Done |
| E | Implement maxpool, avgpool, classifier, argmax in firmware | Done |
| F | Validate full 9-Conv closed-loop path | Done |
| G | Expose maintained entry point `run_vgg_closed_loop.sh` | Done |

## Final Implementation

Entry points:

```bash
./run_vgg_closed_loop.sh [img_idx]
./run_vgg_closed_loop.sh --image <file>
./run_all.sh closed_loop [args...]
```

Main files:

| File | Role |
|---|---|
| `tools/pth/gen_vgg_closed_loop.py` | Generates static assets, firmware, DRAM init, metadata |
| `tb/tb_soc_vgg_closed_loop.v` | Loads firmware/DRAM, monitors marker, reports PASS/FAIL |
| `run_vgg_closed_loop.sh` | Build/run wrapper with timeout and fail detection |
| `run_all.sh` | Unified entry point with `closed_loop` alias |

## Important Correction From Bring-Up

An intermediate version used the hardware per-16-channel `QUANT_CFG` approximation as the expected label. That was misleading because several inputs collapsed to `cat/class 3` while the exact Python model predicted other classes.

The fixed implementation now uses this policy:

- NPU produces raw INT32+bias tile outputs.
- CPU firmware applies ReLU and per-output-channel Q24 requant.
- PASS target is exact Python model prediction.
- Testbench prints both exact-python and fixed-runtime diagnostics.

## Verified Results

| Input | Result | Prediction | Cycles |
|---|---|---|---:|
| `pic/test_cifar10_2.jpg` | PASS | frog/class 6 | 114,014,769 |
| `pic/test_cifar10_4.jpg` | PASS | dog/class 5 | 114,013,544 |

## Remaining Work

The closed-loop inference flow is implemented in simulation. FPGA deployment still requires UART, SPI Flash reader, Boot ROM, and host-side serial tooling. See `doc/uart_spi_fpga_plan.md`.
