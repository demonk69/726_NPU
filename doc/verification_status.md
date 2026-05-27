# Verification Status

Updated: 2026-05-26

This document is the current verification record. Older status files and worklogs in `doc/archive/` are historical and may describe superseded Windows/Icarus flows.

## Maintained Entry Points

| Command | Purpose | Expected status |
|---|---|---|
| `./run_vgg_e2e.sh` | Fast RepOpt VGG baseline | PASS |
| `./run_all.sh standard [idx]` | Unified wrapper for fast baseline | PASS |
| `./run_all.sh image <file>` | Fast baseline with arbitrary image input | Supported |
| `./run_vgg_closed_loop.sh [idx|--image <file>] [--shape <shape>]` | Runtime closed-loop VGG | PASS on tested default shape images |
| `./run_all.sh closed_loop [args...]` | Unified wrapper for runtime closed-loop | PASS on tested images |

`./run_all.sh all` is a fast regression set and intentionally does not include the long runtime closed-loop test.

## Current Verified Results

| Flow | Command | Result | Prediction | Cycles |
|---|---|---|---|---:|
| Fast VGG e2e baseline | `./run_vgg_e2e.sh` | PASS | cat/class 3 | 10,768,727 |
| Runtime closed-loop image2 | `./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg` | PASS | frog/class 6 | 114,014,769 |
| Runtime closed-loop image4 | `./run_all.sh closed_loop --image ./pic/test_cifar10_4.jpg` | PASS | dog/class 5 | 114,013,544 |
| Closed-loop generator sanity image3 | `python3 -B tools/pth/gen_vgg_closed_loop.py --out-dir /tmp/opencode/cl_test_3_fixed --image pic/test_cifar10_3.jpg` | PASS generation | fixed-runtime 6, exact-python 6 | N/A |

## Syntax Checks

| Command | Result |
|---|---|
| `python3 -B -m py_compile tools/pth/gen_vgg_closed_loop.py` | PASS |
| `bash -n run_vgg_closed_loop.sh` | PASS |
| `bash -n run_all.sh` | PASS |

## What Is Being Verified

### Fast E2E Baseline

The e2e flow verifies that the NPU can execute a 1024-tile, 9-Conv RepOpt VGG chain using pre-generated Conv A tiles. The CPU firmware performs avgpool, classifier, and argmax after the NPU tile chain.

This is the fast regression baseline. It is not a complete runtime layer-to-layer deployment because Python pre-generates all Conv A tile streams.

### Runtime Closed Loop

The closed-loop flow verifies that firmware can perform the runtime steps needed for deployment:

- receive/use a dense INT8 input activation buffer
- pack Conv A tiles at runtime
- run NPU tile GEMM with raw INT32+bias result output
- apply ReLU and per-output-channel Q24 requant on the CPU
- scatter INT8 bytes back to dense activation buffers
- run maxpool, avgpool, classifier, and argmax

The current closed-loop validation target is exact Python model output. The older per-16-channel hardware `QUANT_CFG` approximation is no longer used as the PASS criterion.

The default runtime closed-loop shape is `16x16`. The generator and script accept `4x4`, `8x8`, `16x16`, and `8x32`; only the default `16x16` commands above are currently recorded as full long-run RTL PASS results.

## Deployment Status

The RTL SoC is still a simulation SoC. It has PicoRV32, SRAM, DRAM model, and NPU, but no UART, SPI Flash controller, or boot ROM yet.

The planned FPGA deployment path is documented in `doc/uart_spi_fpga_plan.md`:

- SPI Flash stores firmware and static model assets.
- Boot ROM copies firmware and static assets into SRAM/DRAM.
- Upper PC preprocesses each image to 3x32x32 INT8 bytes.
- UART sends one image to FPGA and receives one class byte.

## Regression Notes

- Generated artifacts under `sim/vgg_e2e/` and `sim/vgg_closed_loop/` are not source files.
- `RepOpt/` and image/generated data should not be committed as source changes.
- The untracked `session` file is not part of the project source.
