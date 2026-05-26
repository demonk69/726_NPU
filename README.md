# NPU OpenAI Lab Prototype

Current state: Linux + Verilator simulation of a PicoRV32-controlled NPU SoC for RepOpt VGG-like CIFAR-10 inference.

This README describes the maintained flow. Older Windows/Icarus plans and worklogs are archived under `doc/archive/` and should not be treated as current guidance.

## What Works Now

| Flow | Entry point | Purpose | Current status |
|---|---|---|---|
| Fast VGG e2e | `./run_vgg_e2e.sh` or `./run_all.sh standard` | Python pre-generates all Conv A tiles; CPU firmware runs 1024 NPU tiles plus avgpool/classifier/argmax | PASS, cat/class 3, 10,768,727 cycles |
| Runtime closed-loop VGG | `./run_vgg_closed_loop.sh` or `./run_all.sh closed_loop` | CPU firmware packs A tiles at runtime, runs NPU, performs per-channel requant/scatter, maxpool, avgpool, classifier | PASS on tested images, about 114M cycles |
| Arbitrary image e2e | `./run_all.sh image <file>` | Classify an image after host resize/normalize/quantize | Supported |

The closed-loop flow is the path closest to an FPGA deployment: model assets are static, while each inference only needs a new 3x32x32 INT8 input image.

## Quick Start

Requirements:

- Linux shell
- Python 3 with PyTorch
- Pillow for `--image` input
- Verilator 5.x
- GNU coreutils (`timeout`, `stdbuf`, `tee`, `grep`)

Run the verified fast baseline:

```bash
./run_vgg_e2e.sh
```

Run the unified entry point:

```bash
./run_all.sh standard 0
./run_all.sh image ./pic/test_cifar10_2.jpg
./run_all.sh closed_loop --image ./pic/test_cifar10_2.jpg
```

Run the full fast regression set:

```bash
./run_all.sh all
```

`./run_all.sh all` intentionally does not run the runtime closed-loop flow because it takes much longer than the fast baseline.

## VGG Flow Summary

### Fast E2E Flow

`tools/pth/gen_vgg_e2e.py` generates a DRAM image containing all pre-packed A tiles for all 9 Conv layers. Firmware iterates a 1024-entry tile table, starts the NPU for each tile, then performs avgpool, classifier, and argmax on the CPU.

Use this for quick regression and baseline confidence.

### Runtime Closed-Loop Flow

`tools/pth/gen_vgg_closed_loop.py` generates static model assets and the selected input image. Firmware does the real runtime work:

- pack A tiles from dense activation buffers
- configure and run the NPU tile by tile
- read raw INT32+bias tile results
- apply ReLU and per-output-channel Q24 requant on the CPU
- scatter INT8 bytes back to dense activation buffers
- run maxpool, avgpool, classifier, and argmax

This flow avoids the old per-16-channel hardware `QUANT_CFG` approximation and validates against the exact Python model target.

## FPGA Deployment Direction

The planned board flow is:

- SPI Flash stores firmware and static model assets once.
- Boot ROM copies firmware to SRAM and model assets to DRAM after reset.
- Upper PC preprocesses each image into 3x32x32 INT8 bytes.
- PC sends the bytes over UART.
- FPGA runs closed-loop inference and returns one class byte.

See `doc/uart_spi_fpga_plan.md` for the deployment plan.

## Documentation Map

| Document | Purpose |
|---|---|
| `doc/verification_status.md` | Current verified commands and results |
| `doc/architecture.md` | Current SoC/NPU architecture and address map |
| `doc/vgg_e2e_flow.md` | Fast e2e flow details |
| `doc/vgg_closed_loop_flow.md` | Runtime closed-loop flow details |
| `doc/uart_spi_fpga_plan.md` | UART/SPI Flash FPGA deployment plan |
| `doc/known_issues.md` | Current limitations and risks |
| `doc/user_manual.md` | Register and firmware-facing ABI reference |
| `doc/rtl_reference.md` | RTL module reference |
| `doc/conv_gemm_mapping.md` | Conv2D-to-GEMM mapping reference |
| `doc/archive/` | Historical plans and obsolete worklogs |

## Important Distinctions

- `run_vgg_e2e.sh` is the fast baseline, but Python pre-generates all Conv A tiles.
- `run_vgg_closed_loop.sh` is the runtime closed-loop path and is the basis for FPGA I/O deployment.
- The current SoC still has no UART/SPI peripherals in RTL. The FPGA deployment plan documents the next implementation stage.
- Do not use archived Windows/Icarus documents as current run instructions.
