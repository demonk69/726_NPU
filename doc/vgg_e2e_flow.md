# VGG E2E Flow

Updated: 2026-05-26

This document describes the fast RepOpt VGG end-to-end baseline driven by `run_vgg_e2e.sh` and `./run_all.sh standard`.

## Purpose

The e2e flow is the fast regression baseline for the full 9-Conv VGG-like CIFAR-10 model. It verifies that the RTL NPU can execute the complete Conv tile chain and that the PicoRV32 firmware can finish avgpool, classifier, and argmax.

This flow is not the deployment model. Python pre-generates every Conv A tile stream before simulation.

## Entry Points

```bash
./run_vgg_e2e.sh
./run_vgg_e2e.sh 7
./run_vgg_e2e.sh --image ./pic/test_cifar10_2.jpg
./run_all.sh standard 0
./run_all.sh image ./pic/test_cifar10_2.jpg
```

## Generator

`tools/pth/gen_vgg_e2e.py` generates:

| Artifact | Purpose | Per-image? |
|---|---|---|
| `dram_init.hex` | Input, all pre-packed A tiles, weights, bias, classifier params, tile table | Yes |
| `soc_vgg.hex` | PicoRV32 firmware | No |
| `soc_vgg_params.vh` | Testbench parameters and expected label | Partly |
| `expected.hex` | Golden result data | Yes |

The generator uses `run_repopt_vgg_host.py` as the exact host reference.

## Runtime Data Flow

1. Testbench loads firmware into SRAM via `$readmemh`.
2. Testbench loads generated DRAM image via `$readmemh`.
3. Firmware iterates a 1024-entry tile table.
4. For each tile, firmware writes NPU registers and starts the NPU.
5. NPU reads pre-packed W/A tile streams from DRAM and writes results back to DRAM.
6. Firmware performs final avgpool, classifier, and argmax.
7. Firmware writes the prediction marker for the testbench.

## Tile Table ABI

The e2e firmware uses a VGG-specific 10-word tile table, not the RTL descriptor-v1 format.

| Word | Meaning |
|---:|---|
| 0 | `M_DIM` |
| 1 | `N_DIM` |
| 2 | `K_DIM` |
| 3 | W tile base address |
| 4 | A tile base address |
| 5 | R tile base address |
| 6 | Bias base address |
| 7 | Quant config word |
| 8 | Array config |
| 9 | Shape config |

## Verified Baseline

| Command | Result | Prediction | Cycles |
|---|---|---|---:|
| `./run_vgg_e2e.sh` | PASS | cat/class 3 | 10,768,727 |

## Correct Use

Use this flow when you need a fast known-good baseline before changing RTL, generator code, or firmware.

Do not describe this flow as full runtime closed-loop inference. It is full model classification, but Conv A tile generation is host-side.
