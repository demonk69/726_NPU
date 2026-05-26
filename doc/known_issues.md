# Known Issues

Updated: 2026-05-26

This document lists current limitations. Historical issue lists are archived and should not be used as current status.

## FPGA Deployment Gaps

| Issue | Status | Impact | Planned direction |
|---|---|---|---|
| No UART peripheral | Open | FPGA cannot receive images from an upper PC yet | Add memory-mapped UART at `0x03000000` |
| No SPI Flash reader | Open | Static model assets are loaded by testbench `$readmemh`, not by hardware | Add SPI Flash read-only controller |
| No Boot ROM | Open | Firmware and static assets are simulation-initialized | Add small boot ROM to copy Flash contents into SRAM/DRAM |
| No board constraints | Open | No real Xilinx board build yet | Add XDC after target board is fixed |

## Runtime Flow Limitations

| Issue | Status | Impact |
|---|---|---|
| Closed-loop requant is CPU-side | Expected | Correct and exact, but slower than a future hardware per-channel requant path |
| Closed-loop runtime is long in Verilator | Expected | Around 114M cycles for tested images |
| Arbitrary `--image` inputs have no true CIFAR label | Expected | PASS compares against exact Python model output, not semantic ground truth |

## Tooling Limitations

| Issue | Status | Impact |
|---|---|---|
| Python generator still needs PyTorch | Open | Current simulation generation loads the checkpoint and computes exact/fixed golden values |
| Host preprocessing for FPGA is not implemented yet | Open | Planned host script should use Pillow + numpy only, without PyTorch |
| SPI Flash image packaging is not implemented yet | Open | Static model assets are still generated as DRAM hex |

## Documentation Rules

- Current run commands must be documented in `README.md` and `doc/verification_status.md`.
- Historical plans belong in `doc/archive/`.
- Any doc that says a feature is "planned" must be updated once the feature is implemented.
