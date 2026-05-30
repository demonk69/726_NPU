# Known Issues

Updated: 2026-05-30

This document lists current limitations. Historical issue lists are archived and should not be used as current status.

## FPGA Deployment Gaps

| Issue | Status | Impact | Planned direction |
|---|---|---|---|
| No Vivado/PYNQ integration yet | Open | NPU has not been packaged into a PYNQ-Z2 bitstream | Add PS+PL block design and export bitstream/`.hwh` |
| No UART peripheral | Deferred | Not a blocker for PYNQ; normal PYNQ Python/SSH/notebook path is primary | Add UART later only if a serial transport is required |
| No SPI Flash reader | Deferred | PYNQ route loads assets from PS/DDR rather than PL flash | Keep for future pure-PL boards |
| No Boot ROM | Deferred | PYNQ route uses PS ARM runtime rather than PL firmware boot | Keep for future pure-PL boards |

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
