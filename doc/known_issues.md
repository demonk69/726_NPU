# Known Issues

Updated: 2026-06-01

This document lists current limitations. Historical issue lists are archived and should not be used as current status.

## FPGA Deployment Gaps

| Issue | Status | Impact | Planned direction |
|---|---|---|---|
| No Vivado/PYNQ integration yet | Open | NPU has not been packaged into a PYNQ-Z2 bitstream | Add PS+PL block design and export bitstream/`.hwh` |
| No UART peripheral | Deferred | Not a blocker for PYNQ; normal PYNQ Python/SSH/notebook path is primary | Add UART later only if a serial transport is required |
| No SPI Flash reader | Deferred | PYNQ route loads assets from PS/DDR rather than PL flash | Keep for future pure-PL boards |
| No Boot ROM | Deferred | PYNQ route uses PS ARM runtime rather than PL firmware boot | Keep for future pure-PL boards |

## RTL Circuit Issues Found Before PYNQ Bring-Up

| Issue | Status | Impact | Planned direction |
|---|---|---|---|
| AXI-Lite `BVALID` is not held until `BREADY` | Fixed in RTL | PS `MMIO.write()` can hang if the write response is missed | `npu_axi_lite` now registers `BVALID` and keeps one outstanding write response until handshake |
| PYNQ AXI wrapper is missing | Added, pending Vivado validation | Vivado BD may not connect `npu_top` cleanly to PS GP0/HP0 and address offsets may be wrong | `npu_pynq_wrapper` adds Xilinx AXI sideband defaults and local AXI-Lite offset decode |
| DMA ignores AXI `BRESP/RRESP` | Fixed in RTL | HP/DDR access errors can silently produce wrong results | `npu_dma` emits AXI response error bits and `npu_ctrl` latches them into `ERR_STATUS`/`STATUS.error` |
| DMA assumes 32-bit aligned accesses | Guarded in RTL | Non-4-byte-aligned addresses or lengths can cross 4KB boundaries or overwrite adjacent DDR bytes | Unaligned DMA requests now raise `ERR_STATUS`; byte-strobe support remains deferred until needed |
| Direct-start config lacks illegal dimension checks | Guarded in RTL | Zero `M/N/K` or invalid direct parameters can underflow counters or run an invalid transfer | Direct starts with zero `M/N/K` or zero Conv2D shape/stride/dilation fields now raise `ERR_STATUS` |
| Several controller/DMA lengths are 16-bit limited | Open | Large `K` or byte lengths can silently wrap | Document hard limits and add range checks before widening or splitting transactions |
| Controller has large combinational division/modulo paths | Open | Vivado timing/resource risk on PYNQ-Z2 | Move tile planning to registered/multicycle logic or restrict parameters |
| `pingpong_buf` likely infers registers/muxes instead of BRAM | Open | Area and timing risk for vector reads | Reset pointers only and redesign vector storage as banked memory if resources fail |
| Tile result capture uses dynamic loop bound | Fixed in RTL | Vivado synthesis compatibility risk | `npu_top` now uses a fixed `MAX_TILE_RESULTS` loop with an internal capture enable condition |
| WS tile flow performance counters are inaccurate | Fixed in RTL | `compute_cycles` can be under-counted for WS tile flow | `npu_top` now counts tile WS direct compute/drain cycles in `perf_compute_valid` |

## Runtime Flow Limitations

| Issue | Status | Impact |
|---|---|---|
| Closed-loop requant is CPU-side | Expected | Correct and exact, but slower than a future hardware per-channel requant path |
| Closed-loop runtime is long in Verilator | Expected | Around 114M cycles for default `16x16`; around 161M cycles for tested `4x4` OS/WS, so the default script timeout is 250M cycles |
| Arbitrary `--image` inputs have no true CIFAR label | Expected | PASS compares against exact Python model output, not semantic ground truth |

## Tooling Limitations

| Issue | Status | Impact |
|---|---|---|
| Python generator still needs PyTorch | Expected | Current simulation generation loads the checkpoint and computes exact/fixed golden values; current environment has CPU PyTorch installed |
| Host preprocessing for FPGA is not implemented yet | Open | Planned host script should use Pillow + numpy only, without PyTorch |
| SPI Flash image packaging is not implemented yet | Open | Static model assets are still generated as DRAM hex |

## Documentation Rules

- Current run commands must be documented in `README.md` and `doc/verification_status.md`.
- Historical plans belong in `doc/archive/`.
- Any doc that says a feature is "planned" must be updated once the feature is implemented.
