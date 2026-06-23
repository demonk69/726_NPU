# DRAM Layout for Multi-Core NPU

## Design Rules

- `A_WORK` is shared across cores because A depends on M/K, not N.
- `R_WORK` is per-core because each NPU writes raw INT32 results independently.
- Dense activation buffers are shared and written by disjoint output-channel slices.
- Weights, bias, Q24 multipliers, and classifier data are read-only during inference.
- Simulation memory size must be generated from the maximum used address, not from an old fixed 60KB assumption.

## Current Single-Core Baseline Constants

The maintained closed-loop generator currently uses these base addresses:

| Symbol | Address | Notes |
|--------|--------:|-------|
| `ACT_A` | `0x00010000` | Dense activation buffer A |
| `ACT_B` | `0x00030000` | Dense activation buffer B |
| `A_WORK` | `0x00050000` | Packed A tile workspace |
| `R_WORK` | `0x00068000` | Raw result workspace |
| `FEAT_BASE` | `0x00069000` | Avgpool feature buffer |
| `SCORE_BASE` | `0x0006A000` | Classifier score buffer |
| `MARKER_ADDR` | `0x0006B000` | Progress/result marker |
| `DESC_BASE` | `0x0006C000` | Firmware descriptors |
| `STATIC_BASE` | `0x00070000` | Weights, bias, Q24 multipliers, classifier |

Current generated closed-loop cases already use much more than 60KB of memory.
For example, the current generated default case reports `DRAM_WORDS=1474560` and
`max_addr=0x00580c28`.

## Shared A Workspace Size

For INT8 tile mode, the packed A workspace size is approximately:

```text
A_WORK_BYTES = padded_K * A_PACK_LANES
```

For the largest current VGG layer:

```text
K = 4608
A_PACK_LANES = 16
A_WORK_BYTES = 73728 bytes
```

Allocate at least 128KB for `A_WORK_SHARED` to leave alignment and shape margin.

## Per-Core Result Workspace Size

For the conservative one-N-tile-per-core scheduler:

```text
R_WORK_BYTES = tile_rows * tile_cols * 4
```

The largest current tile result is 1024 bytes (`16x16` or `8x32`). Allocate 16KB
per core so the same layout can also support later contiguous N-range launches.

## Proposed Scalable Layout

Use this layout for 2-core first and keep it valid for 4-core scaling:

```text
0x00010000  ACT_A               128KB
0x00030000  ACT_B               128KB
0x00050000  A_WORK_SHARED       128KB
0x00070000  R_WORK_CORE0         16KB
0x00074000  R_WORK_CORE1         16KB
0x00078000  R_WORK_CORE2         16KB  optional
0x0007C000  R_WORK_CORE3         16KB  optional
0x00080000  FEAT_BASE             4KB
0x00081000  SCORE_BASE            4KB reserved
0x00082000  MARKER_ADDR           4KB reserved
0x00083000  DESC_BASE            52KB reserved to 0x0008FFFF
0x00090000  STATIC_BASE          generated static assets
```

Firmware constants:

```python
ACT_A_BASE      = 0x00010000
ACT_B_BASE      = 0x00030000
A_WORK_SHARED   = 0x00050000
R_WORK_BASE     = [0x00070000, 0x00074000, 0x00078000, 0x0007C000]
R_WORK_STRIDE   = 0x00004000
FEAT_BASE       = 0x00080000
SCORE_BASE      = 0x00081000
MARKER_ADDR     = 0x00082000
DESC_BASE       = 0x00083000
STATIC_BASE     = 0x00090000
```

`STATIC_BASE` moving upward increases the generated maximum address by a small
constant offset. `DRAM_WORDS` should continue to be derived from the generated
maximum address.

## Access Ownership

| Region | Writer | Readers | Sharing rule |
|--------|--------|---------|--------------|
| `ACT_A/ACT_B` | PicoRV32 postprocess/pool | PicoRV32, NPU DMA when used as input | Shared dense buffers |
| `A_WORK_SHARED` | PicoRV32 | All launched NPU cores | Pack once per M tile; do not overwrite until all launched cores are done |
| `R_WORK_CORE[i]` | NPU core i | PicoRV32 | Private per core |
| `STATIC_BASE` | Generator/loader | PicoRV32, all NPU cores | Read-only during inference |
| `MARKER_ADDR` | PicoRV32 | Testbench/debug host | Progress and final result |
