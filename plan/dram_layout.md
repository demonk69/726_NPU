# DRAM Layout for Multi-Core NPU

## Design Constraints

- DRAM is shared between CPU and NUM_CORES NPU DMA ports
- Weight data: read-only, shared across cores
- Bias data: read-only, shared across cores
- Input activations: read-only, shared across cores (only one IFM per layer)
- A tile workspace: per-core (each core packs tiles for its N range)
- Result workspace: per-core (each core writes its N-range results)
- Output activations: shared but written by disjoint N ranges (no conflict)

## Current Single-Core Layout (Baseline)

| Symbol | Start Address | Size | Contents |
|--------|--------------|------|----------|
| ACT_A | `0x00010000` | 32KB | Dense activation buffer A |
| ACT_B | `0x00030000` | 32KB | Dense activation buffer B |
| A_WORK | `0x00050000` | 32KB | Packed A tile workspace |
| R_WORK | `0x00068000` | 16KB | Raw NPU result workspace |
| FEAT_BASE | `0x00069000` | 4KB | Avgpool feature buffer |
| SCORE_BASE | `0x0006A000` | 1KB | Classifier score buffer |
| MARKER_ADDR | `0x0006B000` | 256B | Progress/result marker |
| DESC_BASE | `0x0006C000` | 16KB | Firmware layer descriptor table |
| STATIC_BASE | `0x00070000` | 64KB | Weights, bias, Q24 multipliers, classifier params |

DRAM total: ~60KB (0xF000 bytes used of 60KB).

## 2-Core Layout (Phase 1 Target)

```
0x0000_0000 ┌─────────────────────┐
            │ SRAM (CPU only)     │ 4KB — firmware code
0x0000_1000 ├─────────────────────┤
            │                     │
0x0001_0000 │ ACT_A               │ 32KB — dense IFM/OFM buffer A (shared, read-only)
0x0001_8000 ├─────────────────────┤
            │                     │
0x0003_0000 │ ACT_B               │ 32KB — dense IFM/OFM buffer B (shared, read-only)
0x0003_8000 ├─────────────────────┤
            │                     │
0x0005_0000 │ A_WORK_CORE0        │ 32KB — Core0 A tile workspace
0x0005_8000 ├─────────────────────┤
            │ A_WORK_CORE1        │ 32KB — Core1 A tile workspace
0x0006_0000 ├─────────────────────┤
            │                     │
0x0006_8000 │ R_WORK_CORE0        │ 16KB — Core0 result workspace
0x0006_C000 ├─────────────────────┤
            │ R_WORK_CORE1        │ 16KB — Core1 result workspace
0x0007_0000 ├─────────────────────┤
            │                     │
            │ STATIC_BASE         │ 64KB — W, bias, Q24 multipliers, classifier
            │                     │
0x0008_0000 └─────────────────────┘
```

**Total DRAM needed**: ~40KB (well within 60KB limit)

## Address Constants for Firmware

```python
# Shared buffers (same as single-core)
ACT_A_BASE   = 0x00010000
ACT_B_BASE   = 0x00030000

# Per-core buffers
A_WORK_BASE  = [0x00050000, 0x00058000]       # strided by 32KB
R_WORK_BASE  = [0x00068000, 0x0006C000]       # strided by 16KB

# Shared static data
FEAT_BASE    = 0x00069000  # unchanged (or move up to avoid overlap)
SCORE_BASE   = 0x0006A000
MARKER_ADDR  = 0x0006B000
DESC_BASE    = 0x0006C000
STATIC_BASE  = 0x00070000
```

**Note**: FEAT_BASE through DESC_BASE overlap with R_WORK_CORE1.
These need to be relocated upward. See section below.

## Address Conflict Resolution

Current single-core layout has overlap between R_WORK end (0x0006C000) and
DESC_BASE start (0x0006C000). In the 2-core layout, R_WORK_CORE1 occupies
0x0006C000-0x0006FFFF, conflicting with FEAT_BASE, SCORE_BASE, etc.

### Solution: Relocate fixed buffers upward

```python
# Updated layout (no conflicts):
A_WORK_BASE  = [0x00050000, 0x00058000]       # Core0, Core1
R_WORK_BASE  = [0x00068000, 0x0006C000]       # Core0, Core1
# --- gap ---
FEAT_BASE    = 0x00070000  # moved from 0x00069000
SCORE_BASE   = 0x00071000  # moved from 0x0006A000
MARKER_ADDR  = 0x00072000  # moved from 0x0006B000
DESC_BASE    = 0x00073000  # moved from 0x0006C000
STATIC_BASE  = 0x00080000  # moved from 0x00070000
```

This gives each region clear separation. DRAM model (60KB = 0xF000) supports
addresses up to 0x0000FFFF, so 0x00080000 + 64KB = 0x00090000 stays within
range if we extend DRAM_WORDS. Need to bump DRAM_WORDS from 15360 to ~32768.

## 4-Core Layout (Phase 2)

```python
A_WORK_BASE  = [0x00050000, 0x00058000, 0x00060000, 0x00068000]  # 32KB each
R_WORK_BASE  = [0x00070000, 0x00074000, 0x00078000, 0x0007C000]  # 16KB each
FEAT_BASE    = 0x00080000
SCORE_BASE   = 0x00081000
MARKER_ADDR  = 0x00082000
DESC_BASE    = 0x00083000
STATIC_BASE  = 0x00090000
```

DRAM_WORDS should be ~40960 (160KB) for 4 cores. For CIFAR-10 VGG this is fine
in simulation; on real FPGA Zynq-7000 DDR has 512MB+.

## Python Generator Changes

The `gen_vgg_closed_loop.py` must emit these address constants into the
firmware's data section and use them when computing per-core W/A/R addresses.
