# PicoRV32 Firmware Changes for Multi-Core NPU

## Goal

Keep PicoRV32 as the reference/control CPU and extend the generated closed-loop
firmware so it can schedule multiple replicated `npu_top` cores.

The first version must preserve the current static weight layout and exact model
results. Performance optimization comes after correctness.

## Current Single-Core Loop

The current generated firmware does this per Conv layer:

```text
clear OFM
for m_base in M tiles:
  pack A_WORK from dense IFM
  for n_base in N tiles:
    program one NPU
    start one NPU
    poll STATUS
    read R_WORK
    ReLU + Q24 requant
    scatter into dense OFM
```

## First Multi-Core Loop

Use shared A packing and per-core result buffers:

```text
clear OFM
for m_base in M tiles:
  pack A_WORK_SHARED once

  n_tile = 0
  while n_tile < n_tiles:
    launched_mask = 0

    for core in 0..NUM_CORES-1:
      if n_tile < n_tiles:
        launch core on this N tile
        launched_mask |= (1 << core)
        n_tile += 1

    poll launched cores until all done

    for each launched core:
      postprocess R_WORK_CORE[core]
      scatter using that core's global n_base
```

This scheduler launches one existing N tile per core per round. It avoids the
weight layout problem caused by per-N-tile alignment in the current generator.

## Why `A_WORK` Is Shared

For a fixed `m_base`, packed A contains `A[M,K]`. It is independent of output
channel `N`. All cores should use:

```text
REG_A_ADDR = A_WORK_SHARED
```

Only these fields differ per core launch:

```text
REG_W_ADDR    = w_base + n_tile * w_stride
REG_R_ADDR    = R_WORK_BASE[core]
REG_BIAS_ADDR = bias_base + n_base * 4
Q24 lookup    = qcfg_base + n_base * 4
scatter base  = ofm channel n_base
```

## Per-Core Launch State

The firmware must track, for each core in the current round:

| Field | Purpose |
|-------|---------|
| `active` | Whether this core was launched in this round |
| `n_tile` | Logical N tile index assigned to the core |
| `n_base` | Global output-channel base, `n_tile * TC` |
| `r_base` | `R_WORK_BASE[core]` |
| `status_addr` | Core-local `REG_STATUS` address |

The final round of a layer may launch fewer than `NUM_CORES` cores.

## Register Programming Per Core

For the conservative scheduler, each core still sees one tile-shaped GEMM:

```text
M_DIM     = TR
N_DIM     = TC
K_DIM     = layer K
W_ADDR    = layer w_base + n_tile * w_stride
A_ADDR    = A_WORK_SHARED
R_ADDR    = R_WORK_BASE[core]
BIAS_ADDR = layer bias_base + n_base * 4
ARR_CFG   = ARR_TILE
CFG_SHAPE = selected shape
QUANT_CFG = QUANT_DISABLED
CTRL      = CTRL_BIAS_TILE
```

For `8x32`, `TC=32`, so `n_base = n_tile * 32`.

## Polling And Error Handling

Polling is the first-version synchronization mechanism:

```text
while done_mask != launched_mask:
  for each launched core not done:
    status = read STATUS
    if status.error:
      write FAIL_MARKER and halt
    if status.done:
      clear CTRL/start as current single-core code does
      done_mask |= (1 << core)
```

IRQ can be wired for debug, but firmware should not require it.

## Postprocess And Scatter

For each completed core:

```text
for col in 0..TC-1:
  qmult = Q24[n_base + col]
  for row in 0..TR-1:
    acc = R_WORK_CORE[core][row][col]
    q = relu_q24_requant(acc, qmult)
    OFM[m_base + row][n_base + col] = q
```

Boundary handling remains shape-aware. Current VGG layer dimensions are tile
divisible for maintained shapes, but the implementation should not bake in that
assumption beyond what the generator already validates.

## Generator Changes

Add arguments and emitted constants:

```text
--num-cores N
A_WORK_SHARED
R_WORK_BASE[0..N-1]
NPU_CORE_STRIDE = 0x100
```

Modify code generation in these areas:

| Area | Change |
|------|--------|
| Descriptor constants | Emit moved FEAT/SCORE/MARKER/DESC/STATIC bases |
| Conv loop | Pack A once per M tile, then schedule N tiles across cores |
| Launch helper | Add `core_id`, `n_tile`, and per-core R address |
| Wait helper | Poll only cores launched in the current round |
| Postprocess helper | Use per-core R base and saved global `n_base` |
| Metadata | Record `num_cores`, shared A layout, per-core R layout |

## Later Optimized Firmware

Do not start with multi-N-tile ranges per core. If enabled later, the generator
must also repack weights into a contiguous stream that matches RTL's internal
multi-N-tile stride. Without that repack, layers with per-tile alignment padding
will read wrong weights.

## Firmware Size

Current closed-loop firmware uses `MEM_WORDS=8192`. Multi-core scheduling should
reuse loops and helpers instead of fully unrolling by core. If code size grows,
increase the PicoRV32 SRAM parameter rather than compressing logic prematurely.
