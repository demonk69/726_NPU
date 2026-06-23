# Multi-Core NPU Architecture

## 1. Baseline

The working baseline is:

```text
soc_top style system
├── PicoRV32 reference/control CPU
│   ├── firmware instruction/data SRAM
│   ├── packs runtime A tiles
│   ├── programs NPU MMIO registers
│   ├── polls NPU STATUS
│   └── runs CPU-side requant/scatter/pool/classifier code
├── shared memory model in simulation
└── 1x npu_top
    ├── AXI-Lite register slave
    ├── AXI master DMA
    ├── W/A ping-pong buffers
    ├── reconfigurable 16x16 PE array
    └── result FIFO and writeback path
```

This plan keeps PicoRV32 as the reference/control CPU. ZCU102 is the FPGA
carrier for the PL design and resource target.

## 2. Target Multi-Core Structure

```text
PicoRV32 multi-core NPU system
├── PicoRV32 reference/control CPU
│   ├── firmware SRAM/BRAM
│   ├── simple memory bus
│   └── multi-core MMIO bridge
├── shared memory path
│   ├── simulation: shared DRAM model / multi-port model
│   └── implementation: board memory fabric or shared memory controller
└── npu_mc_top(NUM_CORES)
    ├── npu_top #0
    │   ├── AXI-Lite register window
    │   ├── AXI master memory port
    │   └── IRQ/status output
    ├── npu_top #1
    │   ├── AXI-Lite register window
    │   ├── AXI master memory port
    │   └── IRQ/status output
    └── ...
```

Each `npu_top` is kept unchanged in the first version. The multi-core logic is
outside the core: address decode, replicated ports, memory routing, and firmware
scheduling.

## 3. ZCU102 Boundary

ZCU102-specific board validation is not a planning checkpoint here. The RTL
should still be written so it can map to ZCU102 resources cleanly:

- Use a synthesizable top-level boundary around PicoRV32, `npu_mc_top`, and memory/interconnect ports.
- Keep simulation-only `dram_model` out of the synthesis boundary.
- Keep clock/reset assumptions simple: one system clock and one active-low reset for the first version.
- Avoid assumptions from any previous board-specific runtime path.

## 4. Work Partitioning

Conv2D is mapped to GEMM:

```text
C[M, N] = A[M, K] x W[K, N] + bias[N]
M = OH x OW
K = Cin x KH x KW
N = Cout
```

For a fixed `M` tile:

- `A[M,K]` is identical for all output-channel slices.
- Each core reads a different `W[K,N_tile]`.
- Each core writes a disjoint output-channel slice.
- K accumulation stays inside each NPU core.

Therefore the first multi-core partition is by `N` tile.

## 5. First-Pass Scheduling Model

Use a conservative one-N-tile-per-core-per-round scheduler:

```text
for each layer:
  clear OFM
  for each M tile:
    PicoRV32 packs A_WORK_SHARED once

    for n_round in N tiles grouped by NUM_CORES:
      launch core0 on N tile n_round*NUM_CORES + 0
      launch core1 on N tile n_round*NUM_CORES + 1
      ...
      poll all launched cores
      postprocess each core's R_WORK into dense OFM

  layer barrier is complete when all M/N tiles are done
```

This keeps the current static weight layout valid because each core still runs
one existing N tile at a time.

## 6. Later Optimized Scheduling

After the conservative scheduler is correct, a later optimization can assign a
contiguous N range to each core:

```text
core0: N tiles [0 .. k)
core1: N tiles [k .. 2k)
```

This requires a matching contiguous weight stream layout. The current generator
aligns every N tile independently, while the RTL's internal multi-N-tile stride
does not include that per-tile alignment. Do not enable multi-N-tile-per-core
launches until the weight repack is changed and tested.

## 7. MMIO Map

Use a 256-byte register window per core:

```text
0x02000000 - 0x020000FF : Core 0 registers
0x02000100 - 0x020001FF : Core 1 registers
0x02000200 - 0x020002FF : Core 2 registers
0x02000300 - 0x020003FF : Core 3 registers
```

Decode rule:

```text
offset = addr - NPU_BASE
valid  = offset < NUM_CORES * 0x100
core   = offset[11:8]
local  = offset[7:0]
```

Do not allow high addresses to alias to a valid core.

## 8. Synchronization

The first version uses polling:

- PicoRV32 writes all registers for each launched core.
- PicoRV32 starts the cores.
- PicoRV32 polls each launched core's `STATUS` register.
- If any core reports error, firmware writes the failure marker and halts.

IRQ wiring can exist as an optional debug/status signal, but firmware should not
depend on IRQ delivery for the first multi-core version.

## 9. Resource Notes

The incremental cost of each extra core is dominated by:

- One 16x16 PE array.
- Two ping-pong buffers.
- One DMA engine.
- One AXI-Lite register block and controller.
- One result FIFO/writeback path.

PicoRV32 and the multi-core bridge are small compared with the NPU replicas.
Before moving beyond 2 cores, confirm that buffer storage maps to BRAM or another
intended memory resource rather than registers.
