# PYNQ-Z2 Deployment Plan

Updated: 2026-05-30

This is the current board-deployment plan. Older UART/SPI/Boot-ROM-only plans have been removed to avoid treating obsolete bring-up paths as current guidance.

## Target

Run VGG closed-loop inference on PYNQ-Z2 and return, once per image:

- classification result
- raw performance counters
- host-computed bus utilization, TOPS, and cycle counts

## Architecture Decision

Primary route for PYNQ-Z2:

```text
PC
  |
  | UART 921600 or PYNQ host channel
  v
Zynq PS ARM
  |
  | AXI-Lite: NPU registers and performance counters
  | AXI/DDR: image, weights, activations, results
  v
PL NPU
```

The NPU RTL stays board-independent. PYNQ-Z2-specific work belongs in the PS/PL wrapper, Vivado integration, and host/runtime software.

## CPU Boundary

- Runtime CPU for PYNQ-Z2 is the Zynq PS ARM.
- PicoRV32 remains useful for the current simulation SoC and for a future pure-PL board route, but it is not the primary PYNQ-Z2 deployment CPU.
- Python/golden reference CPU stays on the host PC for validation only. It is not implemented in PL.

The PS ARM runtime takes over the work currently performed by PicoRV32 firmware in closed-loop simulation:

- image receive / buffer fill
- A tile runtime packing
- NPU register programming
- polling or interrupt wait
- ReLU and per-channel Q24 requant
- scatter back to dense activation buffers
- maxpool, avgpool, classifier, argmax
- performance counter readback

## Performance Counter Policy

Counters should not affect the NPU datapath. The final board path should expose raw counters and compute derived metrics on PS or PC.

Keep raw counters:

- `total_cycles`
- `busy_cycles`
- `compute_cycles`
- `dma_cycles`
- `m_axi_rd_bytes`
- `m_axi_wr_bytes`
- `m_axi_rd_beats`
- `m_axi_wr_beats`
- `ops` or `mac_ops`

Avoid hard real-time hardware division for the board build. Current derived RTL outputs such as TOPS, bandwidth, and utilization are useful for simulation, but combinational `/` can consume significant resources and hurt timing closure on FPGA.

Compute on host:

```text
TOPS = ops * f_clk / busy_cycles / 1e12
rd_util = rd_beats / total_cycles
wr_util = wr_beats / total_cycles
```

The bus-utilization scope is the NPU AXI master traffic unless PS/DDR system counters are added separately.

## Counter RTL Work

1. Add `PERF_CTRL`:

| Bit | Meaning |
|---:|---|
| 0 | clear raw counters |
| 1 | snapshot raw counters |
| 2 | optional freeze |

2. Add snapshot registers for all raw counters.
3. Keep existing read addresses where practical.
4. Parameterize or remove board-build hardware division outputs.
5. Verify clear/snapshot/read consistency from AXI-Lite.

Snapshot is required because 64-bit counters are read as two 32-bit words. PS should trigger snapshot once after each inference and read the stable shadow values.

## PC/PS Result Packet

UART target baud: `921600`.

Per image, return one fixed packet:

| Field | Bytes |
|---|---:|
| magic | 1 |
| class_id | 1 |
| status | 1 |
| total_cycles | 4 |
| busy_cycles | 4 |
| compute_cycles | 4 |
| dma_cycles | 4 |
| rd_bytes | 4 |
| wr_bytes | 4 |
| rd_beats | 4 |
| wr_beats | 4 |
| ops_low | 4 |
| ops_high | 4 |
| checksum or crc | 1 |

Input image payload remains 3072 signed INT8 bytes in CHW order after host preprocessing.

## Workstream 1: Documentation

Maintain only the current PYNQ-Z2 deployment plan and current reference docs.

Tasks:

- keep this document as the deployment source of truth
- remove obsolete plan documents
- keep README, architecture, user manual, and RTL reference links current
- document the counter register map once `PERF_CTRL` and snapshots are implemented

Acceptance:

- no current doc links to removed plan files
- ref CPU / PS ARM / PicoRV32 responsibilities are unambiguous
- counter policy states raw-counter readback with host-side derived metrics

## Workstream 2: Circuit Review And Modification

Tasks:

- audit existing counters and AXI-Lite register map
- implement raw-counter clear/snapshot/freeze
- avoid board-build critical paths from division operators
- run Verilator lint
- run micro regressions, especially `8x32 M=13 K=9 N=40 bias`
- run selected closed-loop regressions when runtime is practical

Acceptance:

- NPU classification behavior is unchanged
- raw counters can be read consistently
- snapshot values are stable across multiword reads
- counter logic does not feed back into compute/DMA control

## Workstream 3: PYNQ-Z2 Port

Tasks:

- package PL NPU as Vivado IP or top-level module
- connect AXI-Lite register interface to PS
- connect NPU AXI master to PS DDR via HP/AXI interconnect
- expose optional NPU interrupt to PS
- generate bitstream and `.hwh`
- write PS runtime to perform the current closed-loop firmware work
- write host tool to send image and decode result/perf packet

Bring-up order:

1. PS reads NPU ID/status registers over AXI-Lite.
2. PS runs a tiny GEMM/tile smoke test from DDR.
3. PS verifies counter clear/snapshot/readback.
4. PS runs one fixed image and returns class.
5. PC sends image over UART 921600 and receives class plus raw counters.
6. PC computes and prints cycles, TOPS, read/write bus utilization.
7. Run repeated images to check buffer and counter isolation.

Acceptance:

- PYNQ-Z2 returns correct class for known CIFAR-10 test images
- each image returns one performance packet
- host output includes cycles, TOPS, bus utilization, and raw counter dump
- repeated inference does not require FPGA reconfiguration

## Deferred Pure-PL Route

The previous UART/SPI Flash/Boot ROM/PicoRV32-only board route is deferred. It remains a possible path for non-Zynq pure FPGA boards, but is not the PYNQ-Z2 primary implementation path.
