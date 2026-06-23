# Multi-Core NPU Architecture

## 1. Current Single-Core Architecture (Baseline)

```
SoC (soc_top)
├── PicoRV32 CPU (RV32IMC)
│   ├── IRQ[7] ← NPU done
│   └── Memory bus → SRAM / DRAM / NPU MMIO
├── SRAM (4KB) — firmware code + data
├── DRAM model (60KB) — weights, activations, results
├── AXI-Lite bridge — CPU mem bus → NPU AXI4-Lite
└── 1x npu_top
    ├── npu_axi_lite — MMIO register file @ 0x02000000
    ├── npu_ctrl   — ping-pong overlap FSM
    ├── npu_dma    — AXI4 master DMA (dual FSM: load + writeback)
    ├── W/A pingpong_buf (64-word depth each)
    ├── 16x16 reconfigurable PE array
    └── Result FIFO + tile serializer + post-process
```

Single NPU has:
- 1x AXI4-Lite slave (CPU config)
- 1x AXI4 master (DMA to DRAM)
- 1x IRQ output

## 2. Target Multi-Core Architecture

```
SoC (soc_mc_top)
├── PicoRV32 CPU
│   ├── IRQ[7]   ← Core0 done
│   ├── IRQ[8]   ← Core1 done
│   └── IRQ[9+]  ← Core2+ done (if NUM_CORES > 2)
├── SRAM (4KB)
├── AXI-Lite bridge (multi-core address decode)
├── DRAM model (multi-port: 1 CPU + NUM_CORES NPU ports)
└── npu_mc_top (parameter NUM_CORES=N)
    ├── npu_top #0 (complete instance)
    │   ├── AXI-Lite slave ← bridge (addr matched)
    │   ├── AXI4 master → DRAM port 0
    │   └── IRQ → CPU IRQ[7]
    ├── npu_top #1 (complete instance)
    │   ├── AXI-Lite slave ← bridge (addr matched)
    │   ├── AXI4 master → DRAM port 1
    │   └── IRQ → CPU IRQ[8]
    └── npu_top #N ... (generate)
```

**Key property: each `npu_top` instance is UNMODIFIED from the single-core design.**

## 3. Work Partitioning Strategy

### Why partition by N (output channel)

VGG Conv2D → GEMM mapping:

```
C[M, N] = A_im2col[M, K] × W_col[K, N] + bias[N]
M = OH × OW       (output spatial positions)
K = Cin × KH × KW (convolution window)
N = Cout           (output channels)
```

The tiled decomposition:

```
for m in [0..M step 16]:        ← spatial tile loop
    pack A_tile from dense IFM
    for n in [0..N step 16]:    ← channel tile loop
        C_tile[m][n] = Σ_k A[k] × W[k]
```

Within a layer:
- **Different (M,N) tiles write to DISJOINT OFM regions** — no RAW/WAR/WAW hazard
- **K accumulation is fully internal to PE accumulators** — firmware sees only complete results
- **The OFM is zero-cleared before each layer** — no read-modify-write between tiles

Therefore: **partition the N loop across cores. Zero intra-layer dependency.**

```
Layer L: Cout=128, TR=16, TC=16

Core0: N ∈ [0, 63]   — responsible for output channels 0..63
       Iterates all M tiles, iterates all K (hardware), writes OFM[0..63]

Core1: N ∈ [64, 127] — responsible for output channels 64..127
       Iterates all M tiles, iterates all K (hardware), writes OFM[64..127]

Weights:   read-only, shared, no conflict (each core reads different N slice)
Input:     read-only, shared, no conflict (each core needs full spatial range)
Output:    disjoint N slices, no conflict
```

### Layer barrier

Between layers, cores must synchronize:

```
Layer L complete → all cores done → firmware swaps IFM/OFM buffers
→ Clears new OFM → launches all cores for Layer L+1
```

This barrier is implemented by firmware polling each core's STATUS register.

## 4. Why NOT Partition by K or M

### Partition by K — REQUIRES inter-core dependency

```
Core0: partial_sum[m][n] = A[k=0..7] × W[k=0..7]
Core1: partial_sum[m][n] = A[k=8..15] × W[k=8..15]
                    ↓
Need reduction: full[m][n] = partial_sum_0 + partial_sum_1
```

This requires either:
- Hardware reduction tree (complex, not in current design)
- Firmware read-modify-write accumulation (slow, breaks the DRAM write-back model)
- Changing PE accumulators to support inter-core partial-sum injection

**Not viable without major hardware changes.**

### Partition by M — works, but worse load balance

For many VGG layers, M is small (e.g., OH×OW=4 for early layers with pooling).
Partitioning 4 rows across 2+ cores leaves some cores idle.
N is generally larger (Cout=64..256) and divides better across cores.

## 5. MMIO Address Map

```
0x02000000 - 0x020000FF : Core 0 register file  (256B)
0x02000100 - 0x020001FF : Core 1 register file  (256B)
0x02000200 - 0x020002FF : Core 2 register file  (256B)
...
```

Core selection via `addr[11:8]` in the AXI-Lite bridge.
Lower bits `addr[7:0]` pass through unchanged to the selected core's register file.

## 6. IRQ Mapping

```
npu_irq_core0 → cpu_irq[7]
npu_irq_core1 → cpu_irq[8]
npu_irq_core2 → cpu_irq[9]
npu_irq_core3 → cpu_irq[10]
```

PicoRV32 IRQ handler polls STATUS registers of all cores to determine which
core(s) completed.

## 7. Clock and Reset

All cores share the same `sys_clk` and `sys_rst_n`.
No per-core clock gating at the top level (each core has its own `npu_power`
module internally for PE array row/column CE).
