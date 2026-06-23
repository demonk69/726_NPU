# RTL Changes for Multi-Core NPU

## Principles

- Keep `rtl/top/npu_top.v` unchanged for the first multi-core implementation.
- Keep the existing single-core `soc_top` path working.
- Add multi-core modules instead of mutating the single-core bridge/top whenever practical.
- Treat ZCU102 as a resource/platform target, not as a required validation step.
- Keep simulation-only memory models out of the synthesizable board boundary.

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `rtl/top/npu_top.v` | No change | Known-good NPU core instance |
| `rtl/top/npu_mc_top.v` | New | Replicate `NUM_CORES` `npu_top` instances |
| `rtl/soc/axi_lite_mc_bridge.v` | New | PicoRV32 simple bus to selected core AXI-Lite |
| `rtl/soc/dram_multi_port.v` | New, simulation only | Shared backing memory for CPU and NPU DMA ports |
| `rtl/soc/soc_mc_top.v` | New, simulation top | PicoRV32 plus multi-NPU simulation system |
| `rtl/top/pico_npu_mc_top.v` | New, synthesis boundary | PicoRV32 plus multi-NPU RTL boundary for carrier integration |
| `tools/pth/gen_vgg_closed_loop.py` | Modify | Emit multi-core firmware and layout constants |

## `npu_mc_top.v`

Replicates complete NPU cores:

```text
npu_mc_top(NUM_CORES)
├── npu_top[0]
├── npu_top[1]
└── ...
```

Use flattened Verilog buses for portability:

```verilog
input  wire [NUM_CORES*32-1:0] s_axi_awaddr;
input  wire [NUM_CORES-1:0]    s_axi_awvalid;
output wire [NUM_CORES-1:0]    s_axi_awready;
...
output wire [NUM_CORES*32-1:0] m_axi_awaddr;
output wire [NUM_CORES-1:0]    m_axi_awvalid;
input  wire [NUM_CORES-1:0]    m_axi_awready;
...
output wire [NUM_CORES-1:0]    npu_irq;
```

Each generate instance slices its own AXI-Lite and AXI master signals.

## `axi_lite_mc_bridge.v`

Do not replace the existing single-core `axi_lite_bridge.v` in the first pass.
Create a multi-core bridge with explicit decode:

```text
offset = iomem_addr - NPU_BASE
valid  = iomem_valid && offset < NUM_CORES * 0x100
core   = offset[11:8]
local  = offset[7:0]
```

Only the selected core receives AW/W or AR valid. The bridge returns `DEADBEEF`
or raises a simple firmware-visible failure behavior for invalid core windows.
The first version can keep one outstanding PicoRV32 transaction, matching the
current simple bus behavior.

## `dram_multi_port.v` For Simulation

The simulation memory model should provide one CPU simple port and `NUM_CORES`
NPU AXI ports over one shared backing store.

Minimum behavior:

- CPU reads/writes hit the same memory array used by all NPU ports.
- Multiple reads can be accepted in a simulation-friendly way.
- Writes are serialized if multiple NPU ports write in the same cycle.
- This module is not the board memory implementation.

## `soc_mc_top.v` For Simulation

Simulation integration:

```text
PicoRV32
├── SRAM for firmware
├── shared DRAM model for runtime data
└── axi_lite_mc_bridge -> npu_mc_top registers

npu_mc_top AXI masters -> dram_multi_port NPU ports
```

`NUM_CORES=1` must behave like the current single-core flow before enabling
multi-core firmware.

## `pico_npu_mc_top.v` For Carrier Integration

Create a synthesis-oriented top that contains:

- PicoRV32.
- Firmware SRAM/BRAM interface.
- Multi-core NPU wrapper.
- Core MMIO bridge.
- External/shared memory interface ports.

This top must not instantiate `dram_model`. The actual ZCU102 memory attachment
can be handled outside this plan by the board integration flow.

## IRQ Policy

The first firmware polls STATUS. RTL may expose `npu_irq[NUM_CORES-1:0]`, but
the CPU IRQ vector should be built with one combinational assignment, not by
driving the same vector from multiple continuous assignments.

## Resource Knobs

| Knob | First value | Reason |
|------|-------------|--------|
| `NUM_CORES` | 2 | First multi-core target |
| `FP16_ENABLE` | 0 | Avoid duplicated FP16 datapath cost |
| `PERF_ENABLE_DERIVED` | 0 | Keep counter math out of first build |
| `INT8_SIMD_LANES` | 4 | Match current maintained closed-loop default |
| `PPB_DEPTH` | Current working value | Reduce only after K-split regressions pass |

Before scaling to 4 cores, inspect whether ping-pong buffers infer BRAM or an
acceptable memory primitive. If they infer registers/muxes, fix that first.
