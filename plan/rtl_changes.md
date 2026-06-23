# RTL Changes for Multi-Core NPU

## File Change Summary

| File | Action | Lines (est.) | Description |
|------|--------|-------------|-------------|
| `rtl/top/npu_top.v` | **No change** | 0 | Core black box, used as-is |
| `rtl/top/npu_mc_top.v` | **New** | ~100 | Multi-core NPU wrapper |
| `rtl/soc/dram_multi_port.v` | **New** | ~150 | Multi-port DRAM model |
| `rtl/soc/soc_mc_top.v` | **New** | ~250 | Multi-core SoC top |
| `rtl/soc/axi_lite_bridge.v` | **Modify** | ~20 | Add multi-core address decode |
| `rtl/soc/dram_model.v` | **No change** | 0 | Wrapped by dram_multi_port |
| `rtl/ctrl/npu_ctrl.v` | **No change** | 0 | Unchanged |
| `rtl/axi/npu_dma.v` | **No change** | 0 | Unchanged |
| `rtl/array/reconfig_pe_array.v` | **No change** | 0 | Unchanged |
| `rtl/pe/*.v` | **No change** | 0 | Unchanged |
| `rtl/buf/*.v` | **No change** | 0 | Unchanged |

**Total new RTL: ~500 lines. Modified RTL: ~20 lines.**

---

## Module 1: `rtl/top/npu_mc_top.v`

### Purpose
Wrap `NUM_CORES` identical `npu_top` instances with independent AXI interfaces.

### Parameters

```systemverilog
parameter NUM_CORES        = 2,
parameter PHY_ROWS         = 16,
parameter PHY_COLS         = 16,
parameter DATA_W           = 32,
parameter ACC_W            = 32,
parameter PPB_DEPTH        = 64,
parameter PPB_THRESH       = 16,
parameter INT8_SIMD_LANES  = 4,
parameter PERF_ENABLE_DERIVED = 0,
parameter FP16_ENABLE      = 0,
parameter PPB_SCALAR_READ_ENABLE = 1
```

### Ports

```systemverilog
module npu_mc_top (
    input  wire         sys_clk,
    input  wire         sys_rst_n,

    // AXI4-Lite slaves (one per core) — indexed [NUM_CORES-1:0]
    input  wire [NUM_CORES-1:0][31:0] s_axi_awaddr,
    input  wire [NUM_CORES-1:0]       s_axi_awvalid,
    output wire [NUM_CORES-1:0]       s_axi_awready,
    // ... (all AXI4-Lite channels replicated)

    // AXI4 Masters (one per core) — indexed [NUM_CORES-1:0]
    output wire [NUM_CORES-1:0][31:0] m_axi_awaddr,
    // ... (all AXI4 Master channels replicated)

    // Interrupts (one per core)
    output wire [NUM_CORES-1:0]       npu_irq
);
```

### Implementation

```systemverilog
genvar core_idx;
generate
    for (core_idx = 0; core_idx < NUM_CORES; core_idx = core_idx + 1) begin : gen_cores
        npu_top #(
            .PHY_ROWS(PHY_ROWS),
            .PHY_COLS(PHY_COLS),
            // ... all parameters passed through
        ) u_npu_core (
            .sys_clk       (sys_clk),
            .sys_rst_n     (sys_rst_n),
            .s_axi_awaddr  (s_axi_awaddr[core_idx]),
            .s_axi_awvalid (s_axi_awvalid[core_idx]),
            .s_axi_awready (s_axi_awready[core_idx]),
            // ... all AXI4-Lite ports connected
            .m_axi_awaddr  (m_axi_awaddr[core_idx]),
            // ... all AXI4 Master ports connected
            .npu_irq       (npu_irq[core_idx])
        );
    end
endgenerate
```

---

## Module 2: `rtl/soc/dram_multi_port.v`

### Purpose
Wrap the existing `dram_model` to support `NUM_PORTS` independent AXI4
slave ports (1 CPU simple-port + NUM_CORES NPU AXI4 ports).

### Approach
Instantiate `NUM_PORTS` independent `dram_model` instances, each with a
private memory array that mirrors the full DRAM space.

**Tradeoff**: Memory is duplicated per port, not truly shared. This means:
- CPU writes MUST be broadcast to all NPU port instances so DMA reads see the data
- Each NPU port instance holds its private copy; NPU writes are local

**Alternative (more accurate but complex)**:
Instantiate a single backing store array, and serialize all port accesses
through a round-robin arbiter. This is faithful to real hardware but adds
simulation complexity and a bandwidth bottleneck.

**Chosen approach for phase 1**: Shared backing store with round-robin
arbiter for write conflicts. Read ports operate concurrently.

### Parameters

```systemverilog
parameter WORDS     = 15360,    // DRAM depth in words
parameter DATA_W    = 32,
parameter NUM_PORTS = 3         // 1 CPU + NUM_CORES NPU
```

### Ports

```systemverilog
module dram_multi_port #(...) (
    input  wire         clk,
    input  wire         rst_n,

    // CPU simple port (port 0)
    input  wire         cpu_valid,
    output wire         cpu_ready,
    input  wire         cpu_we,
    input  wire [3:0]   cpu_wstrb,
    input  wire [31:0]  cpu_addr,
    input  wire [31:0]  cpu_wdata,
    output wire [31:0]  cpu_rdata,

    // NPU AXI4 ports [1:NUM_PORTS-1]
    // ... replicated AXI4 slave interface per NPU port
);
```

### Implementation sketch

```
Single reg [DATA_W-1:0] mem [0:WORDS-1];
Round-robin arbiter for writes (CPU + NPU ports).
Concurrent reads from all ports (no arbitration needed in simulation).
```

---

## Module 3: `rtl/soc/soc_mc_top.v`

### Purpose
New SoC top-level for multi-core simulation. Integrates:
- PicoRV32 CPU
- SRAM (unchanged)
- Multi-port DRAM
- Multi-core address decoding AXI-Lite bridge
- `npu_mc_top`

### Parameter

```systemverilog
parameter MEM_WORDS   = 1024,
parameter DRAM_WORDS  = 15360,
parameter NUM_CORES   = 2
```

### Integration sketch

```
PicoRV32
  ├── addr < SRAM_SIZE    → SRAM
  ├── SRAM_SIZE ≤ addr < NPU_BASE → DRAM (CPU port)
  └── addr ≥ NPU_BASE     → AXI-Lite bridge → core-specific MMIO

npu_mc_top
  ├── Core0: AXI4-Lite ← bridge, AXI4 Master → DRAM port1
  ├── Core1: AXI4-Lite ← bridge, AXI4 Master → DRAM port2
  └── IRQ[0..N-1] → CPU IRQ[7..7+N-1]
```

### IRQ connection

```systemverilog
wire [NUM_CORES-1:0] npu_irq_vec;
wire [31:0] cpu_irq;

assign cpu_irq = 32'h0;
genvar i;
generate
    for (i = 0; i < NUM_CORES; i = i + 1) begin : gen_irq
        assign cpu_irq[7 + i] = npu_irq_vec[i];
    end
endgenerate
```

---

## Module 4: `rtl/soc/axi_lite_bridge.v` (Modification)

### Change
Add per-core address decoding.

### Current logic

```systemverilog
wire addr_is_npu = iomem_valid && (iomem_addr >= npu_base_addr);
```

### New logic

```systemverilog
parameter NUM_CORES = 2;
parameter [31:0] NPU_CORE_STRIDE = 32'h100;

wire addr_is_npu = iomem_valid && (iomem_addr >= npu_base_addr);

// Extract core index from addr[11:8], upper bound check
wire [$clog2(NUM_CORES)-1:0] core_sel = iomem_addr[11:8];
wire core_sel_valid = (core_sel < NUM_CORES);

// Route to selected core's AXI-Lite
// Each core sees addr[7:0] (offset within its register file)
```

The output side becomes a `generate` loop producing `NUM_CORES` independent
AXI4-Lite master ports, with only the selected core receiving the transaction.

---

## Verification Impact

### Files NOT changed (no regression risk)

- `rtl/pe/*.v` — PE datapath
- `rtl/array/reconfig_pe_array.v` — PE array
- `rtl/ctrl/npu_ctrl.v` — Controller FSM
- `rtl/axi/npu_dma.v` — DMA engine
- `rtl/buf/pingpong_buf.v` — Ping-pong buffers
- `rtl/power/npu_power.v` — Power/CE module
- `rtl/common/*.v` — Utility modules

### Existing testbenches — still valid

All existing single-core testbenches (`tb_soc.v`, `tb_npu_top.v`, etc.)
remain valid since they instantiate `npu_top` directly, which is unchanged.

### New testbenches needed

| Testbench | Purpose |
|-----------|---------|
| `tb/tb_npu_mc_top.v` | Multi-core NPU basic smoke test (start 2 cores simultaneously) |
| `tb/tb_soc_mc_vgg_closed_loop.v` | Multi-core VGG closed-loop end-to-end test |
