// =============================================================================
// Module  : axi_lite_mc_bridge
// Project : NPU_prj
// Desc    : Multi-core AXI-Lite bridge. Converts a single PicoRV32 iomem
//           transaction into an AXI4-Lite transaction directed to one of
//           NUM_CORES independent AXI-Lite master ports.
//
//           Address decode:
//             offset = iomem_addr - npu_base_addr
//             valid  = offset < NUM_CORES * 0x100
//             core   = offset[11:8]
//             local  = offset[7:0]
//
//           Only the selected core receives the transaction. Reads from
//           invalid core windows return 32'hDEADBEEF.
// =============================================================================

`timescale 1ns/1ps

module axi_lite_mc_bridge #(
    parameter NUM_CORES       = 2,
    parameter NPU_CORE_STRIDE = 256  // 0x100 bytes per core
)(
    input  wire        clk,
    input  wire        rst_n,

    // ---- PicoRV32 iomem side ----
    input  wire        iomem_valid,
    output wire        iomem_ready,
    input  wire [3:0]  iomem_wstrb,
    input  wire [31:0] iomem_addr,
    input  wire [31:0] iomem_wdata,
    output wire [31:0] iomem_rdata,

    // ---- AXI4-Lite masters (flattened, one per core) ----
    output wire [NUM_CORES*32-1:0] m_axi_awaddr,
    output wire [NUM_CORES-1:0]    m_axi_awvalid,
    input  wire [NUM_CORES-1:0]    m_axi_awready,

    output wire [NUM_CORES*32-1:0] m_axi_wdata,
    output wire [NUM_CORES*4-1:0]  m_axi_wstrb,
    output wire [NUM_CORES-1:0]    m_axi_wvalid,
    input  wire [NUM_CORES-1:0]    m_axi_wready,

    input  wire [NUM_CORES*2-1:0]  m_axi_bresp,
    input  wire [NUM_CORES-1:0]    m_axi_bvalid,
    output wire [NUM_CORES-1:0]    m_axi_bready,

    output wire [NUM_CORES*32-1:0] m_axi_araddr,
    output wire [NUM_CORES-1:0]    m_axi_arvalid,
    input  wire [NUM_CORES-1:0]    m_axi_arready,

    input  wire [NUM_CORES*32-1:0] m_axi_rdata,
    input  wire [NUM_CORES*2-1:0]  m_axi_rresp,
    input  wire [NUM_CORES-1:0]    m_axi_rvalid,
    output wire [NUM_CORES-1:0]    m_axi_rready,

    // Configuration
    input  wire [31:0] npu_base_addr    // NPU register base (e.g., 0x02000000)
);

// ---------------------------------------------------------------------------
// Address decode
// ---------------------------------------------------------------------------
wire [31:0] npu_offset    = iomem_addr - npu_base_addr;
wire        valid_core    = (npu_offset < (NUM_CORES * NPU_CORE_STRIDE));
wire        addr_is_npu   = iomem_valid && valid_core;
wire        core_sel_ok   = (npu_offset[11:8] < NUM_CORES);

wire        addr_valid    = iomem_valid && (iomem_addr >= npu_base_addr) && valid_core && core_sel_ok;
wire        addr_is_write = iomem_valid && (|iomem_wstrb);

wire [3:0]  core_idx      = valid_core ? npu_offset[11:8] : 4'd0;
wire [31:0] local_offset  = {24'd0, npu_offset[7:0]};

// ---------------------------------------------------------------------------
// State machine (same protocol as single-core bridge)
// ---------------------------------------------------------------------------
localparam S_IDLE      = 2'd0;
localparam S_WRITE     = 2'd1;
localparam S_READ_ADDR = 2'd2;
localparam S_READ_DATA = 2'd3;

reg [1:0]  state;
reg [31:0] addr_q;
reg [31:0] wdata_q;
reg [3:0]  wstrb_q;
reg        aw_done;
reg        w_done;
reg [3:0]  active_core;

// Per-core handshake signals
wire aw_fire_sel = m_axi_awvalid[active_core] && m_axi_awready[active_core];
wire w_fire_sel  = m_axi_wvalid[active_core]  && m_axi_wready[active_core];
wire ar_fire_sel = m_axi_arvalid[active_core] && m_axi_arready[active_core];
wire r_fire_sel  = m_axi_rvalid[active_core]  && m_axi_rready[active_core];

wire write_complete = (state == S_WRITE) && (aw_done || aw_fire_sel) && (w_done || w_fire_sel);
wire read_complete  = (state == S_READ_DATA) && r_fire_sel;

// ---------------------------------------------------------------------------
// Output to PicoRV32
// ---------------------------------------------------------------------------
assign iomem_ready = addr_valid && (write_complete || read_complete);

// Combinational read data (like original single-core bridge)
reg [31:0] rdata_hold;
assign iomem_rdata = (iomem_addr >= npu_base_addr && !valid_core) ? 32'hDEADBEEF
                   : (iomem_addr >= npu_base_addr && !core_sel_ok) ? 32'hDEADBEEF
                   : read_complete ? m_axi_rdata[active_core*32 +: 32]
                   : rdata_hold;

// ---------------------------------------------------------------------------
// Per-core AXI-Lite output
// ---------------------------------------------------------------------------
genvar g;
generate
    for (g = 0; g < NUM_CORES; g = g + 1) begin : gen_axi
        wire        core_active = (g[3:0] == active_core);

        assign m_axi_awaddr[g*32 +: 32] = core_active ? addr_q   : 32'd0;
        assign m_axi_wdata[g*32 +: 32]  = core_active ? wdata_q  : 32'd0;
        assign m_axi_wstrb[g*4 +: 4]    = core_active ? wstrb_q  : 4'd0;
        assign m_axi_araddr[g*32 +: 32] = core_active ? addr_q   : 32'd0;

        assign m_axi_awvalid[g] = core_active && (state == S_WRITE)      && !aw_done;
        assign m_axi_wvalid[g]  = core_active && (state == S_WRITE)      && !w_done;
        assign m_axi_bready[g]  = 1'b1;     // always accept B response
        assign m_axi_arvalid[g] = core_active && (state == S_READ_ADDR);
        assign m_axi_rready[g]  = core_active && (state == S_READ_DATA);
    end
endgenerate

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        state       <= S_IDLE;
        addr_q      <= 32'd0;
        wdata_q     <= 32'd0;
        wstrb_q     <= 4'd0;
        aw_done     <= 1'b0;
        w_done      <= 1'b0;
        active_core <= 4'd0;
        rdata_hold  <= 32'd0;
    end else begin
        case (state)
            S_IDLE: begin
                if (iomem_valid && addr_valid) begin
                    addr_q      <= local_offset;
                    wdata_q     <= iomem_wdata;
                    wstrb_q     <= iomem_wstrb;
                    aw_done     <= 1'b0;
                    w_done      <= 1'b0;
                    active_core <= core_idx;
                    if (addr_is_write)
                        state <= S_WRITE;
                    else
                        state <= S_READ_ADDR;
                end
            end

            S_WRITE: begin
                if (aw_fire_sel) aw_done <= 1'b1;
                if (w_fire_sel)  w_done  <= 1'b1;
                if (write_complete) begin
                    state <= S_IDLE;
                end
            end

            S_READ_ADDR: begin
                if (ar_fire_sel)
                    state <= S_READ_DATA;
            end

            S_READ_DATA: begin
                if (read_complete) begin
                    rdata_hold <= m_axi_rdata[active_core*32 +: 32];
                    state      <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
