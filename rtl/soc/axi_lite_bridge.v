// =============================================================================
// Module  : axi_lite_bridge
// Project : NPU_prj
// Desc    : Bridge between PicoRV32 iomem interface and AXI4-Lite protocol.
//
//           PicoRV32 iomem:
//             - Simple valid/ready handshake
//             - Single addr + wdata/rdata per transaction
//             - wstrb[3:0] for byte enables
//             - Write when wstrb != 0, Read when wstrb == 0
//
//           AXI4-Lite:
//             - Separate AW/AR address channels
//             - W data channel (for writes)
//             - B response channel (for writes)
//             - R response channel (for reads)
//
//           This bridge converts iomem → AXI4-Lite transparently.
//           NPU register base address is stripped before passing to NPU.
// =============================================================================

`timescale 1ns/1ps

module axi_lite_bridge (
    input  wire        clk,
    input  wire        rst_n,

    // ---- PicoRV32 iomem side ----
    input  wire        iomem_valid,
    output wire        iomem_ready,
    input  wire [3:0]  iomem_wstrb,
    input  wire [31:0] iomem_addr,
    input  wire [31:0] iomem_wdata,
    output reg  [31:0] iomem_rdata,

    // ---- AXI4-Lite master (towards NPU register slave) ----
    // AW
    output wire [31:0] m_axi_awaddr,
    output wire        m_axi_awvalid,
    input  wire        m_axi_awready,
    // W
    output wire [31:0] m_axi_wdata,
    output wire [3:0]  m_axi_wstrb,
    output wire        m_axi_wvalid,
    input  wire        m_axi_wready,
    // B
    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output wire        m_axi_bready,
    // AR
    output wire [31:0] m_axi_araddr,
    output wire        m_axi_arvalid,
    input  wire        m_axi_arready,
    // R
    input  wire [31:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rvalid,
    output wire        m_axi_rready,

    // Configuration
    input  wire [31:0] npu_base_addr    // NPU register base (e.g., 0x02000000)
);

// ---------------------------------------------------------------------------
// Address offset: strip the base address
// ---------------------------------------------------------------------------
wire [31:0] npu_offset = iomem_addr - npu_base_addr;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------
localparam S_IDLE   = 2'd0;
localparam S_WRITE  = 2'd1;
localparam S_READ   = 2'd2;

reg [1:0] state;
wire is_write = iomem_valid && (|iomem_wstrb);
wire is_read  = iomem_valid && (iomem_wstrb == 4'b0);

// ---------------------------------------------------------------------------
// Output assignments
// ---------------------------------------------------------------------------
assign iomem_ready = (state == S_IDLE) ? 1'b1 :
                     (state == S_WRITE) ? (m_axi_awready && m_axi_wready) :  // simplified
                     (state == S_READ)  ? (m_axi_rvalid && m_axi_rready) : 1'b0;

// AXI address: strip base, keep offset
assign m_axi_awaddr = npu_offset;
assign m_axi_araddr = npu_offset;
assign m_axi_wdata  = iomem_wdata;
assign m_axi_wstrb  = iomem_wstrb;

// AXI valid signals
assign m_axi_awvalid = (state == S_WRITE);
assign m_axi_wvalid  = (state == S_WRITE);
assign m_axi_bready  = 1'b1;   // always accept B response
assign m_axi_arvalid = (state == S_READ);
assign m_axi_rready  = (state == S_READ);

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        state      <= S_IDLE;
        iomem_rdata <= 32'h0;
    end else begin
        case (state)
            S_IDLE: begin
                if (iomem_valid) begin
                    if (|iomem_wstrb)
                        state <= S_WRITE;
                    else
                        state <= S_READ;
                end
            end

            S_WRITE: begin
                // AW + W complete in same cycle (axi_lite ready is immediate in sim)
                if (m_axi_awready && m_axi_wready) begin
                    state <= S_IDLE;
                end
            end

            S_READ: begin
                if (m_axi_rvalid) begin
                    iomem_rdata <= m_axi_rdata;
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
