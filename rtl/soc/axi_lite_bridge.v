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
reg [31:0] rdata_hold;

wire aw_fire = m_axi_awvalid && m_axi_awready;
wire w_fire  = m_axi_wvalid && m_axi_wready;
wire ar_fire = m_axi_arvalid && m_axi_arready;
wire r_fire  = m_axi_rvalid && m_axi_rready;
wire write_complete = (state == S_WRITE) && (aw_done || aw_fire) && (w_done || w_fire);
wire read_complete  = (state == S_READ_DATA) && r_fire;

// ---------------------------------------------------------------------------
// Output assignments
// ---------------------------------------------------------------------------
assign iomem_ready = write_complete || read_complete;

// AXI address: strip base, keep offset
assign m_axi_awaddr = addr_q;
assign m_axi_araddr = addr_q;
assign m_axi_wdata  = wdata_q;
assign m_axi_wstrb  = wstrb_q;

// AXI valid signals
assign m_axi_awvalid = (state == S_WRITE) && !aw_done;
assign m_axi_wvalid  = (state == S_WRITE) && !w_done;
assign m_axi_bready  = 1'b1;   // always accept B response
assign m_axi_arvalid = (state == S_READ_ADDR);
assign m_axi_rready  = (state == S_READ_DATA);

always @(*) begin
    iomem_rdata = read_complete ? m_axi_rdata : rdata_hold;
end

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        state      <= S_IDLE;
        addr_q     <= 32'h0;
        wdata_q    <= 32'h0;
        wstrb_q    <= 4'h0;
        aw_done    <= 1'b0;
        w_done     <= 1'b0;
        rdata_hold <= 32'h0;
    end else begin
        case (state)
            S_IDLE: begin
                if (iomem_valid) begin
                    addr_q  <= npu_offset;
                    wdata_q <= iomem_wdata;
                    wstrb_q <= iomem_wstrb;
                    aw_done <= 1'b0;
                    w_done  <= 1'b0;
                    if (|iomem_wstrb)
                        state <= S_WRITE;
                    else
                        state <= S_READ_ADDR;
                end
            end

            S_WRITE: begin
                if (aw_fire) aw_done <= 1'b1;
                if (w_fire)  w_done  <= 1'b1;
                if (write_complete) begin
                    state <= S_IDLE;
                end
            end

            S_READ_ADDR: begin
                if (ar_fire)
                    state <= S_READ_DATA;
            end

            S_READ_DATA: begin
                if (read_complete) begin
                    rdata_hold <= m_axi_rdata;
                    state      <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
