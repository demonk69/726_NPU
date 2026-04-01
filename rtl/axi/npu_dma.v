// =============================================================================
// Module  : npu_dma
// Project : NPU_prj
// Desc    : Simplified AXI4 Master DMA with INCR burst support.
//           3 independent channels: Weight, Activation, Result.
// =============================================================================

`timescale 1ns/1ps

module npu_dma #(
    parameter DATA_W    = 32,
    parameter BURST_MAX = 16,
    parameter FIFO_DEPTH = 64
)(
    input  wire        clk,
    input  wire        rst_n,
    // Channel 0: Weight (Read from DRAM)
    input  wire        w_start,
    input  wire [31:0] w_base_addr,
    input  wire [15:0] w_len_bytes,
    output reg         w_done,
    output wire        w_fifo_empty,
    input  wire        w_fifo_rd_en,
    output wire [DATA_W-1:0] w_fifo_dout,
    // Channel 1: Activation (Read from DRAM)
    input  wire        a_start,
    input  wire [31:0] a_base_addr,
    input  wire [15:0] a_len_bytes,
    output reg         a_done,
    output wire        a_fifo_empty,
    input  wire        a_fifo_rd_en,
    output wire [DATA_W-1:0] a_fifo_dout,
    // Channel 2: Result (Write to DRAM)
    input  wire        r_start,
    input  wire [31:0] r_base_addr,
    input  wire [15:0] r_len_bytes,
    output reg         r_done,
    output wire        r_fifo_full,
    input  wire        r_fifo_wr_en,
    input  wire [DATA_W-1:0] r_fifo_din,
    // AXI4 Master (shared bus, time-division)
    output wire [31:0] m_axi_awaddr,
    output wire [7:0]  m_axi_awlen,
    output wire [2:0]  m_axi_awsize,
    output wire [1:0]  m_axi_awburst,
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,
    output reg  [DATA_W-1:0] m_axi_wdata,
    output reg  [DATA_W/8-1:0] m_axi_wstrb,
    output reg         m_axi_wlast,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,
    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output reg         m_axi_bready,
    output wire [31:0] m_axi_araddr,
    output wire [7:0]  m_axi_arlen,
    output wire [2:0]  m_axi_arsize,
    output wire [1:0]  m_axi_arburst,
    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,
    input  wire [DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rvalid,
    output reg         m_axi_rready,
    input  wire        m_axi_rlast
);

// ---------------------------------------------------------------------------
// Internal FIFOs
// ---------------------------------------------------------------------------
wire w_fifo_afull, a_fifo_afull, r_fifo_aempty;
wire [$clog2(FIFO_DEPTH):0] w_fill, a_fill, r_fill;
wire w_fifo_wr = (dma_state == W_READ) && m_axi_rvalid;
wire a_fifo_wr = (dma_state == A_READ) && m_axi_rvalid;
wire r_fifo_rd_wire = (dma_state == R_WRITE) && m_axi_wvalid && m_axi_wready;

sync_fifo #(.DATA_W(DATA_W), .DEPTH(FIFO_DEPTH), .ALMOST_FULL(4), .ALMOST_EMPTY(2))
    u_w_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(w_fifo_wr), .wr_data(m_axi_rdata),
        .full(w_fifo_full_n), .almost_full(w_fifo_afull),
        .rd_en(w_fifo_rd_en), .rd_data(w_fifo_dout),
        .empty(w_fifo_empty), .almost_empty(), .fill_count(w_fill)
    );

sync_fifo #(.DATA_W(DATA_W), .DEPTH(FIFO_DEPTH), .ALMOST_FULL(4), .ALMOST_EMPTY(2))
    u_a_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(a_fifo_wr), .wr_data(m_axi_rdata),
        .full(a_fifo_full_n), .almost_full(a_fifo_afull),
        .rd_en(a_fifo_rd_en), .rd_data(a_fifo_dout),
        .empty(a_fifo_empty), .almost_empty(), .fill_count(a_fill)
    );

wire [DATA_W-1:0] r_fifo_dout_int;
sync_fifo #(.DATA_W(DATA_W), .DEPTH(FIFO_DEPTH), .ALMOST_FULL(4), .ALMOST_EMPTY(2))
    u_r_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(r_fifo_wr_en), .wr_data(r_fifo_din),
        .full(r_fifo_full), .almost_full(),
        .rd_en(r_fifo_rd_wire), .rd_data(r_fifo_dout_int),
        .empty(r_fifo_empty_n), .almost_empty(r_fifo_aempty), .fill_count(r_fill)
    );

// ---------------------------------------------------------------------------
// FIFO full/empty wires needed by FSM
// ---------------------------------------------------------------------------
wire w_fifo_full_n, a_fifo_full_n, r_fifo_empty_n;

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------
localparam IDLE    = 3'd0;
localparam W_READ  = 3'd1;
localparam A_READ  = 3'd2;
localparam R_WRITE = 3'd3;

reg [2:0] dma_state;
reg [31:0] addr_cnt;
reg [15:0] byte_cnt;
wire [7:0] burst_len = (BURST_MAX > 1) ? (BURST_MAX - 1) : 0;

// ---------------------------------------------------------------------------
// Main FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        dma_state   <= IDLE;
        addr_cnt    <= 0;
        byte_cnt    <= 0;
        w_done      <= 0;
        a_done      <= 0;
        r_done      <= 0;
        m_axi_awvalid <= 0;
        m_axi_wvalid  <= 0;
        m_axi_wlast   <= 0;
        m_axi_bready  <= 0;
        m_axi_arvalid <= 0;
        m_axi_rready  <= 0;
    end else begin
        case (dma_state)
            IDLE: begin
                w_done <= 0; a_done <= 0; r_done <= 0;
                m_axi_awvalid <= 0; m_axi_wvalid <= 0;
                m_axi_arvalid <= 0; m_axi_rready <= 0;
                if (w_start && !w_fifo_full_n) begin
                    dma_state <= W_READ;
                    addr_cnt  <= w_base_addr;
                    byte_cnt  <= 0;
                end else if (a_start && !a_fifo_full_n) begin
                    dma_state <= A_READ;
                    addr_cnt  <= a_base_addr;
                    byte_cnt  <= 0;
                end else if (r_start && !r_fifo_empty_n) begin
                    dma_state <= R_WRITE;
                    addr_cnt  <= r_base_addr;
                    byte_cnt  <= 0;
                end
            end

            W_READ: begin
                // Issue AR
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;
                if (m_axi_rvalid && m_axi_rready) begin
                    byte_cnt <= byte_cnt + DATA_W/8;
                    if (m_axi_rlast || byte_cnt + DATA_W/8 >= w_len_bytes) begin
                        w_done <= 1;
                        dma_state <= IDLE;
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 0;
                    end
                end
            end

            A_READ: begin
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;
                if (m_axi_rvalid && m_axi_rready) begin
                    byte_cnt <= byte_cnt + DATA_W/8;
                    if (m_axi_rlast || byte_cnt + DATA_W/8 >= a_len_bytes) begin
                        a_done <= 1;
                        dma_state <= IDLE;
                    end
                end
            end

            R_WRITE: begin
                m_axi_awvalid <= 1;
                m_axi_bready  <= 1;
                m_axi_wvalid  <= !r_fifo_empty_n;
                m_axi_wdata   <= r_fifo_dout_int;
                m_axi_wstrb   <= {(DATA_W/8){1'b1}};
                m_axi_wlast   <= (byte_cnt + DATA_W/8 >= r_len_bytes);
                if (m_axi_wvalid && m_axi_wready) begin
                    byte_cnt <= byte_cnt + DATA_W/8;
                    if (byte_cnt + DATA_W/8 >= r_len_bytes) begin
                        r_done <= 1;
                        dma_state <= IDLE;
                    end
                end
            end

            default: dma_state <= IDLE;
        endcase
    end
end

// ---------------------------------------------------------------------------
// AXI address outputs
// ---------------------------------------------------------------------------
assign m_axi_awaddr  = addr_cnt;
assign m_axi_awlen   = burst_len;
assign m_axi_awsize  = $clog2(DATA_W/8);
assign m_axi_awburst = 2'b01;

assign m_axi_araddr  = addr_cnt;
assign m_axi_arlen   = burst_len;
assign m_axi_arsize  = $clog2(DATA_W/8);
assign m_axi_arburst = 2'b01;

endmodule
