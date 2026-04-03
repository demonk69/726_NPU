// =============================================================================
// Module  : npu_dma
// Project : NPU_prj
// Desc    : AXI4 Master DMA with Ping-Pong Buffer integration.
//           3 independent channels: Weight, Activation, Result.
//           Weight and Activation channels use pingpong_buf for overlapping
//           DMA load with PE compute.
//           Result channel uses a standard sync_fifo.
//
//           DMA write port → pingpong_buf → PE read port
//           PE result      → sync_fifo   → DMA read port → DRAM write
// =============================================================================

`timescale 1ns/1ps

module npu_dma #(
    parameter DATA_W    = 32,       // AXI data bus width
    parameter PE_DATA_W = 16,       // PE data width
    parameter BURST_MAX = 16,       // max AXI burst length
    parameter PPB_DEPTH = 32,       // ping-pong buffer depth per bank
    parameter PPB_THRESH= 16,       // early-start threshold for PPBuf
    parameter R_FIFO_DEPTH = 64     // result FIFO depth
)(
    input  wire        clk,
    input  wire        rst_n,
    // ---- Channel 0: Weight (Read from DRAM → Ping-Pong Buffer → PE) ----
    input  wire        w_start,
    input  wire [31:0] w_base_addr,
    input  wire [15:0] w_len_bytes,
    output reg         w_done,
    // PPBuf interface (external, connected in npu_top)
    output wire        w_ppb_wr_en,
    output wire [DATA_W-1:0] w_ppb_wr_data,
    input  wire        w_ppb_full,        // from PPBuf
    input  wire        w_ppb_buf_ready,   // from PPBuf (threshold reached)
    input  wire        w_ppb_buf_empty,   // from PPBuf (reader's bank empty)
    input  wire        w_ppb_drain_done,  // from PPBuf (all data consumed)

    // ---- Channel 1: Activation (Read from DRAM → Ping-Pong Buffer → PE) ----
    input  wire        a_start,
    input  wire [31:0] a_base_addr,
    input  wire [15:0] a_len_bytes,
    output reg         a_done,
    // PPBuf interface (external)
    output wire        a_ppb_wr_en,
    output wire [DATA_W-1:0] a_ppb_wr_data,
    input  wire        a_ppb_full,
    input  wire        a_ppb_buf_ready,
    input  wire        a_ppb_buf_empty,
    input  wire        a_ppb_drain_done,

    // ---- Channel 2: Result (PE → FIFO → Write to DRAM) ----
    input  wire        r_start,
    input  wire [31:0] r_base_addr,
    input  wire [15:0] r_len_bytes,
    output reg         r_done,
    input  wire        r_fifo_wr_en,
    input  wire [DATA_W-1:0] r_fifo_din,
    output wire        r_fifo_full,

    // ---- AXI4 Master (shared bus, time-division multiplexing) ----
    // AW channel (write address)
    output wire [31:0] m_axi_awaddr,
    output wire [7:0]  m_axi_awlen,
    output wire [2:0]  m_axi_awsize,
    output wire [1:0]  m_axi_awburst,
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,
    // W channel (write data)
    output reg  [DATA_W-1:0] m_axi_wdata,
    output reg  [DATA_W/8-1:0] m_axi_wstrb,
    output reg         m_axi_wlast,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,
    // B channel (write response)
    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output reg         m_axi_bready,
    // AR channel (read address)
    output wire [31:0] m_axi_araddr,
    output wire [7:0]  m_axi_arlen,
    output wire [2:0]  m_axi_arsize,
    output wire [1:0]  m_axi_arburst,
    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,
    // R channel (read data)
    input  wire [DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rvalid,
    output reg         m_axi_rready,
    input  wire        m_axi_rlast
);

// ---------------------------------------------------------------------------
// Result channel: internal sync FIFO (PE results → DMA write to DRAM)
// ---------------------------------------------------------------------------
wire [$clog2(R_FIFO_DEPTH):0] r_fill;
wire r_fifo_rd_wire = (dma_state == R_WRITE) && m_axi_wvalid && m_axi_wready;

sync_fifo #(.DATA_W(DATA_W), .DEPTH(R_FIFO_DEPTH), .ALMOST_FULL(4), .ALMOST_EMPTY(2))
    u_r_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(r_fifo_wr_en), .wr_data(r_fifo_din),
        .full(r_fifo_full), .almost_full(),
        .rd_en(r_fifo_rd_wire), .rd_data(r_fifo_dout_int),
        .empty(r_fifo_empty_n), .almost_empty(), .fill_count(r_fill)
    );

wire [DATA_W-1:0] r_fifo_dout_int;
wire r_fifo_empty_n;

// ---------------------------------------------------------------------------
// PPBuf write enables: DMA AXI read data → PPBuf
// ---------------------------------------------------------------------------
assign w_ppb_wr_en   = (dma_state == W_READ || (dma_state == WA_READ && reading_w))
                        && m_axi_rvalid && !w_ppb_full;
assign w_ppb_wr_data = m_axi_rdata;  // Full 32-bit AXI word

assign a_ppb_wr_en   = (dma_state == A_READ || (dma_state == WA_READ && !reading_w))
                        && m_axi_rvalid && !a_ppb_full;
assign a_ppb_wr_data = m_axi_rdata;  // Full 32-bit AXI word

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------
localparam IDLE    = 4'd0;
localparam W_READ  = 4'd1;
localparam A_READ  = 4'd3;
localparam WA_READ = 4'd5;
localparam R_WRITE = 4'd7;

reg [3:0] dma_state;
reg [31:0] addr_cnt;
reg [15:0] byte_cnt;
wire [7:0] burst_len = 8'd0;  // Fixed single-beat burst for simplicity

// Track which channel is currently reading
reg reading_w;  // 1=reading weight, 0=reading activation (during WA_READ)

// Track total bytes transferred per channel
reg [15:0] w_bytes_done;
reg [15:0] a_bytes_done;

// ---------------------------------------------------------------------------
// PPBuf swap/clear: now controlled by npu_ctrl, not DMA.
// These outputs are kept for interface compatibility but driven from ctrl.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main DMA FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        dma_state    <= IDLE;
        addr_cnt     <= 0;
        byte_cnt     <= 0;
        w_done       <= 0;
        a_done       <= 0;
        r_done       <= 0;
        w_bytes_done <= 0;
        a_bytes_done <= 0;
        reading_w    <= 1;
        m_axi_awvalid <= 0;
        m_axi_wvalid  <= 0;
        m_axi_wlast   <= 0;
        m_axi_bready  <= 0;
        m_axi_arvalid <= 0;
        m_axi_rready  <= 0;
    end else begin
        case (dma_state)
            IDLE: begin
                w_bytes_done <= 0; a_bytes_done <= 0;
                m_axi_awvalid <= 0; m_axi_wvalid <= 0;
                m_axi_arvalid <= 0; m_axi_rready <= 0;
                reading_w <= 1;
                // Clear done flags so the next start can trigger correctly
                w_done <= 0;
                a_done <= 0;
                r_done <= 0;

                // Result write-back takes priority
                if (r_start && !r_fifo_empty_n) begin
                    dma_state <= R_WRITE;
                    addr_cnt  <= r_base_addr;
                    byte_cnt  <= 0;
                end
                // Weight-only or simultaneous W+A start
                else if (w_start && a_start) begin
                    // Sequential: read W first, then A
                    w_bytes_done <= 0; a_bytes_done <= 0;
                    dma_state <= W_READ;
                    addr_cnt  <= w_base_addr;
                    byte_cnt  <= 0;
                end
                else if (w_start && !w_ppb_full) begin
                    w_bytes_done <= 0;
                    dma_state <= W_READ;
                    addr_cnt  <= w_base_addr;
                    byte_cnt  <= 0;
                end
                else if (a_start && !a_ppb_full) begin
                    a_bytes_done <= 0;
                    dma_state <= A_READ;
                    addr_cnt  <= a_base_addr;
                    byte_cnt  <= 0;
                end
            end

            // ---- Weight-only read ----
            W_READ: begin
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;
                if (m_axi_rvalid && m_axi_rready) begin
                    byte_cnt     <= byte_cnt + DATA_W/8;
                    w_bytes_done <= w_bytes_done + DATA_W/8;
                    if (w_bytes_done + DATA_W/8 >= w_len_bytes) begin
                        w_done      <= 1;
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 0;
                        // If activation also needs loading, chain to A_READ
                        if (a_len_bytes > 0) begin
                            dma_state <= A_READ;
                            addr_cnt  <= a_base_addr;
                            byte_cnt  <= 0;
                        end else begin
                            dma_state <= IDLE;
                        end
                    end else if (m_axi_rlast) begin
                        // Burst complete but more data needed — issue next AR
                        m_axi_arvalid <= 0; // deassert then re-assert next cycle
                        addr_cnt <= w_base_addr + w_bytes_done + DATA_W/8;
                        byte_cnt <= 0;
                    end
                end
            end

            // ---- Activation-only read ----
            A_READ: begin
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;
                if (m_axi_rvalid && m_axi_rready) begin
                    byte_cnt     <= byte_cnt + DATA_W/8;
                    a_bytes_done <= a_bytes_done + DATA_W/8;
                    if (a_bytes_done + DATA_W/8 >= a_len_bytes) begin
                        a_done      <= 1;
                        dma_state   <= IDLE;
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 0;
                    end else if (m_axi_rlast) begin
                        m_axi_arvalid <= 0;
                        addr_cnt <= a_base_addr + a_bytes_done + DATA_W/8;
                        byte_cnt <= 0;
                    end
                end
            end

            // ---- Interleaved Weight + Activation read ----
            // Strategy: read weight burst, then activation burst, alternate.
            // This simplifies AXI arbitration compared to per-beat interleaving.
            WA_READ: begin
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;

                if (m_axi_rvalid && m_axi_rready) begin
                    byte_cnt <= byte_cnt + DATA_W/8;

                    if (reading_w) begin
                        w_bytes_done <= w_bytes_done + DATA_W/8;
                        // After a burst of weight, switch to activation
                        if (w_bytes_done + DATA_W/8 >= w_len_bytes) begin
                            if (a_bytes_done < a_len_bytes) begin
                                reading_w <= 0;
                                addr_cnt  <= a_base_addr + a_bytes_done;
                                byte_cnt  <= 0;
                                m_axi_arvalid <= 0; // re-issue AR for new address
                            end else begin
                                // Both done
                                w_done <= 1;
                                a_done <= 1;
                                dma_state <= IDLE;
                                m_axi_arvalid <= 0;
                                m_axi_rready  <= 0;
                            end
                        end
                    end else begin
                        a_bytes_done <= a_bytes_done + DATA_W/8;
                        // After a burst of activation, switch to weight
                        if (a_bytes_done + DATA_W/8 >= a_len_bytes) begin
                            if (w_bytes_done < w_len_bytes) begin
                                reading_w <= 1;
                                addr_cnt  <= w_base_addr + w_bytes_done;
                                byte_cnt  <= 0;
                                m_axi_arvalid <= 0;
                            end else begin
                                w_done <= 1;
                                a_done <= 1;
                                dma_state <= IDLE;
                                m_axi_arvalid <= 0;
                                m_axi_rready  <= 0;
                            end
                        end
                    end
                end
            end

            // ---- Result write-back ----
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
                        r_done     <= 1;
                        dma_state  <= IDLE;
                        m_axi_awvalid <= 0;
                        m_axi_wvalid  <= 0;
                        m_axi_bready  <= 0;
                    end
                end
            end

            default: dma_state <= IDLE;
        endcase
    end
end

// ---------------------------------------------------------------------------
// AXI address outputs (combinational)
// ---------------------------------------------------------------------------
assign m_axi_awaddr  = addr_cnt;
assign m_axi_awlen   = burst_len;
assign m_axi_awsize  = $clog2(DATA_W/8);
assign m_axi_awburst = 2'b01;   // INCR

assign m_axi_araddr  = addr_cnt;
assign m_axi_arlen   = burst_len;
assign m_axi_arsize  = $clog2(DATA_W/8);
assign m_axi_arburst = 2'b01;   // INCR

endmodule
