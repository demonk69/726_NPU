// =============================================================================
// Module  : npu_dma
// Project : NPU_prj
// Desc    : AXI4 Master DMA with **Dual-FSM architecture** (Strategy C).
//
//           Two fully independent state machines share one AXI4 master port:
//             1. Load-FSM : Weight+Activation reads (AR/R channels)
//             2. WB-FSM   : Result write-back (AW/W/B channels)
//
//           AXI4 read/write channels are architecturally independent,
//           so both FSMs operate simultaneously with ZERO conflict.
//
//           Data flow:
//             DRAM --[Load-FSM]--> pingpong_buf --> PE --> FIFO --[WB-FSM]--> DRAM
// =============================================================================

`timescale 1ns/1ps

module npu_dma #(
    parameter DATA_W    = 32,
    parameter PE_DATA_W = 16,
    parameter BURST_MAX = 16,
    parameter PPB_DEPTH = 32,
    parameter PPB_THRESH= 16,
    parameter R_FIFO_DEPTH = 64
)(
    input  wire        clk,
    input  wire        rst_n,
    // ---- Channel 0: Weight ----
    input  wire        w_start,
    input  wire [31:0] w_base_addr,
    input  wire [15:0] w_len_bytes,
    output reg         w_done,
    output wire        w_ppb_wr_en,
    output wire [DATA_W-1:0] w_ppb_wr_data,
    input  wire        w_ppb_full,
    input  wire        w_ppb_buf_ready,
    input  wire        w_ppb_buf_empty,
    input  wire        w_ppb_drain_done,

    // ---- Channel 1: Activation ----
    input  wire        a_start,
    input  wire [31:0] a_base_addr,
    input  wire [15:0] a_len_bytes,
    output reg         a_done,
    output wire        a_ppb_wr_en,
    output wire [DATA_W-1:0] a_ppb_wr_data,
    input  wire        a_ppb_full,
    input  wire        a_ppb_buf_ready,
    input  wire        a_ppb_buf_empty,
    input  wire        a_ppb_drain_done,

    // ---- Channel 2: Result ----
    input  wire        r_start,
    input  wire [31:0] r_base_addr,
    input  wire [15:0] r_len_bytes,
    output reg         r_done,
    input  wire        r_fifo_clear,      // synchronous clear for result FIFO
    input  wire        r_fifo_wr_en,
    input  wire [DATA_W-1:0] r_fifo_din,
    output wire        r_fifo_full,

    // ---- AXI4 Master (shared) ----
    // AW channel
    output wire [31:0] m_axi_awaddr,
    output wire [7:0]  m_axi_awlen,
    output wire [2:0]  m_axi_awsize,
    output wire [1:0]  m_axi_awburst,
    output reg         m_axi_awvalid,
    input  wire        m_axi_awready,
    // W channel
    output wire [DATA_W-1:0] m_axi_wdata,
    output reg  [DATA_W/8-1:0] m_axi_wstrb,
    output wire        m_axi_wlast,
    output reg         m_axi_wvalid,
    input  wire        m_axi_wready,
    // B channel
    input  wire [1:0]  m_axi_bresp,
    input  wire        m_axi_bvalid,
    output reg         m_axi_bready,
    // AR channel
    output wire [31:0] m_axi_araddr,
    output wire [7:0]  m_axi_arlen,
    output wire [2:0]  m_axi_arsize,
    output wire [1:0]  m_axi_arburst,
    output reg         m_axi_arvalid,
    input  wire        m_axi_arready,
    // R channel
    input  wire [DATA_W-1:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rvalid,
    output reg         m_axi_rready,
    input  wire        m_axi_rlast
);

// ===========================================================================
// Result sync FIFO
// ===========================================================================
wire [$clog2(R_FIFO_DEPTH):0] r_fill;
wire r_fifo_rd_wire = (wb_state == WB_ACTIVE) && m_axi_wvalid && m_axi_wready;

sync_fifo #(.DATA_W(DATA_W), .DEPTH(R_FIFO_DEPTH), .ALMOST_FULL(4), .ALMOST_EMPTY(2))
    u_r_fifo (
        .clk(clk), .rst_n(rst_n),
        .clear(r_fifo_clear),
        .wr_en(r_fifo_wr_en), .wr_data(r_fifo_din),
        .full(r_fifo_full), .almost_full(),
        .rd_en(r_fifo_rd_wire), .rd_data(r_fifo_dout_int),
        .empty(r_fifo_empty_n), .almost_empty(), .fill_count(r_fill)
    );

wire [DATA_W-1:0] r_fifo_dout_int;
wire r_fifo_empty_n;

// ===========================================================================
// PPBuf write enables
// ===========================================================================
assign w_ppb_wr_en = (load_state == L_WREAD || (load_state == L_WA_READ && load_reading_w))
                      && m_axi_rvalid && !w_ppb_full;
assign w_ppb_wr_data = m_axi_rdata;

assign a_ppb_wr_en = (load_state == L_AREAD || (load_state == L_WA_READ && !load_reading_w))
                      && m_axi_rvalid && !a_ppb_full;
assign a_ppb_wr_data = m_axi_rdata;

// ===========================================================================
// Load-FSM states
// ===========================================================================
localparam [2:0] L_IDLE   = 3'd0;
localparam [2:0] L_WREAD  = 3'd1;
localparam [2:0] L_AREAD  = 3'd2;
localparam [2:0] L_WA_READ= 3'd3;

reg [2:0] load_state;
reg [31:0] load_addr_cnt;
reg [15:0] load_byte_cnt;
reg [15:0] w_bytes_done, a_bytes_done;
reg        load_reading_w;

always @(posedge clk) begin
    if (!rst_n) begin
        load_state <= L_IDLE; load_addr_cnt <= 0; load_byte_cnt <= 0;
        w_done <= 0; a_done <= 0;
        w_bytes_done <= 0; a_bytes_done <= 0;
        load_reading_w <= 1;
        m_axi_arvalid <= 0; m_axi_rready <= 0;
    end else begin
        case (load_state)
            L_IDLE: begin
                w_bytes_done <= 0; a_bytes_done <= 0;
                m_axi_arvalid <= 0; m_axi_rready <= 0;
                load_reading_w <= 1;
                w_done <= 0; a_done <= 0;
                // Result WB has priority — handled by separate WB-FSM
                // Weight or simultaneous W+A start
                if (w_start && a_start) begin
                    load_state <= L_WREAD; load_addr_cnt <= w_base_addr; load_byte_cnt <= 0;
                    w_bytes_done <= 0; a_bytes_done <= 0;
                end else if (w_start && !w_ppb_full) begin
                    load_state <= L_WREAD; load_addr_cnt <= w_base_addr; load_byte_cnt <= 0;
                    w_bytes_done <= 0;
                end else if (a_start && !a_ppb_full) begin
                    load_state <= L_AREAD; load_addr_cnt <= a_base_addr; load_byte_cnt <= 0;
                    a_bytes_done <= 0;
                end
            end

            // Weight-only read — identical behavior to HEAD W_READ
            L_WREAD: begin
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;
                if (m_axi_rvalid && m_axi_rready) begin
                    load_byte_cnt  <= load_byte_cnt + DATA_W/8;
                    w_bytes_done   <= w_bytes_done + DATA_W/8;
                    if (w_bytes_done + DATA_W/8 >= w_len_bytes) begin
                        w_done       <= 1;
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 0;
                        if (a_len_bytes > 0) begin
                            load_state    <= L_AREAD;
                            load_addr_cnt <= a_base_addr;
                            load_byte_cnt <= 0;
                        end else begin
                            load_state <= L_IDLE;
                        end
                    end else if (m_axi_rlast) begin
                        m_axi_arvalid <= 0;
                        load_addr_cnt <= w_base_addr + w_bytes_done + DATA_W/8;
                        load_byte_cnt <= 0;
                    end
                end
            end

            // Activation-only read — identical behavior to HEAD A_READ
            L_AREAD: begin
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;
                if (m_axi_rvalid && m_axi_rready) begin
                    load_byte_cnt  <= load_byte_cnt + DATA_W/8;
                    a_bytes_done   <= a_bytes_done + DATA_W/8;
                    if (a_bytes_done + DATA_W/8 >= a_len_bytes) begin
                        a_done       <= 1;
                        load_state   <= L_IDLE;
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 0;
                    end else if (m_axi_rlast) begin
                        m_axi_arvalid <= 0;
                        load_addr_cnt <= a_base_addr + a_bytes_done + DATA_W/8;
                        load_byte_cnt <= 0;
                    end
                end
            end

            // Interleaved W+A read — identical to HEAD WA_READ
            L_WA_READ: begin
                m_axi_arvalid <= 1;
                m_axi_rready  <= 1;
                if (m_axi_rvalid && m_axi_rready) begin
                    load_byte_cnt <= load_byte_cnt + DATA_W/8;
                    if (load_reading_w) begin
                        w_bytes_done <= w_bytes_done + DATA_W/8;
                        if (w_bytes_done + DATA_W/8 >= w_len_bytes) begin
                            if (a_bytes_done < a_len_bytes) begin
                                load_reading_w <= 0;
                                load_addr_cnt  <= a_base_addr + a_bytes_done;
                                load_byte_cnt  <= 0;
                                m_axi_arvalid <= 0;
                            end else begin
                                w_done <= 1; a_done <= 1; load_state <= L_IDLE;
                                m_axi_arvalid <= 0; m_axi_rready <= 0;
                            end
                        end
                    end else begin
                        a_bytes_done <= a_bytes_done + DATA_W/8;
                        if (a_bytes_done + DATA_W/8 >= a_len_bytes) begin
                            if (w_bytes_done < w_len_bytes) begin
                                load_reading_w <= 1;
                                load_addr_cnt  <= w_base_addr + w_bytes_done;
                                load_byte_cnt  <= 0;
                                m_axi_arvalid <= 0;
                            end else begin
                                w_done <= 1; a_done <= 1; load_state <= L_IDLE;
                                m_axi_arvalid <= 0; m_axi_rready <= 0;
                            end
                        end
                    end
                end
            end

            default: load_state <= L_IDLE;
        endcase
    end
end

// ===========================================================================
// WB-FSM states
// ===========================================================================
localparam [1:0] WB_IDLE   = 2'd0;
localparam [1:0] WB_ACTIVE = 2'd1;

reg [1:0] wb_state;
reg [31:0] wb_addr_cnt;
reg [15:0] wb_byte_cnt;
reg [7:0]  wb_burst_len;
reg        wb_aw_sent;

reg r_pending;
reg [31:0] r_base_latch;
reg [15:0] r_len_latch;
reg r_pending_clr;

always @(posedge clk) begin
    if (!rst_n) begin
        r_pending <= 0; r_base_latch <= 0; r_len_latch <= 0;
    end else begin
        if (r_pending_clr) r_pending <= 0;
        // Allow r_start to (re-)arm at any time when WB-FSM is idle.
        // This also clears stale r_pending from previous run and updates
        // r_base_latch/r_len_latch to the current request's parameters.
        else if (r_start && wb_state == WB_IDLE) begin
            r_pending <= 1; r_base_latch <= r_base_addr; r_len_latch <= r_len_bytes;
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        wb_state <= WB_IDLE; wb_addr_cnt <= 0; wb_byte_cnt <= 0;
        r_done <= 0; wb_burst_len <= 0; wb_aw_sent <= 0;
        r_pending_clr <= 0;
        m_axi_awvalid <= 0; m_axi_wvalid <= 0;
        m_axi_bready <= 0; m_axi_wstrb <= 0;
    end else begin
        r_pending_clr <= 0;
        r_done <= 0;   // default: single-cycle pulse (cleared every cycle)
        case (wb_state)
            WB_IDLE: begin
                m_axi_awvalid <= 0; m_axi_wvalid <= 0;
                m_axi_bready  <= 0; m_axi_wstrb   <= 0; wb_aw_sent <= 0;
                if (r_pending && !r_fifo_empty_n) begin
                    wb_state      <= WB_ACTIVE;
                    wb_addr_cnt   <= r_base_latch; wb_byte_cnt   <= 0;
                    wb_burst_len  <= (r_len_latch >> $clog2(DATA_W/8)) - 1;
                    r_pending_clr <= 1;
                end
            end
            WB_ACTIVE: begin
                m_axi_bready <= 1; m_axi_wstrb <= {(DATA_W/8){1'b1}};
                if (!wb_aw_sent) begin
                    m_axi_awvalid <= 1;
                    if (m_axi_awvalid && m_axi_awready) begin wb_aw_sent <= 1; m_axi_awvalid <= 0; end
                end
                m_axi_wvalid <= !r_fifo_empty_n;
                if (m_axi_wvalid && m_axi_wready) begin
                    wb_byte_cnt <= wb_byte_cnt + DATA_W/8;
                    if (wb_byte_cnt + DATA_W/8 >= r_len_latch) begin
                        r_done <= 1; wb_state <= WB_IDLE;
                        m_axi_wvalid <= 0; m_axi_bready <= 0; wb_aw_sent <= 0;
                    end
                end
            end
            default: wb_state <= WB_IDLE;
        endcase
    end
end

// ===========================================================================
// AXI outputs (combinational)
// ===========================================================================
assign m_axi_araddr  = load_addr_cnt;
assign m_axi_arlen   = 8'd0;  // Fixed single-beat (same as HEAD)
assign m_axi_arsize  = $clog2(DATA_W/8);
assign m_axi_arburst = 2'b01;

assign m_axi_awaddr  = wb_addr_cnt;
assign m_axi_awlen   = wb_burst_len;
assign m_axi_awsize  = $clog2(DATA_W/8);
assign m_axi_awburst = 2'b01;

assign m_axi_wdata   = r_fifo_dout_int;
assign m_axi_wlast    = (wb_state == WB_ACTIVE) && (wb_byte_cnt + DATA_W/8 >= r_len_latch);

// Debug: combined state for testbench compatibility
wire [3:0] dma_state = {wb_state, load_state};

endmodule
