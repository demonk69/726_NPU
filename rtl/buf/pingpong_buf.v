// =============================================================================
// Module  : pingpong_buf
// Project : NPU_prj
// Desc    : Dual-buffer Ping-Pong buffer with sub-word read support.
//           - Two independent SRAM banks (BufA, BufB).
//           - Writer (DMA) writes DATA_W words.
//           - Reader (PE) reads OUT_WIDTH sub-words (OUT_WIDTH <= DATA_W).
//           - Each DATA_W word is read SUBW sub-words (DATA_W/OUT_WIDTH).
//           - "swap" toggles which bank each side accesses.
//           - "clear" resets all pointers and fill counts.
//
// Parameters:
//   DATA_W     - write data width (DMA side, e.g. 32)
//   DEPTH      - entries per bank (word count, must be power of 2)
//   OUT_WIDTH  - read data width (PE side, e.g. 16); DATA_W/OUT_WIDTH must be int
//   THRESHOLD  - how many words DMA must fill before buf_ready asserts
// =============================================================================

`timescale 1ns/1ps

module pingpong_buf #(
    parameter DATA_W    = 32,
    parameter DEPTH     = 64,
    parameter OUT_WIDTH = 16,       // PE data width (output)
    parameter THRESHOLD = 16,
    parameter SUBW      = 4          // sub-words per word (4 for INT8 in 32-bit)
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // ---- Write port (DMA side) ----
    input  wire                      wr_en,
    input  wire [DATA_W-1:0]         wr_data,

    // ---- Read port (PE side) ----
    input  wire                      rd_en,
    output wire [OUT_WIDTH-1:0]      rd_data,

    // ---- Control ----
    input  wire                      swap,       // toggle active bank
    input  wire                      clear,      // reset read/write pointers

    // ---- Status ----
    output wire                      buf_empty,  // no more sub-words to read
    output wire                      buf_full,   // writer's bank full
    output wire                      buf_ready,  // writer's bank has >= THRESHOLD words
    output wire [$clog2(DEPTH*DATA_W/OUT_WIDTH):0] rd_fill,  // sub-words remaining
    output wire [$clog2(DEPTH):0]    wr_fill     // words written
);

// ---------------------------------------------------------------------------
// Local parameters
// ---------------------------------------------------------------------------
localparam ADDR_W    = $clog2(DEPTH);
localparam FILL_W    = ADDR_W + 1;
localparam SUBW_W    = $clog2(SUBW);
localparam RD_FILL_W = $clog2(DEPTH * SUBW) + 1;

// ---------------------------------------------------------------------------
// Bank select: 0 = BufA, 1 = BufB
// ---------------------------------------------------------------------------
reg wr_sel;   // which bank DMA writes to
reg rd_sel;   // which bank PE reads from

wire next_wr_sel = ~wr_sel;

// ---------------------------------------------------------------------------
// Write pointer & fill count (per bank)
// ---------------------------------------------------------------------------
reg [ADDR_W-1:0] wr_ptr_a, wr_ptr_b;
reg [FILL_W-1:0] wr_fill_a, wr_fill_b;

wire [ADDR_W-1:0] wr_ptr = wr_sel ? wr_ptr_b : wr_ptr_a;
wire [FILL_W-1:0] cur_wr_fill = wr_sel ? wr_fill_b : wr_fill_a;

// ---------------------------------------------------------------------------
// Read pointer & sub-word counter (per bank)
// rd_ptr: word-level pointer
// rd_sub: sub-word index within current word (0 to SUBW-1)
// rd_fill: total sub-words remaining to read
// ---------------------------------------------------------------------------
reg [ADDR_W-1:0] rd_ptr_a, rd_ptr_b;
reg [SUBW_W-1:0] rd_sub_a, rd_sub_b;  // sub-word index
reg [RD_FILL_W-1:0] rd_fill_a, rd_fill_b;

wire [ADDR_W-1:0] rd_ptr = rd_sel ? rd_ptr_b : rd_ptr_a;
wire [SUBW_W-1:0] rd_sub = rd_sel ? rd_sub_b : rd_sub_a;
wire [RD_FILL_W-1:0] cur_rd_fill = rd_sel ? rd_fill_b : rd_fill_a;

// ---------------------------------------------------------------------------
// Bank storage
// ---------------------------------------------------------------------------
reg [DATA_W-1:0] mem_a [0:DEPTH-1];
reg [DATA_W-1:0] mem_b [0:DEPTH-1];

// ---------------------------------------------------------------------------
// Write logic
// ---------------------------------------------------------------------------
wire do_write = wr_en && !buf_full;

always @(posedge clk) begin
    if (do_write) begin
        if (wr_sel == 1'b0)
            mem_a[wr_ptr] <= wr_data;
        else
            mem_b[wr_ptr] <= wr_data;
    end
end

// Update write pointers & fill counts
always @(posedge clk) begin
    if (!rst_n || clear) begin
        wr_ptr_a   <= 0;
        wr_ptr_b   <= 0;
        wr_fill_a  <= 0;
        wr_fill_b  <= 0;
    end else if (swap) begin
        // Reset the NEW writer's bank (it was just drained by PE)
        // New writer's bank = next_wr_sel (after swap)
        if (next_wr_sel == 1'b0) begin
            wr_ptr_a  <= 0;
        end else begin
            wr_ptr_b  <= 0;
        end
    end else if (do_write) begin
        if (wr_sel == 1'b0) begin
            wr_ptr_a  <= wr_ptr_a + 1'b1;
            wr_fill_a <= wr_fill_a + 1'b1;
        end else begin
            wr_ptr_b  <= wr_ptr_b + 1'b1;
            wr_fill_b <= wr_fill_b + 1'b1;
        end
    end
end

// ---------------------------------------------------------------------------
// Read logic (sub-word)
// ---------------------------------------------------------------------------
wire do_read = rd_en && !buf_empty;

// Output: select sub-word from current word, sign-extend INT8 to DATA_W
wire [DATA_W-1:0] rd_mem = (rd_sel == 1'b0)
    ? mem_a[rd_ptr]
    : mem_b[rd_ptr];

wire [7:0] rd_byte = (rd_sub == 0) ? rd_mem[ 7:0] :
                     (rd_sub == 1) ? rd_mem[15:8] :
                     (rd_sub == 2) ? rd_mem[23:16] :
                                     rd_mem[31:24];

// Sign-extend INT8 to OUT_WIDTH
wire [OUT_WIDTH-1:0] rd_data_raw = {{(OUT_WIDTH-8){rd_byte[7]}}, rd_byte};

// When buffer is empty, output 0 to avoid PE latching stale data
assign rd_data = buf_empty ? {OUT_WIDTH{1'b0}} : rd_data_raw;

// Update read pointers & fill counts
always @(posedge clk) begin
    if (!rst_n || clear) begin
        rd_ptr_a   <= 0;  rd_sub_a <= 0;  rd_fill_a <= 0;
        rd_ptr_b   <= 0;  rd_sub_b <= 0;  rd_fill_b <= 0;
        if (clear) ; // silently clear
    end else if (swap) begin
        // New reader's bank = old writer's bank = wr_sel (before swap)
        // Copy the write fill count as the read fill count (× SUBW)
        if (wr_sel == 1'b0) begin
            rd_ptr_a  <= 0;
            rd_sub_a  <= 0;
            rd_fill_a <= wr_fill_a * SUBW;
        end else begin
            rd_ptr_b  <= 0;
            rd_sub_b  <= 0;
            rd_fill_b <= wr_fill_b * SUBW;
        end
    end else if (do_read) begin
        if (rd_sel == 1'b0) begin
            if (rd_sub_a == (SUBW-1)) begin
                rd_ptr_a  <= rd_ptr_a + 1'b1;
                rd_sub_a  <= 0;
            end else begin
                rd_sub_a  <= rd_sub_a + 1'b1;
            end
            rd_fill_a <= rd_fill_a - 1'b1;
        end else begin
            if (rd_sub_b == (SUBW-1)) begin
                rd_ptr_b  <= rd_ptr_b + 1'b1;
                rd_sub_b  <= 0;
            end else begin
                rd_sub_b  <= rd_sub_b + 1'b1;
            end
            rd_fill_b <= rd_fill_b - 1'b1;
        end
    end
end

// ---------------------------------------------------------------------------
// Swap logic
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n)
        wr_sel <= 0;
    else if (clear)
        wr_sel <= 0;
    else if (swap)
        wr_sel <= next_wr_sel;
end

always @(posedge clk) begin
    if (!rst_n)
        rd_sel <= 0;
    else if (clear)
        rd_sel <= 0;
    else if (swap)
        rd_sel <= wr_sel;  // rd_sel gets the OLD wr_sel
end

// ---------------------------------------------------------------------------
// Status flags
// ---------------------------------------------------------------------------
assign buf_full  = (cur_wr_fill >= DEPTH[FILL_W-1:0]);
assign buf_empty = (cur_rd_fill == 0);
assign buf_ready = (cur_wr_fill >= THRESHOLD[FILL_W-1:0]);
assign rd_fill   = cur_rd_fill;
assign wr_fill   = cur_wr_fill;

endmodule
