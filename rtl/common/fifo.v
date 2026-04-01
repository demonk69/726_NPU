// =============================================================================
// Module  : sync_fifo
// Project : NPU_prj
// Desc    : Parameterised single-clock synchronous FIFO.
//           Uses ring-buffer with head/tail pointers.
//           Flags: full, empty, almost_full, almost_empty.
// =============================================================================

`timescale 1ns/1ps

module sync_fifo #(
    parameter DATA_W       = 16,            // data bit-width
    parameter DEPTH        = 16,            // FIFO depth (must be power of 2)
    parameter ALMOST_FULL  = 2,             // almost_full  threshold (remaining slots)
    parameter ALMOST_EMPTY = 2              // almost_empty threshold (remaining entries)
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // write port
    input  wire                  wr_en,
    input  wire [DATA_W-1:0]     wr_data,
    output wire                  full,
    output wire                  almost_full,
    // read port
    input  wire                  rd_en,
    output wire [DATA_W-1:0]     rd_data,
    output wire                  empty,
    output wire                  almost_empty,
    // debug
    output wire [$clog2(DEPTH):0] fill_count
);

// ---------------------------------------------------------------------------
// Local parameters
// ---------------------------------------------------------------------------
localparam ADDR_W = $clog2(DEPTH);

// ---------------------------------------------------------------------------
// Pointer & storage
// ---------------------------------------------------------------------------
reg [ADDR_W:0] head_ptr, tail_ptr;  // extra MSB for full/empty distinguish
reg [DATA_W-1:0] mem [0:DEPTH-1];

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------
wire do_write = wr_en && !full;
always @(posedge clk) begin
    if (!rst_n)
        head_ptr <= 0;
    else if (do_write)
        head_ptr <= head_ptr + 1'b1;
end

always @(posedge clk) begin
    if (do_write)
        mem[head_ptr[ADDR_W-1:0]] <= wr_data;
end

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------
wire do_read = rd_en && !empty;
assign rd_data = mem[tail_ptr[ADDR_W-1:0]];

always @(posedge clk) begin
    if (!rst_n)
        tail_ptr <= 0;
    else if (do_read)
        tail_ptr <= tail_ptr + 1'b1;
end

// ---------------------------------------------------------------------------
// Status flags
// ---------------------------------------------------------------------------
wire ptr_match     = (head_ptr == tail_ptr);
wire ptr_msb_ne    = (head_ptr[ADDR_W] != tail_ptr[ADDR_W]);

assign empty        = ptr_match && !ptr_msb_ne;
assign full         = ptr_match && ptr_msb_ne;

// fill count: unsigned difference head - tail
wire [ADDR_W:0] fill;
assign fill = head_ptr - tail_ptr;
assign fill_count = fill;

assign almost_full  = (fill >= (DEPTH - ALMOST_FULL));
assign almost_empty = (fill <= ALMOST_EMPTY);

endmodule
