// =============================================================================
// Module  : psum_out_buf
// Project : NPU_prj
// Desc    : Tile-local PSUM/OUT buffer for K-split accumulation.
//           - Stores one accumulator tile per bank.
//           - Default shape is 4x4, 16 words, 32-bit per word.
//           - Two synchronous read/write ports:
//               port A: DMA/load/drain side
//               port B: compute/serializer side
//           - Arithmetic is intentionally outside this module. K-split uses a
//             read-modify-write sequence so INT32 and FP32 accumulator formats
//             can share the same storage.
//           - valid_mask filters edge-tile lanes. Invalid lanes read as zero and
//             writes to invalid lanes are ignored.
// =============================================================================

`timescale 1ns/1ps

module psum_out_buf #(
    parameter ACC_W      = 32,
    parameter TILE_M     = 4,
    parameter TILE_N     = 4,
    parameter BANKS      = 2,
    parameter TILE_WORDS = TILE_M * TILE_N,
    parameter IDX_W      = (TILE_WORDS <= 1) ? 1 : $clog2(TILE_WORDS),
    parameter BANK_W     = (BANKS <= 1) ? 1 : $clog2(BANKS),
    parameter TOTAL_WORDS = BANKS * TILE_WORDS,
    parameter ADDR_W     = (TOTAL_WORDS <= 1) ? 1 : $clog2(TOTAL_WORDS)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    input  wire                         clear,
    input  wire [TILE_WORDS-1:0]        valid_mask,

    input  wire                         tile_clear_en,
    input  wire [BANK_W-1:0]            tile_clear_bank,
    output reg                          tile_clear_done,

    // ---- Port A: DMA/load/drain side ----
    input  wire                         port_a_en,
    input  wire                         port_a_we,
    input  wire [BANK_W-1:0]            port_a_bank,
    input  wire [IDX_W-1:0]             port_a_idx,
    input  wire [ACC_W-1:0]             port_a_wdata,
    output reg  [ACC_W-1:0]             port_a_rdata,
    output reg                          port_a_rvalid,

    // ---- Port B: compute/serializer side ----
    input  wire                         port_b_en,
    input  wire                         port_b_we,
    input  wire [BANK_W-1:0]            port_b_bank,
    input  wire [IDX_W-1:0]             port_b_idx,
    input  wire [ACC_W-1:0]             port_b_wdata,
    output reg  [ACC_W-1:0]             port_b_rdata,
    output reg                          port_b_rvalid,

    // Asserted when both ports write the same valid word in one cycle.
    // The write is deterministic: port B has priority.
    output reg                          write_conflict
);

localparam [ADDR_W-1:0] TILE_WORDS_ADDR = TILE_WORDS[ADDR_W-1:0];

reg [ACC_W-1:0] mem [0:TOTAL_WORDS-1];

wire [ADDR_W-1:0] port_a_addr =
    ({ADDR_W{1'b0}} + port_a_idx) + (port_a_bank * TILE_WORDS_ADDR);
wire [ADDR_W-1:0] port_b_addr =
    ({ADDR_W{1'b0}} + port_b_idx) + (port_b_bank * TILE_WORDS_ADDR);
wire [ADDR_W-1:0] clear_base =
    tile_clear_bank * TILE_WORDS_ADDR;

wire port_a_lane_valid = valid_mask[port_a_idx];
wire port_b_lane_valid = valid_mask[port_b_idx];

integer i;
integer clear_i;

always @(posedge clk) begin
    if (!rst_n || clear) begin
        port_a_rdata     <= {ACC_W{1'b0}};
        port_b_rdata     <= {ACC_W{1'b0}};
        port_a_rvalid    <= 1'b0;
        port_b_rvalid    <= 1'b0;
        tile_clear_done  <= 1'b0;
        write_conflict   <= 1'b0;
        for (i = 0; i < TOTAL_WORDS; i = i + 1)
            mem[i] <= {ACC_W{1'b0}};
    end else begin
        port_a_rvalid    <= 1'b0;
        port_b_rvalid    <= 1'b0;
        tile_clear_done  <= tile_clear_en;
        write_conflict   <= port_a_en && port_a_we && port_a_lane_valid &&
                            port_b_en && port_b_we && port_b_lane_valid &&
                            (port_a_addr == port_b_addr);

        if (tile_clear_en) begin
            for (clear_i = 0; clear_i < TILE_WORDS; clear_i = clear_i + 1)
                mem[clear_base + clear_i[ADDR_W-1:0]] <= {ACC_W{1'b0}};
        end

        if (port_a_en && !port_a_we) begin
            port_a_rdata  <= port_a_lane_valid ? mem[port_a_addr] : {ACC_W{1'b0}};
            port_a_rvalid <= port_a_lane_valid;
        end

        if (port_b_en && !port_b_we) begin
            port_b_rdata  <= port_b_lane_valid ? mem[port_b_addr] : {ACC_W{1'b0}};
            port_b_rvalid <= port_b_lane_valid;
        end

        if (port_a_en && port_a_we && port_a_lane_valid)
            mem[port_a_addr] <= port_a_wdata;

        if (port_b_en && port_b_we && port_b_lane_valid)
            mem[port_b_addr] <= port_b_wdata;
    end
end

endmodule
