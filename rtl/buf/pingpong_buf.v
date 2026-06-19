// =============================================================================
// Module  : pingpong_buf
// Project : NPU_prj
// Desc    : Dual-buffer Ping-Pong buffer with INT8 sub-word read support.
//           - Two independent SRAM banks (BufA, BufB).
//           - Writer (DMA) writes DATA_W words.
//           - Reader (PE) reads OUT_WIDTH sub-words (OUT_WIDTH <= DATA_W).
//           - Each DATA_W word is read in SUBW INT8 sub-words.
//           - "swap" toggles which bank each side accesses.
//           - "clear" resets all pointers and fill counts.
//
// Parameters:
//   DATA_W     - write data width (DMA side, e.g. 32)
//   DEPTH      - entries per bank (word count, must be power of 2)
//   OUT_WIDTH  - read data width (PE side, e.g. 16); DATA_W/OUT_WIDTH must be int
//   THRESHOLD  - how many words DMA must fill before buf_ready asserts
//   SUBW       - INT8 sub-words per DATA_W word
//   VEC_LANES  - maximum number of PE lanes returned by rd_vec
// =============================================================================

`timescale 1ns/1ps

module pingpong_buf #(
    parameter DATA_W    = 32,
    parameter DEPTH     = 64,
    parameter OUT_WIDTH = 16,       // PE data width (output)
    parameter THRESHOLD = 16,
    parameter SUBW      = 4,         // max sub-words per word (INT8 case)
    parameter VEC_LANES = 4,
    parameter SCALAR_READ_ENABLE = 1
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // ---- Write port (DMA side) ----
    input  wire                      wr_en,
    input  wire [DATA_W-1:0]         wr_data,

    // ---- Read port (PE side) ----
    input  wire                      rd_en,
    output wire [OUT_WIDTH-1:0]      rd_data,
    input  wire                      rd_vec_en,     // consume rd_vec_lanes lanes
    input  wire [4:0]                rd_vec_lanes,  // valid range: 1..VEC_LANES
    output wire [VEC_LANES*OUT_WIDTH-1:0] rd_vec,
    output wire                      rd_vec_valid,

    // ---- Control ----
    input  wire                      swap,       // toggle active bank
    input  wire                      clear,      // reset read/write pointers
    input  wire                      packed_int8,  // 1=packed INT8 pairs {odd,even} for SIMD

    // ---- Status ----
    output wire                      buf_empty,  // no more sub-words to read
    output wire                      buf_full,   // writer's bank full
    output wire                      buf_ready,  // writer's bank has >= THRESHOLD words
    output wire [$clog2(DEPTH*SUBW):0] rd_fill,  // sub-words remaining
    output wire [$clog2(DEPTH):0]    wr_fill     // words written
);

// ---------------------------------------------------------------------------
// Local parameters
// ---------------------------------------------------------------------------
localparam ADDR_W    = $clog2(DEPTH);
localparam FILL_W    = ADDR_W + 1;
localparam SUBW_W    = $clog2(SUBW);             // 2 bits for SUBW=4 index
localparam RD_FILL_W = $clog2(DEPTH * SUBW) + 1; // max fill: DEPTH*4 sub-words
localparam VEC_W      = VEC_LANES * OUT_WIDTH;
localparam [4:0] VEC_LANES_5 = VEC_LANES;
localparam [2:0] PACKED_LANES = (OUT_WIDTH + 7) / 8;  // 2 for 16b, 4 for 32b
localparam [SUBW_W-1:0] SUBW_LAST = {SUBW_W{1'b1}};

wire [4:0] rd_vec_lanes_eff =
    (rd_vec_lanes == 5'd0)      ? VEC_LANES_5 :
    (rd_vec_lanes > VEC_LANES_5) ? VEC_LANES_5 :
                                   rd_vec_lanes;
wire [RD_FILL_W-1:0] rd_vec_lanes_fill = packed_int8 ? (rd_vec_lanes_eff * PACKED_LANES) : rd_vec_lanes_eff;
wire [2:0] vec_words_per_k = (rd_vec_lanes_eff <= 5'd4)  ? 3'd1 :
                             (rd_vec_lanes_eff <= 5'd8)  ? 3'd2 :
                             (rd_vec_lanes_eff <= 5'd12) ? 3'd3 : 3'd4;
wire [5:0] vec_group_words = packed_int8 ? ({3'd0, vec_words_per_k} * {3'd0, PACKED_LANES})
                                         : {3'd0, vec_words_per_k};

function [ADDR_W-1:0] vec_group_idx;
    input [ADDR_W-1:0] ptr;
    input [5:0] group_words;
    begin
        case (group_words)
            6'd1:    vec_group_idx = ptr;
            6'd2:    vec_group_idx = ptr >> 1;
            6'd3:    vec_group_idx = ptr / 3;
            6'd4:    vec_group_idx = ptr >> 2;
            6'd6:    vec_group_idx = ptr / 6;
            6'd8:    vec_group_idx = ptr >> 3;
            6'd12:   vec_group_idx = ptr / 12;
            6'd16:   vec_group_idx = ptr >> 4;
            default: vec_group_idx = ptr;
        endcase
    end
endfunction

function [5:0] vec_word_in_group;
    input [ADDR_W-1:0] ptr;
    input [5:0] group_words;
    begin
        case (group_words)
            6'd1:    vec_word_in_group = 6'd0;
            6'd2:    vec_word_in_group = ptr - ((ptr >> 1) << 1);
            6'd3:    vec_word_in_group = ptr - ((ptr / 3) * 3);
            6'd4:    vec_word_in_group = ptr - ((ptr >> 2) << 2);
            6'd6:    vec_word_in_group = ptr - ((ptr / 6) * 6);
            6'd8:    vec_word_in_group = ptr - ((ptr >> 3) << 3);
            6'd12:   vec_word_in_group = ptr - ((ptr / 12) * 12);
            6'd16:   vec_word_in_group = ptr - ((ptr >> 4) << 4);
            default: vec_word_in_group = 6'd0;
        endcase
    end
endfunction

function [2:0] vec_k_idx;
    input [5:0] word_in_group;
    input [2:0] words_per_k;
    begin
        case (words_per_k)
            3'd1:    vec_k_idx = word_in_group[2:0];
            3'd2:    vec_k_idx = word_in_group[3:1];
            3'd3:    vec_k_idx = word_in_group / 3;
            3'd4:    vec_k_idx = word_in_group[4:2];
            default: vec_k_idx = 3'd0;
        endcase
    end
endfunction

function [2:0] vec_lane_word;
    input [5:0] word_in_group;
    input [2:0] words_per_k;
    begin
        case (words_per_k)
            3'd1:    vec_lane_word = 3'd0;
            3'd2:    vec_lane_word = {2'd0, word_in_group[0]};
            3'd3:    vec_lane_word = word_in_group - ((word_in_group / 3) * 3);
            3'd4:    vec_lane_word = {1'd0, word_in_group[1:0]};
            default: vec_lane_word = 3'd0;
        endcase
    end
endfunction

function [VEC_W-1:0] vec_line_update;
    input [VEC_W-1:0] line_in;
    input [DATA_W-1:0] data_in;
    input [2:0] lane_word;
    input [2:0] k_idx;
    input [4:0] lane_limit;
    input       is_packed;
    integer byte_i;
    integer lane_i_local;
    reg [7:0] byte_val;
    begin
        vec_line_update = line_in;
        for (byte_i = 0; byte_i < SUBW; byte_i = byte_i + 1) begin
            lane_i_local = lane_word * SUBW + byte_i;
            byte_val = data_in[byte_i*8 +: 8];
            if (lane_i_local < VEC_LANES && lane_i_local < lane_limit) begin
                if (is_packed)
                    vec_line_update[lane_i_local*OUT_WIDTH + k_idx*8 +: 8] = byte_val;
                else
                    vec_line_update[lane_i_local*OUT_WIDTH +: OUT_WIDTH] = {{(OUT_WIDTH-8){byte_val[7]}}, byte_val};
            end
        end
    end
endfunction

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
// rd_sub: INT8 sub-word index within current word (0 to SUBW-1)
// rd_fill: total sub-words remaining to read
// ---------------------------------------------------------------------------
reg [ADDR_W-1:0] rd_ptr_a, rd_ptr_b;
reg [SUBW_W-1:0] rd_sub_a, rd_sub_b;  // sub-word index
reg [RD_FILL_W-1:0] rd_fill_a, rd_fill_b;

wire [ADDR_W-1:0] rd_ptr = rd_sel ? rd_ptr_b : rd_ptr_a;
wire [SUBW_W-1:0] rd_sub = rd_sel ? rd_sub_b : rd_sub_a;
wire [RD_FILL_W-1:0] cur_rd_fill = rd_sel ? rd_fill_b : rd_fill_a;

// ---------------------------------------------------------------------------
// Vector cache storage
// ---------------------------------------------------------------------------
// Vector cache: built while DMA writes sequential packed tile words.  This
// removes the old rd_vec path's many combinational reads from mem_a/mem_b.
reg [VEC_W-1:0] vec_mem_a [0:DEPTH-1];
reg [VEC_W-1:0] vec_mem_b [0:DEPTH-1];
reg [VEC_W-1:0] vec_build_a;
reg [VEC_W-1:0] vec_build_b;

// ---------------------------------------------------------------------------
// Write logic
// ---------------------------------------------------------------------------
wire do_write = wr_en && !buf_full;
wire [ADDR_W-1:0] wr_vec_group_idx = vec_group_idx(wr_ptr, vec_group_words);
wire [5:0] wr_vec_word_in_group = vec_word_in_group(wr_ptr, vec_group_words);
wire [2:0] wr_vec_k_idx = packed_int8 ? vec_k_idx(wr_vec_word_in_group, vec_words_per_k) : 3'd0;
wire [2:0] wr_vec_lane_word = packed_int8 ? vec_lane_word(wr_vec_word_in_group, vec_words_per_k)
                                          : wr_vec_word_in_group[2:0];
wire [VEC_W-1:0] wr_vec_build_cur = wr_sel ? vec_build_b : vec_build_a;
wire [VEC_W-1:0] wr_vec_build_base = (wr_vec_word_in_group == 6'd0) ? {VEC_W{1'b0}}
                                                                    : wr_vec_build_cur;
wire [VEC_W-1:0] wr_vec_build_next = vec_line_update(wr_vec_build_base, wr_data,
                                                     wr_vec_lane_word, wr_vec_k_idx,
                                                     rd_vec_lanes_eff, packed_int8);

always @(posedge clk) begin
    if (do_write) begin
        if (wr_sel == 1'b0) begin
            vec_build_a <= wr_vec_build_next;
            vec_mem_a[wr_vec_group_idx] <= wr_vec_build_next;
        end else begin
            vec_build_b <= wr_vec_build_next;
            vec_mem_b[wr_vec_group_idx] <= wr_vec_build_next;
        end
    end
end

// Optional scalar read storage. Tile-mode board builds can disable this path
// because rd_vec is the only PPBuf read interface used by the PE array.
generate
    if (SCALAR_READ_ENABLE != 0) begin : gen_scalar_mem
        reg [DATA_W-1:0] mem_a [0:DEPTH-1];
        reg [DATA_W-1:0] mem_b [0:DEPTH-1];

        always @(posedge clk) begin
            if (do_write) begin
                if (wr_sel == 1'b0)
                    mem_a[wr_ptr] <= wr_data;
                else
                    mem_b[wr_ptr] <= wr_data;
            end
        end

        // Current memory word
        wire [DATA_W-1:0] rd_mem = (rd_sel == 1'b0)
            ? mem_a[rd_ptr]
            : mem_b[rd_ptr];

        // ---- INT8 path: read one byte, sign-extend to OUT_WIDTH ----
        wire [7:0] rd_byte = (rd_sub == 0) ? rd_mem[ 7: 0] :
                             (rd_sub == 1) ? rd_mem[15: 8] :
                             (rd_sub == 2) ? rd_mem[23:16] :
                                             rd_mem[31:24];
        wire [OUT_WIDTH-1:0] rd_int8 = {{(OUT_WIDTH-8){rd_byte[7]}}, rd_byte};

        // When buffer is empty, output 0 to avoid PE latching stale data.
        assign rd_data = buf_empty ? {OUT_WIDTH{1'b0}} : rd_int8;
    end else begin : gen_no_scalar_mem
        assign rd_data = {OUT_WIDTH{1'b0}};
    end
endgenerate

// Update write pointers & fill counts. Bank contents are not reset here: fill
// counts define validity, and keeping memory writes in one always block avoids
// multi-driven RAM/register inference.
always @(posedge clk) begin
    if (!rst_n) begin
        wr_ptr_a   <= 0; wr_ptr_b   <= 0; wr_fill_a  <= 0; wr_fill_b  <= 0;
        vec_build_a <= {VEC_W{1'b0}}; vec_build_b <= {VEC_W{1'b0}};
    end else if (clear) begin
        wr_ptr_a   <= 0; wr_ptr_b   <= 0; wr_fill_a  <= 0; wr_fill_b  <= 0;
        vec_build_a <= {VEC_W{1'b0}}; vec_build_b <= {VEC_W{1'b0}};
    end else if (swap) begin
        // Reset the NEW writer's bank (it was just drained by PE)
        // New writer's bank = next_wr_sel (after swap)
        if (next_wr_sel == 1'b0) begin
            wr_ptr_a  <= 0;
            wr_fill_a <= 0;  // Reset fill count for new writer's bank
            vec_build_a <= {VEC_W{1'b0}};
        end else begin
            wr_ptr_b  <= 0;
            wr_fill_b <= 0;  // Reset fill count for new writer's bank
            vec_build_b <= {VEC_W{1'b0}};
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
wire do_vec_read = rd_vec_en && rd_vec_valid;
wire do_read = rd_en && !buf_empty && !do_vec_read;

assign rd_vec_valid = (rd_vec_lanes_eff != 5'd0) && (cur_rd_fill >= rd_vec_lanes_fill);
wire [ADDR_W-1:0] rd_vec_group_idx = vec_group_idx(rd_ptr, vec_group_words);
wire [VEC_W-1:0] rd_vec_cached = (rd_sel == 1'b0) ? vec_mem_a[rd_vec_group_idx]
                                                   : vec_mem_b[rd_vec_group_idx];
assign rd_vec = rd_vec_valid ? rd_vec_cached : {VEC_W{1'b0}};

wire [7:0] vec_abs_next = packed_int8
    ? ({4'b0000, rd_sub} + (rd_vec_lanes_eff * PACKED_LANES))
    : ({4'b0000, rd_sub} + {1'b0, rd_vec_lanes_eff});
wire [ADDR_W:0] vec_word_inc = (vec_abs_next >> 2);
wire [SUBW_W-1:0] vec_next_sub = packed_int8 ? vec_abs_next[SUBW_W-1:0]
    : vec_abs_next[SUBW_W-1:0];

// Update read pointers & fill counts
always @(posedge clk) begin
    if (!rst_n || clear) begin
        rd_ptr_a   <= 0;  rd_sub_a <= 0;  rd_fill_a <= 0;
        rd_ptr_b   <= 0;  rd_sub_b <= 0;  rd_fill_b <= 0;
    end else if (swap) begin
        // New reader's bank = old writer's bank = wr_sel (before swap)
        // Copy the write fill count as the read fill count in INT8 sub-words.
        if (wr_sel == 1'b0) begin
            rd_ptr_a  <= 0;
            rd_sub_a  <= 0;
            rd_fill_a <= wr_fill_a * SUBW;
        end else begin
            rd_ptr_b  <= 0;
            rd_sub_b  <= 0;
            rd_fill_b <= wr_fill_b * SUBW;
        end
    end else if (do_vec_read) begin
        if (rd_sel == 1'b0) begin
            `ifdef DIAG_PPBUF
            $display("[DIAG_PPBUF] vec_read: old_ptr=%0d old_fill=%0d inc=%0d new_ptr=%0d new_fill=%0d",
                     rd_ptr_a, rd_fill_a, vec_word_inc[ADDR_W-1:0],
                     rd_ptr_a + vec_word_inc[ADDR_W-1:0],
                     rd_fill_a - rd_vec_lanes_fill);
            `endif
            rd_ptr_a  <= rd_ptr_a + vec_word_inc[ADDR_W-1:0];
            rd_sub_a  <= vec_next_sub;
            rd_fill_a <= rd_fill_a - rd_vec_lanes_fill;
        end else begin
            rd_ptr_b  <= rd_ptr_b + vec_word_inc[ADDR_W-1:0];
            rd_sub_b  <= vec_next_sub;
            rd_fill_b <= rd_fill_b - rd_vec_lanes_fill;
        end
    end else if (do_read) begin
        if (rd_sel == 1'b0) begin
            // Advance INT8 sub-word index; wrap at SUBW-1.
            if (rd_sub_a == SUBW_LAST) begin
                rd_ptr_a  <= rd_ptr_a + 1'b1;
                rd_sub_a  <= 0;
            end else begin
                rd_sub_a  <= rd_sub_a + 1'b1;
            end
            rd_fill_a <= rd_fill_a - 1'b1;
        end else begin
            if (rd_sub_b == SUBW_LAST) begin
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
