// =============================================================================
// Module  : reconfig_pe_array
// Project : NPU_prj
// Desc    : Reconfigurable 16x16 PE Array with shape control and clock gating.
//
//   Physical: 16x16 WS (Weight-Stationary) PE array, all instances always present.
//
//   cfg_shape modes:
//     2'b00 -> 4x4 mode:  Only rows[3:0], cols[3:0] active. Clock-gate rest.
//     2'b01 -> 8x8 mode:  Only rows[7:0], cols[7:0] active. Clock-gate rest.
//     2'b10 -> 16x16 mode: Full array active.
//     2'b11 -> 8x32 mode (folded): Two 8x16 halves stitched horizontally.
//
//   Clock Gating: Per-PE clock enable derived from cfg_shape.
// =============================================================================

`timescale 1ns/1ps

module reconfig_pe_array #(
    parameter PHY_ROWS = 16,
    parameter PHY_COLS = 16,
    parameter DATA_W   = 16,
    parameter ACC_W    = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    // global controls
    input  wire [1:0]                    cfg_shape,
    input  wire                          mode,        // 0=INT8, 1=FP16
    input  wire                          stat_mode,   // 0=WS, 1=OS
    input  wire                          en,
    input  wire                          flush,
    input  wire                          load_w,
    input  wire                          swap_w,
    // data inputs
    input  wire [PHY_COLS*DATA_W-1:0]    w_in,
    input  wire [PHY_ROWS*DATA_W-1:0]    act_in,
    input  wire [PHY_COLS*ACC_W-1:0]     acc_in,
    // outputs (max width for 8x32 folded mode = 32 columns)
    output reg  [32*ACC_W-1:0]           acc_out,
    output reg  [31:0]                   valid_out,
    // debug
    output wire [PHY_ROWS*PHY_COLS-1:0]  pe_active
);

// ---------------------------------------------------------------------------
// Shape decode
// ---------------------------------------------------------------------------
localparam MODE_4x4   = 2'b00;
localparam MODE_8x8   = 2'b01;
localparam MODE_16x16 = 2'b10;
localparam MODE_8x32  = 2'b11;

// ---------------------------------------------------------------------------
// Internal wires
// ---------------------------------------------------------------------------

// Activation horizontal bus: act_h[row][col]  -- PHY_ROWS x (PHY_COLS+1)
wire [DATA_W-1:0] act_h [0:PHY_ROWS-1][0:PHY_COLS];

// Partial-sum vertical bus: acc_v[row][col]  -- (PHY_ROWS+1) x PHY_COLS
wire [ACC_W-1:0]  acc_v [0:PHY_ROWS][0:PHY_COLS-1];
wire              valid_v[0:PHY_ROWS][0:PHY_COLS-1];

// OS-mode weight horizontal bus
wire [DATA_W-1:0] w_h [0:PHY_ROWS-1][0:PHY_COLS];

// Per-PE combined active flag (for debug)
wire [PHY_ROWS*PHY_COLS-1:0] pe_active_flat;

genvar r, c;

// ---------------------------------------------------------------------------
// Connect external inputs to boundary wires
// ---------------------------------------------------------------------------
generate
    for (r = 0; r < PHY_ROWS; r = r+1) begin : gen_act_in
        assign act_h[r][0] = act_in[r*DATA_W +: DATA_W];
    end

    for (c = 0; c < PHY_COLS; c = c+1) begin : gen_acc_in
        assign acc_v[0][c]   = acc_in[c*ACC_W +: ACC_W];
        assign valid_v[0][c] = 1'b0;
    end

    for (r = 0; r < PHY_ROWS; r = r+1) begin : gen_w_in_os
        assign w_h[r][0] = w_in[0 +: DATA_W];
    end
endgenerate

// ---------------------------------------------------------------------------
// The fold signal for 8x32 mode
// ---------------------------------------------------------------------------
wire fold_enable = (cfg_shape == MODE_8x32);

// Capture row7,col15's activation output for fold-through to row8,col0
wire [DATA_W-1:0] fold_act_from_top;

// ---------------------------------------------------------------------------
// PE instantiation grid (16x16 physical)
// ---------------------------------------------------------------------------
generate
    for (r = 0; r < PHY_ROWS; r = r+1) begin : gen_row
        for (c = 0; c < PHY_COLS; c = c+1) begin : gen_col

            // Row/col active flags based on shape
            wire row_active = (cfg_shape == MODE_4x4) ? (r < 4) :
                              (cfg_shape == MODE_8x8) ? (r < 8) :
                              1'b1;  // 16x16 and 8x32 use all rows
            wire col_active = (cfg_shape == MODE_4x4) ? (c < 4) :
                              (cfg_shape == MODE_8x8) ? (c < 8) :
                              1'b1;  // 16x16 and 8x32 use all cols

            wire pe_clk_en = row_active && col_active;
            assign pe_active_flat[r * PHY_COLS + c] = pe_clk_en;

            // Weight input selection
            wire [DATA_W-1:0] pe_w_in = stat_mode ? w_h[r][c]
                                                   : w_in[c*DATA_W +: DATA_W];

            // Activation input with fold support
            wire [DATA_W-1:0] pe_a_in;
            if (r == 8 && c == 0) begin : fold_a_in
                assign pe_a_in = fold_enable ? fold_act_from_top : act_h[8][0];
            end else begin : normal_a_in
                assign pe_a_in = act_h[r][c];
            end

            // Partial sum input: bottom half in 8x32 mode has no psum from above
            wire [ACC_W-1:0] pe_acc_in;
            if (r >= 8) begin : fold_acc
                assign pe_acc_in = {ACC_W{1'b0}};
            end else begin : normal_acc
                assign pe_acc_in = acc_v[r][c];
            end

            pe_top #(
                .DATA_W(DATA_W),
                .ACC_W (ACC_W)
            ) u_pe (
                .clk      (clk),
                .rst_n    (rst_n),
                .mode     (mode),
                .stat_mode(stat_mode),
                .en       (en && pe_clk_en),
                .flush    (flush),
                .load_w   (load_w),
                .swap_w   (swap_w),
                .w_in     (pe_w_in),
                .a_in     (pe_a_in),
                .acc_in   (pe_acc_in),
                .acc_out  (acc_v[r+1][c]),
                .valid_out(valid_v[r+1][c])
            );

            // --- Horizontal activation shift register ---
            reg [DATA_W-1:0] act_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    act_reg <= 0;
                else
                    act_reg <= act_h[r][c];
            end
            assign act_h[r][c+1] = act_reg;

            // Capture fold source: Row 7, Column 15's activation output
            if (r == 7 && c == 15) begin : fold_src
                assign fold_act_from_top = act_reg;
            end

            // --- OS-mode horizontal weight shift register ---
            reg [DATA_W-1:0] os_w_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    os_w_reg <= 0;
                else
                    os_w_reg <= w_h[r][c];
            end
            assign w_h[r][c+1] = os_w_reg;

        end  // gen_col
    end  // gen_row
endgenerate

// ---------------------------------------------------------------------------
// Output mapping (combinational, using reg type so we can use always_comb)
// ---------------------------------------------------------------------------
integer ci;
always @(*) begin
    // Default: zero-fill everything
    acc_out   = {32*ACC_W{1'b0}};
    valid_out = {32{1'b0}};

    for (ci = 0; ci < PHY_COLS; ci = ci+1) begin
        case (cfg_shape)
            MODE_4x4: begin
                if (ci < 4) begin
                    acc_out[ci*ACC_W +: ACC_W]   = acc_v[4][ci];
                    valid_out[ci]                  = valid_v[4][ci];
                end
            end
            MODE_8x8: begin
                if (ci < 8) begin
                    acc_out[ci*ACC_W +: ACC_W]   = acc_v[8][ci];
                    valid_out[ci]                  = valid_v[8][ci];
                end
            end
            MODE_16x16: begin
                acc_out[ci*ACC_W +: ACC_W]   = acc_v[PHY_ROWS][ci];
                valid_out[ci]                  = valid_v[PHY_ROWS][ci];
            end
            MODE_8x32: begin
                // Left half  (logical cols 0-15):  from top half row 8
                acc_out[ci*ACC_W +: ACC_W]         = acc_v[8][ci];
                valid_out[ci]                       = valid_v[8][ci];
                // Right half (logical cols 16-31): from bottom half row 16
                acc_out[(ci+16)*ACC_W +: ACC_W]     = acc_v[PHY_ROWS][ci];
                valid_out[ci+16]                     = valid_v[PHY_ROWS][ci];
            end
            default: ;
        endcase
    end
end

// ---------------------------------------------------------------------------
// Debug: flatten pe_active
// ---------------------------------------------------------------------------
genvar idx;
generate
    for (idx = 0; idx < PHY_ROWS * PHY_COLS; idx = idx+1) begin : gen_pe_active_flat
        assign pe_active[idx] = pe_active_flat[idx];
    end
endgenerate

endmodule
