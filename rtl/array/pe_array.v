// =============================================================================
// Module  : pe_array
// Project : NPU_prj
// Desc    : MxN systolic PE array.
//
//   Topology (Weight-Stationary example):
//
//      act_in[0] ──► PE[0][0] ──► PE[0][1] ──► … ──► PE[0][N-1]
//                       │                               │
//      act_in[1] ──► PE[1][0] ──► PE[1][1] ──► … ──► PE[1][N-1]
//                       │                               │
//             …         …                               …
//      act_in[M-1]► PE[M-1][0]─► PE[M-1][1]─► … ──► PE[M-1][N-1]
//
//   - Activations flow horizontally (row direction).
//   - Partial sums flow vertically (column direction).
//   - Each PE: pe_top instance.
//
//   OS Mode Note:
//   In OS mode, both weight and activation should flow systolically.
//   Current implementation uses broadcast for weight (simplified), which
//   works for 1x1 PE array but may cause bubbles in larger arrays.
//   For full systolic OS, weight needs horizontal shift registers.
//
// Parameters:
//   ROWS     - number of PE rows (M)
//   COLS     - number of PE columns (N)
//   DATA_W   - data width (16 for FP16/INT8)
//   ACC_W    - accumulator width (32)
// =============================================================================

`timescale 1ns/1ps

module pe_array #(
    parameter ROWS   = 4,
    parameter COLS   = 4,
    parameter DATA_W = 16,
    parameter ACC_W  = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,
    // global mode controls
    input  wire                          mode,       // 0=INT8, 1=FP16
    input  wire                          stat_mode,  // 0=WS, 1=OS
    input  wire                          en,
    input  wire                          flush,
    input  wire                          load_w,     // WS mode: latch w_in into weight_reg this cycle
    // weight inputs: one per column, broadcast to all rows in that column
    input  wire [COLS*DATA_W-1:0]        w_in,       // [col*DATA_W +: DATA_W]
    // activation inputs: one per row, shifted along that row
    input  wire [ROWS*DATA_W-1:0]        act_in,     // [row*DATA_W +: DATA_W]
    // partial sum inputs (bottom feed): one per column
    input  wire [COLS*ACC_W-1:0]         acc_in,     // [col*ACC_W  +: ACC_W ]
    // outputs: one per column (bottom of each column)
    output wire [COLS*ACC_W-1:0]         acc_out,    // [col*ACC_W  +: ACC_W ]
    output wire [COLS-1:0]               valid_out
);

// ---------------------------------------------------------------------------
// Internal wires:
//   act_h[row][col]  : horizontal activation bus  (ROWS × (COLS+1))
//   acc_v[row][col]  : vertical partial-sum bus   ((ROWS+1) × COLS)
// ---------------------------------------------------------------------------

// Activation: act_h[r][c] is the activation arriving at PE[r][c]
wire [DATA_W-1:0] act_h [0:ROWS-1][0:COLS];   // COLS+1 nodes per row
// Partial sum: acc_v[r][c] is the psum arriving at PE[r][c] from above
wire [ACC_W-1:0]  acc_v [0:ROWS][0:COLS-1];   // ROWS+1 nodes per col
wire              valid_v[0:ROWS][0:COLS-1];

// OS mode: weight horizontal shift registers to avoid broadcast bubbles
// w_h[row][col] is the weight arriving at PE[row][col]
wire [DATA_W-1:0] w_h [0:ROWS-1][0:COLS];     // COLS+1 nodes per row

// Connect external inputs
genvar r, c;
generate
    for (r = 0; r < ROWS; r = r+1) begin : gen_act_in
        assign act_h[r][0] = act_in[r*DATA_W +: DATA_W];
    end
    for (c = 0; c < COLS; c = c+1) begin : gen_acc_in
        assign acc_v[0][c]   = acc_in[c*ACC_W +: ACC_W];
        assign valid_v[0][c] = 1'b0;  // no valid from top boundary
    end
    // OS mode: weight flows horizontally like activation
    for (r = 0; r < ROWS; r = r+1) begin : gen_w_in_os
        assign w_h[r][0] = w_in[0 +: DATA_W];  // First column weight input
    end
endgenerate

// ---------------------------------------------------------------------------
// PE instantiation grid
// ---------------------------------------------------------------------------
generate
    for (r = 0; r < ROWS; r = r+1) begin : gen_row
        for (c = 0; c < COLS; c = c+1) begin : gen_col

            // Mode-dependent weight input:
            //   WS: broadcast per column (original behavior)
            //   OS: systolic flow horizontally
            wire [DATA_W-1:0] pe_w_in = stat_mode ? w_h[r][c] : w_in[c*DATA_W +: DATA_W];

            pe_top #(
                .DATA_W(DATA_W),
                .ACC_W (ACC_W)
            ) u_pe (
                .clk      (clk),
                .rst_n    (rst_n),
                .mode     (mode),
                .stat_mode(stat_mode),
                .en       (en),
                .flush    (flush),
                .load_w   (load_w),
                // weight: broadcast per column (WS) or systolic (OS)
                .w_in     (pe_w_in),
                // activation: systolic shift along row
                .a_in     (act_h[r][c]),
                // partial sum: pass down column
                .acc_in   (acc_v[r][c]),
                .acc_out  (acc_v[r+1][c]),
                .valid_out(valid_v[r+1][c])
            );

            // Activation passes to the right (registered inside PE stage-0,
            // so we forward the raw wire; add a register here for timing)
            // CRITICAL: Always propagate data to avoid bubbles, even when en=0
            reg [DATA_W-1:0] act_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    act_reg <= 0;
                else
                    act_reg <= act_h[r][c];  // Always propagate, not gated by en
            end
            assign act_h[r][c+1] = act_reg;

            // OS mode: Weight passes to the right (horizontal systolic)
            // This eliminates broadcast bubbles in OS mode
            reg [DATA_W-1:0] w_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    w_reg <= 0;
                else
                    w_reg <= w_h[r][c];  // Always propagate
            end
            assign w_h[r][c+1] = w_reg;

        end
    end
endgenerate

// ---------------------------------------------------------------------------
// Connect bottom row outputs
// ---------------------------------------------------------------------------
generate
    for (c = 0; c < COLS; c = c+1) begin : gen_out
        assign acc_out  [c*ACC_W +: ACC_W] = acc_v[ROWS][c];
        assign valid_out[c]                = valid_v[ROWS][c];
    end
endgenerate

endmodule
