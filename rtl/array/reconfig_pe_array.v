// =============================================================================
// Module  : reconfig_pe_array
// Project : NPU_prj
// Desc    : Reconfigurable 16x16 PE Array — 2D Systolic Data Flow Edition
//
//   Physical: 16x16 PE array, all instances always present.
//
//   cfg_shape modes:
//     2'b00 -> 4x4 mode:  Only rows[3:0], cols[3:0] active. Clock-gate rest.
//     2'b01 -> 8x8 mode:  Only rows[7:0], cols[7:0] active. Clock-gate rest.
//     2'b10 -> 16x16 mode: Full array active.
//     2'b11 -> 8x32 mode (folded): Two 8x16 halves stitched horizontally.
//
//   ── Data Flow (aligned with textbook figures 5-3 / 5-5) ──
//
//   OS (Output-Stationary) mode — Figure 5-3:
//     Activation: enters from LEFT per row, horizontal shift with column skew.
//                 Row skew via act_row_d delay chain (row r is r cycles delayed).
//     Weight:     enters from TOP per column, vertical shift with row skew.
//                 w_in[c] is the weight stream for column c.
//     Accumulate: each PE accumulates its own result independently.
//     Output:     each PE outputs independently.
//
//   WS (Weight-Stationary) mode — Figure 5-5:
//     Weight:     pre-loaded into each PE row-by-row.
//                 ws_load_row selects which row receives load_w this cycle.
//                 w_in[c] provides the weight for column c.
//                 After R load_w cycles, every PE holds a unique W[r][c].
//     Activation: enters from LEFT per row, row skew via act_row_d.
//                 Same value broadcast to all columns within a row (no column skew).
//     Accumulate: vertical psum chain — PE passes partial sum downward.
//     Output:     column bottom outputs the full dot-product.
//
//   ── Input Data Width ──
//
//     w_in:   PHY_COLS × DATA_W bits (one weight per column, loaded each cycle)
//     act_in: PHY_ROWS × DATA_W bits (one activation per row, loaded each cycle)
//     The upper layers must format data so that:
//       - WS weight load: w_in carries one row of the weight matrix per cycle
//       - OS weight stream: w_in carries one row of weights (one per column)
//       - Activation: act_in carries one column of activations (one per row)
//
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
    input  wire [PHY_COLS*DATA_W-1:0]    w_in,        // weight: per-column
    input  wire [PHY_ROWS*DATA_W-1:0]    act_in,      // activation: per-row
    input  wire [PHY_COLS*ACC_W-1:0]     acc_in,      // initial psum (OS mode)
    // outputs (max width for 8x32 folded mode = 32 columns)
    output reg  [32*ACC_W-1:0]           acc_out,
    output reg  [31:0]                   valid_out,
    // WS load row indicator (for controller to know current load row)
    output wire [3:0]                    ws_load_row_out,
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

// Active row/col counts per shape
wire [4:0] active_rows = (cfg_shape == MODE_4x4)  ? 5'd4  :
                          (cfg_shape == MODE_8x8)  ? 5'd8  :
                          5'd16;  // 16x16 and 8x32

wire [4:0] active_cols = (cfg_shape == MODE_4x4)  ? 5'd4  :
                          (cfg_shape == MODE_8x8)  ? 5'd8  :
                          5'd16;  // 16x16 and 8x32

// ---------------------------------------------------------------------------
// Internal buses
// ---------------------------------------------------------------------------

// Activation horizontal bus: act_h[row][col]  -- PHY_ROWS x (PHY_COLS+1)
// Used by OS mode for column-to-column skew
wire [DATA_W-1:0] act_h [0:PHY_ROWS-1][0:PHY_COLS];

// Row-delayed activation: act_row_skewed[row] -- one value per row after skew
// Used by WS mode (row skew, no column skew)
wire [DATA_W-1:0] act_row_skewed [0:PHY_ROWS-1];

// Partial-sum vertical bus: acc_v[row][col]  -- (PHY_ROWS+1) x PHY_COLS
wire [ACC_W-1:0]  acc_v [0:PHY_ROWS][0:PHY_COLS-1];
wire              valid_v[0:PHY_ROWS][0:PHY_COLS-1];

// OS-mode weight VERTICAL bus: w_v[row][col]  -- (PHY_ROWS+1) x PHY_COLS
// Weight enters from top and shifts downward with row skew
wire [DATA_W-1:0] w_v [0:PHY_ROWS][0:PHY_COLS-1];

// Per-PE combined active flag (for debug)
wire [PHY_ROWS*PHY_COLS-1:0] pe_active_flat;

genvar r, c;

// ---------------------------------------------------------------------------
// Activation row-delay chain (行间错拍)
//
//   WS mode: cascade delay — Row 0 direct, Row 1 gets Row 0's value 1 cycle
//            later, Row 2 gets Row 1's delayed value, etc.
//            This matches Figure 5-5: activation propagates downward row-by-row.
//
//   OS mode: each row uses its own act_in[r] directly (no row-to-row cascade).
//            Row skew is provided by the upper layer feeding different data
//            per row, or by the horizontal shift chain (act_h).
//
//   Implementation: we build a cascade delay chain from act_in[0].
//   WS mode taps from the chain, OS mode uses act_in[r] directly.
// ---------------------------------------------------------------------------

// Cascade delay chain: delay_d[r] = act_in[0] delayed by (r+1) cycles
reg [DATA_W-1:0] act_row_d [0:PHY_ROWS-1];

always @(posedge clk) begin
    if (!rst_n)
        act_row_d[0] <= {DATA_W{1'b0}};
    else
        act_row_d[0] <= act_in[0*DATA_W +: DATA_W];  // Row 0 input delayed 1 cycle
end

generate
    for (r = 1; r < PHY_ROWS; r = r+1) begin : gen_act_row_delay
        always @(posedge clk) begin
            if (!rst_n)
                act_row_d[r] <= {DATA_W{1'b0}};
            else
                act_row_d[r] <= act_row_d[r-1];  // cascade
        end
    end
endgenerate

// Row-skewed activation for WS mode:
//   act_row_skewed[0] = act_in[0]           (no delay)
//   act_row_skewed[1] = act_row_d[0]        (1 cycle delay)
//   act_row_skewed[r] = act_row_d[r-1]      (r cycles delay)
// (act_row_skewed already declared above)

assign act_row_skewed[0] = act_in[0*DATA_W +: DATA_W];

generate
    for (r = 1; r < PHY_ROWS; r = r+1) begin : gen_act_skewed
        assign act_row_skewed[r] = act_row_d[r-1];
    end
endgenerate

// ---------------------------------------------------------------------------
// Connect external inputs to boundary wires
// ---------------------------------------------------------------------------

// --- Activation horizontal bus entry: each row gets its SKEWED activation ---
// OS mode uses act_h for column skew; WS mode bypasses act_h (row broadcast)
generate
    for (r = 0; r < PHY_ROWS; r = r+1) begin : gen_act_h_entry
        assign act_h[r][0] = act_in[r*DATA_W +: DATA_W];
    end
endgenerate

// --- OS weight vertical bus entry: each column gets its own weight stream ---
generate
    for (c = 0; c < PHY_COLS; c = c+1) begin : gen_w_v_entry
        assign w_v[0][c] = w_in[c*DATA_W +: DATA_W];
    end
endgenerate

// --- Partial sum top boundary ---
generate
    for (c = 0; c < PHY_COLS; c = c+1) begin : gen_acc_in
        assign acc_v[0][c]   = acc_in[c*ACC_W +: ACC_W];
        assign valid_v[0][c] = 1'b0;
    end
endgenerate

// ---------------------------------------------------------------------------
// WS weight load row counter
// ---------------------------------------------------------------------------
reg [3:0] ws_load_row;

always @(posedge clk) begin
    if (!rst_n) begin
        ws_load_row <= 4'd0;
    end else if (load_w && !stat_mode) begin
        // WS mode: advance row counter on each load_w pulse
        // Wrap around at active_rows (controller ensures correct count)
        if (ws_load_row >= active_rows - 1)
            ws_load_row <= 4'd0;
        else
            ws_load_row <= ws_load_row + 4'd1;
    end else if (!load_w) begin
        // Reset when not loading (allows fresh start next load sequence)
        ws_load_row <= 4'd0;
    end
end

assign ws_load_row_out = ws_load_row;

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

            // ── Weight input selection ──
            // OS mode: weight from vertical shift chain w_v[r][c]
            // WS mode: weight from w_in per-column, but only current load row accepts
            wire [DATA_W-1:0] pe_w_in = stat_mode ? w_v[r][c]
                                                   : w_in[c*DATA_W +: DATA_W];

            // ── WS load_w gating: only the current row accepts weight ──
            wire pe_load_w = stat_mode ? load_w
                                       : (load_w && (r == ws_load_row));

            // ── Activation input selection ──
            // OS mode: use act_in[r] directly (row skew is provided by act_h
            //          horizontal shift chain and/or upper layer data formatting)
            // WS mode: use cascade-delayed activation (row-to-row propagation)
            wire [DATA_W-1:0] pe_a_in;
            if (r == 8 && c == 0) begin : fold_a_in
                // 8x32 fold: Row8,Col0 gets activation from Row7,Col15's output
                assign pe_a_in = fold_enable ? fold_act_from_top
                              : stat_mode     ? act_in[r*DATA_W +: DATA_W]
                                              : act_row_skewed[r];
            end else begin : normal_a_in
                assign pe_a_in = stat_mode ? act_in[r*DATA_W +: DATA_W]
                                           : act_row_skewed[r];
            end

            // ── Partial sum input: bottom half in 8x32 mode has no psum from above ──
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
                .load_w   (pe_load_w),
                .swap_w   (swap_w),
                .w_in     (pe_w_in),
                .a_in     (pe_a_in),
                .acc_in   (pe_acc_in),
                .acc_out  (acc_v[r+1][c]),
                .valid_out(valid_v[r+1][c])
            );

            // --- Horizontal activation shift register (OS mode column skew) ---
            reg [DATA_W-1:0] act_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    act_reg <= {DATA_W{1'b0}};
                else
                    act_reg <= act_h[r][c];
            end
            assign act_h[r][c+1] = act_reg;

            // Capture fold source: Row 7, Column 15's activation output
            if (r == 7 && c == 15) begin : fold_src
                assign fold_act_from_top = act_reg;
            end

            // --- Vertical weight shift register (OS mode row skew) ---
            reg [DATA_W-1:0] w_v_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    w_v_reg <= {DATA_W{1'b0}};
                else
                    w_v_reg <= w_v[r][c];
            end
            assign w_v[r+1][c] = w_v_reg;

        end  // gen_col
    end  // gen_row
endgenerate

// ---------------------------------------------------------------------------
// Output mapping
// ---------------------------------------------------------------------------
integer ri, ci;
always @(*) begin
    // Default: zero-fill everything
    acc_out   = {32*ACC_W{1'b0}};
    valid_out = {32{1'b0}};

    case (cfg_shape)
        MODE_4x4: begin
            for (ri = 0; ri < 4; ri = ri+1) begin
                for (ci = 0; ci < 4; ci = ci+1) begin
                    // T2.4 serializer expects row-major C tile order:
                    // result_index = ri*4 + ci -> C[m0+ri,n0+ci].
                    acc_out[(ri*4+ci)*ACC_W +: ACC_W] = acc_v[ri+1][ci];
                    valid_out[ri*4+ci]                = valid_v[ri+1][ci];
                end
            end
        end
        MODE_8x8: begin
            for (ci = 0; ci < PHY_COLS; ci = ci+1) begin
                if (ci < 8) begin
                    acc_out[ci*ACC_W +: ACC_W]   = acc_v[8][ci];
                    valid_out[ci]                  = valid_v[8][ci];
                end
            end
        end
        MODE_16x16: begin
            for (ci = 0; ci < PHY_COLS; ci = ci+1) begin
                acc_out[ci*ACC_W +: ACC_W]   = acc_v[PHY_ROWS][ci];
                valid_out[ci]                  = valid_v[PHY_ROWS][ci];
            end
        end
        MODE_8x32: begin
            for (ci = 0; ci < PHY_COLS; ci = ci+1) begin
                // Left half  (logical cols 0-15):  from top half row 8
                acc_out[ci*ACC_W +: ACC_W]         = acc_v[8][ci];
                valid_out[ci]                       = valid_v[8][ci];
                // Right half (logical cols 16-31): from bottom half row 16
                acc_out[(ci+16)*ACC_W +: ACC_W]     = acc_v[PHY_ROWS][ci];
                valid_out[ci+16]                     = valid_v[PHY_ROWS][ci];
            end
        end
        default: ;
    endcase
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
