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
    parameter PHY_ROWS         = 16,
    parameter PHY_COLS         = 16,
    parameter DATA_W           = 16,
    parameter ACC_W            = 32,
    parameter MAX_TILE_RESULTS = 256, // max results per tile (16x16 or 8x32)
    parameter INT8_SIMD_LANES  = (DATA_W >= 64) ? 8 : ((DATA_W >= 32) ? 4 : 2),
    parameter FP16_ENABLE      = 0,
    parameter INT8_SCALAR_SIGNEXT_COMPAT = 1,
    parameter USE_ROUTER_MESH  = 0
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
    input  wire                          ws_direct,   // packed tile WS: load all active rows directly
    input  wire                          acc_init_en,
    input  wire                          half_en,     // 8x32: 0=top half active, 1=bottom half active
    input  wire                          array_ce,
    input  wire                          router_enable,
    input  wire                          os_act_systolic,
    input  wire                          os_weight_broadcast,
    input  wire [PHY_ROWS-1:0]           row_ce,
    input  wire [PHY_COLS-1:0]           col_ce,
    // data inputs
    input  wire [PHY_COLS*DATA_W-1:0]    w_in,        // weight: per-column
    input  wire [PHY_ROWS*DATA_W-1:0]    act_in,      // activation: per-row
    input  wire [PHY_COLS*ACC_W-1:0]     acc_in,      // initial psum (OS mode)
    input  wire [PHY_ROWS*PHY_COLS*ACC_W-1:0] acc_init,
    input  wire [PHY_ROWS*PHY_COLS-1:0]  acc_init_mask,
    // outputs (max width for 16x16 or 8x32 = 256 results)
    output reg  [MAX_TILE_RESULTS*ACC_W-1:0] acc_out,
    output reg  [MAX_TILE_RESULTS-1:0]       valid_out,
    // WS load row indicator (for controller to know current load row)
    output wire [3:0]                    ws_load_row_out,
    // debug
    output wire [PHY_ROWS*PHY_COLS-1:0]  pe_active,
    output wire                          router_ready,
    output wire                          router_overflow
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
                          (cfg_shape == MODE_8x32) ? 5'd8  :
                          5'd16;

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

// Per-PE combined active flags (for debug)
wire [PHY_ROWS*PHY_COLS-1:0] pe_active_flat;
wire [PHY_ROWS*PHY_COLS-1:0] direct_pe_active_flat;
wire [PHY_ROWS*PHY_COLS-1:0] router_pe_active_flat;

wire use_router_mesh = (USE_ROUTER_MESH != 0) && router_enable;
wire direct_path_enable = !use_router_mesh;
wire router_path_ready;
wire router_path_overflow;
wire [PHY_ROWS*PHY_COLS*ACC_W-1:0] router_acc_flat;
wire [PHY_ROWS*PHY_COLS-1:0]       router_valid_flat;

assign router_ready = use_router_mesh ? router_path_ready : 1'b1;
assign router_overflow = use_router_mesh ? router_path_overflow : 1'b0;
assign pe_active_flat = use_router_mesh ? router_pe_active_flat : direct_pe_active_flat;

genvar r, c, sr, sc;

// ---------------------------------------------------------------------------
// Activation row-delay chain (行间错拍)
//
//   True WS mode: row r receives its own boundary activation act_in[r], delayed
//                 to match the registered PE output and psum handoff latency.
//                 This lets rows represent independent K/SIMD groups.
//
//   OS mode: each row uses act_in[r] directly or the horizontal act_h chain.
// ---------------------------------------------------------------------------

assign act_row_skewed[0] = act_in[0*DATA_W +: DATA_W];

generate
    for (r = 1; r < PHY_ROWS; r = r+1) begin : gen_act_row_skewed
        localparam integer WS_ROW_DELAY = r * 3;
        reg [DATA_W-1:0] pipe [0:WS_ROW_DELAY-1];
        integer stage_i;

        always @(posedge clk) begin
            if (!rst_n) begin
                for (stage_i = 0; stage_i < WS_ROW_DELAY; stage_i = stage_i + 1)
                    pipe[stage_i] <= {DATA_W{1'b0}};
            end else if (array_ce) begin
                pipe[0] <= act_in[r*DATA_W +: DATA_W];
                for (stage_i = 1; stage_i < WS_ROW_DELAY; stage_i = stage_i + 1)
                    pipe[stage_i] <= pipe[stage_i-1];
            end
        end

        assign act_row_skewed[r] = pipe[WS_ROW_DELAY-1];
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
    end else if (load_w && !stat_mode && !ws_direct) begin
        // WS mode: advance row counter on each load_w pulse
        // Wrap around at active_rows (controller ensures correct count)
        if (ws_load_row >= active_rows - 1)
            ws_load_row <= 4'd0;
        else
            ws_load_row <= ws_load_row + 4'd1;
    end else if (!load_w && !(use_router_mesh && !stat_mode && !ws_direct)) begin
        // Direct WS has contiguous preload cycles. Router WS can backpressure
        // between accepted W beats, so it must hold the row pointer in gaps.
        ws_load_row <= 4'd0;
    end
end

assign ws_load_row_out = ws_load_row;

// ---------------------------------------------------------------------------
// The fold signal for 8x32 mode
// ---------------------------------------------------------------------------
wire fold_enable = (cfg_shape == MODE_8x32);

// In 8x32 mode the lower half implements logical columns 16..31. Each top
// physical row r continues horizontally into lower physical row r+8.
wire [DATA_W-1:0] fold_act_from_top [0:7];

// ---------------------------------------------------------------------------
// Optional router-mesh PE path.  When compiled in, software still selects it
// at runtime via router_enable; otherwise the direct PE grid remains active.
// ---------------------------------------------------------------------------
generate
    if (USE_ROUTER_MESH != 0) begin : gen_router_mesh_path
`ifdef ROUTER_MESH_ENABLE
        localparam [1:0] ROUTER_ST_IDLE   = 2'd0;
        localparam [1:0] ROUTER_ST_SEND_W = 2'd1;
        localparam [1:0] ROUTER_ST_SEND_A = 2'd2;

        reg [1:0] router_state;
        reg [PHY_ROWS*DATA_W-1:0] router_act_data_q;
        reg [PHY_COLS*DATA_W-1:0] router_weight_data_q;
        reg [PHY_ROWS-1:0] router_row_mask_q;
        reg [PHY_COLS-1:0] router_col_mask_q;
        reg router_flush_q;
        reg router_pending_valid;
        reg [PHY_ROWS*DATA_W-1:0] router_pending_act_data_q;
        reg [PHY_COLS*DATA_W-1:0] router_pending_weight_data_q;
        reg [PHY_ROWS-1:0] router_pending_row_mask_q;
        reg [PHY_COLS-1:0] router_pending_col_mask_q;
        reg router_pending_flush_q;
        reg router_send_w_q;
        reg router_send_a_q;
        reg router_load_w_q;
        reg [3:0] router_ws_load_row_q;
        reg router_pending_send_w_q;
        reg router_pending_send_a_q;
        reg router_pending_load_w_q;
        reg [3:0] router_pending_ws_load_row_q;
        reg router_overflow_q;

        wire [PHY_ROWS-1:0] router_row_mask_next;
        wire [PHY_COLS-1:0] router_col_mask_next;
        wire [PHY_ROWS*DATA_W-1:0] router_act_data_next;
        wire [PHY_ROWS-1:0] router_act_valid;
        wire [PHY_ROWS-1:0] router_act_ready;
        wire [PHY_COLS-1:0] router_weight_valid;
        wire [PHY_COLS-1:0] router_weight_ready;
        wire [PHY_ROWS*PHY_COLS-1:0] router_pe_valid;
        wire [PHY_ROWS*PHY_COLS*ACC_W-1:0] router_pe_acc_out;
        wire router_split_8x32_active = (cfg_shape == MODE_8x32) &&
                                        (PHY_ROWS >= 16) &&
                                        (PHY_COLS >= 16);
        wire [PHY_ROWS-1:0] router_act_valid_full = router_split_8x32_active ?
            {PHY_ROWS{1'b0}} : router_act_valid;
        wire [PHY_COLS-1:0] router_weight_valid_full = router_split_8x32_active ?
            {PHY_COLS{1'b0}} : router_weight_valid;
        wire [PHY_ROWS-1:0] router_act_ready_full;
        wire [PHY_COLS-1:0] router_weight_ready_full;
        wire [PHY_ROWS*PHY_COLS-1:0] router_pe_valid_full;
        wire [PHY_ROWS*PHY_COLS*ACC_W-1:0] router_pe_acc_out_full;
        wire [PHY_ROWS-1:0] router_act_ready_split;
        wire [PHY_COLS-1:0] router_weight_ready_split;
        wire [PHY_ROWS*PHY_COLS-1:0] router_pe_valid_split;
        wire [PHY_ROWS*PHY_COLS*ACC_W-1:0] router_pe_acc_out_split;
        localparam integer ROUTER_LANE_W = (INT8_SIMD_LANES > 0) ?
                                           ((DATA_W + INT8_SIMD_LANES - 1) / INT8_SIMD_LANES) :
                                           DATA_W;
        wire router_true_ws_mode = !stat_mode && !ws_direct;
        wire router_os_like_mode = stat_mode || ws_direct;
        wire router_mode_supported = !mode && (router_os_like_mode || router_true_ws_mode);
        wire router_input_attempt = router_enable && en && array_ce;
        wire router_input_has_payload = router_os_like_mode || load_w || !flush;
        wire router_input_valid = router_input_attempt && router_mode_supported && router_input_has_payload;
        wire router_input_unsupported = router_input_attempt && !router_mode_supported;
        wire router_send_w_next = router_os_like_mode || load_w;
        wire router_send_a_next = router_os_like_mode || (router_true_ws_mode && !load_w && !flush);
        wire router_ready_i = router_mode_supported &&
                              (!router_pending_valid || (router_state == ROUTER_ST_IDLE));
        assign router_act_ready = router_split_8x32_active ? router_act_ready_split
                                                           : router_act_ready_full;
        assign router_weight_ready = router_split_8x32_active ? router_weight_ready_split
                                                              : router_weight_ready_full;
        assign router_pe_valid = router_split_8x32_active ? router_pe_valid_split
                                                          : router_pe_valid_full;
        assign router_pe_acc_out = router_split_8x32_active ? router_pe_acc_out_split
                                                            : router_pe_acc_out_full;

        wire router_weight_done = ((router_weight_ready & router_col_mask_q) == router_col_mask_q);
        wire router_act_done = ((router_act_ready & router_row_mask_q) == router_row_mask_q);

        assign router_path_ready = router_ready_i;
        assign router_path_overflow = router_overflow_q;

        for (r = 0; r < PHY_ROWS; r = r+1) begin : gen_router_rows
            wire row_active = (cfg_shape == MODE_4x4) ? (r < 4) :
                              (cfg_shape == MODE_8x8) ? (r < 8) :
                              (cfg_shape == MODE_8x32) ?
                                  (flush ? (r < 16) : (half_en ? ((r >= 8) && (r < 16))
                                                                : (r < 8))) :
                              (r < 16);
            wire row_power_enabled;
            if (r >= 8) begin : gen_router_8x32_lower_row
                assign row_power_enabled = (cfg_shape == MODE_8x32) ? row_ce[r-8]
                                                                    : row_ce[r];
                assign router_act_data_next[r*DATA_W +: DATA_W] =
                    (cfg_shape == MODE_8x32) ? act_in[(r-8)*DATA_W +: DATA_W]
                                             : act_in[r*DATA_W +: DATA_W];
            end else begin : gen_router_normal_row
                assign row_power_enabled = row_ce[r];
                assign router_act_data_next[r*DATA_W +: DATA_W] = act_in[r*DATA_W +: DATA_W];
            end
            assign router_row_mask_next[r] = row_active && row_power_enabled;
        end

        for (c = 0; c < PHY_COLS; c = c+1) begin : gen_router_cols
            wire col_active = (cfg_shape == MODE_4x4) ? (c < 4) :
                              (cfg_shape == MODE_8x8) ? (c < 8) :
                              ((cfg_shape == MODE_16x16) || (cfg_shape == MODE_8x32));
            assign router_col_mask_next[c] = col_active && col_ce[c];
        end

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                router_state <= ROUTER_ST_IDLE;
                router_act_data_q <= {PHY_ROWS*DATA_W{1'b0}};
                router_weight_data_q <= {PHY_COLS*DATA_W{1'b0}};
                router_row_mask_q <= {PHY_ROWS{1'b0}};
                router_col_mask_q <= {PHY_COLS{1'b0}};
                router_flush_q <= 1'b0;
                router_pending_valid <= 1'b0;
                router_pending_act_data_q <= {PHY_ROWS*DATA_W{1'b0}};
                router_pending_weight_data_q <= {PHY_COLS*DATA_W{1'b0}};
                router_pending_row_mask_q <= {PHY_ROWS{1'b0}};
                router_pending_col_mask_q <= {PHY_COLS{1'b0}};
                router_pending_flush_q <= 1'b0;
                router_send_w_q <= 1'b0;
                router_send_a_q <= 1'b0;
                router_load_w_q <= 1'b0;
                router_ws_load_row_q <= 4'd0;
                router_pending_send_w_q <= 1'b0;
                router_pending_send_a_q <= 1'b0;
                router_pending_load_w_q <= 1'b0;
                router_pending_ws_load_row_q <= 4'd0;
                router_overflow_q <= 1'b0;
            end else if (router_input_unsupported) begin
                router_overflow_q <= 1'b1;
`ifndef SYNTHESIS
                $display("[ERROR] reconfig_pe_array router unsupported input at t=%0t cfg_shape=%0b stat_mode=%0b mode=%0b",
                         $time, cfg_shape, stat_mode, mode);
                $fatal;
`endif
            end else begin
                case (router_state)
                    ROUTER_ST_IDLE: begin
                        if (router_pending_valid) begin
                            router_act_data_q <= router_pending_act_data_q;
                            router_weight_data_q <= router_pending_weight_data_q;
                            router_row_mask_q <= router_pending_row_mask_q;
                            router_col_mask_q <= router_pending_col_mask_q;
                            router_flush_q <= router_pending_flush_q;
                            router_send_w_q <= router_pending_send_w_q;
                            router_send_a_q <= router_pending_send_a_q;
                            router_load_w_q <= router_pending_load_w_q;
                            router_ws_load_row_q <= router_pending_ws_load_row_q;
                            router_state <= router_pending_send_w_q ? ROUTER_ST_SEND_W : ROUTER_ST_SEND_A;
                            router_pending_valid <= router_input_valid;
                            if (router_input_valid) begin
                                router_pending_act_data_q <= router_act_data_next;
                                router_pending_weight_data_q <= w_in;
                                router_pending_row_mask_q <= router_row_mask_next;
                                router_pending_col_mask_q <= router_col_mask_next;
                                router_pending_flush_q <= flush;
                                router_pending_send_w_q <= router_send_w_next;
                                router_pending_send_a_q <= router_send_a_next;
                                router_pending_load_w_q <= load_w;
                                router_pending_ws_load_row_q <= ws_load_row;
                            end
                        end else if (router_input_valid) begin
                            router_act_data_q <= router_act_data_next;
                            router_weight_data_q <= w_in;
                            router_row_mask_q <= router_row_mask_next;
                            router_col_mask_q <= router_col_mask_next;
                            router_flush_q <= flush;
                            router_send_w_q <= router_send_w_next;
                            router_send_a_q <= router_send_a_next;
                            router_load_w_q <= load_w;
                            router_ws_load_row_q <= ws_load_row;
                            router_state <= router_send_w_next ? ROUTER_ST_SEND_W : ROUTER_ST_SEND_A;
                        end
                    end
                    ROUTER_ST_SEND_W: begin
                        if (router_input_valid) begin
                            if (!router_pending_valid) begin
                                router_pending_valid <= 1'b1;
                                router_pending_act_data_q <= router_act_data_next;
                                router_pending_weight_data_q <= w_in;
                                router_pending_row_mask_q <= router_row_mask_next;
                                router_pending_col_mask_q <= router_col_mask_next;
                                router_pending_flush_q <= flush;
                                router_pending_send_w_q <= router_send_w_next;
                                router_pending_send_a_q <= router_send_a_next;
                                router_pending_load_w_q <= load_w;
                                router_pending_ws_load_row_q <= ws_load_row;
                            end else begin
                                router_overflow_q <= 1'b1;
`ifndef SYNTHESIS
                                $display("[ERROR] reconfig_pe_array router input overflow at t=%0t", $time);
                                $fatal;
`endif
                            end
                        end
                        router_col_mask_q <= router_col_mask_q & ~router_weight_ready;
                        if (router_weight_done)
                            router_state <= router_send_a_q ? ROUTER_ST_SEND_A : ROUTER_ST_IDLE;
                    end
                    ROUTER_ST_SEND_A: begin
                        if (router_input_valid) begin
                            if (!router_pending_valid) begin
                                router_pending_valid <= 1'b1;
                                router_pending_act_data_q <= router_act_data_next;
                                router_pending_weight_data_q <= w_in;
                                router_pending_row_mask_q <= router_row_mask_next;
                                router_pending_col_mask_q <= router_col_mask_next;
                                router_pending_flush_q <= flush;
                                router_pending_send_w_q <= router_send_w_next;
                                router_pending_send_a_q <= router_send_a_next;
                                router_pending_load_w_q <= load_w;
                                router_pending_ws_load_row_q <= ws_load_row;
                            end else begin
                                router_overflow_q <= 1'b1;
`ifndef SYNTHESIS
                                $display("[ERROR] reconfig_pe_array router input overflow at t=%0t", $time);
                                $fatal;
`endif
                            end
                        end
                        router_row_mask_q <= router_row_mask_q & ~router_act_ready;
                        if (router_act_done)
                            router_state <= ROUTER_ST_IDLE;
                    end
                    default: begin
                        router_state <= ROUTER_ST_IDLE;
                    end
                endcase
            end
        end

        assign router_weight_valid = ((router_state == ROUTER_ST_SEND_W) && router_send_w_q)
                                   ? router_col_mask_q
                                   : {PHY_COLS{1'b0}};
        assign router_act_valid = ((router_state == ROUTER_ST_SEND_A) && router_send_a_q)
                                ? router_row_mask_q
                                : {PHY_ROWS{1'b0}};

        router_pe_array_lite #(
            .ROWS(PHY_ROWS),
            .COLS(PHY_COLS),
            .XW(4),
            .YW(4),
            .LANES(INT8_SIMD_LANES),
            .LANE_W(ROUTER_LANE_W),
            .PE_DATA_W(DATA_W),
            .ACC_W(ACC_W),
            .INT8_SIMD_LANES(INT8_SIMD_LANES),
            .FP16_ENABLE(FP16_ENABLE),
            .INT8_SCALAR_SIGNEXT_COMPAT(INT8_SCALAR_SIGNEXT_COMPAT),
            .AUTO_FLUSH_ON_COMPUTE(0),
            .PE_STREAM_BUF_DEPTH(32)
        ) u_router_pe_array_lite_full (
            .clk(clk),
            .rst_n(rst_n),
            .flush(router_flush_q),
            .mode(mode),
            .stat_mode(stat_mode),
            .load_w(router_load_w_q),
            .swap_w(swap_w),
            .ws_direct(ws_direct),
            .ws_load_row(router_ws_load_row_q),
            .act_valid(router_act_valid_full),
            .act_ready(router_act_ready_full),
            .act_data(router_act_data_q),
            .weight_valid(router_weight_valid_full),
            .weight_ready(router_weight_ready_full),
            .weight_data(router_weight_data_q),
            .pe_valid(router_pe_valid_full),
            .pe_compute_fire(),
            .pe_acc_out(router_pe_acc_out_full)
        );

        if ((PHY_ROWS >= 16) && (PHY_COLS >= 16)) begin : gen_router_8x32_split
            wire [7:0] split_top_act_valid;
            wire [7:0] split_bot_act_valid;
            wire [7:0] split_top_act_ready;
            wire [7:0] split_bot_act_ready;
            wire [15:0] split_top_weight_valid;
            wire [15:0] split_bot_weight_valid;
            wire [15:0] split_top_weight_ready;
            wire [15:0] split_bot_weight_ready;
            wire [8*DATA_W-1:0] split_top_act_data;
            wire [8*DATA_W-1:0] split_bot_act_data;
            wire [16*DATA_W-1:0] split_weight_data;
            wire [8*16-1:0] split_top_pe_valid;
            wire [8*16-1:0] split_bot_pe_valid;
            wire [8*16*ACC_W-1:0] split_top_pe_acc_out;
            wire [8*16*ACC_W-1:0] split_bot_pe_acc_out;
            wire split_top_active = |router_row_mask_q[7:0];
            wire split_bot_active = |router_row_mask_q[15:8];

            assign split_top_act_valid = router_act_valid[7:0];
            assign split_bot_act_valid = router_act_valid[15:8];
            assign split_top_act_data = router_act_data_q[0 +: 8*DATA_W];
            assign split_bot_act_data = router_act_data_q[8*DATA_W +: 8*DATA_W];
            assign split_weight_data = router_weight_data_q[0 +: 16*DATA_W];
            assign split_top_weight_valid = split_top_active ? router_weight_valid[15:0] : 16'h0000;
            assign split_bot_weight_valid = split_bot_active ? router_weight_valid[15:0] : 16'h0000;

            assign router_act_ready_split[7:0] = split_top_act_ready;
            assign router_act_ready_split[15:8] = split_bot_act_ready;
            assign router_weight_ready_split[15:0] =
                (split_top_active ? split_top_weight_ready : 16'hffff) &
                (split_bot_active ? split_bot_weight_ready : 16'hffff);

            if (PHY_ROWS > 16) begin : gen_router_split_extra_rows
                assign router_act_ready_split[PHY_ROWS-1:16] = {(PHY_ROWS-16){1'b1}};
            end
            if (PHY_COLS > 16) begin : gen_router_split_extra_cols
                assign router_weight_ready_split[PHY_COLS-1:16] = {(PHY_COLS-16){1'b1}};
            end

            router_pe_array_lite #(
                .ROWS(8),
                .COLS(16),
                .XW(4),
                .YW(4),
                .LANES(INT8_SIMD_LANES),
                .LANE_W(ROUTER_LANE_W),
                .PE_DATA_W(DATA_W),
                .ACC_W(ACC_W),
                .INT8_SIMD_LANES(INT8_SIMD_LANES),
                .FP16_ENABLE(FP16_ENABLE),
                .INT8_SCALAR_SIGNEXT_COMPAT(INT8_SCALAR_SIGNEXT_COMPAT),
                .AUTO_FLUSH_ON_COMPUTE(0),
                .PE_STREAM_BUF_DEPTH(32)
            ) u_router_pe_array_lite_top8 (
                .clk(clk),
                .rst_n(rst_n),
                .flush(router_flush_q),
                .mode(mode),
                .stat_mode(stat_mode),
                .load_w(router_load_w_q),
                .swap_w(swap_w),
                .ws_direct(ws_direct),
                .ws_load_row(router_ws_load_row_q),
                .act_valid(split_top_act_valid),
                .act_ready(split_top_act_ready),
                .act_data(split_top_act_data),
                .weight_valid(split_top_weight_valid),
                .weight_ready(split_top_weight_ready),
                .weight_data(split_weight_data),
                .pe_valid(split_top_pe_valid),
                .pe_compute_fire(),
                .pe_acc_out(split_top_pe_acc_out)
            );

            router_pe_array_lite #(
                .ROWS(8),
                .COLS(16),
                .XW(4),
                .YW(4),
                .LANES(INT8_SIMD_LANES),
                .LANE_W(ROUTER_LANE_W),
                .PE_DATA_W(DATA_W),
                .ACC_W(ACC_W),
                .INT8_SIMD_LANES(INT8_SIMD_LANES),
                .FP16_ENABLE(FP16_ENABLE),
                .INT8_SCALAR_SIGNEXT_COMPAT(INT8_SCALAR_SIGNEXT_COMPAT),
                .AUTO_FLUSH_ON_COMPUTE(0),
                .PE_STREAM_BUF_DEPTH(32)
            ) u_router_pe_array_lite_bot8 (
                .clk(clk),
                .rst_n(rst_n),
                .flush(router_flush_q),
                .mode(mode),
                .stat_mode(stat_mode),
                .load_w(router_load_w_q),
                .swap_w(swap_w),
                .ws_direct(ws_direct),
                .ws_load_row(router_ws_load_row_q),
                .act_valid(split_bot_act_valid),
                .act_ready(split_bot_act_ready),
                .act_data(split_bot_act_data),
                .weight_valid(split_bot_weight_valid),
                .weight_ready(split_bot_weight_ready),
                .weight_data(split_weight_data),
                .pe_valid(split_bot_pe_valid),
                .pe_compute_fire(),
                .pe_acc_out(split_bot_pe_acc_out)
            );

            for (sr = 0; sr < PHY_ROWS; sr = sr + 1) begin : gen_router_split_out_rows
                for (sc = 0; sc < PHY_COLS; sc = sc + 1) begin : gen_router_split_out_cols
                    if ((sr < 8) && (sc < 16)) begin : gen_router_split_top_out
                        assign router_pe_acc_out_split[(sr*PHY_COLS+sc)*ACC_W +: ACC_W] =
                            split_top_pe_acc_out[(sr*16+sc)*ACC_W +: ACC_W];
                        assign router_pe_valid_split[sr*PHY_COLS+sc] = split_top_pe_valid[sr*16+sc];
                    end else if ((sr >= 8) && (sr < 16) && (sc < 16)) begin : gen_router_split_bot_out
                        assign router_pe_acc_out_split[(sr*PHY_COLS+sc)*ACC_W +: ACC_W] =
                            split_bot_pe_acc_out[((sr-8)*16+sc)*ACC_W +: ACC_W];
                        assign router_pe_valid_split[sr*PHY_COLS+sc] = split_bot_pe_valid[(sr-8)*16+sc];
                    end else begin : gen_router_split_zero_out
                        assign router_pe_acc_out_split[(sr*PHY_COLS+sc)*ACC_W +: ACC_W] = {ACC_W{1'b0}};
                        assign router_pe_valid_split[sr*PHY_COLS+sc] = 1'b0;
                    end
                end
            end
        end else begin : gen_router_no_8x32_split
            assign router_act_ready_split = {PHY_ROWS{1'b1}};
            assign router_weight_ready_split = {PHY_COLS{1'b1}};
            assign router_pe_acc_out_split = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
            assign router_pe_valid_split = {PHY_ROWS*PHY_COLS{1'b0}};
        end

        for (r = 0; r < PHY_ROWS; r = r+1) begin : gen_router_out_rows
            for (c = 0; c < PHY_COLS; c = c+1) begin : gen_router_out_cols
                assign router_acc_flat[(r*PHY_COLS+c)*ACC_W +: ACC_W] =
                    router_pe_acc_out[(r*PHY_COLS+c)*ACC_W +: ACC_W];
                assign router_valid_flat[r*PHY_COLS+c] = router_pe_valid[r*PHY_COLS+c];
                assign router_pe_active_flat[r*PHY_COLS+c] = router_enable &&
                                                             router_row_mask_next[r] &&
                                                             router_col_mask_next[c];
            end
        end
`else
        initial begin
            $display("[ERROR] reconfig_pe_array USE_ROUTER_MESH requires +define+ROUTER_MESH_ENABLE and router RTL files");
            $finish;
        end

        assign router_path_ready = 1'b0;
        assign router_path_overflow = 1'b1;
        assign router_acc_flat = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
        assign router_valid_flat = {PHY_ROWS*PHY_COLS{1'b0}};
        assign router_pe_active_flat = {PHY_ROWS*PHY_COLS{1'b0}};
`endif
    end else begin : gen_no_router_mesh_path
        assign router_path_ready = 1'b0;
        assign router_path_overflow = 1'b0;
        assign router_acc_flat = {PHY_ROWS*PHY_COLS*ACC_W{1'b0}};
        assign router_valid_flat = {PHY_ROWS*PHY_COLS{1'b0}};
        assign router_pe_active_flat = {PHY_ROWS*PHY_COLS{1'b0}};
    end
endgenerate

// ---------------------------------------------------------------------------
// Direct PE instantiation grid (16x16 physical)
// ---------------------------------------------------------------------------
generate
    if (1) begin : gen_direct_pe_grid
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
            wire pe_power_ce = array_ce && row_ce[r] && col_ce[c];
            wire direct_power_ce = direct_path_enable && pe_power_ce;
            assign direct_pe_active_flat[r * PHY_COLS + c] = direct_path_enable && pe_clk_en;
            // 8x32 two-pass: half_en gates which half-array is active.
            // half_en=0: rows 0-7 active (top half, logical cols 0-15)
            // half_en=1: rows 8-15 active (bottom half, logical cols 16-31)
            // other shapes: fold_enable=0 → all rows always active
            // During flush, enable all PEs regardless of half_en so that
            // valid_out propagates from all rows and capture triggers correctly.
            wire pe_half_active = flush || !fold_enable || (half_en ? (r >= 8) : (r < 8));
            wire pe_acc_init_en = acc_init_en &&
                                  direct_path_enable &&
                                  pe_clk_en &&
                                  pe_half_active &&
                                  pe_power_ce &&
                                  acc_init_mask[r * PHY_COLS + c];

            // ── Weight input selection ──
            // OS mode supports either vertical systolic flow or per-column broadcast.
            // WS mode uses w_in per-column, with load_w gating deciding which PE latches it.
            wire [DATA_W-1:0] pe_w_in = stat_mode
                                      ? (os_weight_broadcast ? w_in[c*DATA_W +: DATA_W]
                                                             : w_v[r][c])
                                      : w_in[c*DATA_W +: DATA_W];

            // ── WS load_w gating: only the current row accepts weight ──
            wire fold_load_row_match = (r < 8) ? (r == ws_load_row)
                                               : ((r - 8) == ws_load_row);
            wire pe_load_w = stat_mode ? load_w
                                       : (ws_direct
                                          ? load_w
                                          : (load_w &&
                                             (fold_enable ? fold_load_row_match
                                                          : (r == ws_load_row))));
            wire pe_load_w_ce = direct_path_enable && pe_load_w && pe_power_ce;
            wire pe_swap_w_ce = direct_path_enable && swap_w && pe_power_ce;
            wire pe_flush_ce  = direct_path_enable && flush && pe_power_ce;
            // Packed tile WS has no external psum-chain schedule yet, so keep
            // that direct feeder on the PE's internal OS accumulator path.
            wire pe_stat_mode = stat_mode || ws_direct;

            // ── Activation input selection ──
            // OS mode: use act_in[r] directly (row skew is provided by act_h
            //          horizontal shift chain and/or upper layer data formatting)
            // WS mode: use cascade-delayed activation (row-to-row propagation)
            wire [DATA_W-1:0] pe_a_in;
            wire [DATA_W-1:0] pe_a_os_in = os_act_systolic ? act_h[r][c]
                                                           : act_in[r*DATA_W +: DATA_W];

            if (r >= 8) begin : fold_a_in
                // 8x32 OS two-pass: route same activation to both halves.
                // The fold delay (16-cycle horizontal shift) is incompatible
                // with the single-fire packed SIMD architecture — bottom half
                // would see activation 16 cycles after weight injection.
                assign pe_a_in = (fold_enable && stat_mode)
                               ? (os_act_systolic ? fold_act_from_top[r-8]
                                                  : act_in[(r-8)*DATA_W +: DATA_W])
                               : (fold_enable && ws_direct)
                               ? act_in[(r-8)*DATA_W +: DATA_W]
                               : (fold_enable
                                   ? act_row_skewed[r-8]
                                   : (stat_mode ? pe_a_os_in
                                                : (ws_direct ? act_in[r*DATA_W +: DATA_W]
                                                             : act_row_skewed[r])));
            end else begin : normal_a_in
                assign pe_a_in = stat_mode ? pe_a_os_in
                                           : (ws_direct ? act_in[r*DATA_W +: DATA_W]
                                                        : act_row_skewed[r]);
            end

            // ── Partial sum input: 8x32 splits the vertical chain at row8 ──
            wire [ACC_W-1:0] pe_acc_in;
            assign pe_acc_in = (fold_enable && (r == 8)) ? {ACC_W{1'b0}}
                                                         : acc_v[r][c];

            pe_top #(
                .DATA_W(DATA_W),
                .ACC_W (ACC_W),
                .INT8_SIMD_LANES(INT8_SIMD_LANES),
                .FP16_ENABLE(FP16_ENABLE),
                .INT8_SCALAR_SIGNEXT_COMPAT(INT8_SCALAR_SIGNEXT_COMPAT)
            ) u_pe (
                .clk      (clk),
                .rst_n    (rst_n),
                .mode     (mode),
                .stat_mode(pe_stat_mode),
                .en       (en && direct_path_enable && pe_clk_en && pe_half_active && pe_power_ce),
                .flush    (pe_flush_ce),
                .load_w   (pe_load_w_ce),
                .swap_w   (pe_swap_w_ce),
                .acc_init_en(pe_acc_init_en),
                .w_in     (pe_w_in),
                .a_in     (pe_a_in),
                .acc_in   (pe_acc_in),
                .acc_init (acc_init[(r*PHY_COLS+c)*ACC_W +: ACC_W]),
                .acc_out  (acc_v[r+1][c]),
                .valid_out(valid_v[r+1][c])
            );

            // --- Horizontal activation shift register (OS mode column skew) ---
            reg [DATA_W-1:0] act_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    act_reg <= {DATA_W{1'b0}};
                else if (direct_power_ce)
                    act_reg <= act_h[r][c];
            end
            assign act_h[r][c+1] = act_reg;

            if (r < 8 && c == PHY_COLS-1) begin : fold_act_src
                assign fold_act_from_top[r] = act_reg;
            end

            // --- Vertical weight shift register (OS mode row skew) ---
            reg [DATA_W-1:0] w_v_reg;
            always @(posedge clk) begin
                if (!rst_n)
                    w_v_reg <= {DATA_W{1'b0}};
                else if (direct_power_ce)
                    w_v_reg <= w_v[r][c];
            end
            if (r == 7) begin : fold_weight_boundary
                // Lower 8x16 half is an independent right-half array.
                assign w_v[r+1][c] = fold_enable ? w_in[c*DATA_W +: DATA_W]
                                                 : w_v_reg;
            end else begin : normal_weight_boundary
                assign w_v[r+1][c] = w_v_reg;
            end

        end  // gen_col
    end  // gen_row
    end
endgenerate

// ---------------------------------------------------------------------------
// Output mapping — per-PE row-major grid for all shapes.
//
//   cfg_shape  | active physical rows | active physical cols | grid rows | grid cols | total results
//   -----------|-----------------------|----------------------|-----------|-----------|---------------
//   4x4        | rows 0..3             | cols 0..3            | 4         | 4         | 16
//   8x8        | rows 0..7             | cols 0..7            | 8         | 8         | 64
//   16x16      | rows 0..15            | cols 0..15           | 16        | 16        | 256
//   8x32 folded | rows 0..7 top+bot    | cols 0..15 ×2 halves | 8         | 32        | 256
//
//   result_index = grid_row * grid_cols + grid_col
//   C[m0 + grid_row, n0 + grid_col]
// ---------------------------------------------------------------------------
integer ri, ci;
always @(*) begin
    // Default: zero-fill everything
    acc_out   = {MAX_TILE_RESULTS*ACC_W{1'b0}};
    valid_out = {MAX_TILE_RESULTS{1'b0}};

    case (cfg_shape)
        MODE_4x4: begin
            for (ri = 0; ri < 4; ri = ri+1) begin
                for (ci = 0; ci < 4; ci = ci+1) begin
                    if ((ri < PHY_ROWS) && (ci < PHY_COLS)) begin
                        if (use_router_mesh) begin
                            acc_out[(ri*4+ci)*ACC_W +: ACC_W] =
                                router_acc_flat[(ri*PHY_COLS+ci)*ACC_W +: ACC_W];
                            valid_out[ri*4+ci] = router_valid_flat[ri*PHY_COLS+ci];
                        end else begin
                            acc_out[(ri*4+ci)*ACC_W +: ACC_W] = acc_v[ri+1][ci];
                            valid_out[ri*4+ci]                = valid_v[ri+1][ci];
                        end
                    end
                end
            end
        end
        MODE_8x8: begin
            for (ri = 0; ri < 8; ri = ri+1) begin
                for (ci = 0; ci < 8; ci = ci+1) begin
                    if ((ri < PHY_ROWS) && (ci < PHY_COLS)) begin
                        if (use_router_mesh) begin
                            acc_out[(ri*8+ci)*ACC_W +: ACC_W] =
                                router_acc_flat[(ri*PHY_COLS+ci)*ACC_W +: ACC_W];
                            valid_out[ri*8+ci] = router_valid_flat[ri*PHY_COLS+ci];
                        end else begin
                            acc_out[(ri*8+ci)*ACC_W +: ACC_W] = acc_v[ri+1][ci];
                            valid_out[ri*8+ci]                = valid_v[ri+1][ci];
                        end
                    end
                end
            end
        end
        MODE_16x16: begin
            for (ri = 0; ri < 16; ri = ri+1) begin
                for (ci = 0; ci < 16; ci = ci+1) begin
                    if ((ri < PHY_ROWS) && (ci < PHY_COLS)) begin
                        if (use_router_mesh) begin
                            acc_out[(ri*16+ci)*ACC_W +: ACC_W] =
                                router_acc_flat[(ri*PHY_COLS+ci)*ACC_W +: ACC_W];
                            valid_out[ri*16+ci] = router_valid_flat[ri*PHY_COLS+ci];
                        end else begin
                            acc_out[(ri*16+ci)*ACC_W +: ACC_W] = acc_v[ri+1][ci];
                            valid_out[ri*16+ci]                = valid_v[ri+1][ci];
                        end
                    end
                end
            end
        end
        MODE_8x32: begin
            // Top half: logical rows 0..7, logical cols 0..15
            for (ri = 0; ri < 8; ri = ri+1) begin
                for (ci = 0; ci < 16; ci = ci+1) begin
                    if ((ri < PHY_ROWS) && (ci < PHY_COLS)) begin
                        if (use_router_mesh) begin
                            acc_out[(ri*32+ci)*ACC_W +: ACC_W] =
                                router_acc_flat[(ri*PHY_COLS+ci)*ACC_W +: ACC_W];
                            valid_out[ri*32+ci] = router_valid_flat[ri*PHY_COLS+ci];
                        end else begin
                            acc_out[(ri*32+ci)*ACC_W +: ACC_W] = acc_v[ri+1][ci];
                            valid_out[ri*32+ci]                = valid_v[ri+1][ci];
                        end
                    end
                end
            end
            // Bottom half: logical rows 0..7, logical cols 16..31 (physical rows 8..15, cols 0..15)
            for (ri = 0; ri < 8; ri = ri+1) begin
                for (ci = 0; ci < 16; ci = ci+1) begin
                    if (((ri + 8) < PHY_ROWS) && (ci < PHY_COLS)) begin
                        if (use_router_mesh) begin
                            acc_out[(ri*32 + ci + 16)*ACC_W +: ACC_W] =
                                router_acc_flat[((ri+8)*PHY_COLS+ci)*ACC_W +: ACC_W];
                            valid_out[ri*32 + ci + 16] = router_valid_flat[(ri+8)*PHY_COLS+ci];
                        end else begin
                            acc_out[(ri*32 + ci + 16)*ACC_W +: ACC_W] = acc_v[ri+9][ci];
                            valid_out[ri*32 + ci + 16]                = valid_v[ri+9][ci];
                        end
                    end
                end
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

// ---------------------------------------------------------------------------
// Diagnostic: log weight vertical chain per cycle
// ---------------------------------------------------------------------------
`ifdef DIAG_WCHAIN
reg [15:0] diag_wcyc;
reg        diag_wactive;
integer    diag_wr, diag_wc;

always @(posedge clk) begin
    if (!rst_n || en === 1'b0) begin
        diag_wcyc    <= 16'd0;
        diag_wactive <= 1'b0;
    end else if (!diag_wactive) begin
        diag_wactive <= 1'b1;
        diag_wcyc    <= 16'd0;
    end else if (diag_wcyc < 20) begin
        diag_wcyc <= diag_wcyc + 16'd1;
    end
end

always @(posedge clk) begin
    if (diag_wactive && diag_wcyc < 20) begin
        for (diag_wr = 0; diag_wr < 16; diag_wr = diag_wr + 1) begin
            if (w_v[diag_wr][0] != 0)
                $display("[DIAG_W] cyc=%0d w_v[%0d][0]=0x%08h",
                         diag_wcyc, diag_wr, w_v[diag_wr][0]);
        end
    end
end
`endif

// ---------------------------------------------------------------------------
// Diagnostic: probe PE(8,0) every clock cycle
// ---------------------------------------------------------------------------
`ifdef DIAG_8X32_PE
always @(posedge clk) begin
    if (en)  // log only when array-level en is active
        $display("[DIAG_PE80] t=%0t half=%0d pe_en=%0d s0a=0x%04h s0w=0x%04h s0v=%0d s1v=%0d s1m=%0d os=%0d acc=%0d",
                 $time, half_en,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.en,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.s0_a,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.s0_w,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.s0_valid,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.s1_valid,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.s1_mul,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.os_acc,
                  gen_direct_pe_grid.gen_row[8].gen_col[0].u_pe.acc_out);
end
`endif

endmodule
