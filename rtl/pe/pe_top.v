// =============================================================================
// Module  : pe_top
// Project : NPU_prj  -- Scalable PE Array
// Desc    : Processing Element (PE) with dual weight register for prefetch hiding.
//
//           Architecture:
//           - Dual Weight Registers: w_reg[0] (active) / w_reg[1] (prefetch)
//             swap_w signal atomically switches which register is used for compute.
//             While computing with w_reg[0], new weights can be loaded into w_reg[1],
//             then swapped in one cycle — hides the reload latency.
//           - 3-stage pipeline: Stage0=Input-Reg, Stage1=MUL, Stage2=ACC/Output
//
//           WS mode: load_w=1 latches w_in into the PREFETCH weight register.
//                    swap_w atomically swaps active/prefetch registers.
//           OS mode: weight streams through like activation.
//           FP16/INT8 support with mixed-precision FP32 accumulation.
// =============================================================================

`timescale 1ns/1ps

module pe_top #(
    parameter DATA_W = 16,   // max data width (FP16)
    parameter ACC_W  = 32    // accumulator width
)(
    input  wire              clk,
    input  wire              rst_n,
    // mode control
    input  wire              mode,      // 0=INT8, 1=FP16
    input  wire              stat_mode, // 0=Weight-Stationary, 1=Output-Stationary
    input  wire              en,        // pipeline enable
    input  wire              flush,     // flush accumulator
    input  wire              load_w,    // WS: latch w_in into PREFETCH weight reg this cycle
    input  wire              swap_w,    // WS: atomic swap of active↔prefetch weight regs
    // data
    input  wire [DATA_W-1:0] w_in,      // weight input
    input  wire [DATA_W-1:0] a_in,      // activation input
    input  wire [ACC_W-1:0]  acc_in,    // incoming partial sum (from above PE)
    output reg  [ACC_W-1:0]  acc_out,   // accumulated result output
    output reg               valid_out
);

// ---------------------------------------------------------------------------
// Dual Weight Register Bank for Prefetch Hiding
//   w_reg[0]: ACTIVE   — currently used by the datapath for multiply
//   w_reg[1]: PREFETCH — being loaded; will become active on next swap_w
//   w_sel: selects which register is currently active (0 or 1)
// ---------------------------------------------------------------------------

reg [DATA_W-1:0] w_reg [0:1];       // dual weight registers
reg              w_sel;              // which register is active

always @(posedge clk) begin
    if (!rst_n) begin
        w_reg[0] <= {DATA_W{1'b0}};
        w_reg[1] <= {DATA_W{1'b0}};
        w_sel   <= 1'b0;
    end else begin
        // Atomic swap on swap_w pulse
        if (swap_w)
            w_sel <= ~w_sel;
        // Load into the INACTIVE (prefetch) register for background prefetch
        // Also update ACTIVE register so load_w is immediately effective
        // (backward compatible: load_w makes weight available this cycle)
        if (load_w) begin
            w_reg[~w_sel] <= w_in;      // preload into inactive
            w_reg[w_sel]  <= w_in;       // also update active for immediate use
        end
    end
end

// The active weight value presented to Stage-0
wire [DATA_W-1:0] active_weight = w_reg[w_sel];

// ---------------------------------------------------------------------------
// Stage-0 : Input register
// ---------------------------------------------------------------------------
reg  [DATA_W-1:0] s0_w, s0_a;
reg               s0_valid;
reg               s0_flush;
reg               s0_mode;
reg               s0_stat;

always @(posedge clk) begin
    if (!rst_n) begin
        s0_w     <= 0;
        s0_a     <= 0;
        s0_valid <= 0;
        s0_flush <= 0;
        s0_mode  <= 0;
        s0_stat  <= 0;
    end else if (en) begin
        s0_mode  <= mode;
        s0_stat <= stat_mode;
        s0_flush <= flush;
        s0_valid <= 1'b1;

        // WS mode: use active weight from dual-reg bank
        // On load_w cycle, use w_in directly (same-cycle availability)
        // On subsequent beats, use the latched active_weight
        if (stat_mode == 1'b0) begin
            s0_w <= load_w ? w_in : active_weight;  // bypass on load cycle
            s0_a <= a_in;
        end
        // OS mode: weight and activation both stream in
        else begin
            s0_w <= w_in;
            s0_a <= a_in;
        end
    end else begin
        s0_valid <= 1'b0;
        s0_flush <= flush;
    end
end

// ---------------------------------------------------------------------------
// Stage-1: Multiply  (INT8 or FP16)
// ---------------------------------------------------------------------------
reg  [ACC_W-1:0]  s1_mul;
reg               s1_valid;
reg               s1_flush;
reg               s1_stat;
reg               s1_mode;
reg  [ACC_W-1:0]  s1_acc_in;

// INT8: signed 8-bit multiply -> 16-bit sign-extended to ACC_W
wire signed [7:0] int8_w = s0_w[7:0];
wire signed [7:0] int8_a = s0_a[7:0];
wire signed [15:0] int8_mul_16 = $signed(int8_w) * $signed(int8_a);
wire [ACC_W-1:0] int8_prod = {{(ACC_W-16){int8_mul_16[15]}}, int8_mul_16};

// FP16: instantiate fp16_mul (combinational)
wire [ACC_W-1:0] fp16_mul_out;
fp16_mul u_fp16_mul (
    .clk     (clk),
    .rst_n   (rst_n),
    .en      (en),
    .a       (s0_w),
    .b       (s0_a),
    .result  (fp16_mul_out)
);

// Mux INT8 / FP16 + register
always @(posedge clk) begin
    if (!rst_n) begin
        s1_mul    <= 0;
        s1_valid  <= 0;
        s1_flush  <= 0;
        s1_stat   <= 0;
        s1_mode   <= 0;
        s1_acc_in <= 0;
    end else if (s0_valid) begin
        s1_valid  <= s0_valid;
        s1_flush  <= s0_flush;
        s1_stat   <= s0_stat;
        s1_mode   <= s0_mode;
        s1_acc_in <= acc_in;
        s1_mul    <= s0_mode ? fp16_mul_out : int8_prod;
    end else begin
        s1_valid <= 0;
    end
end

// ---------------------------------------------------------------------------
// Stage-2: Accumulate / output
//
// WS (Weight-Stationary):
//   Internal ws_acc accumulates products across K beats.
//   flush=1 outputs the accumulated sum and clears ws_acc.
//   FP16 uses fp32_add (mixed-precision); INT8 uses signed integer add.
//
// OS (Output-Stationary):
//   Internal os_acc accumulates products. flush=1 outputs and clears.
// ---------------------------------------------------------------------------

reg [ACC_W-1:0]  os_acc;      // Output-Stationary accumulator
reg [ACC_W-1:0]  ws_acc;      // Weight-Stationary accumulator

// --- FP16 to FP32 conversion ---
function [31:0] fp16_to_fp32;
    input [15:0] fp16;
    reg [4:0] exp16;
    reg is_zero, is_inf, is_nan;
    begin
        exp16  = fp16[14:10];
        is_zero = (exp16 == 5'd0) && (fp16[9:0] == 10'd0);
        is_inf  = (exp16 == 5'h1F) && (fp16[9:0] == 10'd0);
        is_nan  = (exp16 == 5'h1F) && (fp16[9:0] != 10'd0);
        if (is_zero)
            fp16_to_fp32 = {fp16[15], 31'd0};
        else if (is_inf)
            fp16_to_fp32 = {fp16[15], 8'hFF, 23'd0};
        else if (is_nan)
            fp16_to_fp32 = {1'b1, 8'hFF, 23'h400000};
        else if (exp16 == 5'd0)
            fp16_to_fp32 = {fp16[15], 8'd113, fp16[9:0], 13'd0};
        else
            fp16_to_fp32 = {fp16[15],
                            8'd127 + {3'b0, exp16} - 8'd15,
                            fp16[9:0], 13'd0};
    end
endfunction

// FP32 adders
wire [31:0] fp32_a = s1_stat ? os_acc : ws_acc;
wire [31:0] fp32_b = fp16_to_fp32(s1_mul[15:0]);
wire [31:0] fp32_sum;

fp32_add u_fp32_add (
    .a      (fp32_a),
    .b      (fp32_b),
    .result (fp32_sum)
);

wire [31:0] fp32_ws_ext_sum;
fp32_add u_fp32_add_ws_ext (
    .a      (s1_acc_in),
    .b      (fp32_b),
    .result (fp32_ws_ext_sum)
);

always @(posedge clk) begin
    if (!rst_n) begin
        os_acc    <= 0;
        ws_acc    <= 0;
        acc_out   <= 0;
        valid_out <= 0;
    end else begin
        valid_out <= 0;   // default

        if (s1_valid) begin
            if (s1_stat == 1'b0) begin
                // ----- Weight-Stationary -----
                if (s1_flush) begin
                    if (s1_mode) begin
                        acc_out <= ws_acc;
                        ws_acc  <= 32'd0;
                    end else begin
                        acc_out <= ws_acc;
                        ws_acc  <= 32'd0;
                    end
                    valid_out <= 1'b1;
                end else begin
                    if (s1_mode) begin
                        ws_acc <= fp32_sum;
                    end else begin
                        ws_acc <= $signed(ws_acc) + $signed(s1_mul);
                    end
                    valid_out <= 1'b0;
                end
            end else begin
                // ----- Output-Stationary -----
                if (s1_flush) begin
                    if (s1_mode) begin
                        acc_out <= fp32_sum;
                    end else begin
                        acc_out <= $signed(os_acc) + $signed(s1_mul);
                    end
                    os_acc    <= 32'd0;
                    valid_out <= 1'b1;
                end else begin
                    if (s1_mode) begin
                        os_acc <= fp32_sum;
                    end else begin
                        os_acc <= $signed(os_acc) + $signed(s1_mul);
                    end
                end
            end
        end
    end
end

endmodule
