// =============================================================================
// Module  : pe_top
// Project : NPU_prj  -- Scalable PE Array
// Author  : Auto-generated
// Date    : 2026-04-03
// Desc    : Processing Element (PE) top module.
//           Supports Weight-Stationary (WS) and Output-Stationary (OS) modes.
//           Supports FP16 and INT8 multiply-accumulate.
//           3-stage pipeline: Stage0=Input-Reg, Stage1=MUL, Stage2=ACC/Output
//
//           WS mode: load_w=1 latches weight into PE; subsequent beats reuse
//                    the stored weight. FP16 accumulation uses fp16_add.
//           OS mode: internal os_acc accumulates products. FP16 uses fp16_add.
//           FP16 values are ALWAYS zero-extended (never sign-extended).
// =============================================================================
//
// Port Summary:
//   clk        - clock
//   rst_n      - active-low synchronous reset
//   mode       - data type: 0=INT8, 1=FP16
//   stat_mode  - stationary: 0=Weight-Stationary, 1=Output-Stationary
//   en         - pipeline enable
//   flush      - flush accumulator / output registers
//   load_w     - (WS only) latch w_in into weight register this cycle
//   w_in       - weight input  (16-bit; INT8 uses [7:0])
//   a_in       - activation input (16-bit; INT8 uses [7:0])
//   acc_in     - partial sum passed in (32-bit)
//   acc_out    - accumulated result output (32-bit)
//   valid_out  - output valid
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
    input  wire              load_w,    // WS: latch weight this cycle
    // data
    input  wire [DATA_W-1:0] w_in,      // weight
    input  wire [DATA_W-1:0] a_in,      // activation
    input  wire [ACC_W-1:0]  acc_in,    // incoming partial sum
    output reg  [ACC_W-1:0]  acc_out,   // result
    output reg               valid_out
);

// ---------------------------------------------------------------------------
// Weight register — true Weight-Stationary storage
// ---------------------------------------------------------------------------
reg [DATA_W-1:0] weight_reg;  // latched weight for WS mode

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
        weight_reg <= 0;
    end else if (en || flush) begin
        s0_mode  <= mode;
        s0_stat  <= stat_mode;
        s0_flush <= flush;
        // flush=1 must propagate even when PPBuf is empty (en=1 but no data).
        // Treat flush beat as valid so Stage-1 sees s0_flush and propagates it.
        s0_valid <= en ? 1'b1 : 1'b0;

        if (en) begin
            // WS mode: latch weight when load_w=1, otherwise use stored weight
            if (stat_mode == 1'b0) begin
                s0_w <= load_w ? w_in : weight_reg;
                s0_a <= a_in;
            end
            // OS mode: weight and activation both stream in
            else begin
                s0_w <= w_in;
                s0_a <= a_in;
            end

            // Update weight register
            if (stat_mode == 1'b0 && load_w) begin
                weight_reg <= w_in;
            end
        end
    end else begin
        s0_valid <= 1'b0;
        s0_flush <= 1'b0;
    end
end

// ---------------------------------------------------------------------------
// Stage-1: Multiply  (INT8 or FP16)
// ---------------------------------------------------------------------------
reg  [ACC_W-1:0]  s1_mul;
reg               s1_valid;
reg               s1_flush;
reg               s1_stat;
reg               s1_mode;       // data type for stage-2 (0=INT8, 1=FP16)
reg  [ACC_W-1:0]  s1_acc_in;

// INT8: signed 8-bit multiply -> 16-bit sign-extended to ACC_W
wire signed [7:0] int8_w = s0_w[7:0];
wire signed [7:0] int8_a = s0_a[7:0];
wire signed [15:0] int8_mul_16 = $signed(int8_w) * $signed(int8_a);
wire [ACC_W-1:0] int8_prod = {{(ACC_W-16){int8_mul_16[15]}}, int8_mul_16};

// FP16: instantiate fp16_mul (combinational, zero-extended output)
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
    end else if (s0_valid || s0_flush) begin
        // Propagate flush unconditionally (flush must pass through even when
        // PPBuf is empty and s0_valid=0 — otherwise OS mode acc never outputs).
        s1_flush  <= s0_flush;
        s1_stat   <= s0_stat;
        s1_mode   <= s0_mode;
        s1_acc_in <= acc_in;
        if (s0_valid) begin
            s1_valid <= 1'b1;
            s1_mul   <= s0_mode ? fp16_mul_out : int8_prod;
        end else begin
            // flush-only beat: s1_mul should keep the last product from previous cycle
            // so that flush logic can accumulate it before clearing
            s1_valid <= 1'b0;
            // s1_mul NOT updated - keep its previous value for flush accumulation
        end
    end else begin
        s1_valid <= 0;
        s1_flush <= 0;
    end
end

// ---------------------------------------------------------------------------
// Stage-2: Accumulate / output
//
// WS (Weight-Stationary):
//   Internal ws_acc accumulates products (same as OS for 1x1 PE).
//   flush=1 outputs accumulated result and clears.
//   FP16 uses fp16_add; INT8 uses signed integer add.
//
// OS (Output-Stationary):
//   Internal os_acc accumulates products. flush=1 outputs and clears.
//   FP16 uses fp16_add; INT8 uses signed integer add.
// ---------------------------------------------------------------------------

reg [ACC_W-1:0]  os_acc;      // Output-Stationary accumulator
reg [ACC_W-1:0]  ws_acc;      // Weight-Stationary accumulator (for 1x1 PE)

// --- FP16 to FP32 conversion (for mixed-precision accumulation) ---
// Correctly re-maps exponent bias from 15 (FP16) to 127 (FP32)
// and zero-extends mantissa from 10 bits to 23 bits.
// Handles special cases: zero, Inf, NaN.
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
            fp16_to_fp32 = {fp16[15], 31'd0};                     // preserve sign of zero
        else if (is_inf)
            fp16_to_fp32 = {fp16[15], 8'hFF, 23'd0};             // Inf
        else if (is_nan)
            fp16_to_fp32 = {1'b1, 8'hFF, 23'h400000};            // quiet NaN
        else if (exp16 == 5'd0)                                    // subnormal
            fp16_to_fp32 = {fp16[15], 8'd113, fp16[9:0], 13'd0}; // exp=1-15+127=113
        else
            fp16_to_fp32 = {fp16[15],                              // sign
                            8'd127 + {3'b0, exp16} - 8'd15,        // re-bias exponent
                            fp16[9:0], 13'd0};                     // extend mantissa
    end
endfunction

// --- FP32 adder for FP16 mixed-precision accumulation ---
// OS path: os_acc (already FP32) + FP16 mul product (converted to FP32)
// WS path: acc_in (already FP32) + FP16 mul product (converted to FP32)
// For INT8 WS: acc_in (INT32) + INT8 product (INT32) - handled separately
wire [31:0] fp32_a = s1_stat ? os_acc : s1_acc_in;  // WS uses acc_in, not ws_acc
wire [31:0] fp32_b = fp16_to_fp32(s1_mul[15:0]);
wire [31:0] fp32_sum;

fp32_add u_fp32_add (
    .a      (fp32_a),
    .b      (fp32_b),
    .result (fp32_sum)
);

always @(posedge clk) begin
    if (!rst_n) begin
        os_acc    <= 0;
        ws_acc    <= 0;
        acc_out   <= 0;
        valid_out <= 0;
    end else begin
        valid_out <= 0;   // default

        if (s1_valid || s1_flush) begin
            if (s1_stat == 1'b0) begin
                // ----- Weight-Stationary mode -----
                // In WS mode, the PE should output acc_in + product
                // This allows partial sums to flow through the PE array
                // For 1x1 PE, acc_in comes from external accumulator (testbench)
                // For multi-column PE array, acc_in comes from left PE's acc_out
                if (s1_flush) begin
                    // Flush: output acc_in + product, then reset ws_acc
                    if (s1_mode) begin
                        // FP16: output acc_in + product
                        acc_out <= fp32_sum;  // fp32_sum = s1_acc_in + fp16_to_fp32(s1_mul[15:0])
                    end else begin
                        // INT8: output acc_in + product
                        acc_out <= $signed(s1_acc_in) + $signed(s1_mul);
                    end
                    ws_acc    <= 32'd0;
                    valid_out <= 1'b1;
                end else if (s1_valid) begin
                    // Normal operation: always output acc_in + product
                    // This is the key difference from OS mode
                    if (s1_mode) begin
                        // FP16: output acc_in + product
                        acc_out <= fp32_sum;  // fp32_sum = s1_acc_in + fp16_to_fp32(s1_mul[15:0])
                        ws_acc <= fp32_sum;   // For internal tracking if needed
                    end else begin
                        // INT8: output acc_in + product
                        acc_out <= $signed(s1_acc_in) + $signed(s1_mul);
                        ws_acc <= $signed(s1_acc_in) + $signed(s1_mul);
                    end
                end
            end else begin
                // ----- Output-Stationary -----
                if (s1_flush) begin
                    // Flush: accumulate current product into os_acc, output, then clear.
                    // This avoids the "data black hole" where s1_mul would be lost.
                    if (s1_mode) begin
                        // FP16: FP32 mixed-precision, output then clear
                        acc_out <= fp32_sum;
                    end else begin
                        // INT8: add s1_mul to os_acc, output, clear
                        acc_out <= $signed(os_acc) + $signed(s1_mul);
                    end
                    os_acc    <= 32'd0;
                    valid_out <= 1'b1;
                end else if (s1_valid) begin
                    // Normal accumulation (no flush)
                    if (s1_mode) begin
                        // FP16: FP32 mixed-precision accumulation
                        os_acc <= fp32_sum;
                    end else begin
                        // INT8: signed integer accumulation
                        os_acc <= $signed(os_acc) + $signed(s1_mul);
                    end
                end
            end
        end
    end
end

endmodule
