// =============================================================================
// Module  : pe_top
// Project : NPU_prj  -- Scalable PE Array
// Author  : Auto-generated
// Date    : 2026-04-01
// Desc    : Processing Element (PE) top module.
//           Supports Weight-Stationary (WS) and Output-Stationary (OS) modes.
//           Supports FP16 and INT8 multiply-accumulate.
//           3-stage pipeline: Stage0=Input-Reg, Stage1=MUL, Stage2=ACC/Output
// =============================================================================
//
// Port Summary:
//   clk        - clock
//   rst_n      - active-low synchronous reset
//   mode       - data type: 0=INT8, 1=FP16
//   stat_mode  - stationary: 0=Weight-Stationary, 1=Output-Stationary
//   en         - pipeline enable
//   flush      - flush accumulator / output registers
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
    // data
    input  wire [DATA_W-1:0] w_in,      // weight
    input  wire [DATA_W-1:0] a_in,      // activation
    input  wire [ACC_W-1:0]  acc_in,    // incoming partial sum (OS mode)
    output reg  [ACC_W-1:0]  acc_out,   // result
    output reg               valid_out
);

// ---------------------------------------------------------------------------
// Internal wires
// ---------------------------------------------------------------------------
wire [ACC_W-1:0] mac_result;   // output from MAC unit
wire             mac_valid;

// Weight-Stationary: weight is registered once; Output-Stationary: activation pass-through
reg  [DATA_W-1:0] weight_reg;  // stored weight (WS mode)
reg  [DATA_W-1:0] act_reg;     // registered activation

// Stage-0 : Input register
reg  [DATA_W-1:0] s0_w, s0_a;
reg               s0_valid;
reg               s0_flush;
reg               s0_mode;
reg               s0_stat;

// Stage-1 : Multiplier result
reg  [ACC_W-1:0]  s1_mul;
reg               s1_valid;
reg               s1_flush;
reg               s1_stat;
reg  [ACC_W-1:0]  s1_acc_in;

// Stage-2 : Accumulator (OS mode internal acc, WS forwards acc_in)
reg  [ACC_W-1:0]  os_acc;      // Output-Stationary accumulator

// ---------------------------------------------------------------------------
// Stage-0: Latch inputs, handle WS weight storage
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        s0_w     <= 0;
        s0_a     <= 0;
        s0_valid <= 0;
        s0_flush <= 0;
        s0_mode  <= 0;
        s0_stat  <= 0;
        weight_reg <= 0;
    end else if (en) begin
        s0_mode  <= mode;
        s0_stat  <= stat_mode;
        s0_flush <= flush;
        s0_valid <= 1'b1;
        // Weight-Stationary: capture weight on flush (load phase)
        if (stat_mode == 1'b0) begin
            if (flush)
                weight_reg <= w_in;
            s0_w <= weight_reg;
            s0_a <= a_in;
        end
        // Output-Stationary: weight streams in, activation is stored or streamed
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
wire [ACC_W-1:0] int8_prod;
wire [ACC_W-1:0] fp16_prod;

// INT8: signed 8-bit multiply -> 16-bit sign-extended to ACC_W
wire signed [7:0] int8_w = s0_w[7:0];
wire signed [7:0] int8_a = s0_a[7:0];
assign int8_prod = {{(ACC_W-16){int8_w[7] ^ int8_a[7]}},
                    $signed(int8_w) * $signed(int8_a)};

// FP16: instantiate fp16 MAC
wire [ACC_W-1:0] fp16_mul_out;
fp16_mul u_fp16_mul (
    .clk     (clk),
    .rst_n   (rst_n),
    .en      (en),
    .a       (s0_w),
    .b       (s0_a),
    .result  (fp16_mul_out)
);

// Mux INT8 / FP16
always @(posedge clk) begin
    if (!rst_n) begin
        s1_mul    <= 0;
        s1_valid  <= 0;
        s1_flush  <= 0;
        s1_stat   <= 0;
        s1_acc_in <= 0;
    end else begin
        s1_valid  <= s0_valid;
        s1_flush  <= s0_flush;
        s1_stat   <= s0_stat;
        s1_acc_in <= acc_in;   // carry through partial sum
        s1_mul    <= s0_mode ? fp16_mul_out : int8_prod;
    end
end

// ---------------------------------------------------------------------------
// Stage-2: Accumulate / output
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (!rst_n) begin
        os_acc    <= 0;
        acc_out   <= 0;
        valid_out <= 0;
    end else begin
        valid_out <= s1_valid;
        if (s1_valid) begin
            if (s1_stat == 1'b0) begin
                // Weight-Stationary: acc_out = s1_acc_in + product
                if (s1_flush)
                    acc_out <= s1_mul;
                else
                    acc_out <= s1_acc_in + s1_mul;
            end else begin
                // Output-Stationary: internal accumulator
                if (s1_flush) begin
                    acc_out <= os_acc;   // output accumulated result
                    os_acc  <= s1_mul;   // start new accumulation
                end else begin
                    os_acc  <= os_acc + s1_mul;
                    acc_out <= os_acc;   // pass previous value downstream
                end
            end
        end
    end
end

endmodule
