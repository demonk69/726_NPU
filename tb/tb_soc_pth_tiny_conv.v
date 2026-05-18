// =============================================================================
// Module  : tb_soc_pth_tiny_conv
// Project : NPU_prj
// Desc    : SoC smoke for a tiny .pth-converted Conv2D+ReLU layer.
//
// The case assets are generated under sim/pth_tiny_conv by:
//   python tools/pth/gen_tiny_conv_soc_case.py --out-dir sim/pth_tiny_conv
// =============================================================================

`timescale 1ns/1ps

`include "soc_pth_tiny_conv_params.vh"

module tb_soc_pth_tiny_conv;

localparam CLK_PERIOD     = 10;
localparam TIMEOUT_CYCLES = 300000;
localparam MEM_WORDS      = 1024;
localparam DRAM_WORDS     = 15360;
localparam PASS_MARKER    = 32'h0000_00AA;
localparam FAIL_MARKER    = 32'h0000_00FF;
localparam FW_LAST_WORD   = `PTH_TINY_FW_WORDS - 1;

reg clk;
reg rst_n;
integer cycle_count;

soc_top #(
    .MEM_WORDS      (MEM_WORDS),
    .DRAM_WORDS     (DRAM_WORDS),
    .NPU_ROWS       (4),
    .NPU_COLS       (4),
    .NPU_DATA_W     (32),
    .NPU_ACC_W      (32),
    .NPU_PPB_DEPTH  (64),
    .NPU_PPB_THRESH (16)
) u_soc (
    .clk   (clk),
    .rst_n (rst_n)
);

initial clk = 0;
always #(CLK_PERIOD/2) clk = ~clk;

initial begin
    rst_n = 0;
    cycle_count = 0;
    #100;
    rst_n = 1;
end

initial begin
    $readmemh(`PTH_TINY_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST_WORD);
end

initial begin : init_dram
    integer i;
    for (i = 0; i < DRAM_WORDS; i = i + 1)
        u_soc.u_dram.mem[i] = 32'h0;
    $readmemh(`PTH_TINY_DRAM_HEX, u_soc.u_dram.mem);
end

always @(posedge clk) begin
    if (!rst_n)
        cycle_count <= 0;
    else
        cycle_count <= cycle_count + 1;
end

reg pass_seen;
reg fail_seen;

initial begin
    pass_seen = 0;
    fail_seen = 0;
end

always @(posedge clk) begin
    if (rst_n) begin
        if (u_soc.u_dram.mem[`PTH_TINY_MARKER_ADDR >> 2] == PASS_MARKER)
            pass_seen <= 1;
        if (u_soc.u_dram.mem[`PTH_TINY_MARKER_ADDR >> 2] == FAIL_MARKER)
            fail_seen <= 1;
    end
end

function [31:0] expected_word;
    input integer idx;
    begin
        case (idx)
            0: expected_word = `PTH_TINY_EXPECTED_0;
            1: expected_word = `PTH_TINY_EXPECTED_1;
            2: expected_word = `PTH_TINY_EXPECTED_2;
            3: expected_word = `PTH_TINY_EXPECTED_3;
            4: expected_word = `PTH_TINY_EXPECTED_4;
            5: expected_word = `PTH_TINY_EXPECTED_5;
            6: expected_word = `PTH_TINY_EXPECTED_6;
            7: expected_word = `PTH_TINY_EXPECTED_7;
            default: expected_word = 32'h0;
        endcase
    end
endfunction

function results_match;
    integer i;
    reg ok;
    begin
        ok = 1'b1;
        for (i = 0; i < `PTH_TINY_RESULT_COUNT; i = i + 1) begin
            if (u_soc.u_dram.mem[(`PTH_TINY_RESULT_BASE >> 2) + i] !== expected_word(i))
                ok = 1'b0;
        end
        results_match = ok;
    end
endfunction

task print_results;
    integer i;
    begin
        for (i = 0; i < `PTH_TINY_RESULT_COUNT; i = i + 1) begin
            $display("    R[%0d] = %0d expected %0d",
                     i,
                     $signed(u_soc.u_dram.mem[(`PTH_TINY_RESULT_BASE >> 2) + i]),
                     $signed(expected_word(i)));
        end
    end
endtask

initial begin
    wait (rst_n);

    fork
        begin : wait_done
            #100;
            forever begin
                if (pass_seen) begin
                    $display("");
                    $display("========================================");
                    if (results_match()) begin
                        $display("  [PASS] PTH tiny Conv SoC test PASSED!");
                    end else begin
                        $display("  [FAIL] PASS marker seen but result memory mismatched!");
                    end
                    $display("  Cycles: %0d", cycle_count);
                    print_results();
                    $display("========================================");
                    $finish;
                end
                if (fail_seen) begin
                    $display("");
                    $display("========================================");
                    $display("  [FAIL] PTH tiny Conv SoC firmware reported failure!");
                    $display("  Cycles: %0d", cycle_count);
                    print_results();
                    $display("========================================");
                    $finish;
                end
                #(CLK_PERIOD);
            end
        end
        begin : wait_timeout
            #(TIMEOUT_CYCLES * CLK_PERIOD);
            $display("");
            $display("========================================");
            $display("  [TIMEOUT] PTH tiny Conv SoC test exceeded %0d cycles", TIMEOUT_CYCLES);
            $display("  pass_seen=%0b fail_seen=%0b", pass_seen, fail_seen);
            print_results();
            $display("========================================");
            $finish;
        end
    join
end

initial begin
`ifdef DUMP_VCD
    $dumpfile("soc_pth_tiny_conv.vcd");
    $dumpvars(0, tb_soc_pth_tiny_conv);
`endif
end

endmodule
