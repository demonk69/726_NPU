// tb_soc_multilayer_tile.v — Multi-layer tile-mode SoC closed-loop test
`timescale 1ns/1ps
`include "soc_multilayer_params.vh"

module tb_soc_multilayer_tile;
    localparam CLK_PERIOD     = 10;
    localparam TIMEOUT_CYCLES = `MULTILAYER_TIMEOUT_CYCLES;
    localparam MEM_WORDS      = 1024;
    localparam DRAM_WORDS     = `MULTILAYER_DRAM_WORDS;
    localparam PASS_MARKER    = 32'h0000_00AA;
    localparam FAIL_MARKER    = 32'h0000_00FF;
    localparam FW_LAST_WORD   = `MULTILAYER_FW_WORDS - 1;

    reg clk, rst_n;
    reg pass_seen, fail_seen;
    integer cycle_count, i;

    soc_top #(
        .MEM_WORDS     (MEM_WORDS),    .DRAM_WORDS    (DRAM_WORDS),
        .NPU_ROWS      (4),            .NPU_COLS      (4),
        .NPU_DATA_W    (32),           .NPU_ACC_W     (32),
        .NPU_PPB_DEPTH (64),           .NPU_PPB_THRESH(16)
    ) u_soc (.clk(clk), .rst_n(rst_n));

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    initial begin
        rst_n = 0; cycle_count = 0; pass_seen = 0; fail_seen = 0;
        #100; rst_n = 1;
    end

    initial $readmemh(`MULTILAYER_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST_WORD);

    initial begin
        integer j;
        for (j = 0; j < DRAM_WORDS; j = j + 1) u_soc.u_dram.mem[j] = 32'h0;
        $readmemh(`MULTILAYER_DRAM_HEX, u_soc.u_dram.mem);
    end

    reg [31:0] expected [0:`MULTILAYER_RESULT_COUNT-1];
    initial $readmemh(`MULTILAYER_EXPECTED_HEX, expected);

    always @(posedge clk) begin
        if (rst_n) begin
            cycle_count <= cycle_count + 1;
            if (u_soc.u_dram.mem[`MULTILAYER_MARKER_ADDR >> 2] == PASS_MARKER)
                pass_seen <= 1;
            if (u_soc.u_dram.mem[`MULTILAYER_MARKER_ADDR >> 2] == FAIL_MARKER)
                fail_seen <= 1;
        end
    end

    function results_match;
        integer k; reg ok;
        begin
            ok = 1'b1;
            for (k = 0; k < `MULTILAYER_RESULT_COUNT; k = k + 1)
                if (u_soc.u_dram.mem[(`MULTILAYER_R_ADDR >> 2) + k] !== expected[k])
                    ok = 1'b0;
            results_match = ok;
        end
    endfunction

    initial begin
        wait(rst_n);
        fork
            begin : wait_done
                #100;
                forever begin
                    if (pass_seen) begin
                        if (results_match())
                            $display("[PASS] Multi-layer tile SoC test PASSED");
                        else begin
                            $display("[FAIL] PASS marker seen but result mismatched");
                            for (i = 0; i < `MULTILAYER_RESULT_COUNT; i = i + 1)
                                if (u_soc.u_dram.mem[(`MULTILAYER_R_ADDR >> 2) + i] !== expected[i]) begin
                                    $display("  R[%0d]=%0d exp=%0d", i,
                                        $signed(u_soc.u_dram.mem[(`MULTILAYER_R_ADDR >> 2) + i]),
                                        $signed(expected[i]));
                                    if (i > 4) break;
                                end
                        end
                        $display("  Cycles: %0d", cycle_count);
                        $finish;
                    end
                    if (fail_seen) begin
                        $display("[FAIL] Multi-layer firmware reported failure. Cycles: %0d", cycle_count);
                        $finish;
                    end
                    #(CLK_PERIOD);
                end
            end
            begin : wait_timeout
                #(TIMEOUT_CYCLES * CLK_PERIOD);
                $display("[TIMEOUT] Multi-layer test exceeded %0d cycles", TIMEOUT_CYCLES);
                $finish;
            end
        join
    end
endmodule
