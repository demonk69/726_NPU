// =============================================================================
// tb_soc_mc_shared_a - SoC test: shared A_WORK, per-core R_WORK
// =============================================================================
`timescale 1ns/1ps

module tb_soc_mc_shared_a;
    localparam CLK_T = 10, TIMEOUT = 20000, MEM_WORDS = 1024, DRAM_WORDS = 4096;
    localparam PASS_MARKER = 32'hAA, FAIL_MARKER = 32'hFF, MARKER_ADDR = 32'h1000;
    localparam FW_LAST = 129;

    reg clk, rst_n; integer cyc; reg pass_seen, fail_seen;

    soc_mc_top #(.MEM_WORDS(MEM_WORDS),.DRAM_WORDS(DRAM_WORDS),.NUM_CORES(2),
                 .NPU_PPB_DEPTH(8192),.NPU_PPB_THRESH(16)) u_soc (.clk(clk),.rst_n(rst_n));

    initial clk = 0; always #(CLK_T/2) clk = ~clk;

    initial begin
        integer i;
        rst_n = 0; cyc = 0; pass_seen = 0; fail_seen = 0;
        $display("[INIT] loading firmware"); $fflush();
        $readmemh("sim/mc_tests/mc_shared_a_fw.hex", u_soc.u_sram.mem);
        $display("[INIT] fw loaded"); $fflush();
        $display("[INIT] loading DRAM"); $fflush();
        for (i = 0; i < DRAM_WORDS; i = i + 1) u_soc.u_dram.mem[i] = 32'h0;
        $readmemh("sim/mc_tests/mc_shared_a_dram.hex", u_soc.u_dram.mem);
        $display("[INIT] ready"); $fflush();
        repeat(5) @(posedge clk);
        rst_n = 1;
    end

    always @(posedge clk) if (rst_n) cyc <= cyc + 1;

    always @(posedge clk)
        if (rst_n) begin
            if (u_soc.u_dram.mem[MARKER_ADDR >> 2] == PASS_MARKER) pass_seen <= 1;
            if (u_soc.u_dram.mem[MARKER_ADDR >> 2] == FAIL_MARKER) fail_seen <= 1;
        end

    initial begin
        wait(rst_n);
        forever begin
            #(CLK_T);
            if (pass_seen) begin
                $display("[PASS] tb_soc_mc_shared_a: shared A_WORK, per-core R_WORK verified  cycles=%0d", cyc);
                $finish;
            end
            if (fail_seen) begin
                $display("[FAIL] tb_soc_mc_shared_a: firmware reported failure");
                $finish;
            end
        end
    end

    initial begin #(CLK_T * TIMEOUT); $display("[FAIL] tb_soc_mc_shared_a timeout at cycle %0d", cyc); $finish; end
endmodule
