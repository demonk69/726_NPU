// tb_soc_vgg_e2e.v — RepOpt VGG end-to-end classification test
`timescale 1ns/1ps
`include "soc_vgg_params.vh"

module tb_soc_vgg_e2e;
    localparam CLK_T = 10;
    localparam TIMEOUT = `VGG_TIMEOUT_CYCLES;
    localparam DRAM_W = `VGG_DRAM_WORDS;
    localparam FW_LAST = `VGG_FW_WORDS - 1;
    localparam MARKER_OK = `VGG_LABEL + 32'h100;

    reg clk, rst_n, pass_seen, fail_seen;
    integer cyc, i;

    soc_top #(
        .MEM_WORDS(1024), .DRAM_WORDS(DRAM_W),
        .NPU_ROWS(4), .NPU_COLS(4), .NPU_DATA_W(32), .NPU_ACC_W(32),
        .NPU_PPB_DEPTH(1024), .NPU_PPB_THRESH(16)
    ) u_soc (.clk(clk), .rst_n(rst_n));

    initial clk=0; always #(CLK_T/2) clk=~clk;
    initial begin rst_n=0; cyc=0; pass_seen=0; fail_seen=0; #100; rst_n=1; end
    initial $readmemh(`VGG_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST);

    initial begin
        integer j;
        for (j=0; j<DRAM_W; j=j+1) u_soc.u_dram.mem[j]=32'h0;
        $readmemh(`VGG_DRAM_HEX, u_soc.u_dram.mem);
    end

    always @(posedge clk) begin
        if (rst_n) begin
            cyc<=cyc+1;
            if (u_soc.u_dram.mem[`VGG_MARKER_ADDR>>2] == MARKER_OK) pass_seen<=1;
            if (u_soc.u_dram.mem[`VGG_MARKER_ADDR>>2] == 32'h000000FF) fail_seen<=1;
        end
    end

    initial begin
        wait(rst_n);
        fork
            begin : wait_done
                #100;
                forever begin
                    if (pass_seen) begin
                        $display("[PASS] RepOpt VGG end-to-end classification PASSED");
                        $display("  Predicted class: %0d (PyTorch label: %0d)", `VGG_LABEL, `VGG_LABEL);
                        $display("  Cycles: %0d", cyc);
                        // Also check L1 raw MAC
                        for (i=0; i<`VGG_RESULT_COUNT; i=i+1)
                            if (u_soc.u_dram.mem[(`VGG_R_ADDR>>2)+i] !== 32'h0 && i<3)
                                $display("  L1 R[%0d]=%0d", i, $signed(u_soc.u_dram.mem[(`VGG_R_ADDR>>2)+i]));
                        $finish;
                    end
                    if (fail_seen) begin $display("[FAIL] Firmware failure"); $finish; end
                    #(CLK_T);
                end
            end
            begin #(TIMEOUT*CLK_T); $display("[TIMEOUT] %0d cycles", TIMEOUT); $finish; end
        join
    end
endmodule
