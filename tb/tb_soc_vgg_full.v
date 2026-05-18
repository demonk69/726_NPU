// tb_soc_vgg_full.v — RepOpt VGG Layer-0 SoC test
`timescale 1ns/1ps
`include "soc_vgg_params.vh"

module tb_soc_vgg_full;
    localparam CLK_T = 10;
    localparam TIMEOUT = `VGG_TIMEOUT_CYCLES;
    localparam DRAM_W = `VGG_DRAM_WORDS;
    localparam FW_LAST = `VGG_FW_WORDS - 1;

    reg clk, rst_n, pass_seen, fail_seen;
    integer cyc, i;

    soc_top #(
        .MEM_WORDS(1024), .DRAM_WORDS(DRAM_W),
        .NPU_ROWS(4), .NPU_COLS(4), .NPU_DATA_W(32), .NPU_ACC_W(32),
        .NPU_PPB_DEPTH(64), .NPU_PPB_THRESH(16)
    ) u_soc (.clk(clk), .rst_n(rst_n));

    initial clk=0; always #(CLK_T/2) clk=~clk;
    initial begin rst_n=0; cyc=0; pass_seen=0; fail_seen=0; #100; rst_n=1; end
    initial $readmemh(`VGG_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST);

    initial begin
        integer j;
        for (j=0; j<DRAM_W; j=j+1) u_soc.u_dram.mem[j]=32'h0;
        $readmemh(`VGG_DRAM_HEX, u_soc.u_dram.mem);
    end

    reg [31:0] exp [0:`VGG_RESULT_COUNT-1];
    initial $readmemh(`VGG_EXPECTED_HEX, exp);

    always @(posedge clk) begin
        if (rst_n) begin
            cyc<=cyc+1;
            if (u_soc.u_dram.mem[`VGG_MARKER_ADDR>>2]==32'hAA) pass_seen<=1;
            if (u_soc.u_dram.mem[`VGG_MARKER_ADDR>>2]==32'hFF) fail_seen<=1;
        end
    end

    function match;
        integer k; reg ok;
        begin ok=1'b1;
            for (k=0; k<`VGG_RESULT_COUNT; k=k+1)
                if (u_soc.u_dram.mem[(`VGG_R_ADDR>>2)+k]!==exp[k]) ok=1'b0;
            match=ok;
        end
    endfunction

    initial begin
        wait(rst_n);
        fork
            begin : wait_done
                #100;
                forever begin
                    if (pass_seen) begin
                        if (match()) $display("[PASS] RepOpt VGG Layer-0 SoC PASS");
                        else begin
                            $display("[FAIL] Result mismatch:");
                            for (i=0; i<`VGG_RESULT_COUNT; i=i+1)
                                if (u_soc.u_dram.mem[(`VGG_R_ADDR>>2)+i]!==exp[i])
                                    $display("  R[%0d]=%0d exp=%0d", i,
                                        $signed(u_soc.u_dram.mem[(`VGG_R_ADDR>>2)+i]),
                                        $signed(exp[i]));
                        end
                        $display("  Cycles: %0d", cyc); $finish;
                    end
                    if (fail_seen) begin $display("[FAIL] Firmware reported failure. cyc=%0d",cyc); $finish; end
                    #(CLK_T);
                end
            end
            begin
                #(TIMEOUT*CLK_T);
                $display("[TIMEOUT] %0d cycles", TIMEOUT); $finish;
            end
        join
    end
endmodule
