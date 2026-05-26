// tb_soc_vgg_closed_loop.v - runtime closed-loop RepOpt VGG test
`timescale 1ns/1ps
`include "soc_vgg_closed_loop_params.vh"

module tb_soc_vgg_closed_loop;
    localparam CLK_T = 10;
    localparam TIMEOUT = `VGG_CLOSED_TIMEOUT_CYCLES;
    localparam [63:0] CLK_T_TICKS = CLK_T;
    localparam [63:0] TIMEOUT_CYCLES = TIMEOUT;
    localparam [63:0] TIMEOUT_TICKS = TIMEOUT_CYCLES * CLK_T_TICKS;
    localparam DRAM_W = `VGG_CLOSED_DRAM_WORDS;
    localparam FW_LAST = `VGG_CLOSED_FW_WORDS - 1;
    localparam MARKER_OK = `VGG_CLOSED_LABEL + 32'h100;
    localparam MARKER_CLASS_MIN = 32'h100;
    localparam MARKER_CLASS_MAX = 32'h109;

    reg clk, rst_n, pass_seen, fail_seen;
    reg [31:0] last_marker;
    integer cyc, j;

    soc_top #(
        .MEM_WORDS(`VGG_CLOSED_MEM_WORDS), .DRAM_WORDS(DRAM_W),
        .NPU_ROWS(4), .NPU_COLS(4), .NPU_DATA_W(32), .NPU_ACC_W(32),
        .NPU_PPB_DEPTH(8192), .NPU_PPB_THRESH(16)
    ) u_soc (.clk(clk), .rst_n(rst_n));

    initial clk = 0;
    always #(CLK_T/2) clk = ~clk;

    initial begin
        rst_n = 0;
        cyc = 0;
        pass_seen = 0;
        fail_seen = 0;
        last_marker = 0;
        #100;
        rst_n = 1;
    end

    initial $readmemh(`VGG_CLOSED_FW_HEX, u_soc.u_sram.mem, 0, FW_LAST);

    initial begin
        $display("[INIT] clear DRAM words=%0d", DRAM_W);
        $fflush();
        for (j = 0; j < DRAM_W; j = j + 1) u_soc.u_dram.mem[j] = 32'h0;
        $display("[INIT] read DRAM hex");
        $fflush();
        $readmemh(`VGG_CLOSED_DRAM_HEX, u_soc.u_dram.mem);
        $display("[INIT] DRAM ready");
        $fflush();
    end

    always @(posedge clk) begin
        if (rst_n) begin
            cyc <= cyc + 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] == MARKER_OK)
                pass_seen <= 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] == 32'h000000FF)
                fail_seen <= 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] >= MARKER_CLASS_MIN &&
                u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] <= MARKER_CLASS_MAX &&
                u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] != MARKER_OK)
                fail_seen <= 1;
            if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] != last_marker) begin
                last_marker <= u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2];
                if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] >= 32'h200)
                    $display("[PROGRESS] marker=0x%08h cycles=%0d",
                             u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2], cyc);
                $fflush();
            end
        end
    end

    initial begin
        wait(rst_n);
        fork
            begin : wait_done
                #100;
                forever begin
                    if (pass_seen) begin
                        $display("[PASS] Runtime closed-loop VGG classification PASSED");
                        $display("  Predicted class: %0d (expected: %0d, exact-python: %0d)",
                                 `VGG_CLOSED_LABEL, `VGG_CLOSED_LABEL, `VGG_CLOSED_EXACT_LABEL);
                        $display("  Cycles: %0d", cyc);
                        $fflush();
                        $finish;
                    end
                    if (fail_seen) begin
                        if (u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] == 32'h000000FF) begin
                            $display("[FAIL] Runtime closed-loop firmware failure");
                        end else begin
                            $display("[FAIL] Runtime closed-loop classification mismatch");
                            $display("  Predicted class: %0d (expected: %0d, exact-python: %0d)",
                                     u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2] - 32'h100,
                                     `VGG_CLOSED_LABEL, `VGG_CLOSED_EXACT_LABEL);
                        end
                        $display("  Cycles: %0d", cyc);
                        $fflush();
                        $finish;
                    end
                    #(CLK_T);
                end
            end
            begin
                #(TIMEOUT_TICKS);
                $display("[TIMEOUT] %0d cycles", TIMEOUT);
                $display("  marker=0x%08h pc=0x%08h trap=%0d mem_valid=%0d mem_addr=0x%08h",
                         u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR >> 2],
                         u_soc.u_cpu.reg_pc, u_soc.u_cpu.trap,
                         u_soc.mem_valid, u_soc.mem_addr);
                $display("  npu busy=%0d done=%0d error=%0d err=0x%08h state=%0d tile_i=%0d tile_j=%0d k_tile=%0d",
                         u_soc.u_npu.status_busy, u_soc.u_npu.status_done,
                         u_soc.u_npu.status_error, u_soc.u_npu.err_status,
                         u_soc.u_npu.u_ctrl.state, u_soc.u_npu.u_ctrl.tile_i,
                         u_soc.u_npu.u_ctrl.tile_j, u_soc.u_npu.u_ctrl.k_tile_idx);
                $fflush();
                $finish;
            end
        join
    end
endmodule
