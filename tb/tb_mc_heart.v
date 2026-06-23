`timescale 1ns/1ps
`include "soc_vgg_closed_loop_params.vh"
`ifndef MC_HEARTBEAT_INTERVAL
`define MC_HEARTBEAT_INTERVAL 10000
`endif
module tb_mc_heart;
    localparam CLK_T = 10; localparam DRAM_W = `VGG_CLOSED_DRAM_WORDS;
    localparam MARKER_OK = `VGG_CLOSED_LABEL + 32'h100;
    localparam NC = `VGG_CLOSED_NUM_CORES;
    reg clk=0, rst_n=0, pass=0, fail=0; integer cyc=0; integer hb=0;
    always #(CLK_T/2) clk = ~clk;
    soc_mc_top #(.MEM_WORDS(`VGG_CLOSED_MEM_WORDS),.DRAM_WORDS(DRAM_W),.NUM_CORES(NC),
                  .NPU_PPB_DEPTH(8192),.NPU_PPB_THRESH(16)) u_soc (.clk(clk),.rst_n(rst_n));
    wire busy0 = u_soc.u_npu_mc.gen_cores[0].u_npu_core.u_ctrl.busy;
    wire busy1;
    generate
        if (NC > 1) begin : gen_busy1
            assign busy1 = u_soc.u_npu_mc.gen_cores[1].u_npu_core.u_ctrl.busy;
        end else begin : gen_no_busy1
            assign busy1 = 1'b0;
        end
    endgenerate
    initial begin $readmemh(`VGG_CLOSED_FW_HEX, u_soc.u_sram.mem, 0, `VGG_CLOSED_FW_WORDS-1); repeat(5) @(posedge clk); rst_n = 1; end
    initial begin integer i; for(i=0; i<DRAM_W; i=i+1) u_soc.u_dram.mem[i]=0; $readmemh(`VGG_CLOSED_DRAM_HEX, u_soc.u_dram.mem); end
    always @(posedge clk) if(rst_n) begin
        cyc<=cyc+1; hb<=hb+1;
        if(hb >= (`MC_HEARTBEAT_INTERVAL-1)) begin hb<=0; $display("[HB] cyc=%0d pc=0x%08h marker=0x%08h busy0=%b busy1=%b", cyc, u_soc.u_cpu.reg_pc, u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR>>2], busy0, busy1); end
        if(u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR>>2]==MARKER_OK) pass<=1;
        if(u_soc.u_dram.mem[`VGG_CLOSED_MARKER_ADDR>>2]==32'hFF) fail<=1;
    end
    initial begin wait(rst_n); #100; forever begin #(CLK_T); if(pass) begin $display("[PASS] %0d-core VGG cycles=%0d",NC,cyc); $finish; end if(fail) begin $display("[FAIL]"); $finish; end end end
    initial #(CLK_T*`VGG_CLOSED_TIMEOUT_CYCLES) begin $display("[FAIL] timeout %0d",cyc); $finish; end
endmodule
