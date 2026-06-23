// =============================================================================
// tb_dram_multi_port - unit test for shared multi-port DRAM simulation model
// =============================================================================
`timescale 1ns/1ps

module tb_dram_multi_port;
    localparam CLK_T = 10, WORDS = 1024, DATA_W = 32, NUM_CORES = 2, TIMEOUT = 2000;

    reg clk = 1'b0; always #(CLK_T/2) clk = ~clk;
    reg rst_n = 1'b0; initial begin repeat(4) @(posedge clk); rst_n = 1'b1; end
    reg cpu_valid, cpu_we; reg [3:0] cpu_wstrb; reg [31:0] cpu_addr, cpu_wdata; wire [31:0] cpu_rdata;
    wire cpu_ready;

    reg [63:0] axi_awaddr; reg [1:0] axi_awvalid; wire [1:0] axi_awready;
    reg [63:0] axi_wdata; reg [7:0] axi_wstrb; reg [1:0] axi_wlast, axi_wvalid;
    wire [1:0] axi_wready, axi_bvalid; wire [3:0] axi_bresp; reg [1:0] axi_bready;
    reg [63:0] axi_araddr; reg [15:0] axi_arlen; reg [1:0] axi_arvalid;
    wire [1:0] axi_arready, axi_rvalid, axi_rlast; wire [63:0] axi_rdata; wire [3:0] axi_rresp;
    reg [1:0] axi_rready;

    integer pass_cnt, fail_cnt; reg [31:0] rd_val;

    dram_multi_port #(.WORDS(WORDS), .DATA_W(DATA_W), .NUM_CORES(NUM_CORES)) dut (
        .clk(clk), .rst_n(rst_n),
        .cpu_valid(cpu_valid), .cpu_ready(cpu_ready),
        .cpu_we(cpu_we), .cpu_wstrb(cpu_wstrb), .cpu_addr(cpu_addr),
        .cpu_wdata(cpu_wdata), .cpu_rdata(cpu_rdata),
        .axi_awaddr(axi_awaddr), .axi_awvalid(axi_awvalid), .axi_awready(axi_awready),
        .axi_wdata(axi_wdata), .axi_wstrb(axi_wstrb), .axi_wlast(axi_wlast),
        .axi_wvalid(axi_wvalid), .axi_wready(axi_wready),
        .axi_bresp(axi_bresp), .axi_bvalid(axi_bvalid), .axi_bready(axi_bready),
        .axi_araddr(axi_araddr), .axi_arlen(axi_arlen),
        .axi_arvalid(axi_arvalid), .axi_arready(axi_arready),
        .axi_rdata(axi_rdata), .axi_rresp(axi_rresp),
        .axi_rvalid(axi_rvalid), .axi_rlast(axi_rlast), .axi_rready(axi_rready)
    );

    task pass;
        begin pass_cnt = pass_cnt + 1; end
    endtask
    task fail;
        input [1023:0] msg;
        begin $display("[FAIL] %0s", msg); fail_cnt = fail_cnt + 1; end
    endtask
    task check;
        input ok;
        input [1023:0] msg;
        begin if (ok) pass; else fail(msg); end
    endtask

    initial begin
        pass_cnt = 0; fail_cnt = 0;
        cpu_valid = 0; cpu_we = 0; cpu_wstrb = 0; cpu_addr = 0; cpu_wdata = 0;
        axi_awaddr = 0; axi_awvalid = 0; axi_wdata = 0; axi_wstrb = 0;
        axi_wlast = 0; axi_wvalid = 0; axi_bready = 0;
        axi_araddr = 0; axi_arlen = 0; axi_arvalid = 0; axi_rready = 0;
        wait(rst_n); repeat(3) @(posedge clk);

        // CPU write addr 16
        cpu_valid = 1; cpu_we = 1; cpu_wstrb = 4'hF; cpu_addr = 32'd16; cpu_wdata = 32'h1234_5678;
        @(posedge clk); cpu_valid = 0; @(posedge clk);
        // CPU readback
        cpu_valid = 1; cpu_we = 0; cpu_addr = 32'd16;
        @(posedge clk); cpu_valid = 0; @(posedge clk);
        check(cpu_rdata === 32'h1234_5678, "CPU self readback");

        // NPU core0 read addr 16
        axi_araddr[0*32 +: 32] = 32'd16; axi_arlen[0*8 +: 8] = 8'd0;
        axi_arvalid[0] = 1'b1; axi_rready[0] = 1'b1;
        @(posedge clk); axi_arvalid[0] = 1'b0;
        repeat(3) @(posedge clk);
        check(axi_rdata[0*DATA_W +: DATA_W] === 32'h1234_5678, "NPU core0 read wrong");
        axi_rready[0] = 1'b0; @(posedge clk);

        // NPU core1 write addr 64
        axi_awaddr[1*32 +: 32] = 32'd64; axi_awvalid[1] = 1'b1;
        axi_wdata[1*DATA_W +: DATA_W] = 32'hDEAD_BEEF;
        axi_wstrb[1*(DATA_W/8) +: DATA_W/8] = {DATA_W/8{1'b1}};
        axi_wlast[1] = 1'b1; axi_wvalid[1] = 1'b1; axi_bready[1] = 1'b1;
        @(posedge clk); axi_awvalid[1] = 1'b0;
        @(posedge clk); axi_wvalid[1] = 1'b0;
        repeat(5) @(posedge clk); axi_bready[1] = 1'b0; @(posedge clk);

        // CPU read addr 64
        cpu_valid = 1; cpu_we = 0; cpu_addr = 32'd64;
        @(posedge clk); cpu_valid = 0; @(posedge clk);
        check(cpu_rdata === 32'hDEAD_BEEF, "CPU read of NPU write wrong");

        // Burst write 2 beats at addr 128
        axi_awaddr[0*32 +: 32] = 32'd128; axi_awvalid[0] = 1'b1;
        axi_wdata[0*DATA_W +: DATA_W] = 32'hAAAA_0001;
        axi_wstrb[0*(DATA_W/8) +: DATA_W/8] = {DATA_W/8{1'b1}};
        axi_wlast[0] = 1'b0; axi_wvalid[0] = 1'b0; axi_bready[0] = 1'b1;
        @(posedge clk); axi_awvalid[0] = 1'b0;
        // W beat 1
        axi_wvalid[0] = 1'b1; @(posedge clk);
        // W beat 2
        axi_wdata[0*DATA_W +: DATA_W] = 32'hAAAA_0002; axi_wlast[0] = 1'b1;
        @(posedge clk); axi_wvalid[0] = 1'b0;
        repeat(5) @(posedge clk); axi_bready[0] = 1'b0; @(posedge clk);

        // CPU readback of burst data
        cpu_valid = 1; cpu_we = 0; cpu_addr = 32'd128;
        @(posedge clk); cpu_valid = 0; @(posedge clk);
        check(cpu_rdata === 32'hAAAA_0001, "burst CPU readback beat0 bad");
        cpu_valid = 1; cpu_we = 0; cpu_addr = 32'd132;
        @(posedge clk); cpu_valid = 0; @(posedge clk);
        check(cpu_rdata === 32'hAAAA_0002, "burst CPU readback beat1 bad");

        // Burst read back: do NOT keep rready high at AR fire
        axi_araddr[0*32 +: 32] = 32'd128; axi_arlen[0*8 +: 8] = 8'd1;
        axi_arvalid[0] = 1'b1; axi_rready[0] = 1'b0;
        @(posedge clk); axi_arvalid[0] = 1'b0;
        // Now wait a cycle, then pulse rready for each beat
        @(posedge clk);
        axi_rready[0] = 1'b1; @(posedge clk);
        check(axi_rdata[0*DATA_W +: DATA_W] === 32'hAAAA_0001, "burst beat0 mismatch");
        check(axi_rlast[0] === 1'b0, "burst beat0 rlast wrong");
        @(posedge clk);
        check(axi_rdata[0*DATA_W +: DATA_W] === 32'hAAAA_0002, "burst beat1 mismatch");
        check(axi_rlast[0] === 1'b1, "burst rlast missing");
        axi_rready[0] = 1'b0; @(posedge clk);

        if (fail_cnt == 0) $display("[PASS] tb_dram_multi_port: %0d checks", pass_cnt);
        else $display("[FAIL] tb_dram_multi_port: %0d passed, %0d failed", pass_cnt, fail_cnt);
        $finish;
    end

    initial #(CLK_T * TIMEOUT) begin $display("[FAIL] tb_dram_multi_port timeout"); $finish; end
endmodule
