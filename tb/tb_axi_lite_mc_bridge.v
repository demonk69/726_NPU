// =============================================================================
// tb_axi_lite_mc_bridge - unit test for multi-core AXI-Lite bridge
// =============================================================================
`timescale 1ns/1ps

module tb_axi_lite_mc_bridge;
    localparam CLK_T = 10, NUM_CORES = 2, NPU_BASE = 32'h0200_0000, TIMEOUT = 2000;

    reg clk = 1'b0; always #(CLK_T/2) clk = ~clk;
    reg rst_n = 1'b0;
    initial begin repeat(4) @(posedge clk); rst_n = 1'b1; end

    reg        iomem_valid; wire iomem_ready;
    reg [3:0]  iomem_wstrb; reg [31:0] iomem_addr, iomem_wdata; wire [31:0] iomem_rdata;
    wire [63:0] m_axi_awaddr; wire [1:0] m_axi_awvalid, m_axi_wvalid, m_axi_arvalid;
    reg  [1:0] m_axi_awready, m_axi_wready, m_axi_arready;
    wire [63:0] m_axi_wdata; wire [7:0] m_axi_wstrb;
    reg  [3:0]  m_axi_bresp; reg [1:0] m_axi_bvalid; wire [1:0] m_axi_bready;
    wire [63:0] m_axi_araddr; reg [63:0] m_axi_rdata; reg [3:0] m_axi_rresp;
    reg  [1:0] m_axi_rvalid; wire [1:0] m_axi_rready;

    integer pass_cnt, fail_cnt, cyc;

    axi_lite_mc_bridge #(.NUM_CORES(NUM_CORES), .NPU_CORE_STRIDE(256)) dut (
        .clk(clk), .rst_n(rst_n), .iomem_valid(iomem_valid), .iomem_ready(iomem_ready),
        .iomem_wstrb(iomem_wstrb), .iomem_addr(iomem_addr),
        .iomem_wdata(iomem_wdata), .iomem_rdata(iomem_rdata),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .npu_base_addr(NPU_BASE)
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

    reg [31:0] val;

    initial begin
        pass_cnt = 0; fail_cnt = 0; cyc = 0;
        iomem_valid = 0; iomem_wstrb = 4'd0; iomem_addr = 0; iomem_wdata = 0;
        m_axi_awready = {NUM_CORES{1'b1}}; m_axi_wready = {NUM_CORES{1'b1}};
        m_axi_arready = {NUM_CORES{1'b0}}; m_axi_rdata = 0; m_axi_rresp = 0;
        m_axi_rvalid = {NUM_CORES{1'b0}}; m_axi_bresp = 0; m_axi_bvalid = {NUM_CORES{1'b0}};
        wait(rst_n); repeat(3) @(posedge clk);

        $display("--- Core0 write ---");
        iomem_valid = 1; iomem_wstrb = 4'hF; iomem_addr = NPU_BASE + 32'h10; iomem_wdata = 32'hAAAA_5555;
        @(posedge clk); repeat(2) @(posedge clk);
        iomem_valid = 0; @(posedge clk);
        // After 3 cycles from valid, the addr should be latched
        check(m_axi_awaddr[0*32 +: 32] === 32'h00000010, "core0 write local addr mismatch");
        check(m_axi_wdata[0*32 +: 32] === 32'hAAAA_5555, "core0 write data mismatch");

        $display("--- Core1 write ---");
        iomem_valid = 1; iomem_wstrb = 4'hF; iomem_addr = NPU_BASE + 32'h110; iomem_wdata = 32'hDEAD_BEEF;
        @(posedge clk); repeat(2) @(posedge clk);
        iomem_valid = 0; @(posedge clk);
        check(m_axi_awaddr[1*32 +: 32] === 32'h00000010, "core1 write local addr mismatch");

        $display("--- Core isolation ---");
        iomem_valid = 1; iomem_wstrb = 4'hF; iomem_addr = NPU_BASE + 32'h18; iomem_wdata = 32'h1111_2222;
        @(posedge clk); repeat(2) @(posedge clk);
        iomem_valid = 0; @(posedge clk);
        // Core0 write should not assert core1 valid
        check(m_axi_awvalid[1] === 1'b0 && m_axi_wvalid[1] === 1'b0, "core0 write leaked to core1 valid");

        $display("--- Core0 read ---");
        // We don't have a real NPU, but we can check AR channel
        iomem_valid = 1; iomem_wstrb = 4'h0; iomem_addr = NPU_BASE + 32'h04;
        @(posedge clk); #1;
        check(m_axi_arvalid[0] === 1'b1 && m_axi_arvalid[1] === 1'b0, "core0 read should only assert core0 AR");
        // Respond to read
        m_axi_arready[0] = 1'b1; @(posedge clk); m_axi_arready[0] = 1'b0;
        @(posedge clk);
        m_axi_rdata[0*32 +: 32] = 32'hCCCC_DDDD; m_axi_rvalid[0] = 1'b1;
        @(posedge clk); m_axi_rvalid[0] = 1'b0;
        repeat(2) @(posedge clk);
        iomem_valid = 0; @(posedge clk);
        check(iomem_rdata === 32'hCCCC_DDDD, "core0 read rdata mismatch");

        $display("--- Invalid window read ---");
        iomem_valid = 1; iomem_wstrb = 4'h0; iomem_addr = NPU_BASE + NUM_CORES * 32'h100;
        @(posedge clk); repeat(2) @(posedge clk);
        val = iomem_rdata;
        iomem_valid = 0; @(posedge clk);
        check(val === 32'hDEADBEEF, "invalid window not DEADBEEF");

        $display("--- Invalid window write ---");
        iomem_valid = 1; iomem_wstrb = 4'hF; iomem_addr = NPU_BASE + NUM_CORES * 32'h100; iomem_wdata = 32'hBAAD;
        @(posedge clk); repeat(2) @(posedge clk);
        check(m_axi_awvalid[0] === 1'b0 && m_axi_awvalid[1] === 1'b0, "invalid write fired a core");
        iomem_valid = 0; @(posedge clk);

        if (fail_cnt == 0) $display("[PASS] tb_axi_lite_mc_bridge: %0d checks", pass_cnt);
        else $display("[FAIL] tb_axi_lite_mc_bridge: %0d passed, %0d failed", pass_cnt, fail_cnt);
        $finish;
    end

    always @(posedge clk) if (rst_n) cyc <= cyc + 1;
    initial #(CLK_T * TIMEOUT) begin $display("[FAIL] tb_axi_lite_mc_bridge timeout"); $finish; end
endmodule
