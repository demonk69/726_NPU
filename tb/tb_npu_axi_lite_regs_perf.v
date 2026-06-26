`timescale 1ns/1ps

module tb_npu_axi_lite_regs_perf;

localparam CLK_T = 10;

reg clk = 1'b0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 1'b0;
initial begin
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
end

reg  [31:0] awaddr;
reg         awvalid;
wire        awready;
reg  [31:0] wdata;
reg  [3:0]  wstrb;
reg         wvalid;
wire        wready;
wire [1:0]  bresp;
wire        bvalid;
reg         bready;
reg  [31:0] araddr;
reg         arvalid;
wire        arready;
wire [31:0] rdata;
wire [1:0]  rresp;
wire        rvalid;
reg         rready;

wire [31:0] ctrl_reg;
wire [31:0] m_dim, n_dim, k_dim;
wire [31:0] w_addr, a_addr, r_addr;
wire [7:0]  arr_cfg;
wire [2:0]  clk_div;
wire        cg_en;
wire [1:0]  cfg_shape;
wire [31:0] desc_base, desc_count;
wire [31:0] conv_ifm_shape, conv_channels, conv_kernel, conv_out_shape;
wire [31:0] conv_stride_pad, conv_dilation, bias_addr, quant_cfg;
wire        err_clear;
wire [31:0] err_clear_mask;
wire        perf_clear;
wire        npu_irq;

reg status_busy, status_done, status_error, irq_flag;
reg [31:0] err_status;

npu_axi_lite dut (
    .aclk(clk),
    .aresetn(rst_n),
    .awaddr(awaddr),
    .awvalid(awvalid),
    .awready(awready),
    .wdata(wdata),
    .wstrb(wstrb),
    .wvalid(wvalid),
    .wready(wready),
    .bresp(bresp),
    .bvalid(bvalid),
    .bready(bready),
    .araddr(araddr),
    .arvalid(arvalid),
    .arready(arready),
    .rdata(rdata),
    .rresp(rresp),
    .rvalid(rvalid),
    .rready(rready),
    .ctrl_reg(ctrl_reg),
    .m_dim(m_dim),
    .n_dim(n_dim),
    .k_dim(k_dim),
    .w_addr(w_addr),
    .a_addr(a_addr),
    .r_addr(r_addr),
    .arr_cfg(arr_cfg),
    .clk_div(clk_div),
    .cg_en(cg_en),
    .cfg_shape(cfg_shape),
    .desc_base(desc_base),
    .desc_count(desc_count),
    .conv_ifm_shape(conv_ifm_shape),
    .conv_channels(conv_channels),
    .conv_kernel(conv_kernel),
    .conv_out_shape(conv_out_shape),
    .conv_stride_pad(conv_stride_pad),
    .conv_dilation(conv_dilation),
    .bias_addr(bias_addr),
    .quant_cfg(quant_cfg),
    .status_busy(status_busy),
    .status_done(status_done),
    .status_error(status_error),
    .err_status(err_status),
    .err_clear(err_clear),
    .err_clear_mask(err_clear_mask),
    .irq_flag(irq_flag),
    .perf_cycles(32'h0000_0101),
    .perf_m_axi_rd_beats(32'h0000_0202),
    .perf_m_axi_wr_beats(32'h0000_0303),
    .perf_m_axi_rd_bytes(32'h0000_0404),
    .perf_m_axi_wr_bytes(32'h0000_0505),
    .perf_m_axi_rd_bw(32'h0000_0606),
    .perf_m_axi_wr_bw(32'h0000_0707),
    .perf_m_axi_rd_util(32'h0000_0808),
    .perf_m_axi_wr_util(32'h0000_0909),
    .perf_m_axi_rd_bursts(32'h0000_0A0A),
    .perf_m_axi_wr_bursts(32'h0000_0B0B),
    .perf_mac_ops(64'h1111_2222_3333_4444),
    .perf_ops(64'h5555_6666_7777_8888),
    .perf_busy_cycles(32'h0000_0C0C),
    .perf_compute_cycles(32'h0000_0D0D),
    .perf_dma_cycles(32'h0000_0E0E),
    .perf_tops_x1e6(32'h0000_0F0F),
    .perf_compute_util_bp(32'h0000_1010),
    .perf_e2e_util_bp(32'h0000_1111),
    .perf_peak_ops_per_cycle(32'h0000_1212),
    .perf_clear(perf_clear),
    .npu_irq(npu_irq)
);

task axi_write;
    input [31:0] addr;
    input [31:0] data;
    input [3:0]  strb;
    integer guard;
    begin
        awaddr <= addr;
        awvalid <= 1'b1;
        wdata <= data;
        wstrb <= strb;
        wvalid <= 1'b1;
        bready <= 1'b1;
        @(posedge clk);
        while (!awready) @(posedge clk);
        @(posedge clk);
        awvalid <= 1'b0;
        wvalid <= 1'b0;
        guard = 0;
        while (!bvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        if (guard >= 100) begin
            $display("[FAIL] write timeout addr=0x%08h", addr);
            $fatal;
        end
        @(posedge clk);
        bready <= 1'b0;
    end
endtask

task axi_read;
    input [31:0] addr;
    output [31:0] data;
    integer guard;
    begin
        araddr <= addr;
        arvalid <= 1'b1;
        rready <= 1'b1;
        @(posedge clk);
        while (!arready) @(posedge clk);
        @(posedge clk);
        arvalid <= 1'b0;
        guard = 0;
        while (!rvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        data = rdata;
        if (guard >= 100) begin
            $display("[FAIL] read timeout addr=0x%08h", addr);
            $fatal;
        end
        @(posedge clk);
        rready <= 1'b0;
    end
endtask

task expect_read;
    input [31:0] addr;
    input [31:0] exp;
    reg [31:0] got;
    begin
        axi_read(addr, got);
        if (got !== exp) begin
            $display("[FAIL] read addr=0x%08h got=0x%08h exp=0x%08h", addr, got, exp);
            $fatal;
        end
    end
endtask

initial begin
    awaddr = 0;
    awvalid = 0;
    wdata = 0;
    wstrb = 0;
    wvalid = 0;
    bready = 0;
    araddr = 0;
    arvalid = 0;
    rready = 0;
    status_busy = 0;
    status_done = 0;
    status_error = 0;
    irq_flag = 0;
    err_status = 32'hCAFE_0001;

    @(posedge rst_n);
    repeat (2) @(posedge clk);

    axi_write(32'h14, 32'h0000_0014, 4'hF);
    axi_write(32'h18, 32'h0000_0018, 4'hF);
    axi_write(32'h20, 32'h1000_0020, 4'hF);
    axi_write(32'h24, 32'h1000_0024, 4'hF);
    axi_write(32'h28, 32'h1000_0028, 4'hF);
    axi_write(32'h30, 32'h0000_00A5, 4'hF);
    axi_write(32'h34, 32'h0000_0007, 4'hF);
    axi_write(32'h38, 32'h0000_0001, 4'hF);
    axi_write(32'h3C, 32'h0000_0003, 4'hF);
    axi_write(32'h80, 32'h0010_0020, 4'hF);
    axi_write(32'h84, 32'h0002_0003, 4'hF);
    axi_write(32'h88, 32'h0003_0003, 4'hF);
    axi_write(32'h8C, 32'h000E_001E, 4'hF);
    axi_write(32'h90, 32'h0102_0304, 4'hF);
    axi_write(32'h94, 32'h0001_0002, 4'hF);
    axi_write(32'h98, 32'h2000_0098, 4'hF);
    axi_write(32'h9C, 32'h1234_0503, 4'hF);

    expect_read(32'h14, 32'h0000_0014);
    expect_read(32'h18, 32'h0000_0018);
    expect_read(32'h20, 32'h1000_0020);
    expect_read(32'h24, 32'h1000_0024);
    expect_read(32'h28, 32'h1000_0028);
    expect_read(32'h30, 32'h0000_00A5);
    expect_read(32'h34, 32'h0000_0007);
    expect_read(32'h38, 32'h0000_0001);
    expect_read(32'h3C, 32'h0000_0003);
    expect_read(32'h80, 32'h0010_0020);
    expect_read(32'h84, 32'h0002_0003);
    expect_read(32'h88, 32'h0003_0003);
    expect_read(32'h8C, 32'h000E_001E);
    expect_read(32'h90, 32'h0102_0304);
    expect_read(32'h94, 32'h0001_0002);
    expect_read(32'h98, 32'h2000_0098);
    expect_read(32'h9C, 32'h1234_0503);

    axi_write(32'h78, 32'h0000_0003, 4'hF);
    @(posedge clk);
    if (perf_clear !== 1'b0) begin
        $display("[FAIL] perf_clear should be one-cycle pulse");
        $fatal;
    end

    expect_read(32'h48, 32'h0000_0101);
    expect_read(32'h4C, 32'h0000_0202);
    expect_read(32'h50, 32'h0000_0303);
    expect_read(32'h54, 32'h0000_0404);
    expect_read(32'h58, 32'h0000_0505);
    expect_read(32'h5C, 32'h0000_0606);
    expect_read(32'h60, 32'h0000_0707);
    expect_read(32'h64, 32'h0000_0808);
    expect_read(32'h68, 32'h0000_0909);
    expect_read(32'h6C, 32'h0000_0A0A);
    expect_read(32'h70, 32'h0000_0B0B);
    expect_read(32'h78, 32'h0000_0000);
    expect_read(32'hA0, 32'h3333_4444);
    expect_read(32'hA4, 32'h1111_2222);
    expect_read(32'hA8, 32'h7777_8888);
    expect_read(32'hAC, 32'h5555_6666);
    expect_read(32'hB0, 32'h0000_0C0C);
    expect_read(32'hB4, 32'h0000_0D0D);
    expect_read(32'hB8, 32'h0000_0E0E);
    expect_read(32'hBC, 32'h0000_0F0F);
    expect_read(32'hC0, 32'h0000_1010);
    expect_read(32'hC4, 32'h0000_1111);
    expect_read(32'hC8, 32'h0000_1212);
    expect_read(32'hCC, 32'hDEAD_BEEF);

    axi_write(32'h08, 32'h0000_0002, 4'hF);
    @(negedge clk);
    status_error = 1'b1;
    @(negedge clk);
    status_error = 1'b0;
    @(posedge clk);
    if (npu_irq !== 1'b1) begin
        $display("[FAIL] status_error IRQ path did not set npu_irq");
        $fatal;
    end

    $display("[PASS] tb_npu_axi_lite_regs_perf: full register and perf snapshot coverage passed");
    $finish;
end

initial begin
    #(CLK_T * 3000);
    $display("[FAIL] tb_npu_axi_lite_regs_perf timeout");
    $finish;
end

endmodule
