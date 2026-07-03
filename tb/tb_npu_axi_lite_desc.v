`timescale 1ns/1ps

module tb_npu_axi_lite_desc;

localparam CLK_T = 10;
localparam REG_CTRL       = 32'h00;
localparam REG_STATUS     = 32'h04;
localparam REG_INT_EN     = 32'h08;
localparam REG_INT_CLR    = 32'h0C;
localparam REG_M_DIM      = 32'h10;
localparam REG_ARR_CFG    = 32'h30;
localparam REG_DESC_BASE  = 32'h40;
localparam REG_DESC_COUNT = 32'h44;
localparam REG_ERR_STATUS = 32'h74;

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
wire [31:0] m_dim;
wire [31:0] n_dim;
wire [31:0] k_dim;
wire [31:0] w_addr;
wire [31:0] a_addr;
wire [31:0] r_addr;
wire [7:0]  arr_cfg;
wire [2:0]  clk_div;
wire        cg_en;
wire [1:0]  cfg_shape;
wire [31:0] desc_base;
wire [31:0] desc_count;
wire        npu_irq;
wire        err_clear;
wire [31:0] err_clear_mask;

reg status_busy;
reg status_done;
reg status_error;
reg [31:0] err_status;
reg irq_flag;
reg saw_err_clear;
reg [31:0] saw_err_clear_mask;

always @(posedge clk) begin
    if (err_clear) begin
        saw_err_clear <= 1'b1;
        saw_err_clear_mask <= err_clear_mask;
    end
end

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
    .status_busy(status_busy),
    .status_done(status_done),
    .status_error(status_error),
    .err_status(err_status),
    .err_clear(err_clear),
    .err_clear_mask(err_clear_mask),
    .irq_flag(irq_flag),
    .perf_cycles(32'd0),
    .perf_m_axi_rd_beats(32'd0),
    .perf_m_axi_wr_beats(32'd0),
    .perf_m_axi_rd_bytes(32'd0),
    .perf_m_axi_wr_bytes(32'd0),
    .perf_m_axi_rd_bw(32'd0),
    .perf_m_axi_wr_bw(32'd0),
    .perf_m_axi_rd_util(32'd0),
    .perf_m_axi_wr_util(32'd0),
    .perf_m_axi_rd_bursts(32'd0),
    .perf_m_axi_wr_bursts(32'd0),
    .npu_irq(npu_irq)
);

task axi_write;
    input [31:0] addr;
    input [31:0] data;
    integer guard;
    begin
        awaddr  <= addr;
        awvalid <= 1'b1;
        wdata   <= data;
        wstrb   <= 4'hF;
        wvalid  <= 1'b1;
        bready  <= 1'b1;
        @(posedge clk);
        while (!awready) @(posedge clk);
        @(posedge clk);
        awvalid <= 1'b0;
        wvalid  <= 1'b0;
        guard = 0;
        while (!bvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        bready <= 1'b0;
        if (guard >= 100) begin
            $display("[FAIL] AXI-Lite write timeout at 0x%08h", addr);
            $fatal;
        end
    end
endtask

task axi_write_delayed_bready;
    input [31:0] addr;
    input [31:0] data;
    integer i;
    begin
        awaddr  <= addr;
        awvalid <= 1'b1;
        wdata   <= data;
        wstrb   <= 4'hF;
        wvalid  <= 1'b0;
        bready  <= 1'b0;
        @(posedge clk);
        awvalid <= 1'b0;
        wvalid  <= 1'b1;
        @(posedge clk);
        wvalid <= 1'b0;
        @(posedge clk);

        if (bvalid !== 1'b1) begin
            $display("[FAIL] delayed-BREADY write did not assert BVALID");
            $fatal;
        end

        for (i = 0; i < 3; i = i + 1) begin
            @(posedge clk);
            if (bvalid !== 1'b1) begin
                $display("[FAIL] BVALID dropped before BREADY, hold_cycle=%0d", i);
                $fatal;
            end
        end

        bready <= 1'b1;
        @(posedge clk);
        bready <= 1'b0;
        @(posedge clk);
        if (bvalid !== 1'b0) begin
            $display("[FAIL] BVALID did not clear after BREADY handshake");
            $fatal;
        end
    end
endtask

task pulse_done_irq;
    begin
        @(negedge clk);
        irq_flag = 1'b1;
        @(negedge clk);
        irq_flag = 1'b0;
        @(posedge clk);
    end
endtask

task pulse_error_status;
    input [31:0] mask;
    begin
        @(negedge clk);
        err_status = mask;
        status_error = 1'b1;
        @(negedge clk);
        status_error = 1'b0;
        @(posedge clk);
    end
endtask

task axi_read;
    input  [31:0] addr;
    output [31:0] data;
    integer guard;
    begin
        araddr  <= addr;
        arvalid <= 1'b1;
        rready  <= 1'b1;
        @(posedge clk);
        guard = 0;
        while (!arready && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        @(posedge clk);
        arvalid <= 1'b0;
        guard = 0;
        while (!rvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        data = rdata;
        rready <= 1'b0;
        if (guard >= 100) begin
            $display("[FAIL] AXI-Lite read timeout at 0x%08h", addr);
            $fatal;
        end
    end
endtask

task expect_read;
    input [31:0] addr;
    input [31:0] exp;
    reg [31:0] got;
    begin
        axi_read(addr, got);
        if (got !== exp) begin
            $display("[FAIL] read 0x%08h got=0x%08h exp=0x%08h", addr, got, exp);
            $fatal;
        end
    end
endtask

initial begin
    awaddr = 32'd0;
    awvalid = 1'b0;
    wdata = 32'd0;
    wstrb = 4'd0;
    wvalid = 1'b0;
    bready = 1'b0;
    araddr = 32'd0;
    arvalid = 1'b0;
    rready = 1'b0;
    status_busy = 1'b0;
    status_done = 1'b0;
    status_error = 1'b0;
    err_status = 32'd0;
    irq_flag = 1'b0;
    saw_err_clear = 1'b0;
    saw_err_clear_mask = 32'd0;

    @(posedge rst_n);
    repeat (2) @(posedge clk);

    expect_read(REG_DESC_BASE, 32'd0);
    expect_read(REG_DESC_COUNT, 32'd0);
    expect_read(REG_ARR_CFG, 32'd0);

    axi_write(REG_ARR_CFG, 32'h0000_00C0);
    expect_read(REG_ARR_CFG, 32'h0000_00C0);
    if (arr_cfg !== 8'hC0) begin
        $display("[FAIL] ARR_CFG router/tile bits not preserved arr_cfg=0x%02h", arr_cfg);
        $fatal;
    end

    axi_write_delayed_bready(REG_M_DIM, 32'd17);
    expect_read(REG_M_DIM, 32'd17);

    axi_write(REG_DESC_BASE, 32'h0000_4000);
    axi_write(REG_DESC_COUNT, 32'd3);

    expect_read(REG_DESC_BASE, 32'h0000_4000);
    expect_read(REG_DESC_COUNT, 32'd3);

    if (desc_base !== 32'h0000_4000 || desc_count !== 32'd3) begin
        $display("[FAIL] desc outputs base=0x%08h count=%0d", desc_base, desc_count);
        $fatal;
    end

    axi_write(REG_CTRL, 32'h0000_0091); // start + OS + desc_mode; irq_clr bit remains W1C.
    expect_read(REG_CTRL, 32'h0000_0091);

    status_busy = 1'b1;
    status_done = 1'b0;
    status_error = 1'b0;
    expect_read(REG_STATUS, 32'h0000_0001);

    status_busy = 1'b0;
    status_done = 1'b1;
    expect_read(REG_STATUS, 32'h0000_0002);

    axi_write(REG_INT_EN, 32'h0000_0001);
    pulse_done_irq();
    if (npu_irq !== 1'b1) begin
        $display("[FAIL] done IRQ did not set npu_irq");
        $fatal;
    end
    expect_read(REG_INT_CLR, 32'h0000_0001);
    axi_write(REG_INT_CLR, 32'h0000_0001);
    @(posedge clk);
    if (npu_irq !== 1'b0) begin
        $display("[FAIL] INT_CLR did not clear npu_irq");
        $fatal;
    end

    axi_write(REG_INT_EN, 32'h0000_0002);
    pulse_error_status(32'h0000_0005);
    status_error = 1'b1;
    expect_read(REG_STATUS, 32'h0000_0006);
    expect_read(REG_ERR_STATUS, 32'h0000_0005);
    if (npu_irq !== 1'b1) begin
        $display("[FAIL] error IRQ did not set npu_irq");
        $fatal;
    end

    axi_write(REG_CTRL, 32'h0000_0040);
    @(posedge clk);
    if (npu_irq !== 1'b0 || ctrl_reg[6] !== 1'b0) begin
        $display("[FAIL] CTRL irq_clr W1C failed npu_irq=%b ctrl=0x%08h", npu_irq, ctrl_reg);
        $fatal;
    end

    saw_err_clear = 1'b0;
    saw_err_clear_mask = 32'd0;
    axi_write(REG_ERR_STATUS, 32'h0000_0004);
    repeat (2) @(posedge clk);
    if (saw_err_clear !== 1'b1 || saw_err_clear_mask !== 32'h0000_0004) begin
        $display("[FAIL] ERR_STATUS W1C pulse clear=%b mask=0x%08h", saw_err_clear, saw_err_clear_mask);
        $fatal;
    end

    $display("[PASS] tb_npu_axi_lite_desc: descriptor regs, ARR_CFG, IRQ, STATUS, and ERR_STATUS W1C passed");
    $finish;
end

initial begin
    #(CLK_T * 2000);
    $display("[FAIL] tb_npu_axi_lite_desc timeout");
    $finish;
end

endmodule
