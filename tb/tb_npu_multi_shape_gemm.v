// Stripped testbench without npu_top to isolate vvp hang
`timescale 1ns/1ps
module tb_npu_multi_shape_gemm;

`include "test_params.vh"

initial $display("=== MODULE ELABORATED ===");

localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;

localparam REG_STATUS = 32'h04;
localparam REG_CTRL   = 32'h00;

reg clk = 1'b0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 1'b0;
initial begin
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
end

reg [31:0] s_awaddr, s_wdata, s_araddr;
reg [3:0]  s_wstrb;
reg        s_awvalid, s_wvalid, s_bready;
reg        s_arvalid, s_rready;
wire       s_awready, s_wready, s_bvalid;
wire       s_arready, s_rvalid;
wire [1:0] s_bresp, s_rresp;
wire [31:0] s_rdata;

// Simplified: connect npu_top
wire [31:0] m_awaddr, m_wdata, m_araddr;
wire [7:0]  m_awlen, m_arlen;
wire [2:0]  m_awsize, m_arsize;
wire [1:0]  m_awburst, m_arburst;
wire        m_awvalid, m_wlast, m_wvalid, m_arvalid, m_rready, m_bready;
wire [3:0]  m_wstrb;
reg         m_awready, m_wready, m_arready;
reg  [31:0] m_rdata;
reg         m_bvalid, m_rvalid, m_rlast;
wire [1:0]  m_bresp, m_rresp;
wire        npu_irq;

reg [31:0] dram [0:`DRAM_SIZE-1];
reg [31:0] expected [0:`NUM_RESULTS-1];

initial begin: load_data
    $readmemh(`DRAM_HEX, dram);
    $readmemh(`EXPECTED_HEX, expected);
    $display("[INFO] DRAM loaded: %0d entries", `DRAM_SIZE);
end

npu_top #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_npu (
    .sys_clk(clk), .sys_rst_n(rst_n),
    .s_axi_awaddr(s_awaddr), .s_axi_awvalid(s_awvalid), .s_axi_awready(s_awready),
    .s_axi_wdata(s_wdata), .s_axi_wstrb(s_wstrb), .s_axi_wvalid(s_wvalid), .s_axi_wready(s_wready),
    .s_axi_bresp(s_bresp), .s_axi_bvalid(s_bvalid), .s_axi_bready(s_bready),
    .s_axi_araddr(s_araddr), .s_axi_arvalid(s_arvalid), .s_axi_arready(s_arready),
    .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp), .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready),
    .m_axi_awaddr(m_awaddr), .m_axi_awlen(m_awlen), .m_axi_awsize(m_awsize), .m_axi_awburst(m_awburst),
    .m_axi_awvalid(m_awvalid), .m_axi_awready(m_awready),
    .m_axi_wdata(m_wdata), .m_axi_wstrb(m_wstrb), .m_axi_wlast(m_wlast),
    .m_axi_wvalid(m_wvalid), .m_axi_wready(m_wready),
    .m_axi_bresp(m_bresp), .m_axi_bvalid(m_bvalid), .m_axi_bready(m_bready),
    .m_axi_araddr(m_araddr), .m_axi_arlen(m_arlen), .m_axi_arsize(m_arsize), .m_axi_arburst(m_arburst),
    .m_axi_arvalid(m_arvalid), .m_axi_arready(m_arready),
    .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp), .m_axi_rvalid(m_rvalid), .m_axi_rready(m_rready), .m_axi_rlast(m_rlast),
    .npu_irq(npu_irq)
);

// Simple AXI slave
reg [31:0] rd_base;
reg [7:0]  rd_len, rd_cnt;
reg        rd_active;

always @(posedge clk) begin
    if (!rst_n) begin
        m_arready <= 1'b0; m_rvalid <= 1'b0; m_rlast <= 1'b0; rd_active <= 1'b0; rd_cnt <= 8'd0;
    end else begin
        m_arready <= 1'b1;
        if (m_rvalid && m_rready && m_rlast) begin m_rvalid <= 1'b0; m_rlast <= 1'b0; end
        else if (!rd_active && !m_rvalid && m_arvalid && m_arready) begin
            rd_active <= 1'b1; rd_base <= m_araddr; rd_len <= m_arlen; rd_cnt <= 8'd0; m_rvalid <= 1'b0; m_rlast <= 1'b0;
        end else if (rd_active && (!m_rvalid || (m_rvalid && m_rready))) begin
            m_rdata <= dram[((rd_base >> 2) + rd_cnt) % `DRAM_SIZE];
            m_rvalid <= 1'b1; m_rlast <= (rd_cnt >= rd_len);
            if (rd_cnt >= rd_len) rd_active <= 1'b0; else rd_cnt <= rd_cnt + 1'b1;
        end
    end
end

assign m_bresp = 2'b00; assign m_rresp = 2'b00;

reg [31:0] wr_base;
reg [7:0]  wr_cnt;
reg wr_phase, b_pending;

always @(posedge clk) begin
    if (!rst_n) begin
        m_awready <= 1'b0; m_wready <= 1'b0; m_bvalid <= 1'b0; wr_phase <= 1'b0; b_pending <= 1'b0; wr_cnt <= 8'd0;
    end else begin
        m_awready <= 1'b1;
        if (m_awvalid && m_awready && !wr_phase) begin
            wr_phase <= 1'b1; wr_base <= m_awaddr; wr_cnt <= 8'd0; m_wready <= 1'b1;
        end
        if (wr_phase && m_wvalid && m_wready) begin
            dram[((wr_base >> 2) + wr_cnt) % `DRAM_SIZE] <= m_wdata;
            wr_cnt <= wr_cnt + 1'b1;
            if (m_wlast) begin wr_phase <= 1'b0; m_wready <= 1'b0; b_pending <= 1'b1; end
        end
        if (b_pending && !m_bvalid) begin m_bvalid <= 1'b1; b_pending <= 1'b0; end
        else if (m_bvalid && m_bready) begin m_bvalid <= 1'b0; end
    end
end

task axi_write;
    input [31:0] addr, data;
    integer guard;
    begin
        s_awaddr <= addr; s_wdata <= data; s_wstrb <= 4'hF; s_awvalid <= 1'b1; s_wvalid <= 1'b1; s_bready <= 1'b1;
        @(posedge clk);
        while (!s_awready || !s_wready) @(posedge clk);
        @(posedge clk);
        s_awvalid <= 1'b0; s_wvalid <= 1'b0;
        guard = 0;
        while (!s_bvalid && guard < 100) begin @(posedge clk); guard = guard + 1; end
        s_bready <= 1'b0;
        if (guard >= 100) begin $display("[FAIL] AXI-Lite write timeout at 0x%08h", addr); $finish; end
    end
endtask

task axi_read;
    input [31:0] addr;
    output [31:0] data;
    integer guard;
    begin
        s_araddr <= addr; s_arvalid <= 1'b1; s_rready <= 1'b1;
        @(posedge clk);
        while (!s_arready) @(posedge clk);
        @(posedge clk);
        s_arvalid <= 1'b0;
        guard = 0;
        while (!s_rvalid && guard < 100) begin @(posedge clk); guard = guard + 1; end
        data = s_rdata; s_rready <= 1'b0;
        if (guard >= 100) begin $display("[FAIL] AXI-Lite read timeout at 0x%08h", addr); $finish; end
    end
endtask

task wait_done;
    input [31:0] timeout;
    integer guard;
    reg [31:0] status;
    begin
        status = 32'd0; guard = 0;
        while (!status[1] && guard < timeout) begin
            axi_read(REG_STATUS, status);
            @(posedge clk);
            guard = guard + 1;
        end
        if (!status[1]) begin $display("[FAIL] NPU timeout state=%0d", u_npu.u_ctrl.state); $finish; end
        axi_write(REG_CTRL, 32'h0);
    end
endtask

initial begin
    pass_cnt = 0; fail_cnt = 0;
    s_awaddr <= 32'd0; s_wdata <= 32'd0; s_wstrb <= 4'h0;
    s_awvalid <= 1'b0; s_wvalid <= 1'b0; s_bready <= 1'b0;
    s_araddr <= 32'd0; s_arvalid <= 1'b0; s_rready <= 1'b0;

    @(posedge rst_n);
    repeat (10) @(posedge clk);
    $display("[INFO] Starting NPU configuration: M=%0d N=%0d K=%0d", `M_DIM, `N_DIM, `K_DIM);

    axi_write(32'h10, `M_DIM);   // M_DIM
    axi_write(32'h14, `N_DIM);   // N_DIM
    axi_write(32'h18, `K_DIM);   // K_DIM
    axi_write(32'h20, `W_ADDR);  // W_ADDR
    axi_write(32'h24, `A_ADDR);  // A_ADDR
    axi_write(32'h28, `R_ADDR);  // R_ADDR
    axi_write(32'h3C, `CFG_SHAPE);
    axi_write(32'h30, `ARR_CFG);
    axi_write(32'h00, 32'h11);   // CTRL: INT8 OS start

    wait_done(50000);

    // Check results
    for (check_idx = 0; check_idx < `NUM_RESULTS; check_idx = check_idx + 1) begin
        check_got = dram[(`R_ADDR >> 2) + check_idx];
        check_exp = expected[check_idx];
        if (check_got === check_exp) pass_cnt = pass_cnt + 1;
        else begin
            check_r = check_idx / `N_DIM; check_c = check_idx % `N_DIM;
            $display("[FAIL] C[%0d][%0d] got=%0d exp=%0d", check_r, check_c, $signed(check_got), $signed(check_exp));
            fail_cnt = fail_cnt + 1;
            if (fail_cnt >= 16) begin $display("[FAIL] Too many mismatches"); $finish; end
        end
    end

    if (fail_cnt == 0)
        $display("[PASS] %s: ALL %0d CHECKS PASSED", `TEST_NAME, pass_cnt);
    else
        $display("[FAIL] %s: %0d/%0d", `TEST_NAME, fail_cnt, pass_cnt);
    $finish;
end

integer pass_cnt, fail_cnt;
integer check_idx, check_r, check_c;
reg [31:0] check_got, check_exp;

endmodule