`timescale 1ns/1ps

module tb_npu_tile_gemm;

`include "test_params.vh"

localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;

localparam REG_CTRL      = 32'h00;
localparam REG_STATUS    = 32'h04;
localparam REG_M_DIM     = 32'h10;
localparam REG_N_DIM     = 32'h14;
localparam REG_K_DIM     = 32'h18;
localparam REG_W_ADDR    = 32'h20;
localparam REG_A_ADDR    = 32'h24;
localparam REG_R_ADDR    = 32'h28;
localparam REG_ARR_CFG   = 32'h30;
localparam REG_CFG_SHAPE = 32'h3C;

localparam ARR_TILE4     = 32'h80; // ARR_CFG[7]: enable 4x4 tile planner/data path
localparam CFG_4X4       = 32'h0;  // CFG_SHAPE=00: use the left-top 4x4 PE array

reg clk = 1'b0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 1'b0;
initial begin
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
end

reg  [31:0] s_awaddr, s_wdata;
reg  [3:0]  s_wstrb;
reg         s_awvalid, s_wvalid, s_bready;
wire        s_awready, s_wready, s_bvalid;
wire [1:0]  s_bresp;

reg  [31:0] s_araddr;
reg         s_arvalid, s_rready;
wire        s_arready, s_rvalid;
wire [31:0] s_rdata;
wire [1:0]  s_rresp;

wire [31:0] m_awaddr;
wire [7:0]  m_awlen;
wire [2:0]  m_awsize;
wire [1:0]  m_awburst;
wire        m_awvalid;
reg         m_awready;
wire [31:0] m_wdata;
wire [3:0]  m_wstrb;
wire        m_wlast;
wire        m_wvalid;
reg         m_wready;
wire [1:0]  m_bresp;
reg         m_bvalid;
wire        m_bready;

wire [31:0] m_araddr;
wire [7:0]  m_arlen;
wire [2:0]  m_arsize;
wire [1:0]  m_arburst;
wire        m_arvalid;
reg         m_arready;
reg  [31:0] m_rdata;
wire [1:0]  m_rresp;
reg         m_rvalid;
wire        m_rready;
reg         m_rlast;

wire npu_irq;

reg [31:0] dram [0:`DRAM_SIZE-1];
reg [31:0] expected [0:`NUM_RESULTS-1];
integer aw_count;   // number of result row write bursts observed
integer pass_cnt;
integer fail_cnt;

initial begin
    $readmemh(`DRAM_HEX, dram);
    $readmemh(`EXPECTED_HEX, expected);
end

npu_top #(
    .DATA_W(DATA_W),
    .ACC_W (ACC_W)
) u_npu (
    .sys_clk       (clk),
    .sys_rst_n     (rst_n),
    .s_axi_awaddr  (s_awaddr),
    .s_axi_awvalid (s_awvalid),
    .s_axi_awready (s_awready),
    .s_axi_wdata   (s_wdata),
    .s_axi_wstrb   (s_wstrb),
    .s_axi_wvalid  (s_wvalid),
    .s_axi_wready  (s_wready),
    .s_axi_bresp   (s_bresp),
    .s_axi_bvalid  (s_bvalid),
    .s_axi_bready  (s_bready),
    .s_axi_araddr  (s_araddr),
    .s_axi_arvalid (s_arvalid),
    .s_axi_arready (s_arready),
    .s_axi_rdata   (s_rdata),
    .s_axi_rresp   (s_rresp),
    .s_axi_rvalid  (s_rvalid),
    .s_axi_rready  (s_rready),
    .m_axi_awaddr  (m_awaddr),
    .m_axi_awlen   (m_awlen),
    .m_axi_awsize  (m_awsize),
    .m_axi_awburst (m_awburst),
    .m_axi_awvalid (m_awvalid),
    .m_axi_awready (m_awready),
    .m_axi_wdata   (m_wdata),
    .m_axi_wstrb   (m_wstrb),
    .m_axi_wlast   (m_wlast),
    .m_axi_wvalid  (m_wvalid),
    .m_axi_wready  (m_wready),
    .m_axi_bresp   (m_bresp),
    .m_axi_bvalid  (m_bvalid),
    .m_axi_bready  (m_bready),
    .m_axi_araddr  (m_araddr),
    .m_axi_arlen   (m_arlen),
    .m_axi_arsize  (m_arsize),
    .m_axi_arburst (m_arburst),
    .m_axi_arvalid (m_arvalid),
    .m_axi_arready (m_arready),
    .m_axi_rdata   (m_rdata),
    .m_axi_rresp   (m_rresp),
    .m_axi_rvalid  (m_rvalid),
    .m_axi_rready  (m_rready),
    .m_axi_rlast   (m_rlast),
    .npu_irq       (npu_irq)
);

assign m_bresp = 2'b00;
assign m_rresp = 2'b00;

reg [31:0] rd_base;
reg [7:0]  rd_len, rd_cnt;
reg        rd_active;

always @(posedge clk) begin
    if (!rst_n) begin
        m_arready <= 1'b0;
        m_rvalid  <= 1'b0;
        m_rlast   <= 1'b0;
        rd_active <= 1'b0;
        rd_cnt    <= 8'd0;
    end else begin
        m_arready <= 1'b1;
        if (m_rvalid && m_rready && m_rlast) begin
            m_rvalid <= 1'b0;
            m_rlast  <= 1'b0;
        end else if (!rd_active && !m_rvalid && m_arvalid && m_arready) begin
            // Model a simple AXI read burst. m_arlen is beats-1.
            rd_active <= 1'b1;
            rd_base   <= m_araddr;
            rd_len    <= m_arlen;
            rd_cnt    <= 8'd0;
            m_rvalid  <= 1'b0;
            m_rlast   <= 1'b0;
        end else if (rd_active && (!m_rvalid || (m_rvalid && m_rready))) begin
            m_rdata  <= dram[((rd_base >> 2) + rd_cnt) % `DRAM_SIZE];
            m_rvalid <= 1'b1;
            m_rlast  <= (rd_cnt >= rd_len);
            if (rd_cnt >= rd_len)
                rd_active <= 1'b0;
            else
                rd_cnt <= rd_cnt + 1'b1;
        end
    end
end

reg [31:0] wr_base;
reg [7:0]  wr_cnt;
reg        wr_phase, b_pending;

always @(posedge clk) begin
    if (!rst_n) begin
        m_awready <= 1'b0;
        m_wready  <= 1'b0;
        m_bvalid  <= 1'b0;
        wr_phase  <= 1'b0;
        b_pending <= 1'b0;
        wr_cnt    <= 8'd0;
        aw_count  <= 0;
    end else begin
        m_awready <= 1'b1;
        if (m_awvalid && m_awready && !wr_phase) begin
            // Tile mode writes one short burst per valid C row.
            wr_phase <= 1'b1;
            wr_base  <= m_awaddr;
            wr_cnt   <= 8'd0;
            m_wready <= 1'b1;
            aw_count <= aw_count + 1;
        end

        if (wr_phase && m_wvalid && m_wready) begin
            dram[((wr_base >> 2) + wr_cnt) % `DRAM_SIZE] <= m_wdata;
            wr_cnt <= wr_cnt + 1'b1;
            if (m_wlast) begin
                wr_phase  <= 1'b0;
                m_wready  <= 1'b0;
                b_pending <= 1'b1;
            end
        end

        if (b_pending && !m_bvalid) begin
            m_bvalid  <= 1'b1;
            b_pending <= 1'b0;
        end else if (m_bvalid && m_bready) begin
            m_bvalid <= 1'b0;
        end
    end
end

task axi_write;
    input [31:0] addr;
    input [31:0] data;
    integer guard;
    begin
        s_awaddr  <= addr;
        s_awvalid <= 1'b1;
        s_wdata   <= data;
        s_wstrb   <= 4'hF;
        s_wvalid  <= 1'b1;
        s_bready  <= 1'b1;
        @(posedge clk);
        while (!s_awready) @(posedge clk);
        @(posedge clk);
        s_awvalid <= 1'b0;
        s_wvalid  <= 1'b0;
        guard = 0;
        while (!s_bvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        s_bready <= 1'b0;
        if (guard >= 100) begin
            $display("[FAIL] AXI-Lite write timeout at 0x%08h", addr);
            $finish;
        end
    end
endtask

task axi_read;
    input  [31:0] addr;
    output [31:0] data;
    integer guard;
    begin
        s_araddr  <= addr;
        s_arvalid <= 1'b1;
        s_rready  <= 1'b1;
        @(posedge clk);
        while (!s_arready) @(posedge clk);
        @(posedge clk);
        s_arvalid <= 1'b0;
        guard = 0;
        while (!s_rvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        data = s_rdata;
        s_rready <= 1'b0;
        if (guard >= 100) begin
            $display("[FAIL] AXI-Lite read timeout at 0x%08h", addr);
            $finish;
        end
    end
endtask

task wait_done;
    input [31:0] timeout;
    integer guard;
    reg [31:0] status;
    begin
        status = 32'd0;
        guard = 0;
        while (!status[1] && guard < timeout) begin
            axi_read(REG_STATUS, status);
            @(posedge clk);
            guard = guard + 1;
        end
        if (!status[1]) begin
            $display("[FAIL] NPU timeout state=%0d dma=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $finish;
        end
        axi_write(REG_CTRL, 32'h0);
    end
endtask

function [31:0] fp32_ordered;
    input [31:0] x;
    begin
        fp32_ordered = x[31] ? ~x : (x | 32'h8000_0000);
    end
endfunction

function fp32_close;
    input [31:0] got;
    input [31:0] exp;
    reg [31:0] got_ord, exp_ord, diff;
    begin
        got_ord = fp32_ordered(got);
        exp_ord = fp32_ordered(exp);
        diff = (got_ord > exp_ord) ? (got_ord - exp_ord) : (exp_ord - got_ord);
        fp32_close = (got === exp) || (diff <= 32'd8);
    end
endfunction

task check_result;
    input integer idx;
    reg [31:0] got;
    reg [31:0] exp;
    begin
        // expected[] is row-major C[r,c], so idx/4 is r and idx%4 is c.
        got = dram[(`R_ADDR >> 2) + idx];
        exp = expected[idx];
        if (`IS_FP16) begin
            if (fp32_close(got, exp)) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s C[%0d][%0d] got=0x%08h exp=0x%08h",
                         `TEST_NAME, idx / 4, idx % 4, got, exp);
                fail_cnt = fail_cnt + 1;
            end
        end else begin
            if (got === exp) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s C[%0d][%0d] got=%0d (0x%08h) exp=%0d (0x%08h)",
                         `TEST_NAME, idx / 4, idx % 4,
                         $signed(got), got, $signed(exp), exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

integer i;

initial begin
    s_awaddr = 0; s_wdata = 0; s_wstrb = 0;
    s_awvalid = 0; s_wvalid = 0; s_bready = 0;
    s_araddr = 0; s_arvalid = 0; s_rready = 0;
    pass_cnt = 0;
    fail_cnt = 0;

    @(posedge rst_n);
    repeat (4) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  4x4 Tile GEMM Test: %s", `TEST_NAME);
    $display("################################################################");

    axi_write(REG_M_DIM, `M_DIM);
    axi_write(REG_N_DIM, `N_DIM);
    axi_write(REG_K_DIM, `K_DIM);
    axi_write(REG_W_ADDR, `W_ADDR);
    axi_write(REG_A_ADDR, `A_ADDR);
    axi_write(REG_R_ADDR, `R_ADDR);
    axi_write(REG_ARR_CFG, ARR_TILE4);
    axi_write(REG_CFG_SHAPE, CFG_4X4);
    axi_write(REG_CTRL, `CTRL);

    wait_done(5000);

    if (aw_count !== 4) begin
        // For a full 4x4 tile, npu_ctrl should issue 4 row-wise write bursts.
        $display("[FAIL] %s expected 4 row write bursts, got %0d", `TEST_NAME, aw_count);
        $finish;
    end

    for (i = 0; i < `NUM_RESULTS; i = i + 1)
        check_result(i);

    if (fail_cnt == 0) begin
        $display("[PASS] %s: ALL %0d CHECKS PASSED", `TEST_NAME, pass_cnt);
    end else begin
        $display("[FAIL] %s: %0d passed, %0d failed", `TEST_NAME, pass_cnt, fail_cnt);
        $fatal;
    end

    $finish;
end

initial begin
    #(CLK_T * 200000);
    $display("[FAIL] %s global timeout", `TEST_NAME);
    $finish;
end

endmodule
