// =============================================================================
// tb_conv2d_two_layer.v - T6.6 two-layer Conv2D end-to-end testbench
//
// Layer 0 uses direct scalar Conv2D on-the-fly im2col with bias, ReLU and INT8
// quantization. Layer 1 consumes the layer0 output address directly as its A
// matrix and verifies the final result against software golden data.
// =============================================================================

`timescale 1ns/1ps

module tb_conv2d_two_layer;

`include "test_params.vh"

parameter ROWS   = 1;
parameter COLS   = 1;
parameter DATA_W = 16;
parameter ACC_W  = 32;
parameter CLK_T  = 10;

localparam REG_CTRL   = 32'h00;
localparam REG_STATUS = 32'h04;
localparam REG_M_DIM  = 32'h10;
localparam REG_N_DIM  = 32'h14;
localparam REG_K_DIM  = 32'h18;
localparam REG_W_ADDR = 32'h20;
localparam REG_A_ADDR = 32'h24;
localparam REG_R_ADDR = 32'h28;
localparam REG_CONV_IFM_SHAPE  = 32'h80;
localparam REG_CONV_CHANNELS   = 32'h84;
localparam REG_CONV_KERNEL     = 32'h88;
localparam REG_CONV_OUT_SHAPE  = 32'h8C;
localparam REG_CONV_STRIDE_PAD = 32'h90;
localparam REG_CONV_DILATION   = 32'h94;
localparam REG_BIAS_ADDR       = 32'h98;
localparam REG_QUANT_CFG       = 32'h9C;

// ---------------------------------------------------------------------------
// DRAM and expected data
// ---------------------------------------------------------------------------
reg [31:0] dram [0:`DRAM_SIZE-1];
reg [31:0] layer0_expected [0:`L0_NUM_RESULTS-1];
reg [31:0] expected [0:`L1_NUM_RESULTS-1];

initial begin
    $readmemh("dram_init.hex", dram);
    $readmemh("layer0_expected.hex", layer0_expected);
    $readmemh("expected.hex", expected);
end

// ---------------------------------------------------------------------------
// Clock and reset
// ---------------------------------------------------------------------------
reg clk = 0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 0;
initial begin
    #(CLK_T*5);
    rst_n = 1;
end

// ---------------------------------------------------------------------------
// AXI4-Lite signals
// ---------------------------------------------------------------------------
reg  [31:0] s_awaddr, s_wdata;
reg  [3:0]  s_wstrb;
reg         s_awvalid, s_wvalid, s_bready;
reg  [31:0] s_araddr;
reg         s_arvalid, s_rready;
wire        s_awready, s_wready, s_bvalid;
wire [1:0]  s_bresp;
wire        s_arready, s_rvalid;
wire [31:0] s_rdata;
wire [1:0]  s_rresp;

// ---------------------------------------------------------------------------
// AXI4 master signals
// ---------------------------------------------------------------------------
wire [31:0]        m_awaddr, m_araddr;
wire [7:0]         m_awlen, m_arlen;
wire [2:0]         m_awsize, m_arsize;
wire [1:0]         m_awburst, m_arburst;
wire               m_awvalid, m_wvalid, m_bready, m_arvalid, m_rready;
wire [ACC_W-1:0]   m_wdata, m_rdata;
wire [ACC_W/8-1:0] m_wstrb;
wire               m_wlast, m_bvalid;
wire [1:0]         m_bresp, m_rresp;
reg                m_awready, m_wready, m_bvalid_r, m_arready;
reg  [ACC_W-1:0]   m_rdata_r;
reg                m_rvalid, m_rlast;
wire               npu_irq;

assign m_bresp = 2'b00;
assign m_rresp = 2'b00;
assign m_rdata = m_rdata_r;
assign m_bvalid = m_bvalid_r;

// ---------------------------------------------------------------------------
// AXI4 DRAM read model
// ---------------------------------------------------------------------------
reg [31:0] rd_base, rd_len_r;
reg [7:0]  rd_cnt;
reg        rd_active;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_active <= 1'b0;
        rd_cnt    <= 8'd0;
        m_rvalid  <= 1'b0;
        m_rlast   <= 1'b0;
        m_rdata_r <= {ACC_W{1'b0}};
    end else if (!rd_active && m_arvalid && m_arready) begin
        rd_active <= 1'b1;
        rd_base   <= m_araddr;
        rd_len_r  <= m_arlen;
        rd_cnt    <= 8'd0;
    end else if (rd_active) begin
        if (m_rvalid && m_rready) begin
            if (m_rlast || rd_cnt >= rd_len_r) begin
                rd_active <= 1'b0;
                m_rvalid  <= 1'b0;
                m_rlast   <= 1'b0;
            end else begin
                rd_cnt    <= rd_cnt + 1'b1;
                m_rdata_r <= dram[((rd_base >> 2) + rd_cnt + 1'b1) % `DRAM_SIZE];
                m_rlast   <= ((rd_cnt + 1'b1) >= rd_len_r);
            end
        end else if (!m_rvalid) begin
            m_rvalid  <= 1'b1;
            m_rdata_r <= dram[((rd_base >> 2) + rd_cnt) % `DRAM_SIZE];
            m_rlast   <= (rd_cnt >= rd_len_r);
        end
    end
end

// ---------------------------------------------------------------------------
// AXI4 DRAM write model
// ---------------------------------------------------------------------------
reg [31:0] wr_base;
reg [7:0]  wr_len_r, wr_cnt;
reg        wr_phase, b_pending;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        m_awready <= 1'b0;
        m_arready <= 1'b0;
    end else begin
        m_awready <= 1'b1;
        m_arready <= 1'b1;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_phase   <= 1'b0;
        wr_cnt     <= 8'd0;
        m_wready   <= 1'b0;
        m_bvalid_r <= 1'b0;
        b_pending  <= 1'b0;
    end else begin
        if (m_awvalid && m_awready && !wr_phase) begin
            wr_phase <= 1'b1;
            wr_base  <= m_awaddr;
            wr_len_r <= m_awlen;
            wr_cnt   <= 8'd0;
            m_wready <= 1'b1;
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

        if (b_pending && !m_bvalid_r) begin
            m_bvalid_r <= 1'b1;
            b_pending  <= 1'b0;
        end else if (m_bvalid_r && m_bready) begin
            m_bvalid_r <= 1'b0;
        end
    end
end

// ---------------------------------------------------------------------------
// DUT
// ---------------------------------------------------------------------------
npu_top #(
    .PHY_ROWS(ROWS),
    .PHY_COLS(COLS),
    .DATA_W(DATA_W),
    .ACC_W(ACC_W)
) u_npu (
    .sys_clk       (clk),
    .sys_rst_n     (rst_n),
    .s_axi_awaddr  (s_awaddr),  .s_axi_awvalid (s_awvalid),
    .s_axi_awready (s_awready),
    .s_axi_wdata   (s_wdata),   .s_axi_wstrb   (s_wstrb),
    .s_axi_wvalid  (s_wvalid),  .s_axi_wready  (s_wready),
    .s_axi_bresp   (s_bresp),   .s_axi_bvalid  (s_bvalid),
    .s_axi_bready  (s_bready),
    .s_axi_araddr  (s_araddr),  .s_axi_arvalid (s_arvalid),
    .s_axi_arready (s_arready),
    .s_axi_rdata   (s_rdata),   .s_axi_rresp   (s_rresp),
    .s_axi_rvalid  (s_rvalid),  .s_axi_rready  (s_rready),
    .m_axi_awaddr  (m_awaddr),  .m_axi_awlen   (m_awlen),
    .m_axi_awsize  (m_awsize),  .m_axi_awburst (m_awburst),
    .m_axi_awvalid (m_awvalid), .m_axi_awready (m_awready),
    .m_axi_wdata   (m_wdata),   .m_axi_wstrb   (m_wstrb),
    .m_axi_wlast   (m_wlast),   .m_axi_wvalid  (m_wvalid),
    .m_axi_wready  (m_wready),
    .m_axi_bresp   (m_bresp),   .m_axi_bvalid  (m_bvalid),
    .m_axi_bready  (m_bready),
    .m_axi_araddr  (m_araddr),  .m_axi_arlen   (m_arlen),
    .m_axi_arsize  (m_arsize),  .m_axi_arburst (m_arburst),
    .m_axi_arvalid (m_arvalid), .m_axi_arready (m_arready),
    .m_axi_rdata   (m_rdata),   .m_axi_rresp   (m_rresp),
    .m_axi_rvalid  (m_rvalid),  .m_axi_rready  (m_rready),
    .m_axi_rlast   (m_rlast),
    .npu_irq       (npu_irq)
);

// ---------------------------------------------------------------------------
// BFM tasks
// ---------------------------------------------------------------------------
reg [31:0] rdata_tmp;
integer bfm_cnt;

task axi_write;
    input [31:0] addr;
    input [31:0] data;
    begin
        s_awaddr  <= addr;
        s_awvalid <= 1'b1;
        s_wdata   <= data;
        s_wstrb   <= 4'hF;
        s_wvalid  <= 1'b1;
        s_bready  <= 1'b1;
        @(posedge clk);
        @(posedge clk);
        s_awvalid <= 1'b0;
        s_wvalid  <= 1'b0;
        bfm_cnt = 0;
        while (!s_bvalid && bfm_cnt < 200) begin
            @(posedge clk);
            bfm_cnt = bfm_cnt + 1;
        end
        s_bready <= 1'b0;
    end
endtask

task axi_read;
    input  [31:0] addr;
    output [31:0] data;
    begin
        s_araddr  <= addr;
        s_arvalid <= 1'b1;
        s_rready  <= 1'b1;
        @(posedge clk);
        @(posedge clk);
        s_arvalid <= 1'b0;
        bfm_cnt = 0;
        while (!s_rvalid && bfm_cnt < 200) begin
            @(posedge clk);
            bfm_cnt = bfm_cnt + 1;
        end
        data = s_rdata;
        s_rready <= 1'b0;
    end
endtask

task wait_done;
    input [31:0] timeout;
    integer wdg;
    begin
        wdg = 0;
        rdata_tmp = 32'd0;
        while (!rdata_tmp[1] && wdg < timeout) begin
            @(posedge clk);
            axi_read(REG_STATUS, rdata_tmp);
            wdg = wdg + 1;
        end

        if (rdata_tmp[1]) begin
            axi_write(REG_CTRL, 32'h0);
        end else begin
            $display("    *** TIMEOUT *** ctrl_state=%0d dma_state=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.load_state);
            fail_cnt = fail_cnt + 1;
        end
    end
endtask

task run_layer;
    input [31:0] layer_id;
    input [31:0] m_dim;
    input [31:0] n_dim;
    input [31:0] k_dim;
    input [31:0] w_addr;
    input [31:0] a_addr;
    input [31:0] r_addr;
    input [31:0] bias_addr;
    input [31:0] quant_cfg;
    input [31:0] ctrl_val;
    input        conv_en;
    input [31:0] conv_ifm_shape;
    input [31:0] conv_channels;
    input [31:0] conv_kernel;
    input [31:0] conv_out_shape;
    input [31:0] conv_stride_pad;
    input [31:0] conv_dilation;
    begin
        $display("");
        $display("  Running layer%0d: M=%0d N=%0d K=%0d W=0x%08h A=0x%08h R=0x%08h CTRL=0x%08h",
                 layer_id, m_dim, n_dim, k_dim, w_addr, a_addr, r_addr, ctrl_val);

        axi_write(REG_CTRL, 32'h0);
        axi_write(REG_M_DIM, m_dim);
        axi_write(REG_N_DIM, n_dim);
        axi_write(REG_K_DIM, k_dim);
        axi_write(REG_W_ADDR, w_addr);
        axi_write(REG_A_ADDR, a_addr);
        axi_write(REG_R_ADDR, r_addr);
        axi_write(REG_BIAS_ADDR, bias_addr);
        axi_write(REG_QUANT_CFG, quant_cfg);

        if (conv_en) begin
            axi_write(REG_CONV_IFM_SHAPE,  conv_ifm_shape);
            axi_write(REG_CONV_CHANNELS,   conv_channels);
            axi_write(REG_CONV_KERNEL,     conv_kernel);
            axi_write(REG_CONV_OUT_SHAPE,  conv_out_shape);
            axi_write(REG_CONV_STRIDE_PAD, conv_stride_pad);
            axi_write(REG_CONV_DILATION,   conv_dilation);
        end else begin
            axi_write(REG_CONV_IFM_SHAPE,  32'h0);
            axi_write(REG_CONV_CHANNELS,   32'h0);
            axi_write(REG_CONV_KERNEL,     32'h0);
            axi_write(REG_CONV_OUT_SHAPE,  32'h0);
            axi_write(REG_CONV_STRIDE_PAD, 32'h0);
            axi_write(REG_CONV_DILATION,   32'h0);
        end

        axi_write(REG_CTRL, ctrl_val);
        wait_done((m_dim * n_dim * k_dim * 10000) + 200000);
    end
endtask

// ---------------------------------------------------------------------------
// Result checking
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;
integer i;
reg [31:0] got_val, exp_val;

task check_layer0;
    begin
        $display("  Checking layer0 intermediate OFM at 0x%08h", `L0_R_ADDR);
        for (i = 0; i < `L0_NUM_RESULTS; i = i + 1) begin
            got_val = dram[(`L0_R_ADDR >> 2) + i];
            exp_val = layer0_expected[i];
            if (got_val === exp_val) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("    [FAIL] L0[%0d]: got=%0d (0x%08h) exp=%0d (0x%08h)",
                         i, $signed(got_val), got_val, $signed(exp_val), exp_val);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

task check_layer1;
    integer idx;
    begin
        $display("  Checking layer1 final OFM at 0x%08h", `L1_R_ADDR);
        for (idx = 0; idx < `L1_NUM_RESULTS; idx = idx + 1) begin
            got_val = dram[(`L1_R_ADDR >> 2) + idx];
            exp_val = expected[idx];
            if (got_val === exp_val) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("    [FAIL] L1[%0d]: got=%0d (0x%08h) exp=%0d (0x%08h)",
                         idx, $signed(got_val), got_val, $signed(exp_val), exp_val);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
initial begin
    s_awaddr<=0; s_awvalid<=0; s_wdata<=0; s_wstrb<=0; s_wvalid<=0; s_bready<=0;
    s_araddr<=0; s_arvalid<=0; s_rready<=0;
    m_awready<=1; m_wready<=0;
    pass_cnt=0; fail_cnt=0;

    `ifdef DUMP_VCD
    $dumpfile("tb_conv2d_two_layer.vcd");
    $dumpvars(0, tb_conv2d_two_layer);
    `endif

    @(posedge rst_n);
    repeat (3) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  T6.6 Two-Layer Conv2D End-to-End Test");
    $display("  Layer0 R_ADDR = 0x%08h", `L0_R_ADDR);
    $display("  Layer1 A_ADDR = 0x%08h", `L1_A_ADDR);
    $display("################################################################");

    run_layer(
        32'd0,
        `L0_M_DIM, `L0_N_DIM, `L0_K_DIM,
        `L0_W_ADDR, `L0_A_ADDR, `L0_R_ADDR,
        `L0_BIAS_ADDR, `L0_QUANT_CFG, `L0_CTRL,
        1'b1,
        `L0_CONV_IFM_SHAPE, `L0_CONV_CHANNELS, `L0_CONV_KERNEL,
        `L0_CONV_OUT_SHAPE, `L0_CONV_STRIDE_PAD, `L0_CONV_DILATION
    );
    check_layer0();

    run_layer(
        32'd1,
        `L1_M_DIM, `L1_N_DIM, `L1_K_DIM,
        `L1_W_ADDR, `L1_A_ADDR, `L1_R_ADDR,
        `L1_BIAS_ADDR, `L1_QUANT_CFG, `L1_CTRL,
        1'b0,
        32'h0, 32'h0, 32'h0, 32'h0, 32'h0, 32'h0
    );
    check_layer1();

    $display("");
    $display("################################################################");
    if (fail_cnt == 0)
        $display("  ALL %0d CHECKS PASSED", pass_cnt);
    else
        $display("  RESULT: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    $display("################################################################");

    #(CLK_T*20);
    $finish;
end

initial begin
    #(CLK_T * 10000000);
    $display("\nFATAL: Global timeout!");
    $finish;
end

endmodule
