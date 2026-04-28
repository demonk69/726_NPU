// =============================================================================
// tb_matmul_os.v - Matrix multiplication testbench (OS mode)
//
// Verifies C[M×N] = A[M×K] × B[K×N] using the tile-loop OS architecture.
// Uses gen_matmul_data.py generated data files.
//
// Parameterized via `include test_params.vh.
// =============================================================================

`timescale 1ns/1ps

module tb_matmul_os;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
`include "test_params.vh"

parameter ROWS   = 1;
parameter COLS   = 1;
parameter DATA_W = 16;
parameter ACC_W  = 32;
parameter CLK_T  = 10;

// Register addresses (from npu_axi_lite)
localparam REG_CTRL   = 32'h00;
localparam REG_STATUS = 32'h04;
localparam REG_INT_EN = 32'h08;
localparam REG_INT_CLR= 32'h0C;
localparam REG_M_DIM  = 32'h10;
localparam REG_N_DIM  = 32'h14;
localparam REG_K_DIM  = 32'h18;
localparam REG_W_ADDR = 32'h20;
localparam REG_A_ADDR = 32'h24;
localparam REG_R_ADDR = 32'h28;

// ---------------------------------------------------------------------------
// DRAM and Expected arrays
// ---------------------------------------------------------------------------
reg [31:0] dram [0:`DRAM_SIZE-1];
reg [31:0] expected [0:`NUM_RESULTS-1];

initial begin
    $readmemh("dram_init.hex", dram);
    $readmemh("expected.hex", expected);
end

// ---------------------------------------------------------------------------
// Clock & Reset
// ---------------------------------------------------------------------------
reg clk = 0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 0;
initial begin #(CLK_T*5); rst_n = 1; end;

// ---------------------------------------------------------------------------
// AXI4-Lite Signals
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
// AXI4 Master Signals (DMA <-> DRAM)
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

// ---------------------------------------------------------------------------
// AXI4 DRAM Read Model
// ---------------------------------------------------------------------------
reg [31:0] rd_base, rd_len_r;
reg [7:0]  rd_cnt;
reg        rd_active;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_active <= 0; rd_cnt <= 0;
        m_rvalid  <= 0; m_rlast <= 0; m_rdata_r <= 0;
    end else if (!rd_active && m_arvalid && m_arready) begin
        rd_active <= 1;
        rd_base   <= m_araddr;
        rd_len_r  <= m_arlen;
        rd_cnt    <= 0;
    end else if (rd_active) begin
        if (m_rvalid && m_rready) begin
            if (m_rlast || rd_cnt >= rd_len_r) begin
                rd_active <= 0; m_rvalid <= 0; m_rlast <= 0;
            end else begin
                rd_cnt    <= rd_cnt + 1;
                m_rdata_r <= dram[((rd_base >> 2) + rd_cnt + 1) % `DRAM_SIZE];
                m_rlast   <= ((rd_cnt + 1) >= rd_len_r);
            end
        end else if (!m_rvalid) begin
            m_rvalid  <= 1;
            m_rdata_r <= dram[((rd_base >> 2) + rd_cnt) % `DRAM_SIZE];
            m_rlast   <= (rd_cnt >= rd_len_r);
        end
    end
end

// ---------------------------------------------------------------------------
// AXI4 DRAM Write Model
// ---------------------------------------------------------------------------
reg [31:0] wr_base;
reg [7:0]  wr_len_r, wr_cnt;
reg        wr_phase, b_pending;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        m_awready <= 0; m_arready <= 0;
    end else begin
        m_awready <= 1; m_arready <= 1;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_phase <= 0; wr_cnt <= 0;
        m_wready <= 0; m_bvalid_r <= 0; b_pending <= 0;
    end else begin
        if (m_awvalid && m_awready && !wr_phase) begin
            wr_phase <= 1; wr_base <= m_awaddr;
            wr_len_r <= m_awlen; wr_cnt <= 0; m_wready <= 1;
        end
        if (wr_phase && m_wvalid && m_wready) begin
            dram[((wr_base >> 2) + wr_cnt) % `DRAM_SIZE] <= m_wdata;
            wr_cnt <= wr_cnt + 1;
            if (m_wlast) begin
                wr_phase <= 0; m_wready <= 0; b_pending <= 1;
            end
        end
        if (b_pending && !m_bvalid_r) begin
            m_bvalid_r <= 1; b_pending <= 0;
        end else if (m_bvalid_r && m_bready) begin
            m_bvalid_r <= 0;
        end
    end
end

assign m_bvalid = m_bvalid_r;

// ---------------------------------------------------------------------------
// NPU DUT
// ---------------------------------------------------------------------------
npu_top #(
    .PHY_ROWS(ROWS), .PHY_COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)
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
    .m_axi_bresp   (m_bresp),   .m_axi_bvalid   (m_bvalid),
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
// BFM Tasks
// ---------------------------------------------------------------------------
reg [31:0] rdata_tmp;
integer    bfm_cnt;

task axi_write;
    input [31:0] addr, data;
    begin
        s_awaddr <= addr; s_awvalid <= 1;
        s_wdata  <= data; s_wstrb   <= 4'hF; s_wvalid <= 1;
        s_bready <= 1;
        @(posedge clk); @(posedge clk);
        s_awvalid <= 0; s_wvalid <= 0;
        bfm_cnt = 0;
        while (!s_bvalid && bfm_cnt < 200) begin
            @(posedge clk); bfm_cnt = bfm_cnt + 1;
        end
        s_bready <= 0;
    end
endtask

task axi_read;
    input  [31:0] addr;
    output [31:0] data;
    begin
        s_araddr <= addr; s_arvalid <= 1; s_rready <= 1;
        @(posedge clk); @(posedge clk);
        s_arvalid <= 0;
        bfm_cnt = 0;
        while (!s_rvalid && bfm_cnt < 200) begin
            @(posedge clk); bfm_cnt = bfm_cnt + 1;
        end
        data = s_rdata;
        s_rready <= 0;
    end
endtask

task wait_done;
    input [31:0] timeout;
    integer wdg;
    begin
        wdg = 0; rdata_tmp = 0;
        while (!rdata_tmp[1] && wdg < timeout) begin
            @(posedge clk);
            axi_read(REG_STATUS, rdata_tmp);
            wdg = wdg + 1;
        end
        if (rdata_tmp[1])
            axi_write(REG_CTRL, 32'h00);
        else begin
            $display("    *** TIMEOUT *** ctrl_state=%0d dma_state=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
        end
    end
endtask

// ---------------------------------------------------------------------------
// FP32 ULP comparison
// ---------------------------------------------------------------------------
function [31:0] fp32_ordered;
    input [31:0] x;
    begin
        if (x[31]) fp32_ordered = ~x;
        else       fp32_ordered = x | 32'h8000_0000;
    end
endfunction

function fp32_close;
    input [31:0] got, exp;
    reg [31:0] got_ord, exp_ord, ulp_diff;
    reg gz, ez, got_nan, exp_nan, got_inf, exp_inf;
    begin
        gz      = (got[30:0] == 0);
        ez      = (exp[30:0] == 0);
        got_nan = (&got[30:23]) && (got[22:0] != 0);
        exp_nan = (&exp[30:23]) && (exp[22:0] != 0);
        got_inf = (&got[30:23]) && (got[22:0] == 0);
        exp_inf = (&exp[30:23]) && (exp[22:0] == 0);

        if (got_nan || exp_nan) fp32_close = 0;
        else if (gz && ez) fp32_close = 1;
        else if (got_inf || exp_inf) fp32_close = (got === exp);
        else begin
            got_ord = fp32_ordered(got);
            exp_ord = fp32_ordered(exp);
            ulp_diff = (got_ord > exp_ord) ? (got_ord - exp_ord) : (exp_ord - got_ord);
            fp32_close = (ulp_diff <= 32'd8);
        end
    end
endfunction

// ---------------------------------------------------------------------------
// Matrix multiplication test
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;
reg [31:0] got_val;
integer i, j;

task do_matmul_test;
    input [31:0] m_dim, n_dim, k_dim_val;
    input [31:0] w_addr, a_addr, r_addr, ctrl_val;
    input [31:0] timeout;
    input is_fp16;
    integer idx;
    reg [31:0] exp_val;
    begin
        // Configure NPU
        axi_write(REG_M_DIM, m_dim);
        axi_write(REG_N_DIM, n_dim);
        axi_write(REG_K_DIM, k_dim_val);
        axi_write(REG_W_ADDR, w_addr);
        axi_write(REG_A_ADDR, a_addr);
        axi_write(REG_R_ADDR, r_addr);

        // Debug: verify register writes took effect
        axi_read(REG_M_DIM, rdata_tmp); $display("    M_DIM readback = %0d", rdata_tmp);
        axi_read(REG_N_DIM, rdata_tmp); $display("    N_DIM readback = %0d", rdata_tmp);
        axi_read(REG_K_DIM, rdata_tmp); $display("    K_DIM readback = %0d", rdata_tmp);

        axi_write(REG_CTRL, ctrl_val);

        // Wait for completion
        wait_done(timeout);

        // Debug: print NPU state
        $display("    NPU ctrl state=%0d, tile_i=%0d, tile_j=%0d, tile_count=%0d/%0d",
                 u_npu.u_ctrl.state, u_npu.u_ctrl.tile_i, u_npu.u_ctrl.tile_j,
                 u_npu.u_ctrl.tile_count, u_npu.u_ctrl.tile_total);
        $display("    DMA state=%0d, r_pending=%0d",
                 u_npu.u_dma.dma_state, u_npu.u_dma.r_pending);

        // Check all M×N results
        $display("    Checking %0dx%0d results...", m_dim, n_dim);
        for (i = 0; i < m_dim; i = i + 1) begin
            for (j = 0; j < n_dim; j = j + 1) begin
                idx = i * n_dim + j;
                got_val = dram[(r_addr >> 2) + idx];
                exp_val = expected[idx];
                $display("    C[%0d][%0d]: addr=0x%08h got=0x%08h exp=0x%08h",
                         i, j, r_addr + idx * 4, got_val, exp_val);

                if (is_fp16) begin
                    if (fp32_close(got_val, exp_val)) begin
                        pass_cnt = pass_cnt + 1;
                    end else begin
                        $display("    [FAIL] C[%0d][%0d]: got=0x%08h exp=0x%08h",
                                 i, j, got_val, exp_val);
                        fail_cnt = fail_cnt + 1;
                    end
                end else begin
                    if (got_val === exp_val) begin
                        pass_cnt = pass_cnt + 1;
                    end else begin
                        $display("    [FAIL] C[%0d][%0d]: got=%0d (0x%08h) exp=%0d (0x%08h)",
                                 i, j, $signed(got_val), got_val,
                                 $signed(exp_val), exp_val);
                        fail_cnt = fail_cnt + 1;
                    end
                end
            end
        end
        $display("    Results: %0d pass, %0d fail", pass_cnt, fail_cnt);
    end
endtask

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
initial begin
    s_awaddr<=0;s_awvalid<=0;s_wdata<=0;s_wstrb<=0;s_wvalid<=0;s_bready<=0;
    s_araddr<=0;s_arvalid<=0;s_rready<=0;
    m_awready<=1; m_wready<=0;
    pass_cnt=0; fail_cnt=0;

    `ifdef DUMP_VCD
    $dumpfile("tb_matmul_os.vcd");
    $dumpvars(0, tb_matmul_os);
    `endif

    @(posedge rst_n);
    repeat(3) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  Matrix Multiplication OS Test: %0dx%0d x %0dx%0d = %0dx%0d",
             `M_DIM, `K_DIM, `K_DIM, `N_DIM, `M_DIM, `N_DIM);
    $display("  Data type: %0s", `IS_FP16 ? "FP16" : "INT8");
    $display("################################################################");

    do_matmul_test(
        `M_DIM, `N_DIM, `K_DIM,
        `W_ADDR, `A_ADDR, `R_ADDR,
        `CTRL,
        `M_DIM * `N_DIM * `K_DIM * 10000 + 200000,
        `IS_FP16
    );

    // Summary
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

// Global timeout
initial begin
    #(CLK_T * 10000000);
    $display("\nFATAL: Global timeout!");
    $finish;
end

endmodule
