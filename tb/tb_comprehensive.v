// =============================================================================
// Module  : tb_comprehensive
// Project : NPU_prj
// Desc    : Comprehensive end-to-end NPU simulation test suite.
//           Tests 6 scenarios covering edge cases, multi-result, and
//           back-to-back operation.
//
//   Test 1: INT8 正数点积 (K=8, 全正)
//   Test 2: INT8 混合正负点积 (K=8, 有负数)
//   Test 3: INT8 边界值 (INT8_MAX/INT8_MIN)
//   Test 4: INT8 零值 (全零权重, 验证结果为0)
//   Test 5: INT8 交替正负 (K=16, 交替 +-1)
//   Test 6: 连续运算 (两次启动, 验证状态机复位)
// =============================================================================

`timescale 1ns/1ps

module tb_comprehensive;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
localparam ROWS    = 1;
localparam COLS    = 1;
localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;
localparam DRAM_SZ = 4096;

// NPU AXI-Lite register map
localparam REG_CTRL    = 32'h00;
localparam REG_STATUS  = 32'h04;
localparam REG_M_DIM   = 32'h10;
localparam REG_N_DIM   = 32'h14;
localparam REG_K_DIM   = 32'h18;
localparam REG_W_ADDR  = 32'h20;
localparam REG_A_ADDR  = 32'h24;
localparam REG_R_ADDR  = 32'h28;

localparam CTRL_START  = 32'h01;
localparam CTRL_OS     = 32'h10;  // bit[4]=1 → OS mode

// ---------------------------------------------------------------------------
// Clock & Reset
// ---------------------------------------------------------------------------
reg clk = 0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 0;
initial begin #(CLK_T*5); rst_n = 1; end

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
wire [31:0]        m_awaddr;
wire [7:0]         m_awlen;
wire [2:0]         m_awsize;
wire [1:0]         m_awburst;
wire               m_awvalid;
reg                m_awready;
wire [ACC_W-1:0]   m_wdata;
wire [ACC_W/8-1:0] m_wstrb;
wire               m_wlast;
wire               m_wvalid;
reg                m_wready;
wire [1:0]         m_bresp_out;
reg                m_bvalid;
wire               m_bready;
wire [31:0]        m_araddr;
wire [7:0]         m_arlen;
wire [2:0]         m_arsize;
wire [1:0]         m_arburst;
wire               m_arvalid;
reg                m_arready;
reg  [ACC_W-1:0]   m_rdata_r;
wire [ACC_W-1:0]   m_rdata;
wire [1:0]         m_rresp;
reg                m_rvalid;
reg                m_rlast;
wire               m_rready;
wire               npu_irq;

assign m_bresp_out = 2'b00;
assign m_rresp     = 2'b00;
assign m_rdata     = m_rdata_r;

// ---------------------------------------------------------------------------
// DRAM Model (32-bit word addressed)
// ---------------------------------------------------------------------------
reg [ACC_W-1:0] dram [0:DRAM_SZ-1];

integer dram_i;
initial begin
    for (dram_i = 0; dram_i < DRAM_SZ; dram_i = dram_i+1)
        dram[dram_i] = 32'h0;

    // =================================================================
    // Test 1: INT8 正数点积 K=8
    //   W = [1, 2, 3, 4, 5, 6, 7, 8]
    //   A = [10, 20, 30, 40, 50, 60, 70, 80]
    //   Expected: 1*10 + 2*20 + 3*30 + 4*40 + 5*50 + 6*60 + 7*70 + 8*80
    //           = 10 + 40 + 90 + 160 + 250 + 360 + 490 + 640 = 2040
    //
    //   W = 2 words (8 INT8), A = 2 words (8 INT8)
    //   W at 0x2000..0x2007, A at 0x2020..0x2027 (separated by 0x20)
    //   Result at 0x2040
    // =================================================================
    // W: bytes [1,2,3,4] [5,6,7,8]
    dram[32'h2000 >> 2] = 32'h04030201;
    dram[(32'h2000 >> 2) + 1] = 32'h08070605;
    // A: bytes [10,20,30,40] = [0x0A,0x14,0x1E,0x28]
    dram[32'h2020 >> 2] = 32'h281E140A;
    dram[(32'h2020 >> 2) + 1] = 32'h50463C32;

    // =================================================================
    // Test 1b: K=4 regression (same as tb_classifier, expected 310)
    //   W = [3, 7, -2, 5], A = [10, 20, 30, 40]
    //   Expected: 3*10 + 7*20 + (-2)*30 + 5*40 = 310
    // =================================================================
    dram[32'h2080 >> 2] = 32'h05FE0703;
    dram[32'h2084 >> 2] = 32'h281E140A;

    // =================================================================
    // Test 2: INT8 混合正负点积 K=8
    //   W = [10, -20, 30, -40, 50, -60, 70, -80]
    //   A = [3,   5, -7,   9, -11,  13, -15,  17]
    //   Expected: 30-100-210-360-550-780-1050-1360 = -4380
    //   -4380 = 0xFFFFEEE4 in 32-bit signed
    // =================================================================
    dram[32'h2100 >> 2] = 32'hD81EEC0A;
    dram[(32'h2100 >> 2) + 1] = 32'hB046C432;
    dram[32'h2120 >> 2] = 32'h09F90503;
    dram[(32'h2120 >> 2) + 1] = 32'h11F10DF5;

    // =================================================================
    // Test 3: INT8 边界值 K=8
    //   W = [127, -128, 127, -128, 1, 0, -1, 0]
    //   A = [127, -128, -128, 127, 127, 1, 127, 1]
    //   Expected: 127*127+(-128)*(-128)+127*(-128)+(-128)*127+1*127+0*1+(-1)*127+0*1
    //           = 16129+16384-16256-16256+127+0-127+0 = 1
    //   A word1 bytes [127,1,127,1] = [0x7F,0x01,0x7F,0x01] = 0x017F017F
    // =================================================================
    dram[32'h2200 >> 2] = 32'h807F807F;
    dram[(32'h2200 >> 2) + 1] = 32'h00FF0001;
    dram[32'h2220 >> 2] = 32'h7F80807F;
    dram[(32'h2220 >> 2) + 1] = 32'h017F017F;

    // =================================================================
    // Test 4: INT8 零值 K=8
    //   W = [0,0,0,0,0,0,0,0] A = [10,20,30,40,50,60,70,80]
    //   Expected: 0
    // =================================================================
    dram[32'h2300 >> 2] = 32'h00000000;
    dram[(32'h2300 >> 2) + 1] = 32'h00000000;
    dram[32'h2320 >> 2] = 32'h281E140A;
    dram[(32'h2320 >> 2) + 1] = 32'h50463C32;

    // =================================================================
    // Test 5: INT8 交替正负 K=16 (4 words per vector)
    //   W = [1,-1,1,-1, 2,-2,2,-2, 3,-3,3,-3, 4,-4,4,-4]
    //   A = [1,-1,1,-1, 1,-1,1,-1, 1,-1,1,-1, 1,-1,1,-1]
    //   Each pair: 1*1+(-1)*(-1)=2, weighted: 2*(1+2+3+4)=20
    //   Expected: 20
    // =================================================================
    dram[32'h2400 >> 2]       = 32'hFF01FF01;
    dram[(32'h2400 >> 2) + 1] = 32'hFE02FE02;
    dram[(32'h2400 >> 2) + 2] = 32'hFD03FD03;
    dram[(32'h2400 >> 2) + 3] = 32'hFC04FC04;
    dram[32'h2440 >> 2]       = 32'hFF01FF01;   // A
    dram[(32'h2440 >> 2) + 1] = 32'hFF01FF01;
    dram[(32'h2440 >> 2) + 2] = 32'hFF01FF01;
    dram[(32'h2440 >> 2) + 3] = 32'hFF01FF01;

    // =================================================================
    // Test 6: 连续运算 — 使用 test 1 和 test 2 的数据
    // =================================================================
end

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
                rd_active <= 0;
                m_rvalid  <= 0;
                m_rlast   <= 0;
            end else begin
                rd_cnt    <= rd_cnt + 1;
                m_rdata_r <= dram[((rd_base >> 2) + rd_cnt + 1) % DRAM_SZ];
                m_rlast   <= ((rd_cnt + 1) >= rd_len_r);
            end
        end else if (!m_rvalid) begin
            m_rvalid  <= 1;
            m_rdata_r <= dram[((rd_base >> 2) + rd_cnt) % DRAM_SZ];
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
        m_awready <= 1;
        m_arready <= 1;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_phase <= 0; wr_cnt <= 0;
        m_wready <= 0; m_bvalid <= 0; b_pending <= 0;
    end else begin
        if (m_awvalid && m_awready && !wr_phase) begin
            wr_phase <= 1;
            wr_base  <= m_awaddr;
            wr_len_r <= m_awlen;
            wr_cnt   <= 0;
            m_wready <= 1;
        end
        if (wr_phase && m_wvalid && m_wready) begin
            dram[((wr_base >> 2) + wr_cnt) % DRAM_SZ] <= m_wdata;
            wr_cnt <= wr_cnt + 1;
            if (m_wlast) begin
                wr_phase  <= 0;
                m_wready  <= 0;
                b_pending <= 1;
            end
        end
        if (b_pending && !m_bvalid) begin
            m_bvalid  <= 1;
            b_pending <= 0;
        end else if (m_bvalid && m_bready) begin
            m_bvalid <= 0;
        end
    end
end

// ---------------------------------------------------------------------------
// NPU DUT
// ---------------------------------------------------------------------------
npu_top #(
    .ROWS(ROWS), .COLS(COLS), .DATA_W(DATA_W), .ACC_W(ACC_W)
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
    .m_axi_bresp   (m_bresp_out), .m_axi_bvalid (m_bvalid),
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
    input [31:0] addr;
    input [31:0] data;
    begin
        s_awaddr <= addr;  s_awvalid <= 1;
        s_wdata  <= data;  s_wstrb   <= 4'hF; s_wvalid <= 1;
        s_bready <= 1;
        @(posedge clk);
        @(posedge clk);
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
        @(posedge clk);
        @(posedge clk);
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
    input [31:0] timeout_cyc;
    integer wdg;
    begin
        wdg = 0;
        rdata_tmp = 0;
        while (!rdata_tmp[1] && wdg < timeout_cyc) begin
            @(posedge clk);
            axi_read(REG_STATUS, rdata_tmp);
            wdg = wdg + 1;
        end
        if (rdata_tmp[1]) begin
            // Clear ctrl_reg to deassert done
            axi_write(REG_CTRL, 32'h00);
        end else begin
            $display("    *** TIMEOUT after %0d cycles! ***", timeout_cyc);
            $display("    ctrl_state=%0d, dma_state=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $finish;
        end
    end
endtask

// Run one NPU inference: configure, start, wait
// Positional args: m_dim, n_dim, k_dim, w_addr, a_addr, r_addr, ctrl_val, timeout
task run_npu;
    input [31:0] t_m_dim;
    input [31:0] t_n_dim;
    input [31:0] t_k_dim;
    input [31:0] t_w_addr;
    input [31:0] t_a_addr;
    input [31:0] t_r_addr;
    input [31:0] t_ctrl_val;
    input [31:0] t_timeout;
    begin
        axi_write(REG_M_DIM, t_m_dim);
        axi_write(REG_N_DIM, t_n_dim);
        axi_write(REG_K_DIM, t_k_dim);
        axi_write(REG_W_ADDR, t_w_addr);
        axi_write(REG_A_ADDR, t_a_addr);
        axi_write(REG_R_ADDR, t_r_addr);
        axi_write(REG_CTRL, t_ctrl_val);
        wait_done(t_timeout);
    end
endtask

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;
reg [31:0] got_val, exp_val;

initial begin
    s_awaddr<=0;s_awvalid<=0;s_wdata<=0;s_wstrb<=0;s_wvalid<=0;s_bready<=0;
    s_araddr<=0;s_arvalid<=0;s_rready<=0;
    m_awready<=1; m_wready<=0;
    pass_cnt=0; fail_cnt=0;

    `ifdef DUMP_VCD
    $dumpfile("tb_comprehensive.vcd");
    $dumpvars(0, tb_comprehensive);
    `endif

    @(posedge rst_n);
    repeat(3) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  NPU Comprehensive Test Suite");
    $display("  PE Array: %0dx%0d, DATA_W=%0d, ACC_W=%0d", ROWS, COLS, DATA_W, ACC_W);
    $display("################################################################");

    // =================================================================
    // Test 1: INT8 正数点积 K=8
    //   W = [1,2,3,4,5,6,7,8], A = [10,20,30,40,50,60,70,80]
    //   Expected: 2040
    // =================================================================
    $display("");
    $display("--- Test 1: INT8 Positive Dot Product (K=8) ---");
    $display("  W=[1,2,3,4,5,6,7,8] A=[10,20,30,40,50,60,70,80]");
    $display("  Expected: 2040");
    run_npu(32'd1, 32'd1, 32'd8,
            32'h2000, 32'h2020, 32'h2040,
            CTRL_START | CTRL_OS, 100000);
    got_val = dram[32'h2040 >> 2];
    exp_val = 32'd2040;
    if (got_val === exp_val) begin
        $display("  [PASS] T1_PosDotK8: got %0d (0x%08h)", $signed(got_val), got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T1_PosDotK8: got %0d (0x%08h), exp %0d, diff=%0d",
                 $signed(got_val), got_val, $signed(exp_val), $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 1b: K=4 regression
    // =================================================================
    $display("");
    $display("--- Test 1b: K=4 Regression (expected 310) ---");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h2080, 32'h2084, 32'h2088,
            CTRL_START | CTRL_OS, 100000);
    got_val = dram[32'h2088 >> 2];
    exp_val = 32'd310;
    if (got_val === exp_val) begin
        $display("  [PASS] T1b_K4_Regress: got %0d", $signed(got_val));
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T1b_K4_Regress: got %0d, exp %0d, diff=%0d",
                 $signed(got_val), $signed(exp_val), $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 2: INT8 混合正负 K=8
    //   Expected: -4380 (0xFFFFEEA4)
    // =================================================================
    $display("");
    $display("--- Test 2: INT8 Mixed Signed Dot Product (K=8) ---");
    $display("  W=[10,-20,30,-40,50,-60,70,-80]");
    $display("  A=[3,5,-7,9,-11,13,-15,17]");
    $display("  Expected: -4380");
    run_npu(32'd1, 32'd1, 32'd8,
            32'h2100, 32'h2120, 32'h2140,
            CTRL_START | CTRL_OS, 100000);
    got_val = dram[32'h2140 >> 2];
    exp_val = 32'hFFFFEEE4;  // -4380 signed
    if (got_val === exp_val) begin
        $display("  [PASS] T2_MixDotK8: got %0d (0x%08h)", $signed(got_val), got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T2_MixDotK8: got %0d (0x%08h), exp %0d (0x%08h), diff=%0d",
                 $signed(got_val), got_val, $signed(exp_val), exp_val,
                 $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 3: INT8 边界值 K=8
    //   W=[127,-128,127,-128,1,0,-1,0], A=[127,-128,-128,127,127,1,127,1]
    //   Expected: 1
    // =================================================================
    $display("");
    $display("--- Test 3: INT8 Boundary Values (K=8) ---");
    $display("  W=[127,-128,127,-128,1,0,-1,0]");
    $display("  A=[127,-128,-128,127,127,1,127,1]");
    $display("  Expected: 1");
    run_npu(32'd1, 32'd1, 32'd8,
            32'h2200, 32'h2220, 32'h2240,
            CTRL_START | CTRL_OS, 100000);
    got_val = dram[32'h2240 >> 2];
    exp_val = 32'd1;
    if (got_val === exp_val) begin
        $display("  [PASS] T3_BoundaryK8: got %0d (0x%08h)", $signed(got_val), got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T3_BoundaryK8: got %0d (0x%08h), exp %0d, diff=%0d",
                 $signed(got_val), got_val, $signed(exp_val), $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 4: INT8 零值权重 K=8
    //   W=[0,0,0,0,0,0,0,0] A=[10,20,30,40,50,60,70,80]
    //   Expected: 0
    // =================================================================
    $display("");
    $display("--- Test 4: INT8 Zero Weights (K=8) ---");
    $display("  W=[0,0,0,0,0,0,0,0] A=[10,20,30,40,50,60,70,80]");
    $display("  Expected: 0");
    run_npu(32'd1, 32'd1, 32'd8,
            32'h2300, 32'h2320, 32'h2340,
            CTRL_START | CTRL_OS, 100000);
    got_val = dram[32'h2340 >> 2];
    exp_val = 32'd0;
    if (got_val === exp_val) begin
        $display("  [PASS] T4_ZeroWeights: got %0d (0x%08h)", $signed(got_val), got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T4_ZeroWeights: got %0d (0x%08h), exp %0d, diff=%0d",
                 $signed(got_val), got_val, $signed(exp_val), $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 5: INT8 交替正负 K=16 (4 words per vector)
    //   W = [1,-1,1,-1, 2,-2,2,-2, 3,-3,3,-3, 4,-4,4,-4]
    //   A = [1,-1,1,-1, 1,-1,1,-1, 1,-1,1,-1, 1,-1,1,-1]
    //   Each pair W[i]*A[i]: 1+1+1+1+2+2+2+2+3+3+3+3+4+4+4+4 = 40
    //   Expected: 40
    // =================================================================
    $display("");
    $display("--- Test 5: INT8 Alternating Sign (K=16) ---");
    $display("  W=[1,-1,1,-1,2,-2,2,-2,3,-3,3,-3,4,-4,4,-4]");
    $display("  A=[1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]");
    $display("  Expected: 40");
    run_npu(32'd1, 32'd1, 32'd16,
            32'h2400, 32'h2440, 32'h2480,
            CTRL_START | CTRL_OS, 200000);
    got_val = dram[32'h2480 >> 2];
    exp_val = 32'd40;
    if (got_val === exp_val) begin
        $display("  [PASS] T5_Alternating: got %0d (0x%08h)", $signed(got_val), got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T5_Alternating: got %0d (0x%08h), exp %0d, diff=%0d",
                 $signed(got_val), got_val, $signed(exp_val), $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 6: 连续运算 (back-to-back)
    //   Run 1: same as test 1 → expected 2040
    //   Run 2: same as test 2 → expected -4380 (0xFFFFEEA4)
    // =================================================================
    $display("");
    $display("--- Test 6: Back-to-Back Operations ---");
    $display("  Run 1: same as Test 1 -> expected 2040");
    $display("  Run 2: same as Test 2 -> expected -4380 (0xFFFFEEE4)");

    // Run 1
    run_npu(32'd1, 32'd1, 32'd8,
            32'h2000, 32'h2020, 32'h2500,
            CTRL_START | CTRL_OS, 100000);
    got_val = dram[32'h2500 >> 2];
    exp_val = 32'd2040;
    if (got_val === exp_val) begin
        $display("  [PASS] T6_Back2Back_1: got %0d (0x%08h)", $signed(got_val), got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T6_Back2Back_1: got %0d (0x%08h), exp %0d, diff=%0d",
                 $signed(got_val), got_val, $signed(exp_val), $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // Run 2
    run_npu(32'd1, 32'd1, 32'd8,
            32'h2100, 32'h2120, 32'h2504,
            CTRL_START | CTRL_OS, 100000);
    got_val = dram[32'h2504 >> 2];
    exp_val = 32'hFFFFEEE4;  // -4380
    if (got_val === exp_val) begin
        $display("  [PASS] T6_Back2Back_2: got %0d (0x%08h)", $signed(got_val), got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T6_Back2Back_2: got %0d (0x%08h), exp %0d (0x%08h), diff=%0d",
                 $signed(got_val), got_val, $signed(exp_val), exp_val,
                 $signed(got_val)-$signed(exp_val));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Summary
    // =================================================================
    $display("");
    $display("################################################################");
    if (fail_cnt == 0)
        $display("  ALL %0d TESTS PASSED", pass_cnt);
    else
        $display("  RESULT: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    $display("################################################################");

    #(CLK_T*20);
    $finish;
end

// Global timeout
initial begin
    #(CLK_T * 2000000);
    $display("\nFATAL: Global timeout (2M cycles)!");
    $finish;
end

endmodule
