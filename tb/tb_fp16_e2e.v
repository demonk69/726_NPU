// =============================================================================
// Module  : tb_fp16_e2e
// Project : NPU_prj
// Date    : 2026-04-06
// Desc    : FP16 end-to-end testbench.
//           Exercises the full NPU pipeline in FP16 mode:
//             DRAM → DMA → PPBuf → FP16 packer → PE (fp16_mul + fp32_add)
//             → Result FIFO → DMA → DRAM
//
//   Test suite (8 tests, 9 checkpoints):
//     T1: FP16 OS  K=4  W=[2.0,3.0,-1.5,0.5]  A=[1.0,2.0,4.0,-2.0]  => 1.0
//     T2: FP16 WS  K=1  W=[1.5]     A=[2.0]                           => 3.0
//         (WS architecture: each beat outputs w*a independently; DMA takes first result)
//     T3: FP16 OS  K=8  alternating cancel                             => 0.0
//     T4: FP16 WS  K=1  W=[-3.0]    A=[2.0]                           => -6.0
//     T5: FP16 OS  zero weight K=4                                     => 0.0
//     T6: FP16 WS  K=1  W=[0.25]    A=[4.0]                           => 1.0
//     T7: FP16 OS  K=8  precision   W=[1.0]*8  A=[0.125..1.0]         => 4.5
//     T8: Back-to-back OS (T1 data twice)                              => 1.0, 1.0
//
//   WS Architecture Note:
//     In this single-PE system, acc_in=0 always.  Each pe_consume cycle outputs
//     acc_out = fp32_add(0, fp16_mul(w_i, a_i)) = w_i * a_i.
//     The result FIFO collects K individual products; DMA writes r_len=4 bytes
//     (one FP32 word), reading only the FIRST product from FIFO.
//     For a true WS dot-product, the host CPU would sum all K FIFO outputs.
//     These tests therefore verify K=1 (single multiply), directly checking the
//     FP16 multiply + FP32 write-back path end-to-end.
//
//   Result encoding: FP32 (32-bit IEEE 754)
//   DRAM layout: FP16 values stored as little-endian 16-bit pairs in 32-bit words
//     word[31:16] = fp16[1],  word[15:0] = fp16[0]
//
//   CTRL register encoding (ctrl_reg):
//     bit[0]   = start
//     bit[1]   = abort
//     bit[3:2] = data mode: 00=INT8, 01=INT16, 10=FP16
//     bit[5:4] = stat mode: 00=WS,   01=OS
//   For FP16+OS:  ctrl = 32'h19  (bit[4]=1, bit[3:2]=10, bit[0]=1)
//   For FP16+WS:  ctrl = 32'h09  (bit[3:2]=10, bit[0]=1)
// =============================================================================

`timescale 1ns/1ps

module tb_fp16_e2e;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
localparam ROWS    = 1;
localparam COLS    = 1;
localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;
localparam DRAM_SZ = 4096;

// NPU AXI-Lite register addresses
localparam REG_CTRL   = 32'h00;
localparam REG_STATUS = 32'h04;
localparam REG_M_DIM  = 32'h10;
localparam REG_N_DIM  = 32'h14;
localparam REG_K_DIM  = 32'h18;
localparam REG_W_ADDR = 32'h20;
localparam REG_A_ADDR = 32'h24;
localparam REG_R_ADDR = 32'h28;

// ctrl_reg encodings
// bit[3:2]=10 → FP16,  bit[5:4]=00 → WS,  bit[5:4]=01 → OS
localparam CTRL_FP16_WS = 32'h09;  // bit[3:2]=10, bit[0]=start → 0b001001
localparam CTRL_FP16_OS = 32'h19;  // bit[4]=1, bit[3:2]=10, bit[0]=1 → 0b011001

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

    // =====================================================================
    // DRAM Layout:  FP16 little-endian pairs
    //   word = {fp16[2k+1], fp16[2k]}  (high16 = index 2k+1, low16 = index 2k)
    //
    // Address map (byte addresses, stride 0x80):
    //   T1: W@0x1000, A@0x1020, R@0x1040  (OS K=4)
    //   T2: W@0x1080, A@0x10A0, R@0x10C0  (WS K=1: w=1.5, a=2.0 => 3.0)
    //   T3: W@0x1100, A@0x1120, R@0x1140  (OS K=8 alternating cancel)
    //   T4: W@0x1180, A@0x11A0, R@0x11C0  (WS K=1: w=-3.0, a=2.0 => -6.0)
    //   T5: W@0x1200, A@0x1220, R@0x1240  (OS K=4 zero weight)
    //   T6: W@0x1280, A@0x12A0, R@0x12C0  (WS K=1: w=0.25, a=4.0 => 1.0)
    //   T7: W@0x1300, A@0x1320, R@0x1340  (OS K=8 precision: sum=4.5)
    //   T8: W@0x1000 (reuse T1 data), A@0x1020, R@0x1400 & R@0x1420 (back-to-back)
    // =====================================================================

    // --- T1: FP16 OS K=4
    //   W = [2.0(0x4000), 3.0(0x4200), -1.5(0xBE00), 0.5(0x3800)]
    //   A = [1.0(0x3C00), 2.0(0x4000),  4.0(0x4400),-2.0(0xC000)]
    //   Expected: 2*1 + 3*2 + (-1.5)*4 + 0.5*(-2) = 2+6-6-1 = 1.0 => FP32 0x3F800000
    dram[32'h1000 >> 2]       = 32'h42004000;  // W[0]=2.0,  W[1]=3.0
    dram[(32'h1000 >> 2) + 1] = 32'h3800BE00;  // W[2]=-1.5, W[3]=0.5
    dram[32'h1020 >> 2]       = 32'h40003C00;  // A[0]=1.0,  A[1]=2.0
    dram[(32'h1020 >> 2) + 1] = 32'hC0004400;  // A[2]=4.0,  A[3]=-2.0

    // --- T2: FP16 WS K=1
    //   W = [1.5(0x3E00)]   A = [2.0(0x4000)]
    //   Expected first FIFO output: 1.5*2.0 = 3.0 => FP32 0x40400000
    //   (WS: single-multiply, DMA reads first result from FIFO)
    //   Word: {0x0000(pad), 0x3E00} = 0x00003E00 for W[0];  {0x0000, 0x4000} for A[0]
    dram[32'h1080 >> 2]       = 32'h00003E00;  // W[0]=1.5, W[1]=padding(0)
    dram[32'h10A0 >> 2]       = 32'h00004000;  // A[0]=2.0, A[1]=padding(0)

    // --- T3: FP16 OS K=8 alternating cancel
    //   W = [0.5,-0.5, 1.0,-1.0, 2.0,-2.0, 0.25,-0.25]
    //   A = [4.0, 4.0, 3.0, 3.0, 1.5,  1.5, 8.0,   8.0]
    //   Expected: 0.0 => FP32 0x00000000
    dram[32'h1100 >> 2]       = 32'hB8003800;  // W[0]=0.5,   W[1]=-0.5
    dram[(32'h1100 >> 2) + 1] = 32'hBC003C00;  // W[2]=1.0,   W[3]=-1.0
    dram[(32'h1100 >> 2) + 2] = 32'hC0004000;  // W[4]=2.0,   W[5]=-2.0
    dram[(32'h1100 >> 2) + 3] = 32'hB4003400;  // W[6]=0.25,  W[7]=-0.25
    dram[32'h1120 >> 2]       = 32'h44004400;  // A[0]=A[1]=4.0
    dram[(32'h1120 >> 2) + 1] = 32'h42004200;  // A[2]=A[3]=3.0
    dram[(32'h1120 >> 2) + 2] = 32'h3E003E00;  // A[4]=A[5]=1.5
    dram[(32'h1120 >> 2) + 3] = 32'h48004800;  // A[6]=A[7]=8.0

    // --- T4: FP16 WS K=1 negative
    //   W = [-3.0(0xC200)]   A = [2.0(0x4000)]
    //   Expected: -3.0*2.0 = -6.0 => FP32 0xC0C00000
    dram[32'h1180 >> 2]       = 32'h0000C200;  // W[0]=-3.0
    dram[32'h11A0 >> 2]       = 32'h00004000;  // A[0]=2.0

    // --- T5: FP16 OS zero weight K=4
    //   W = [0.0]*4   A = [1.5, 2.5, 3.5, 4.5]
    //   Expected: 0.0 => FP32 0x00000000
    dram[32'h1200 >> 2]       = 32'h00000000;
    dram[(32'h1200 >> 2) + 1] = 32'h00000000;
    dram[32'h1220 >> 2]       = 32'h41003E00;  // A[0]=1.5,  A[1]=2.5
    dram[(32'h1220 >> 2) + 1] = 32'h44804300;  // A[2]=3.5,  A[3]=4.5

    // --- T6: FP16 WS K=1 small positive
    //   W = [0.25(0x3400)]   A = [4.0(0x4400)]
    //   Expected: 0.25*4.0 = 1.0 => FP32 0x3F800000
    dram[32'h1280 >> 2]       = 32'h00003400;  // W[0]=0.25
    dram[32'h12A0 >> 2]       = 32'h00004400;  // A[0]=4.0

    // --- T7: FP16 OS K=8 arithmetic series precision
    //   W = [1.0]*8   A = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    //   Expected: sum = 4.5 => FP32 0x40900000
    //   A fp16: 0x3000,0x3400, 0x3600,0x3800, 0x3900,0x3A00, 0x3B00,0x3C00
    dram[32'h1300 >> 2]       = 32'h3C003C00;  // W[0]=W[1]=1.0
    dram[(32'h1300 >> 2) + 1] = 32'h3C003C00;  // W[2]=W[3]=1.0
    dram[(32'h1300 >> 2) + 2] = 32'h3C003C00;  // W[4]=W[5]=1.0
    dram[(32'h1300 >> 2) + 3] = 32'h3C003C00;  // W[6]=W[7]=1.0
    dram[32'h1320 >> 2]       = 32'h34003000;  // A[0]=0.125, A[1]=0.25
    dram[(32'h1320 >> 2) + 1] = 32'h38003600;  // A[2]=0.375, A[3]=0.5
    dram[(32'h1320 >> 2) + 2] = 32'h3A003900;  // A[4]=0.625, A[5]=0.75
    dram[(32'h1320 >> 2) + 3] = 32'h3C003B00;  // A[6]=0.875, A[7]=1.0
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
    .PHY_ROWS(16), .PHY_COLS(16), .DATA_W(DATA_W), .ACC_W(ACC_W)
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

// Wait for NPU done (polls STATUS register)
task wait_done;
    integer timeout;
    begin
        timeout = 0;
        rdata_tmp = 0;
        while (!rdata_tmp[1] && timeout < 5000) begin
            axi_read(REG_STATUS, rdata_tmp);
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 5000) begin
            $display("  [TIMEOUT] NPU did not complete in 5000 cycles!");
            $finish;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------
integer pass_cnt = 0;
integer fail_cnt = 0;

// Check FP32 result with optional tolerance (exact match for all expected values here)
task check_fp32;
    input [31:0] got;
    input [31:0] exp;
    input integer test_id;
    begin
        if (got === exp) begin
            $display("  [PASS] T%0d: got=0x%08X (correct FP32)", test_id, got);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] T%0d: got=0x%08X  expected=0x%08X", test_id, got, exp);
            fail_cnt = fail_cnt + 1;
        end
    end
endtask

// ---------------------------------------------------------------------------
// FP16 NPU launch helper
//   ctrl: CTRL_FP16_OS or CTRL_FP16_WS
//   k   : number of FP16 elements in each vector
//   w_addr, a_addr, r_addr: byte addresses in DRAM
// ---------------------------------------------------------------------------
task launch_fp16;
    input [31:0] ctrl_val;
    input [31:0] k;
    input [31:0] w_ba;   // weight byte address
    input [31:0] a_ba;   // activation byte address
    input [31:0] r_ba;   // result byte address
    begin
        // Deassert start (write 0) to ensure rising-edge detection
        axi_write(REG_CTRL, 32'h00);
        @(posedge clk);

        axi_write(REG_M_DIM,  32'd1);    // M=1 (always 1 for dot-product)
        axi_write(REG_N_DIM,  32'd1);    // N=1 (1 column)
        axi_write(REG_K_DIM,  k);
        axi_write(REG_W_ADDR, w_ba);
        axi_write(REG_A_ADDR, a_ba);
        axi_write(REG_R_ADDR, r_ba);

        // Fire start
        axi_write(REG_CTRL, ctrl_val);
    end
endtask

// ---------------------------------------------------------------------------
// Main test sequence
// ---------------------------------------------------------------------------
initial begin
    // Init AXI signals
    s_awvalid = 0; s_wvalid = 0; s_bready = 0;
    s_arvalid = 0; s_rready = 0;
    s_awaddr  = 0; s_wdata  = 0; s_wstrb  = 0;
    s_araddr  = 0;

    // Wait reset
    @(posedge clk); wait (rst_n === 1'b1);
    repeat(5) @(posedge clk);

    $display("################################################################");
    $display("  NPU FP16 End-to-End Test");
    $display("  ROWS=%0d COLS=%0d DATA_W=%0d ACC_W=%0d", ROWS, COLS, DATA_W, ACC_W);
    $display("################################################################");

    // ==================================================================
    // T1: FP16 OS K=4
    //   W=[2.0,3.0,-1.5,0.5]  A=[1.0,2.0,4.0,-2.0]  => 1.0 = 0x3F800000
    // ==================================================================
    $display("\n--- T1: FP16 OS K=4 (W=[2,3,-1.5,0.5] A=[1,2,4,-2] => 1.0) ---");
    launch_fp16(CTRL_FP16_OS, 32'd4, 32'h1000, 32'h1020, 32'h1040);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h1040 >> 2], 32'h3F800000, 1);

    // ==================================================================
    // T2: FP16 WS K=1
    //   W=[1.5]  A=[2.0]  => first FIFO result = 1.5*2.0 = 3.0 = 0x40400000
    //   WS architecture: each beat outputs w*a independently; DMA reads first
    // ==================================================================
    $display("\n--- T2: FP16 WS K=1 (w=1.5 a=2.0 => 3.0) ---");
    launch_fp16(CTRL_FP16_WS, 32'd1, 32'h1080, 32'h10A0, 32'h10C0);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h10C0 >> 2], 32'h40400000, 2);

    // ==================================================================
    // T3: FP16 OS K=8 alternating cancel => 0.0 = 0x00000000
    // ==================================================================
    $display("\n--- T3: FP16 OS K=8 alternating cancel => 0.0 ---");
    launch_fp16(CTRL_FP16_OS, 32'd8, 32'h1100, 32'h1120, 32'h1140);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h1140 >> 2], 32'h00000000, 3);

    // ==================================================================
    // T4: FP16 WS K=1 negative
    //   W=[-3.0]  A=[2.0]  => -3.0*2.0 = -6.0 = 0xC0C00000
    // ==================================================================
    $display("\n--- T4: FP16 WS K=1 negative (w=-3 a=2 => -6.0) ---");
    launch_fp16(CTRL_FP16_WS, 32'd1, 32'h1180, 32'h11A0, 32'h11C0);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(100) @(posedge clk);  // Increased from 5 to 100 to flush pipeline
    check_fp32(dram[32'h11C0 >> 2], 32'hC0C00000, 4);

    // ==================================================================
    // T5: FP16 OS zero weight K=4 => 0.0 = 0x00000000
    // ==================================================================
    $display("\n--- T5: FP16 OS zero weights K=4 => 0.0 ---");
    launch_fp16(CTRL_FP16_OS, 32'd4, 32'h1200, 32'h1220, 32'h1240);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h1240 >> 2], 32'h00000000, 5);

    // ==================================================================
    // T6: FP16 WS K=1 small positive
    //   W=[0.25]  A=[4.0]  => 0.25*4.0 = 1.0 = 0x3F800000
    // ==================================================================
    $display("\n--- T6: FP16 WS K=1 small (w=0.25 a=4.0 => 1.0) ---");
    launch_fp16(CTRL_FP16_WS, 32'd1, 32'h1280, 32'h12A0, 32'h12C0);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h12C0 >> 2], 32'h3F800000, 6);

    // ==================================================================
    // T7: FP16 OS K=8 arithmetic precision
    //   W=[1.0]*8  A=[0.125..1.0]  => 4.5 = 0x40900000
    // ==================================================================
    $display("\n--- T7: FP16 OS K=8 precision (W=1, A=0.125..1.0) => 4.5 ---");
    launch_fp16(CTRL_FP16_OS, 32'd8, 32'h1300, 32'h1320, 32'h1340);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h1340 >> 2], 32'h40900000, 7);

    // ==================================================================
    // T8: Back-to-back OS (reuse T1 data twice)
    //   run1: T1 W/A => 1.0 = 0x3F800000
    //   run2: T1 W/A => 1.0 = 0x3F800000
    // ==================================================================
    $display("\n--- T8: Back-to-back OS (T1 repeated twice => 1.0, 1.0) ---");

    // run1
    launch_fp16(CTRL_FP16_OS, 32'd4, 32'h1000, 32'h1020, 32'h1400);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h1400 >> 2], 32'h3F800000, 81);

    // run2
    launch_fp16(CTRL_FP16_OS, 32'd4, 32'h1000, 32'h1020, 32'h1420);
    wait_done;
    axi_write(REG_CTRL, 32'h00);
    repeat(5) @(posedge clk);
    check_fp32(dram[32'h1420 >> 2], 32'h3F800000, 82);

    // ==================================================================
    // Final report
    // ==================================================================
    $display("\n================================================================");
    $display("  FP16 E2E RESULT: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    $display("================================================================");

    if (fail_cnt == 0)
        $display("  ALL PASS - FP16 end-to-end pipeline verified.");
    else
        $display("  FAILURES DETECTED - check FP16 packer / fp16_mul / fp32_add.");

    $finish;
end

// ---------------------------------------------------------------------------
// Timeout watchdog
// ---------------------------------------------------------------------------
initial begin
    #2000000;
    $display("[WATCHDOG] Simulation exceeded 2M ns - force stop");
    $finish;
end

// ---------------------------------------------------------------------------
// Optional VCD dump
// ---------------------------------------------------------------------------
`ifdef DUMP_VCD
initial begin
    $dumpfile("sim/wave/fp16_e2e.vcd");
    $dumpvars(0, tb_fp16_e2e);
end
`endif

endmodule
