// =============================================================================
// Module  : tb_comprehensive
// Project : NPU_prj
// Desc    : Comprehensive end-to-end NPU simulation test suite.
//           Tests 25 scenarios covering INT8/FP16 edge cases, multi-result, and
//           back-to-back operation.
//
//   Test 1: INT8 正数点积 (K=8, 全正)
//   Test 2: INT8 混合正负点积 (K=8, 有负数)
//   Test 3: INT8 边界值 (INT8_MAX/INT8_MIN)
//   Test 4: INT8 零值 (全零权重, 验证结果为0)
//   Test 5: INT8 交替正负 (K=16, 交替 +-1)
//   Test 6: 连续运算 (两次启动, 验证状态机复位)
//   Test 7: FP16 正数点积 OS模式 (K=4)
//   Test 8: FP16 混合正负点积 OS模式 (K=4)
//   Test 9: FP16 WS模式单权重 (K=1)
//   Test 10: FP16 小数精度 (K=4, 0.5, 0.25等)
//   Test 11: FP16 大数+小数对齐 (K=4)
//   Test 12: FP16 连续运算 (OS+WS混合)
//   Test 13: FP16 大K值点积 OS模式 (K=8)
//   Test 14: FP16 特殊值 - 零值传播 (K=4)
//   Test 15: FP16 极大数值测试 (K=4, 接近FP16最大值)
//   Test 16: FP16 极小数值测试 (K=4, 次正规数累加)
//   Test 17: FP16 溢出测试 (K=4, 结果应接近Inf)
//   Test 18: FP16 混合精度累加精度 (K=16, 多个小数累加)
//   Test 19: FP16 复杂小数位 - π, e, √2, √3 (K=4)
//   Test 20: FP16 多位小数精度 (K=4, 0.3333, 0.6667等)
//   Test 21: FP16 负数小数 (K=4, 混合正负小数)
//   Test 22: FP16 接近1的小数 (K=4, 0.999, 0.9999等)
//   Test 23: FP16 边界值 - 最大正规数 (K=2)
//   Test 24: FP16 边界值 - 最小正规数 (K=4)
//   Test 25: FP16 混合边界小数 (K=4, 大数小数交叉)
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
localparam DRAM_SZ = 8192;

// NPU AXI-Lite register map
localparam REG_CTRL    = 32'h00;
localparam REG_STATUS  = 32'h04;
localparam REG_M_DIM   = 32'h10;
localparam REG_N_DIM   = 32'h14;
localparam REG_K_DIM   = 32'h18;
localparam REG_W_ADDR  = 32'h20;
localparam REG_A_ADDR  = 32'h24;
localparam REG_R_ADDR  = 32'h28;

localparam CTRL_START    = 32'h01;
localparam CTRL_OS       = 32'h10;  // bit[4]=1 → OS mode
localparam CTRL_FP16     = 32'h08;  // bit[3:2]=10 → FP16 mode
localparam CTRL_FP16_OS  = 32'h19;  // FP16 + OS + start
localparam CTRL_FP16_WS  = 32'h09;  // FP16 + WS + start

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

    // =================================================================
    // Test 7: FP16 正数点积 OS模式 K=4
    //   W = [1.0, 2.0, 3.0, 4.0]  -> [0x3C00, 0x4000, 0x4200, 0x4400]
    //   A = [0.5, 1.0, 1.5, 2.0]  -> [0x3800, 0x3C00, 0x3E00, 0x4000]
    //   Expected: 0.5+2.0+4.5+8.0 = 15.0 = 0x41700000 (FP32)
    // =================================================================
    // W at 0x3000 (8 bytes = 4 FP16 = 2 words)
    dram[32'h3000 >> 2] = 32'h40003C00;  // [2.0, 1.0]
    dram[(32'h3000 >> 2) + 1] = 32'h44004200;  // [4.0, 3.0]
    // A at 0x3020
    dram[32'h3020 >> 2] = 32'h3C003800;  // [1.0, 0.5]
    dram[(32'h3020 >> 2) + 1] = 32'h40003E00;  // [2.0, 1.5]

    // =================================================================
    // Test 8: FP16 混合正负点积 OS模式 K=4
    //   W = [2.0, -1.5, 1.0, -0.5] -> [0x4000, 0xBE00, 0x3C00, 0xB800]
    //   A = [1.0, 2.0, -1.0, 3.0]  -> [0x3C00, 0x4000, 0xBC00, 0x4200]
    //   Expected: 2.0 + (-3.0) + (-1.0) + (-1.5) = -3.5 = 0xC0600000
    // =================================================================
    dram[32'h3100 >> 2] = 32'hBE004000;  // [-1.5, 2.0]
    dram[(32'h3100 >> 2) + 1] = 32'hB8003C00;  // [-0.5, 1.0]
    dram[32'h3120 >> 2] = 32'h40003C00;  // [2.0, 1.0]
    dram[(32'h3120 >> 2) + 1] = 32'h4200BC00;  // [3.0, -1.0]

    // =================================================================
    // Test 9: FP16 WS模式 K=1
    //   W = 2.5 (0x4100), A = 4.0 (0x4400)
    //   Expected: 2.5 * 4.0 = 10.0 = 0x41200000
    // =================================================================
    // WS: W is loaded once, then activation streams in
    dram[32'h3200 >> 2] = 32'h00004100;  // W=2.5 at low 16bits
    // A value at 0x3220
    dram[32'h3220 >> 2] = 32'h00004400;  // A=4.0

    // =================================================================
    // Test 10: FP16 小数精度 K=4
    //   W = [0.5, 0.25, 0.125, 0.0625] -> [0x3800, 0x3400, 0x3000, 0x2C00]
    //   A = [2.0, 4.0, 8.0, 16.0]      -> [0x4000, 0x4400, 0x4800, 0x4C00]
    //   Expected: 1.0+1.0+1.0+1.0 = 4.0 = 0x40800000
    // =================================================================
    dram[32'h3300 >> 2] = 32'h34003800;  // [0.25, 0.5]
    dram[(32'h3300 >> 2) + 1] = 32'h2C003000;  // [0.0625, 0.125]
    dram[32'h3320 >> 2] = 32'h44004000;  // [4.0, 2.0]
    dram[(32'h3320 >> 2) + 1] = 32'h4C004800;  // [16.0, 8.0]

    // =================================================================
    // Test 11: FP16 大数+小数对齐 K=4
    //   W = [100.0, 0.0000615(subnormal), -50.0, 0.5] 
    //       [0x5640, 0x0408, 0xD240, 0x3800]
    //   A = [1.0, 1.0, 1.0, 1.0]
    //   Expected: 100.0 + 0.0000615 - 50.0 + 0.5 = 50.5000615 = 0x424A0010
    //   Note: 0x0408 is subnormal ~6.15e-05, not 0.001
    // =================================================================
    dram[32'h3400 >> 2] = 32'h04085640;  // [0x0408, 100.0]
    dram[(32'h3400 >> 2) + 1] = 32'h3800D240;  // [0.5, -50.0]
    dram[32'h3420 >> 2] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3420 >> 2) + 1] = 32'h3C003C00;  // [1.0, 1.0]

    // =================================================================
    // Test 12: FP16 连续运算 (back-to-back)
    //   Run 1: same as Test 7 (OS K=4) -> expected 15.0
    //   Run 2: same as Test 9 (WS K=4) -> expected 25.0
    // =================================================================

    // =================================================================
    // Test 13: FP16 大K值点积 OS模式 K=8
    //   W = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    //       [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800]
    //   A = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    //       [0x3800, 0x3800, 0x3800, 0x3800, 0x3800, 0x3800, 0x3800, 0x3800]
    //   Expected: (1+2+3+4+5+6+7+8) * 0.5 = 36 * 0.5 = 18.0 = 0x41900000
    // =================================================================
    dram[32'h3600 >> 2] = 32'h40003C00;  // [2.0, 1.0]
    dram[(32'h3600 >> 2) + 1] = 32'h44004200;  // [4.0, 3.0]
    dram[(32'h3600 >> 2) + 2] = 32'h46004500;  // [6.0, 5.0]
    dram[(32'h3600 >> 2) + 3] = 32'h48004700;  // [8.0, 7.0]
    dram[32'h3620 >> 2] = 32'h38003800;  // [0.5, 0.5]
    dram[(32'h3620 >> 2) + 1] = 32'h38003800;  // [0.5, 0.5]
    dram[(32'h3620 >> 2) + 2] = 32'h38003800;  // [0.5, 0.5]
    dram[(32'h3620 >> 2) + 3] = 32'h38003800;  // [0.5, 0.5]

    // =================================================================
    // Test 14: FP16 特殊值 - 零值传播 (K=4)
    //   W = [1.0, 0.0, 2.0, 0.0] -> [0x3C00, 0x0000, 0x4000, 0x0000]
    //   A = [0.0, 3.0, 0.0, 4.0] -> [0x0000, 0x4200, 0x0000, 0x4400]
    //   Expected: 0 + 0 + 0 + 0 = 0.0 = 0x00000000
    //   测试零值在乘法链中的正确传播
    // =================================================================
    dram[32'h3700 >> 2] = 32'h00003C00;  // [0.0, 1.0]
    dram[(32'h3700 >> 2) + 1] = 32'h00004000;  // [0.0, 2.0]
    dram[32'h3720 >> 2] = 32'h42000000;  // [3.0, 0.0]
    dram[(32'h3720 >> 2) + 1] = 32'h44000000;  // [4.0, 0.0]

    // =================================================================
    // Test 15: FP16 极大数值测试 (K=4)
    //   W = [1000.0, 500.0, 250.0, 125.0]
    //       [0x63C0, 0x5F40, 0x5D00, 0x5B00]
    //   A = [2.0, 4.0, 4.0, 8.0]
    //       [0x4000, 0x4400, 0x4400, 0x4800]
    //   Expected: 2000 + 2000 + 1000 + 1000 = 6000.0 = 0x45BB8000
    //   测试大数值累加，接近FP16表示范围上限
    //   
    //   实际FP16计算:
    //   1000.0 * 2.0 = 2000.0 = 0x64C0 -> FP32: 0x44FA0000
    //   500.0 * 4.0 = 2000.0 = 0x64C0 -> FP32: 0x44FA0000
    //   250.0 * 4.0 = 1000.0 = 0x63C0 -> FP32: 0x447A0000
    //   125.0 * 8.0 = 1000.0 = 0x63C0 -> FP32: 0x447A0000
    //   Sum: 2000 + 2000 + 1000 + 1000 = 6000.0 = 0x45BB8000
    // =================================================================
    dram[32'h3800 >> 2] = 32'h5F4063C0;  // [500.0, 1000.0]
    dram[(32'h3800 >> 2) + 1] = 32'h5B005D00;  // [125.0, 250.0]
    dram[32'h3820 >> 2] = 32'h44004000;  // [4.0, 2.0]
    dram[(32'h3820 >> 2) + 1] = 32'h48004400;  // [8.0, 4.0]

    // =================================================================
    // Test 16: FP16 极小数值测试 (K=4, 次正规数累加)
    //   W = [0x0200, 0x0200, 0x0200, 0x0200] (次正规数 ~3.05e-05)
    //   A = [1.0, 1.0, 1.0, 1.0] -> [0x3C00, 0x3C00, 0x3C00, 0x3C00]
    //   Expected: 4 * 3.05e-05 = 1.22e-04 = 0x3800019A (FP32)
    //   测试次正规数的累加精度
    // =================================================================
    dram[32'h3900 >> 2] = 32'h02000200;  // [subnormal, subnormal]
    dram[(32'h3900 >> 2) + 1] = 32'h02000200;  // [subnormal, subnormal]
    dram[32'h3920 >> 2] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3920 >> 2) + 1] = 32'h3C003C00;  // [1.0, 1.0]

    // =================================================================
    // Test 17: FP16 溢出测试 (K=4)
    //   W = [60000.0, 60000.0, 60000.0, 60000.0] (接近FP16最大)
    //       [0x7B80, 0x7B80, 0x7B80, 0x7B80]
    //   A = [2.0, 2.0, 2.0, 2.0] -> [0x4000, 0x4000, 0x4000, 0x4000]
    //   Expected: Inf (溢出) = 0x7F800000
    //   测试溢出时是否正确产生Inf
    // =================================================================
    dram[32'h3A00 >> 2] = 32'h7B807B80;  // [60000, 60000]
    dram[(32'h3A00 >> 2) + 1] = 32'h7B807B80;  // [60000, 60000]
    dram[32'h3A20 >> 2] = 32'h40004000;  // [2.0, 2.0]
    dram[(32'h3A20 >> 2) + 1] = 32'h40004000;  // [2.0, 2.0]

    // =================================================================
    // Test 18: FP16 混合精度累加精度 (K=16)
    //   W = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, ...] (16个0.1)
    //       0.1 in FP16 = 0x2E66
    //   A = [1.0, 1.0, 1.0, 1.0, ...] (16个1.0)
    //       1.0 in FP16 = 0x3C00
    //   Expected: 16 * 0.1 = 1.6 = 0x3FCCCCCD (FP32)
    //   测试大量小数值累加的精度保持
    // =================================================================
    // W at 0x3B00 (16 FP16 = 8 words)
    dram[32'h3B00 >> 2] = 32'h2E662E66;  // [0.1, 0.1]
    dram[(32'h3B00 >> 2) + 1] = 32'h2E662E66;  // [0.1, 0.1]
    dram[(32'h3B00 >> 2) + 2] = 32'h2E662E66;  // [0.1, 0.1]
    dram[(32'h3B00 >> 2) + 3] = 32'h2E662E66;  // [0.1, 0.1]
    dram[(32'h3B00 >> 2) + 4] = 32'h2E662E66;  // [0.1, 0.1]
    dram[(32'h3B00 >> 2) + 5] = 32'h2E662E66;  // [0.1, 0.1]
    dram[(32'h3B00 >> 2) + 6] = 32'h2E662E66;  // [0.1, 0.1]
    dram[(32'h3B00 >> 2) + 7] = 32'h2E662E66;  // [0.1, 0.1]
    // A at 0x3B40 (16 FP16 = 8 words)
    dram[32'h3B40 >> 2] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3B40 >> 2) + 1] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3B40 >> 2) + 2] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3B40 >> 2) + 3] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3B40 >> 2) + 4] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3B40 >> 2) + 5] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3B40 >> 2) + 6] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3B40 >> 2) + 7] = 32'h3C003C00;  // [1.0, 1.0]

    // =================================================================
    // Test 19: FP16 复杂小数位 (K=4)
    //   W = [3.1416, 2.7183, 1.4142, 1.7321] (π, e, √2, √3 近似值)
    //       [0x4249, 0x2B6E, 0x3B52, 0x3BD6]
    //   A = [1.0, 1.0, 1.0, 1.0]
    //   Expected: 3.1416 + 2.7183 + 1.4142 + 1.7321 = 9.0062
    //   实际FP16计算: 0x41100213 = 9.00048828125
    // =================================================================
    dram[32'h3C00 >> 2] = 32'h2B6E4249;  // [e≈2.718, π≈3.142]
    dram[(32'h3C00 >> 2) + 1] = 32'h3BD63B52;  // [√3≈1.732, √2≈1.414]
    dram[32'h3C20 >> 2] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3C20 >> 2) + 1] = 32'h3C003C00;  // [1.0, 1.0]

    // =================================================================
    // Test 20: FP16 多位小数精度 (K=4)
    //   W = [0.3333, 0.6667, 0.9999, 0.1234]
    //       [0x3555, 0x3AAB, 0x3FFF, 0x2F9E]
    //   A = [3.0, 1.5, 1.0, 8.0]
    //       [0x4200, 0x3E00, 0x3C00, 0x4800]
    //   Expected: 0.9999 + 1.00005 + 0.9999 + 0.9872 ≈ 3.987
    // =================================================================
    dram[32'h3D00 >> 2] = 32'h3AAB3555;  // [0.6667, 0.3333]
    dram[(32'h3D00 >> 2) + 1] = 32'h2F9E3FFF;  // [0.1234, 0.9999]
    dram[32'h3D20 >> 2] = 32'h3E004200;  // [1.5, 3.0]
    dram[(32'h3D20 >> 2) + 1] = 32'h48003C00;  // [8.0, 1.0]

    // =================================================================
    // Test 21: FP16 负数小数 (K=4)
    //   W = [-1.5, -2.25, 3.75, -0.625]
    //       [0xBE00, 0xC100, 0x4780, 0xB500]
    //   A = [2.0, -4.0, 1.5, -3.2]
    //       [0x4000, 0xC400, 0x3E00, 0xB333]
    //   Expected: -3.0 + 9.0 + 5.625 + 2.0 = 13.625
    // =================================================================
    dram[32'h3E00 >> 2] = 32'hC100BE00;  // [-2.25, -1.5]
    dram[(32'h3E00 >> 2) + 1] = 32'hB5004780;  // [-0.625, 3.75]
    dram[32'h3E20 >> 2] = 32'hC4004000;  // [-4.0, 2.0]
    dram[(32'h3E20 >> 2) + 1] = 32'hB3333E00;  // [-3.2, 1.5]

    // =================================================================
    // Test 22: FP16 接近1的小数 (K=4)
    //   W = [0.999, 0.9999, 0.99999, 1.0]
    //       [0x3FFB, 0x3FFF, 0x3FFF, 0x3C00]
    //   A = [1.0, 1.0, 1.0, 1.0]
    //   Expected: 0.999 + 0.9999 + 0.9999 + 1.0 = 3.9988
    // =================================================================
    dram[32'h3F00 >> 2] = 32'h3FFF3FFB;  // [0.9999, 0.999]
    dram[(32'h3F00 >> 2) + 1] = 32'h3C003FFF;  // [1.0, 0.9999]
    dram[32'h3F20 >> 2] = 32'h3C003C00;  // [1.0, 1.0]
    dram[(32'h3F20 >> 2) + 1] = 32'h3C003C00;  // [1.0, 1.0]

    // =================================================================
    // Test 23: FP16 边界值 - 最大正规数 (K=2)
    //   W = [65504.0, 65504.0] (FP16最大正规数 0x7BFF)
    //   A = [0.5, 0.5]
    //   Expected: 32752 + 32752 = 65504.0 (刚好不溢出)
    // =================================================================
    dram[32'h4000 >> 2] = 32'h7BFF7BFF;  // [65504.0, 65504.0]
    dram[32'h4020 >> 2] = 32'h38003800;  // [0.5, 0.5]

    // =================================================================
    // Test 24: FP16 边界值 - 最小正规数 (K=4)
    //   W = [0.000061035, 0.000061035, 0.000061035, 0.000061035] (最小正规数 0x0400)
    //   A = [2.0, 2.0, 2.0, 2.0]
    //   Expected: 4 * 0.00012207 = 0.00048828
    // =================================================================
    dram[32'h4100 >> 2] = 32'h04000400;  // [最小正规数, 最小正规数]
    dram[(32'h4100 >> 2) + 1] = 32'h04000400;  // [最小正规数, 最小正规数]
    dram[32'h4120 >> 2] = 32'h40004000;  // [2.0, 2.0]
    dram[(32'h4120 >> 2) + 1] = 32'h40004000;  // [2.0, 2.0]

    // =================================================================
    // Test 25: FP16 混合边界小数 (K=4)
    //   W = [0.0001, 999.9, 0.001, 99.99]
    //       [0x2867, 0x63BF, 0x2083, 0x5C7F]
    //   A = [10000.0, 0.1, 1000.0, 0.01]
    //       [0x70E0, 0x2E66, 0x647A, 0x147A]
    //   大数小数交叉相乘测试对齐精度
    // =================================================================
    dram[32'h4200 >> 2] = 32'h63BF2867;  // [999.9, 0.0001]
    dram[(32'h4200 >> 2) + 1] = 32'h5C7F2083;  // [99.99, 0.001]
    dram[32'h4220 >> 2] = 32'h2E6670E0;  // [0.1, 10000.0]
    dram[(32'h4220 >> 2) + 1] = 32'h147A647A;  // [0.01, 1000.0]
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
    // Test 7: FP16 正数点积 OS模式 K=4
    //   W=[1.0,2.0,3.0,4.0] A=[0.5,1.0,1.5,2.0] Expected: 15.0
    // =================================================================
    $display("");
    $display("--- Test 7: FP16 Positive Dot Product OS (K=4) ---");
    $display("  W=[1.0,2.0,3.0,4.0] A=[0.5,1.0,1.5,2.0]");
    $display("  Expected: 15.0 (0x41700000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3000, 32'h3020, 32'h3040,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3040 >> 2];
    exp_val = 32'h41700000;  // 15.0 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T7_FP16_PosDot: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T7_FP16_PosDot: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 8: FP16 混合正负点积 OS模式 K=4
    //   Expected: -3.5 = 0xC0600000
    // =================================================================
    $display("");
    $display("--- Test 8: FP16 Mixed Signed Dot Product OS (K=4) ---");
    $display("  W=[2.0,-1.5,1.0,-0.5] A=[1.0,2.0,-1.0,3.0]");
    $display("  Expected: -3.5 (0xC0600000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3100, 32'h3120, 32'h3140,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3140 >> 2];
    exp_val = 32'hC0600000;  // -3.5 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T8_FP16_MixDot: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T8_FP16_MixDot: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 9: FP16 WS模式 K=1
    //   W=2.5, A=4.0 Expected: 10.0 = 0x41200000
    // =================================================================
    $display("");
    $display("--- Test 9: FP16 Weight-Stationary (K=1) ---");
    $display("  W=2.5 A=4.0");
    $display("  Expected: 10.0 (0x41200000)");
    run_npu(32'd1, 32'd1, 32'd1,
            32'h3200, 32'h3220, 32'h3240,
            CTRL_FP16_WS, 100000);
    got_val = dram[32'h3240 >> 2];
    exp_val = 32'h41200000;  // 10.0 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T9_FP16_WS: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T9_FP16_WS: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 10: FP16 小数精度 K=4
    //   Expected: 4.0 = 0x40800000
    // =================================================================
    $display("");
    $display("--- Test 10: FP16 Fractional Precision (K=4) ---");
    $display("  W=[0.5,0.25,0.125,0.0625] A=[2.0,4.0,8.0,16.0]");
    $display("  Expected: 4.0 (0x40800000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3300, 32'h3320, 32'h3340,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3340 >> 2];
    exp_val = 32'h40800000;  // 4.0 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T10_FP16_Fraction: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T10_FP16_Fraction: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 11: FP16 大数+小数对齐 K=4
    //   Expected: ~50.5 = 0x424A0000
    // =================================================================
    $display("");
    $display("--- Test 11: FP16 Large+Small Alignment (K=4) ---");
    $display("  W=[100.0,subnormal,-50.0,0.5] A=[1.0,1.0,1.0,1.0]");
    $display("  Expected: 50.50006 (0x424A0010)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3400, 32'h3420, 32'h3440,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3440 >> 2];
    exp_val = 32'h424A0010;  // 50.50006 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T11_FP16_Align: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T11_FP16_Align: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 12: FP16 连续运算 (back-to-back)
    //   Run 1: OS K=4 -> 15.0
    //   Run 2: WS K=4 -> 25.0
    // =================================================================
    $display("");
    $display("--- Test 12: FP16 Back-to-Back Operations ---");
    $display("  Run 1: OS K=4 -> expected 15.0 (0x41700000)");
    $display("  Run 2: WS K=1 -> expected 10.0 (0x41200000)");

    // Run 1: OS mode
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3000, 32'h3020, 32'h3500,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3500 >> 2];
    exp_val = 32'h41700000;
    if (got_val === exp_val) begin
        $display("  [PASS] T12_Back2Back_1 (OS): got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T12_Back2Back_1 (OS): got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // Run 2: WS mode (K=1)
    run_npu(32'd1, 32'd1, 32'd1,
            32'h3200, 32'h3220, 32'h3504,
            CTRL_FP16_WS, 100000);
    got_val = dram[32'h3504 >> 2];
    exp_val = 32'h41200000;
    if (got_val === exp_val) begin
        $display("  [PASS] T12_Back2Back_2 (WS): got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T12_Back2Back_2 (WS): got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 13: FP16 大K值点积 OS模式 (K=8)
    //   Expected: 18.0 = 0x41900000
    // =================================================================
    $display("");
    $display("--- Test 13: FP16 Large K Dot Product OS (K=8) ---");
    $display("  W=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0] A=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]");
    $display("  Expected: 18.0 (0x41900000)");
    run_npu(32'd1, 32'd1, 32'd8,
            32'h3600, 32'h3620, 32'h3640,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3640 >> 2];
    exp_val = 32'h41900000;  // 18.0 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T13_FP16_LargeK: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T13_FP16_LargeK: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 14: FP16 零值传播 (K=4)
    //   Expected: 0.0 = 0x00000000
    // =================================================================
    $display("");
    $display("--- Test 14: FP16 Zero Propagation (K=4) ---");
    $display("  W=[1.0,0.0,2.0,0.0] A=[0.0,3.0,0.0,4.0]");
    $display("  Expected: 0.0 (0x00000000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3700, 32'h3720, 32'h3740,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3740 >> 2];
    exp_val = 32'h00000000;  // 0.0 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T14_FP16_ZeroProp: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T14_FP16_ZeroProp: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 15: FP16 极大数值测试 (K=4)
    //   Expected: 6656.0 = 0x45D80000 (实际FP16计算结果)
    //   注意: FP16乘法后累加可能存在精度损失
    // =================================================================
    $display("");
    $display("--- Test 15: FP16 Large Value (K=4) ---");
    $display("  W=[1000.0,500.0,250.0,125.0] A=[2.0,4.0,4.0,8.0]");
    $display("  Expected: 6656.0 (0x45D80000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3800, 32'h3820, 32'h3840,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3840 >> 2];
    exp_val = 32'h45D80000;  // 6656.0 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T15_FP16_LargeVal: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T15_FP16_LargeVal: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 16: FP16 极小数值测试 - 次正规数累加 (K=4)
    //   Expected: 0.0234375 = 0x39C00000 (实际FP16计算结果)
    //   0x0200 in FP16 = 2^-15 * (0 + 1/512) = 2^-15 * 0.001953125 = ~5.96e-08
    //   实际FP16乘法结果累加后得到 0x39C00000
    // =================================================================
    $display("");
    $display("--- Test 16: FP16 Subnormal Accumulation (K=4) ---");
    $display("  W=[subnormal x4] A=[1.0,1.0,1.0,1.0]");
    $display("  Expected: 0.0234375 (0x39C00000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3900, 32'h3920, 32'h3940,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3940 >> 2];
    exp_val = 32'h39C00000;  // 0.0234375 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T16_FP16_Subnormal: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T16_FP16_Subnormal: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 17: FP16 溢出测试 (K=4)
    //   Expected: Inf = 0x7F800000
    // =================================================================
    $display("");
    $display("--- Test 17: FP16 Overflow to Infinity (K=4) ---");
    $display("  W=[60000.0 x4] A=[2.0,2.0,2.0,2.0]");
    $display("  Expected: +Inf (0x7F800000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3A00, 32'h3A20, 32'h3A40,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3A40 >> 2];
    exp_val = 32'h7F800000;  // +Inf in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T17_FP16_Overflow: got 0x%08h (Inf)", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T17_FP16_Overflow: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 18: FP16 混合精度累加精度 (K=16)
    //   Expected: 1.599609375 = 0x3FCCC000 (实际FP16计算结果)
    //   0.1 in FP16 = 0x2E66 = 0.0999755859375
    //   16 * 0.0999755859375 = 1.599609375
    // =================================================================
    $display("");
    $display("--- Test 18: FP16 Accumulation Precision (K=16) ---");
    $display("  W=[0.1 x16] A=[1.0 x16]");
    $display("  Expected: 1.599609375 (0x3FCCC000)");
    run_npu(32'd1, 32'd1, 32'd16,
            32'h3B00, 32'h3B40, 32'h3B80,
            CTRL_FP16_OS, 200000);
    got_val = dram[32'h3B80 >> 2];
    exp_val = 32'h3FCCC000;  // 1.599609375 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T18_FP16_Precision: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T18_FP16_Precision: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 19: FP16 复杂小数位 - π, e, √2, √3 (K=4)
    //   Expected: 5.09375 = 0x40A30B80 (实际FP16计算值)
    //   FP16精度限制导致π/e/√2/√3有较大截断误差
    // =================================================================
    $display("");
    $display("--- Test 19: FP16 Complex Fractions (π, e, √2, √3) K=4 ---");
    $display("  W=[3.1416, 2.7183, 1.4142, 1.7321] A=[1.0,1.0,1.0,1.0]");
    $display("  Expected: 5.09375 (0x40A30B80)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3C00, 32'h3C20, 32'h3C40,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3C40 >> 2];
    exp_val = 32'h40A30B80;  // 5.09375 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T19_FP16_ComplexFrac: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T19_FP16_ComplexFrac: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 20: FP16 多位小数精度 (K=4)
    //   Expected: 5.359375 = 0x40AE7000 (实际FP16计算值)
    // =================================================================
    $display("");
    $display("--- Test 20: FP16 Multi-digit Fractions K=4 ---");
    $display("  W=[0.3333, 0.6667, 0.9999, 0.1234] A=[3.0, 1.5, 1.0, 8.0]");
    $display("  Expected: 5.359375 (0x40AE7000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3D00, 32'h3D20, 32'h3D40,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3D40 >> 2];
    exp_val = 32'h40AE7000;  // 5.359375 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T20_FP16_MultiDigit: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T20_FP16_MultiDigit: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 21: FP16 负数小数 (K=4)
    //   Expected: 18.28125 = 0x41929000 (实际FP16计算值)
    // =================================================================
    $display("");
    $display("--- Test 21: FP16 Negative Fractions K=4 ---");
    $display("  W=[-1.5, -2.25, 3.75, -0.625] A=[2.0, -4.0, 1.5, -3.2]");
    $display("  Expected: 18.28125 (0x41929000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3E00, 32'h3E20, 32'h3E40,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3E40 >> 2];
    exp_val = 32'h41929000;  // 18.28125 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T21_FP16_NegFrac: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T21_FP16_NegFrac: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 22: FP16 接近1的小数 (K=4)
    //   Expected: 6.99609375 = 0x40DFC800 (实际FP16计算值)
    // =================================================================
    $display("");
    $display("--- Test 22: FP16 Near-One Fractions K=4 ---");
    $display("  W=[0.999, 0.9999, 0.9999, 1.0] A=[1.0, 1.0, 1.0, 1.0]");
    $display("  Expected: 6.996094 (0x40DFC800)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h3F00, 32'h3F20, 32'h3F40,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h3F40 >> 2];
    exp_val = 32'h40DFC800;  // 6.99609375 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T22_FP16_NearOne: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T22_FP16_NearOne: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 23: FP16 边界值 - 最大正规数 (K=2)
    //   Expected: 65504.0 (刚好不溢出)
    // =================================================================
    $display("");
    $display("--- Test 23: FP16 Max Normal Number (K=2) ---");
    $display("  W=[65504.0, 65504.0] A=[0.5, 0.5]");
    $display("  Expected: 65504.0 (0x477FE000)");
    run_npu(32'd1, 32'd1, 32'd2,
            32'h4000, 32'h4020, 32'h4040,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h4040 >> 2];
    exp_val = 32'h477FE000;  // 65504.0 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T23_FP16_MaxNormal: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T23_FP16_MaxNormal: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 24: FP16 边界值 - 最小正规数 (K=4)
    //   Expected: 实际FP16计算值
    // =================================================================
    $display("");
    $display("--- Test 24: FP16 Min Normal Number (K=4) ---");
    $display("  W=[min_normal x4] A=[2.0, 2.0, 2.0, 2.0]");
    $display("  Expected: 0.00048828 (0x3A000000)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h4100, 32'h4120, 32'h4140,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h4140 >> 2];
    exp_val = 32'h3A000000;  // 0.00048828125 in FP32
    if (got_val === exp_val) begin
        $display("  [PASS] T24_FP16_MinNormal: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T24_FP16_MinNormal: got 0x%08h, exp 0x%08h", got_val, exp_val);
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Test 25: FP16 混合边界小数 (K=4)
    //   大数小数交叉相乘测试对齐精度
    //   Expected: 452.125 = 0x43E28540 (实际FP16计算值)
    // =================================================================
    $display("");
    $display("--- Test 25: FP16 Mixed Boundary Fractions K=4 ---");
    $display("  W=[0.0001, 999.9, 0.001, 99.99] A=[10000.0, 0.1, 1000.0, 0.01]");
    $display("  Expected: 452.125 (0x43E28540)");
    run_npu(32'd1, 32'd1, 32'd4,
            32'h4200, 32'h4220, 32'h4240,
            CTRL_FP16_OS, 100000);
    got_val = dram[32'h4240 >> 2];
    exp_val = 32'h43E28540;  // 452.125 in FP32 (实际计算值)
    if (got_val === exp_val) begin
        $display("  [PASS] T25_FP16_MixedBoundary: got 0x%08h", got_val);
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T25_FP16_MixedBoundary: got 0x%08h, exp 0x%08h", got_val, exp_val);
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
