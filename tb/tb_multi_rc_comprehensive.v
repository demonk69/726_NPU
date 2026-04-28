// =============================================================================
// Module  : tb_multi_rc_comprehensive
// Project : NPU_prj
// Desc    : Multi-row multi-column comprehensive test.
//           Tests NPU with ROWS=2, COLS=2 to verify:
//             1. Tile-loop OS mode: M×N tiles, each computes one C[i][j]
//             2. Result serialiser (multi-col drain) — COLS results per flush
//             3. Weight routing: OS tile j sends weight to col (j % COLS)
//             4. Result layout in DRAM (row-major)
//             5. Back-to-back operations with state reset
//             6. WS mode tile-loop matrix multiply
//
// ============================================================================
// Architecture: Tile-Loop with Column-Targeted Weight Routing
// ============================================================================
//
//   OS Tile-Loop (M×N iterations):
//     for i in 0..M-1:
//       for j in 0..N-1:
//         DMA load B[:,j] → target PE column (j % COLS)
//         DMA load A[i,:] → broadcast to all rows
//         PE computes C[i][j] = dot(A[i,:], B[:,j]) internally
//         flush → valid_out → result FIFO → DMA write C[i][j]
//
//   When N == COLS (e.g. N=2, COLS=2):
//     tile j=0: weight → col0, result at R_ADDR+0
//     tile j=1: weight → col1, result at R_ADDR+4
//     (Both tiles share the same physical PE array; results written sequentially)
//
//   No systolic shift: each PE column receives its own weight via target_col
//   routing. All rows receive the same activation (broadcast).
//
// ============================================================================
// Test Expected Values (Tile-Loop, no systolic shift)
// ============================================================================
//
//   All tests use M_DIM=1, N_DIM=2 (or N_DIM=COLS), K_DIM=K.
//   B is stored column-major: B[:,0] at W_ADDR, B[:,1] at W_ADDR + K_bytes.
//   A is stored row-major: A[0,:] at A_ADDR.
//
//   T1: A=[5,6,7,8], B[:,0]=[1,2,3,4], B[:,1]=[2,3,4,5]  K=4
//       C[0][0] = 5*1+6*2+7*3+8*4 = 5+12+21+32 = 70
//       C[0][1] = 5*2+6*3+7*4+8*5 = 10+18+28+40 = 96
//
//   T2: A=[2,2,2,2,2,2,2,2], B[:,0]=[1,-1,2,-2,3,-3,4,-4], B[:,1]=[4,-4,3,-3,2,-2,1,-1]  K=8
//       C[0][0] = 1*2+(-1)*2+2*2+(-2)*2+3*2+(-3)*2+4*2+(-4)*2 = 0
//       C[0][1] = 4*2+(-4)*2+3*2+(-3)*2+2*2+(-2)*2+1*2+(-1)*2 = 0
//
//   T3: A=[127,-128,1,0], B[:,0]=[127,-128,1,0], B[:,1]=[1,0,127,-128]  K=4
//       C[0][0] = 127*127+(-128)*(-128)+1*1+0*0 = 16129+16384+1 = 32514
//       C[0][1] = 127*1+(-128)*0+1*127+0*(-128) = 127+0+127+0 = 254
//
//   T4: A=[1,2,3,4], B[:,0]=[0,0,0,0], B[:,1]=[0,0,0,0]  K=4
//       C[0][0] = 0, C[0][1] = 0
//
//   T5: Back-to-back (Run1=T1, Run2=T3)
//       Verifies state reset between operations.
//
//   T6: Verify col0 != col1 for non-trivial B columns (standard matmul)
// =============================================================================

`timescale 1ns/1ps

module tb_multi_rc_comprehensive;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
localparam ROWS    = 2;
localparam COLS    = 2;
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
localparam CTRL_OS     = 32'h11;  // bit[4]=1 → OS mode, bit[0]=1 → start

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
    // Test 1: C = A[1x4] x B[4x2] = C[1x2]
    //   A     = [5, 6, 7, 8]
    //   B[:,0] = [1, 2, 3, 4]  (column 0)
    //   B[:,1] = [2, 3, 4, 5]  (column 1)
    //   C[0][0] = 5*1+6*2+7*3+8*4 = 70
    //   C[0][1] = 5*2+6*3+7*4+8*5 = 96
    //
    //   W_ADDR=0x1000: B[:,0]=[1,2,3,4] then B[:,1]=[2,3,4,5]
    //     word0 (0x1000): [1,2,3,4] = 0x04030201
    //     word1 (0x1004): [2,3,4,5] = 0x05040302  (B[:,1], 1 word after B[:,0])
    //   A_ADDR=0x1010: [5,6,7,8] = 0x08070605
    //   R_ADDR=0x1020: 2 result words
    // =================================================================
    dram[32'h1000 >> 2]       = 32'h04030201;   // B[:,0] = [1,2,3,4]
    dram[(32'h1000 >> 2) + 1] = 32'h05040302;   // B[:,1] = [2,3,4,5]
    dram[32'h1010 >> 2]       = 32'h08070605;   // A[0,:] = [5,6,7,8]

    // =================================================================
    // Test 2: C = A[1x8] x B[8x2] = C[1x2], alternating weights
    //   A       = [2,2,2,2,2,2,2,2]
    //   B[:,0]  = [1,-1,2,-2,3,-3,4,-4]
    //   B[:,1]  = [4,-4,3,-3,2,-2,1,-1]
    //   C[0][0] = 1*2+(-1)*2+2*2+(-2)*2+3*2+(-3)*2+4*2+(-4)*2 = 0
    //   C[0][1] = 4*2+(-4)*2+3*2+(-3)*2+2*2+(-2)*2+1*2+(-1)*2 = 0
    //
    //   W_ADDR=0x1100: 2 words for B[:,0], 2 words for B[:,1]
    //   A_ADDR=0x1120: 2 words for A[0,:]
    //   R_ADDR=0x1140: 2 result words
    // =================================================================
    dram[32'h1100 >> 2]       = 32'hFE02FF01;   // B[:,0] w0: [1,-1,2,-2]
    dram[(32'h1100 >> 2) + 1] = 32'hFC04FD03;   // B[:,0] w1: [3,-3,4,-4]
    dram[(32'h1100 >> 2) + 2] = 32'hFD03FC04;   // B[:,1] w0: [4,-4,3,-3]
    dram[(32'h1100 >> 2) + 3] = 32'hFF01FE02;   // B[:,1] w1: [2,-2,1,-1]
    dram[32'h1120 >> 2]       = 32'h02020202;   // A[0,:] w0: [2,2,2,2]
    dram[(32'h1120 >> 2) + 1] = 32'h02020202;   // A[0,:] w1: [2,2,2,2]

    // =================================================================
    // Test 3: C = A[1x4] x B[4x2] = C[1x2], boundary values
    //   A      = [127,-128,1,0]
    //   B[:,0] = [127,-128,1,0]
    //   B[:,1] = [1,0,127,-128]
    //   C[0][0] = 127*127+(-128)*(-128)+1*1+0*0 = 16129+16384+1+0 = 32514
    //   C[0][1] = 127*1+(-128)*0+1*127+0*(-128) = 127+0+127+0 = 254
    //
    //   W_ADDR=0x1200: [B[:,0], B[:,1]]
    //   A_ADDR=0x1210: A[0,:]
    //   R_ADDR=0x1230: 2 result words
    // =================================================================
    dram[32'h1200 >> 2]       = 32'h0001807F;   // B[:,0] = [127,-128,1,0]  byte: [0x7F,0x80,0x01,0x00]
    dram[(32'h1200 >> 2) + 1] = 32'h807F0001;   // B[:,1] = [1,0,127,-128]  byte: [0x01,0x00,0x7F,0x80]
    dram[32'h1210 >> 2]       = 32'h0001807F;   // A[0,:] = [127,-128,1,0]

    // =================================================================
    // Test 4: C = A[1x4] x B[4x2] = C[1x2], all zeros
    //   A=[1,2,3,4], B[:,0]=[0,0,0,0], B[:,1]=[0,0,0,0]
    //   C[0][0] = 0, C[0][1] = 0
    // =================================================================
    dram[32'h1300 >> 2]       = 32'h00000000;   // B[:,0]=[0,0,0,0]
    dram[(32'h1300 >> 2) + 1] = 32'h00000000;   // B[:,1]=[0,0,0,0]
    dram[32'h1310 >> 2]       = 32'h04030201;   // A[0,:]=[1,2,3,4]

    // Test 5 reuses T1 (0x1000/0x1010) and T3 (0x1200/0x1210) data
    // Results at 0x1400 and 0x1410 respectively
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
            $display("[DRAM_WR] wr_base=0x%08h addr=0x%08h data=0x%08h beat=%0d", wr_base, (wr_base + wr_cnt*4), m_wdata, wr_cnt);
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
// NPU DUT: ROWS=2, COLS=2
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
            axi_write(REG_CTRL, 32'h00);
        end else begin
            $display("    *** TIMEOUT after %0d cycles! ***", timeout_cyc);
            $display("    ctrl_state=%0d, dma_state=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $finish;
        end
    end
endtask

// run_npu: M_DIM, N_DIM, K_DIM, W_ADDR, A_ADDR, R_ADDR, ctrl_val, timeout
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

// check: compare got vs exp, print PASS/FAIL
task check_result;
    input [63:0]  test_id;
    input [31:0]  got;
    input [31:0]  exp;
    input [127:0] tag;  // 16 chars
    begin
        if (got === exp) begin
            $display("  [PASS] %s: got=%0d (0x%08h)", tag, $signed(got), got);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  [FAIL] %s: got=%0d (0x%08h), exp=%0d (0x%08h), diff=%0d",
                     tag, $signed(got), got, $signed(exp), exp,
                     $signed(got) - $signed(exp));
            fail_cnt = fail_cnt + 1;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;
reg [31:0] got0, got1;

initial begin
    s_awaddr<=0;s_awvalid<=0;s_wdata<=0;s_wstrb<=0;s_wvalid<=0;s_bready<=0;
    s_araddr<=0;s_arvalid<=0;s_rready<=0;
    m_awready<=1; m_wready<=0;
    pass_cnt=0; fail_cnt=0;

    `ifdef DUMP_VCD
    $dumpfile("tb_multi_rc_comprehensive.vcd");
    $dumpvars(0, tb_multi_rc_comprehensive);
    `endif

    @(posedge rst_n);
    repeat(3) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  NPU Multi-Row/Col Comprehensive Test (Tile-Loop Architecture)");
    $display("  PE Array: %0dx%0d, DATA_W=%0d, ACC_W=%0d", ROWS, COLS, DATA_W, ACC_W);
    $display("  Architecture: tile-loop OS, weight routed to target_col");
    $display("  Standard matmul: C[M×N] = A[M×K] × B[K×N]");
    $display("################################################################");

    // =================================================================
    // Test 1: OS tile-loop M=1, N=2, K=4
    //   A[0,:]  = [5,6,7,8]
    //   B[:,0]  = [1,2,3,4]   C[0][0] = 5+12+21+32 = 70
    //   B[:,1]  = [2,3,4,5]   C[0][1] = 10+18+28+40 = 96
    //   (Standard matmul, no systolic shift)
    // =================================================================
    $display("");
    $display("--- Test 1: OS tile-loop M=1,N=2,K=4 ---");
    $display("  A=[5,6,7,8]  B[:,0]=[1,2,3,4] B[:,1]=[2,3,4,5]");
    $display("  C[0][0]=70  C[0][1]=96");
    // N=2 tiles: j=0 routes to col0, j=1 routes to col1
    run_npu(32'd1, 32'd2, 32'd4,
            32'h1000, 32'h1010, 32'h1020,
            CTRL_OS, 200000);
    got0 = dram[32'h1020 >> 2];
    got1 = dram[32'h1024 >> 2];
    check_result(1, got0, 32'd70, "T1_C[0][0]");
    check_result(2, got1, 32'd96, "T1_C[0][1]");
    $display("  C[0][0]=%0d C[0][1]=%0d", $signed(got0), $signed(got1));

    // =================================================================
    // Test 2: OS tile-loop M=1, N=2, K=8, alternating weights
    //   A=[2,2,2,2,2,2,2,2]
    //   B[:,0]=[1,-1,2,-2,3,-3,4,-4]  C[0][0]=0
    //   B[:,1]=[4,-4,3,-3,2,-2,1,-1]  C[0][1]=0
    // =================================================================
    $display("");
    $display("--- Test 2: OS tile-loop M=1,N=2,K=8 Alternating+- ---");
    $display("  C[0][0]=0  C[0][1]=0 (symmetric cancellation)");
    run_npu(32'd1, 32'd2, 32'd8,
            32'h1100, 32'h1120, 32'h1140,
            CTRL_OS, 200000);
    got0 = dram[32'h1140 >> 2];
    got1 = dram[32'h1144 >> 2];
    check_result(3, got0, 32'd0,  "T2_C[0][0]");
    check_result(4, got1, 32'd0,  "T2_C[0][1]");

    // =================================================================
    // Test 3: OS tile-loop M=1, N=2, K=4, boundary values
    //   A=[127,-128,1,0]
    //   B[:,0]=[127,-128,1,0]  C[0][0]=127*127+128*128+1+0=32514
    //   B[:,1]=[1,0,127,-128]  C[0][1]=127+0+127+0=254
    // =================================================================
    $display("");
    $display("--- Test 3: OS tile-loop M=1,N=2,K=4 Boundary ---");
    $display("  A=[127,-128,1,0]  B[:,0]=[127,-128,1,0]  B[:,1]=[1,0,127,-128]");
    $display("  C[0][0]=32514  C[0][1]=254");
    run_npu(32'd1, 32'd2, 32'd4,
            32'h1200, 32'h1210, 32'h1230,
            CTRL_OS, 200000);
    got0 = dram[32'h1230 >> 2];
    got1 = dram[32'h1234 >> 2];
    check_result(5, got0, 32'd32514,  "T3_C[0][0]");
    check_result(6, got1, 32'd254,    "T3_C[0][1]");

    // =================================================================
    // Test 4: OS tile-loop M=1, N=2, K=4 Zero weights
    //   C[0][0]=0, C[0][1]=0
    // =================================================================
    $display("");
    $display("--- Test 4: OS tile-loop M=1,N=2,K=4 Zero Weights ---");
    run_npu(32'd1, 32'd2, 32'd4,
            32'h1300, 32'h1310, 32'h1330,
            CTRL_OS, 200000);
    got0 = dram[32'h1330 >> 2];
    got1 = dram[32'h1334 >> 2];
    check_result(7, got0, 32'd0, "T4_C[0][0]");
    check_result(8, got1, 32'd0, "T4_C[0][1]");

    // =================================================================
    // Test 5: Back-to-Back
    //   Run 1: same as T1 → C[0][0]=70, C[0][1]=96
    //   Run 2: same as T3 → C[0][0]=32514, C[0][1]=254
    // =================================================================
    $display("");
    $display("--- Test 5: Back-to-Back ---");
    $display("  Run 1: T1 data -> C[0][0]=70, C[0][1]=96");
    run_npu(32'd1, 32'd2, 32'd4,
            32'h1000, 32'h1010, 32'h1400,
            CTRL_OS, 200000);
    got0 = dram[32'h1400 >> 2];
    got1 = dram[32'h1404 >> 2];
    check_result(9,  got0, 32'd70, "T5_R1_C[0][0]");
    check_result(10, got1, 32'd96, "T5_R1_C[0][1]");

    $display("  Run 2: T3 data -> C[0][0]=32514, C[0][1]=254");
    run_npu(32'd1, 32'd2, 32'd4,
            32'h1200, 32'h1210, 32'h1410,
            CTRL_OS, 200000);
    got0 = dram[32'h1410 >> 2];
    got1 = dram[32'h1414 >> 2];
    check_result(11, got0, 32'd32514, "T5_R2_C[0][0]");
    check_result(12, got1, 32'd254,   "T5_R2_C[0][1]");

    // =================================================================
    // Test 6: Verify col0 != col1 when B columns are distinct
    //   T1: C[0][0]=70, C[0][1]=96 → different columns → different results
    // =================================================================
    $display("");
    $display("--- Test 6: Different B columns → different C results ---");
    $display("  T1 results: C[0][0]=%0d C[0][1]=%0d",
             $signed(dram[32'h1020 >> 2]), $signed(dram[32'h1024 >> 2]));
    if (dram[32'h1020 >> 2] !== dram[32'h1024 >> 2]) begin
        $display("  [PASS] T6: C[0][0](%0d) != C[0][1](%0d) — independent column routing works",
                 $signed(dram[32'h1020 >> 2]), $signed(dram[32'h1024 >> 2]));
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("  [FAIL] T6: C[0][0]==C[0][1]==%0d — column routing not working!",
                 $signed(dram[32'h1020 >> 2]));
        fail_cnt = fail_cnt + 1;
    end

    // =================================================================
    // Summary
    // =================================================================
    $display("");
    $display("################################################################");
    $display("  PE Array: %0dx%0d", ROWS, COLS);
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
    #(CLK_T * 3000000);
    $display("\nFATAL: Global timeout (3M cycles)!");
    $finish;
end

endmodule
