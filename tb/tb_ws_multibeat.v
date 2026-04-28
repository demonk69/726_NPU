// =============================================================================
// Module  : tb_ws_multibeat
// Project : NPU_prj
// Desc    : WS Multi-Round Dot Product Verification
//
//   WS Mode Full Specification Validation:
//   - Verifies TRUE Weight-Stationary dot product semantics
//   - Weight vector W (K elements) is loaded ONCE per NPU invocation.
//     Each beat outputs: acc_out = acc_in(=0) + W[i] * A[i]
//     The DMA captures ALL K beat results (r_len = K*4 bytes).
//     CPU-side accumulates the K results → dot product.
//
//   Test structure:
//     M rounds × K elements per round:
//       Round m: send W and A_m vectors, collect K individual products.
//                CPU accumulates: dot_m = sum(W[i]*A_m[i] for i=0..K-1)
//     Reference: dot_m = sum(W[i]*A_m[i])   (computed in Python / integer)
//
//   4 test scenarios:
//     TC0: INT8  K=4  M=4  (small, easy to trace)
//     TC1: INT8  K=8  M=3
//     TC2: INT8  K=16 M=2
//     TC3: INT8  K=4  M=8  (many rounds)
//
//   This testbench uses ROWS=1, COLS=1 (single PE).
//   DMA r_len = K * 4 bytes (capture all K beat results).
// =============================================================================

`timescale 1ns/1ps

module tb_ws_multibeat;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
localparam ROWS   = 1;
localparam COLS   = 1;
localparam DATA_W = 16;
localparam ACC_W  = 32;
localparam CLK_T  = 10;
localparam DRAM_SZ = 16384;

// ---------------------------------------------------------------------------
// Register offsets
// ---------------------------------------------------------------------------
localparam REG_CTRL   = 32'h00;
localparam REG_STATUS = 32'h04;
localparam REG_M_DIM  = 32'h10;
localparam REG_N_DIM  = 32'h14;
localparam REG_K_DIM  = 32'h18;
localparam REG_W_ADDR = 32'h20;
localparam REG_A_ADDR = 32'h24;
localparam REG_R_ADDR = 32'h28;

// WS ctrl: bit0=start, [3:2]=dtype(00=INT8), [5:4]=stat(00=WS)
localparam CTRL_INT8_WS = 32'h01;  // start=1, INT8, WS

// ---------------------------------------------------------------------------
// Clock & Reset
// ---------------------------------------------------------------------------
reg clk = 0;
always #(CLK_T/2) clk = ~clk;

reg rst_n = 0;
initial begin #(CLK_T*5); rst_n = 1; end

// ---------------------------------------------------------------------------
// AXI4-Lite (CPU config port)
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
// AXI4 Master (DMA <-> DRAM)
// ---------------------------------------------------------------------------
wire [31:0]       m_awaddr, m_araddr;
wire [7:0]        m_awlen, m_arlen;
wire [2:0]        m_awsize, m_arsize;
wire [1:0]        m_awburst, m_arburst;
wire              m_awvalid, m_wvalid, m_bready, m_arvalid, m_rready;
wire [ACC_W-1:0]  m_wdata, m_rdata;
wire [ACC_W/8-1:0] m_wstrb;
wire              m_wlast, m_bvalid;
wire [1:0]        m_bresp, m_rresp;
reg               m_awready, m_wready, m_bvalid_r, m_arready;
reg  [ACC_W-1:0]  m_rdata_r;
reg               m_rvalid, m_rlast;
wire              npu_irq;

assign m_bresp  = 2'b00;
assign m_rresp  = 2'b00;
assign m_rdata  = m_rdata_r;

// ---------------------------------------------------------------------------
// DRAM model
// ---------------------------------------------------------------------------
reg [ACC_W-1:0] dram [0:DRAM_SZ-1];
integer dram_i;
initial begin
    for (dram_i = 0; dram_i < DRAM_SZ; dram_i = dram_i + 1)
        dram[dram_i] = 32'h0;
end

// AXI4 Read model
reg [31:0] rd_base, rd_len_r;
reg [7:0]  rd_cnt;
reg        rd_active;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_active <= 0; rd_cnt <= 0;
        m_rvalid <= 0; m_rlast <= 0; m_rdata_r <= 0;
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

// AXI4 Write model
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
            dram[((wr_base >> 2) + wr_cnt) % DRAM_SZ] <= m_wdata;
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
npu_top #(.PHY_ROWS(16), .PHY_COLS(16), .DATA_W(DATA_W), .ACC_W(ACC_W))
u_npu (
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
// AXI-Lite BFM
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
            $display("    *** TIMEOUT *** ctrl=%0d dma=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $finish;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Test data / helpers
// ---------------------------------------------------------------------------

// INT8 pack 4 elements per 32-bit word (SUBW=4)
function [31:0] int8_pack;
    input [7:0] b0, b1, b2, b3;
    begin
        int8_pack = {b3, b2, b1, b0};
    end
endfunction

// Signed INT8: clip to [-128,127] → used for address writing
function [7:0] s8;
    input integer v;
    begin s8 = v[7:0]; end
endfunction

// ---------------------------------------------------------------------------
// Write INT8 vector (K elements, 4 bytes/word, starting at DRAM word index idx)
// ---------------------------------------------------------------------------
task write_int8_vec;
    input [31:0] base_word;   // DRAM word index (byte_addr >> 2)
    input integer K;
    input integer v0, v1, v2, v3, v4, v5, v6, v7,
                  v8, v9, v10, v11, v12, v13, v14, v15,
                  v16, v17, v18, v19, v20, v21, v22, v23,
                  v24, v25, v26, v27, v28, v29, v30, v31;
    integer arr[0:31];
    integer i, word_i;
    reg [31:0] w;
    begin
        arr[0]=v0;   arr[1]=v1;   arr[2]=v2;   arr[3]=v3;
        arr[4]=v4;   arr[5]=v5;   arr[6]=v6;   arr[7]=v7;
        arr[8]=v8;   arr[9]=v9;   arr[10]=v10; arr[11]=v11;
        arr[12]=v12; arr[13]=v13; arr[14]=v14; arr[15]=v15;
        arr[16]=v16; arr[17]=v17; arr[18]=v18; arr[19]=v19;
        arr[20]=v20; arr[21]=v21; arr[22]=v22; arr[23]=v23;
        arr[24]=v24; arr[25]=v25; arr[26]=v26; arr[27]=v27;
        arr[28]=v28; arr[29]=v29; arr[30]=v30; arr[31]=v31;
        for (i = 0; i < K; i = i + 4) begin
            word_i = i / 4;
            w = { s8(arr[i+3]), s8(arr[i+2]), s8(arr[i+1]), s8(arr[i]) };
            dram[base_word + word_i] = w;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Run one WS NPU invocation:
//   W at w_word_addr, A at a_word_addr, result at r_word_addr.
//   r_len = K * 4 bytes (capture all K beat products).
//   Returns: sum of K beat products (CPU-side dot product).
// ---------------------------------------------------------------------------
task run_ws_and_accumulate;
    input [31:0] w_word_addr;   // DRAM word index for W
    input [31:0] a_word_addr;   // DRAM word index for A
    input [31:0] r_word_addr;   // DRAM word index for result
    input integer K;
    output integer dot_sum;     // CPU-accumulated dot product
    integer i;
    integer prod;
    integer timeout;
    begin
        timeout = K * 5000 + 50000;
        axi_write(REG_M_DIM, 32'd1);
        axi_write(REG_N_DIM, 32'd1);
        axi_write(REG_K_DIM, K);
        axi_write(REG_W_ADDR, w_word_addr << 2);  // byte address
        axi_write(REG_A_ADDR, a_word_addr << 2);
        axi_write(REG_R_ADDR, r_word_addr << 2);
        axi_write(REG_CTRL, CTRL_INT8_WS);
        wait_done(timeout);

        // CPU accumulates K beat results from DRAM
        dot_sum = 0;
        for (i = 0; i < K; i = i + 1) begin
            prod = $signed(dram[r_word_addr + i]);
            dot_sum = dot_sum + prod;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Test counters
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;
integer dot_got, dot_exp;

// ---------------------------------------------------------------------------
// Test Case 0: INT8 WS, K=4, M=4 rounds
//
//  W = [2, 3, -1, 4]
//  A[0] = [1, 2, 3, 4]    dot = 2*1 + 3*2 + (-1)*3 + 4*4 = 2+6-3+16 = 21
//  A[1] = [-1, 1, -1, 1]  dot = -2+3+1+4 = 6
//  A[2] = [10, 0, -5, 2]  dot = 20+0+5+8 = 33
//  A[3] = [1, 1, 1, 1]    dot = 2+3-1+4 = 8
// ---------------------------------------------------------------------------
task test_case_0;
    integer m;
    integer dot_exp_arr[0:3];
    begin
        $display("");
        $display("--- TC0: INT8 WS K=4 M=4 ---");
        $display("    W=[2,3,-1,4], 4 activation rows, CPU-side dot accumulation");

        // W at DRAM word 0x100 (byte 0x400)
        // A rows at 0x110, 0x120, 0x130, 0x140
        // R at 0x150 (each K=4 words = 0x10 words)
        // Enough spacing so they don't overlap

        // Write W = [2, 3, -1, 4] as one 32-bit word (4 bytes)
        dram[32'h100] = int8_pack(8'd2, 8'd3, 8'hFF, 8'd4); // -1 = 0xFF

        // Write A rows
        dram[32'h110] = int8_pack(8'd1, 8'd2, 8'd3, 8'd4);
        dram[32'h120] = int8_pack(8'hFF, 8'd1, 8'hFF, 8'd1); // -1,1,-1,1
        dram[32'h130] = int8_pack(8'd10, 8'd0, 8'hFB, 8'd2); // 10,0,-5,2
        dram[32'h140] = int8_pack(8'd1, 8'd1, 8'd1, 8'd1);

        dot_exp_arr[0] = 21;
        dot_exp_arr[1] = 6;
        dot_exp_arr[2] = 33;
        dot_exp_arr[3] = 8;

        for (m = 0; m < 4; m = m + 1) begin
            run_ws_and_accumulate(32'h100, 32'h110 + m*32'h10,
                                  32'h150 + m*32'h10, 4, dot_got);
            dot_exp = dot_exp_arr[m];
            if (dot_got === dot_exp) begin
                $display("  [PASS] round %0d: dot=%0d", m, dot_got);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  [FAIL] round %0d: got=%0d exp=%0d", m, dot_got, dot_exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

// ---------------------------------------------------------------------------
// Test Case 1: INT8 WS, K=8, M=3 rounds
//
//  W = [1, -2, 3, -4, 5, -6, 7, -8]
//  A[0] = [1,1,1,1,1,1,1,1]   dot = 1-2+3-4+5-6+7-8 = -4
//  A[1] = [2,2,2,2,2,2,2,2]   dot = 2*(-4) = -8
//  A[2] = [1,0,1,0,1,0,1,0]   dot = 1+3+5+7 = 16
// ---------------------------------------------------------------------------
task test_case_1;
    integer m;
    integer dot_exp_arr[0:2];
    begin
        $display("");
        $display("--- TC1: INT8 WS K=8 M=3 ---");
        $display("    W=[1,-2,3,-4,5,-6,7,-8], 3 activation rows");

        // W at 0x200 (2 words), A at 0x210/0x220/0x230 (2 words each), R at 0x240+
        dram[32'h200] = int8_pack(8'd1, 8'hFE, 8'd3, 8'hFC);  // 1,-2,3,-4
        dram[32'h201] = int8_pack(8'd5, 8'hFA, 8'd7, 8'hF8);  // 5,-6,7,-8

        // A[0] = [1,1,1,1,1,1,1,1]
        dram[32'h210] = int8_pack(8'd1, 8'd1, 8'd1, 8'd1);
        dram[32'h211] = int8_pack(8'd1, 8'd1, 8'd1, 8'd1);
        // A[1] = [2,2,2,2,2,2,2,2]
        dram[32'h220] = int8_pack(8'd2, 8'd2, 8'd2, 8'd2);
        dram[32'h221] = int8_pack(8'd2, 8'd2, 8'd2, 8'd2);
        // A[2] = [1,0,1,0,1,0,1,0]
        dram[32'h230] = int8_pack(8'd1, 8'd0, 8'd1, 8'd0);
        dram[32'h231] = int8_pack(8'd1, 8'd0, 8'd1, 8'd0);

        dot_exp_arr[0] = -4;
        dot_exp_arr[1] = -8;
        dot_exp_arr[2] = 16;

        for (m = 0; m < 3; m = m + 1) begin
            run_ws_and_accumulate(32'h200, 32'h210 + m*32'h10,
                                  32'h240 + m*32'h10, 8, dot_got);
            dot_exp = dot_exp_arr[m];
            if (dot_got === dot_exp) begin
                $display("  [PASS] round %0d: dot=%0d", m, dot_got);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  [FAIL] round %0d: got=%0d exp=%0d", m, dot_got, dot_exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

// ---------------------------------------------------------------------------
// Test Case 2: INT8 WS, K=16, M=2 rounds
//
//  W = [1,1,1,1, -1,-1,-1,-1, 2,2,2,2, -2,-2,-2,-2]
//  A[0] = [10 repeated 16]
//          dot = (4*1 - 4*1 + 4*2 - 4*2)*10 = 0
//  A[1] = [i for i in 1..16]
//          dot = sum(W[i]*A[i]) = 1+2+3+4 - 5-6-7-8 + 2*9+2*10+2*11+2*12 - 2*13-2*14-2*15-2*16
//              = 10 - 26 + 84 - 116 = -48
// ---------------------------------------------------------------------------
task test_case_2;
    integer m;
    integer dot_exp_arr[0:1];
    begin
        $display("");
        $display("--- TC2: INT8 WS K=16 M=2 ---");
        $display("    W=[1,1,1,1,-1,-1,-1,-1,2,2,2,2,-2,-2,-2,-2]");

        // W at 0x300 (4 words)
        dram[32'h300] = int8_pack(8'd1,   8'd1,   8'd1,   8'd1);
        dram[32'h301] = int8_pack(8'hFF,  8'hFF,  8'hFF,  8'hFF); // -1,-1,-1,-1
        dram[32'h302] = int8_pack(8'd2,   8'd2,   8'd2,   8'd2);
        dram[32'h303] = int8_pack(8'hFE,  8'hFE,  8'hFE,  8'hFE); // -2,-2,-2,-2

        // A[0] = [10,10,...,10] (16 elements)
        dram[32'h310] = int8_pack(8'd10, 8'd10, 8'd10, 8'd10);
        dram[32'h311] = int8_pack(8'd10, 8'd10, 8'd10, 8'd10);
        dram[32'h312] = int8_pack(8'd10, 8'd10, 8'd10, 8'd10);
        dram[32'h313] = int8_pack(8'd10, 8'd10, 8'd10, 8'd10);

        // A[1] = [1..16]
        dram[32'h320] = int8_pack(8'd1,  8'd2,  8'd3,  8'd4);
        dram[32'h321] = int8_pack(8'd5,  8'd6,  8'd7,  8'd8);
        dram[32'h322] = int8_pack(8'd9,  8'd10, 8'd11, 8'd12);
        dram[32'h323] = int8_pack(8'd13, 8'd14, 8'd15, 8'd16);

        dot_exp_arr[0] = 0;
        dot_exp_arr[1] = 1+2+3+4 - 5-6-7-8 + 2*(9+10+11+12) - 2*(13+14+15+16);
        // = 10 - 26 + 84 - 116 = -48

        for (m = 0; m < 2; m = m + 1) begin
            run_ws_and_accumulate(32'h300, 32'h310 + m*32'h10,
                                  32'h340 + m*32'h40, 16, dot_got);
            dot_exp = dot_exp_arr[m];
            if (dot_got === dot_exp) begin
                $display("  [PASS] round %0d: dot=%0d", m, dot_got);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  [FAIL] round %0d: got=%0d exp=%0d", m, dot_got, dot_exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

// ---------------------------------------------------------------------------
// Test Case 3: INT8 WS, K=4, M=8 rounds (verify clean state between rounds)
//
//  W = [1, 2, 3, 4]
//  A[m] = [m, m, m, m]   dot = (1+2+3+4)*m = 10*m
// ---------------------------------------------------------------------------
task test_case_3;
    integer m;
    integer dot_exp_val;
    begin
        $display("");
        $display("--- TC3: INT8 WS K=4 M=8 (state isolation between rounds) ---");
        $display("    W=[1,2,3,4], A_m=[m,m,m,m], expected dot=10*m");

        // W at 0x400
        dram[32'h400] = int8_pack(8'd1, 8'd2, 8'd3, 8'd4);

        // A rows at 0x410, 0x420, ..., 0x480
        // A_0=[0,0,0,0], A_1=[1,1,1,1], ..., A_7=[7,7,7,7]
        dram[32'h410] = int8_pack(8'd0, 8'd0, 8'd0, 8'd0);
        dram[32'h420] = int8_pack(8'd1, 8'd1, 8'd1, 8'd1);
        dram[32'h430] = int8_pack(8'd2, 8'd2, 8'd2, 8'd2);
        dram[32'h440] = int8_pack(8'd3, 8'd3, 8'd3, 8'd3);
        dram[32'h450] = int8_pack(8'd4, 8'd4, 8'd4, 8'd4);
        dram[32'h460] = int8_pack(8'd5, 8'd5, 8'd5, 8'd5);
        dram[32'h470] = int8_pack(8'd6, 8'd6, 8'd6, 8'd6);
        dram[32'h480] = int8_pack(8'd7, 8'd7, 8'd7, 8'd7);

        for (m = 0; m < 8; m = m + 1) begin
            run_ws_and_accumulate(32'h400, 32'h410 + m*32'h10,
                                  32'h490 + m*32'h10, 4, dot_got);
            dot_exp_val = 10 * m;
            if (dot_got === dot_exp_val) begin
                $display("  [PASS] round %0d: dot=%0d (10*%0d)", m, dot_got, m);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  [FAIL] round %0d: got=%0d exp=%0d", m, dot_got, dot_exp_val);
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
    $dumpfile("tb_ws_multibeat.vcd");
    $dumpvars(0, tb_ws_multibeat);
    `endif

    @(posedge rst_n);
    repeat(3) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  WS Multi-Round Dot Product Verification (ROWS=%0d, COLS=%0d)", ROWS, COLS);
    $display("  Tests: TC0(K=4,M=4) TC1(K=8,M=3) TC2(K=16,M=2) TC3(K=4,M=8)");
    $display("  WS semantics: each beat outputs W[i]*A[i]; CPU sums K beats");
    $display("################################################################");

    test_case_0;
    test_case_1;
    test_case_2;
    test_case_3;

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
    #(CLK_T * 10000000);
    $display("\nFATAL: Global timeout!");
    $finish;
end

endmodule
