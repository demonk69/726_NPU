`timescale 1ns/1ps

module tb_npu_tile_ksplit_gemm;

localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;
localparam INT8_SIMD_LANES = (DATA_W >= 32) ? 4 : 2;
localparam DRAM_SZ = 2048;

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

localparam CTRL_START_OS_INT8 = 32'h11;
localparam ARR_TILE4          = 32'h80;
localparam CFG_4X4            = 32'h0;

localparam M_DIM  = 4;
localparam N_DIM  = 4;
localparam K_DIM  = 10;
localparam W_ADDR = 32'h00000100;
localparam A_ADDR = 32'h00000180;
localparam R_ADDR = 32'h00000200;

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

reg [31:0] dram [0:DRAM_SZ-1];
reg [31:0] expected [0:M_DIM*N_DIM-1];

integer a_mat [0:M_DIM*K_DIM-1];
integer b_mat [0:K_DIM*N_DIM-1];
integer pass_cnt;
integer fail_cnt;
integer ar_count;
integer aw_count;

reg [31:0] ar_addr_seen [0:7];
reg [7:0]  ar_len_seen  [0:7];
reg [31:0] aw_addr_seen [0:7];
reg [7:0]  aw_len_seen  [0:7];

npu_top #(
    .DATA_W(DATA_W),
    .ACC_W (ACC_W),
    .INT8_SIMD_LANES(INT8_SIMD_LANES),
    .PPB_DEPTH(4),
    .PPB_THRESH(4)
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

function [31:0] pack4_int8;
    input integer v0;
    input integer v1;
    input integer v2;
    input integer v3;
    begin
        pack4_int8 = ((v0 & 8'hff)      ) |
                     ((v1 & 8'hff) <<  8) |
                     ((v2 & 8'hff) << 16) |
                     ((v3 & 8'hff) << 24);
    end
endfunction

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

task expect_read;
    input integer idx;
    input [31:0] exp_addr;
    input [7:0]  exp_len;
    begin
        if (ar_addr_seen[idx] !== exp_addr || ar_len_seen[idx] !== exp_len) begin
            $display("[FAIL] read%0d addr=%08h len=%0d, expected addr=%08h len=%0d",
                     idx, ar_addr_seen[idx], ar_len_seen[idx], exp_addr, exp_len);
            fail_cnt = fail_cnt + 1;
        end else begin
            pass_cnt = pass_cnt + 1;
        end
    end
endtask

task expect_write;
    input integer idx;
    input [31:0] exp_addr;
    begin
        if (aw_addr_seen[idx] !== exp_addr || aw_len_seen[idx] !== 8'd3) begin
            $display("[FAIL] write%0d addr=%08h len=%0d, expected addr=%08h len=3",
                     idx, aw_addr_seen[idx], aw_len_seen[idx], exp_addr);
            fail_cnt = fail_cnt + 1;
        end else begin
            pass_cnt = pass_cnt + 1;
        end
    end
endtask

task check_result;
    input integer idx;
    reg [31:0] got;
    reg [31:0] exp;
    begin
        got = dram[(R_ADDR >> 2) + idx];
        exp = expected[idx];
        if (got !== exp) begin
            $display("[FAIL] C[%0d][%0d] got=%0d (0x%08h), expected=%0d (0x%08h)",
                     idx / N_DIM, idx % N_DIM,
                     $signed(got), got, $signed(exp), exp);
            fail_cnt = fail_cnt + 1;
        end else begin
            pass_cnt = pass_cnt + 1;
        end
    end
endtask

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
        ar_count  <= 0;
    end else begin
        m_arready <= 1'b1;
        if (m_arvalid && m_arready && ar_count < 8) begin
            ar_addr_seen[ar_count] <= m_araddr;
            ar_len_seen[ar_count]  <= m_arlen;
            ar_count <= ar_count + 1;
        end

        if (m_rvalid && m_rready && m_rlast) begin
            m_rvalid <= 1'b0;
            m_rlast  <= 1'b0;
        end else if (!rd_active && !m_rvalid && m_arvalid && m_arready) begin
            rd_active <= 1'b1;
            rd_base   <= m_araddr;
            rd_len    <= m_arlen;
            rd_cnt    <= 8'd0;
            m_rvalid  <= 1'b0;
            m_rlast   <= 1'b0;
        end else if (rd_active && (!m_rvalid || (m_rvalid && m_rready))) begin
            m_rdata  <= dram[((rd_base >> 2) + rd_cnt) % DRAM_SZ];
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
            wr_phase <= 1'b1;
            wr_base  <= m_awaddr;
            wr_cnt   <= 8'd0;
            m_wready <= 1'b1;
            if (aw_count < 8) begin
                aw_addr_seen[aw_count] <= m_awaddr;
                aw_len_seen[aw_count]  <= m_awlen;
            end
            aw_count <= aw_count + 1;
        end

        if (wr_phase && m_wvalid && m_wready) begin
            dram[((wr_base >> 2) + wr_cnt) % DRAM_SZ] <= m_wdata;
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

`ifdef DEBUG_KSPLIT
reg [3:0] dbg_state;
reg [31:0] dbg_k_index;
always @(posedge clk) begin
    if (!rst_n) begin
        dbg_state <= 4'hf;
        dbg_k_index <= 32'hffff_ffff;
    end else begin
        if (dbg_state !== u_npu.u_ctrl.state ||
            dbg_k_index !== u_npu.u_ctrl.tile_k_index ||
            u_npu.u_ctrl.dma_w_start || u_npu.u_ctrl.dma_a_start ||
            u_npu.u_ctrl.dma_w_done || u_npu.u_ctrl.dma_a_done ||
            u_npu.u_ctrl.w_ppb_swap || u_npu.u_ctrl.a_ppb_swap) begin
            $display("[DBG] t=%0t state=%0d kidx=%0d kbase=%0d klen=%0d dma=%0d ws=%0b as=%0b wd=%0b ad=%0b dl=%0b sw=%0b",
                     $time, u_npu.u_ctrl.state, u_npu.u_ctrl.tile_k_index,
                     u_npu.u_ctrl.tile_k_base, u_npu.u_ctrl.tile_k_len,
                     u_npu.u_dma.dma_state,
                     u_npu.u_ctrl.dma_w_start, u_npu.u_ctrl.dma_a_start,
                     u_npu.u_ctrl.dma_w_done, u_npu.u_ctrl.dma_a_done,
                     u_npu.u_ctrl.dma_load_done,
                     u_npu.u_ctrl.w_ppb_swap);
            dbg_state <= u_npu.u_ctrl.state;
            dbg_k_index <= u_npu.u_ctrl.tile_k_index;
        end
    end
end
`endif

integer i;
integer r;
integer c;
integer k;
integer acc;

initial begin
    for (i = 0; i < DRAM_SZ; i = i + 1)
        dram[i] = 32'd0;
    for (i = 0; i < 8; i = i + 1) begin
        ar_addr_seen[i] = 32'd0;
        ar_len_seen[i]  = 8'd0;
        aw_addr_seen[i] = 32'd0;
        aw_len_seen[i]  = 8'd0;
    end

    a_mat[0]  =  3; a_mat[1]  = -2; a_mat[2]  =  5; a_mat[3]  =  7; a_mat[4]  = -1;
    a_mat[5]  =  4; a_mat[6]  =  2; a_mat[7]  = -3; a_mat[8]  =  6; a_mat[9]  =  1;
    a_mat[10] = -1; a_mat[11] =  4; a_mat[12] =  0; a_mat[13] =  6; a_mat[14] = -5;
    a_mat[15] =  2; a_mat[16] =  3; a_mat[17] = -4; a_mat[18] =  1; a_mat[19] =  5;
    a_mat[20] =  8; a_mat[21] = -3; a_mat[22] =  2; a_mat[23] = -5; a_mat[24] =  7;
    a_mat[25] = -6; a_mat[26] =  1; a_mat[27] =  4; a_mat[28] = -2; a_mat[29] =  3;
    a_mat[30] =  0; a_mat[31] =  9; a_mat[32] = -4; a_mat[33] =  1; a_mat[34] =  3;
    a_mat[35] = -2; a_mat[36] =  5; a_mat[37] = -7; a_mat[38] =  2; a_mat[39] =  6;

    b_mat[0]  =  2; b_mat[1]  = -1; b_mat[2]  =  3; b_mat[3]  =  0;
    b_mat[4]  =  5; b_mat[5]  =  4; b_mat[6]  = -2; b_mat[7]  =  1;
    b_mat[8]  = -3; b_mat[9]  =  6; b_mat[10] =  1; b_mat[11] = -4;
    b_mat[12] =  7; b_mat[13] =  0; b_mat[14] = -5; b_mat[15] =  2;
    b_mat[16] = -2; b_mat[17] =  3; b_mat[18] =  4; b_mat[19] = -1;
    b_mat[20] =  1; b_mat[21] = -5; b_mat[22] =  2; b_mat[23] =  6;
    b_mat[24] =  4; b_mat[25] =  2; b_mat[26] = -3; b_mat[27] =  5;
    b_mat[28] = -6; b_mat[29] =  1; b_mat[30] =  7; b_mat[31] = -2;
    b_mat[32] =  3; b_mat[33] = -4; b_mat[34] =  0; b_mat[35] =  2;
    b_mat[36] = -1; b_mat[37] =  5; b_mat[38] = -6; b_mat[39] =  4;

    for (k = 0; k < K_DIM; k = k + 1) begin
        dram[(W_ADDR >> 2) + k] = pack4_int8(
            b_mat[k*N_DIM + 0], b_mat[k*N_DIM + 1],
            b_mat[k*N_DIM + 2], b_mat[k*N_DIM + 3]);
        dram[(A_ADDR >> 2) + k] = pack4_int8(
            a_mat[0*K_DIM + k], a_mat[1*K_DIM + k],
            a_mat[2*K_DIM + k], a_mat[3*K_DIM + k]);
    end

    for (r = 0; r < M_DIM; r = r + 1) begin
        for (c = 0; c < N_DIM; c = c + 1) begin
            acc = 0;
            for (k = 0; k < K_DIM; k = k + 1)
                acc = acc + a_mat[r*K_DIM + k] * b_mat[k*N_DIM + c];
            expected[r*N_DIM + c] = acc;
        end
    end

    s_awaddr = 0; s_wdata = 0; s_wstrb = 0;
    s_awvalid = 0; s_wvalid = 0; s_bready = 0;
    s_araddr = 0; s_arvalid = 0; s_rready = 0;
    pass_cnt = 0;
    fail_cnt = 0;

    @(posedge rst_n);
    repeat (4) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  4x4 Tile K-split GEMM Test: int8_4x4x10_ppb4");
    $display("################################################################");

    axi_write(REG_M_DIM, M_DIM);
    axi_write(REG_N_DIM, N_DIM);
    axi_write(REG_K_DIM, K_DIM);
    axi_write(REG_W_ADDR, W_ADDR);
    axi_write(REG_A_ADDR, A_ADDR);
    axi_write(REG_R_ADDR, R_ADDR);
    axi_write(REG_ARR_CFG, ARR_TILE4);
    axi_write(REG_CFG_SHAPE, CFG_4X4);
    axi_write(REG_CTRL, CTRL_START_OS_INT8);

    wait_done(8000);

    if (ar_count !== 6) begin
        $display("[FAIL] expected 6 K-split read bursts, got %0d", ar_count);
        fail_cnt = fail_cnt + 1;
    end else begin
        pass_cnt = pass_cnt + 1;
    end
    expect_read(0, W_ADDR + 32'h00, 8'd3);
    expect_read(1, A_ADDR + 32'h00, 8'd3);
    expect_read(2, W_ADDR + 32'h10, 8'd3);
    expect_read(3, A_ADDR + 32'h10, 8'd3);
    expect_read(4, W_ADDR + 32'h20, 8'd1);
    expect_read(5, A_ADDR + 32'h20, 8'd1);

    if (aw_count !== 4) begin
        $display("[FAIL] expected final-only 4 row write bursts, got %0d", aw_count);
        fail_cnt = fail_cnt + 1;
    end else begin
        pass_cnt = pass_cnt + 1;
    end
    expect_write(0, R_ADDR + 32'h00);
    expect_write(1, R_ADDR + 32'h10);
    expect_write(2, R_ADDR + 32'h20);
    expect_write(3, R_ADDR + 32'h30);

    for (i = 0; i < M_DIM*N_DIM; i = i + 1)
        check_result(i);

    if (fail_cnt == 0) begin
        $display("[PASS] tb_npu_tile_ksplit_gemm: ALL %0d CHECKS PASSED", pass_cnt);
    end else begin
        $display("[FAIL] tb_npu_tile_ksplit_gemm: %0d passed, %0d failed",
                 pass_cnt, fail_cnt);
        $fatal;
    end

    $finish;
end

initial begin
    #(CLK_T * 200000);
    $display("[FAIL] tb_npu_tile_ksplit_gemm global timeout");
    $finish;
end

endmodule
