// =============================================================================
// Module  : tb_classifier
// Project : NPU_prj
// Desc    : Simple end-to-end test: dot product y = W·x (M=1, N=1, K=4).
//           Uses INT8 mode, OS-like internal accumulation.
//           Tests the full data path: DRAM → DMA → PPBuf → PE → FIFO → DRAM.
// =============================================================================

`timescale 1ns/1ps

module tb_classifier;

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

// Test data: y = [3, 7, -2, 5] · [10, 20, 30, 40] = 30+140-60+200 = 310
localparam EXP_RESULT  = 32'd310;

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
wire [31:0]       m_awaddr;
wire [7:0]        m_awlen;
wire [2:0]        m_awsize;
wire [1:0]        m_awburst;
wire              m_awvalid;
reg               m_awready;
wire [ACC_W-1:0]  m_wdata;
wire [ACC_W/8-1:0] m_wstrb;
wire              m_wlast;
wire              m_wvalid;
reg               m_wready;
wire [1:0]        m_bresp_out;
reg               m_bvalid;
wire              m_bready;
wire [31:0]       m_araddr;
wire [7:0]        m_arlen;
wire [2:0]        m_arsize;
wire [1:0]        m_arburst;
wire              m_arvalid;
reg               m_arready;
reg  [ACC_W-1:0]  m_rdata_r;
wire [ACC_W-1:0]  m_rdata;
wire [1:0]        m_rresp;
reg               m_rvalid;
reg               m_rlast;
wire              m_rready;
wire              npu_irq;

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
    // W = [3, 7, -2, 5]: 4 INT8 packed in one 32-bit word (little-endian)
    // PPBuf 4-way read: each 32-bit word → 4 sign-extended INT8
    // byte[0]=3, byte[1]=7, byte[2]=0xFE(-2), byte[3]=5
    // DMA reads from 0x1000 → dram[0x1000>>2] = dram[1024]
    dram[1024] = 32'h05FE0703;  // W at 0x1000
    // A = [10, 20, 30, 40]: DMA reads from 0x1004 → dram[0x1004>>2] = dram[1025]
    dram[1025] = 32'h281E140A;  // A at 0x1004
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
        if (bfm_cnt >= 200)
            $display("    [BFM-W] TIMEOUT addr=0x%08h", addr);
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
            $display("    [NPU] DONE after %0d polls", wdg);
            axi_write(REG_CTRL, 32'h00);
        end else begin
            $display("    [NPU] *** TIMEOUT after %0d cycles! ***", timeout_cyc);
            $display("    ctrl_state=%0d, dma_state=%0d", u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $finish;
        end
    end
endtask

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;

initial begin
    s_awaddr<=0;s_awvalid<=0;s_wdata<=0;s_wstrb<=0;s_wvalid<=0;s_bready<=0;
    s_araddr<=0;s_arvalid<=0;s_rready<=0;
    m_awready<=1; m_wready<=0;
    pass_cnt=0; fail_cnt=0;

    // Dump waveform
    $dumpfile("tb_classifier.vcd");
    $dumpvars(0, tb_classifier);

    @(posedge rst_n);
    repeat(3) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  Simple Dot Product Test: y = W . x");
    $display("  W = [3, 7, -2, 5], x = [10, 20, 30, 40]");
    $display("  Expected: 3*10 + 7*20 + (-2)*30 + 5*40 = 310");
    $display("  PE Array: %0dx%0d, DATA_W=%0d, ACC_W=%0d", ROWS, COLS, DATA_W, ACC_W);
    $display("################################################################");

    // Configure NPU: M=1, N=1, K=4, INT8, OS mode
    $display("\n[STEP] Configuring NPU...");
    axi_write(REG_M_DIM, 32'd1);
    axi_write(REG_N_DIM, 32'd1);
    axi_write(REG_K_DIM, 32'd4);
    axi_write(REG_W_ADDR, 32'h00001000);  // dram[0] = W
    axi_write(REG_A_ADDR, 32'h00001004);  // dram[1] = A
    axi_write(REG_R_ADDR, 32'h00001008);  // result at dram[2]
    // Start with OS mode (bit[4]=1) + INT8
    axi_write(REG_CTRL, CTRL_START | CTRL_OS);

    // Wait for NPU done
    $display("[STEP] Waiting for NPU...");
    wait_done(50000);

    // Check result (R_ADDR=0x1008 → dram[0x1008>>2]=dram[1026])
    $display("\n[CHECK] Result at DRAM[1026] (addr 0x1008):");
    $display("    Got:      0x%08h (%0d)", dram[1026], $signed(dram[1026]));
    $display("    Expected: 0x%08h (%0d)", EXP_RESULT[31:0], $signed(EXP_RESULT[31:0]));
    if (dram[1026] === EXP_RESULT[31:0]) begin
        $display("    [PASS] Dot product CORRECT!");
        pass_cnt = pass_cnt + 1;
    end else begin
        $display("    [FAIL] Dot product WRONG! diff=%0d", $signed(dram[2]) - $signed(EXP_RESULT[31:0]));
        fail_cnt = fail_cnt + 1;
    end

    $display("");
    if (fail_cnt == 0)
        $display("  ALL %0d CHECKS PASSED", pass_cnt);
    else
        $display("  RESULT: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);

    #(CLK_T*20);
    $finish;
end

// Global timeout
initial begin
    #(CLK_T * 200000);
    $display("\nFATAL: Global timeout!");
    $finish;
end

endmodule
