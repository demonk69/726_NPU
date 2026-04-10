// =============================================================================
// tb_array_scale_body.v - Common testbench body for 1x1 PE K-depth verification
//
// This file is included by per-K wrappers (N<K>/tb_wrapper.v).
// It expects the following to be already defined:

//   ROWS, COLS, DATA_W, ACC_W, CLK_T, DRAM_SZ
//   REG_CTRL, REG_STATUS, REG_M_DIM, REG_N_DIM, REG_K_DIM, REG_W_ADDR, REG_A_ADDR, REG_R_ADDR
//   test_params.vh macros (NUM_TESTS, T<n>_*, TEST_<n>)
//   reg [ACC_W-1:0] dram [0:DRAM_SZ-1] (with $readmemh already loaded)
//   reg [31:0] expected [0:`NUM_TESTS-1] (with $readmemh already loaded)

// =============================================================================

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
            $display("    *** TIMEOUT *** ctrl=%0d dma=%0d",
                     u_npu.u_ctrl.state, u_npu.u_dma.dma_state);
            $finish;
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
// Test execution
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;
reg [31:0] got_val;

// Helper: run a single test by index
task do_test;
    input integer idx;
    input [31:0] w_addr, a_addr, r_addr, ctrl_val;
    input [31:0] k_dim, timeout;
    input [31:0] exp_val;
    input is_fp16;
    begin
        axi_write(REG_M_DIM, 32'd1);
        axi_write(REG_N_DIM, 32'd1);
        axi_write(REG_K_DIM, k_dim);
        axi_write(REG_W_ADDR, w_addr);
        axi_write(REG_A_ADDR, a_addr);
        axi_write(REG_R_ADDR, r_addr);
        axi_write(REG_CTRL, ctrl_val);
        wait_done(timeout);

        got_val = dram[r_addr >> 2];

        if (is_fp16) begin
            if (fp32_close(got_val, exp_val)) begin
                $display("  [PASS] T%0d: got=0x%08h exp=0x%08h", idx, got_val, exp_val);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  [FAIL] T%0d: got=0x%08h exp=0x%08h", idx, got_val, exp_val);
                fail_cnt = fail_cnt + 1;
            end
        end else begin
            if (got_val === exp_val) begin
                $display("  [PASS] T%0d: got=%0d", idx, $signed(got_val));
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  [FAIL] T%0d: got=%0d (0x%08h) exp=%0d (0x%08h) diff=%0d",
                         idx, $signed(got_val), got_val, $signed(exp_val), exp_val,
                         $signed(got_val) - $signed(exp_val));
                fail_cnt = fail_cnt + 1;
            end
        end
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
    $dumpfile("tb_array_scale.vcd");
    $dumpvars(0, tb_array_scale);
    `endif

    @(posedge rst_n);
    repeat(3) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  K-Depth Verification: physical PE = %0dx%0d", ROWS, COLS);
    $display("  Running %0d tests...", `NUM_TESTS);
    $display("################################################################");


`ifdef TEST_0
    $display("");
    $display("--- Test 0: %0s%0s K=%0d ---",
             `T0_IS_FP16 ? "FP16" : "INT8",
             `T0_IS_OS ? "_OS" : "_WS",
             `T0_K_DIM);
    do_test(0, `T0_W_ADDR, `T0_A_ADDR, `T0_R_ADDR,
            `T0_CTRL, `T0_K_DIM, `T0_K_DIM * 5000 + 50000,
            expected[0], `T0_IS_FP16);
`endif

`ifdef TEST_1
    $display("");
    $display("--- Test 1: %0s%0s K=%0d ---",
             `T1_IS_FP16 ? "FP16" : "INT8",
             `T1_IS_OS ? "_OS" : "_WS",
             `T1_K_DIM);
    do_test(1, `T1_W_ADDR, `T1_A_ADDR, `T1_R_ADDR,
            `T1_CTRL, `T1_K_DIM, `T1_K_DIM * 5000 + 50000,
            expected[1], `T1_IS_FP16);
`endif

`ifdef TEST_2
    $display("");
    $display("--- Test 2: %0s%0s K=%0d ---",
             `T2_IS_FP16 ? "FP16" : "INT8",
             `T2_IS_OS ? "_OS" : "_WS",
             `T2_K_DIM);
    do_test(2, `T2_W_ADDR, `T2_A_ADDR, `T2_R_ADDR,
            `T2_CTRL, `T2_K_DIM, `T2_K_DIM * 5000 + 50000,
            expected[2], `T2_IS_FP16);
`endif

`ifdef TEST_3
    $display("");
    $display("--- Test 3: %0s%0s K=%0d ---",
             `T3_IS_FP16 ? "FP16" : "INT8",
             `T3_IS_OS ? "_OS" : "_WS",
             `T3_K_DIM);
    do_test(3, `T3_W_ADDR, `T3_A_ADDR, `T3_R_ADDR,
            `T3_CTRL, `T3_K_DIM, `T3_K_DIM * 5000 + 50000,
            expected[3], `T3_IS_FP16);
`endif

`ifdef TEST_4
    $display("");
    $display("--- Test 4: %0s%0s K=%0d ---",
             `T4_IS_FP16 ? "FP16" : "INT8",
             `T4_IS_OS ? "_OS" : "_WS",
             `T4_K_DIM);
    do_test(4, `T4_W_ADDR, `T4_A_ADDR, `T4_R_ADDR,
            `T4_CTRL, `T4_K_DIM, `T4_K_DIM * 5000 + 50000,
            expected[4], `T4_IS_FP16);
`endif

`ifdef TEST_5
    $display("");
    $display("--- Test 5: %0s%0s K=%0d ---",
             `T5_IS_FP16 ? "FP16" : "INT8",
             `T5_IS_OS ? "_OS" : "_WS",
             `T5_K_DIM);
    do_test(5, `T5_W_ADDR, `T5_A_ADDR, `T5_R_ADDR,
            `T5_CTRL, `T5_K_DIM, `T5_K_DIM * 5000 + 50000,
            expected[5], `T5_IS_FP16);
`endif

`ifdef TEST_6
    $display("");
    $display("--- Test 6: %0s%0s K=%0d ---",
             `T6_IS_FP16 ? "FP16" : "INT8",
             `T6_IS_OS ? "_OS" : "_WS",
             `T6_K_DIM);
    do_test(6, `T6_W_ADDR, `T6_A_ADDR, `T6_R_ADDR,
            `T6_CTRL, `T6_K_DIM, `T6_K_DIM * 5000 + 50000,
            expected[6], `T6_IS_FP16);
`endif

`ifdef TEST_7
    $display("");
    $display("--- Test 7: %0s%0s K=%0d ---",
             `T7_IS_FP16 ? "FP16" : "INT8",
             `T7_IS_OS ? "_OS" : "_WS",
             `T7_K_DIM);
    do_test(7, `T7_W_ADDR, `T7_A_ADDR, `T7_R_ADDR,
            `T7_CTRL, `T7_K_DIM, `T7_K_DIM * 5000 + 50000,
            expected[7], `T7_IS_FP16);
`endif

`ifdef TEST_8
    $display("");
    $display("--- Test 8: %0s%0s K=%0d ---",
             `T8_IS_FP16 ? "FP16" : "INT8",
             `T8_IS_OS ? "_OS" : "_WS",
             `T8_K_DIM);
    do_test(8, `T8_W_ADDR, `T8_A_ADDR, `T8_R_ADDR,
            `T8_CTRL, `T8_K_DIM, `T8_K_DIM * 5000 + 50000,
            expected[8], `T8_IS_FP16);
`endif

`ifdef TEST_9
    $display("");
    $display("--- Test 9: %0s%0s K=%0d ---",
             `T9_IS_FP16 ? "FP16" : "INT8",
             `T9_IS_OS ? "_OS" : "_WS",
             `T9_K_DIM);
    do_test(9, `T9_W_ADDR, `T9_A_ADDR, `T9_R_ADDR,
            `T9_CTRL, `T9_K_DIM, `T9_K_DIM * 5000 + 50000,
            expected[9], `T9_IS_FP16);
`endif

`ifdef TEST_10
    $display("");
    $display("--- Test 10: %0s%0s K=%0d ---",
             `T10_IS_FP16 ? "FP16" : "INT8",
             `T10_IS_OS ? "_OS" : "_WS",
             `T10_K_DIM);
    do_test(10, `T10_W_ADDR, `T10_A_ADDR, `T10_R_ADDR,
            `T10_CTRL, `T10_K_DIM, `T10_K_DIM * 5000 + 50000,
            expected[10], `T10_IS_FP16);
`endif

`ifdef TEST_11
    $display("");
    $display("--- Test 11: %0s%0s K=%0d ---",
             `T11_IS_FP16 ? "FP16" : "INT8",
             `T11_IS_OS ? "_OS" : "_WS",
             `T11_K_DIM);
    do_test(11, `T11_W_ADDR, `T11_A_ADDR, `T11_R_ADDR,
            `T11_CTRL, `T11_K_DIM, `T11_K_DIM * 5000 + 50000,
            expected[11], `T11_IS_FP16);
`endif

    // Summary
    $display("");
    $display("################################################################");
    if (fail_cnt == 0)
        $display("  ALL %0d TESTS PASSED (%0dx%0d array)", pass_cnt, ROWS, COLS);
    else
        $display("  RESULT: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    $display("################################################################");

    #(CLK_T*20);
    $finish;
end

// Global timeout
initial begin
    #(CLK_T * 5000000);
    $display("\nFATAL: Global timeout!");
    $finish;
end

// Debug trace: print ALL activity from rst_n onward (first 70 cycles covers T0 completely)
reg [31:0] dbg_cycle = 0;
reg [3:0]  dbg_prev_st = 0;
always @(posedge clk) begin
    dbg_cycle <= dbg_cycle + 1;
    if (rst_n && dbg_cycle < 300) begin
        // State transitions
        if (u_npu.u_ctrl.state != dbg_prev_st) begin
            $display("[DBG] cyc=%0d ST %0d->%0d en=%b fl=%b lw=%b st=%b md=%b",
                dbg_cycle, dbg_prev_st, u_npu.u_ctrl.state,
                u_npu.pe_en, u_npu.pe_flush, u_npu.pe_load_w,
                u_npu.pe_stat, u_npu.pe_mode);
            dbg_prev_st <= u_npu.u_ctrl.state;
        end
        // During S_COMPUTE/S_DRAIN: show PE internals every cycle
        if (u_npu.u_ctrl.state >= 3 && u_npu.u_ctrl.state <= 9) begin
            $display("[DBG] cyc=%0d | s0_v=%b s0_w=0x%04h s0_a=0x%04h | s1_v=%b s1_m=0x%08h | ws_acc=0x%08h | v=%b acc=0x%08h",
                dbg_cycle,
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.s0_valid,
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.s0_w[15:0],
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.s0_a[15:0],
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.s1_valid,
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.s1_mul[31:0],
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.ws_acc,
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.valid_out,
                u_npu.u_pe_array.gen_row[0].gen_col[0].u_pe.acc_out);
        end
        // Show FIFO writes and AXI writes
        if (u_npu.r_fifo_wr_en)
            $display("[FIFO-WR] cyc=%0d din=0x%08h", dbg_cycle, u_npu.r_fifo_din);
        if (u_npu.u_dma.m_axi_wvalid && u_npu.u_dma.m_axi_wready)
            $display("[AXI-WD] cyc=%0d waddr=%h wdata=0x%08h", dbg_cycle, u_npu.u_dma.wb_addr_cnt, u_npu.u_dma.m_axi_wdata);
    end
end
