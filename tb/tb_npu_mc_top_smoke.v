// =============================================================================
// tb_npu_mc_top_smoke - wrapper independence smoke for multi-core NPU
// =============================================================================
`timescale 1ns/1ps

module tb_npu_mc_top_smoke;
    localparam CLK_T      = 10;
    localparam NUM_CORES  = 2;
    localparam PHY_ROWS   = 16; localparam PHY_COLS = 16;
    localparam DATA_W     = 32; localparam ACC_W = 32;
    localparam PPB_DEPTH  = 8192; localparam PPB_THRESH = 16;
    localparam INT8_LANES = 4; localparam DRAM_SZ = 4096;
    localparam TIMEOUT    = 50000;

    localparam REG_CTRL = 32'h00; localparam REG_STATUS = 32'h04;
    localparam REG_M_DIM = 32'h10; localparam REG_N_DIM = 32'h14;
    localparam REG_K_DIM = 32'h18; localparam REG_W_ADDR = 32'h20;
    localparam REG_A_ADDR = 32'h24; localparam REG_R_ADDR = 32'h28;
    localparam REG_ARR_CFG = 32'h30; localparam REG_CFG_SHAPE = 32'h3C;
    localparam REG_BIAS_ADDR = 32'h98; localparam REG_QUANT_CFG = 32'h9C;

    reg clk = 1'b0; always #(CLK_T/2) clk = ~clk;
    reg rst_n = 1'b0;

    reg  [NUM_CORES*32-1:0] s_awaddr, s_wdata, s_araddr;
    reg  [NUM_CORES-1:0]    s_awvalid, s_wvalid, s_bready, s_arvalid, s_rready;
    reg  [NUM_CORES*4-1:0]  s_wstrb;
    wire [NUM_CORES-1:0]    s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
    wire [NUM_CORES*2-1:0]  s_bresp, s_rresp; wire [NUM_CORES*32-1:0] s_rdata;

    wire [NUM_CORES*32-1:0] m_awaddr, m_araddr;
    wire [NUM_CORES*8-1:0]  m_awlen, m_arlen;
    wire [NUM_CORES*3-1:0]  m_awsize, m_arsize; wire [NUM_CORES*2-1:0] m_awburst, m_arburst;
    wire [NUM_CORES-1:0]    m_awvalid, m_wlast, m_wvalid, m_bready, m_arvalid, m_rready;
    reg  [NUM_CORES-1:0]    m_awready, m_wready, m_arready;
    wire [NUM_CORES*ACC_W-1:0] m_wdata; wire [NUM_CORES*(ACC_W/8)-1:0] m_wstrb;
    reg  [NUM_CORES*2-1:0]  m_bresp; reg  [NUM_CORES-1:0] m_bvalid;
    reg  [NUM_CORES*ACC_W-1:0] m_rdata; reg  [NUM_CORES*2-1:0] m_rresp;
    reg  [NUM_CORES-1:0]    m_rvalid, m_rlast; wire [NUM_CORES-1:0] npu_irq;

    integer pass_cnt, fail_cnt, cyc;
    reg [ACC_W-1:0] dram [0:DRAM_SZ-1];

    npu_mc_top #(
        .NUM_CORES(NUM_CORES), .PHY_ROWS(PHY_ROWS), .PHY_COLS(PHY_COLS),
        .DATA_W(DATA_W), .ACC_W(ACC_W), .PPB_DEPTH(PPB_DEPTH), .PPB_THRESH(PPB_THRESH),
        .INT8_SIMD_LANES(INT8_LANES), .PERF_ENABLE_DERIVED(0), .FP16_ENABLE(0),
        .PPB_SCALAR_READ_ENABLE(1)
    ) u_dut (
        .sys_clk(clk), .sys_rst_n(rst_n),
        .s_axi_awaddr(s_awaddr), .s_axi_awvalid(s_awvalid), .s_axi_awready(s_awready),
        .s_axi_wdata(s_wdata), .s_axi_wstrb(s_wstrb),
        .s_axi_wvalid(s_wvalid), .s_axi_wready(s_wready),
        .s_axi_bresp(s_bresp), .s_axi_bvalid(s_bvalid), .s_axi_bready(s_bready),
        .s_axi_araddr(s_araddr), .s_axi_arvalid(s_arvalid), .s_axi_arready(s_arready),
        .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp),
        .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready),
        .m_axi_awaddr(m_awaddr), .m_axi_awlen(m_awlen), .m_axi_awsize(m_awsize),
        .m_axi_awburst(m_awburst), .m_axi_awvalid(m_awvalid), .m_axi_awready(m_awready),
        .m_axi_wdata(m_wdata), .m_axi_wstrb(m_wstrb), .m_axi_wlast(m_wlast),
        .m_axi_wvalid(m_wvalid), .m_axi_wready(m_wready),
        .m_axi_bresp(m_bresp), .m_axi_bvalid(m_bvalid), .m_axi_bready(m_bready),
        .m_axi_araddr(m_araddr), .m_axi_arlen(m_arlen), .m_axi_arsize(m_arsize),
        .m_axi_arburst(m_arburst), .m_axi_arvalid(m_arvalid), .m_axi_arready(m_arready),
        .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp),
        .m_axi_rvalid(m_rvalid), .m_axi_rlast(m_rlast), .m_axi_rready(m_rready),
        .npu_irq(npu_irq)
    );

    task pass; begin pass_cnt = pass_cnt + 1; end endtask
    task fail; input [1023:0] msg; begin $display("[FAIL] %0s", msg); fail_cnt = fail_cnt + 1; end endtask
    task check; input ok; input [1023:0] msg; begin if (ok) pass; else fail(msg); end endtask

    task core_write; input integer core; input [31:0] off; input [31:0] val;
        begin
            // Cycle 0: assert AW+W
            s_awaddr[core*32 +: 32] = {24'd0, off}; s_awvalid[core] = 1'b1;
            s_wdata[core*32 +: 32] = val; s_wstrb[core*4 +: 4] = 4'hF;
            s_wvalid[core] = 1'b1; s_bready[core] = 1'b1;
            @(posedge clk); s_awvalid[core] = 1'b0;  // AW fire done
            @(posedge clk); s_wvalid[core] = 1'b0;     // W fire done
            repeat(10) @(posedge clk); s_bready[core] = 1'b0;  // B handshake
        end
    endtask

    task core_read; input integer core; input [31:0] off; output [31:0] val;
        begin
            s_araddr[core*32 +: 32] = {24'd0, off}; s_arvalid[core] = 1'b1;
            s_rready[core] = 1'b1;
            @(posedge clk); s_arvalid[core] = 1'b0;
            repeat(5) @(posedge clk);
            val = s_rdata[core*32 +: 32]; s_rready[core] = 1'b0;
        end
    endtask

    // Per-core DMA responder state
    reg [7:0]  rd_len_d [0:1]; reg [7:0] rd_idx_d [0:1];
    reg [31:0] rd_base_d [0:1]; reg rd_active_d [0:1];
    reg wr_aw_d [0:1]; reg [31:0] wr_base_d [0:1];
    genvar gv;
    generate for (gv = 0; gv < NUM_CORES; gv = gv + 1) begin : gen_dma
        always @(posedge clk) begin
            if (!rst_n) begin
                rd_active_d[gv] <= 0; rd_idx_d[gv] <= 0;
                m_rvalid[gv] <= 0; m_rlast[gv] <= 0;
                wr_aw_d[gv] <= 0; m_bvalid[gv] <= 0;
            end else begin
                if (!rd_active_d[gv] && m_arvalid[gv] && m_arready[gv]) begin
                    rd_active_d[gv] <= 1; rd_base_d[gv] <= m_araddr[gv*32 +: 32];
                    rd_len_d[gv] <= m_arlen[gv*8 +: 8]; rd_idx_d[gv] <= 0;
                end
                if (rd_active_d[gv]) begin
                    m_rvalid[gv] <= 1;
                    m_rdata[gv*ACC_W +: ACC_W] <= dram[(rd_base_d[gv] >> 2) + rd_idx_d[gv]];
                    m_rlast[gv] <= (rd_idx_d[gv] >= rd_len_d[gv]);
                    if (m_rvalid[gv] && m_rready[gv]) begin
                        if (m_rlast[gv]) begin rd_active_d[gv] <= 0; m_rvalid[gv] <= 0; end
                        else rd_idx_d[gv] <= rd_idx_d[gv] + 1;
                    end
                end
                if (m_awvalid[gv] && m_awready[gv] && !wr_aw_d[gv]) begin
                    wr_aw_d[gv] <= 1; wr_base_d[gv] <= m_awaddr[gv*32 +: 32];
                end
                if (wr_aw_d[gv] && m_wvalid[gv] && m_wready[gv]) begin
                    dram[(wr_base_d[gv] >> 2)] <= m_wdata[gv*ACC_W +: ACC_W];
                    if (m_wlast[gv]) begin wr_aw_d[gv] <= 0; m_bvalid[gv] <= 1; end
                end
                if (m_bvalid[gv] && m_bready[gv]) m_bvalid[gv] <= 0;
            end
        end
    end endgenerate

    task poll_done; input integer core;
        reg [31:0] st; integer g;
        begin g = 0;
            while (g < TIMEOUT) begin
                core_read(core, REG_STATUS, st);
                if (st[1]) g = TIMEOUT; else g = g + 1;
            end
        end
    endtask

    task prog_core; input integer core; input integer w_offs, a_offs, r_offs;
        begin
            core_write(core, REG_CTRL, 32'd0);  // clear CTRL to ensure fresh start edge
            core_write(core, REG_M_DIM, 32'd4); core_write(core, REG_N_DIM, 32'd4);
            core_write(core, REG_K_DIM, 32'd4); core_write(core, REG_W_ADDR, w_offs);
            core_write(core, REG_A_ADDR, a_offs); core_write(core, REG_R_ADDR, r_offs);
            core_write(core, REG_BIAS_ADDR, 32'd0); core_write(core, REG_ARR_CFG, 32'h80);
            core_write(core, REG_CFG_SHAPE, 32'd0); core_write(core, REG_QUANT_CFG, 32'h0001_0000);
        end
    endtask

    initial begin : test_seq
        pass_cnt = 0; fail_cnt = 0; cyc = 0;
        s_awaddr = 0; s_wdata = 0; s_araddr = 0; s_wstrb = 0;
        s_awvalid = 0; s_wvalid = 0; s_arvalid = 0; s_rready = 0; s_bready = 0;
        m_awready = {NUM_CORES{1'b1}}; m_wready = {NUM_CORES{1'b1}};
        m_bresp = 0; m_bvalid = 0; m_arready = {NUM_CORES{1'b1}}; m_rresp = 0;
        m_rvalid = 0; m_rlast = 0;
        rd_active_d[0] = 0; rd_active_d[1] = 0; rd_idx_d[0] = 0; rd_idx_d[1] = 0;
        wait(rst_n); repeat(4) @(posedge clk);

        // Pre-load DRAM: core0 W@0x0 A@0x20 R@0x100, core1 W@0x200 A@0x220 R@0x300
        dram[0]=32'h0202_0202; dram[1]=0; dram[2]=32'h0101_0101; dram[3]=0;
        dram[128]=32'h0303_0303; dram[129]=0; dram[130]=32'h0101_0101; dram[131]=0;

        $display("--- Core0 only ---");
        prog_core(0, 0, 8, 32'h100); core_write(0, REG_CTRL, 32'h211);
        poll_done(0);
        begin reg [31:0] s0, s1;
            core_read(0, REG_STATUS, s0); core_read(1, REG_STATUS, s1);
            check(s0[1] === 1'b1, "core0 not done"); check(s1[1] === 1'b0, "core1 spuriously done");
        end

        $display("--- Core1 only ---");
        core_write(0, REG_CTRL, 32'd0);  // clear core0 done
        prog_core(1, 32'h200, 32'h208, 32'h300); core_write(1, REG_CTRL, 32'h211);
        poll_done(1);
        begin reg [31:0] s0, s1;
            core_read(0, REG_STATUS, s0); core_read(1, REG_STATUS, s1);
            check(s1[1] === 1'b1, "core1 not done"); check(s0[1] === 1'b0, "core0 spuriously done");
        end

        $display("--- Both cores ---");
        core_write(0, REG_CTRL, 32'd0);
        core_write(1, REG_CTRL, 32'd0);
        prog_core(0, 0, 8, 32'h500); prog_core(1, 32'h200, 32'h208, 32'h600);
        core_write(0, REG_CTRL, 32'h211); core_write(1, REG_CTRL, 32'h211);
        fork poll_done(0); poll_done(1); join
        begin reg [31:0] s0, s1;
            core_read(0, REG_STATUS, s0); core_read(1, REG_STATUS, s1);
            check(s0[1] && s1[1], "both cores not done");
        end
        check(dram[32'h500 >> 2] !== 0, "core0 result zero");
        check(dram[32'h600 >> 2] !== 0, "core1 result zero");

        $display("--- Error isolation ---");
        prog_core(0, 0, 0, 32'h700); // bad A addr
        core_write(0, REG_CTRL, 32'h211);
        repeat(50) @(posedge clk);
        begin reg [31:0] s1;
            core_read(1, REG_STATUS, s1);
            check(s1[2] === 1'b0, "core1 error from core0 fault");
        end

        if (fail_cnt == 0) $display("[PASS] tb_npu_mc_top_smoke: %0d checks passed", pass_cnt);
        else $display("[FAIL] tb_npu_mc_top_smoke: %0d passed, %0d failed", pass_cnt, fail_cnt);
        $finish;
    end

    always @(posedge clk) if (rst_n) cyc <= cyc + 1;
    initial begin repeat(4) @(posedge clk); rst_n = 1'b1; end
    initial begin #(CLK_T * TIMEOUT); $display("[FAIL] tb_npu_mc_top_smoke timeout at %0d", cyc); $finish; end
endmodule
