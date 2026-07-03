`timescale 1ns/1ps

module tb_npu_tile_gemm_v2;

`include "test_params.vh"

`ifndef OUTPUT_HEX
`define OUTPUT_HEX "npu_output.hex"
`endif

`ifdef DATA_W_VAL
localparam DATA_W  = `DATA_W_VAL;
`else
localparam DATA_W  = 32;
`endif
localparam ACC_W   = 32;
localparam CLK_T   = 10;
`ifdef INT8_SIMD_LANES_VAL
localparam INT8_SIMD_LANES = `INT8_SIMD_LANES_VAL;
`else
localparam INT8_SIMD_LANES = 4;
`endif
`ifdef PERF_ONLY
localparam PERF_ONLY_MODE = 1;
`else
localparam PERF_ONLY_MODE = 0;
`endif
`ifdef WAIT_TIMEOUT_VAL
localparam [31:0] WAIT_TIMEOUT = `WAIT_TIMEOUT_VAL;
`else
localparam [31:0] WAIT_TIMEOUT = 32'd50000;
`endif
`ifdef TB_TIMEOUT_CYCLES_VAL
localparam [63:0] TB_TIMEOUT_CYCLES = `TB_TIMEOUT_CYCLES_VAL;
`else
localparam [63:0] TB_TIMEOUT_CYCLES = 64'd1000000;
`endif
`ifdef STRICT_AW_EXPECT
localparam STRICT_AW_EXPECT_MODE = 1;
`else
localparam STRICT_AW_EXPECT_MODE = 0;
`endif

localparam REG_CTRL      = 32'h00;
localparam REG_STATUS    = 32'h04;
localparam REG_M_DIM     = 32'h10;
localparam REG_N_DIM     = 32'h14;
localparam REG_K_DIM     = 32'h18;
localparam REG_W_ADDR    = 32'h20;
localparam REG_A_ADDR    = 32'h24;
localparam REG_R_ADDR    = 32'h28;
localparam REG_BIAS_ADDR = 32'h98;
localparam REG_QUANT_CFG = 32'h9C;
localparam REG_ARR_CFG   = 32'h30;
localparam REG_CLK_DIV   = 32'h34;
localparam REG_CFG_SHAPE = 32'h3C;
localparam REG_PERF_CYCLES         = 32'h48;
localparam REG_PERF_RD_BEATS       = 32'h4C;
localparam REG_PERF_WR_BEATS       = 32'h50;
localparam REG_PERF_RD_BYTES       = 32'h54;
localparam REG_PERF_WR_BYTES       = 32'h58;
localparam REG_PERF_RD_BURSTS      = 32'h6C;
localparam REG_PERF_WR_BURSTS      = 32'h70;
localparam REG_ERR_STATUS          = 32'h74;
localparam REG_PERF_CTRL           = 32'h78;
localparam REG_PERF_MAC_OPS_LO     = 32'hA0;
localparam REG_PERF_MAC_OPS_HI     = 32'hA4;
localparam REG_PERF_OPS_LO         = 32'hA8;
localparam REG_PERF_OPS_HI         = 32'hAC;
localparam REG_PERF_BUSY_CYCLES    = 32'hB0;
localparam REG_PERF_COMPUTE_CYCLES = 32'hB4;
localparam REG_PERF_DMA_CYCLES     = 32'hB8;
localparam REG_PERF_PEAK_OPS_CYCLE = 32'hC8;

`ifdef ARR_CFG
localparam ARR_TILE     = `ARR_CFG;
`else
localparam ARR_TILE     = 32'h80; // ARR_CFG[7]: enable tile planner/data path
`endif
`ifdef USE_ROUTER_MESH_VAL
localparam TB_USE_ROUTER_MESH = `USE_ROUTER_MESH_VAL;
`else
localparam TB_USE_ROUTER_MESH = 0;
`endif
`ifdef DUT_PHY_ROWS
localparam TB_PHY_ROWS = `DUT_PHY_ROWS;
`else
localparam TB_PHY_ROWS = 16;
`endif
`ifdef DUT_PHY_COLS
localparam TB_PHY_COLS = `DUT_PHY_COLS;
`else
localparam TB_PHY_COLS = 16;
`endif
`ifdef CFG_SHAPE_VAL
localparam CFG_SHAPE    = `CFG_SHAPE_VAL;
`else
localparam CFG_SHAPE    = 32'h0;  // default 4x4
`endif
`ifdef GRID_COLS_VAL
localparam GRID_COLS    = `GRID_COLS_VAL;
`else
localparam GRID_COLS    = 4;      // default 4
`endif
`ifdef CLK_DIV_VAL
localparam CLK_DIV = `CLK_DIV_VAL;
`else
localparam CLK_DIV = 0;
`endif
`ifdef AW_EXPECT_VAL
localparam AW_EXPECT    = `AW_EXPECT_VAL;
`else
localparam AW_EXPECT    = 4;
`endif

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

reg [31:0] dram [0:`DRAM_SIZE-1];
`ifdef PERF_ONLY
reg [31:0] expected [0:0];
`else
reg [31:0] expected [0:`NUM_RESULTS-1];
`endif
integer aw_count;   // number of result row write bursts observed
integer pass_cnt;
integer fail_cnt;
integer dump_fd;
reg [63:0] cycle_count;
reg [63:0] start_cycle;
reg [63:0] done_cycle;
reg [63:0] run_cycles;
reg [31:0] perf_cycles;
reg [31:0] perf_rd_beats;
reg [31:0] perf_wr_beats;
reg [31:0] perf_rd_bytes;
reg [31:0] perf_wr_bytes;
reg [31:0] perf_rd_bursts;
reg [31:0] perf_wr_bursts;
reg [31:0] perf_mac_ops_lo;
reg [31:0] perf_mac_ops_hi;
reg [31:0] perf_ops_lo;
reg [31:0] perf_ops_hi;
reg [31:0] perf_busy_cycles;
reg [31:0] perf_compute_cycles;
reg [31:0] perf_dma_cycles;
reg [31:0] perf_peak_ops_cycle;
reg [31:0] err_status;

always @(posedge clk) begin
    if (!rst_n)
        cycle_count <= 64'd0;
    else
        cycle_count <= cycle_count + 64'd1;
end

// Diagnostic: snapshot PPBuf output on fire cycles (ifdef DIAG_TRACE only)
`ifdef DIAG_TRACE
reg [16*32-1:0] dbg_a_snap;
reg [16*32-1:0] dbg_w_snap;
reg [16*32-1:0] dbg_a_snap2;
reg [16*32-1:0] dbg_w_snap2;
reg             dbg_a_snapped;
reg             dbg_a_snapped2;

always @(posedge clk) begin
    if (!rst_n) begin
        dbg_a_snapped  <= 1'b0;
        dbg_a_snapped2 <= 1'b0;
        dbg_a_snap     <= 0;
        dbg_w_snap     <= 0;
        dbg_a_snap2    <= 0;
        dbg_w_snap2    <= 0;
    end else if (!dbg_a_snapped && u_npu.a_ppb_rd_vec_valid) begin
        dbg_a_snapped <= 1'b1;
        dbg_a_snap    <= u_npu.a_ppb_rd_vec;
        dbg_w_snap    <= u_npu.w_ppb_rd_vec;
    end else if (dbg_a_snapped && !dbg_a_snapped2 && u_npu.a_ppb_rd_vec_valid) begin
        dbg_a_snapped2 <= 1'b1;
        dbg_a_snap2    <= u_npu.a_ppb_rd_vec;
        dbg_w_snap2    <= u_npu.w_ppb_rd_vec;
    end
end
`endif

initial begin
    $readmemh(`DRAM_HEX, dram);
`ifndef PERF_ONLY
    $readmemh(`EXPECTED_HEX, expected);
`endif
end

npu_top #(
    .PHY_ROWS(TB_PHY_ROWS),
    .PHY_COLS(TB_PHY_COLS),
    .DATA_W(DATA_W),
    .ACC_W (ACC_W),
    .INT8_SIMD_LANES(INT8_SIMD_LANES),
    .USE_ROUTER_MESH(TB_USE_ROUTER_MESH)
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
    end else begin
        m_arready <= 1'b1;
        if (m_rvalid && m_rready && m_rlast) begin
            m_rvalid <= 1'b0;
            m_rlast  <= 1'b0;
        end else if (!rd_active && !m_rvalid && m_arvalid && m_arready) begin
            // Model a simple AXI read burst. m_arlen is beats-1.
            rd_active <= 1'b1;
            rd_base   <= m_araddr;
            rd_len    <= m_arlen;
            rd_cnt    <= 8'd0;
            m_rvalid  <= 1'b0;
            m_rlast   <= 1'b0;
        end else if (rd_active && (!m_rvalid || (m_rvalid && m_rready))) begin
            m_rdata  <= dram[((rd_base >> 2) + rd_cnt) % `DRAM_SIZE];
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
            // Tile mode writes one short burst per valid C row.
            wr_phase <= 1'b1;
            wr_base  <= m_awaddr;
            wr_cnt   <= 8'd0;
            m_wready <= 1'b1;
            aw_count <= aw_count + 1;
        end

        if (wr_phase && m_wvalid && m_wready) begin
`ifndef PERF_ONLY
            dram[((wr_base >> 2) + wr_cnt) % `DRAM_SIZE] <= m_wdata;
`endif
            `ifdef DIAG_DRAM_WR
            $display("[DIAG_WR] aw=%0d addr=0x%08h cnt=%0d data=0x%08h",
                     aw_count, wr_base, wr_cnt, m_wdata);
            `endif
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

task axi_write;
    input [31:0] addr;
    input [31:0] data;
    integer guard;
    begin
        // Drive at negedge for clean setup before posedge
        @(negedge clk);
        s_awaddr  = addr;
        s_awvalid = 1'b1;
        s_wdata   = data;
        s_wstrb   = 4'hF;
        s_wvalid  = 1'b1;
        s_bready  = 1'b1;
        // Wait for DUT to accept and respond
        @(posedge clk);
        guard = 0;
        while (!s_awready && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        guard = 0;
        while (!s_wready && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        guard = 0;
        while (!s_bvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        // After bvalid seen, de-assert on negedge for clean teardown
        @(negedge clk);
        s_awvalid = 1'b0;
        s_wvalid  = 1'b0;
        s_bready  = 1'b0;
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
        @(negedge clk);
        s_araddr  = addr;
        s_arvalid = 1'b1;
        s_rready  = 1'b1;
        @(posedge clk);
        guard = 0;
        while (!s_arready && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        guard = 0;
        while (!s_rvalid && guard < 100) begin
            @(posedge clk);
            guard = guard + 1;
        end
        data = s_rdata;
        @(negedge clk);
        s_arvalid = 1'b0;
        s_rready  = 1'b0;
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
            $display("[FAIL] NPU timeout (status[1]=%0d)", status[1]);
`ifdef DIAG_TIMEOUT
            $display("[DIAG_TIMEOUT] ctrl state=%0d busy=%0d done=%0d pe_en=%0d pe_flush=%0d pe_ready=%0d tile_i=%0d tile_j=%0d kidx=%0d pass=%0d kcyc=%0d wb_row=%0d",
                     u_npu.u_ctrl.state,
                     u_npu.u_ctrl.busy,
                     u_npu.u_ctrl.done,
                     u_npu.u_ctrl.pe_en,
                     u_npu.u_ctrl.pe_flush,
                     u_npu.u_ctrl.pe_array_ready,
                     u_npu.u_ctrl.tile_i,
                     u_npu.u_ctrl.tile_j,
                     u_npu.u_ctrl.k_tile_idx,
                     u_npu.u_ctrl.pass_idx,
                     u_npu.u_ctrl.tile_k_cycle,
                     u_npu.u_ctrl.wb_row);
            $display("[DIAG_TIMEOUT] dma_state=%0d wb_state=%0d r_pending=%0d r_fill=%0d awvalid=%0d wvalid=%0d bready=%0d",
                     u_npu.u_dma.dma_state,
                     u_npu.u_dma.wb_state,
                     u_npu.u_dma.r_pending,
                     u_npu.u_dma.r_fill,
                     m_awvalid,
                     m_wvalid,
                     m_bready);
            $display("[DIAG_TIMEOUT] ser_busy=%0d ser_row=%0d ser_col=%0d router_collect=%0d router_seen_lo=0x%016h pe_valid_lo=0x%016h router_ready=%0d overflow=%0d",
                     u_npu.tile_ser_busy,
                     u_npu.tile_ser_row,
                     u_npu.tile_ser_col,
                     u_npu.tile_router_collect_active,
                     u_npu.tile_router_seen[63:0],
                     u_npu.pe_array_valid[63:0],
                     u_npu.pe_array_input_ready,
                     u_npu.pe_array_router_overflow);
`endif
            $finish;
        end
        axi_write(REG_CTRL, 32'h0);
    end
endtask

function [31:0] fp32_ordered;
    input [31:0] x;
    begin
        fp32_ordered = x[31] ? ~x : (x | 32'h8000_0000);
    end
endfunction

function fp32_close;
    input [31:0] got;
    input [31:0] exp;
    reg [31:0] got_ord, exp_ord, diff;
    begin
        got_ord = fp32_ordered(got);
        exp_ord = fp32_ordered(exp);
        diff = (got_ord > exp_ord) ? (got_ord - exp_ord) : (exp_ord - got_ord);
        fp32_close = (got === exp) || (diff <= 32'd8);
    end
endfunction

task check_result;
    input integer idx;
    reg [31:0] got;
    reg [31:0] exp;
    begin
        // expected[] is row-major C[r,c], so idx/cols is r and idx%cols is c.
        got = dram[(`R_ADDR >> 2) + idx];
        exp = expected[idx];
        if (`IS_FP16) begin
            if (fp32_close(got, exp)) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s C[%0d][%0d] got=0x%08h exp=0x%08h",
                         `TEST_NAME, idx / GRID_COLS, idx % GRID_COLS, got, exp);
                fail_cnt = fail_cnt + 1;
            end
        end else begin
            if (got === exp) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s C[%0d][%0d] got=%0d (0x%08h) exp=%0d (0x%08h)",
                         `TEST_NAME, idx / GRID_COLS, idx % GRID_COLS,
                         $signed(got), got, $signed(exp), exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

integer i;

initial begin
    s_awaddr = 0; s_wdata = 0; s_wstrb = 0;
    s_awvalid = 0; s_wvalid = 0; s_bready = 0;
    s_araddr = 0; s_arvalid = 0; s_rready = 0;
    pass_cnt = 0;
    fail_cnt = 0;
    start_cycle = 64'd0;
    done_cycle = 64'd0;
    run_cycles = 64'd0;

    @(posedge rst_n);
    repeat (4) @(posedge clk);

    $display("");
    $display("################################################################");
    $display("  Tile GEMM Test: %s DATA_W=%0d INT8_SIMD_LANES=%0d PERF_ONLY=%0d USE_ROUTER_MESH=%0d ARR_CFG=0x%02h",
             `TEST_NAME, DATA_W, INT8_SIMD_LANES, PERF_ONLY_MODE,
             TB_USE_ROUTER_MESH, ARR_TILE[7:0]);
    $display("################################################################");

    axi_write(REG_CLK_DIV, CLK_DIV);
    axi_write(REG_M_DIM, `M_DIM);
    axi_write(REG_N_DIM, `N_DIM);
    axi_write(REG_K_DIM, `K_DIM);
    axi_write(REG_W_ADDR, `W_ADDR);
    axi_write(REG_A_ADDR, `A_ADDR);
    axi_write(REG_R_ADDR, `R_ADDR);
`ifdef BIAS_ADDR_VAL
    axi_write(REG_BIAS_ADDR, `BIAS_ADDR_VAL);
`endif
`ifdef QUANT_CFG_VAL
    axi_write(REG_QUANT_CFG, `QUANT_CFG_VAL);
`endif
    axi_write(REG_ARR_CFG, ARR_TILE);
    axi_write(REG_CFG_SHAPE, CFG_SHAPE);
    axi_write(REG_PERF_CTRL, 32'h1);
    start_cycle = cycle_count;
`ifdef CTRL_VAL
    axi_write(REG_CTRL, `CTRL_VAL);
`else
    axi_write(REG_CTRL, 32'h11);
`endif

    wait_done(WAIT_TIMEOUT);
    done_cycle = cycle_count;
    run_cycles = done_cycle - start_cycle;
    axi_write(REG_PERF_CTRL, 32'h2);
    axi_read(REG_PERF_CYCLES, perf_cycles);
    axi_read(REG_PERF_RD_BEATS, perf_rd_beats);
    axi_read(REG_PERF_WR_BEATS, perf_wr_beats);
    axi_read(REG_PERF_RD_BYTES, perf_rd_bytes);
    axi_read(REG_PERF_WR_BYTES, perf_wr_bytes);
    axi_read(REG_PERF_RD_BURSTS, perf_rd_bursts);
    axi_read(REG_PERF_WR_BURSTS, perf_wr_bursts);
    axi_read(REG_PERF_MAC_OPS_LO, perf_mac_ops_lo);
    axi_read(REG_PERF_MAC_OPS_HI, perf_mac_ops_hi);
    axi_read(REG_PERF_OPS_LO, perf_ops_lo);
    axi_read(REG_PERF_OPS_HI, perf_ops_hi);
    axi_read(REG_PERF_BUSY_CYCLES, perf_busy_cycles);
    axi_read(REG_PERF_COMPUTE_CYCLES, perf_compute_cycles);
    axi_read(REG_PERF_DMA_CYCLES, perf_dma_cycles);
    axi_read(REG_PERF_PEAK_OPS_CYCLE, perf_peak_ops_cycle);
    axi_read(REG_ERR_STATUS, err_status);

    `ifdef DIAG_TRACE
    begin
        integer pe_c;
        $display("[PE_ACC] Post-sim PE accumulators:");
        for (pe_c = 0; pe_c < 4; pe_c = pe_c + 1) begin
            $display("[PE_ACC] PE(0,%0d)=0x%08h PE(8,%0d)=0x%08h",
                     pe_c,
                     u_npu.u_pe_array.acc_out[(0*32+pe_c)*32 +: 32],
                     pe_c,
                     u_npu.u_pe_array.acc_out[(0*32 + pe_c + 16)*32 +: 32]);
        end
    end
    `endif

    if (aw_count !== AW_EXPECT) begin
        if (STRICT_AW_EXPECT_MODE) begin
            $display("[FAIL] %s expected %0d row write bursts, got %0d", `TEST_NAME, AW_EXPECT, aw_count);
            $finish;
        end else begin
            $display("[WARN] %s expected %0d row write bursts, got %0d", `TEST_NAME, AW_EXPECT, aw_count);
        end
    end

`ifndef PERF_ONLY
    for (i = 0; i < `NUM_RESULTS; i = i + 1)
        check_result(i);
`else
    $display("[INFO] %s perf-only: skipped %0d golden checks", `TEST_NAME, `NUM_RESULTS);
`endif

    `ifdef DUMP_RESULT_HEX
`ifndef PERF_ONLY
    dump_fd = $fopen(`OUTPUT_HEX, "w");
    if (dump_fd == 0) begin
        $display("[FAIL] %s could not open output dump: %s", `TEST_NAME, `OUTPUT_HEX);
        fail_cnt = fail_cnt + 1;
    end else begin
        for (i = 0; i < `NUM_RESULTS; i = i + 1)
            $fdisplay(dump_fd, "%08h", dram[(`R_ADDR >> 2) + i]);
        $fclose(dump_fd);
        $display("[DUMP] %s wrote %s", `TEST_NAME, `OUTPUT_HEX);
    end
`else
    $display("[INFO] %s perf-only: result dump skipped", `TEST_NAME);
`endif
    `endif

    if (fail_cnt == 0) begin
        $display("[RESULT] %s", `TEST_NAME);
        $display("| M               | %-10d |", `M_DIM);
        $display("| N               | %-10d |", `N_DIM);
        $display("| K               | %-10d |", `K_DIM);
        $display("| cfg_shape       | %-10d |", CFG_SHAPE);
        $display("| lanes           | %-10d |", INT8_SIMD_LANES);
        $display("| data_w          | %-10d |", DATA_W);
        $display("| run_cycles      | %-10d |", run_cycles);
        $display("| perf_cycles     | %-10d |", perf_cycles);
        $display("| busy_cycles     | %-10d |", perf_busy_cycles);
        $display("| compute_cycles  | %-10d |", perf_compute_cycles);
        $display("| dma_cycles      | %-10d |", perf_dma_cycles);
        $display("| rd_beats        | %-10d |", perf_rd_beats);
        $display("| wr_beats        | %-10d |", perf_wr_beats);
        $display("| rd_bytes        | %-10d |", perf_rd_bytes);
        $display("| wr_bytes        | %-10d |", perf_wr_bytes);
        $display("| rd_bursts       | %-10d |", perf_rd_bursts);
        $display("| wr_bursts       | %-10d |", perf_wr_bursts);
        $display("| mac_ops         | %-10d |", {perf_mac_ops_hi, perf_mac_ops_lo});
        $display("| ops             | %-10d |", {perf_ops_hi, perf_ops_lo});
        $display("| peak_ops_cycle  | %-10d |", perf_peak_ops_cycle);
        $display("| aw_count        | %-10d |", aw_count);
        $display("| err_status      | 0x%08h |", err_status);
        $display("| status          | PASS     |");
`ifdef PERF_ONLY
        $display("[PASS] %s: PERF-ONLY COMPLETE", `TEST_NAME);
`else
        $display("[PASS] %s: ALL %0d CHECKS PASSED", `TEST_NAME, pass_cnt);
`endif
    end else begin
        $display("[FAIL] %s: %0d passed, %0d failed", `TEST_NAME, pass_cnt, fail_cnt);
        $fatal;
    end

    $finish;
end

initial begin
    #(CLK_T * TB_TIMEOUT_CYCLES);
    $display("[FAIL] %s global timeout", `TEST_NAME);
    $finish;
end

endmodule
