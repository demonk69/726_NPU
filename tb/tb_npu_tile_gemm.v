`timescale 1ns/1ps

module tb_npu_tile_gemm;

`include "test_params.vh"

`ifdef OUTPUT_HEX_VH
`include "output_hex_define.vh"
`endif

`ifndef OUTPUT_HEX
`define OUTPUT_HEX "npu_output.hex"
`endif

localparam DATA_W  = 16;
localparam ACC_W   = 32;
localparam CLK_T   = 10;
localparam PE_ACTIVE_W = 256;

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
localparam WS_RUN_W_ADDR = 32'h00006000;
localparam WS_RUN_A_ADDR = 32'h00006400;
localparam WS_RUN_R_ADDR = 32'h00006800;

localparam ARR_TILE4     = 32'h80; // ARR_CFG[7]: enable 4x4 tile planner/data path
localparam CFG_4X4       = 32'h0;  // CFG_SHAPE=00: use the left-top 4x4 PE array

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
reg [31:0] expected [0:`NUM_RESULTS-1];
integer aw_count;   // number of result row write bursts observed
integer pass_cnt;
integer fail_cnt;
integer dump_fd;
integer ws_accum [0:15];

`ifdef EDGE_SHAPE
reg edge_full_tile_seen;
`endif

initial begin
    $readmemh(`DRAM_HEX, dram);
    $readmemh(`EXPECTED_HEX, expected);
end

npu_top #(
    .DATA_W(DATA_W),
    .ACC_W (ACC_W)
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
            dram[((wr_base >> 2) + wr_cnt) % `DRAM_SIZE] <= m_wdata;
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

`ifdef EDGE_SHAPE
generate
    if ((`EDGE_PE_ACTIVE_EXPECT == 16) &&
        (`EDGE_PE_VALID_EXPECT == 16) &&
        (`CTRL == 32'h00000011)) begin : gen_edge_4x4_full_tile_monitor
        localparam [31:0] EDGE_VALID_MASK = 32'h0000_FFFF;

        function [PE_ACTIVE_W-1:0] expected_active_mask;
            integer rr;
            integer cc;
            begin
                expected_active_mask = {PE_ACTIVE_W{1'b0}};
                for (rr = 0; rr < `TILE_M; rr = rr + 1) begin
                    for (cc = 0; cc < `TILE_N; cc = cc + 1) begin
                        expected_active_mask[(rr * 16) + cc] = 1'b1;
                    end
                end
            end
        endfunction

        always @(posedge clk) begin
            if (!rst_n) begin
                edge_full_tile_seen <= 1'b0;
            end else if (!edge_full_tile_seen &&
                         u_npu.status_busy &&
                         u_npu.ctrl_tile_mode &&
                         (u_npu.pe_array_valid === EDGE_VALID_MASK)) begin
                if (u_npu.pe_active_dbg !== expected_active_mask()) begin
                    $display("[FAIL] %s full-tile active mask mismatch: got=0x%064h exp=0x%064h",
                             `TEST_NAME, u_npu.pe_active_dbg, expected_active_mask());
                    fail_cnt = fail_cnt + 1;
                end
                edge_full_tile_seen <= 1'b1;
            end
        end
    end
endgenerate
`endif

task check_result;
    input integer idx;
    reg [31:0] got;
    reg [31:0] exp;
    begin
        // expected[] is row-major C[r,c], so idx/4 is r and idx%4 is c.
        got = dram[(`R_ADDR >> 2) + idx];
        exp = expected[idx];
        if (`IS_FP16) begin
            if (fp32_close(got, exp)) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s C[%0d][%0d] got=0x%08h exp=0x%08h",
                         `TEST_NAME, idx / 4, idx % 4, got, exp);
                fail_cnt = fail_cnt + 1;
            end
        end else begin
            if (got === exp) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s C[%0d][%0d] got=%0d (0x%08h) exp=%0d (0x%08h)",
                         `TEST_NAME, idx / 4, idx % 4,
                         $signed(got), got, $signed(exp), exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask

`ifdef EDGE_SHAPE
task prepare_ws_block;
    input integer row_idx;
    input integer k_base;
    input integer blk_len;
    integer t;
    reg [31:0] src_word;
    reg [7:0]  lane_byte;
    begin
        for (t = 0; t < `TILE_N; t = t + 1)
            dram[(WS_RUN_R_ADDR >> 2) + t] = 32'd0;
        for (t = 0; t < 4; t = t + 1) begin
            dram[(WS_RUN_W_ADDR >> 2) + t] = 32'd0;
            dram[(WS_RUN_A_ADDR >> 2) + t] = 32'd0;
        end
        if (blk_len > 0) begin
            src_word = dram[(`A_ADDR >> 2) + k_base];
            case (row_idx)
                0: lane_byte = src_word[7:0];
                1: lane_byte = src_word[15:8];
                2: lane_byte = src_word[23:16];
                default: lane_byte = src_word[31:24];
            endcase
            dram[(WS_RUN_A_ADDR >> 2)] = {24'd0, lane_byte};

            dram[(WS_RUN_W_ADDR >> 2)] = dram[(`W_ADDR >> 2) + k_base];
        end
    end
endtask

task run_ws_block;
    input integer row_idx;
    input integer k_base;
    input integer blk_len;
    integer aw_before;
    integer col_idx;
    begin
        prepare_ws_block(row_idx, k_base, blk_len);
        aw_before = aw_count;

        axi_write(REG_M_DIM, 32'd1);
        axi_write(REG_N_DIM, `TILE_N);
        axi_write(REG_K_DIM, 32'd1);
        axi_write(REG_W_ADDR, WS_RUN_W_ADDR);
        axi_write(REG_A_ADDR, WS_RUN_A_ADDR);
        axi_write(REG_R_ADDR, WS_RUN_R_ADDR);
        axi_write(REG_ARR_CFG, ARR_TILE4);
        axi_write(REG_CFG_SHAPE, `CFG_SHAPE);
        axi_write(REG_CTRL, `CTRL);
        wait_done(5000);

        if (aw_count !== aw_before + 1) begin
            $display("[FAIL] %s WS row%0d k%0d expected one row write burst, got delta=%0d",
                     `TEST_NAME, row_idx, k_base,
                     aw_count - aw_before);
            fail_cnt = fail_cnt + 1;
        end

        for (col_idx = 0; col_idx < `TILE_N; col_idx = col_idx + 1)
            ws_accum[row_idx * `TILE_N + col_idx] =
                ws_accum[row_idx * `TILE_N + col_idx] +
                $signed(dram[(WS_RUN_R_ADDR >> 2) + col_idx]);
    end
endtask

task check_ws_accum_result;
    input integer idx;
    begin
        if (ws_accum[idx] === $signed(expected[idx])) begin
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("[FAIL] %s WS C[%0d][%0d] got=%0d exp=%0d",
                     `TEST_NAME, idx / `TILE_N, idx % `TILE_N,
                     ws_accum[idx], $signed(expected[idx]));
            fail_cnt = fail_cnt + 1;
        end
    end
endtask

task check_internal_result;
    input integer r;
    input integer c;
    reg [31:0] got;
    reg [31:0] exp;
    begin
        got = u_npu.u_pe_array.acc_v[r+1][c];
        exp = expected[r * `TILE_N + c];
        if (`IS_FP16) begin
            if (fp32_close(got, exp)) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s internal C[%0d][%0d] got=0x%08h exp=0x%08h",
                         `TEST_NAME, r, c, got, exp);
                fail_cnt = fail_cnt + 1;
            end
        end else begin
            if (got === exp) begin
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s internal C[%0d][%0d] got=%0d (0x%08h) exp=%0d (0x%08h)",
                         `TEST_NAME, r, c, $signed(got), got, $signed(exp), exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    end
endtask
`endif

integer i;
integer ws_row_idx;
integer ws_k_base;
integer ws_blk_len;

initial begin
    s_awaddr = 0; s_wdata = 0; s_wstrb = 0;
    s_awvalid = 0; s_wvalid = 0; s_bready = 0;
    s_araddr = 0; s_arvalid = 0; s_rready = 0;
    pass_cnt = 0;
    fail_cnt = 0;
`ifdef EDGE_SHAPE
    edge_full_tile_seen = 1'b0;
    for (i = 0; i < 16; i = i + 1)
        ws_accum[i] = 0;
`endif

    @(posedge rst_n);
    repeat (4) @(posedge clk);

    $display("");
    $display("################################################################");
`ifdef EDGE_SHAPE
    $display("  Tile GEMM Test (%s): %s", `EDGE_SHAPE, `TEST_NAME);
`else
    $display("  4x4 Tile GEMM Test: %s", `TEST_NAME);
`endif
    $display("################################################################");

    axi_write(REG_M_DIM, `M_DIM);
    axi_write(REG_N_DIM, `N_DIM);
    axi_write(REG_K_DIM, `K_DIM);
    axi_write(REG_W_ADDR, `W_ADDR);
    axi_write(REG_A_ADDR, `A_ADDR);
    axi_write(REG_R_ADDR, `R_ADDR);
    axi_write(REG_ARR_CFG, ARR_TILE4);
`ifdef EDGE_SHAPE
    axi_write(REG_CFG_SHAPE, `CFG_SHAPE);
`else
    axi_write(REG_CFG_SHAPE, CFG_4X4);
`endif
`ifdef EDGE_SHAPE
    if (`CTRL != 32'h00000001) begin
        axi_write(REG_CTRL, `CTRL);
        wait_done((`TILE_M == 4) ? 5000 : ((`TILE_M == 8) ? 20000 : 50000));
    end
`else
    axi_write(REG_CTRL, `CTRL);
    wait_done(5000);
`endif

`ifdef EDGE_SHAPE
    if ((`TILE_M == 4) && (`TILE_N == 4) && (`CTRL == 32'h00000011) && !edge_full_tile_seen) begin
        $display("[FAIL] %s did not observe a full-tile valid cycle", `TEST_NAME);
        fail_cnt = fail_cnt + 1;
    end
`endif

`ifdef EDGE_SHAPE
    if (`CTRL == 32'h00000001) begin
        for (ws_row_idx = 0; ws_row_idx < `TILE_M; ws_row_idx = ws_row_idx + 1) begin
            ws_k_base = 0;
            while (ws_k_base < `K_DIM) begin
                ws_blk_len = 1;
                run_ws_block(ws_row_idx, ws_k_base, ws_blk_len);
                ws_k_base = ws_k_base + 1;
            end
        end
        for (i = 0; i < `NUM_RESULTS; i = i + 1)
            check_ws_accum_result(i);
    end else if ((`TILE_M == 4) && (`TILE_N == 4)) begin
        if (aw_count !== 4) begin
            $display("[FAIL] %s expected 4 row write bursts, got %0d", `TEST_NAME, aw_count);
            $finish;
        end
        for (i = 0; i < `NUM_RESULTS; i = i + 1)
            check_result(i);
    end else begin
        $display("[INFO] %s skipping row-writeback compare; checking internal PE array results", `TEST_NAME);
        for (i = 0; i < `TILE_M; i = i + 1) begin
            integer j;
            for (j = 0; j < `TILE_N; j = j + 1)
                check_internal_result(i, j);
        end
    end
`else
    if (aw_count !== 4) begin
        // For a full 4x4 tile, npu_ctrl should issue 4 row-wise write bursts.
        $display("[FAIL] %s expected 4 row write bursts, got %0d", `TEST_NAME, aw_count);
        $finish;
    end

    for (i = 0; i < `NUM_RESULTS; i = i + 1)
        check_result(i);
`endif

    `ifdef DUMP_RESULT_HEX
    dump_fd = $fopen(`OUTPUT_HEX, "w");
    if (dump_fd == 0) begin
        $display("[FAIL] %s could not open output dump: %s", `TEST_NAME, `OUTPUT_HEX);
        fail_cnt = fail_cnt + 1;
    end else begin
        `ifdef EDGE_SHAPE
        if (`CTRL == 32'h00000001) begin
            for (i = 0; i < `NUM_RESULTS; i = i + 1)
                $fdisplay(dump_fd, "%08h", ws_accum[i][31:0]);
        end else if ((`TILE_M == 4) && (`TILE_N == 4)) begin
            for (i = 0; i < `NUM_RESULTS; i = i + 1)
                $fdisplay(dump_fd, "%08h", dram[(`R_ADDR >> 2) + i]);
        end else begin
            for (i = 0; i < `TILE_M; i = i + 1) begin
                integer j;
                for (j = 0; j < `TILE_N; j = j + 1)
                    $fdisplay(dump_fd, "%08h", u_npu.u_pe_array.acc_v[i+1][j]);
            end
        end
        `else
        for (i = 0; i < `NUM_RESULTS; i = i + 1)
            $fdisplay(dump_fd, "%08h", dram[(`R_ADDR >> 2) + i]);
        `endif
        $fclose(dump_fd);
        $display("[DUMP] %s wrote %s", `TEST_NAME, `OUTPUT_HEX);
    end
    `endif

    if (fail_cnt == 0) begin
        `ifdef EDGE_SHAPE
        if (`CTRL == 32'h00000001) begin
            $display("[PASS] %s: WS row-pass compare passed (%0d checks)", `TEST_NAME, pass_cnt);
        end else if ((`TILE_M == 4) && (`TILE_N == 4)) begin
            $display("[PASS] %s: ALL %0d CHECKS PASSED", `TEST_NAME, pass_cnt);
        end else begin
            $display("[PASS] %s: internal PE-array compare passed (%0d checks)", `TEST_NAME, pass_cnt);
        end
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
    #(CLK_T * 200000);
    $display("[FAIL] %s global timeout", `TEST_NAME);
    $finish;
end

endmodule
